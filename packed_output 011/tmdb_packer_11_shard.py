#!/usr/bin/env python3
"""
tmdb_packer.py (sharded)

Pack NDJSON of TMDB similar-lists into compact binary shards.

Default behavior: create 4 shards (popularity-contiguous), stored in OUTDIR/shards/.
Also writes a compact source->shard index (binary + CSV) and shards_index.json.

Each shard is a self-contained SIML binary (header + source_keys + offsets + lengths + remap_table + values_blob).
Remap tables are created per-shard (so shards are independent).
"""
import os
import sys
import json
import struct
import gzip
import zlib
import csv
import math
from collections import Counter
from pathlib import Path

# Optional brotli compression
have_brotli = False
try:
    import brotli
    have_brotli = True
except Exception:
    have_brotli = False

def choose_file_via_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select NDJSON file",
            filetypes=[("NDJSON", "*.ndjson *.jsonl *.json"), ("All files", "*.*")])
        root.update()
        root.destroy()
        return path or None
    except Exception:
        return None

def packed_value(tmdb_id, typ):
    return (int(tmdb_id) << 1) | (1 if str(typ).lower().startswith('tv') else 0)

def parse_ndjson(path):
    """
    Parse NDJSON. Preserves input order (important for popularity-contiguous sharding).
    Returns: rows (list of dicts), total_similar E, global_freq Counter, freq_by_type Counter, max_list_len
    Each row: { source_tmdb_id, source_type, packed_source_key, packed_list }
    """
    rows = []
    freq = Counter()
    freq_by_type = Counter()
    total_similar = 0
    max_len = 0
    line_no = 0
    with open(path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            line_no += 1
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] skipping line {line_no}: JSON parse error: {e}")
                continue
            src_raw = obj.get('tmdb_id') if 'tmdb_id' in obj else obj.get('id') if 'id' in obj else None
            if src_raw is None:
                # tolerant: search keys with spaces
                for k in obj.keys():
                    if k.strip().lower() in ('tmdb_id', 'tmdbid', 'id'):
                        src_raw = obj[k]
                        break
            if src_raw is None:
                print(f"[warn] skipping line {line_no}: missing tmdb_id")
                continue
            try:
                src_id = int(src_raw)
            except Exception:
                print(f"[warn] skipping line {line_no}: invalid tmdb_id '{src_raw}'")
                continue
            src_type = obj.get('type', 'movie') or 'movie'
            similars = obj.get('similar', []) or []
            packed_list = []
            for s in similars:
                # tolerant parsing of similar entries (allow dicts or integers)
                try:
                    if isinstance(s, dict):
                        sid_raw = s.get('tmdb_id') if 'tmdb_id' in s else s.get('id') if 'id' in s else None
                        if sid_raw is None:
                            # try scanning keys
                            for k in s.keys():
                                if k.strip().lower() in ('tmdb_id', 'tmdbid', 'id'):
                                    sid_raw = s[k]; break
                        if sid_raw is None:
                            continue
                        sid = int(sid_raw)
                        stype = s.get('type', 'movie') or 'movie'
                    elif isinstance(s, int):
                        sid = s
                        stype = 'movie'
                    elif isinstance(s, str):
                        ss = s.strip()
                        if not ss:
                            continue
                        # try "123" or "123 movie"
                        parts = ss.split()
                        sid = int(parts[0])
                        stype = parts[1] if len(parts) > 1 else 'movie'
                    else:
                        continue
                except Exception:
                    continue
                pv = packed_value(sid, stype)
                packed_list.append(pv)
                freq[pv] += 1
                freq_by_type[stype] += 1
            psrc = packed_value(int(src_id), src_type)
            rows.append({
                "source_tmdb_id": int(src_id),
                "source_type": src_type,
                "packed_source_key": psrc,
                "packed_list": packed_list
            })
            total_similar += len(packed_list)
            if len(packed_list) > max_len:
                max_len = len(packed_list)
    return rows, total_similar, freq, freq_by_type, max_len

def build_remap(freq_counter):
    """
    Given a Counter of packed_value -> freq produce:
      packed_vals (list), freqs (list), pv_to_idx (dict), items (list of tuples)
    Sorted by frequency desc, then packed_val asc.
    """
    items = sorted(freq_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    packed_vals = [pv for pv, _ in items]
    freqs = [cnt for _, cnt in items]
    pv_to_idx = {pv: idx for idx, pv in enumerate(packed_vals)}
    return packed_vals, freqs, pv_to_idx, items

def choose_remap_width(U):
    if U <= 0xFFFF:
        return 2
    if U <= 0xFFFFFF:
        return 3
    return 4

def write_uint_bytes(fh, value, width):
    if width == 2:
        fh.write(struct.pack('<H', value & 0xFFFF))
    elif width == 3:
        v = value & 0xFFFFFF
        fh.write(bytes((v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF)))
    elif width == 4:
        fh.write(struct.pack('<I', value & 0xFFFFFFFF))
    else:
        raise ValueError("Unsupported width")

def write_bin_file(out_path, rows_sorted, offsets, header_info, packed_vals, pv_to_idx, remap_width, lengths_type, omit_offsets=False):
    """
    Write a SIML-format binary file (single shard or whole dataset).
    rows_sorted: list of rows sorted by packed_source_key (required for binary-search reader).
    offsets: list of offsets corresponding to rows_sorted (in indices within values_blob)
    header_info: dict containing R,E,U
    packed_vals: remap table (list of packed_values) for this file
    pv_to_idx: mapping from packed_value to remap index for this file
    remap_width: 2/3/4 bytes per remap index
    lengths_type: 0 uint8 else uint16
    omit_offsets: if True, set flag bit and don't write offsets
    Returns: (crc32, file_size)
    """
    R = header_info['R']; E = header_info['E']; U = header_info['U']
    header_size = 28
    magic = b"SIML"
    version = 1
    endian = 1
    flags = 1 if omit_offsets else 0
    lengths_byte = 0 if lengths_type == 0 else 1
    remap_index_width = remap_width
    reserved = 0
    header_crc_placeholder = 0
    header = struct.pack('<4s B B H I I I B B H I',
                         magic, version, endian, flags, R, E, U,
                         lengths_byte, remap_index_width, reserved, header_crc_placeholder)
    assert len(header) == header_size

    with open(out_path, 'wb') as fh:
        fh.write(header)

        # source_keys (R * uint32)
        for r in rows_sorted:
            fh.write(struct.pack('<I', r['packed_source_key']))

        # OFFSETS: optional (omitted if omit_offsets True)
        if not omit_offsets:
            for off in offsets:
                fh.write(struct.pack('<I', off))

        # lengths array
        if lengths_type == 0:
            for r in rows_sorted:
                l = len(r['packed_list'])
                if l >= 256:
                    raise ValueError("List length exceeds 255 but lengths_type is uint8")
                fh.write(struct.pack('<B', l))
        else:
            for r in rows_sorted:
                l = len(r['packed_list'])
                fh.write(struct.pack('<H', l))

        # remap_table (U * uint32)
        for pv in packed_vals:
            fh.write(struct.pack('<I', pv))

        # values_blob: remap indices (E entries) at remap_width bytes each
        for r in rows_sorted:
            for pv in r['packed_list']:
                idx = pv_to_idx[pv]
                if remap_width == 2 and idx >= (1 << 16):
                    raise ValueError("remap index overflow for uint16")
                if remap_width == 3 and idx >= (1 << 24):
                    raise ValueError("remap index overflow for 24-bit")
                write_uint_bytes(fh, idx, remap_width)

    # Compute CRC of file (with placeholder zero) and patch it into header offset 24
    with open(out_path, 'rb') as fh:
        data = fh.read()
    crc = zlib.crc32(data) & 0xFFFFFFFF
    with open(out_path, 'r+b') as fh:
        fh.seek(24)
        fh.write(struct.pack('<I', crc))

    return crc, os.path.getsize(out_path)

def write_csv_remap(out_csv, packed_vals, freqs):
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['remap_index', 'tmdb_id', 'type', 'packed_value', 'frequency'])
        for idx, pv in enumerate(packed_vals):
            tmdb = pv >> 1
            typ = 'tvshow' if (pv & 1) else 'movie'
            writer.writerow([idx, tmdb, typ, pv, freqs[idx]])

def write_csv_source_keys(out_csv, rows_sorted, offsets):
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['source_tmdb_id', 'source_type', 'packed_source_key', 'offset_index', 'length'])
        for r, off in zip(rows_sorted, offsets):
            writer.writerow([r['source_tmdb_id'], r['source_type'], r['packed_source_key'], off, len(r['packed_list'])])

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def write_source_to_shard_bin(out_path, mapping):
    """
    mapping: iterable of (packed_key:int, shard_id:int)
    Writes binary file: repeated records [uint32 packed_key][uint8 shard_id]
    (packed tight; file length = len(mapping) * 5)
    """
    with open(out_path, 'wb') as fh:
        for k, sid in mapping:
            fh.write(struct.pack('<I B', k & 0xFFFFFFFF, sid & 0xFF))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pack NDJSON TMDB similar-lists into compact binary shards.")
    parser.add_argument('ndjson', nargs='?', help='Path to NDJSON file. If omitted a file picker will be shown (if available).')
    parser.add_argument('--outdir', default='packed_output', help='Output folder')
    parser.add_argument('--no-gzip', action='store_true', help='Skip gzip output')
    parser.add_argument('--no-br', action='store_true', help='Skip brotli output even if available')
    parser.add_argument('--omit-offsets', action='store_true', help='Do not write offsets array to the binary (reader must reconstruct offsets from lengths)')
    parser.add_argument('--shards', type=int, default=4, help='Number of shards to produce (default 4)')
    args = parser.parse_args()

    ndjson_path = args.ndjson
    if not ndjson_path:
        ndjson_path = choose_file_via_dialog()
        if not ndjson_path:
            if len(sys.argv) > 1:
                ndjson_path = sys.argv[1]
            else:
                print("No NDJSON file specified. Exiting.")
                return
    ndjson_path = os.path.abspath(ndjson_path)
    if not os.path.isfile(ndjson_path):
        print("File not found:", ndjson_path)
        return

    outdir = os.path.abspath(args.outdir)
    ensure_dir(outdir)
    shards_dir = os.path.join(outdir, 'shards')
    ensure_dir(shards_dir)

    print("Parsing NDJSON:", ndjson_path)
    rows, E_total, freq_total, freq_by_type, max_list_len = parse_ndjson(ndjson_path)
    R_total = len(rows)
    U_total = len(freq_total)
    print(f"Rows R={R_total}, total similar entries E={E_total}, unique similar ids U={U_total}, max_list_len={max_list_len}")

    # ======= Decide sharding by popularity-contiguous (input order preserved) =======
    num_shards = max(1, int(args.shards))
    shard_size = math.ceil(R_total / num_shards)
    print(f"Sharding: {num_shards} shards, approx {shard_size} rows per shard")

    # We'll produce per-shard binaries. For each shard we build a per-shard remap table.
    shards_meta = []
    source_to_shard_pairs = []  # list of (packed_key, shard_id)

    # Keep global summary outputs too (single dataset.bin) as before:
    remap_width_global = choose_remap_width(len(freq_total))
    lengths_type_global = 0 if max_list_len < 256 else 1
    # We'll still write the original dataset.bin for compatibility
    rows_sorted_global = sorted(rows, key=lambda r: r['packed_source_key'])
    offsets_global = []
    cur = 0
    for r in rows_sorted_global:
        offsets_global.append(cur)
        cur += len(r['packed_list'])
    if cur != E_total:
        print("[warn] computed E mismatch (global):", cur, "!=", E_total)
        E_total = cur

    header_info_global = {'R': R_total, 'E': E_total, 'U': len(freq_total)}

    # Write the single combined dataset.bin (unchanged behavior)
    bin_path = os.path.join(outdir, 'dataset.bin')
    print("Writing single combined binary to:", bin_path)
    # Build global remap arrays for this file
    packed_vals_global, freqs_global, pv_to_idx_global, _items = build_remap(freq_total)
    crc, size = write_bin_file(bin_path, rows_sorted_global, offsets_global, header_info_global,
                               packed_vals_global, pv_to_idx_global, remap_width_global, lengths_type_global,
                               omit_offsets=args.omit_offsets)
    print(f"Wrote {bin_path} ({size} bytes) CRC32=0x{crc:08x}")

    # per-shard processing
    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min((shard_id + 1) * shard_size, R_total)
        shard_rows = rows[start:end]
        if not shard_rows:
            continue
        # For reader binary search inside shard, sort shard rows by packed_source_key
        shard_rows_sorted = sorted(shard_rows, key=lambda r: r['packed_source_key'])

        # Build per-shard frequency counter & remap table
        freq_shard = Counter()
        max_len_shard = 0
        E_shard = 0
        for r in shard_rows_sorted:
            for pv in r['packed_list']:
                freq_shard[pv] += 1
            if len(r['packed_list']) > max_len_shard:
                max_len_shard = len(r['packed_list'])
            E_shard += len(r['packed_list'])
        packed_vals_shard, freqs_shard, pv_to_idx_shard, _items_shard = build_remap(freq_shard)
        U_shard = len(packed_vals_shard)
        remap_width_shard = choose_remap_width(U_shard)
        lengths_type_shard = 0 if max_len_shard < 256 else 1

        # compute offsets (in indices) for shard
        offsets_shard = []
        cur_sh = 0
        for r in shard_rows_sorted:
            offsets_shard.append(cur_sh)
            cur_sh += len(r['packed_list'])
        if cur_sh != E_shard:
            print(f"[warn] computed E mismatch (shard {shard_id}):", cur_sh, "!=", E_shard)
            E_shard = cur_sh

        header_info_shard = {'R': len(shard_rows_sorted), 'E': E_shard, 'U': U_shard}

        shard_fname = f"shard-{shard_id:03d}.bin"
        shard_path = os.path.join(shards_dir, shard_fname)
        print(f"Writing shard {shard_id}: rows {start}..{end-1} -> {shard_path} (R={header_info_shard['R']}, E={E_shard}, U={U_shard})")

        crc_sh, size_sh = write_bin_file(shard_path, shard_rows_sorted, offsets_shard, header_info_shard,
                                         packed_vals_shard, pv_to_idx_shard, remap_width_shard, lengths_type_shard,
                                         omit_offsets=args.omit_offsets)
        print(f"  Wrote {shard_path} ({size_sh} bytes) CRC32=0x{crc_sh:08x}")

        # gzip and brotli for shard (optional)
        if not args.no_gzip:
            gz_path = shard_path + '.gz'
            with open(shard_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
                f_out.writelines(f_in)
            print("  Wrote gzip:", gz_path, "size:", os.path.getsize(gz_path))
        if have_brotli and not args.no_br:
            br_path = shard_path + '.br'
            with open(shard_path, 'rb') as f_in:
                data = f_in.read()
            with open(br_path, 'wb') as f_out:
                f_out.write(brotli.compress(data))
            print("  Wrote brotli:", br_path, "size:", os.path.getsize(br_path))

        # shard metadata for index
        first_packed = shard_rows_sorted[0]['packed_source_key']
        last_packed = shard_rows_sorted[-1]['packed_source_key']
        shards_meta.append({
            "shard_id": shard_id,
            "file": os.path.relpath(shard_path, outdir),
            "start_rank": start,
            "end_rank": end - 1,
            "entries": header_info_shard['R'],
            "first_packed_source_key": first_packed,
            "last_packed_source_key": last_packed,
            "file_size_bytes": size_sh,
            "estimated_expand_ram_bytes": (size_sh * 2)  # rough heuristic (you can tune)
        })

        # produce mapping entries packed_key -> shard_id
        for r in shard_rows_sorted:
            source_to_shard_pairs.append((r['packed_source_key'], shard_id))

        # write per-shard csv helpers
        remap_csv = os.path.join(shards_dir, f"remap_table_shard_{shard_id:03d}.csv")
        write_csv_remap(remap_csv, packed_vals_shard, freqs_shard)
        source_keys_csv = os.path.join(shards_dir, f"source_keys_shard_{shard_id:03d}.csv")
        write_csv_source_keys(source_keys_csv, shard_rows_sorted, offsets_shard)

    # write shards_index.json (sorted by shard_id)
    shards_index_path = os.path.join(shards_dir, 'shards_index.json')
    with open(shards_index_path, 'w', encoding='utf-8') as fh:
        json.dump({"shards": sorted(shards_meta, key=lambda s: s['shard_id'])}, fh, indent=2)
    print("Wrote shards_index.json:", shards_index_path)

    # write source_to_shard binary and CSV (sorted by packed_key)
    source_to_shard_pairs.sort(key=lambda x: x[0])
    s2s_bin = os.path.join(shards_dir, 'source_to_shard.bin')
    write_source_to_shard_bin(s2s_bin, source_to_shard_pairs)
    s2s_csv = os.path.join(shards_dir, 'source_to_shard.csv')
    with open(s2s_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['packed_source_key', 'shard_id'])
        for k, sid in source_to_shard_pairs:
            writer.writerow([k, sid])
    print("Wrote source_to_shard.bin and CSV in shards folder")

    # write top-level helper CSVs and metadata
    remap_csv_global = os.path.join(outdir, 'remap_table.csv')
    write_csv_remap(remap_csv_global, packed_vals_global, freqs_global)
    print("Wrote remap_table.csv (global)")

    source_keys_csv_global = os.path.join(outdir, 'source_keys.csv')
    write_csv_source_keys(source_keys_csv_global, rows_sorted_global, offsets_global)
    print("Wrote source_keys.csv (global)")

    metadata = {
        "bin": os.path.abspath(bin_path),
        "bin_size_bytes": os.path.getsize(bin_path),
        "rows": R_total,
        "total_similar_entries": E_total,
        "unique_similar_ids": len(freq_total),
        "remap_index_width_bytes": remap_width_global,
        "lengths_type": "uint8" if lengths_type_global == 0 else "uint16",
        "flags": (1 if args.omit_offsets else 0),
        "offsets_omitted": bool(args.omit_offsets),
        "shards_folder": os.path.relpath(shards_dir, outdir),
        "num_shards": num_shards
    }
    with open(os.path.join(outdir, 'metadata.json'), 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2)
    print("Wrote metadata.json")

    analysis_stats = {
        "freq_by_type": dict(freq_by_type.most_common()),
        "top_100_remap": [
            {"remap_index": i, "packed": packed_vals_global[i], "tmdb_id": packed_vals_global[i] >> 1,
             "type": ("tvshow" if (packed_vals_global[i] & 1) else "movie"), "frequency": freqs_global[i]}
            for i in range(min(100, len(packed_vals_global)))
        ]
    }
    with open(os.path.join(outdir, 'analysis_stats.json'), 'w', encoding='utf-8') as fh:
        json.dump(analysis_stats, fh, indent=2)
    print("Wrote analysis_stats.json")

    print("\nAll outputs written to:", outdir)
    print("Shards written to:", shards_dir)
    print("Shard index (shards_index.json) and source_to_shard.bin allow loader to pick the correct shard quickly.")
    print("Binary format: header + source_keys + offsets (if included) + lengths + remap_table + values_blob")
    print("Use --shards N to change number of shards (default 4).")
    return

if __name__ == '__main__':
    main()