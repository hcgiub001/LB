#!/usr/bin/env python3
"""
tmdb_packer_sharded_fast.py

Faster sharder version of the packer: incremental sharding (linear time).
"""
import os
import sys
import json
import struct
import gzip
import zlib
import csv
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
            src_id = obj.get('tmdb_id')
            if src_id is None:
                print(f"[warn] skipping line {line_no}: missing tmdb_id")
                continue
            src_type = obj.get('type', 'movie')
            similars = obj.get('similar', []) or []
            packed_list = []
            for s in similars:
                try:
                    sid = int(s.get('tmdb_id'))
                except Exception:
                    continue
                stype = s.get('type', 'movie')
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

def estimate_shard_size(rows_subset, omit_offsets=False):
    """
    Keep this for debugging parity — exact compute for a sublist.
    Not used inside the fast hot loop.
    """
    header_size = 28
    R = len(rows_subset)
    E = sum(len(r['packed_list']) for r in rows_subset)
    U = len({pv for r in rows_subset for pv in r['packed_list']})
    remap_width = choose_remap_width(U)
    max_list_len = 0
    for r in rows_subset:
        if len(r['packed_list']) > max_list_len:
            max_list_len = len(r['packed_list'])
    lengths_type = 0 if max_list_len < 256 else 1
    size = header_size
    size += R * 4  # source_keys
    if not omit_offsets:
        size += R * 4  # offsets
    size += R * (1 if lengths_type == 0 else 2)  # lengths
    size += U * 4  # remap_table
    size += E * remap_width  # values_blob
    return size

def shard_rows(rows_sorted, max_shard_bytes, omit_offsets=False):
    """
    Fast incremental sharder.

    For each shard we maintain:
      - cur_rows (list)
      - cur_R, cur_E (counts)
      - cur_freq (dict pv -> count)
      - cur_U (unique count)
      - cur_max_list_len
    We test adding the next row by computing the new sizes from these incremental values.
    """
    shards = []
    cur = []
    cur_R = 0
    cur_E = 0
    cur_freq = {}  # pv -> count
    cur_U = 0
    cur_max_len = 0

    header_size = 28
    def compute_size(R, E, U, max_len):
        size = header_size
        size += R * 4  # source_keys
        if not omit_offsets:
            size += R * 4  # offsets
        lengths_type = 0 if max_len < 256 else 1
        size += R * (1 if lengths_type == 0 else 2)
        size += U * 4  # remap table
        remap_w = choose_remap_width(U)
        size += E * remap_w
        return size

    total_rows = len(rows_sorted)
    printed = 0
    for idx, r in enumerate(rows_sorted):
        # compute incremental stats if we add r
        rlen = len(r['packed_list'])
        new_R = cur_R + 1
        new_E = cur_E + rlen
        # count how many new unique pvs this row would introduce
        new_unique = 0
        # iterate pvs once
        for pv in r['packed_list']:
            if cur_freq.get(pv, 0) == 0:
                new_unique += 1
        new_U = cur_U + new_unique
        new_max_len = cur_max_len if cur_max_len >= rlen else rlen

        new_size = compute_size(new_R, new_E, new_U, new_max_len)

        if new_size <= max_shard_bytes:
            # accept into current shard and update state
            cur.append(r)
            cur_R = new_R
            cur_E = new_E
            for pv in r['packed_list']:
                cur_freq[pv] = cur_freq.get(pv, 0) + 1
            cur_U = new_U
            cur_max_len = new_max_len
        else:
            if cur_R == 0:
                # single row itself exceeds shard limit - still accept it as a single shard
                shards.append([r])
                # cur remains empty
                cur = []
                cur_R = cur_E = cur_U = cur_max_len = 0
                cur_freq = {}
            else:
                # flush current shard
                shards.append(cur)
                # start new shard with this row
                cur = [r]
                cur_R = 1
                cur_E = rlen
                cur_freq = {}
                for pv in r['packed_list']:
                    cur_freq[pv] = cur_freq.get(pv, 0) + 1
                cur_U = len(cur_freq)
                cur_max_len = rlen

        # occasional progress print (every 2000 rows)
        if (idx - printed) >= 2000:
            printed = idx
            print(f"Sharding: processed {idx+1}/{total_rows} rows, planned shards so far: {len(shards) + (1 if cur else 0)}")

    if cur:
        shards.append(cur)
    return shards

def write_sharded_bins(outdir, base_name, shards, args):
    manifest = []
    for i, shard_rows in enumerate(shards):
        shard_id = f"shard_{i:03d}"
        shard_bin = os.path.join(outdir, f"{base_name}_{shard_id}.bin")
        # per-shard frequency / remap
        freq = Counter()
        for r in shard_rows:
            for pv in r['packed_list']:
                freq[pv] += 1
        packed_vals, freqs, pv_to_idx, items = build_remap(freq)
        U = len(packed_vals)
        E = sum(len(r['packed_list']) for r in shard_rows)
        R = len(shard_rows)
        remap_width = choose_remap_width(U)
        max_list_len = max((len(r['packed_list']) for r in shard_rows), default=0)
        lengths_type = 0 if max_list_len < 256 else 1

        # compute offsets (in indices)
        offsets = []
        cur_idx = 0
        for r in shard_rows:
            offsets.append(cur_idx)
            cur_idx += len(r['packed_list'])
        if cur_idx != E:
            E = cur_idx

        header_info = {'R': R, 'E': E, 'U': U}

        print(f"Writing shard {i+1}/{len(shards)} -> {os.path.basename(shard_bin)} R={R} E={E} U={U} remap_w={remap_width} lengths={'uint8' if lengths_type==0 else 'uint16'}")
        crc, size = write_bin_file(shard_bin, shard_rows, offsets, header_info,
                                   packed_vals, pv_to_idx, remap_width, lengths_type,
                                   omit_offsets=args.omit_offsets)
        # gzip
        gz_path = None
        if not args.no_gzip:
            gz_path = shard_bin + '.gz'
            with open(shard_bin, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
                f_out.writelines(f_in)
            gz_size = os.path.getsize(gz_path)
        else:
            gz_size = None

        # brotli
        br_path = None
        if have_brotli and not args.no_br:
            br_path = shard_bin + '.br'
            with open(shard_bin, 'rb') as f_in:
                data = f_in.read()
            with open(br_path, 'wb') as f_out:
                f_out.write(brotli.compress(data))
            br_size = os.path.getsize(br_path)
        else:
            br_size = None

        # per-shard CSVs and analysis
        remap_csv = os.path.join(outdir, f"{base_name}_{shard_id}_remap_table.csv")
        write_csv_remap(remap_csv, packed_vals, freqs)
        source_keys_csv = os.path.join(outdir, f"{base_name}_{shard_id}_source_keys.csv")
        write_csv_source_keys(source_keys_csv, shard_rows, offsets)

        shard_meta = {
            "shard_id": shard_id,
            "bin": os.path.abspath(shard_bin),
            "bin_size_bytes": size,
            "crc32": f"0x{crc:08x}",
            "gz": os.path.abspath(gz_path) if gz_path else None,
            "gz_size": gz_size,
            "br": os.path.abspath(br_path) if br_path else None,
            "br_size": br_size,
            "rows": R,
            "total_similar_entries": E,
            "unique_similar_ids": U,
            "remap_index_width_bytes": remap_width,
            "lengths_type": "uint8" if lengths_type == 0 else "uint16",
            "flags": (1 if args.omit_offsets else 0),
            "offsets_omitted": bool(args.omit_offsets),
            "remap_csv": os.path.abspath(remap_csv),
            "source_keys_csv": os.path.abspath(source_keys_csv)
        }
        manifest.append(shard_meta)
    return manifest

def write_manifest(outdir, manifest, global_meta):
    manifest_path = os.path.join(outdir, 'manifest.json')
    out = {
        "generated": True,
        "shards": manifest,
        "global": global_meta
    }
    with open(manifest_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print("Wrote manifest:", manifest_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pack NDJSON TMDB similar-lists into compact binary (sharded, fast).")
    parser.add_argument('ndjson', nargs='?', help='Path to NDJSON file. If omitted a file picker will be shown (if available).')
    parser.add_argument('--outdir', default='packed_output', help='Output folder')
    parser.add_argument('--no-gzip', action='store_true', help='Skip gzip output')
    parser.add_argument('--no-br', action='store_true', help='Skip brotli output even if available')
    parser.add_argument('--omit-offsets', action='store_true', help='Do not write offsets array to the binary (reader must reconstruct offsets from lengths)')
    parser.add_argument('--shard-bytes', type=int, default=512*1024, help='Target maximum shard size in bytes (default 524288)')
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

    print("Parsing NDJSON:", ndjson_path)
    rows, E_total, freq_global, freq_by_type, max_list_len = parse_ndjson(ndjson_path)
    R_total = len(rows)
    U_total = len(freq_global)
    print(f"Total rows R={R_total}, total similar entries E={E_total}, unique similar ids U={U_total}, max_list_len={max_list_len}")

    rows_sorted = sorted(rows, key=lambda r: r['packed_source_key'])

    print(f"Sharding into max {args.shard_bytes} bytes per shard (omit_offsets={args.omit_offsets})")
    shards = shard_rows(rows_sorted, args.shard_bytes, omit_offsets=args.omit_offsets)
    print(f"Planned {len(shards)} shard(s)")

    base_name = 'dataset'
    manifest = write_sharded_bins(outdir, base_name, shards, args)

    global_meta = {
        "input_ndjson": os.path.abspath(ndjson_path),
        "rows": R_total,
        "total_similar_entries": E_total,
        "unique_similar_ids": U_total,
        "requested_shard_bytes": args.shard_bytes,
        "num_shards": len(manifest),
        "omit_offsets_used": bool(args.omit_offsets)
    }
    write_manifest(outdir, manifest, global_meta)

    # summary analysis stats (global, based on all data)
    analysis_stats = {
        "freq_by_type": dict(freq_by_type.most_common()),
        "top_100_remap_global": [
            {"packed": packed, "tmdb_id": packed >> 1, "type": ("tvshow" if packed & 1 else "movie"), "frequency": freq}
            for packed, freq in freq_global.most_common(100)
        ]
    }
    with open(os.path.join(outdir, 'analysis_stats.json'), 'w', encoding='utf-8') as fh:
        json.dump(analysis_stats, fh, indent=2)
    print("Wrote analysis_stats.json")
    print("\nAll outputs written to:", outdir)
    return

if __name__ == '__main__':
    main()