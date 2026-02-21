#!/usr/bin/env python3
"""
tmdb_packer.py

Pack NDJSON of TMDB similar-lists into compact binary, with optional sharding
and pre-compressed outputs (gzip / brotli). This script writes:

- remap.bin             (global remap table: uint32 packed values)
- shard-000.bin ...     (sharded dataset .bin files, header + keys + offsets + lengths + values_blob)
- shard-000.bin.gz/.br  (optional)
- dataset.bin           (optional full dataset, for compatibility)
- dataset.bin.gz/.br
- remap_table.csv
- source_keys.csv
- shards_index.json
- metadata.json
- analysis_stats.json

Usage examples:
  python tmdb_packer.py data.ndjson
  python tmdb_packer.py data.ndjson --target-shard-bytes 262144 --no-br
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

# -------------------------
# remap + shard writers
# -------------------------
def write_remap_file(outdir, packed_vals):
    """
    Write remap.bin: magic 'REMP' + uint32 count + U * uint32 packed values
    """
    out = os.path.join(outdir, 'remap.bin')
    with open(out, 'wb') as fh:
        fh.write(b'REMP')                       # 4 bytes magic
        fh.write(struct.pack('<I', len(packed_vals)))  # 4 bytes: U
        for pv in packed_vals:
            fh.write(struct.pack('<I', pv))
    return out, os.path.getsize(out)

def estimate_row_size(row, remap_width, lengths_type, include_offsets=True):
    # estimate per-row storage inside a shard (bytes)
    # source_key: 4, offset: 4 (if included), length: 1 or 2, values: remap_width * list_len
    s = 4
    if include_offsets:
        s += 4
    s += 1 if lengths_type == 0 else 2
    s += remap_width * len(row['packed_list'])
    return s

def patch_crc(filepath):
    # Compute CRC across file with placeholder zero located at header offset 24 (like original format)
    # If file is too small or header not present, skip.
    try:
        with open(filepath, 'rb') as fh:
            data = fh.read()
        crc = zlib.crc32(data) & 0xFFFFFFFF
        with open(filepath, 'r+b') as fh:
            if os.path.getsize(filepath) >= 28:
                fh.seek(24)
                fh.write(struct.pack('<I', crc))
        return crc
    except Exception as e:
        print("[warn] failed to patch CRC for", filepath, ":", e)
        return None

def gzip_file(src, keep_original=True):
    gz_path = src + '.gz'
    with open(src, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        f_out.writelines(f_in)
    if not keep_original:
        os.remove(src)
    return gz_path, os.path.getsize(gz_path)

def brotli_file(src, keep_original=True, quality=11):
    br_path = src + '.br'
    with open(src, 'rb') as f_in:
        data = f_in.read()
    comp = brotli.compress(data, quality=quality)
    with open(br_path, 'wb') as f_out:
        f_out.write(comp)
    if not keep_original:
        os.remove(src)
    return br_path, os.path.getsize(br_path)

def write_shards(outdir, rows_sorted, packed_vals, pv_to_idx, remap_width, lengths_type,
                 target_shard_bytes=512*1024, include_offsets=True,
                 produce_gzip=True, produce_br=True, keep_original=True, shard_prefix='shard'):
    """
    Partition rows_sorted into shards aiming for target_shard_bytes (estimated).
    Each shard is written as header+source_keys+offsets+lengths+values_blob (remap indices).
    Returns list of shard metadata dicts and index_path.
    """
    ensure_dir(outdir)
    shards_meta = []
    cur_shard_rows = []
    cur_shard_bytes = 0
    shard_id = 0

    def flush_shard(shard_rows, sid):
        if not shard_rows:
            return None
        fname = f"{shard_prefix}-{sid:03d}.bin"
        path = os.path.join(outdir, fname)
        # header same layout as original format
        magic = b"SIML"
        version = 1; endian = 1; flags = 0 if include_offsets else 1
        R = len(shard_rows)
        E = sum(len(r['packed_list']) for r in shard_rows)
        U = len(packed_vals)
        lengths_byte = 0 if lengths_type == 0 else 1
        remap_index_width = remap_width
        reserved = 0
        header_crc_placeholder = 0
        header = struct.pack('<4s B B H I I I B B H I',
                             magic, version, endian, flags, R, E, U,
                             lengths_byte, remap_index_width, reserved, header_crc_placeholder)
        assert len(header) == 28
        with open(path, 'wb') as fh:
            fh.write(header)
            # source_keys
            for r in shard_rows:
                fh.write(struct.pack('<I', r['packed_source_key']))
            # offsets (indices) if included
            offsets = []
            cur = 0
            for r in shard_rows:
                offsets.append(cur)
                cur += len(r['packed_list'])
            if include_offsets:
                for off in offsets:
                    fh.write(struct.pack('<I', off))
            # lengths
            if lengths_type == 0:
                for r in shard_rows:
                    l = len(r['packed_list'])
                    if l >= 256:
                        raise ValueError("List length exceeds 255 but lengths_type is uint8")
                    fh.write(struct.pack('<B', l))
            else:
                for r in shard_rows:
                    fh.write(struct.pack('<H', len(r['packed_list'])))
            # remap indices (values_blob)
            for r in shard_rows:
                for pv in r['packed_list']:
                    idx = pv_to_idx[pv]
                    write_uint_bytes(fh, idx, remap_index_width)

        crc = patch_crc(path)
        size = os.path.getsize(path)

        gz_info = None
        br_info = None
        if produce_gzip:
            gz_path, gz_size = gzip_file(path, keep_original=keep_original)
            gz_info = {'path': os.path.basename(gz_path), 'size': gz_size}
        if produce_br and have_brotli:
            br_path, br_size = brotli_file(path, keep_original=keep_original)
            br_info = {'path': os.path.basename(br_path), 'size': br_size}

        return {
            'filename': fname,
            'rows': len(shard_rows),
            'entries': E,
            'size': size,
            'crc32': (crc if crc is not None else 0),
            'min_key': shard_rows[0]['packed_source_key'],
            'max_key': shard_rows[-1]['packed_source_key'],
            'gz': gz_info,
            'br': br_info
        }

    # accumulate rows until estimated bytes exceed target_shard_bytes
    for r in rows_sorted:
        est = estimate_row_size(r, remap_width, lengths_type, include_offsets)
        if cur_shard_rows and (cur_shard_bytes + est > target_shard_bytes):
            meta = flush_shard(cur_shard_rows, shard_id)
            if meta:
                shards_meta.append(meta)
            shard_id += 1
            cur_shard_rows = []
            cur_shard_bytes = 0
        cur_shard_rows.append(r)
        cur_shard_bytes += est

    # flush last shard
    meta = flush_shard(cur_shard_rows, shard_id)
    if meta:
        shards_meta.append(meta)

    # write shards index JSON
    index_path = os.path.join(outdir, 'shards_index.json')
    with open(index_path, 'w', encoding='utf-8') as fh:
        json.dump({'shards': shards_meta, 'remap_file': 'remap.bin'}, fh, indent=2)
    return shards_meta, index_path

# -------------------------
# (optional) combined dataset.bin writer (compat)
# -------------------------
def write_combined_bin(out_path, rows_sorted, offsets, header_info, packed_vals, pv_to_idx, remap_width, lengths_type, omit_offsets=False):
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

    crc = patch_crc(out_path)
    size = os.path.getsize(out_path)
    return crc, size

# -------------------------
# main CLI
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pack NDJSON TMDB similar-lists into compact binary with sharding.")
    parser.add_argument('ndjson', nargs='?', help='Path to NDJSON file. If omitted a file picker will be shown (if available).')
    parser.add_argument('--outdir', default='packed_output', help='Output folder')
    parser.add_argument('--no-gzip', action='store_true', help='Skip gzip output')
    parser.add_argument('--no-br', action='store_true', help='Skip brotli output even if available')
    parser.add_argument('--omit-offsets', action='store_true', help='Do not write offsets array to the binary (reader must reconstruct offsets from lengths)')
    parser.add_argument('--target-shard-bytes', type=int, default=512*1024, help='Approx target size (bytes) per shard (estimation)')
    parser.add_argument('--keep-original', action='store_true', help='Keep original .bin files in addition to .gz/.br (default: keep originals)')
    parser.add_argument('--write-dataset', action='store_true', help='Also write combined dataset.bin for backward compatibility')
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
    rows, E, freq, freq_by_type, max_list_len = parse_ndjson(ndjson_path)
    R = len(rows)
    U = len(freq)
    print(f"Rows R={R}, total similar entries E={E}, unique similar ids U={U}, max_list_len={max_list_len}")

    print("Building remap table (sorted by frequency)...")
    packed_vals, freqs, pv_to_idx, items = build_remap(freq)

    remap_width = choose_remap_width(U)  # 2/3/4 bytes
    lengths_type = 0 if max_list_len < 256 else 1

    # Sort rows by source key, compute offsets (in indices) for full dataset (used by combined writer)
    rows_sorted = sorted(rows, key=lambda r: r['packed_source_key'])
    offsets = []
    cur = 0
    for r in rows_sorted:
        offsets.append(cur)
        cur += len(r['packed_list'])
    if cur != E:
        print("[warn] computed E mismatch:", cur, "!=", E)
        E = cur  # fix

    header_info = {'R': R, 'E': E, 'U': U}

    # write remap.bin
    remap_path, remap_size = write_remap_file(outdir, packed_vals)
    print("Wrote remap:", remap_path, "size:", remap_size)

    # write shards
    print("Writing shards (target bytes ~ {})".format(args.target_shard_bytes))
    shards_meta, index_path = write_shards(
        outdir, rows_sorted, packed_vals, pv_to_idx, remap_width, lengths_type,
        target_shard_bytes=args.target_shard_bytes,
        include_offsets=(not args.omit_offsets),
        produce_gzip=(not args.no_gzip),
        produce_br=(not args.no_br),
        keep_original=args.keep_original
    )
    print("Wrote shards index:", index_path)
    for s in shards_meta:
        print("  ", s['filename'], "rows:", s['rows'], "entries:", s['entries'], "size:", s['size'],
              "crc:0x{:08x}".format(s['crc32'] if s.get('crc32') else 0),
              "gz:", s.get('gz', None), "br:", s.get('br', None))

    # optional combined dataset (for backward compatibility)
    if args.write_dataset:
        bin_path = os.path.join(outdir, 'dataset.bin')
        print("Writing combined dataset to:", bin_path, " (omit_offsets=" + str(args.omit_offsets) + ")")
        crc, size = write_combined_bin(bin_path, rows_sorted, offsets, header_info, packed_vals, pv_to_idx, remap_width, lengths_type, omit_offsets=args.omit_offsets)
        print(f"Wrote {bin_path} ({size} bytes) CRC32=0x{crc:08x}")
        if not args.no_gzip:
            gz_path = bin_path + '.gz'
            with open(bin_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
                f_out.writelines(f_in)
            print("Wrote gzip:", gz_path, "size:", os.path.getsize(gz_path))
        if have_brotli and not args.no_br:
            br_path = bin_path + '.br'
            with open(bin_path, 'rb') as f_in:
                data = f_in.read()
            with open(bin_path + '.br', 'wb') as f_out:
                f_out.write(brotli.compress(data))
            print("Wrote brotli:", br_path, "size:", os.path.getsize(br_path))

    # helper CSVs and metadata
    remap_csv = os.path.join(outdir, 'remap_table.csv')
    write_csv_remap(remap_csv, packed_vals, freqs)
    print("Wrote remap_table.csv")

    source_keys_csv = os.path.join(outdir, 'source_keys.csv')
    # For CSV of source keys, we will write per-shard offset index = index within values array for the combined dataset (best-effort)
    write_csv_source_keys(source_keys_csv, rows_sorted, offsets)
    print("Wrote source_keys.csv")

    metadata = {
        "outdir": outdir,
        "remap": os.path.basename(remap_path),
        "remap_size_bytes": remap_size,
        "shards_index": os.path.basename(index_path),
        "shards_count": len(shards_meta),
        "rows": R,
        "total_similar_entries": E,
        "unique_similar_ids": U,
        "remap_index_width_bytes": remap_width,
        "lengths_type": "uint8" if lengths_type == 0 else "uint16",
        "flags": (1 if args.omit_offsets else 0),
        "offsets_omitted": bool(args.omit_offsets),
        "target_shard_bytes": args.target_shard_bytes
    }
    with open(os.path.join(outdir, 'metadata.json'), 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2)
    print("Wrote metadata.json")

    analysis_stats = {
        "freq_by_type": dict(freq_by_type.most_common()),
        "top_100_remap": [
            {"remap_index": i, "packed": packed_vals[i], "tmdb_id": packed_vals[i] >> 1,
             "type": ("tvshow" if (packed_vals[i] & 1) else "movie"), "frequency": freqs[i]}
            for i in range(min(100, len(packed_vals)))
        ]
    }
    with open(os.path.join(outdir, 'analysis_stats.json'), 'w', encoding='utf-8') as fh:
        json.dump(analysis_stats, fh, indent=2)
    print("Wrote analysis_stats.json")

    print("\nAll outputs written to:", outdir)
    print("Shard count:", len(shards_meta))
    print("Remap width (bytes):", remap_width)
    print("Lengths type:", metadata['lengths_type'])
    if args.omit_offsets:
        print("Note: offsets were omitted in shards; readers must reconstruct offsets from lengths (flags bit0 == 1).")
    else:
        print("Note: offsets included in shards (flags bit0 == 0); readers can create zero-copy Uint32Array views.")
    return

if __name__ == '__main__':
    main()