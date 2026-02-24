#!/usr/bin/env python3
"""
tmdb_packer.py
Pack NDJSON of TMDB similar-lists into a compact binary format optimized for Tampermonkey.

This variant includes offsets by default to allow zero-copy readers to create a Uint32Array
view of the offsets block (avoiding an extra allocation and the small reconstruction pass).

If you prefer the smaller raw .bin you can still pass --omit-offsets on the command line;
that will set header.flags bit0=1 and readers must reconstruct offsets from lengths.
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


def write_bin_file(out_path, rows_sorted, offsets, header_info,
                   packed_vals, pv_to_idx, remap_width, lengths_type,
                   omit_offsets=False):
    R = header_info['R']
    E = header_info['E']
    U = header_info['U']

    header_size = 28
    magic = b"SIML"
    version = 1
    endian = 1
    # flags bit0 indicates offsets omitted when set
    flags = 1 if omit_offsets else 0
    lengths_byte = 0 if lengths_type == 0 else 1
    remap_index_width = remap_width
    reserved = 0
    header_crc_placeholder = 0

    header = struct.pack('<4s B B H I I I B B H I',
                         magic, version, endian, flags,
                         R, E, U,
                         lengths_byte, remap_index_width,
                         reserved, header_crc_placeholder)
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
            writer.writerow([r['source_tmdb_id'], r['source_type'],
                             r['packed_source_key'], off, len(r['packed_list'])])


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Pack NDJSON TMDB similar-lists into compact binary.")
    parser.add_argument('ndjson', nargs='?',
                        help='Path to NDJSON file. If omitted a file picker will be shown (if available).')
    parser.add_argument('--outdir', default='packed_output', help='Output folder')
    parser.add_argument('--no-gzip', action='store_true', help='Skip gzip output')
    parser.add_argument('--no-br', action='store_true', help='Skip brotli output even if available')
    parser.add_argument('--omit-offsets', action='store_true',
                        help='Do not write offsets array to the binary (reader must reconstruct offsets from lengths)')
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

    # Sort rows by source key, compute offsets (in indices)
    rows_sorted = sorted(rows, key=lambda r: r['packed_source_key'])
    offsets = []
    cur = 0
    for r in rows_sorted:
        offsets.append(cur)
        cur += len(r['packed_list'])
    if cur != E:
        print("[warn] computed E mismatch:", cur, "!=", E)
        E = cur

    header_info = {'R': R, 'E': E, 'U': U}
    bin_path = os.path.join(outdir, 'dataset.bin')
    print("Writing binary to:", bin_path, " (omit_offsets=" + str(args.omit_offsets) + ")")
    crc, size = write_bin_file(bin_path, rows_sorted, offsets, header_info,
                               packed_vals, pv_to_idx, remap_width, lengths_type,
                               omit_offsets=args.omit_offsets)
    print(f"Wrote {bin_path} ({size} bytes) CRC32=0x{crc:08x}")

    # gzip
    if not args.no_gzip:
        gz_path = bin_path + '.gz'
        with open(bin_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
            f_out.writelines(f_in)
        print("Wrote gzip:", gz_path, "size:", os.path.getsize(gz_path))

    # brotli
    if have_brotli and not args.no_br:
        br_path = bin_path + '.br'
        with open(bin_path, 'rb') as f_in:
            data = f_in.read()
        with open(bin_path + '.br', 'wb') as f_out:
            f_out.write(brotli.compress(data))
        print("Wrote brotli:", br_path, "size:", os.path.getsize(br_path))

    # Write helper CSVs and JSON metadata
    remap_csv = os.path.join(outdir, 'remap_table.csv')
    write_csv_remap(remap_csv, packed_vals, freqs)
    print("Wrote remap_table.csv")

    source_keys_csv = os.path.join(outdir, 'source_keys.csv')
    write_csv_source_keys(source_keys_csv, rows_sorted, offsets)
    print("Wrote source_keys.csv")

    metadata = {
        "bin": os.path.abspath(bin_path),
        "bin_size_bytes": os.path.getsize(bin_path),
        "rows": R,
        "total_similar_entries": E,
        "unique_similar_ids": U,
        "remap_index_width_bytes": remap_width,
        "lengths_type": "uint8" if lengths_type == 0 else "uint16",
        "flags": (1 if args.omit_offsets else 0),
        "offsets_omitted": bool(args.omit_offsets)
    }
    with open(os.path.join(outdir, 'metadata.json'), 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2)
    print("Wrote metadata.json")

    analysis_stats = {
        "freq_by_type": dict(freq_by_type.most_common()),
        "top_100_remap": [
            {
                "remap_index": i,
                "packed": packed_vals[i],
                "tmdb_id": packed_vals[i] >> 1,
                "type": ("tvshow" if (packed_vals[i] & 1) else "movie"),
                "frequency": freqs[i]
            }
            for i in range(min(100, len(packed_vals)))
        ]
    }
    with open(os.path.join(outdir, 'analysis_stats.json'), 'w', encoding='utf-8') as fh:
        json.dump(analysis_stats, fh, indent=2)
    print("Wrote analysis_stats.json")

    print("\nAll outputs written to:", outdir)
    print("Binary format: header + source_keys + offsets (if included) + lengths + remap_table + values_blob")
    print("Remap width (bytes):", remap_width)
    print("Lengths type:", metadata['lengths_type'])
    if args.omit_offsets:
        print("Note: offsets were omitted; readers must reconstruct offsets from lengths (flags bit0 == 1).")
    else:
        print("Note: offsets included on disk (flags bit0 == 0); readers can create zero-copy Uint32Array views.")


if __name__ == '__main__':
    main()