#!/usr/bin/env python3
"""
tmdb_packer_patched.py

Updated packer that keeps header + source_keys + offsets raw (zero-copy),
and writes a compressed tail containing: lengths, remap_table (3- or 4-byte
entries), a packed type bitfield, and values_blob encoded as LEB128 (varint).

Tail is compressed with zlib (level=9) and appended, followed by a 16-byte
TAIL footer describing compression. Header.flags bit0==1 => offsets omitted,
bit1==1 => tail is compressed, remap_index_width==0 => values_blob uses varints.

This version is tuned to allow a compact on-disk file while preserving
zero-copy offsets; the loader can mmap the file, read offsets as uint32, read
and decompress the tail once, parse into compact C-style buffers (array,
bytearray) and then free temp buffers.

Usage: same CLI as before. Produces dataset.bin (with compressed tail),
remap_table.csv, source_keys.csv, metadata.json and analysis_stats.json.
"""

import os
import sys
import json
import struct
import gzip
import zlib
import csv
import io
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


def _write_uvarint_to_buf(buf, n):
    # LEB128 unsigned
    while True:
        to_write = n & 0x7F
        n >>= 7
        if n:
            buf.write(bytes((to_write | 0x80,)))
        else:
            buf.write(bytes((to_write,)))
            break


def write_bin_file(out_path, rows_sorted, offsets, header_info, packed_vals, pv_to_idx, remap_width_guess, lengths_type, omit_offsets=False, use_brotli=False):
    R = header_info['R']; E = header_info['E']; U = header_info['U']
    header_size = 28
    magic = b"SIML"
    version = 1
    endian = 1
    # flags bit0 indicates offsets omitted when set; bit1 indicates tail compressed
    flags = 1 if omit_offsets else 0
    lengths_byte = 0 if lengths_type == 0 else 1
    # We'll set remap_index_width==0 to indicate values_blob uses varints.
    remap_index_width = 0
    reserved = 0
    header_crc_placeholder = 0
    header = struct.pack('<4s B B H I I I B B H I',
                         magic, version, endian, flags, R, E, U,
                         lengths_byte, remap_index_width, reserved, header_crc_placeholder)
    assert len(header) == header_size

    # Decide remap table entry width (3 or 4 bytes) based on max packed value
    max_pv = max(packed_vals) if packed_vals else 0
    use_3byte_remap = (max_pv <= 0xFFFFFF)
    remap_table_entry_bytes = 3 if use_3byte_remap else 4

    # Build tail in memory: lengths + remap_table_width_marker + remap_table + type_bitfield + values_blob (uvarints)
    tail_buf = io.BytesIO()

    # 1) lengths
    if lengths_type == 0:
        for r in rows_sorted:
            l = len(r['packed_list'])
            if l >= 256:
                raise ValueError("List length exceeds 255 but lengths_type is uint8")
            tail_buf.write(struct.pack('<B', l))
    else:
        for r in rows_sorted:
            l = len(r['packed_list'])
            tail_buf.write(struct.pack('<H', l))

    # 2) remap_table width marker (1 byte): 1 => 3-byte entries, 0 => 4-byte entries
    tail_buf.write(struct.pack('<B', 1 if use_3byte_remap else 0))

    # 3) remap_table entries
    if use_3byte_remap:
        for pv in packed_vals:
            v = pv & 0xFFFFFF
            tail_buf.write(bytes((v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF)))
    else:
        for pv in packed_vals:
            tail_buf.write(struct.pack('<I', pv))

    # 4) type bitfield (U bits)
    bitfield_bytes = (U + 7) // 8
    tb = bytearray(bitfield_bytes)
    for idx, pv in enumerate(packed_vals):
        # pv LSB is type bit
        bit = pv & 1
        if bit:
            tb[idx >> 3] |= (1 << (idx & 7))
    tail_buf.write(tb)

    # 5) values_blob: write remap indices as LEB128 varints
    for r in rows_sorted:
        for pv in r['packed_list']:
            idx = pv_to_idx[pv]
            _write_uvarint_to_buf(tail_buf, idx)

    tail_bytes = tail_buf.getvalue()

    # Compress tail (zlib by default, brotli if requested and available)
    if use_brotli and have_brotli:
        comp_bytes = brotli.compress(tail_bytes)
        comp_type = 2
    else:
        comp_bytes = zlib.compress(tail_bytes, level=9)
        comp_type = 1

    # Set compressed tail flag in header.flags
    flags |= 2
    # rewrite header with updated flags
    header = struct.pack('<4s B B H I I I B B H I',
                         magic, version, endian, flags, R, E, U,
                         lengths_byte, remap_index_width, reserved, header_crc_placeholder)
    assert len(header) == header_size

    # Write file: header + source_keys + offsets + compressed_tail + TAIL footer
    with open(out_path, 'wb') as fh:
        fh.write(header)

        # source_keys (R * uint32)
        for r in rows_sorted:
            fh.write(struct.pack('<I', r['packed_source_key']))

        # OFFSETS: optional (omitted if omit_offsets True)
        if not omit_offsets:
            for off in offsets:
                fh.write(struct.pack('<I', off))

        # Append compressed tail
        fh.write(comp_bytes)

        # TAIL footer: 4s B 3x I I
        tail_magic = b'TAIL'
        comp_len = len(comp_bytes)
        uncomp_len = len(tail_bytes)
        fh.write(struct.pack('<4s B 3x I I', tail_magic, comp_type, comp_len, uncomp_len))

    # Compute CRC of file (with placeholder zero in header) and patch it into header offset 24
    with open(out_path, 'rb') as fh:
        data = fh.read()
    crc = zlib.crc32(data) & 0xFFFFFFFF
    with open(out_path, 'r+b') as fh:
        fh.seek(24)
        fh.write(struct.pack('<I', crc))

    return crc, os.path.getsize(out_path), remap_table_entry_bytes, comp_type


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pack NDJSON TMDB similar-lists into compact binary.")
    parser.add_argument('ndjson', nargs='?', help='Path to NDJSON file. If omitted a file picker will be shown (if available).')
    parser.add_argument('--outdir', default='packed_output', help='Output folder')
    parser.add_argument('--no-gzip', action='store_true', help='Skip gzip output')
    parser.add_argument('--no-br', action='store_true', help='Skip brotli output even if available')
    parser.add_argument('--omit-offsets', action='store_true', help='Do not write offsets array to the binary (reader must reconstruct offsets from lengths)')
    parser.add_argument('--brotli', action='store_true', help='Use brotli for tail compression if available')
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

    remap_width = choose_remap_width(U)  # suggested width (not used for values_blob when varints used)
    lengths_type = 0 if max_list_len < 256 else 1

    # Sort rows by source key, compute offsets (in indices)
    rows_sorted = sorted(rows, key=lambda r: r['packed_source_key'])
    offsets = []
    cur = 0
    for r in rows_sorted:
        offsets.append(cur)
        cur += len(r['packed_list'])
    if cur != E:
        print("[warn] computed E mismatch:", cur, "!=" , E)
        E = cur  # fix

    header_info = {'R': R, 'E': E, 'U': U}

    bin_path = os.path.join(outdir, 'dataset.bin')
    print("Writing binary to:", bin_path, " (omit_offsets=" + str(args.omit_offsets) + ")")
    crc, size, remap_entry_bytes, comp_type = write_bin_file(bin_path, rows_sorted, offsets, header_info, packed_vals, pv_to_idx, remap_width, lengths_type, omit_offsets=args.omit_offsets, use_brotli=(args.brotli and have_brotli))
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
        "remap_table_entry_bytes": remap_entry_bytes,
        "values_blob_encoding": "varint-LEB128",
        "lengths_type": "uint8" if lengths_type == 0 else "uint16",
        "flags": (1 if args.omit_offsets else 0) | 2,  # bit1 set for compressed tail
        "offsets_omitted": bool(args.omit_offsets),
        "tail_compression": ("brotli" if (comp_type == 2) else "zlib")
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
    print("Binary format: header + source_keys + offsets (if included) + compressed_tail + TAIL_footer")
    print("Remap table entry bytes (stored inside tail):", remap_entry_bytes)
    print("Values blob encoding:", "LEB128 varint")
    if args.omit_offsets:
        print("Note: offsets were omitted in this build; readers must reconstruct offsets from lengths (flags bit0 == 1).")
    else:
        print("Note: offsets included on disk (flags bit0 == 0); readers can create zero-copy Uint32Array views.")
    print("Note: tail compressed (flags bit1 == 1). Readers must read TAIL footer to locate and decompress tail.")
    return


if __name__ == '__main__':
    main()
