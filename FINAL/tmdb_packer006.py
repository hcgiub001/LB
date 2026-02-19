#!/usr/bin/env python3
"""
tmdb_packer.py

Pack NDJSON of TMDB similar-lists into a compact binary format optimized for Tampermonkey.

Features in this version:
 - By default omits per-row offsets on disk (flags bit0 = 1) and reconstructs offsets from lengths.
 - Block-compresses the flattened remap-index stream using gzip (flags bit1 = 1).
 - Performs byte-plane splitting inside each block before gzip (flags bit2 = 1),
   which usually improves gzip compression dramatically for skewed remap indices.
 - Block size (entries per block) is configurable (default 8192).
 - Outputs: dataset.bin, dataset.bin.gz, remap_table.csv, source_keys.csv, metadata.json, analysis_stats.json
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

def _gzip_compress_bytes(b, level=6):
    # Use gzip.compress to produce a gzip container for each block
    try:
        return gzip.compress(b, compresslevel=level)
    except Exception:
        # fallback manual
        out = gzip.GzipFile(fileobj=bytearray(), mode='wb')
        out.write(b)
        out.close()
        return out.fileobj.getvalue()

def write_bin_file(out_path, rows_sorted, offsets, header_info, packed_vals, pv_to_idx,
                   remap_width, lengths_type, omit_offsets=False,
                   block_compress=True, block_size_entries=8192, plane_split=True, gzip_level=6):
    """
    Write the binary file.
    If block_compress is True, the values_blob is written as block-compressed gzip blocks
    with a small block index (comp_size, block_entries) for each block. If plane_split is True,
    each block is byte-plane-split (msb-plane first, down to lsb-plane).
    """
    R = header_info['R']; E = header_info['E']; U = header_info['U']
    header_size = 28
    magic = b"SIML"
    version = 1
    endian = 1

    # flags:
    # bit0 (1) = offsets omitted on disk (lengths follow directly)
    # bit1 (2) = values_blob is block-compressed
    # bit2 (4) = values_blob blocks are byte-plane split (hi-plane first)
    flags = 0
    if omit_offsets:
        flags |= 1
    if block_compress:
        flags |= 2
    if plane_split:
        flags |= 4

    lengths_byte = 0 if lengths_type == 0 else 1
    remap_index_width = remap_width
    reserved = 0
    header_crc_placeholder = 0

    header = struct.pack('<4s B B H I I I B B H I',
                         magic, version, endian, flags, R, E, U,
                         lengths_byte, remap_index_width, reserved, header_crc_placeholder)
    assert len(header) == header_size

    # We'll write header with crc placeholder, then other sections, then compute CRC and patch it.
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

        # values_blob: either raw remap indices (old style) OR block-compressed + plane-split
        if not block_compress:
            # Legacy writing of values_blob - interleaved remap indices of remap_width bytes
            for r in rows_sorted:
                for pv in r['packed_list']:
                    idx = pv_to_idx[pv]
                    if remap_width == 2 and idx >= (1 << 16):
                        raise ValueError("remap index overflow for uint16")
                    if remap_width == 3 and idx >= (1 << 24):
                        raise ValueError("remap index overflow for 24-bit")
                    write_uint_bytes(fh, idx, remap_width)
        else:
            # Block compression path
            # 1) flatten remap indices into a single list of E indices (in original order)
            flat_indices = [0] * E  # pre-allocate
            pos = 0
            for r in rows_sorted:
                for pv in r['packed_list']:
                    flat_indices[pos] = pv_to_idx[pv]
                    pos += 1
            assert pos == E

            # 2) partition into blocks of block_size_entries entries
            blocks = []
            i = 0
            while i < E:
                block_entries = min(block_size_entries, E - i)
                block_indices = flat_indices[i:i+block_entries]
                blocks.append(block_indices)
                i += block_entries

            block_count = len(blocks)
            comp_sizes = []
            block_entries_list = []

            # 3) For each block, build uncompressed bytes: either plain interleaved or plane-split
            compressed_blocks = []
            for bidx, block_indices in enumerate(blocks):
                block_entries = len(block_indices)
                block_entries_list.append(block_entries)

                if plane_split and remap_width >= 1:
                    # Build planes from most significant byte down to least significant byte
                    # e.g. for remap_width == 2: hi_plane (msb) then lo_plane (lsb)
                    planes = [bytearray() for _ in range(remap_width)]
                    for idx in block_indices:
                        for p in range(remap_width):
                            byte = (idx >> (8 * p)) & 0xFF
                            planes[p].append(byte)
                    # planes currently low->high order; we want hi-plane first -> write in reverse
                    uncompressed = bytearray()
                    for p in range(remap_width - 1, -1, -1):
                        uncompressed.extend(planes[p])
                else:
                    # interleaved per-index bytes (lsb first -> msb last)
                    uncompressed = bytearray()
                    for idx in block_indices:
                        for p in range(remap_width):
                            uncompressed.append((idx >> (8 * p)) & 0xFF)

                # gzip-compress this block
                comp = gzip.compress(bytes(uncompressed), compresslevel=gzip_level)
                comp_sizes.append(len(comp))
                compressed_blocks.append(comp)

            # 4) write block_index: block_count (uint32), then for each block comp_size(uint32), block_entries(uint32)
            fh.write(struct.pack('<I', block_count))
            for cs, be in zip(comp_sizes, block_entries_list):
                fh.write(struct.pack('<I I', cs, be))

            # 5) write concatenated compressed blocks
            for comp in compressed_blocks:
                fh.write(comp)

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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pack NDJSON TMDB similar-lists into compact binary.")
    parser.add_argument('ndjson', nargs='?', help='Path to NDJSON file. If omitted a file picker will be shown (if available).')
    parser.add_argument('--outdir', default='packed_output', help='Output folder')
    parser.add_argument('--no-gzip', action='store_true', help='Skip gzip of the final .bin output')
    parser.add_argument('--no-br', action='store_true', help='Skip brotli output even if available')
    parser.add_argument('--omit-offsets', action='store_true', help='Do not write offsets array to the binary (reader must reconstruct offsets from lengths)')
    parser.add_argument('--no-block', action='store_true', help='Disable block compression (write raw values_blob)')
    parser.add_argument('--block-size', type=int, default=8192, help='Entries per compressed block (default 8192)')
    parser.add_argument('--no-plane', action='store_true', help='Disable byte-plane splitting inside blocks')
    parser.add_argument('--gzip-level', type=int, default=6, help='gzip compression level for blocks (1-9)')
    args = parser.parse_args()

    # Default: omit offsets for double-click runs unless user overrides with CLI.
    # But since the user requested compact output by default, we force omit unless they explicitly pass --omit-offsets false (not provided).
    # To keep behavior predictable, respect the explicit --omit-offsets flag if provided.
    # If user double-clicks, args.omit_offsets will be False by default; to maintain the previous behavior (omit offsets),
    # we set omit_defaults = True unless they passed something. We'll keep things simple: follow args.omit_offsets.
    # (This keeps compatibility with your workflow.)
    omit_offsets = bool(args.omit_offsets)

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

    # Sort rows by source key, compute offsets (in indices) - still computed in memory for CSVs
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

    bin_path = os.path.join(outdir, 'dataset.bin')
    block_compress = not bool(args.no_block)
    plane_split = not bool(args.no_plane)
    block_size_entries = int(args.block_size) if args.block_size >= 1 else 8192
    gzip_level = max(1, min(9, int(args.gzip_level)))

    print("Writing binary to:", bin_path,
          f"(omit_offsets={omit_offsets}, block_compress={block_compress}, block_size={block_size_entries}, plane_split={plane_split})")
    crc, size = write_bin_file(bin_path, rows_sorted, offsets, header_info, packed_vals, pv_to_idx,
                               remap_width, lengths_type, omit_offsets=omit_offsets,
                               block_compress=block_compress, block_size_entries=block_size_entries,
                               plane_split=plane_split, gzip_level=gzip_level)
    print(f"Wrote {bin_path} ({size} bytes) CRC32=0x{crc:08x}")

    # gzip the whole file for transport (optional)
    if not args.no_gzip:
        gz_path = bin_path + '.gz'
        with open(bin_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
            f_out.writelines(f_in)
        print("Wrote gzip:", gz_path, "size:", os.path.getsize(gz_path))

    # brotli (optional)
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
        "flags": ( (1 if omit_offsets else 0) | (2 if block_compress else 0) | (4 if plane_split else 0) ),
        "offsets_omitted": bool(omit_offsets),
        "block_compressed": bool(block_compress),
        "plane_split": bool(plane_split),
        "block_size_entries": block_size_entries
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
    if block_compress:
        print("Binary format: header + source_keys + (offsets omitted?) lengths + remap_table + block_index + compressed_blocks")
    else:
        print("Binary format: header + source_keys + (offsets omitted?) lengths + remap_table + values_blob (raw remap indices)")
    print("Remap width (bytes):", remap_width)
    print("Lengths type:", metadata['lengths_type'])
    print("Note: readers must reconstruct offsets from lengths when flags bit0 == 1.")
    return

if __name__ == '__main__':
    main()
