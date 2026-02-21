#!/usr/bin/env python3
"""
tmdb_packer.py  —  S905/Kodi Shard Edition

Pack NDJSON of TMDB similar-lists into a compact binary format split into
N shards so that Kodi on an Amlogic S905 (very limited free RAM) only needs
to load the single shard that contains the queried source-key.

Shard layout (one file per shard):
  dataset_s{i:04d}_of_{N:04d}.bin

Each shard file is a self-contained SIML binary with its own header,
source_keys, offsets, lengths, remap_table and values_blob.  The remap
table inside every shard is LOCAL to that shard (re-indexed from 0) so the
file is independently decodable without the other shards.

A small JSON manifest  (dataset_manifest.json)  is written alongside the
shards.  The manifest maps every packed_source_key range to the shard file
that holds it, so the reader can open the right file in O(log N).

Shard selection strategy:
  --shards N        produce exactly N shards (rows split as evenly as possible)
  --shard-size K    produce ceil(R / K) shards of at most K rows each

Within each shard rows keep their sort order (ascending packed_source_key),
so binary-search inside a shard still works.

Binary format per shard  (identical to the single-file format):
  [28-byte header]
  [R_s * uint32  source_keys]
  [R_s * uint32  offsets]        (unless --omit-offsets)
  [R_s * uint8/uint16  lengths]
  [U_s * uint32  remap_table]   (local, re-indexed)
  [E_s * rw bytes  values_blob]

Header fields (same magic "SIML", version 1):
  R  = rows in THIS shard
  E  = total similar-entries in THIS shard
  U  = unique packed values in THIS shard
  flags bit0 = 1 if offsets omitted (reader reconstructs from lengths)

Usage examples:
  python tmdb_packer.py data.ndjson --shards 8
  python tmdb_packer.py data.ndjson --shard-size 5000 --omit-offsets
  python tmdb_packer.py data.ndjson --shards 1          # single-file, same as original
"""

import os
import sys
import json
import struct
import gzip
import zlib
import csv
import re
import math
from collections import Counter
from pathlib import Path

# Optional brotli
have_brotli = False
try:
    import brotli
    have_brotli = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# File picker (GUI fallback)
# ---------------------------------------------------------------------------
def choose_file_via_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select NDJSON file",
            filetypes=[("NDJSON", "*.ndjson *.jsonl *.json"), ("All files", "*.*")])
        root.update(); root.destroy()
        return path or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core packing helpers  (unchanged from original)
# ---------------------------------------------------------------------------
def packed_value(tmdb_id, typ):
    t = str(typ).strip().lower()
    return (int(tmdb_id) << 1) | (1 if t.startswith('tv') else 0)


def parse_tmdb_entry(entry):
    if entry is None:
        return None, None
    if isinstance(entry, int):
        return entry, 'movie'
    if isinstance(entry, dict):
        tmdb_raw = (entry.get('tmdb_id') if 'tmdb_id' in entry
                    else entry.get('id') if 'id' in entry else None)
        if tmdb_raw is None:
            for k in entry.keys():
                if k.strip().lower() in ('tmdb_id', 'tmdbid', 'id'):
                    tmdb_raw = entry[k]; break
        if tmdb_raw is None:
            return None, None
        try:
            sid = int(str(tmdb_raw).strip())
        except Exception:
            return None, None
        typ_raw = entry.get('type', 'movie')
        typ = typ_raw.strip() if isinstance(typ_raw, str) else str(typ_raw)
        return sid, typ or 'movie'
    if isinstance(entry, str):
        s = entry.strip()
        if not s:
            return None, None
        if s.isdigit():
            return int(s), 'movie'
        m = re.match(r'^\s*(\d+)\s*(?:[,|:\-]?\s*([A-Za-z0-9_ ]+))?\s*$', s)
        if m:
            return int(m.group(1)), (m.group(2) or 'movie').strip()
        m2 = re.search(r'(\d+)', s)
        if m2:
            sid = int(m2.group(1))
            remainder = s[m2.end():].strip()
            tok = remainder.split()[0] if remainder else 'movie'
            return sid, tok.strip()
        return None, None
    return None, None


def parse_ndjson(path):
    rows, freq, freq_by_type = [], Counter(), Counter()
    total_similar = max_len = line_no = 0
    with open(path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            line_no += 1
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] line {line_no}: JSON error: {e}"); continue

            src_raw = (obj.get('tmdb_id') if 'tmdb_id' in obj
                       else obj.get('id') if 'id' in obj else None)
            if src_raw is None:
                for k in obj.keys():
                    if k.strip().lower() in ('tmdb_id', 'tmdbid', 'id'):
                        src_raw = obj[k]; break
            src_id, src_type = (
                parse_tmdb_entry({'tmdb_id': src_raw,
                                  'type': obj.get('type', 'movie')})
                if src_raw is not None else (None, None))
            if src_id is None:
                print(f"[warn] line {line_no}: missing/invalid tmdb_id"); continue

            packed_list = []
            for s in (obj.get('similar', []) or []):
                sid, stype = parse_tmdb_entry(s)
                if sid is None:
                    continue
                try:
                    pv = packed_value(sid, stype)
                except Exception:
                    continue
                packed_list.append(pv)
                freq[pv] += 1
                stype_norm = (stype or 'movie').strip().lower()
                freq_by_type[stype_norm] += 1

            psrc = packed_value(int(src_id), src_type)
            rows.append({
                'source_tmdb_id': int(src_id),
                'source_type': (src_type or 'movie').strip(),
                'packed_source_key': psrc,
                'packed_list': packed_list,
            })
            total_similar += len(packed_list)
            if len(packed_list) > max_len:
                max_len = len(packed_list)
    return rows, total_similar, freq, freq_by_type, max_len


def build_remap(freq_counter):
    items = sorted(freq_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    packed_vals = [pv for pv, _ in items]
    freqs       = [cnt for _, cnt in items]
    pv_to_idx   = {pv: i for i, pv in enumerate(packed_vals)}
    return packed_vals, freqs, pv_to_idx, items


def choose_remap_width(U):
    if U <= 0xFFFF:   return 2
    if U <= 0xFFFFFF: return 3
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


# ---------------------------------------------------------------------------
# Single-shard writer
# ---------------------------------------------------------------------------
def write_shard_file(out_path, shard_rows, omit_offsets=False):
    """
    Write one shard.  The remap table is LOCAL to this shard (re-indexed
    from 0 using only the packed values that appear in shard_rows).

    Returns (crc32, file_size, shard_meta_dict).
    """
    # Build a local frequency counter and remap for this shard only
    local_freq = Counter()
    for r in shard_rows:
        for pv in r['packed_list']:
            local_freq[pv] += 1

    packed_vals, freqs, pv_to_idx, _ = build_remap(local_freq)

    R_s  = len(shard_rows)
    E_s  = sum(len(r['packed_list']) for r in shard_rows)
    U_s  = len(packed_vals)

    # Compute offsets (in entry-index space) for this shard
    offsets = []
    cur = 0
    for r in shard_rows:
        offsets.append(cur)
        cur += len(r['packed_list'])

    max_len_s   = max((len(r['packed_list']) for r in shard_rows), default=0)
    remap_width = choose_remap_width(U_s)
    lengths_type = 0 if max_len_s < 256 else 1  # 0=uint8, 1=uint16

    HEADER_SIZE = 28
    magic     = b"SIML"
    version   = 1
    endian    = 1
    flags     = 1 if omit_offsets else 0
    lengths_b = lengths_type
    reserved  = 0
    crc_ph    = 0

    header = struct.pack('<4s B B H I I I B B H I',
                         magic, version, endian, flags,
                         R_s, E_s, U_s,
                         lengths_b, remap_width, reserved, crc_ph)
    assert len(header) == HEADER_SIZE

    with open(out_path, 'wb') as fh:
        fh.write(header)

        # source_keys
        for r in shard_rows:
            fh.write(struct.pack('<I', r['packed_source_key']))

        # offsets (optional)
        if not omit_offsets:
            for off in offsets:
                fh.write(struct.pack('<I', off))

        # lengths
        if lengths_type == 0:
            for r in shard_rows:
                fh.write(struct.pack('<B', len(r['packed_list'])))
        else:
            for r in shard_rows:
                fh.write(struct.pack('<H', len(r['packed_list'])))

        # remap_table (local)
        for pv in packed_vals:
            fh.write(struct.pack('<I', pv))

        # values_blob
        for r in shard_rows:
            for pv in r['packed_list']:
                write_uint_bytes(fh, pv_to_idx[pv], remap_width)

    # patch CRC
    with open(out_path, 'rb') as fh:
        data = fh.read()
    crc = zlib.crc32(data) & 0xFFFFFFFF
    with open(out_path, 'r+b') as fh:
        fh.seek(24)
        fh.write(struct.pack('<I', crc))

    size = os.path.getsize(out_path)

    # key range for manifest
    first_key = shard_rows[0]['packed_source_key']  if R_s else 0
    last_key  = shard_rows[-1]['packed_source_key'] if R_s else 0

    shard_meta = {
        'rows':         R_s,
        'entries':      E_s,
        'unique':       U_s,
        'remap_width':  remap_width,
        'lengths_type': 'uint8' if lengths_type == 0 else 'uint16',
        'first_packed_source_key': first_key,
        'last_packed_source_key':  last_key,
        'file_size_bytes':         size,
        'crc32':                   f'0x{crc:08x}',
    }
    return crc, size, shard_meta


# ---------------------------------------------------------------------------
# Shard-split entry point
# ---------------------------------------------------------------------------
def split_into_shards(rows_sorted, num_shards):
    """
    Split rows_sorted into num_shards groups of as-equal-size-as-possible rows.
    Returns list-of-lists.
    """
    R = len(rows_sorted)
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    num_shards = min(num_shards, R) if R else 1
    base, extra = divmod(R, num_shards)
    shards, start = [], 0
    for i in range(num_shards):
        end = start + base + (1 if i < extra else 0)
        shards.append(rows_sorted[start:end])
        start = end
    return shards


def compress_shard(bin_path, no_gzip, no_br):
    if not no_gzip:
        gz_path = bin_path + '.gz'
        with open(bin_path, 'rb') as fi, gzip.open(gz_path, 'wb') as fo:
            fo.writelines(fi)
        print(f"  gzip  -> {gz_path}  ({os.path.getsize(gz_path)} bytes)")
    if have_brotli and not no_br:
        br_path = bin_path + '.br'
        with open(bin_path, 'rb') as fi:
            data = fi.read()
        with open(br_path, 'wb') as fo:
            fo.write(brotli.compress(data))
        print(f"  brotli-> {br_path}  ({os.path.getsize(br_path)} bytes)")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def write_csv_remap(out_csv, packed_vals, freqs):
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['remap_index', 'tmdb_id', 'type', 'packed_value', 'frequency'])
        for i, pv in enumerate(packed_vals):
            w.writerow([i, pv >> 1, 'tvshow' if (pv & 1) else 'movie', pv, freqs[i]])


def write_csv_source_keys(out_csv, rows_sorted, shard_idx_per_row):
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['source_tmdb_id', 'source_type', 'packed_source_key',
                    'shard_index', 'length'])
        for r, si in zip(rows_sorted, shard_idx_per_row):
            w.writerow([r['source_tmdb_id'], r['source_type'],
                        r['packed_source_key'], si, len(r['packed_list'])])


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Pack NDJSON TMDB similar-lists into sharded binary (S905/Kodi edition).")
    parser.add_argument('ndjson', nargs='?',
        help='Path to NDJSON file. Omit to open a file picker.')
    parser.add_argument('--outdir',      default='packed_output',
        help='Output folder  (default: packed_output)')
    parser.add_argument('--shards',      type=int, default=0,
        help='Number of shard files to produce. 0 = auto based on --shard-size.')
    parser.add_argument('--shard-size',  type=int, default=5000,
        help='Max rows per shard when --shards is 0  (default: 5000).')
    parser.add_argument('--no-gzip',     action='store_true',
        help='Skip gzip output.')
    parser.add_argument('--no-br',       action='store_true',
        help='Skip brotli output.')
    parser.add_argument('--omit-offsets', action='store_true',
        help='Omit offsets array from each shard (reader must reconstruct).')
    args = parser.parse_args()

    # --- resolve input file ---
    ndjson_path = args.ndjson
    if not ndjson_path:
        ndjson_path = choose_file_via_dialog()
    if not ndjson_path:
        print("No NDJSON file specified. Exiting."); return
    ndjson_path = os.path.abspath(ndjson_path)
    if not os.path.isfile(ndjson_path):
        print("File not found:", ndjson_path); return

    outdir = os.path.abspath(args.outdir)
    ensure_dir(outdir)

    # --- parse ---
    print("Parsing NDJSON:", ndjson_path)
    rows, E, freq, freq_by_type, max_list_len = parse_ndjson(ndjson_path)
    R = len(rows)
    U = len(freq)
    print(f"Rows R={R}, total E={E}, unique U={U}, max_list_len={max_list_len}")

    # --- sort globally by packed_source_key ---
    rows_sorted = sorted(rows, key=lambda r: r['packed_source_key'])

    # --- determine shard count ---
    if args.shards > 0:
        num_shards = args.shards
    else:
        num_shards = max(1, math.ceil(R / args.shard_size)) if R else 1
    print(f"Splitting into {num_shards} shard(s) "
          f"(~{math.ceil(R/num_shards) if num_shards else R} rows/shard)")

    shards = split_into_shards(rows_sorted, num_shards)
    actual_N = len(shards)

    # --- write shards ---
    manifest_shards = []
    shard_idx_per_row = []
    total_bin_bytes = 0

    for i, shard_rows in enumerate(shards):
        fname    = f"dataset_s{i:04d}_of_{actual_N:04d}.bin"
        bin_path = os.path.join(outdir, fname)
        print(f"  Shard {i+1}/{actual_N}: {len(shard_rows)} rows -> {fname}")
        crc, size, smeta = write_shard_file(bin_path, shard_rows,
                                             omit_offsets=args.omit_offsets)
        print(f"    {size} bytes  CRC32=0x{crc:08x}  "
              f"U={smeta['unique']}  rw={smeta['remap_width']}")
        total_bin_bytes += size
        compress_shard(bin_path, args.no_gzip, args.no_br)

        smeta['filename'] = fname
        smeta['shard_index'] = i
        manifest_shards.append(smeta)
        for _ in shard_rows:
            shard_idx_per_row.append(i)

    # --- manifest ---
    # The manifest lets the reader do: binary-search shards by
    # first_packed_source_key / last_packed_source_key to find the file.
    manifest = {
        'version': 1,
        'total_rows':    R,
        'total_entries': E,
        'num_shards':    actual_N,
        'omit_offsets':  args.omit_offsets,
        'flags_bit0_meaning': '1=offsets omitted',
        'shards': manifest_shards,
    }
    manifest_path = os.path.join(outdir, 'dataset_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote manifest: {manifest_path}")

    # --- global remap CSV (informational, uses full-dataset freq) ---
    packed_vals_g, freqs_g, _, _ = build_remap(freq)
    remap_csv = os.path.join(outdir, 'remap_table_global.csv')
    write_csv_remap(remap_csv, packed_vals_g, freqs_g)
    print("Wrote remap_table_global.csv")

    source_keys_csv = os.path.join(outdir, 'source_keys.csv')
    write_csv_source_keys(source_keys_csv, rows_sorted, shard_idx_per_row)
    print("Wrote source_keys.csv")

    # --- metadata ---
    metadata = {
        'ndjson':            ndjson_path,
        'total_rows':        R,
        'total_entries':     E,
        'unique_global':     U,
        'max_list_len':      max_list_len,
        'num_shards':        actual_N,
        'omit_offsets':      args.omit_offsets,
        'total_bin_bytes':   total_bin_bytes,
        'freq_by_type':      dict(freq_by_type.most_common()),
        'top_100_global': [
            {'remap_index': i, 'packed': packed_vals_g[i],
             'tmdb_id':     packed_vals_g[i] >> 1,
             'type':        'tvshow' if (packed_vals_g[i] & 1) else 'movie',
             'frequency':   freqs_g[i]}
            for i in range(min(100, len(packed_vals_g)))
        ],
    }
    with open(os.path.join(outdir, 'metadata.json'), 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2)
    print("Wrote metadata.json")

    print()
    print("=" * 60)
    print(f"Done.  {actual_N} shard(s) in: {outdir}")
    print(f"Total binary size : {total_bin_bytes:,} bytes")
    print(f"offsets omitted   : {args.omit_offsets}")
    print()
    print("Reader hint (S905/Kodi):")
    print("  1. Load dataset_manifest.json  (~small, always in RAM)")
    print("  2. Binary-search manifest['shards'] by first/last_packed_source_key")
    print("  3. Open only the matching dataset_s*.bin shard")
    print("  4. Parse that shard with parse_dataset_bytes() as before")
    print("  => Only 1/N of the data is ever loaded into RAM")


if __name__ == '__main__':
    main()
