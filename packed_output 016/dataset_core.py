#!/usr/bin/env python3
"""
dataset_core.py — copy-safe, mmap-backed reader for the packed TMDB similar dataset.

GOAL / GUARANTEE
- This module is written to *avoid creating large Python copies* of mapped file data.
- It uses memoryview.cast, struct.unpack_from on buffer objects, and integer indexing.
- **No operation in the normal path will allocate bytes or copy the values_blob/remap_table/source_keys into Python bytes.**
- The only potentially-copying code path is when the file omits the offsets array; in that case the reader reconstructs a compact `array('I')` of offsets (minimal unavoidable copy). If you always write offsets (recommended), that path is never used.

USAGE
- load_or_fetch(url, cache_path, use_mmap=True) -> (Dataset, metadata)
- parse_dataset_file(mm) -> parsed dict (views are memoryviews / arrays)
- Dataset.query_similar(tmdb_id, kind) -> (results_list, timings)
- Dataset.close() to release mmap and file
- now_ms(), format_ms(), describe_memory(ds, include_process=False)

IMPLEMENTATION NOTES
- Do NOT call any method that slices the mmap (e.g., mm[:n]) — those create bytes copies. Use memoryview(mm) and slice the memoryview.
- memoryview.cast(...) is used to expose typed arrays without copy.
- For 3-byte remap indices we assemble ints from the memoryview bytes — no temporary bytes objects are created.
"""
from __future__ import annotations
import os
import sys
import time
import struct
import mmap
import urllib.request
import shutil
import tempfile
from array import array
from typing import Tuple, List, Dict, Any

# ---- Small developer lint comment ----
# LINTER: Avoid bytes() / mm[a:b] / mm.read() on the mmap. Use memoryview(mm) and struct.unpack_from.
# If you add code here later, ensure it does not create large bytes objects from slices.

# ---- Helpers matching the packer ----
def packed_value(tmdb_id: int, typ: str) -> int:
    return (int(tmdb_id) << 1) | (1 if str(typ).lower().startswith('tv') else 0)

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def format_ms(ms: float) -> str:
    return f"{ms:.3f} ms"

# ---- Header struct (must match packer) ----
HEADER_STRUCT = '<4s B B H I I I B B H I'
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 28

# ---- Dataset object ----
class Dataset:
    """
    Zero-copy dataset backed by an mmap.
    Public API:
      - query_similar(tmdb_id, kind) -> (results_list, timings_dict)
      - close()
      - attributes: _mmap, _fileobj, parsed, sizes
    """
    def __init__(self, fileobj, mm: mmap.mmap, parsed: Dict[str, Any]):
        self._fileobj = fileobj
        self._mmap = mm
        self.parsed = parsed
        self.sizes = parsed.get('sizes', {})
        views = parsed['views']

        # typed zero-copy views
        self._source_keys = views['source_keys']      # memoryview cast('I')
        self._offsets = views['offsets']              # memoryview cast('I') or array('I') if reconstructed
        self._lengths = views['lengths']              # memoryview (uint8) or cast('H')
        self._remap = views['remap_table']            # memoryview cast('I')
        self._values_blob = views['values_blob']      # raw memoryview('B') of remap indices bytes

        self.remap_index_width = parsed['remap_index_width']
        self.lengths_type = parsed['lengths_type']    # 0=uint8,1=uint16

        # cached typed views for remap indices (created lazily, no copies)
        self._values_u16 = None
        self._values_u32 = None

    def close(self):
        try:
            if getattr(self, '_mmap', None) is not None:
                try:
                    self._mmap.close()
                except Exception:
                    pass
                self._mmap = None
        finally:
            if getattr(self, '_fileobj', None) is not None:
                try:
                    self._fileobj.close()
                except Exception:
                    pass
                self._fileobj = None

    # ---- Binary search over source_keys (zero-copy) ----
    def _binary_search_source(self, packed_src_key: int) -> int:
        lo = 0
        hi = len(self._source_keys) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            v = self._source_keys[mid]  # memoryview cast('I') indexing -> int
            if v == packed_src_key:
                return mid
            if v < packed_src_key:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    # ---- Read a remap index at logical position (zero-copy, no temporary bytes) ----
    def _read_remap_index_at(self, idx_pos: int) -> int:
        w = self.remap_index_width
        if w == 2:
            # lazily expose as uint16 view (zero-copy)
            if self._values_u16 is None:
                # memoryview.cast returns a view, not a copy
                self._values_u16 = self._values_blob.cast('H')
            return int(self._values_u16[idx_pos])
        elif w == 4:
            if self._values_u32 is None:
                self._values_u32 = self._values_blob.cast('I')
            return int(self._values_u32[idx_pos])
        elif w == 3:
            # assemble u24 little-endian directly from the byte memoryview
            mv = self._values_blob  # memoryview('B')
            off = idx_pos * 3
            # Accessing mv[...] returns small ints (no copy)
            b0 = mv[off]
            b1 = mv[off + 1]
            b2 = mv[off + 2]
            return b0 | (b1 << 8) | (b2 << 16)
        else:
            raise ValueError("Unsupported remap_index_width")

    def query_similar(self, tmdb_id: int, kind: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        t0 = now_ms()
        packed_src = packed_value(int(tmdb_id), kind)

        t_search_start = now_ms()
        idx = self._binary_search_source(packed_src)
        t_search_end = now_ms()

        if idx == -1:
            timings = {
                'search_ms': t_search_end - t_search_start,
                'query_ms': 0.0,
                'total_ms': (now_ms() - t0)
            }
            return [], timings

        # offsets are logical indices into remap indices (not byte offsets)
        off = int(self._offsets[idx])
        if self.lengths_type == 0:
            length = int(self._lengths[idx])  # memoryview('B') indexing -> int
        else:
            length = int(self._lengths[idx])  # memoryview('H') indexing -> int

        t_query_start = now_ms()
        results = []
        # iterate remap indices and map to packed values using remap_table (zero-copy)
        for i in range(length):
            ridx = self._read_remap_index_at(off + i)
            pv = int(self._remap[ridx])  # remap_table is memoryview cast('I'); indexing -> int
            tmdb = pv >> 1
            typ = 'tvshow' if (pv & 1) else 'movie'
            results.append({'tmdb_id': tmdb, 'type': typ, 'packed': pv})
        t_query_end = now_ms()
        timings = {
            'search_ms': t_search_end - t_search_start,
            'query_ms': t_query_end - t_query_start,
            'total_ms': (now_ms() - t0)
        }
        return results, timings

# ---- Parse function that creates zero-copy views (no copies) ----
def parse_dataset_file(mm: mmap.mmap) -> Dict[str, Any]:
    """
    Parse header and return a dict with zero-copy memoryviews into the mmap.
    IMPORTANT: This function avoids producing bytes copies from the mmap.
    """
    if not mm or mm.size() < HEADER_SIZE:
        raise ValueError("Mapped file too small or invalid")

    # Use a memoryview over mmap and struct.unpack_from on that view (no bytes copy).
    full_mv = memoryview(mm)
    # struct.unpack_from accepts a buffer supporting buffer protocol (memoryview is fine)
    (magic, version, endian, flags, R, E, U,
     lengths_byte, remap_index_width, reserved, header_crc) = struct.unpack_from(HEADER_STRUCT, full_mv, 0)

    if magic != b"SIML":
        raise ValueError("Bad magic (not SIML)")

    pos = HEADER_SIZE
    sizes = {}
    sizes['header'] = HEADER_SIZE

    # source_keys: R * uint32
    source_keys_bytes = R * 4
    source_keys_off = pos
    pos += source_keys_bytes
    sizes['source_keys'] = source_keys_bytes

    # offsets: optional (we expect them present normally)
    offsets_present = not bool(flags & 1)
    offsets_bytes = R * 4 if offsets_present else 0
    offsets_off = pos if offsets_present else None
    pos += offsets_bytes
    sizes['offsets'] = offsets_bytes

    # lengths
    if lengths_byte == 0:
        lengths_type = 0
        lengths_bytes = R * 1
    else:
        lengths_type = 1
        lengths_bytes = R * 2
    lengths_off = pos
    pos += lengths_bytes
    sizes['lengths'] = lengths_bytes

    # remap_table
    remap_bytes = U * 4
    remap_off = pos
    pos += remap_bytes
    sizes['remap_table'] = remap_bytes

    # values_blob: remaining bytes
    values_off = pos
    values_bytes = mm.size() - pos
    sizes['values_blob'] = values_bytes

    # sanity check (best-effort)
    expected_values_bytes = E * remap_index_width
    # If mismatch, we still proceed but keep sizes; caller should be cautious.

    # Create zero-copy memoryviews (use full_mv slices, not mm slices)
    source_keys_mv = full_mv[source_keys_off:source_keys_off + source_keys_bytes].cast('I')

    if offsets_present:
        offsets_mv = full_mv[offsets_off:offsets_off + offsets_bytes].cast('I')
    else:
        # If offsets omitted we must reconstruct them into a compact array('I').
        # NOTE: this is the only unavoidable copy: constructing array('I') uses ~4*R bytes.
        # You can avoid this by writing offsets into the file (recommended).
        lengths_raw_mv = full_mv[lengths_off:lengths_off + lengths_bytes]
        arr = array('I')
        cur = 0
        if lengths_type == 0:
            # lengths_raw_mv is a memoryview('B')
            for i in range(R):
                l = int(lengths_raw_mv[i])
                arr.append(cur)
                cur += l
        else:
            lengths_u16 = lengths_raw_mv.cast('H')
            for i in range(R):
                l = int(lengths_u16[i])
                arr.append(cur)
                cur += l
        offsets_mv = arr  # array supports indexing; this is a compact in-memory copy

    lengths_mv = full_mv[lengths_off:lengths_off + lengths_bytes]
    if lengths_type == 1:
        lengths_mv = lengths_mv.cast('H')

    remap_mv = full_mv[remap_off:remap_off + remap_bytes].cast('I')
    values_mv = full_mv[values_off:values_off + values_bytes]  # memoryview('B')

    parsed = {
        'magic': magic,
        'version': version,
        'endian': endian,
        'flags': flags,
        'R': R, 'E': E, 'U': U,
        'lengths_type': lengths_type,
        'remap_index_width': remap_index_width,
        'offsets_present': offsets_present,
        'sizes': sizes,
        'views': {
            'source_keys': source_keys_mv,
            'offsets': offsets_mv,
            'lengths': lengths_mv,
            'remap_table': remap_mv,
            'values_blob': values_mv
        }
    }
    return parsed

# ---- Atomic download + mmap loader (no copies) ----
def _atomic_write_temp(target_path: str, data_stream):
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix='.tmp_ds_', suffix='.bin')
    os.close(fd)
    try:
        with open(tmp_path, 'wb') as out_f:
            # streaming copy from response to file (no intermediate full-file bytes in memory)
            shutil.copyfileobj(data_stream, out_f)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def load_or_fetch(url: str, cache_path: str, use_mmap: bool = True) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Download url into cache_path if missing and memory-map it.
    Returns (Dataset, metadata). No large memory copies are made.
    """
    metadata = {'from_cache': False, 'size_bytes': None, 'path': cache_path}
    if not os.path.exists(cache_path):
        req = urllib.request.Request(url, headers={'User-Agent': 'dataset_core/1.0'})
        with urllib.request.urlopen(req) as resp:
            _atomic_write_temp(cache_path, resp)
        metadata['from_cache'] = False
    else:
        metadata['from_cache'] = True

    size = os.path.getsize(cache_path)
    metadata['size_bytes'] = size

    f = open(cache_path, 'rb')
    try:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise

    parsed = parse_dataset_file(mm)
    ds = Dataset(f, mm, parsed)
    ds.sizes = parsed.get('sizes', {})
    return ds, metadata

# ---- Memory/diagnostic helper (does not copy) ----
def describe_memory(ds: Dataset, include_process: bool = False) -> Dict[str, Any]:
    """
    Lightweight diagnostics. If include_process and psutil present, includes process RSS/VMS.
    This function avoids creating copies of mapped data.
    """
    p = ds.parsed
    sizes = p.get('sizes', {}).copy()
    sizes_summary = {
        'file_total_bytes': ds._mmap.size() if getattr(ds, '_mmap', None) is not None else None,
        'header_bytes': sizes.get('header'),
        'source_keys_bytes': sizes.get('source_keys'),
        'offsets_bytes': sizes.get('offsets'),
        'lengths_bytes': sizes.get('lengths'),
        'remap_table_bytes': sizes.get('remap_table'),
        'values_blob_bytes': sizes.get('values_blob'),
        'remap_index_width': ds.remap_index_width,
        'lengths_type': ('uint8' if ds.lengths_type == 0 else 'uint16'),
        'rows_R': p.get('R'),
        'unique_U': p.get('U'),
        'entries_E': p.get('E'),
        'offsets_present': p.get('offsets_present'),
    }
    out = {'parsed_sizes': sizes_summary}

    if not include_process:
        return out

    try:
        import psutil
        proc = psutil.Process()
        mi = proc.memory_info()
        out['process_memory'] = {'rss': getattr(mi, 'rss', None), 'vms': getattr(mi, 'vms', None)}
    except Exception as e:
        out['process_memory'] = {'error': f'psutil not available or failed: {e}'}

    # On Linux you could parse /proc/self/smaps; on Windows use external tools (RAMMap) for per-file cache details.
    return out

# ---- Quick CLI sanity check (no copies) ----
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python dataset_core.py <path-to-dataset.bin>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        parsed = parse_dataset_file(mm)
        print("Parsed header:", {k: parsed[k] for k in ('R','E','U','remap_index_width','lengths_type','offsets_present')})
        print("Sizes:", parsed.get('sizes'))
        mm.close()