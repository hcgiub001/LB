#!/usr/bin/env python3
"""
dataset_core.py — copy-safe, mmap-backed reader with safe optional RAM-copy prefetch.

Enhancements (backwards-compatible):
  * graceful fallback if array copy fails (keep mmap-backed views)
  * explicit per-region lazy prefetch methods:
        - prefetch_remap_table() -> returns ms spent (float)
        - prefetch_source_keys() -> returns ms spent (float)
  * ram_copy flag at load time to control eager copy (keeps default behavior)
  * load_or_fetch(...) now returns metadata['load_ms'] (ms to open+parse+prefetch)
"""

from __future__ import annotations
import os
import time
import struct
import mmap
import urllib.request
import shutil
import tempfile
from array import array
from typing import Tuple, List, Dict, Any, Union

# ---- helpers ----
def packed_value(tmdb_id: int, typ: str) -> int:
    return (int(tmdb_id) << 1) | (1 if str(typ).lower().startswith('tv') else 0)

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def format_ms(ms: float) -> str:
    return f"{ms:.3f} ms"

# ---- header ----
HEADER_STRUCT = '<4s B B H I I I B B H I'
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 28

# ---- Dataset ----
class Dataset:
    """
    Zero-copy mmap reader with optional (safe) RAM-copy prefetch for hot regions.

    Constructor:
        Dataset(fileobj, mm, parsed, ram_copy=True, prefetch_remap=False, prefetch_source_keys=False)

    - ram_copy=True will attempt to copy source_keys + remap_table into array('I')
      at construction time (graceful fallback to mmap-backed views on error).
    - prefetch_remap / prefetch_source_keys allow explicit lazy prefetching (if you
      prefer to copy only when needed). Both prefetch methods return ms spent copying.
    """

    def __init__(
        self,
        fileobj,
        mm: mmap.mmap,
        parsed: Dict[str, Any],
        ram_copy: bool = True,
        prefetch_remap: bool = False,
        prefetch_source_keys: bool = False,
    ):
        self._fileobj = fileobj
        self._mmap = mm
        self.parsed = parsed
        self.sizes = parsed.get('sizes', {})

        views = parsed['views']
        self.remap_index_width = parsed['remap_index_width']
        self.lengths_type = parsed['lengths_type']

        # Default to mmap-backed views; may be replaced with array('I') copies
        self._source_keys: Union[array, memoryview] = views['source_keys']
        self._remap: Union[array, memoryview] = views['remap_table']
        self._offsets = views['offsets']    # memoryview cast('I') or array('I') if reconstructed
        self._lengths = views['lengths']    # memoryview('B') or cast('H')
        self._values_blob = views['values_blob']  # memoryview('B')

        # lazily created typed views for values_blob (zero-copy)
        self._values_u16 = None
        self._values_u32 = None

        # flag indicating whether we have a heap copy active (best-effort)
        self._ram_copy = False

        # If requested, attempt eager ram_copy; measure time isn't returned here,
        # but load_or_fetch will include constructor time in metadata['load_ms'].
        if ram_copy:
            try:
                self._source_keys = array('I', views['source_keys'])
                self._remap = array('I', views['remap_table'])
                self._ram_copy = True
            except Exception:
                # fallback to mmap-backed views (keep running)
                self._source_keys = views['source_keys']
                self._remap = views['remap_table']
                self._ram_copy = False

        # Optionally support explicit prefetch calls after construction (if requested)
        if prefetch_remap:
            try:
                self.prefetch_remap_table()
            except Exception:
                pass
        if prefetch_source_keys:
            try:
                self.prefetch_source_keys()
            except Exception:
                pass

    # ---- lazy prefetch helpers (return ms spent) ----
    def prefetch_remap_table(self) -> float:
        """
        Make an in-RAM copy of remap_table if it's not already an array('I').
        Returns milliseconds spent copying (0.0 if already copied or on failure).
        """
        if isinstance(self._remap, array):
            return 0.0
        t0 = time.perf_counter()
        try:
            arr = array('I', self._remap)
            self._remap = arr
            self._ram_copy = True
            return (time.perf_counter() - t0) * 1000.0
        except Exception:
            return 0.0

    def prefetch_source_keys(self) -> float:
        """
        Make an in-RAM copy of source_keys if it's not already an array('I').
        Returns milliseconds spent copying (0.0 if already copied or on failure).
        """
        if isinstance(self._source_keys, array):
            return 0.0
        t0 = time.perf_counter()
        try:
            arr = array('I', self._source_keys)
            self._source_keys = arr
            self._ram_copy = True
            return (time.perf_counter() - t0) * 1000.0
        except Exception:
            return 0.0

    def close(self) -> None:
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

    # ---- binary search over source_keys ----
    def _binary_search_source(self, packed_src_key: int) -> int:
        lo, hi = 0, len(self._source_keys) - 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            v = self._source_keys[mid]
            if v == packed_src_key:
                return mid
            if v < packed_src_key:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    # ---- read remap index ----
    def _read_remap_index_at(self, idx_pos: int) -> int:
        w = self.remap_index_width
        if w == 2:
            if self._values_u16 is None:
                self._values_u16 = self._values_blob.cast('H')
            return int(self._values_u16[idx_pos])
        elif w == 4:
            if self._values_u32 is None:
                self._values_u32 = self._values_blob.cast('I')
            return int(self._values_u32[idx_pos])
        elif w == 3:
            mv = self._values_blob
            off = idx_pos * 3
            return mv[off] | (mv[off + 1] << 8) | (mv[off + 2] << 16)
        else:
            raise ValueError("Unsupported remap_index_width")

    # ---- query ----
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

        off = int(self._offsets[idx])
        if self.lengths_type == 0:
            length = int(self._lengths[idx])
        else:
            length = int(self._lengths[idx])

        t_query_start = now_ms()
        results = []
        for i in range(length):
            ridx = self._read_remap_index_at(off + i)
            pv = int(self._remap[ridx])
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

# ---- parse (unchanged, zero-copy views) ----
def parse_dataset_file(mm: mmap.mmap) -> Dict[str, Any]:
    if not mm or mm.size() < HEADER_SIZE:
        raise ValueError("Mapped file too small or invalid")

    full_mv = memoryview(mm)
    (magic, version, endian, flags, R, E, U,
     lengths_byte, remap_index_width, reserved, header_crc) = struct.unpack_from(HEADER_STRUCT, full_mv, 0)

    if magic != b"SIML":
        raise ValueError("Bad magic (not SIML)")

    pos = HEADER_SIZE
    sizes = {'header': HEADER_SIZE}

    # source_keys
    source_keys_bytes = R * 4
    source_keys_off = pos; pos += source_keys_bytes
    sizes['source_keys'] = source_keys_bytes

    offsets_present = not bool(flags & 1)
    offsets_bytes = R * 4 if offsets_present else 0
    offsets_off = pos if offsets_present else None
    pos += offsets_bytes
    sizes['offsets'] = offsets_bytes

    if lengths_byte == 0:
        lengths_type, lengths_bytes = 0, R * 1
    else:
        lengths_type, lengths_bytes = 1, R * 2
    lengths_off = pos; pos += lengths_bytes
    sizes['lengths'] = lengths_bytes

    remap_bytes = U * 4
    remap_off = pos; pos += remap_bytes
    sizes['remap_table'] = remap_bytes

    values_off = pos
    values_bytes = mm.size() - pos
    sizes['values_blob'] = values_bytes

    # zero-copy memoryview slices
    source_keys_mv = full_mv[source_keys_off:source_keys_off + source_keys_bytes].cast('I')

    if offsets_present:
        offsets_mv = full_mv[offsets_off:offsets_off + offsets_bytes].cast('I')
    else:
        lengths_raw_mv = full_mv[lengths_off:lengths_off + lengths_bytes]
        arr = array('I')
        cur = 0
        if lengths_type == 0:
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
        offsets_mv = arr

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

# ---- atomic download + mmap loader ----
def _atomic_write_temp(target_path: str, data_stream) -> None:
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix='.tmp_ds', suffix='.bin')
    os.close(fd)
    try:
        with open(tmp_path, 'wb') as out_f:
            shutil.copyfileobj(data_stream, out_f)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def load_or_fetch(
    url: str,
    cache_path: str,
    use_mmap: bool = True,
    ram_copy: bool = True,
    prefetch_remap: bool = False,
    prefetch_source_keys: bool = False,
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Download url into cache_path if missing and memory-map it.
    Returns (Dataset, metadata). metadata includes:
      - from_cache (bool)
      - size_bytes (int)
      - path (str)
      - load_ms (float) : milliseconds spent in load_or_fetch (open+parse+construct+prefetch)
    """
    metadata = {'from_cache': False, 'size_bytes': None, 'path': cache_path, 'load_ms': None}
    t_start = time.perf_counter()

    if not os.path.exists(cache_path):
        req = urllib.request.Request(url, headers={'User-Agent': 'dataset_core/1.0'})
        with urllib.request.urlopen(req) as resp:
            _atomic_write_temp(cache_path, resp)
        metadata['from_cache'] = False
    else:
        metadata['from_cache'] = True

    metadata['size_bytes'] = os.path.getsize(cache_path)

    f = open(cache_path, 'rb')
    try:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise

    parsed = parse_dataset_file(mm)
    # Construct dataset; constructor may perform eager ram_copy (included in load_ms)
    ds = Dataset(f, mm, parsed, ram_copy=ram_copy,
                 prefetch_remap=prefetch_remap, prefetch_source_keys=prefetch_source_keys)
    ds.sizes = parsed.get('sizes', {})

    metadata['load_ms'] = (time.perf_counter() - t_start) * 1000.0
    return ds, metadata

# ---- diagnostics ----
def describe_memory(ds: Dataset, include_process: bool = False) -> Dict[str, Any]:
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
        'ram_copy_active': getattr(ds, '_ram_copy', False),
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
    return out

# ---- microbenchmark helper ----
def benchmark_cold_vs_warm(
    ds: Dataset,
    tmdb_id: int,
    kind: str = 'movie',
    n_warm: int = 100,
) -> Dict[str, Any]:
    """
    Run one "first" query (approx cold) then n_warm queries to characterise steady-state latency.
    For true cold measurement on Windows: use RAMMap -> Empty Standby List before loading.
    """
    import statistics
    cold_t0 = time.perf_counter()
    _, timings = ds.query_similar(tmdb_id, kind)
    cold_ms = (time.perf_counter() - cold_t0) * 1000

    warm_times = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        ds.query_similar(tmdb_id, kind)
        warm_times.append((time.perf_counter() - t0) * 1000)

    wt = sorted(warm_times)
    # safe p95 calculation
    if wt:
        idx95 = max(0, min(len(wt)-1, int(0.95 * len(wt))))
        p95 = wt[idx95]
    else:
        p95 = 0.0

    return {
        'first_ms': round(cold_ms, 4),
        'warm_mean_ms': round(statistics.mean(warm_times), 4) if warm_times else 0.0,
        'warm_p50_ms': round(statistics.median(warm_times), 4) if warm_times else 0.0,
        'warm_p95_ms': round(p95, 4),
        'warm_min_ms': round(min(warm_times), 4) if warm_times else 0.0,
        'ram_copy_active': getattr(ds, '_ram_copy', False),
    }

# ---- CLI ----
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset_core.py <path-to-dataset.bin> [tmdb_id] [movie|tv]")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        parsed = parse_dataset_file(mm)
        print("Parsed:", {k: parsed[k] for k in ('R', 'E', 'U', 'remap_index_width', 'lengths_type', 'offsets_present')})
        print("Sizes:", parsed.get('sizes'))

        if len(sys.argv) >= 3:
            tmdb_id = int(sys.argv[2])
            kind = sys.argv[3] if len(sys.argv) >= 4 else 'movie'
            ds, meta = load_or_fetch('file://'+os.path.abspath(path), path, use_mmap=True, ram_copy=True)
            bench = benchmark_cold_vs_warm(ds, tmdb_id, kind, n_warm=200)
            print("Benchmark (ram_copy=True):", bench)
        else:
            mm.close()