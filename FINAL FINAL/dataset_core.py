# dataset_core.py
# dataset_core.py this is mmap with on demand expand, should stay 4MB but slower, only takes a few KB in RAM

"""
Optimized Shared dataset reader + query core.
Zero-copy, on-demand decode by default, with lazy values_idx expansion.

Changes in this variant:
 - Expansion is *on-demand* by default (background_expand=False).
 - _ensure_values_idx will block and perform the one-time expansion on first
   query if values_idx is not present.
 - The 3-byte expansion path avoids creating a huge temporary `bytes()` object,
   lowering transient peak memory use during expansion.
"""
from __future__ import annotations
import os
import mmap
import time
import urllib.request
import array
import threading
import sys
from typing import Tuple, List, Optional, Dict, Any

HEADER_SIZE = 28

# Pre-computed type strings to avoid repeated string creation
_TYPE_MOVIE = 'movie'
_TYPE_TV = 'tv'
_TYPE_STRINGS = (_TYPE_MOVIE, _TYPE_TV)  # Index by type_int

# optional psutil for real RSS reporting in describe_memory
try:
    import psutil
except Exception:
    psutil = None


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def format_ms(x: float) -> str:
    return f"{x:.3f}"


# ---------------- Dataset ----------------
class Dataset:
    __slots__ = (
        'header', 'buffer', 'mv', 'R', 'E', 'U', 'remap_width',
        'source_keys', 'offsets', 'lengths', 'remap',
        'values_blob_offset', 'values_idx',
        'sizes', '_mmap', '_fileobj',
        # diagnostics / expansion flags
        '_expansion_ms', '_expansion_in_progress'
    )

    def __init__(self):
        self.header: Optional[Dict] = None
        self.buffer = None
        self.mv = None
        self.R: int = 0
        self.E: int = 0
        self.U: int = 0
        self.remap_width: int = 0
        self.source_keys: Optional[array.array] = None
        self.offsets: Optional[array.array] = None
        self.lengths: Optional[array.array] = None
        self.remap: Optional[array.array] = None
        self.values_blob_offset: int = 0
        # Lazy expansion target: array('I') or None
        self.values_idx: Optional[array.array] = None

        self.sizes: Dict[str, int] = {}
        self._mmap = None
        self._fileobj = None

        # diagnostics / expansion tracking
        self._expansion_ms: Optional[float] = None
        self._expansion_in_progress: bool = False

    def decode_entry(self, i: int) -> Tuple[int, int]:
        """
        Decode a single entry at index i.
        Uses values_idx if available for fast decode; otherwise reads from mv.
        Returns (tmdb_id, type_int).
        """
        # prefer values_idx if present
        vals = self.values_idx
        if vals is not None:
            ri = vals[i]
            packed = self.remap[ri]
            return (packed >> 1, packed & 1)

        mv = self.mv
        rw = self.remap_width
        pos = self.values_blob_offset + i * rw

        if rw == 2:
            ri = mv[pos] | (mv[pos + 1] << 8)
        elif rw == 3:
            ri = mv[pos] | (mv[pos + 1] << 8) | (mv[pos + 2] << 16)
        elif rw == 4:
            ri = mv[pos] | (mv[pos + 1] << 8) | (mv[pos + 2] << 16) | (mv[pos + 3] << 24)
        else:
            ri = 0
            for b in range(rw):
                ri |= mv[pos + b] << (b * 8)

        packed = self.remap[ri]
        return (packed >> 1, packed & 1)

    def query_similar(self, tmdb_id: int, kind: str) -> Tuple[List[Dict], Dict]:
        """
        Query similar items. Returns (list of dicts, timings dict).
        Each result dict: {'tmdb_id': int, 'type': 'movie'|'tv'}
        """
        t0 = now_ms()
        packed_key = (tmdb_id << 1) | (1 if kind == _TYPE_TV else 0)

        # Inline binary search
        t_search_start = now_ms()
        source_keys = self.source_keys
        lo = 0
        hi = self.R
        idx = -1
        while lo < hi:
            mid = (lo + hi) >> 1
            v = source_keys[mid]
            if v < packed_key:
                lo = mid + 1
            elif v > packed_key:
                hi = mid
            else:
                idx = mid
                break
        t_search_end = now_ms()

        if idx < 0:
            return [], {
                'total_ms': format_ms(now_ms() - t0),
                'search_ms': format_ms(t_search_end - t_search_start),
                'query_ms': format_ms(0.0)
            }

        # On-demand expansion: block here and create values_idx if missing.
        _ensure_values_idx(self)

        t_query_start = now_ms()
        off = int(self.offsets[idx])
        length = int(self.lengths[idx])

        # Attempt to use values_idx if available (fast path)
        vals = self.values_idx
        rm = self.remap
        type_strs = _TYPE_STRINGS

        results: List[Dict] = []
        append = results.append

        if vals is not None:
            # C-indexed fast loop
            for j in range(off, off + length):
                p = rm[vals[j]]
                append({'tmdb_id': p >> 1, 'type': type_strs[p & 1]})
        else:
            # Zero-copy on-demand decode (original slower path)
            mv = self.mv
            voff = self.values_blob_offset
            rw = self.remap_width
            if rw == 2:
                i = off
                while i < off + length:
                    pos = voff + i * 2
                    p = rm[mv[pos] | (mv[pos + 1] << 8)]
                    append({'tmdb_id': p >> 1, 'type': type_strs[p & 1]})
                    i += 1
            elif rw == 3:
                i = off
                while i < off + length:
                    pos = voff + i * 3
                    p = rm[mv[pos] | (mv[pos + 1] << 8) | (mv[pos + 2] << 16)]
                    append({'tmdb_id': p >> 1, 'type': type_strs[p & 1]})
                    i += 1
            elif rw == 4:
                i = off
                while i < off + length:
                    pos = voff + i * 4
                    p = rm[mv[pos] | (mv[pos + 1] << 8) | (mv[pos + 2] << 16) | (mv[pos + 3] << 24)]
                    append({'tmdb_id': p >> 1, 'type': type_strs[p & 1]})
                    i += 1
            else:
                i = off
                while i < off + length:
                    pos = voff + i * rw
                    val = 0
                    for b in range(rw):
                        val |= mv[pos + b] << (b * 8)
                    p = rm[val]
                    append({'tmdb_id': p >> 1, 'type': type_strs[p & 1]})
                    i += 1

        t_query_end = now_ms()

        return results, {
            'total_ms': format_ms(now_ms() - t0),
            'search_ms': format_ms(t_search_end - t_search_start),
            'query_ms': format_ms(t_query_end - t_query_start)
        }

    def query_similar_fast(self, tmdb_id: int, kind: str) -> Tuple[List[Tuple[int, int]], Dict]:
        """
        Fast query returning tuples instead of dicts.
        Returns (list of (tmdb_id, type_int), timings).
        """
        t0 = now_ms()
        packed_key = (tmdb_id << 1) | (1 if kind == _TYPE_TV else 0)

        # Inline binary search
        source_keys = self.source_keys
        lo = 0
        hi = self.R
        idx = -1
        while lo < hi:
            mid = (lo + hi) >> 1
            v = source_keys[mid]
            if v < packed_key:
                lo = mid + 1
            elif v > packed_key:
                hi = mid
            else:
                idx = mid
                break
        t_search_end = now_ms()

        if idx < 0:
            return [], {
                'total_ms': format_ms(now_ms() - t0),
                'search_ms': format_ms(t_search_end - t0),
                'query_ms': format_ms(0.0)
            }

        # Ensure values_idx (blocking on-demand)
        _ensure_values_idx(self)

        off = int(self.offsets[idx])
        length = int(self.lengths[idx])

        vals = self.values_idx
        rm = self.remap

        results: List[Tuple[int, int]] = []
        append = results.append

        if vals is not None:
            for j in range(off, off + length):
                p = rm[vals[j]]
                append((p >> 1, p & 1))
        else:
            mv = self.mv
            voff = self.values_blob_offset
            rw = self.remap_width
            if rw == 2:
                i = off
                while i < off + length:
                    pos = voff + i * 2
                    p = rm[mv[pos] | (mv[pos + 1] << 8)]
                    append((p >> 1, p & 1))
                    i += 1
            elif rw == 3:
                i = off
                while i < off + length:
                    pos = voff + i * 3
                    p = rm[mv[pos] | (mv[pos + 1] << 8) | (mv[pos + 2] << 16)]
                    append((p >> 1, p & 1))
                    i += 1
            elif rw == 4:
                i = off
                while i < off + length:
                    pos = voff + i * 4
                    p = rm[mv[pos] | (mv[pos + 1] << 8) | (mv[pos + 2] << 16) | (mv[pos + 3] << 24)]
                    append((p >> 1, p & 1))
                    i += 1
            else:
                i = off
                while i < off + length:
                    pos = voff + i * rw
                    val = 0
                    for b in range(rw):
                        val |= mv[pos + b] << (b * 8)
                    p = rm[val]
                    append((p >> 1, p & 1))
                    i += 1

        t_end = now_ms()

        return results, {
            'total_ms': format_ms(t_end - t0),
            'search_ms': format_ms(t_search_end - t0),
            'query_ms': format_ms(t_end - t_search_end)
        }

    def query_similar_raw(self, tmdb_id: int, kind: str) -> Tuple[int, int]:
        """
        Fastest query - returns (offset, length) for manual iteration.
        """
        packed = (tmdb_id << 1) | (1 if kind == _TYPE_TV else 0)
        source_keys = self.source_keys
        lo = 0
        hi = self.R
        while lo < hi:
            mid = (lo + hi) >> 1
            v = source_keys[mid]
            if v < packed:
                lo = mid + 1
            elif v > packed:
                hi = mid
            else:
                return (self.offsets[mid], self.lengths[mid])
        return (-1, 0)


# ---------------- helpers for parsing ----------------

def _read_header_from_mv(mv: memoryview) -> Dict:
    if len(mv) < HEADER_SIZE:
        raise ValueError("Buffer too small for header")
    magic = bytes(mv[0:4]).decode('ascii')
    if magic != 'SIML':
        raise ValueError(f"Bad magic: {magic}")
    return {
        'magic': magic,
        'version': mv[4],
        'endianness': mv[5],
        'flags': int.from_bytes(mv[6:8], 'little'),
        'R': int.from_bytes(mv[8:12], 'little'),
        'E': int.from_bytes(mv[12:16], 'little'),
        'U': int.from_bytes(mv[16:20], 'little'),
        'lengths_type': mv[20],
        'remap_index_width': mv[21],
        'header_crc': int.from_bytes(mv[24:28], 'little')
    }


def _make_uint32_array(mv_slice: memoryview) -> array.array:
    try:
        vcast = mv_slice.cast('I')
        if vcast.itemsize == 4:
            return array.array('I', vcast)
    except Exception:
        pass
    a = array.array('I')
    a.frombytes(bytes(mv_slice))
    return a


def _make_uint16_array(mv_slice: memoryview) -> array.array:
    try:
        vcast = mv_slice.cast('H')
        if vcast.itemsize == 2:
            return array.array('H', vcast)
    except Exception:
        pass
    a = array.array('H')
    a.frombytes(bytes(mv_slice))
    return a


def _make_uint8_array(mv_slice: memoryview) -> array.array:
    return array.array('B', mv_slice)


def _do_expand_values_idx(ds: Dataset) -> None:
    """
    Internal expansion routine that actually performs the one-time copy into
    ds.values_idx (array('I')). This is the heavy work and may fault pages.

    Note: This implementation avoids creating one giant temporary `bytes()`
    for the 3-byte case; it reads directly from the memoryview to build the
    target array.
    """
    if ds.values_idx is not None:
        return

    mv = ds.mv
    E = ds.E
    off = ds.values_blob_offset
    w = ds.remap_width

    # Fast path for 4-byte on-disk
    if w == 4:
        slice_mv = mv[off: off + E * 4]
        try:
            vcast = slice_mv.cast('I')
        except Exception:
            vcast = None
        if vcast is not None:
            ds.values_idx = array.array('I', vcast)
            return
        a = array.array('I')
        a.frombytes(bytes(slice_mv))
        ds.values_idx = a
        return

    # 2-byte -> promote to 32-bit
    elif w == 2:
        slice_mv = mv[off: off + E * 2]
        try:
            vcast = slice_mv.cast('H')
            ds.values_idx = array.array('I', vcast)
            return
        except Exception:
            tmp = array.array('H')
            tmp.frombytes(bytes(slice_mv))
            ds.values_idx = array.array('I', tmp)
            return

    # 3-byte packed indices -- build without huge bytes() temporary
    elif w == 3:
        # allocate target array with zeros
        a = array.array('I', [0]) * E
        # fill directly from memoryview without creating an intermediate bytes object
        p = off
        for i in range(E):
            # mv[...] yields integer 0..255
            b0 = mv[p]
            b1 = mv[p + 1]
            b2 = mv[p + 2]
            a[i] = b0 | (b1 << 8) | (b2 << 16)
            p += 3
        ds.values_idx = a
        return

    # other widths (1,5..n) handled generically
    else:
        a = array.array('I', [0]) * E
        for i in range(E):
            s = mv[off + i * w: off + (i + 1) * w]
            a[i] = int.from_bytes(bytes(s), 'little')
        ds.values_idx = a
        return


def _ensure_values_idx(ds: Dataset) -> None:
    """
    Ensure values_idx exists. This blocks and performs the expansion in the
    caller thread if values_idx isn't present. If another expansion is already
    in progress by another thread, wait for it to finish and use its result.
    """
    if ds.values_idx is not None:
        return

    # If another thread is expanding, wait until it finishes (so we don't race).
    # This is a simple spin-wait with sleeps to avoid busy-waiting.
    if getattr(ds, '_expansion_in_progress', False):
        waited = 0.0
        while getattr(ds, '_expansion_in_progress', False):
            time.sleep(0.001)
            waited += 0.001
            # small timeout to avoid waiting forever; if it continues, fall back to on-demand decode
            if waited > 5.0:
                return
        # if expansion finished and values_idx exists, return
        if ds.values_idx is not None:
            return

    # Attempt expansion in this thread (blocking)
    ds._expansion_in_progress = True
    t0 = now_ms()
    try:
        _do_expand_values_idx(ds)
        ds._expansion_ms = now_ms() - t0
        # update sizes if possible
        try:
            if ds.values_idx is not None:
                ds.sizes['decoded_arrays'] = len(ds.values_idx) * 4
        except Exception:
            pass
    except Exception:
        # on failure, ensure values_idx remains None and allow on-demand decode to continue
        ds.values_idx = None
    finally:
        ds._expansion_in_progress = False


def parse_dataset_bytes(
    buf,
    predecode: bool = True,       # Ignored — kept for backward compat
    preexpand_3byte: bool = True   # Ignored — kept for backward compat
) -> Dataset:
    """
    Parse dataset from bytes-like object.
    On-demand decode: values are read directly from the buffer at query time.
    """
    if isinstance(buf, memoryview):
        mv = buf
        buffer_ref = buf
    else:
        mv = memoryview(buf)
        buffer_ref = buf

    header = _read_header_from_mv(mv)
    R = header['R']
    E = header['E']
    U = header['U']
    lengths_type = header['lengths_type']
    remap_width = header['remap_index_width']
    offsets_omitted = (header['flags'] & 1) != 0

    offset = HEADER_SIZE

    # Source keys (R x uint32)
    source_keys = _make_uint32_array(mv[offset:offset + R * 4])
    offset += R * 4

    # Offsets (optional, R x uint32)
    offsets_on_disk = 0
    if not offsets_omitted:
        offsets_arr = _make_uint32_array(mv[offset:offset + R * 4])
        offset += R * 4
        offsets_on_disk = R * 4
    else:
        offsets_arr = None

    # Lengths
    if lengths_type == 0:
        lengths_arr = _make_uint8_array(mv[offset:offset + R])
        lengths_bytes = R
    else:
        lengths_arr = _make_uint16_array(mv[offset:offset + R * 2])
        lengths_bytes = R * 2
    offset += lengths_bytes

    # Reconstruct offsets if omitted
    if offsets_omitted:
        offsets_arr = array.array('I', [0] * R)
        cur = 0
        for i in range(R):
            offsets_arr[i] = cur
            cur += int(lengths_arr[i])

    # Remap table (U x uint32)
    remap_arr = _make_uint32_array(mv[offset:offset + U * 4])
    offset += U * 4

    # Values blob — record start offset; read on demand or expand lazily
    values_blob_offset = offset
    values_blob_bytes = E * remap_width

    # Build dataset
    ds = Dataset()
    ds.header = header
    ds.buffer = buffer_ref
    ds.mv = mv
    ds.R = R
    ds.E = E
    ds.U = U
    ds.remap_width = remap_width
    ds.source_keys = source_keys
    ds.offsets = offsets_arr
    ds.lengths = lengths_arr
    ds.remap = remap_arr
    ds.values_blob_offset = values_blob_offset
    ds.values_idx = None  # lazy

    # Size summary — no decoded arrays unless expanded
    ds.sizes = {
        'header': HEADER_SIZE,
        'source_keys': R * 4,
        'offsets_on_disk': offsets_on_disk,
        'offsets_in_memory': R * 4,
        'lengths': lengths_bytes,
        'remap_table': U * 4,
        'values_blob': values_blob_bytes,
        'decoded_arrays': 0,
        'estimated_total_ram': (
            R * 4 + R * 4 + lengths_bytes + U * 4
        )
    }

    return ds


def parse_dataset_file(
    path: str,
    use_mmap: bool = True,
    predecode: bool = True,       # Ignored
    preexpand_3byte: bool = True   # Ignored
) -> Dataset:
    """
    Load dataset from file path.
    """
    f = open(path, 'rb')
    try:
        if use_mmap:
            try:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                ds = parse_dataset_bytes(mm)
                ds._mmap = mm
                ds._fileobj = f
                return ds
            except Exception:
                f.seek(0)
                data = f.read()
                ds = parse_dataset_bytes(data)
                ds._fileobj = f
                return ds
        else:
            data = f.read()
            ds = parse_dataset_bytes(data)
            ds._fileobj = f
            return ds
    except Exception:
        f.close()
        raise


def _background_expand_and_record(ds: Dataset, meta: Dict[str, Any] = None) -> None:
    """
    Expand values_idx in background and optionally write timing info to meta.
    Left for compatibility if you explicitly request background expansion.
    """
    ds._expansion_in_progress = True
    t0 = now_ms()
    try:
        _do_expand_values_idx(ds)
        t1 = now_ms()
        ds._expansion_ms = (t1 - t0)
        try:
            ds.sizes['decoded_arrays'] = (len(ds.values_idx) * 4) if ds.values_idx is not None else 0
        except Exception:
            pass
        if meta is not None:
            try:
                meta['timings']['background_expand_ms'] = format_ms(ds._expansion_ms)
            except Exception:
                pass
    finally:
        ds._expansion_in_progress = False


def load_or_fetch(
    url: str,
    cache_path: str,
    use_mmap: bool = True,
    predecode: bool = True,       # Ignored
    preexpand_3byte: bool = True, # Ignored
    background_expand: bool = False  # default: do not background-expand; expand on-demand
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Load dataset from cache or fetch from URL.

    background_expand: if True, start a daemon thread that expands the
                      values_idx in the background (non-blocking). If False
                      (the default), expansion will only happen on-demand on
                      the first query that needs it.
    """
    meta: Dict[str, Any] = {
        'from_cache': False,
        'source': None,
        'size_bytes': 0,
        'timings': {}
    }

    t0 = now_ms()

    if os.path.exists(cache_path):
        meta['from_cache'] = True
        meta['source'] = 'file'
        meta['size_bytes'] = os.path.getsize(cache_path)
        t1 = now_ms()
        ds = parse_dataset_file(cache_path, use_mmap=use_mmap)
        meta['timings']['load_ms'] = format_ms(now_ms() - t1)

        # Optionally start background expansion (daemon thread) to avoid blocking GUI
        if background_expand:
            try:
                th = threading.Thread(target=_background_expand_and_record, args=(ds, meta), daemon=True)
                th.start()
            except Exception:
                pass

        return ds, meta

    # Download from URL
    meta['from_cache'] = False
    meta['source'] = url
    t_fetch_start = now_ms()
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            f.write(data)
        meta['size_bytes'] = len(data)
        t_fetch_end = now_ms()

        if use_mmap:
            ds = parse_dataset_file(cache_path, use_mmap=use_mmap)
        else:
            ds = parse_dataset_bytes(data)

        # background expansion
        if background_expand:
            try:
                th = threading.Thread(target=_background_expand_and_record, args=(ds, meta), daemon=True)
                th.start()
            except Exception:
                pass

        t_parse_end = now_ms()
        meta['timings']['fetch_ms'] = format_ms(t_fetch_end - t_fetch_start)
        meta['timings']['parse_ms'] = format_ms(t_parse_end - t_fetch_end)
        return ds, meta
    except Exception as e:
        raise RuntimeError(f"Failed to fetch dataset: {e}")


# ---------- Benchmark utility ----------
def benchmark_query(ds: Dataset, tmdb_id: int, kind: str, iterations: int = 1000) -> Dict:
    """
    Benchmark all query methods.
    """
    import gc
    gc.collect()

    results = {}

    # Warm-up
    for _ in range(10):
        ds.query_similar(tmdb_id, kind)
        ds.query_similar_fast(tmdb_id, kind)
        ds.query_similar_raw(tmdb_id, kind)

    # Benchmark query_similar (dict output)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iterations):
        ds.query_similar(tmdb_id, kind)
    t1 = time.perf_counter()
    results['query_similar_ms'] = (t1 - t0) / iterations * 1000

    # Benchmark query_similar_fast (tuple output)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iterations):
        ds.query_similar_fast(tmdb_id, kind)
    t1 = time.perf_counter()
    results['query_similar_fast_ms'] = (t1 - t0) / iterations * 1000

    # Benchmark query_similar_raw (offset/length only)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iterations):
        ds.query_similar_raw(tmdb_id, kind)
    t1 = time.perf_counter()
    results['query_similar_raw_ms'] = (t1 - t0) / iterations * 1000

    return results


# ---------------- diagnostic helper ----------------
def describe_memory(ds: Dataset, include_process: bool = True) -> Dict[str, Any]:
    """
    Return a dict describing dataset-owned bytes and (optionally) process RSS.
    """
    info: Dict[str, Any] = {}

    def _arr_info(name: str, arr: Optional[array.array]):
        if arr is None:
            return {
                'present': False,
                'typecode': None,
                'length': 0,
                'itemsize': 0,
                'elements_bytes': 0,
                'sys_getsizeof': 0
            }
        try:
            length = len(arr)
            itemsize = arr.itemsize
            elements_bytes = length * itemsize
            sys_size = sys.getsizeof(arr)
            return {
                'present': True,
                'typecode': arr.typecode,
                'length': int(length),
                'itemsize': int(itemsize),
                'elements_bytes': int(elements_bytes),
                'sys_getsizeof': int(sys_size)
            }
        except Exception:
            return {
                'present': True,
                'typecode': None,
                'length': 0,
                'itemsize': 0,
                'elements_bytes': 0,
                'sys_getsizeof': sys.getsizeof(arr)
            }

    # arrays we know about
    ak = _arr_info('source_keys', getattr(ds, 'source_keys', None))
    ao = _arr_info('offsets', getattr(ds, 'offsets', None))
    al = _arr_info('lengths', getattr(ds, 'lengths', None))
    ar = _arr_info('remap', getattr(ds, 'remap', None))
    av = _arr_info('values_idx', getattr(ds, 'values_idx', None))

    # raw values blob on disk (from sizes)
    sizes = ds.sizes if hasattr(ds, 'sizes') else {}
    values_blob_on_disk = int(sizes.get('values_blob', 0))

    # sum of element bytes (pure C buffers)
    sum_elements_bytes = ak['elements_bytes'] + ao['elements_bytes'] + al['elements_bytes'] + ar['elements_bytes'] + av['elements_bytes']

    # sum of sys.getsizeof (Python object size, includes buffer for arrays)
    sum_sys_sizes = ak['sys_getsizeof'] + ao['sys_getsizeof'] + al['sys_getsizeof'] + ar['sys_getsizeof'] + av['sys_getsizeof']

    info['arrays'] = {
        'source_keys': ak,
        'offsets': ao,
        'lengths': al,
        'remap': ar,
        'values_idx': av
    }
    info['values_blob_on_disk_bytes'] = values_blob_on_disk
    info['elements_bytes_sum'] = int(sum_elements_bytes)
    info['sys_getsizeof_sum'] = int(sum_sys_sizes)
    info['estimated_dataset_bytes_tracked'] = int(
        sizes.get('header', 0) +
        sizes.get('source_keys', 0) +
        sizes.get('offsets_on_disk', 0) +  # on-disk size if present
        sizes.get('offsets_in_memory', 0) +  # what we account as in-memory offsets
        sizes.get('lengths', 0) +
        sizes.get('remap_table', 0) +
        values_blob_on_disk
    )

    # If values_idx present, include its element-bytes explicitly as decoded arrays
    info['decoded_arrays_bytes'] = av['elements_bytes']
    info['total_dataset_elements_bytes'] = int(values_blob_on_disk + (sum_elements_bytes - av['elements_bytes']))
    info['total_dataset_elements_mb'] = round(info['total_dataset_elements_bytes'] / 1024.0 / 1024.0, 4)

    # Python-level object overhead vs raw elements
    info['python_objects_sys_getsizeof_bytes'] = int(sum_sys_sizes)
    info['python_objects_sys_getsizeof_mb'] = round(sum_sys_sizes / 1024.0 / 1024.0, 4)

    # Process RSS / OS-level bytes (if requested and psutil available)
    if include_process and psutil is not None:
        p = psutil.Process(os.getpid())
        rss = p.memory_info().rss
        info['process_rss_bytes'] = int(rss)
        info['process_rss_mb'] = round(rss / 1024.0 / 1024.0, 4)
    else:
        info['process_rss_bytes'] = None
        info['process_rss_mb'] = None
        if include_process and psutil is None:
            info['psutil_missing'] = True

    # include expansion diagnostics if present
    info['expansion_in_progress'] = getattr(ds, '_expansion_in_progress', False)
    info['expansion_ms'] = getattr(ds, '_expansion_ms', None)

    return info


if __name__ == "__main__":
    print("dataset_core (on-demand expansion; 3-byte path optimized) loaded.")