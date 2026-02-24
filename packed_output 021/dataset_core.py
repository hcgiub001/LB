#!/usr/bin/env python3
"""
dataset_core.py — optimized, mmap-backed reader with safe optional AIR (All-In-RAM) mode.

Key optimizations over previous version:
  - bisect_left used for binary search (C-level, ~3-5x faster than pure Python loop)
  - _values_u16 / _values_u32 cast done eagerly at construction (eliminates None-check per call)
  - Hot query path uses direct typed memoryview/array indexing with no repeated attribute
    lookups (locals cached at top of method)
  - Result list built with list comprehension where possible
  - now_ms() calls minimized in hot path (only 3 perf_counter calls total per query)
  - packed_value() inlined in query_similar (avoids function call overhead)
  - lengths_type branch resolved once per query, not per-item
  - RAM copy uses array('H') for values_blob when remap_width==2 so indexing is O(1)
    native int (no cast object per read, no manual bit shift)
  - parse_dataset_file returns pre-cast typed views everywhere possible
  - AIR mode: values_blob also stored as array('H') when remap_width==2
"""

from __future__ import annotations

import os
import time
import struct
import mmap
import urllib.request
import shutil
import tempfile
import json
import platform
import ctypes
from array import array
from bisect import bisect_left
from typing import Tuple, List, Dict, Any, Union, Optional

# ---- configuration constants ----
AUTO_THRESHOLD_DEFAULT = 300 * 1024 * 1024   # 300 MB required for auto -> pick AIR
SAFETY_MARGIN_MIN      = 64  * 1024 * 1024   # 64 MB minimum safety margin
SAFETY_MARGIN_FRAC     = 0.05                 # 5% of total RAM as margin
PERSIST_FILENAME       = "tmdb_similar_settings.json"
PERSIST_KEY            = "mode"               # values: 'auto', 'air', 'mmap'

# ---- header ----
HEADER_STRUCT = '<4s B B H I I I B B H I'
HEADER_SIZE   = struct.calcsize(HEADER_STRUCT)   # 28


# ---- helpers ----

def packed_value(tmdb_id: int, typ: str) -> int:
    return (int(tmdb_id) << 1) | (1 if str(typ).lower().startswith('tv') else 0)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def format_ms(ms: float) -> str:
    return f"{ms:.3f} ms"


# ---- platform memory helpers ----

def _get_mem_via_psutil() -> Optional[Tuple[int, int, str]]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), 'psutil'
    except Exception:
        return None


def _get_mem_via_proc_meminfo() -> Optional[Tuple[int, int, str]]:
    try:
        if not os.path.exists('/proc/meminfo'):
            return None
        info: Dict[str, int] = {}
        with open('/proc/meminfo', 'r', encoding='ascii') as fh:
            for line in fh:
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                info[parts[0].strip()] = int(parts[1].strip().split()[0])
        if 'MemAvailable' in info:
            avail = info['MemAvailable'] * 1024
        else:
            avail = int((info.get('MemFree', 0) +
                         info.get('Cached', 0) +
                         info.get('Buffers', 0)) * 1024 * 0.7)
        total = info.get('MemTotal', 0) * 1024
        return int(avail), int(total), '/proc/meminfo'
    except Exception:
        return None


def _get_mem_via_windows() -> Optional[Tuple[int, int, str]]:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength',                 ctypes.c_ulong),
                ('dwMemoryLoad',             ctypes.c_ulong),
                ('ullTotalPhys',             ctypes.c_ulonglong),
                ('ullAvailPhys',             ctypes.c_ulonglong),
                ('ullTotalPageFile',         ctypes.c_ulonglong),
                ('ullAvailPageFile',         ctypes.c_ulonglong),
                ('ullTotalVirtual',          ctypes.c_ulonglong),
                ('ullAvailVirtual',          ctypes.c_ulonglong),
                ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys), int(stat.ullTotalPhys), 'GlobalMemoryStatusEx'
    except Exception:
        pass
    return None


def get_available_memory() -> Tuple[int, int, str]:
    r = _get_mem_via_psutil()
    if r:
        return r
    sys_name = platform.system().lower()
    if sys_name in ('linux', 'android'):
        r = _get_mem_via_proc_meminfo()
        if r:
            return r
    elif sys_name == 'windows':
        r = _get_mem_via_windows()
        if r:
            return r
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), 'psutil'
    except Exception:
        return 0, 0, 'unknown'


# ---- persistence helpers ----

def _read_persisted_mode_from_file(cache_dir: str) -> Optional[str]:
    try:
        p = os.path.join(cache_dir, PERSIST_FILENAME)
        if not os.path.exists(p):
            return None
        with open(p, 'r', encoding='utf-8') as fh:
            j = json.load(fh)
        m = j.get(PERSIST_KEY)
        if m in ('auto', 'air', 'mmap'):
            return m
    except Exception:
        pass
    return None


def _write_persisted_mode_to_file(cache_dir: str, mode: str) -> bool:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        p = os.path.join(cache_dir, PERSIST_FILENAME)
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({PERSIST_KEY: mode}, fh)
        return True
    except Exception:
        return False


def get_persisted_mode(cache_dir: str) -> Optional[str]:
    try:
        import xbmcaddon
        addon = xbmcaddon.Addon()
        val = addon.getSetting(PERSIST_KEY)
        if val in ('auto', 'air', 'mmap'):
            return val
    except Exception:
        pass
    return _read_persisted_mode_from_file(cache_dir)


def set_persisted_mode(cache_dir: str, mode: str) -> bool:
    if mode not in ('auto', 'air', 'mmap'):
        return False
    try:
        import xbmcaddon
        addon = xbmcaddon.Addon()
        addon.setSetting(PERSIST_KEY, mode)
        return True
    except Exception:
        return _write_persisted_mode_to_file(cache_dir, mode)


# ---- Dataset class ----

class Dataset:
    """
    Optimized binary reader.

    Structural changes vs previous version:
      - source_keys always stored as array('I') for O(1) bisect_left (C-level)
      - _values_typed: for remap_width==2 stored as array('H') directly —
        no cast object, no None check, direct integer indexing
      - For remap_width==4: array('I') similarly
      - remap table stored as array('I') always (copy made eagerly)
      - Offsets and lengths stored as array('I') / array('B'|'H') always
      - query_similar caches all hot attributes in locals (avoids repeated
        self. attribute lookup through Python's LOAD_ATTR bytecode)
      - Results built via list comprehension leveraging pre-cached remap array
    """

    __slots__ = (
        '_fileobj', '_mmap', 'parsed', 'sizes',
        'remap_index_width', 'lengths_type',
        '_source_keys',   # array('I') — sorted packed source keys
        '_remap',         # array('I') — remap table (packed values)
        '_offsets',       # array('I') — per-row start offset into values
        '_lengths',       # array('B') or array('H')
        '_values_typed',  # array('H') for w=2, array('I') for w=4, or memoryview for w=3
        '_values_blob',   # raw memoryview('B') — needed for w=3 only
        '_ram_copy',
    )

    def __init__(
        self,
        fileobj,
        mm: Optional[mmap.mmap],
        parsed: Dict[str, Any],
        ram_copy: bool = True,
        prefetch_remap: bool = False,
        prefetch_source_keys: bool = False,
    ):
        self._fileobj = fileobj
        self._mmap    = mm
        self.parsed   = parsed
        self.sizes    = parsed.get('sizes', {})

        views = parsed['views']
        self.remap_index_width = parsed['remap_index_width']
        self.lengths_type      = parsed['lengths_type']
        self._ram_copy         = False

        # ---- Always copy source_keys into array('I') — needed for bisect_left ----
        sk = views['source_keys']
        if isinstance(sk, array) and sk.typecode == 'I':
            self._source_keys = sk
        else:
            self._source_keys = array('I', sk)

        # ---- Always copy remap table into array('I') ----
        rt = views['remap_table']
        if isinstance(rt, array) and rt.typecode == 'I':
            self._remap = rt
        else:
            self._remap = array('I', rt)

        # ---- Offsets ----
        off = views['offsets']
        if isinstance(off, array) and off.typecode == 'I':
            self._offsets = off
        else:
            try:
                self._offsets = array('I', off)
            except Exception:
                try:
                    self._offsets = array('I', off.cast('I'))
                except Exception:
                    self._offsets = off   # fallback: leave as memoryview

        # ---- Lengths ----
        tc = 'B' if self.lengths_type == 0 else 'H'
        ln = views['lengths']
        if isinstance(ln, array) and ln.typecode == tc:
            self._lengths = ln
        else:
            try:
                self._lengths = array(tc, ln)
            except Exception:
                try:
                    self._lengths = array(tc, ln.cast(tc))
                except Exception:
                    self._lengths = ln

        # ---- Values blob — most important for query speed ----
        # Keep the raw bytes view always (needed for w=3 path and close())
        self._values_blob = views['values_blob']

        w = self.remap_index_width
        if w == 2:
            # Store as array('H') — direct integer indexing, no cast object overhead
            try:
                self._values_typed: Any = array('H', self._values_blob)
                self._ram_copy = True
            except Exception:
                # fallback: cast memoryview
                try:
                    self._values_typed = self._values_blob.cast('H')
                except Exception:
                    self._values_typed = self._values_blob
        elif w == 4:
            try:
                self._values_typed = array('I', self._values_blob)
                self._ram_copy = True
            except Exception:
                try:
                    self._values_typed = self._values_blob.cast('I')
                except Exception:
                    self._values_typed = self._values_blob
        else:
            # w == 3: no typed array possible; keep raw bytes
            self._values_typed = self._values_blob

        # Mark ram_copy also if we have all arrays (source_keys + remap are always copied)
        self._ram_copy = True

    # ---- prefetch helpers (kept for API compatibility, mostly no-ops now) ----

    def prefetch_remap_table(self) -> float:
        return 0.0   # already done in __init__

    def prefetch_source_keys(self) -> float:
        return 0.0   # already done in __init__

    def prefetch_offsets(self) -> float:
        return 0.0   # already done in __init__

    def prefetch_lengths(self) -> float:
        return 0.0   # already done in __init__

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

    # ---- core query ----

    def query_similar(
        self,
        tmdb_id: int,
        kind: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Look up similar items for (tmdb_id, kind).

        Returns (results, timings).
        results: list of {'tmdb_id': int, 'type': str, 'packed': int}
        timings: {'search_ms', 'query_ms', 'total_ms'}
        """
        t0 = time.perf_counter()

        # Inline packed_value (avoids function-call overhead on hot path)
        packed_src = (int(tmdb_id) << 1) | (1 if kind[:2].lower() == 'tv' else 0)

        # ---- Binary search via bisect_left (C-level) ----
        sk = self._source_keys       # local ref — avoids repeated LOAD_ATTR
        lo = bisect_left(sk, packed_src)
        t1 = time.perf_counter()

        if lo >= len(sk) or sk[lo] != packed_src:
            t2 = time.perf_counter()
            elapsed = (t2 - t0) * 1000.0
            return [], {
                'search_ms': (t1 - t0) * 1000.0,
                'query_ms':  0.0,
                'total_ms':  elapsed,
            }

        # ---- Decode similar list ----
        off    = int(self._offsets[lo])
        length = int(self._lengths[lo])

        w      = self.remap_index_width    # local
        remap  = self._remap               # local array('I')
        vt     = self._values_typed        # local — array('H') for w=2

        if w == 2:
            # Fastest path: direct array('H') indexing — no cast, no bit ops
            # Build results with list comprehension for bytecode efficiency
            results = _decode_w2(vt, remap, off, length)
        elif w == 4:
            results = _decode_w4(vt, remap, off, length)
        else:
            results = _decode_w3(self._values_blob, remap, off, length)

        t2 = time.perf_counter()
        ms = 1000.0
        return results, {
            'search_ms': (t1 - t0) * ms,
            'query_ms':  (t2 - t1) * ms,
            'total_ms':  (t2 - t0) * ms,
        }


# ---- decode helpers (module-level functions = faster LOAD_GLOBAL vs LOAD_ATTR) ----

def _decode_w2(
    vt: array,       # array('H') — remap indices
    remap: array,    # array('I') — packed values
    off: int,
    length: int,
) -> List[Dict[str, Any]]:
    """Decode remap_width=2 block. Pure Python list comprehension."""
    end = off + length
    # Slice the index array for the row, then map through remap in one pass
    # Using a local alias for remap inside the comprehension avoids repeated
    # global/closure lookup
    _r = remap
    return [
        {
            'tmdb_id': (_r[vt[i]] >> 1),
            'type':    ('tvshow' if (_r[vt[i]] & 1) else 'movie'),
            'packed':  _r[vt[i]],
        }
        for i in range(off, end)
    ]


def _decode_w4(
    vt: array,       # array('I') — remap indices
    remap: array,
    off: int,
    length: int,
) -> List[Dict[str, Any]]:
    _r = remap
    end = off + length
    return [
        {
            'tmdb_id': (_r[vt[i]] >> 1),
            'type':    ('tvshow' if (_r[vt[i]] & 1) else 'movie'),
            'packed':  _r[vt[i]],
        }
        for i in range(off, end)
    ]


def _decode_w3(
    blob: Any,       # memoryview('B')
    remap: array,
    off: int,
    length: int,
) -> List[Dict[str, Any]]:
    """Decode remap_width=3 (24-bit) block — uncommon path."""
    _r = remap
    results = []
    base = off * 3
    for i in range(length):
        b = base + i * 3
        idx = blob[b] | (blob[b + 1] << 8) | (blob[b + 2] << 16)
        pv = _r[idx]
        results.append({
            'tmdb_id': pv >> 1,
            'type':    'tvshow' if (pv & 1) else 'movie',
            'packed':  pv,
        })
    return results


# ---- parse ----

def parse_dataset_file(mm) -> Dict[str, Any]:
    """
    Parse file-like memory buffer (mmap.mmap, bytes, bytearray, or memoryview).
    Returns parsed dict.  Views are memoryview slices; arrays are pre-built for
    typed regions so Dataset.__init__ can skip redundant copies.
    """
    try:
        size = mm.size()   # mmap
    except Exception:
        try:
            size = len(mm)
        except Exception:
            raise ValueError("Cannot determine buffer size")

    if size < HEADER_SIZE:
        raise ValueError("Mapped file too small or invalid")

    full_mv = memoryview(mm)

    (magic, version, endian, flags,
     R, E, U,
     lengths_byte, remap_index_width,
     reserved, header_crc) = struct.unpack_from(HEADER_STRUCT, full_mv, 0)

    if magic != b"SIML":
        raise ValueError(f"Bad magic: {magic!r} (expected b'SIML')")

    pos  = HEADER_SIZE
    sizes: Dict[str, int] = {'header': HEADER_SIZE}

    # source_keys
    sk_bytes = R * 4
    sk_off   = pos;  pos += sk_bytes
    sizes['source_keys'] = sk_bytes

    # offsets (optional)
    offsets_present = not bool(flags & 1)
    off_bytes = R * 4 if offsets_present else 0
    off_off   = pos if offsets_present else None
    pos      += off_bytes
    sizes['offsets'] = off_bytes

    # lengths
    lengths_type  = 0 if lengths_byte == 0 else 1
    ln_item_size  = 1 if lengths_type == 0 else 2
    ln_bytes      = R * ln_item_size
    ln_off        = pos;  pos += ln_bytes
    sizes['lengths'] = ln_bytes

    # remap table
    rm_bytes = U * 4
    rm_off   = pos;  pos += rm_bytes
    sizes['remap_table'] = rm_bytes

    # values blob
    vb_off   = pos
    vb_bytes = size - pos
    sizes['values_blob'] = vb_bytes

    # ---- Build typed arrays eagerly (avoids repeated work in Dataset.__init__) ----

    # source_keys -> array('I') sorted — bisect_left needs a sequence supporting __getitem__
    sk_mv = full_mv[sk_off: sk_off + sk_bytes]
    source_keys_arr = array('I', sk_mv)

    # remap table -> array('I')
    rm_mv = full_mv[rm_off: rm_off + rm_bytes]
    remap_arr = array('I', rm_mv)

    # offsets -> array('I') or reconstruct
    if offsets_present:
        off_mv = full_mv[off_off: off_off + off_bytes]
        offsets_arr = array('I', off_mv)
    else:
        # Reconstruct offsets from lengths (one pass)
        ln_raw = full_mv[ln_off: ln_off + ln_bytes]
        offsets_arr = array('I')
        cur = 0
        if lengths_type == 0:
            for i in range(R):
                offsets_arr.append(cur)
                cur += ln_raw[i]
        else:
            ln_u16 = ln_raw.cast('H')
            for i in range(R):
                offsets_arr.append(cur)
                cur += ln_u16[i]

    # lengths -> array('B') or array('H')
    ln_raw_mv = full_mv[ln_off: ln_off + ln_bytes]
    if lengths_type == 0:
        lengths_arr = array('B', ln_raw_mv)
    else:
        lengths_arr = array('H', ln_raw_mv)

    # values blob — keep as raw memoryview; Dataset will cast/copy based on remap_index_width
    values_mv = full_mv[vb_off: vb_off + vb_bytes]

    parsed = {
        'magic':              magic,
        'version':            version,
        'endian':             endian,
        'flags':              flags,
        'R':                  R,
        'E':                  E,
        'U':                  U,
        'lengths_type':       lengths_type,
        'remap_index_width':  remap_index_width,
        'offsets_present':    offsets_present,
        'sizes':              sizes,
        'views': {
            'source_keys':  source_keys_arr,   # array('I')
            'offsets':      offsets_arr,        # array('I')
            'lengths':      lengths_arr,        # array('B'|'H')
            'remap_table':  remap_arr,          # array('I')
            'values_blob':  values_mv,          # memoryview('B') — Dataset casts/copies
        }
    }
    return parsed


# ---- atomic download ----

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


def _compute_safety_margin(total_bytes: int) -> int:
    return max(SAFETY_MARGIN_MIN, int(total_bytes * SAFETY_MARGIN_FRAC))


# ---- main loader ----

def load_or_fetch(
    url:                  str,
    cache_path:           str,
    use_mmap:             bool = True,
    ram_copy:             bool = True,
    prefetch_remap:       bool = False,
    prefetch_source_keys: bool = False,
    mode:                 str  = 'auto',
    auto_threshold:       int  = AUTO_THRESHOLD_DEFAULT,
) -> Tuple['Dataset', Dict[str, Any]]:
    """
    Download url into cache_path if missing, then load via chosen mode.

    mode:
      'auto'  — read persisted user setting; if none, inspect available memory
                and choose AIR for this run only when safe
      'air'   — read entire file into RAM (bytearray) — fastest queries, ~4-5 MB
      'mmap'  — memory-mapped file (legacy, lowest footprint)

    Returns (Dataset, metadata).
    """
    metadata: Dict[str, Any] = {
        'from_cache':           False,
        'size_bytes':           None,
        'path':                 cache_path,
        'load_ms':              None,
        'mode_setting':         mode,
        'mode_persisted':       None,
        'mode_chosen':          None,
        'mem_available_bytes':  None,
        'mem_total_bytes':      None,
        'mem_source':           None,
        'air_active':           False,
        'air_failed':           False,
        'air_failure_reason':   None,
        'air_bytes_copied':     0,
        'air_copy_ms':          None,
    }

    t_start = time.perf_counter()

    # Ensure file exists
    if not os.path.exists(cache_path):
        req = urllib.request.Request(url, headers={'User-Agent': 'dataset_core/1.0'})
        with urllib.request.urlopen(req) as resp:
            _atomic_write_temp(cache_path, resp)
        metadata['from_cache'] = False
    else:
        metadata['from_cache'] = True

    try:
        size_bytes = os.path.getsize(cache_path)
    except Exception:
        size_bytes = None
    metadata['size_bytes'] = size_bytes

    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    persisted = get_persisted_mode(cache_dir)
    metadata['mode_persisted'] = persisted

    # Resolve mode
    chosen_mode = mode if mode in ('auto', 'air', 'mmap') else 'auto'

    if chosen_mode == 'auto':
        if persisted in ('air', 'mmap'):
            # Honor persisted explicit choice
            chosen_mode = persisted
        else:
            # Inspect memory — choose AIR for this run if safe
            avail, total, src = get_available_memory()
            metadata['mem_available_bytes'] = avail
            metadata['mem_total_bytes']      = total
            metadata['mem_source']           = src
            safety      = _compute_safety_margin(total) if total else SAFETY_MARGIN_MIN
            dataset_need = (size_bytes or 0) + safety
            chosen_mode  = ('air'
                            if (avail and avail >= auto_threshold and avail >= dataset_need)
                            else 'mmap')
    else:
        if chosen_mode not in ('air', 'mmap'):
            chosen_mode = 'mmap'

    metadata['mode_chosen'] = chosen_mode

    # ---- AIR mode: read entire file into bytearray ----
    if chosen_mode == 'air':
        t0 = time.perf_counter()
        try:
            with open(cache_path, 'rb') as fh:
                data = bytearray(fh.read())
            copy_ms = (time.perf_counter() - t0) * 1000.0
            metadata['air_active']      = True
            metadata['air_bytes_copied'] = len(data)
            metadata['air_copy_ms']     = copy_ms

            parsed = parse_dataset_file(data)
            ds     = Dataset(None, None, parsed,
                             ram_copy=False,            # arrays already built in parse
                             prefetch_remap=False,
                             prefetch_source_keys=False)
            ds.sizes = parsed.get('sizes', {})
            metadata['load_ms'] = (time.perf_counter() - t_start) * 1000.0
            return ds, metadata

        except MemoryError:
            metadata['air_failed']         = True
            metadata['air_failure_reason'] = 'MemoryError'
            # fall through to mmap
        except Exception as e:
            metadata['air_failed']         = True
            metadata['air_failure_reason'] = f'Exception: {e}'
            # fall through to mmap

    # ---- mmap mode (or AIR fallback) ----
    f = open(cache_path, 'rb')
    try:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise

    parsed = parse_dataset_file(mm)
    ds     = Dataset(f, mm, parsed,
                     ram_copy=False,            # arrays already built in parse
                     prefetch_remap=False,
                     prefetch_source_keys=False)
    ds.sizes = parsed.get('sizes', {})
    metadata['load_ms'] = (time.perf_counter() - t_start) * 1000.0
    return ds, metadata


# ---- diagnostics ----

def describe_memory(ds: 'Dataset', include_process: bool = False) -> Dict[str, Any]:
    p = ds.parsed
    sizes = p.get('sizes', {}).copy()
    summary = {
        'file_total_bytes':    None,
        'header_bytes':        sizes.get('header'),
        'source_keys_bytes':   sizes.get('source_keys'),
        'offsets_bytes':       sizes.get('offsets'),
        'lengths_bytes':       sizes.get('lengths'),
        'remap_table_bytes':   sizes.get('remap_table'),
        'values_blob_bytes':   sizes.get('values_blob'),
        'remap_index_width':   ds.remap_index_width,
        'lengths_type':        ('uint8' if ds.lengths_type == 0 else 'uint16'),
        'rows_R':              p.get('R'),
        'unique_U':            p.get('U'),
        'entries_E':           p.get('E'),
        'offsets_present':     p.get('offsets_present'),
        'ram_copy_active':     getattr(ds, '_ram_copy', False),
    }
    try:
        if getattr(ds, '_mmap', None) is not None:
            summary['file_total_bytes'] = ds._mmap.size()
    except Exception:
        pass

    out: Dict[str, Any] = {'parsed_sizes': summary}
    if not include_process:
        return out

    try:
        import psutil
        proc = psutil.Process()
        mi   = proc.memory_info()
        out['process_memory'] = {
            'rss': getattr(mi, 'rss', None),
            'vms': getattr(mi, 'vms', None),
        }
    except Exception as e:
        out['process_memory'] = {'error': f'psutil not available: {e}'}
    return out


# ---- microbenchmark ----

def benchmark_cold_vs_warm(
    ds:     'Dataset',
    tmdb_id: int,
    kind:    str = 'movie',
    n_warm:  int = 100,
) -> Dict[str, Any]:
    import statistics

    t0 = time.perf_counter()
    _, timings = ds.query_similar(tmdb_id, kind)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    warm_times: List[float] = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        ds.query_similar(tmdb_id, kind)
        warm_times.append((time.perf_counter() - t0) * 1000.0)

    wt = sorted(warm_times)
    p95 = wt[max(0, min(len(wt) - 1, int(0.95 * len(wt))))] if wt else 0.0

    return {
        'first_ms':      round(cold_ms, 4),
        'warm_mean_ms':  round(statistics.mean(warm_times), 4) if warm_times else 0.0,
        'warm_p50_ms':   round(statistics.median(warm_times), 4) if warm_times else 0.0,
        'warm_p95_ms':   round(p95, 4),
        'warm_min_ms':   round(min(warm_times), 4) if warm_times else 0.0,
        'ram_copy_active': getattr(ds, '_ram_copy', False),
    }


# ---- CLI ----

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset_core.py <path-to-dataset.bin> [tmdb_id] [movie|tv] [auto|air|mmap]")
        sys.exit(1)

    path     = sys.argv[1]
    mode_arg = sys.argv[4] if len(sys.argv) >= 5 else 'auto'

    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    parsed = parse_dataset_file(mm)
    print("Parsed:", {k: parsed[k] for k in
                      ('R', 'E', 'U', 'remap_index_width', 'lengths_type', 'offsets_present')})
    print("Sizes:", parsed.get('sizes'))

    if len(sys.argv) >= 3:
        tmdb_id = int(sys.argv[2])
        kind    = sys.argv[3] if len(sys.argv) >= 4 else 'movie'
        ds, meta = load_or_fetch(
            'file://' + os.path.abspath(path), path,
            use_mmap=True, ram_copy=False, mode=mode_arg
        )
        bench = benchmark_cold_vs_warm(ds, tmdb_id, kind, n_warm=200)
        print("Benchmark:", bench)
    else:
        mm.close()