#!/usr/bin/env python3
"""
dataset_core.py — copy-safe, mmap-backed reader with safe optional AIR (All-In-RAM) mode.

This file extends the previous dataset_core with an "AIR" (All-In-RAM) mode that
reads the entire dataset.bin into process RAM at load time (approx 4–5 MB for your
dataset) so queries never fault on values_blob. There's also an "auto" mode that
will pick AIR for this run only when the device reports sufficient free memory.

Behavior summary:
  - load_or_fetch(..., mode='auto') -> reads persisted user setting (if any),
    otherwise inspects available memory and chooses AIR only for this run when safe.
  - load_or_fetch(..., mode='air') -> blocking, reads whole file into RAM (bytearray)
  - load_or_fetch(..., mode='mmap') -> legacy mmap-backed behavior
Persistence:
  - If xbmcaddon is available (running inside Kodi) we try to read/write the
    Addon setting 'tmdb_similar_mode'. Otherwise we persist to <cache_dir>/settings.json.
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
from typing import Tuple, List, Dict, Any, Union, Optional

# ---- configuration constants ----
AUTO_THRESHOLD_DEFAULT = 300 * 1024 * 1024  # 300 MB required for auto -> pick AIR
SAFETY_MARGIN_MIN = 64 * 1024 * 1024       # 64 MB minimum safety margin
SAFETY_MARGIN_FRAC = 0.05                  # 5% of total RAM as margin
PERSIST_FILENAME = "tmdb_similar_settings.json"
PERSIST_KEY = "mode"  # values: 'auto', 'air', 'mmap'

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

# ---- platform memory helpers ----
def _get_mem_via_psutil() -> Optional[Tuple[int,int,str]]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), 'psutil'
    except Exception:
        return None

def _get_mem_via_proc_meminfo() -> Optional[Tuple[int,int,str]]:
    # Linux / Android fallback using /proc/meminfo
    try:
        if not os.path.exists('/proc/meminfo'):
            return None
        info = {}
        with open('/proc/meminfo', 'r', encoding='ascii') as fh:
            for line in fh:
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                info[key] = int(val)  # value in kB
        # Prefer MemAvailable if present (Linux >= 3.14)
        if 'MemAvailable' in info:
            avail = info['MemAvailable'] * 1024
        else:
            # Conservative fallback: free + cached + buffers
            free = info.get('MemFree', 0)
            cached = info.get('Cached', 0)
            buffers = info.get('Buffers', 0)
            avail = (free + cached + buffers) * 1024
            # conservative factor: only 70% of cache is considered quickly reclaimable
            avail = int(avail * 0.7)
        total = info.get('MemTotal', 0) * 1024
        return int(avail), int(total), '/proc/meminfo'
    except Exception:
        return None

def _get_mem_via_windows() -> Optional[Tuple[int,int,str]]:
    # Use GlobalMemoryStatusEx via ctypes
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            avail = int(stat.ullAvailPhys)
            total = int(stat.ullTotalPhys)
            return avail, total, 'GlobalMemoryStatusEx'
    except Exception:
        return None

def get_available_memory() -> Tuple[int,int,str]:
    """
    Return (available_bytes, total_bytes, source_str).
    Best-effort cross-platform detection; psutil preferred, then platform-specific.
    On failure returns (0, 0, 'unknown').
    """
    # Try psutil first
    r = _get_mem_via_psutil()
    if r:
        return r
    # Platform specific
    sys = platform.system().lower()
    if sys == 'linux' or sys == 'android':
        r = _get_mem_via_proc_meminfo()
        if r:
            return r
    elif sys == 'windows':
        r = _get_mem_via_windows()
        if r:
            return r
    # macOS / fallback: try psutil again or give conservative fallback
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), 'psutil'
    except Exception:
        # last resort guess: return 0 available to avoid choosing AIR incorrectly
        return 0, 0, 'unknown'

# ---- persistence helpers (xbmc or file fallback) ----
def _read_persisted_mode_from_file(cache_dir: str) -> Optional[str]:
    try:
        p = os.path.join(cache_dir, PERSIST_FILENAME)
        if not os.path.exists(p):
            return None
        with open(p, 'r', encoding='utf-8') as fh:
            j = json.load(fh)
            m = j.get(PERSIST_KEY)
            if m in ('auto','air','mmap'):
                return m
    except Exception:
        pass
    return None

def _write_persisted_mode_to_file(cache_dir: str, mode: str) -> bool:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        p = os.path.join(cache_dir, PERSIST_FILENAME)
        d = {PERSIST_KEY: mode}
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(d, fh)
        return True
    except Exception:
        return False

def get_persisted_mode(cache_dir: str) -> Optional[str]:
    # Try Kodi xbmcaddon if available
    try:
        import xbmcaddon
        addon = xbmcaddon.Addon()
        val = addon.getSetting(PERSIST_KEY)
        if val:
            if val in ('auto','air','mmap'):
                return val
    except Exception:
        pass
    # Fallback to file
    return _read_persisted_mode_from_file(cache_dir)

def set_persisted_mode(cache_dir: str, mode: str) -> bool:
    if mode not in ('auto','air','mmap'):
        return False
    try:
        import xbmcaddon
        addon = xbmcaddon.Addon()
        addon.setSetting(PERSIST_KEY, mode)
        return True
    except Exception:
        return _write_persisted_mode_to_file(cache_dir, mode)

# ---- Dataset class (mostly unchanged, with AIR support handled at loader) ----
class Dataset:
    """
    Zero-copy mmap reader with optional (safe) RAM-copy prefetch for hot regions.

    Constructor:
        Dataset(fileobj, mm, parsed, ram_copy=True, prefetch_remap=False, prefetch_source_keys=False)

    Note: fileobj/mm may be None in AIR mode (parsed views can be backed by a bytearray).
    """

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

        # If requested, attempt eager ram_copy of all 4 small regions; measure time isn't returned here,
        # but load_or_fetch will include constructor time in metadata['load_ms'].
        if ram_copy:
            try:
                # copy source_keys and remap_table (uint32 arrays)
                self._source_keys = array('I', views['source_keys'])
                self._remap = array('I', views['remap_table'])

                # copy offsets (uint32). views['offsets'] may already be an array('I') or memoryview('I')
                try:
                    self._offsets = array('I', views['offsets'])
                except Exception:
                    # fallback: if offsets is a memoryview of bytes, ensure cast
                    try:
                        self._offsets = array('I', views['offsets'].cast('I'))
                    except Exception:
                        # leave as-is (mmap-backed or memory-backed) on failure
                        self._offsets = views['offsets']

                # copy lengths: respect lengths_type (0 => uint8 / 'B', 1 => uint16 / 'H')
                try:
                    if self.lengths_type == 0:
                        # lengths is 1 byte per entry
                        self._lengths = array('B', views['lengths'])
                    else:
                        self._lengths = array('H', views['lengths'])
                except Exception:
                    # memoryview might need casting first
                    try:
                        if self.lengths_type == 0:
                            self._lengths = array('B', views['lengths'].cast('B'))
                        else:
                            self._lengths = array('H', views['lengths'].cast('H'))
                    except Exception:
                        # fallback: keep mmap-backed lengths
                        self._lengths = views['lengths']

                self._ram_copy = True
            except Exception:
                # fallback to mmap-backed views (keep running)
                self._source_keys = views['source_keys']
                self._remap = views['remap_table']
                self._offsets = views['offsets']
                self._lengths = views['lengths']
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

    def prefetch_offsets(self) -> float:
        if isinstance(self._offsets, array):
            return 0.0
        t0 = time.perf_counter()
        try:
            arr = array('I', self._offsets)
            self._offsets = arr
            self._ram_copy = True
            return (time.perf_counter() - t0) * 1000.0
        except Exception:
            try:
                arr = array('I', self._offsets.cast('I'))
                self._offsets = arr
                self._ram_copy = True
                return (time.perf_counter() - t0) * 1000.0
            except Exception:
                return 0.0

    def prefetch_lengths(self) -> float:
        desired_type = 'B' if self.lengths_type == 0 else 'H'
        if isinstance(self._lengths, array) and self._lengths.typecode == desired_type:
            return 0.0
        t0 = time.perf_counter()
        try:
            if self.lengths_type == 0:
                arr = array('B', self._lengths)
            else:
                arr = array('H', self._lengths)
            self._lengths = arr
            self._ram_copy = True
            return (time.perf_counter() - t0) * 1000.0
        except Exception:
            try:
                if self.lengths_type == 0:
                    arr = array('B', self._lengths.cast('B'))
                else:
                    arr = array('H', self._lengths.cast('H'))
                self._lengths = arr
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

# ---- parse (updated to accept mmap or any buffer supporting len()) ----
def parse_dataset_file(mm) -> Dict[str, Any]:
    """
    Parse file-like memory buffer (mmap.mmap or bytes/bytearray/memoryview).
    Returns parsed dict with zero-copy memoryviews into the provided buffer.
    """
    # determine size
    try:
        size = mm.size()  # works for mmap
    except Exception:
        try:
            size = len(mm)
        except Exception:
            raise ValueError("Mapped file too small or invalid (cannot determine size)")

    if size < HEADER_SIZE:
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
    values_bytes = size - pos
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

# ---- atomic download + loader with mode selection (mmap | air | auto) ----
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

def load_or_fetch(
    url: str,
    cache_path: str,
    use_mmap: bool = True,
    ram_copy: bool = True,
    prefetch_remap: bool = False,
    prefetch_source_keys: bool = False,
    mode: str = 'auto',
    auto_threshold: int = AUTO_THRESHOLD_DEFAULT,
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Download url into cache_path if missing and memory-map it OR load into RAM in AIR mode.

    mode:
      - 'auto'  : read persisted mode or choose AIR for this run only if memory available
      - 'air'   : read entire file into RAM (blocking)
      - 'mmap'  : mmap-backed (default legacy)

    Returns (Dataset, metadata), metadata includes information about the chosen mode,
    memory detection, and AIR copy statistics if applicable.
    """
    metadata: Dict[str, Any] = {
        'from_cache': False,
        'size_bytes': None,
        'path': cache_path,
        'load_ms': None,
        'mode_setting': mode,
        'mode_persisted': None,
        'mode_chosen': None,
        'mem_available_bytes': None,
        'mem_total_bytes': None,
        'mem_source': None,
        'air_active': False,
        'air_failed': False,
        'air_failure_reason': None,
        'air_bytes_copied': 0,
        'air_copy_ms': None,
    }
    t_start = time.perf_counter()

    # ensure file exists (download if needed)
    if not os.path.exists(cache_path):
        req = urllib.request.Request(url, headers={'User-Agent': 'dataset_core/1.0'})
        with urllib.request.urlopen(req) as resp:
            _atomic_write_temp(cache_path, resp)
        metadata['from_cache'] = False
    else:
        metadata['from_cache'] = True

    # dataset size known early
    try:
        size_bytes = os.path.getsize(cache_path)
    except Exception:
        size_bytes = None
    metadata['size_bytes'] = size_bytes

    # Determine persisted user mode if any (explicit choice)
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    persisted = get_persisted_mode(cache_dir)
    metadata['mode_persisted'] = persisted if persisted else None

    # Resolve initial choice
    chosen_mode = mode if mode in ('auto','air','mmap') else 'auto'

    # If user has persisted explicit choice and mode == 'auto', honor persisted as explicit
    if chosen_mode == 'auto' and persisted in ('air','mmap'):
        chosen_mode = persisted
        metadata['mode_chosen'] = chosen_mode
        # indicate we used a persisted explicit choice (still show mode_setting=auto)
    else:
        # If still auto: perform inspection to pick for this run only.
        if chosen_mode == 'auto':
            # get available memory
            avail, total, src = get_available_memory()
            metadata['mem_available_bytes'] = avail
            metadata['mem_total_bytes'] = total
            metadata['mem_source'] = src

            # compute safety margin using total (if available)
            if total and total > 0:
                safety = _compute_safety_margin(total)
            else:
                safety = SAFETY_MARGIN_MIN

            # require both a minimum threshold (auto_threshold) and enough to hold dataset + margin
            dataset_need = (size_bytes or 0) + safety
            if avail and avail >= auto_threshold and avail >= dataset_need:
                chosen_mode = 'air'
            else:
                chosen_mode = 'mmap'
            metadata['mode_chosen'] = chosen_mode
        else:
            metadata['mode_chosen'] = chosen_mode

    # Now perform load according to chosen_mode
    if chosen_mode == 'air':
        # Blocking read entire file into RAM (bytearray)
        t0 = time.perf_counter()
        try:
            with open(cache_path, 'rb') as fh:
                data = bytearray(fh.read())  # read entire file into RAM
            copy_ms = (time.perf_counter() - t0) * 1000.0
            metadata['air_active'] = True
            metadata['air_bytes_copied'] = len(data)
            metadata['air_copy_ms'] = copy_ms
            # Parse from the in-RAM buffer
            parsed = parse_dataset_file(data)
            # Construct Dataset backed by in-RAM buffer (no mmap/fileobj)
            ds = Dataset(None, None, parsed, ram_copy=ram_copy,
                         prefetch_remap=prefetch_remap, prefetch_source_keys=prefetch_source_keys)
            ds.sizes = parsed.get('sizes', {})
            metadata['load_ms'] = (time.perf_counter() - t_start) * 1000.0
            return ds, metadata
        except MemoryError as me:
            metadata['air_failed'] = True
            metadata['air_failure_reason'] = 'MemoryError'
            # fallback to mmap below
        except Exception as e:
            metadata['air_failed'] = True
            metadata['air_failure_reason'] = f'Exception: {e}'
            # fallback to mmap below

    # Fallback or chosen mmap mode path: mmap the file and parse
    f = open(cache_path, 'rb')
    try:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise
    parsed = parse_dataset_file(mm)
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
        'file_total_bytes': None,
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
    try:
        if getattr(ds, '_mmap', None) is not None:
            sizes_summary['file_total_bytes'] = ds._mmap.size()
    except Exception:
        sizes_summary['file_total_bytes'] = None

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

# ---- microbenchmark helper (unchanged) ----
def benchmark_cold_vs_warm(
    ds: Dataset,
    tmdb_id: int,
    kind: str = 'movie',
    n_warm: int = 100,
) -> Dict[str, Any]:
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

# ---- CLI (unchanged) ----
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset_core.py <path-to-dataset.bin> [tmdb_id] [movie|tv]")
        sys.exit(1)
    path = sys.argv[1]
    mode_arg = 'auto'
    if len(sys.argv) >= 5:
        mode_arg = sys.argv[4]
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        parsed = parse_dataset_file(mm)
        print("Parsed:", {k: parsed[k] for k in ('R', 'E', 'U', 'remap_index_width', 'lengths_type', 'offsets_present')})
        print("Sizes:", parsed.get('sizes'))

        if len(sys.argv) >= 3:
            tmdb_id = int(sys.argv[2])
            kind = sys.argv[3] if len(sys.argv) >= 4 else 'movie'
            ds, meta = load_or_fetch('file://'+os.path.abspath(path), path, use_mmap=True, ram_copy=True, mode=mode_arg)
            bench = benchmark_cold_vs_warm(ds, tmdb_id, kind, n_warm=200)
            print("Benchmark (ram_copy=True):", bench)
        else:
            mm.close()