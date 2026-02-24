# -*- coding: utf-8 -*-
"""
likely_api.py — fenlight-friendly adapter for the packed "likely/similar" dataset.

This file is intentionally *single-file*: it embeds the minimal dataset reader/loader
(previously in dataset_core.py) directly inside this module so callers only need
"likely_api.py".

Public surface (recommended):
  - ensure_loaded(force=False) -> None
  - get_setting_mode() -> str              # returns user's persisted choice: 'auto'|'RAM'|'mmap'
  - set_setting_mode(mode: str) -> bool    # persist user's explicit choice ('auto','RAM','mmap')
  - get_runtime_mode() -> str              # returns what was actually chosen for this run: 'RAM'|'mmap'
  - query_likely_packed(tmdb_id, kind) -> List[int]                # packed ints (id<<1 | typebit)
  - query_likely_pairs(tmdb_id, kind, timing=False) -> dict       # {"count":N,"results":[[id,type],...]}
  - get_likely_for_addon(tmdb_id, kind, timing=False) -> dict     # {"results":[{"id":..,"media_type":..}],"total_results":N}
  - clear_likely_cache() -> None
  - reload_dataset() -> None

Design notes:
 - Uses fenlight caches (main_cache) and settings (get_setting / set_setting).
 - Caches packed-int results (small) keyed by dataset id + cache version.
 - Does not add CLI/pipe surface — this API is intended to be used in-process by the addon.
 - Default behaviour is minimal and matches fenlight patterns.

Dataset reader/loader notes (embedded):
 - Reads the packed binary dataset format (magic 'SIML').
 - Supports two loading modes:
     * 'air'  => read entire dataset into RAM (fastest queries, uses memory)
     * 'mmap' => memory-map the file (lower RAM use, still fast)
   'auto' chooses based on available memory and dataset size.
 - Exposes Dataset.query_similar_packed(tmdb_id, kind) -> List[int]

"""

from __future__ import annotations

import os
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Fenlight imports
# ---------------------------------------------------------------------
# These are expected to exist inside the addon. If you're running this file
# outside Fenlight/Kodi, provide compatible shims with the same names.
from caches.main_cache import main_cache
from caches.settings_cache import get_setting, set_setting
from modules.kodi_utils import translate_path, kodi_dialog, kodi_log  # noqa: F401


# ---------------------------------------------------------------------
# Embedded dataset loader (minimal subset of dataset_core.py)
# ---------------------------------------------------------------------
# We embed only what likely_api needs:
#   - load_or_fetch(...) that downloads (if missing) and loads dataset.bin
#   - Dataset with query_similar_packed(...)
#   - a small 'auto' mode heuristic using available memory
#   - the dataset binary parser
#
# This keeps the public likely_api surface identical, while eliminating the
# extra dataset_core.py dependency.

import struct
import mmap
import urllib.request
import shutil
import tempfile
import platform
import ctypes
import sys
from array import array
from bisect import bisect_left


# ---- dataset header constants ----
_HEADER_STRUCT = "<4s B B H I I I B B H I"
_HEADER_SIZE = struct.calcsize(_HEADER_STRUCT)  # 28


# ---- memory heuristics (small + robust) ----
# If the system reports enough *available* memory, 'auto' prefers RAM mode.
_AUTO_THRESHOLD_DEFAULT = 300 * 1024 * 1024  # 300MB minimum to prefer RAM
_SAFETY_MARGIN_MIN = 64 * 1024 * 1024
_SAFETY_MARGIN_FRAC = 0.05


def _get_mem_via_psutil() -> Optional[Tuple[int, int, str]]:
    """Best-effort available memory via psutil if installed."""
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), "psutil"
    except Exception:
        return None


def _get_mem_via_proc_meminfo() -> Optional[Tuple[int, int, str]]:
    """Linux/Android fallback memory reading."""
    try:
        if not os.path.exists("/proc/meminfo"):
            return None
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                try:
                    info[key] = int(val)  # kB
                except Exception:
                    pass

        if "MemAvailable" in info:
            avail = info["MemAvailable"] * 1024
        else:
            # Conservative estimate if MemAvailable is not present.
            free = info.get("MemFree", 0)
            cached = info.get("Cached", 0)
            buffers = info.get("Buffers", 0)
            avail = int((free + cached + buffers) * 1024 * 0.7)

        total = info.get("MemTotal", 0) * 1024
        return int(avail), int(total), "/proc/meminfo"
    except Exception:
        return None


def _get_mem_via_windows() -> Optional[Tuple[int, int, str]]:
    """Windows fallback memory reading."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullAvailPhys), int(stat.ullTotalPhys), "GlobalMemoryStatusEx"
        return None
    except Exception:
        return None


def _get_available_memory() -> Tuple[int, int, str]:
    """Return (available_bytes, total_bytes, source_string)."""
    r = _get_mem_via_psutil()
    if r:
        return r

    sysname = platform.system().lower()
    if sysname in ("linux", "android"):
        r = _get_mem_via_proc_meminfo()
        if r:
            return r
    elif sysname == "windows":
        r = _get_mem_via_windows()
        if r:
            return r

    return 0, 0, "unknown"


def _compute_safety_margin(total_bytes: int) -> int:
    """Keep some RAM headroom so RAM mode doesn't push the device into OOM."""
    return max(_SAFETY_MARGIN_MIN, int(total_bytes * _SAFETY_MARGIN_FRAC))


def _packed_value(tmdb_id: int, kind: str) -> int:
    """Pack (tmdb_id, kind) into a single integer: (tmdb_id << 1) | type_bit."""
    # type_bit: 0 == movie, 1 == tv
    return (int(tmdb_id) << 1) | (1 if str(kind).lower().startswith("tv") else 0)


def _atomic_write_temp(target_path: str, data_stream) -> None:
    """Write to a temp file and atomically replace the target (safe against interruptions)."""
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=".tmp_ds_", suffix=".bin")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as out_f:
            shutil.copyfileobj(data_stream, out_f)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _parse_dataset_file(buf) -> Dict[str, Any]:
    """Parse the dataset file buffer and return zero-copy views for fast query."""
    if sys.byteorder != "little":
        raise RuntimeError("Big-endian platform not supported.")

    try:
        size = buf.size()  # mmap
    except Exception:
        size = len(buf)

    if size < _HEADER_SIZE:
        raise ValueError("Buffer too small")

    full_mv = memoryview(buf)

    (
        magic,
        version,
        endian,
        flags,
        R,
        E,
        U,
        lengths_byte,
        remap_index_width,
        reserved,
        header_crc,
    ) = struct.unpack_from(_HEADER_STRUCT, full_mv, 0)

    if magic != b"SIML":
        raise ValueError("Bad magic (not SIML)")

    pos = _HEADER_SIZE

    # Rows are indexed by 'source_keys' which are packed ints (src_tmdb_id<<1 | type_bit)
    source_keys_bytes = R * 4
    source_keys_off = pos
    pos += source_keys_bytes

    # Some datasets store offsets explicitly; others store lengths only and offsets are reconstructible.
    offsets_present = not bool(flags & 1)
    offsets_bytes = R * 4 if offsets_present else 0
    offsets_off = pos if offsets_present else None
    pos += offsets_bytes

    # lengths: either 1 byte each or 2 bytes each.
    if lengths_byte == 0:
        lengths_type = 0
        lengths_bytes = R
    else:
        lengths_type = 1
        lengths_bytes = R * 2

    lengths_off = pos
    pos += lengths_bytes

    # remap table maps “value indices” to packed tmdb ids (same packing scheme).
    remap_bytes = U * 4
    remap_off = pos
    pos += remap_bytes

    # remaining bytes are the values blob: indices into remap table (width = 2,3,4 bytes)
    values_off = pos
    values_bytes = size - pos

    source_keys_mv = full_mv[source_keys_off : source_keys_off + source_keys_bytes].cast("I")

    lengths_raw_mv = full_mv[lengths_off : lengths_off + lengths_bytes]
    lengths_mv = lengths_raw_mv.cast("B") if lengths_type == 0 else lengths_raw_mv.cast("H")

    if offsets_present:
        offsets_mv = full_mv[offsets_off : offsets_off + offsets_bytes].cast("I")  # type: ignore[index]
    else:
        # reconstruct offsets from cumulative lengths.
        offsets_arr = array("I")
        cur = 0
        app = offsets_arr.append
        for l in lengths_mv:
            app(cur)
            cur += int(l)
        offsets_mv = offsets_arr

    remap_mv = full_mv[remap_off : remap_off + remap_bytes].cast("I")
    values_mv = full_mv[values_off : values_off + values_bytes]

    return {
        "magic": magic,
        "version": int(version),
        "endian": int(endian),
        "flags": int(flags),
        "R": int(R),
        "E": int(E),
        "U": int(U),
        "lengths_type": int(lengths_type),
        "remap_index_width": int(remap_index_width),
        "offsets_present": bool(offsets_present),
        "views": {
            "source_keys": source_keys_mv,
            "offsets": offsets_mv,
            "lengths": lengths_mv,
            "remap_table": remap_mv,
            "values_blob": values_mv,
        },
    }


class Dataset:
    """Fast in-process query engine for the packed dataset."""

    __slots__ = (
        "_fileobj",
        "_mmap",
        "parsed",
        "remap_index_width",
        "lengths_type",
        "_source_keys",
        "_offsets",
        "_lengths",
        "_remap",
        "_values_blob",
        "_values_u16",
        "_values_u32",
        "_ram_copy",
    )

    def __init__(self, fileobj, mm: Optional[mmap.mmap], parsed: Dict[str, Any], ram_copy: bool = True):
        self._fileobj = fileobj
        self._mmap = mm
        self.parsed = parsed

        self.remap_index_width = int(parsed["remap_index_width"])
        self.lengths_type = int(parsed["lengths_type"])

        views = parsed["views"]
        self._source_keys = views["source_keys"]
        self._offsets = views["offsets"]
        self._lengths = views["lengths"]
        self._remap = views["remap_table"]
        self._values_blob = views["values_blob"]

        # optional cached casts
        self._values_u16 = None
        self._values_u32 = None

        # RAM-copy tries to turn memoryviews into Python arrays to avoid
        # holding onto mmap and to speed up access on some platforms.
        self._ram_copy = False
        if ram_copy:
            self._attempt_ram_copy()

        # Prepare typed views for common widths (2 and 4).
        if self.remap_index_width == 2:
            try:
                self._values_u16 = self._values_blob.cast("H")
            except Exception:
                self._values_u16 = None
        elif self.remap_index_width == 4:
            try:
                self._values_u32 = self._values_blob.cast("I")
            except Exception:
                self._values_u32 = None

    def _attempt_ram_copy(self) -> None:
        """Best-effort conversion of memoryviews into arrays (optional optimization)."""
        try:
            self._source_keys = array("I", self._source_keys)
            self._remap = array("I", self._remap)
            self._offsets = array("I", self._offsets)
            if self.lengths_type == 0:
                self._lengths = array("B", self._lengths)
            else:
                self._lengths = array("H", self._lengths)
            self._ram_copy = True
        except Exception:
            self._ram_copy = False

    def close(self) -> None:
        """Release mmap and file handles (safe to call multiple times)."""
        try:
            if self._mmap is not None:
                try:
                    self._mmap.close()
                except Exception:
                    pass
                self._mmap = None
        finally:
            if self._fileobj is not None:
                try:
                    self._fileobj.close()
                except Exception:
                    pass
                self._fileobj = None

    def _find_row_index(self, packed_src_key: int) -> int:
        """Binary-search the row index by packed source key."""
        sk = self._source_keys
        i = bisect_left(sk, packed_src_key)
        if i != len(sk) and sk[i] == packed_src_key:
            return i
        return -1

    def query_similar_packed(self, tmdb_id: int, kind: str) -> List[int]:
        """Return ordered list of packed ints (tmdb_id<<1 | typebit)."""
        packed_src = _packed_value(int(tmdb_id), kind)
        idx = self._find_row_index(packed_src)
        if idx < 0:
            return []

        off = int(self._offsets[idx])
        length = int(self._lengths[idx])

        remap = self._remap
        w = self.remap_index_width

        out: List[int] = []
        append = out.append

        # values blob stores indices into remap table; decode according to width.
        if w == 2:
            v = self._values_u16
            if v is None:
                v = self._values_blob.cast("H")
                self._values_u16 = v
            end = off + length
            for j in range(off, end):
                append(remap[v[j]])
        elif w == 4:
            v = self._values_u32
            if v is None:
                v = self._values_blob.cast("I")
                self._values_u32 = v
            end = off + length
            for j in range(off, end):
                append(remap[v[j]])
        elif w == 3:
            mv = self._values_blob
            b0 = off * 3
            for _ in range(length):
                ridx = mv[b0] | (mv[b0 + 1] << 8) | (mv[b0 + 2] << 16)
                append(remap[ridx])
                b0 += 3
        else:
            raise ValueError("Unsupported remap_index_width")

        return out


def load_or_fetch(
    url: str,
    cache_path: str,
    ram_copy: bool = True,
    mode: str = "auto",
    auto_threshold: int = _AUTO_THRESHOLD_DEFAULT,
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Ensure dataset is present on disk, then load it.

    Args:
      url: source URL (used only if cache_path does not exist)
      cache_path: local file path to store dataset.bin
      ram_copy: attempt to copy index tables into RAM (usually good)
      mode: 'auto'|'air'|'mmap'

    Returns:
      (Dataset, metadata)
      metadata: {'from_cache': bool, 'size_bytes': int|None, 'mode_chosen': 'air'|'mmap'}
    """
    metadata: Dict[str, Any] = {"from_cache": False, "size_bytes": None, "mode_chosen": None}

    # Download only if missing.
    if not os.path.exists(cache_path):
        req = urllib.request.Request(url, headers={"User-Agent": "likely_api/1.0"})
        with urllib.request.urlopen(req) as resp:
            _atomic_write_temp(cache_path, resp)
        metadata["from_cache"] = False
    else:
        metadata["from_cache"] = True

    try:
        size_bytes = os.path.getsize(cache_path)
    except Exception:
        size_bytes = None
    metadata["size_bytes"] = size_bytes

    chosen_mode = mode if mode in ("auto", "air", "mmap") else "auto"

    # Auto mode: if we have enough headroom, prefer RAM.
    if chosen_mode == "auto":
        avail, total, _src = _get_available_memory()
        safety = _compute_safety_margin(total) if total else _SAFETY_MARGIN_MIN
        dataset_need = (size_bytes or 0) + safety
        if avail and avail >= auto_threshold and avail >= dataset_need:
            chosen_mode = "air"
        else:
            chosen_mode = "mmap"

    metadata["mode_chosen"] = chosen_mode

    if chosen_mode == "air":
        with open(cache_path, "rb") as fh:
            data = bytearray(fh.read())
        parsed = _parse_dataset_file(data)
        ds = Dataset(None, None, parsed, ram_copy=ram_copy)
        return ds, metadata

    # mmap
    f = open(cache_path, "rb")
    try:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise

    parsed = _parse_dataset_file(mm)
    ds = Dataset(f, mm, parsed, ram_copy=ram_copy)
    return ds, metadata


# ---------------------------------------------------------------------
# Configuration / defaults
# ---------------------------------------------------------------------
# Where to store dataset.bin inside addon profile (uses Kodi translate_path)
_DEFAULT_CACHE_SUBDIR = "likely"
_DEFAULT_FILENAME = "dataset.bin"

# Setting keys in fenlight settings (persisted user preference)
_SETTING_KEY_MODE = "fenlight.likely.mode"  # values: 'auto' (default), 'RAM', 'mmap'
_SETTING_KEY_ENABLED = "fenlight.likely.enabled"  # reserved; not used here

# main_cache key used to ring-version our likely cache (so clear is cheap)
_CACHE_VERSION_KEY = "likely:cache_version"

# Cached packed entries expiration (in hours) — mirror other apis (use 24h)
_DEFAULT_EXPIRATION_HOURS = 24

# URL for dataset (replace with your canonical source)
DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin"


# ---------------------------------------------------------------------
# Internal module state
# ---------------------------------------------------------------------
_lock = threading.RLock()
_dataset: Optional[Dataset] = None
_dataset_meta: Dict[str, Any] = {}
_dataset_id: Optional[str] = None
_runtime_mode: Optional[str] = None  # 'RAM' or 'mmap' chosen at load time


# ---------------------------------------------------------------------
# Path helpers (Kodi profile cache)
# ---------------------------------------------------------------------
def _ensure_cache_dir() -> str:
    """Create and return the addon cache folder used to store dataset.bin."""
    # translate_path uses Kodi VFS; keep this lazy.
    folder = translate_path("special://profile/addon_data/plugin.video.fenlight/")
    data_folder = os.path.join(folder, "cache", _DEFAULT_CACHE_SUBDIR)
    try:
        os.makedirs(data_folder, exist_ok=True)
    except Exception:
        pass
    return data_folder


def _dataset_cache_path() -> str:
    return os.path.join(_ensure_cache_dir(), _DEFAULT_FILENAME)


# ---------------------------------------------------------------------
# Cache version helpers (cheap invalidate)
# ---------------------------------------------------------------------
def _get_cache_version() -> str:
    """Return a token used inside cache keys; create it if missing."""
    v = main_cache.get(_CACHE_VERSION_KEY)
    if v is None:
        v = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, v, expiration=0)  # persist until bumped
        except Exception:
            pass
    return str(v)


def clear_likely_cache() -> None:
    """Cheaply clear likely caches by bumping a version token."""
    with _lock:
        newv = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, newv, expiration=0)
        except Exception:
            try:
                main_cache.delete(_CACHE_VERSION_KEY)
            except Exception:
                pass


# ---------------------------------------------------------------------
# Dataset id helpers (used in cache keys to detect dataset changes)
# ---------------------------------------------------------------------
def _compute_dataset_id(path: str) -> str:
    """Compute lightweight id from file size and mtime (fast + deterministic)."""
    try:
        st = os.stat(path)
        return f"{int(st.st_mtime)}:{int(st.st_size)}"
    except Exception:
        return str(int(time.time()))


# ---------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------
def get_setting_mode() -> str:
    """Return persisted user setting for likely mode: 'auto'|'RAM'|'mmap'."""
    v = get_setting(_SETTING_KEY_MODE, "auto")
    v = (v or "auto").strip()
    if v.lower() == "ram":
        return "RAM"
    if v.lower() == "mmap":
        return "mmap"
    return "auto"


def set_setting_mode(mode: str) -> bool:
    """Persist user preference. Accepts 'auto', 'RAM', 'mmap' (case-insensitive)."""
    m = (mode or "auto").strip()
    if m.lower() not in ("auto", "ram", "mmap"):
        return False

    store = "RAM" if m.lower() == "ram" else ("mmap" if m.lower() == "mmap" else "auto")
    try:
        set_setting(_SETTING_KEY_MODE, store)
        return True
    except Exception:
        return False


def get_runtime_mode() -> Optional[str]:
    """Return runtime-chosen mode for current dataset: 'RAM'|'mmap'|None."""
    return _runtime_mode


# ---------------------------------------------------------------------
# Dataset lifecycle (load / reload / close)
# ---------------------------------------------------------------------
def ensure_loaded(url: Optional[str] = None, mode: Optional[str] = None, force: bool = False) -> None:
    """
    Ensure dataset is loaded and parsed. Safe to call multiple times.

    Args:
      url: optional override for dataset source
      mode: explicit run override: 'auto'|'RAM'|'mmap'
      force: reload even if already loaded

    Side-effects:
      Sets module-level _dataset, _dataset_meta, _dataset_id, _runtime_mode.
    """
    global _dataset, _dataset_meta, _dataset_id, _runtime_mode

    with _lock:
        if _dataset is not None and not force:
            return

        dataset_url = url or DEFAULT_DATASET_URL
        cache_path = _dataset_cache_path()

        # Decide mode: explicit param > persisted setting > default 'auto'
        mode_setting = get_setting_mode() if mode is None else mode

        # Map UI modes to loader modes: RAM -> 'air', mmap -> 'mmap'
        loader_mode = "auto" if mode_setting == "auto" else ("air" if mode_setting == "RAM" else "mmap")

        ds, meta = load_or_fetch(dataset_url, cache_path, ram_copy=True, mode=loader_mode)

        _dataset = ds
        _dataset_meta = meta or {}
        _dataset_id = _compute_dataset_id(cache_path)

        chosen = meta.get("mode_chosen") if isinstance(meta, dict) else None
        if chosen == "air":
            _runtime_mode = "RAM"
        elif chosen == "mmap":
            _runtime_mode = "mmap"
        else:
            _runtime_mode = "RAM" if loader_mode == "air" else "mmap"


def reload_dataset(url: Optional[str] = None, mode: Optional[str] = None) -> None:
    """Force reload of dataset and bump the cache-version so cached results won't be reused."""
    global _dataset, _dataset_meta, _dataset_id, _runtime_mode

    with _lock:
        if _dataset is not None:
            try:
                _dataset.close()
            except Exception:
                pass

        _dataset = None
        _dataset_meta = {}
        _dataset_id = None
        _runtime_mode = None

        clear_likely_cache()
        ensure_loaded(url=url, mode=mode, force=True)


def close_dataset() -> None:
    """Close dataset and release resources (mmap etc)."""
    global _dataset, _dataset_meta, _dataset_id, _runtime_mode

    with _lock:
        if _dataset is not None:
            try:
                _dataset.close()
            except Exception:
                pass
        _dataset = None
        _dataset_meta = {}
        _dataset_id = None
        _runtime_mode = None


# ---------------------------------------------------------------------
# Caching + query helpers
# ---------------------------------------------------------------------
def _packed_cache_key(tmdb_id: int, kind: str) -> str:
    """Cache key includes cache-version token + dataset identifier."""
    v = _get_cache_version()
    did = _dataset_id or "nodata"
    return f"likely:packed:{v}:{did}:{tmdb_id}:{kind}"


def query_likely_packed(
    tmdb_id: int,
    kind: str,
    use_cache: bool = True,
    expiration_hours: int = _DEFAULT_EXPIRATION_HOURS,
) -> List[int]:
    """
    Low-level fast path: return ordered list of packed ints (tmdb_id<<1 | typebit).

    Results are tiny (<= ~82 items typically) so caching them in main_cache is cheap.
    """
    if _dataset is None:
        ensure_loaded()

    key = _packed_cache_key(tmdb_id, kind)
    if use_cache:
        try:
            cached = main_cache.get(key)
            if cached is not None:
                return list(cached)
        except Exception:
            pass

    # Dataset lookup (fast, in-process)
    packed = _dataset.query_similar_packed(tmdb_id, kind)  # type: ignore[union-attr]

    try:
        main_cache.set(key, packed, expiration=expiration_hours)
    except Exception:
        pass

    return packed


def query_likely_pairs(tmdb_id: int, kind: str, timing: bool = False) -> Any:
    """
    Public compact shape:
      {"count": N, "results": [[tmdb_id, type_bit], ...]}

    If timing==True returns (result_dict, total_ms).
    """
    t0 = time.perf_counter()
    packed = query_likely_packed(tmdb_id, kind)
    res = [[(pv >> 1), (pv & 1)] for pv in packed]
    out = {"count": len(res), "results": res}
    total_ms = (time.perf_counter() - t0) * 1000.0
    return (out, total_ms) if timing else out


def get_likely_for_addon(tmdb_id: int, kind: str, timing: bool = False) -> Any:
    """
    Addon-friendly wrapper:
      {"results": [{"id": 4257, "media_type": "movie"|"tv"}, ...], "total_results": N}

    If timing==True returns (result_dict, total_ms).
    """
    t0 = time.perf_counter()
    pairs = query_likely_packed(tmdb_id, kind)
    results: List[Dict[str, Any]] = []
    for pv in pairs:
        _id = (pv >> 1)
        typebit = (pv & 1)
        results.append({"id": _id, "media_type": "tv" if typebit == 1 else "movie"})

    out = {"results": results, "total_results": len(results)}
    total_ms = (time.perf_counter() - t0) * 1000.0
    return (out, total_ms) if timing else out


# ---------------------------------------------------------------------
# Utility / debugging helpers
# ---------------------------------------------------------------------
def dataset_info() -> Dict[str, Any]:
    """Return a tiny dict describing loaded dataset (for GUI/status)."""
    return {
        "loaded": _dataset is not None,
        "dataset_id": _dataset_id,
        "runtime_mode": _runtime_mode,
        "meta": _dataset_meta or {},
    }


# ---------------------------------------------------------------------
# Example direct usage (when imported by other modules)
# ---------------------------------------------------------------------
# from apis import likely_api
# likely_api.ensure_loaded()                # load dataset according to persisted setting (auto/RAM/mmap)
# pairs = likely_api.query_likely_pairs(4257, 'movie')    # compact pairs JSON shape
# addon_ready = likely_api.get_likely_for_addon(4257, 'movie')  # wrapper for addon consumers
# ---------------------------------------------------------------------
