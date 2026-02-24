#!/usr/bin/env python3
"""
dataset_core.py — fast mmap/AIR reader for TMDB similar-lists packed dataset.

Key perf tweaks vs your draft:
- bisect_left for source lookup (C-accelerated)
- tight inner loops for values decoding (no per-item function calls)
- cached typed views into values_blob for width 2/4
- optional fast query that returns packed ints without dict allocations
- fixed syntax/correctness issues in the pasted version

Binary format:
header (28 bytes) +
source_keys (R * u32) +
offsets (R * u32) [optional; omitted if flags bit0 == 1] +
lengths (R * u8/u16) +
remap_table (U * u32) +
values_blob (E * remap_index_width bytes)

Header struct: <4s B B H I I I B B H I
magic, version, endian, flags, R, E, U, lengths_byte, remap_index_width, reserved, header_crc32
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
import sys
from array import array
from bisect import bisect_left
from typing import Tuple, List, Dict, Any, Union, Optional


# ---- configuration constants ----
AUTO_THRESHOLD_DEFAULT = 300 * 1024 * 1024  # 300 MB required for auto -> pick AIR
SAFETY_MARGIN_MIN = 64 * 1024 * 1024        # 64 MB minimum safety margin
SAFETY_MARGIN_FRAC = 0.05                   # 5% of total RAM as margin

PERSIST_FILENAME = "tmdb_similar_settings.json"
PERSIST_KEY = "mode"  # values: 'auto', 'air', 'mmap'


# ---- helpers ----
def packed_value(tmdb_id: int, typ: str) -> int:
    # bit0 = 1 for tv; 0 for movie
    return (int(tmdb_id) << 1) | (1 if str(typ).lower().startswith("tv") else 0)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


# ---- header ----
HEADER_STRUCT = "<4s B B H I I I B B H I"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 28


# ---- platform memory helpers ----
def _get_mem_via_psutil() -> Optional[Tuple[int, int, str]]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), "psutil"
    except Exception:
        return None


def _get_mem_via_proc_meminfo() -> Optional[Tuple[int, int, str]]:
    # Linux / Android fallback using /proc/meminfo
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
                info[key] = int(val)  # kB

        if "MemAvailable" in info:
            avail = info["MemAvailable"] * 1024
        else:
            free = info.get("MemFree", 0)
            cached = info.get("Cached", 0)
            buffers = info.get("Buffers", 0)
            avail = int((free + cached + buffers) * 1024 * 0.7)

        total = info.get("MemTotal", 0) * 1024
        return int(avail), int(total), "/proc/meminfo"
    except Exception:
        return None


def _get_mem_via_windows() -> Optional[Tuple[int, int, str]]:
    # GlobalMemoryStatusEx via ctypes
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


def get_available_memory() -> Tuple[int, int, str]:
    """
    Return (available_bytes, total_bytes, source_str).
    Best-effort cross-platform detection; psutil preferred, then platform-specific.
    On failure returns (0, 0, 'unknown') to avoid choosing AIR incorrectly.
    """
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

    # last resort
    return 0, 0, "unknown"


# ---- persistence helpers (xbmc or file fallback) ----
def _read_persisted_mode_from_file(cache_dir: str) -> Optional[str]:
    try:
        p = os.path.join(cache_dir, PERSIST_FILENAME)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        m = j.get(PERSIST_KEY)
        if m in ("auto", "air", "mmap"):
            return m
    except Exception:
        pass
    return None


def _write_persisted_mode_to_file(cache_dir: str, mode: str) -> bool:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        p = os.path.join(cache_dir, PERSIST_FILENAME)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({PERSIST_KEY: mode}, fh)
        return True
    except Exception:
        return False


def get_persisted_mode(cache_dir: str) -> Optional[str]:
    # Try Kodi xbmcaddon if available
    try:
        import xbmcaddon  # type: ignore
        addon = xbmcaddon.Addon()
        val = addon.getSetting(PERSIST_KEY)
        if val in ("auto", "air", "mmap"):
            return val
    except Exception:
        pass

    return _read_persisted_mode_from_file(cache_dir)


def set_persisted_mode(cache_dir: str, mode: str) -> bool:
    if mode not in ("auto", "air", "mmap"):
        return False
    try:
        import xbmcaddon  # type: ignore
        addon = xbmcaddon.Addon()
        addon.setSetting(PERSIST_KEY, mode)
        return True
    except Exception:
        return _write_persisted_mode_to_file(cache_dir, mode)


# ---- Dataset ----
class Dataset:
    """
    Fast reader.
    Views are zero-copy memoryviews into mmap or AIR buffer.
    Optional ram_copy converts some views to array('I') / array('B') / array('H') for speed.
    """

    __slots__ = (
        "_fileobj",
        "_mmap",
        "parsed",
        "sizes",
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

    def __init__(
        self,
        fileobj,
        mm: Optional[mmap.mmap],
        parsed: Dict[str, Any],
        ram_copy: bool = True,
    ):
        self._fileobj = fileobj
        self._mmap = mm
        self.parsed = parsed
        self.sizes = parsed.get("sizes", {})
        self.remap_index_width = int(parsed["remap_index_width"])
        self.lengths_type = int(parsed["lengths_type"])

        views = parsed["views"]
        self._source_keys = views["source_keys"]   # memoryview('I') or array('I')
        self._offsets = views["offsets"]           # memoryview('I') or array('I')
        self._lengths = views["lengths"]           # memoryview('B') or ('H') or array
        self._remap = views["remap_table"]         # memoryview('I') or array('I')
        self._values_blob = views["values_blob"]   # memoryview('B')

        self._values_u16 = None
        self._values_u32 = None
        self._ram_copy = False

        if ram_copy:
            self._attempt_ram_copy()

        # Pre-cache typed view for fastest common case (width=2)
        # (No copy; just a cast)
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
        try:
            # array('I', memoryview('I')) is fast and yields quick indexing
            self._source_keys = array("I", self._source_keys)
            self._remap = array("I", self._remap)

            # offsets might already be array (if reconstructed); copy anyway for uniformity
            self._offsets = array("I", self._offsets)

            if self.lengths_type == 0:
                self._lengths = array("B", self._lengths)
            else:
                self._lengths = array("H", self._lengths)

            self._ram_copy = True
        except Exception:
            # keep views as-is
            self._ram_copy = False

    def close(self) -> None:
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

    # ---- queries ----
    def _find_row_index(self, packed_src_key: int) -> int:
        # bisect_left is in C; faster than Python while-loop binary search.
        sk = self._source_keys
        i = bisect_left(sk, packed_src_key)
        if i != len(sk) and sk[i] == packed_src_key:
            return i
        return -1

    def query_similar_packed(self, tmdb_id: int, kind: str) -> Tuple[List[int], Dict[str, float]]:
        """
        Fastest query: returns list of packed values (tmdb_id<<1 | typebit).
        Avoids per-result dict allocations and string work.
        """
        t0 = now_ms()
        packed_src = packed_value(int(tmdb_id), kind)

        t_search0 = now_ms()
        idx = self._find_row_index(packed_src)
        t_search1 = now_ms()

        if idx < 0:
            return [], {
                "search_ms": t_search1 - t_search0,
                "query_ms": 0.0,
                "total_ms": now_ms() - t0,
            }

        off = self._offsets[idx]
        length = self._lengths[idx]

        remap = self._remap

        w = self.remap_index_width
        t_q0 = now_ms()

        out: List[int] = []
        out_extend = out.append

        if w == 2:
            v = self._values_u16
            if v is None:
                v = self._values_blob.cast("H")
                self._values_u16 = v
            end = off + length
            for j in range(off, end):
                out_extend(remap[v[j]])
        elif w == 4:
            v = self._values_u32
            if v is None:
                v = self._values_blob.cast("I")
                self._values_u32 = v
            end = off + length
            for j in range(off, end):
                out_extend(remap[v[j]])
        elif w == 3:
            mv = self._values_blob
            b0 = off * 3
            # loop in bytes; avoid multiplications inside too much
            for _ in range(length):
                ridx = mv[b0] | (mv[b0 + 1] << 8) | (mv[b0 + 2] << 16)
                out_extend(remap[ridx])
                b0 += 3
        else:
            raise ValueError("Unsupported remap_index_width")

        t_q1 = now_ms()
        return out, {
            "search_ms": t_search1 - t_search0,
            "query_ms": t_q1 - t_q0,
            "total_ms": now_ms() - t0,
        }

    def query_similar(self, tmdb_id: int, kind: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Backwards-compatible query: returns list of dicts:
          {'tmdb_id': int, 'type': 'movie'|'tvshow', 'packed': int}
        """
        packed_list, timings = self.query_similar_packed(tmdb_id, kind)
        if not packed_list:
            return [], timings

        # decode (this is where most remaining time goes: dict allocations)
        MOVIE = "movie"
        TVSHOW = "tvshow"
        results: List[Dict[str, Any]] = []
        append = results.append
        for pv in packed_list:
            append({
                "tmdb_id": (pv >> 1),
                "type": (TVSHOW if (pv & 1) else MOVIE),
                "packed": pv,
            })
        return results, timings


# ---- parse ----
def parse_dataset_file(buf) -> Dict[str, Any]:
    """
    Parse memory buffer (mmap.mmap OR bytes/bytearray/memoryview).
    Returns dict with zero-copy views into that buffer.

    Note: memoryview.cast('I'/'H') uses native endianness.
    This format is little-endian; we reject big-endian platforms.
    """
    if sys.byteorder != "little":
        raise RuntimeError("Big-endian platform not supported by this zero-copy parser.")

    # determine size
    try:
        size = buf.size()  # mmap
    except Exception:
        size = len(buf)

    if size < HEADER_SIZE:
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
    ) = struct.unpack_from(HEADER_STRUCT, full_mv, 0)

    if magic != b"SIML":
        raise ValueError("Bad magic (not SIML)")

    # endian field is informational; we only support little-endian encoding here
    # (the struct unpack is already little-endian due to HEADER_STRUCT)
    pos = HEADER_SIZE

    sizes: Dict[str, int] = {"header": HEADER_SIZE}

    # source_keys
    source_keys_bytes = R * 4
    source_keys_off = pos
    pos += source_keys_bytes
    sizes["source_keys"] = source_keys_bytes

    offsets_present = not bool(flags & 1)
    offsets_bytes = R * 4 if offsets_present else 0
    offsets_off = pos if offsets_present else None
    pos += offsets_bytes
    sizes["offsets"] = offsets_bytes

    # lengths
    if lengths_byte == 0:
        lengths_type = 0
        lengths_bytes = R
    else:
        lengths_type = 1
        lengths_bytes = R * 2

    lengths_off = pos
    pos += lengths_bytes
    sizes["lengths"] = lengths_bytes

    # remap_table
    remap_bytes = U * 4
    remap_off = pos
    pos += remap_bytes
    sizes["remap_table"] = remap_bytes

    # values_blob
    values_off = pos
    values_bytes = size - pos
    sizes["values_blob"] = values_bytes

    # --- create views ---
    source_keys_mv = full_mv[source_keys_off:source_keys_off + source_keys_bytes].cast("I")

    lengths_raw_mv = full_mv[lengths_off:lengths_off + lengths_bytes]
    if lengths_type == 0:
        lengths_mv = lengths_raw_mv.cast("B")
    else:
        lengths_mv = lengths_raw_mv.cast("H")

    if offsets_present:
        offsets_mv = full_mv[offsets_off:offsets_off + offsets_bytes].cast("I")  # type: ignore[index]
    else:
        # reconstruct offsets from lengths (one-time cost)
        # store in array('I') so indexing is fast
        offsets_arr = array("I")
        cur = 0
        offsets_arr_append = offsets_arr.append
        for l in lengths_mv:
            offsets_arr_append(cur)
            cur += int(l)
        offsets_mv = offsets_arr

    remap_mv = full_mv[remap_off:remap_off + remap_bytes].cast("I")
    values_mv = full_mv[values_off:values_off + values_bytes]  # memoryview('B')

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
        "sizes": sizes,
        "views": {
            "source_keys": source_keys_mv,
            "offsets": offsets_mv,
            "lengths": lengths_mv,
            "remap_table": remap_mv,
            "values_blob": values_mv,
        },
    }


# ---- atomic download + loader (mmap | air | auto) ----
def _atomic_write_temp(target_path: str, data_stream) -> None:
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


def _compute_safety_margin(total_bytes: int) -> int:
    return max(SAFETY_MARGIN_MIN, int(total_bytes * SAFETY_MARGIN_FRAC))


def load_or_fetch(
    url: str,
    cache_path: str,
    use_mmap: bool = True,          # kept for compatibility; mode controls behavior
    ram_copy: bool = True,
    mode: str = "auto",
    auto_threshold: int = AUTO_THRESHOLD_DEFAULT,
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    mode:
      - 'auto': honor persisted explicit mode if set; else choose AIR for this run if safe
      - 'air' : read entire file into RAM (bytearray)
      - 'mmap': memory-map file

    Returns (Dataset, metadata)
    """
    metadata: Dict[str, Any] = {
        "from_cache": False,
        "size_bytes": None,
        "path": cache_path,
        "load_ms": None,
        "mode_setting": mode,
        "mode_persisted": None,
        "mode_chosen": None,
        "mem_available_bytes": None,
        "mem_total_bytes": None,
        "mem_source": None,
        "air_active": False,
        "air_failed": False,
        "air_failure_reason": None,
        "air_bytes_copied": 0,
        "air_copy_ms": None,
    }

    t_start = time.perf_counter()

    # ensure file exists
    if not os.path.exists(cache_path):
        req = urllib.request.Request(url, headers={"User-Agent": "dataset_core/1.0"})
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

    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    persisted = get_persisted_mode(cache_dir)
    metadata["mode_persisted"] = persisted

    chosen_mode = mode if mode in ("auto", "air", "mmap") else "auto"

    # If auto and persisted explicit exists, honor it
    if chosen_mode == "auto" and persisted in ("air", "mmap"):
        chosen_mode = persisted

    # Still auto: inspect memory
    if chosen_mode == "auto":
        avail, total, src = get_available_memory()
        metadata["mem_available_bytes"] = avail
        metadata["mem_total_bytes"] = total
        metadata["mem_source"] = src

        safety = _compute_safety_margin(total) if total else SAFETY_MARGIN_MIN
        dataset_need = (size_bytes or 0) + safety

        if avail and avail >= auto_threshold and avail >= dataset_need:
            chosen_mode = "air"
        else:
            chosen_mode = "mmap"

    metadata["mode_chosen"] = chosen_mode

    # --- AIR ---
    if chosen_mode == "air":
        t0 = time.perf_counter()
        try:
            with open(cache_path, "rb") as fh:
                data = bytearray(fh.read())
            metadata["air_active"] = True
            metadata["air_bytes_copied"] = len(data)
            metadata["air_copy_ms"] = (time.perf_counter() - t0) * 1000.0

            parsed = parse_dataset_file(data)
            ds = Dataset(None, None, parsed, ram_copy=ram_copy)
            ds.sizes = parsed.get("sizes", {})

            metadata["load_ms"] = (time.perf_counter() - t_start) * 1000.0
            return ds, metadata

        except MemoryError:
            metadata["air_failed"] = True
            metadata["air_failure_reason"] = "MemoryError"
        except Exception as e:
            metadata["air_failed"] = True
            metadata["air_failure_reason"] = f"Exception: {e}"

        # fall through to mmap

    # --- MMAP ---
    f = open(cache_path, "rb")
    try:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise

    parsed = parse_dataset_file(mm)
    ds = Dataset(f, mm, parsed, ram_copy=ram_copy)
    ds.sizes = parsed.get("sizes", {})

    metadata["load_ms"] = (time.perf_counter() - t_start) * 1000.0
    return ds, metadata


# ---- diagnostics ----
def describe_memory(ds: Dataset, include_process: bool = False) -> Dict[str, Any]:
    p = ds.parsed
    sizes = p.get("sizes", {}).copy()
    sizes_summary = {
        "file_total_bytes": None,
        "header_bytes": sizes.get("header"),
        "source_keys_bytes": sizes.get("source_keys"),
        "offsets_bytes": sizes.get("offsets"),
        "lengths_bytes": sizes.get("lengths"),
        "remap_table_bytes": sizes.get("remap_table"),
        "values_blob_bytes": sizes.get("values_blob"),
        "remap_index_width": ds.remap_index_width,
        "lengths_type": ("uint8" if ds.lengths_type == 0 else "uint16"),
        "rows_R": p.get("R"),
        "unique_U": p.get("U"),
        "entries_E": p.get("E"),
        "offsets_present": p.get("offsets_present"),
        "ram_copy_active": getattr(ds, "_ram_copy", False),
    }
    try:
        if getattr(ds, "_mmap", None) is not None:
            sizes_summary["file_total_bytes"] = ds._mmap.size()  # type: ignore[union-attr]
    except Exception:
        pass

    out: Dict[str, Any] = {"parsed_sizes": sizes_summary}
    if not include_process:
        return out

    try:
        import psutil  # type: ignore
        proc = psutil.Process()
        mi = proc.memory_info()
        out["process_memory"] = {"rss": getattr(mi, "rss", None), "vms": getattr(mi, "vms", None)}
    except Exception as e:
        out["process_memory"] = {"error": f"psutil not available or failed: {e}"}

    return out


# ---- microbenchmark ----
def benchmark_cold_vs_warm(ds: Dataset, tmdb_id: int, kind: str = "movie", n_warm: int = 100) -> Dict[str, Any]:
    import statistics

    cold_t0 = time.perf_counter()
    ds.query_similar(tmdb_id, kind)
    cold_ms = (time.perf_counter() - cold_t0) * 1000.0

    warm_times = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        ds.query_similar(tmdb_id, kind)
        warm_times.append((time.perf_counter() - t0) * 1000.0)

    wt = sorted(warm_times)
    p95 = wt[int(0.95 * (len(wt) - 1))] if wt else 0.0

    return {
        "first_ms": round(cold_ms, 4),
        "warm_mean_ms": round(statistics.mean(warm_times), 4) if warm_times else 0.0,
        "warm_p50_ms": round(statistics.median(warm_times), 4) if warm_times else 0.0,
        "warm_p95_ms": round(p95, 4),
        "warm_min_ms": round(min(warm_times), 4) if warm_times else 0.0,
        "ram_copy_active": getattr(ds, "_ram_copy", False),
    }


# ---- CLI ----
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dataset_core.py <path-to-dataset.bin> [tmdb_id] [movie|tv]")
        sys.exit(1)

    path = sys.argv[1]
    tmdb_id = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    kind = sys.argv[3] if len(sys.argv) >= 4 else "movie"

    url = "file://" + os.path.abspath(path)
    ds, meta = load_or_fetch(url, path, ram_copy=True, mode="mmap")
    print("Load meta:", meta)
    print("Memory:", describe_memory(ds, include_process=False))

    if tmdb_id is not None:
        res, timings = ds.query_similar(tmdb_id, kind)
        print("timings:", timings, "count:", len(res))
        # show first few
        print(res[:10])

    ds.close()