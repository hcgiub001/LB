#!/usr/bin/env python3
"""
dataset_core.py — compact, fast reader that returns ordered [[tmdb_id,type_bit], ...] pairs.

Public API:
  - load_or_fetch(url, cache_path, mode="auto", ram_copy=True) -> (Dataset, meta)
  - Dataset.query_similar_pairs(tmdb_id, kind) -> {"count": N, "results": [[id, type_bit], ...]}

Notes:
 - type_bit: 0 == movie, 1 == tv
 - result ordering is preserved
 - no debug/diagnostic blobs are returned by default
"""
from __future__ import annotations

import os
import struct
import mmap
import urllib.request
import shutil
import tempfile
import time
import platform
import ctypes
import sys
from array import array
from bisect import bisect_left
from typing import Any, Dict, List, Optional, Tuple

# ---- header ----
HEADER_STRUCT = "<4s B B H I I I B B H I"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 28

# ---- memory heuristics (small, robust) ----
AUTO_THRESHOLD_DEFAULT = 300 * 1024 * 1024  # 300MB minimum to prefer AIR
SAFETY_MARGIN_MIN = 64 * 1024 * 1024
SAFETY_MARGIN_FRAC = 0.05


def _get_mem_via_psutil() -> Optional[Tuple[int, int, str]]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), "psutil"
    except Exception:
        return None


def _get_mem_via_proc_meminfo() -> Optional[Tuple[int, int, str]]:
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
            free = info.get("MemFree", 0)
            cached = info.get("Cached", 0)
            buffers = info.get("Buffers", 0)
            avail = int((free + cached + buffers) * 1024 * 0.7)
        total = info.get("MemTotal", 0) * 1024
        return int(avail), int(total), "/proc/meminfo"
    except Exception:
        return None


def _get_mem_via_windows() -> Optional[Tuple[int, int, str]]:
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


# ---- utilities ----
def packed_value(tmdb_id: int, kind: str) -> int:
    """Pack (tmdb_id, kind) into integer: (tmdb_id << 1) | type_bit"""
    return (int(tmdb_id) << 1) | (1 if str(kind).lower().startswith("tv") else 0)


def _compute_safety_margin(total_bytes: int) -> int:
    return max(SAFETY_MARGIN_MIN, int(total_bytes * SAFETY_MARGIN_FRAC))


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


# ---- parse ----
def parse_dataset_file(buf) -> Dict[str, Any]:
    """
    Parse memory buffer (mmap.mmap OR bytes/bytearray/memoryview).
    Returns dict with zero-copy views into that buffer.
    """
    if sys.byteorder != "little":
        raise RuntimeError("Big-endian platform not supported.")

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

    # create views (zero-copy where possible)
    source_keys_mv = full_mv[source_keys_off:source_keys_off + source_keys_bytes].cast("I")

    lengths_raw_mv = full_mv[lengths_off:lengths_off + lengths_bytes]
    lengths_mv = lengths_raw_mv.cast("B") if lengths_type == 0 else lengths_raw_mv.cast("H")

    if offsets_present:
        offsets_mv = full_mv[offsets_off:offsets_off + offsets_bytes].cast("I")  # type: ignore[index]
    else:
        # reconstruct offsets
        offsets_arr = array("I")
        cur = 0
        app = offsets_arr.append
        for l in lengths_mv:
            app(cur)
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


# ---- Dataset class ----
class Dataset:
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

    def __init__(self, fileobj, mm: Optional[mmap.mmap], parsed: Dict[str, Any], ram_copy: bool = True):
        self._fileobj = fileobj
        self._mmap = mm
        self.parsed = parsed
        self.sizes = parsed.get("sizes", {})
        self.remap_index_width = int(parsed["remap_index_width"])
        self.lengths_type = int(parsed["lengths_type"])

        views = parsed["views"]
        self._source_keys = views["source_keys"]
        self._offsets = views["offsets"]
        self._lengths = views["lengths"]
        self._remap = views["remap_table"]
        self._values_blob = views["values_blob"]

        self._values_u16 = None
        self._values_u32 = None
        self._ram_copy = False

        if ram_copy:
            self._attempt_ram_copy()

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
        sk = self._source_keys
        i = bisect_left(sk, packed_src_key)
        if i != len(sk) and sk[i] == packed_src_key:
            return i
        return -1

    def query_similar_packed(self, tmdb_id: int, kind: str) -> List[int]:
        """
        Very fast: return ordered list of packed ints (tmdb_id<<1 | typebit).
        """
        packed_src = packed_value(int(tmdb_id), kind)
        idx = self._find_row_index(packed_src)
        if idx < 0:
            return []

        off = int(self._offsets[idx])
        length = int(self._lengths[idx])

        remap = self._remap
        w = self.remap_index_width

        out: List[int] = []
        append = out.append

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

    def query_similar_pairs(self, tmdb_id: int, kind: str) -> Dict[str, Any]:
        """
        Public compact result format:
          {"count": N, "results": [[tmdb_id,int_typebit], ...]}
        Preserves ordering. typebit: 0=movie,1=tv
        """
        packed = self.query_similar_packed(tmdb_id, kind)
        if not packed:
            return {"count": 0, "results": []}

        # produce list of small two-element lists (pairs)
        # Use local variables for speed
        res = [[(pv >> 1), (pv & 1)] for pv in packed]
        return {"count": len(res), "results": res}


# ---- loader (atomic download + choice of AIR or MMAP) ----
def load_or_fetch(
    url: str,
    cache_path: str,
    ram_copy: bool = True,
    mode: str = "auto",
    auto_threshold: int = AUTO_THRESHOLD_DEFAULT,
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Load dataset and return (Dataset, metadata).
    metadata contains minimal fields: {'from_cache': bool, 'size_bytes': int or None, 'mode_chosen': 'air'|'mmap'}
    """
    metadata: Dict[str, Any] = {"from_cache": False, "size_bytes": None, "mode_chosen": None}

    # fetch if not present
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

    chosen_mode = mode if mode in ("auto", "air", "mmap") else "auto"

    if chosen_mode == "auto":
        avail, total, _src = get_available_memory()
        safety = _compute_safety_margin(total) if total else SAFETY_MARGIN_MIN
        dataset_need = (size_bytes or 0) + safety
        if avail and avail >= auto_threshold and avail >= dataset_need:
            chosen_mode = "air"
        else:
            chosen_mode = "mmap"

    metadata["mode_chosen"] = chosen_mode

    if chosen_mode == "air":
        with open(cache_path, "rb") as fh:
            data = bytearray(fh.read())
        parsed = parse_dataset_file(data)
        ds = Dataset(None, None, parsed, ram_copy=ram_copy)
        ds.sizes = parsed.get("sizes", {})
        return ds, metadata

    # mmap
    f = open(cache_path, "rb")
    try:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
    except Exception:
        f.close()
        raise
    parsed = parse_dataset_file(mm)
    ds = Dataset(f, mm, parsed, ram_copy=ram_copy)
    ds.sizes = parsed.get("sizes", {})
    return ds, metadata


# ---- minimal CLI for testing ----
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset_core.py <path-or-url-to-dataset.bin> [tmdb_id] [movie|tv]")
        sys.exit(1)

    path = sys.argv[1]
    tmdb_id = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    kind = sys.argv[3] if len(sys.argv) >= 4 else "movie"

    # Accept file:// or local path
    if path.startswith("file://"):
        cache_path = path[7:]
        url = path
    elif os.path.exists(path):
        cache_path = path
        url = "file://" + os.path.abspath(path)
    else:
        # treat as URL to download into temp file
        temp_dir = tempfile.gettempdir()
        cache_path = os.path.join(temp_dir, "dataset.bin")
        url = path

    ds, meta = load_or_fetch(url, cache_path, ram_copy=True, mode="mmap")
    if tmdb_id is not None:
        out = ds.query_similar_pairs(tmdb_id, kind)
        # compact JSON for addons: no spaces
        print(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    else:
        print("Loaded dataset. Call with a TMDB id to print compact result JSON.")
        print("meta:", meta)