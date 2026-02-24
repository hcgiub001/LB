# -*- coding: utf-8 -*-
"""
likely_api.py — fenlight-friendly adapter for the packed "likely/similar" dataset.

Public surface (recommended):
  - ensure_loaded(force=False) -> None
  - get_setting_mode() -> str              # returns user's persisted choice: 'auto'|'RAM'|'mmap'
  - set_setting_mode(mode: str) -> bool    # persist user's explicit choice ('auto','RAM','mmap')
  - get_runtime_mode() -> str              # returns what was actually chosen for this run: 'RAM'|'mmap'
  - query_likely_packed(tmdb_id, kind) -> List[int]                # packed ints (id<<1 | typebit)
  - query_likely_pairs(tmdb_id, kind, timing=False) -> dict       # {"count":N,"results":[[id,type],...]}
  - get_likely_for_addon(tmdb_id, kind, timing=False) -> dict     # {"results":[{"id":..,"media_type":..}], "total_results":N}
  - clear_likely_cache() -> None
  - reload_dataset() -> None

Design notes:
 - Uses fenlight caches (main_cache) and settings (get_setting / set_setting).
 - Caches packed-int results (small) keyed by dataset id + cache version.
 - Does not add CLI/pipe surface — this API is intended to be used in-process by the addon.
 - Default behaviour is minimal and matches fenlight patterns.
"""
from __future__ import annotations

import os
import time
import threading
import json
from typing import Any, Dict, List, Optional, Tuple

# fenlight helpers / caches (expected to exist inside the addon)
from caches.main_cache import main_cache
from caches.settings_cache import get_setting, set_setting
from modules.kodi_utils import translate_path, kodi_dialog, kodi_log  # kodi_log may be present as logger wrapper

# dataset loader (must be added to the same lib or importable)
# dataset_core must expose load_or_fetch(...) as in earlier drafts
try:
    from dataset_core import load_or_fetch  # type: ignore
except Exception:
    load_or_fetch = None  # will raise if used without dataset_core present

# ---------------------------------------------------------------------
# Configuration / defaults
# ---------------------------------------------------------------------
# Where to store dataset.bin inside addon profile (uses Kodi translate_path)
_DEFAULT_CACHE_SUBDIR = 'likely'
_DEFAULT_FILENAME = 'dataset.bin'

# Setting keys in fenlight settings (persisted user preference)
_SETTING_KEY_MODE = 'fenlight.likely.mode'   # values: 'auto' (default), 'RAM', 'mmap'
_SETTING_KEY_ENABLED = 'fenlight.likely.enabled'  # optional boolean setting (not used here but reserved)

# main_cache key used to ring-version our likely cache (so clear is cheap)
_CACHE_VERSION_KEY = 'likely:cache_version'

# Cached packed entries expiration (in hours) — mirror other apis (use 24h)
_DEFAULT_EXPIRATION_HOURS = 24

# URL for dataset if you want default (replace with your canonical source)
DEFAULT_DATASET_URL = 'https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin'

# ---------------------------------------------------------------------
# Internal module state
# ---------------------------------------------------------------------
_lock = threading.RLock()
_dataset = None            # Dataset instance returned by dataset_core.load_or_fetch
_dataset_meta: Dict[str, Any] = {}
_dataset_id: Optional[str] = None
_runtime_mode: Optional[str] = None  # 'RAM' or 'mmap' chosen at load time

# Helpers to build the local cache path for dataset.bin (lazy)
def _ensure_cache_dir() -> str:
    # keep path resolution lazy because translate_path uses Kodi vfs
    folder = translate_path('special://profile/addon_data/plugin.video.fenlight/')
    data_folder = os.path.join(folder, 'cache', _DEFAULT_CACHE_SUBDIR)
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
    """Return a string token used as part of cache keys; create if missing."""
    v = main_cache.get(_CACHE_VERSION_KEY)
    if v is None:
        v = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, v, expiration=0)  # persistent until bumped
        except Exception:
            pass
    return str(v)

def clear_likely_cache() -> None:
    """
    Cheaply clear likely caches by bumping our cache-version token.
    This avoids scanning DB keys and matches fenlight cache patterns.
    """
    with _lock:
        newv = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, newv, expiration=0)
        except Exception:
            # last-resort: try to delete (not ideal) — avoid wiping other caches
            try:
                main_cache.delete(_CACHE_VERSION_KEY)
            except Exception:
                pass

# ---------------------------------------------------------------------
# Dataset id helpers (used in cache key to detect dataset changes)
# ---------------------------------------------------------------------
def _compute_dataset_id(path: str) -> str:
    """
    Compute a lightweight dataset identifier used in cache keys.
    Uses file size and mtime (fast), which changes when the dataset file is replaced.
    """
    try:
        st = os.stat(path)
        return f"{int(st.st_mtime)}:{int(st.st_size)}"
    except Exception:
        # fallback: timestamp now (safe but will invalidate)
        return str(int(time.time()))

# ---------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------
def get_setting_mode() -> str:
    """
    Return persisted user setting for likely mode.
    Values: 'auto' (default), 'RAM', 'mmap'
    """
    v = get_setting(_SETTING_KEY_MODE, 'auto')
    # backward-compat: allow lowercase, normalize
    v = (v or 'auto').strip()
    if v.lower() == 'ram':
        return 'RAM'
    if v.lower() == 'mmap':
        return 'mmap'
    return 'auto'

def set_setting_mode(mode: str) -> bool:
    """
    Persist user preference. Accepts 'auto', 'RAM', or 'mmap' (case-insensitive).
    Returns True on success.
    """
    m = (mode or 'auto').strip()
    if m.lower() not in ('auto', 'ram', 'mmap'):
        return False
    # store canonical casing
    store = 'RAM' if m.lower() == 'ram' else ('mmap' if m.lower() == 'mmap' else 'auto')
    try:
        set_setting(_SETTING_KEY_MODE, store)
        return True
    except Exception:
        return False

def get_runtime_mode() -> Optional[str]:
    """Return runtime-chosen mode for the currently loaded dataset: 'RAM' or 'mmap' or None."""
    return _runtime_mode

# ---------------------------------------------------------------------
# Dataset lifecycle (load / reload / close)
# ---------------------------------------------------------------------
def ensure_loaded(url: Optional[str] = None, mode: Optional[str] = None, force: bool = False) -> None:
    """
    Ensure dataset is loaded and parsed. This is safe to call multiple times.
      - url: optional override for dataset source (if none, DEFAULT_DATASET_URL is used or cached file path)
      - mode: explicit mode override for this run ('auto'|'RAM'|'mmap'); if None will read persisted setting
      - force: if True, reload even if already loaded
    After a successful load, module-level _dataset and _dataset_meta and _dataset_id are set.
    """
    global _dataset, _dataset_meta, _dataset_id, _runtime_mode

    if load_or_fetch is None:
        raise RuntimeError("dataset_core.load_or_fetch not available; ensure dataset_core.py is in the addon lib/ path.")

    with _lock:
        if _dataset is not None and not force:
            return  # already loaded

        # choose dataset URL and cache path
        dataset_url = url or DEFAULT_DATASET_URL
        cache_path = _dataset_cache_path()

        # decide mode: explicit param > persisted setting > default 'auto'
        if mode is None:
            mode_setting = get_setting_mode()
        else:
            mode_setting = mode

        # map UI 'RAM' to underlying loader 'air'
        loader_mode = 'auto' if mode_setting == 'auto' else ('air' if mode_setting == 'RAM' else 'mmap')

        # call load_or_fetch which handles AUTO decision if requested
        ds, meta = load_or_fetch(dataset_url, cache_path, ram_copy=True, mode=loader_mode)

        # store dataset and tiny metadata
        _dataset = ds
        _dataset_meta = meta or {}
        # dataset id: prefer file-based id (mtime:size) so cache keys are deterministic
        _dataset_id = _compute_dataset_id(cache_path)
        # runtime mode: meta['mode_chosen'] maps to 'air' or 'mmap' — we expose 'RAM' when 'air'
        chosen = meta.get('mode_chosen') if isinstance(meta, dict) else None
        if chosen == 'air':
            _runtime_mode = 'RAM'
        elif chosen == 'mmap':
            _runtime_mode = 'mmap'
        else:
            # fallback to our chosen loader_mode mapping
            _runtime_mode = 'RAM' if loader_mode == 'air' else 'mmap'

def reload_dataset(url: Optional[str] = None, mode: Optional[str] = None) -> None:
    """
    Force a reload of dataset (close old, clear in-memory handle, load anew).
    Also bumps the likely cache-version token so stale per-id entries won't be used.
    """
    global _dataset, _dataset_meta, _dataset_id, _runtime_mode
    with _lock:
        # close/cleanup old dataset
        if _dataset is not None:
            try:
                _dataset.close()
            except Exception:
                pass
            _dataset = None
            _dataset_meta = {}
            _dataset_id = None
            _runtime_mode = None

        # bump cache version to clear per-id cached results
        clear_likely_cache()

        # re-load
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
    """
    Internal canonical cache key for packed ints.
    Includes cache-version token and dataset identifier to ensure safe invalidation.
    """
    v = _get_cache_version()
    did = _dataset_id or 'nodata'
    return f"likely:packed:{v}:{did}:{tmdb_id}:{kind}"

def query_likely_packed(tmdb_id: int, kind: str, use_cache: bool = True, expiration_hours: int = _DEFAULT_EXPIRATION_HOURS) -> List[int]:
    """
    Low-level fast path: return ordered list of packed ints (tmdb_id<<1 | typebit).
    - use_cache: if True, check main_cache for a cached value first.
    - expiration_hours: how long to store results when caching (hours).
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

    # perform lookup (Dataset.query_similar_packed returns List[int])
    try:
        packed = _dataset.query_similar_packed(tmdb_id, kind)
    except Exception as exc:
        # bubble up error or return empty
        raise

    # store in main_cache (small list)
    try:
        main_cache.set(key, packed, expiration=expiration_hours)
    except Exception:
        pass

    return packed

def query_likely_pairs(tmdb_id: int, kind: str, timing: bool = False) -> Any:
    """
    Public compact shape: returns {"count": N, "results": [[tmdb_id, type_bit], ...]}
    If timing==True returns tuple (result_dict, total_ms).
    """
    t0 = time.perf_counter()
    packed = query_likely_packed(tmdb_id, kind)
    # convert to pairs preserving ordering
    res = [[(pv >> 1), (pv & 1)] for pv in packed]
    out = {"count": len(res), "results": res}
    total_ms = (time.perf_counter() - t0) * 1000.0
    if timing:
        return out, total_ms
    return out

def get_likely_for_addon(tmdb_id: int, kind: str, timing: bool = False) -> Any:
    """
    Addon-friendly wrapper:
      {"results":[{"id": 4257, "media_type": "movie" | "tv"}, ...], "total_results": N}

    This is a tiny conversion layer over the packed ints; conversion cost is negligible for <=82 items.
    If timing==True returns tuple (result_dict, total_ms).
    """
    t0 = time.perf_counter()
    pairs = query_likely_packed(tmdb_id, kind)
    results = []
    for pv in pairs:
        _id = (pv >> 1)
        typebit = (pv & 1)
        results.append({
            "id": _id,
            "media_type": "tv" if typebit == 1 else "movie"
        })
    out = {"results": results, "total_results": len(results)}
    total_ms = (time.perf_counter() - t0) * 1000.0
    if timing:
        return out, total_ms
    return out

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