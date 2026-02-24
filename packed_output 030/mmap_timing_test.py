#!/usr/bin/env python3
# mmap_timing_test.py — timing-only test for on-demand mmap behavior
# Usage: python mmap_timing_test.py
# Requires: psutil (optional but recommended): pip install psutil

import os
import time
import gc
import json

try:
    import psutil
except Exception:
    psutil = None

from dataset_core import load_or_fetch, now_ms, describe_memory

URL = 'https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin'
CACHE = os.path.expanduser('~/.tmdb_similar_cache/dataset.bin')

def fmt_bytes(n):
    if n is None:
        return 'n/a'
    for unit in ('B','KB','MB','GB'):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"

def print_proc(p):
    try:
        mi = p.memory_info()
        print("Process RSS:", fmt_bytes(mi.rss), " VMS:", fmt_bytes(mi.vms))
    except Exception as e:
        print("psutil error:", e)

def main():
    print("PID:", os.getpid())
    if psutil:
        p = psutil.Process(os.getpid())
        try:
            print("Process name:", p.name())
        except Exception:
            pass
    else:
        p = None
        print("psutil not installed — install with: pip install psutil (recommended)")

    gc.collect()
    if p:
        print("Baseline:")
        print_proc(p)

    print("\nLoading dataset (may download if missing)...")
    ds, meta = load_or_fetch(URL, CACHE, use_mmap=True)
    print("Loaded. file size:", fmt_bytes(meta.get('size_bytes')))
    parsed = getattr(ds, 'parsed', None)
    if parsed:
        print("Parsed R,E,U:", parsed.get('R'), parsed.get('E'), parsed.get('U'))
    else:
        print("No parsed metadata available.")

    # small cooldown
    time.sleep(0.05)
    gc.collect()
    if p:
        print("\nAfter mmap (no queries yet):")
        print_proc(p)

    # pick a sample id from the dataset (first source key)
    if parsed and parsed.get('views') and parsed['views'].get('source_keys') and len(parsed['views']['source_keys'])>0:
        pk = int(parsed['views']['source_keys'][0])
        sample_id = pk >> 1
        sample_kind = 'tv' if (pk & 1) else 'movie'
    else:
        # fallback
        sample_id = 550
        sample_kind = 'movie'

    print("\nSample id:", sample_id, sample_kind)

    # First query (may cause page faults)
    t0 = now_ms()
    res, timings = ds.query_similar(sample_id, sample_kind)
    t1 = now_ms()
    total_ms = t1 - t0
    print("\nFirst query:")
    print("  returned", len(res), "results")
    print("  timings (from Dataset):", json.dumps(timings, indent=2))
    print("  total measured ms:", f"{total_ms:.3f}")
    if p:
        print("  RSS after first query:")
        print_proc(p)

    # small pause
    time.sleep(0.02)
    gc.collect()

    # Second query (same id)
    t0 = now_ms()
    res2, timings2 = ds.query_similar(sample_id, sample_kind)
    t1 = now_ms()
    total_ms2 = t1 - t0
    print("\nSecond query (same id):")
    print("  returned", len(res2), "results")
    print("  timings (from Dataset):", json.dumps(timings2, indent=2))
    print("  total measured ms:", f"{total_ms2:.3f}")
    if p:
        print("  RSS after second query:")
        print_proc(p)

    # Print a small memory diagnostic summary
    try:
        mem_diag = describe_memory(ds, include_process=bool(p))
        print("\nMemory diagnostic summary:")
        print(json.dumps(mem_diag, indent=2))
    except Exception as e:
        print("Could not run describe_memory():", e)

    # Done — cleanup
    ds.close()

if __name__ == '__main__':
    main()