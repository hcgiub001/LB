#!/usr/bin/env python3
# mmap_only_test.py
import os, time, gc
import psutil
import mmap

CACHE = os.path.expanduser('~/.tmdb_similar_cache/dataset.bin')

def fmt(n):
    for unit in ('B','KB','MB','GB'):
        if n < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}TB"

p = psutil.Process(os.getpid())
gc.collect()
print("PID", os.getpid(), "baseline RSS", fmt(p.memory_info().rss))

# ensure file exists (if not, run mmap_timing_test to download once)
if not os.path.exists(CACHE):
    raise SystemExit("Cache missing: run one download first (or use your GUI).")

f = open(CACHE, 'rb')
mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
time.sleep(0.05)
gc.collect()
print("After mmap (no parsing) RSS:", fmt(p.memory_info().rss))

# cleanup
mm.close()
f.close()