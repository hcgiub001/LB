#!/usr/bin/env python3
# parse_only_test_fixed_close.py
# Maps the cached file, runs parse_dataset_file(mm) and reports RSS,
# then releases parsed views safely before closing the mmap.

import os, time, gc
import psutil
import mmap
from dataset_core import parse_dataset_file

CACHE = os.path.expanduser('~/.tmdb_similar_cache/dataset.bin')

def fmt_bytes(n):
    if n is None:
        return 'n/a'
    for unit in ('B','KB','MB','GB'):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"

p = psutil.Process(os.getpid())
gc.collect()
print("PID", os.getpid(), "baseline RSS", fmt_bytes(p.memory_info().rss))

if not os.path.exists(CACHE):
    raise SystemExit("Cache missing: run one download first to create the cache file.")

# open + mmap
f = open(CACHE, 'rb')
mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
time.sleep(0.05)
gc.collect()
print("After mmap RSS:", fmt_bytes(p.memory_info().rss))

# call the actual parser from dataset_core (this mirrors your real code)
parsed = parse_dataset_file(mm)
time.sleep(0.05)
gc.collect()
print("After parse_dataset_file RSS:", fmt_bytes(p.memory_info().rss))

# parsed header summary
print("Parsed header (R,E,U):", parsed.get('R'), parsed.get('E'), parsed.get('U'))
print("sizes:", parsed.get('sizes'))

# --- release the returned memoryviews before closing mmap ---
# parsed['views'] holds memoryviews that keep exported pointers active.
try:
    # delete references to the views and parsed dict then GC
    parsed_views = parsed.get('views')
    if parsed_views:
        # delete each view reference
        for k in list(parsed_views.keys()):
            parsed_views[k] = None
    parsed = None
    gc.collect()
except Exception:
    pass

# Now close mmap and file
mm.close()
f.close()
print("Closed mmap/file successfully. Final RSS:", fmt_bytes(p.memory_info().rss))