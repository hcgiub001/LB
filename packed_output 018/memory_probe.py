#!/usr/bin/env python3
# memory_probe.py — updated for zero-RAM mmap reader
import os
import time
import gc
try:
    import psutil
except Exception:
    psutil = None

from dataset_core import load_or_fetch, describe_memory

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

print("PID:", os.getpid())
if psutil:
    p = psutil.Process(os.getpid())
    try:
        print("psutil says process.exe:", p.name())
    except Exception:
        print("psutil present but failed to query process name.")
else:
    print("psutil not installed. Install with: pip install psutil")

print("\nLoading dataset (will stream download if not cached) ...")
try:
    ds, meta = load_or_fetch(URL, CACHE, use_mmap=True)
except Exception as e:
    print("Failed to load dataset:", e)
    raise SystemExit(1)

print("Loaded. meta timings:", meta.get('timings', {}))
print("File size on disk:", fmt_bytes(meta.get('size_bytes')))

# show parsed dataset summary (keeps mmap open)
try:
    print("Dataset summary: R=%d, E=%d, U=%d, remap_width=%d" % (ds.R, ds.E, ds.U, ds.remap_width))
    print("Reported sizes (ds.sizes):", ds.sizes)
except Exception:
    print("Could not read dataset summary fields (unexpected).")

# short pause to allow any background activity (if any) to start
time.sleep(0.15)

# force GC and sample memory
gc.collect()
print("\n--- Immediate memory snapshot ---")
if psutil:
    try:
        mi = p.memory_info()
        print("psutil memory_info() -> rss:", getattr(mi, 'rss', None),
              " vms:", getattr(mi, 'vms', None))
    except Exception as e:
        print("psutil memory_info() failed:", e)
else:
    print("psutil missing - cannot print process RSS")

try:
    mem_report = describe_memory(ds, include_process=True)
    print("describe_memory():")
    import json
    print(json.dumps(mem_report, indent=2))
except Exception as e:
    print("describe_memory() failed:", e)

# wait and sample again
time.sleep(1.0)
gc.collect()
print("\n--- Memory snapshot after 1s ---")
if psutil:
    try:
        mi = p.memory_info()
        print("psutil memory_info() -> rss:", mi.rss)
    except Exception as e:
        print("psutil memory_info() failed:", e)
else:
    print("psutil missing - cannot print process RSS")

try:
    mem_report2 = describe_memory(ds, include_process=True)
    import json
    print("describe_memory() after 1s:")
    print(json.dumps(mem_report2, indent=2))
except Exception as e:
    print("describe_memory() failed:", e)

print("\nNow: open Task Manager (or your OS monitor). Find the PID above and compare its Working Set / RSS with the 'rss' values printed here.")
print("Sleeping 60 seconds so you can inspect Task Manager... (press Ctrl-C to quit early)")

try:
    time.sleep(60)
except KeyboardInterrupt:
    print("Interrupted by user; exiting.")