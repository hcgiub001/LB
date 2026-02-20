# memory_probe.py
import os, time, gc
try:
    import psutil
except Exception:
    psutil = None

from dataset_core import load_or_fetch, describe_memory

URL = 'https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin'
CACHE = os.path.expanduser('~/.tmdb_similar_cache/dataset.bin')

print("PID:", os.getpid())
if psutil:
    p = psutil.Process(os.getpid())
    print("psutil says process.exe:", p.name())
else:
    print("psutil not installed. Install with: pip install psutil")

# Load dataset (this will start background expand if enabled in load_or_fetch)
ds, meta = load_or_fetch(URL, CACHE, use_mmap=True, preexpand_3byte=True)

# wait briefly to allow background expand to start/finish
print("Loaded. meta timings:", meta.get('timings'))
time.sleep(0.15)

# run gc then take snapshots
gc.collect()
if psutil:
    mi = p.memory_info()
    print("psutil memory_info() -> rss, vms, private: ", getattr(mi, 'rss', None), getattr(mi, 'vms', None))
else:
    print("psutil missing - cannot print process RSS")

print("describe_memory() snapshot:")
print(describe_memory(ds))

# Wait, print again in 1s to show any drift
time.sleep(1.0)
gc.collect()
if psutil:
    mi = p.memory_info()
    print("psutil memory_info() after 1s -> rss:", mi.rss)
print("describe_memory() after 1s:")
print(describe_memory(ds))

print("\nNow: open Task Manager -> Details tab -> find PID above. Compare the 'Working set' and 'Private Working Set' columns to the 'rss' above.")

print("Sleeping 60 seconds so you can inspect Task Manager...")
time.sleep(60)