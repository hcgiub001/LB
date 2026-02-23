#!/usr/bin/env python3
# simulate_browsing.py
# Simulate browsing many distinct rows and report process RSS growth.
# Usage: python simulate_browsing.py
#
# Requirements: dataset_core.py in same folder, psutil recommended (pip install psutil)

import os, time, gc, math
import sys
try:
    import psutil
except Exception:
    psutil = None

from dataset_core import load_or_fetch, now_ms

URL = 'https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin'
CACHE = os.path.expanduser('~/.tmdb_similar_cache/dataset.bin')

def fmt_bytes(n):
    if n is None: return 'n/a'
    for unit in ('B','KB','MB','GB'):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"

def proc_rss():
    if psutil:
        p = psutil.Process(os.getpid())
        mi = p.memory_info()
        return mi.rss, getattr(mi, 'vms', None)
    else:
        return None, None

def linux_mapping_rss():
    # best-effort: try smaps_rollup
    if not sys.platform.startswith('linux'):
        return None
    try:
        with open('/proc/self/smaps_rollup', 'rt') as fh:
            text = fh.read()
        for line in text.splitlines():
            if line.startswith('Rss:'):
                parts = line.split()
                return int(parts[1]) * 1024
    except Exception:
        pass
    return None

def touch_all(mm):
    # read one byte per page to fault pages into process working set
    page = 4096
    size = mm.size()
    for off in range(0, size, page):
        _ = mm[off]
    return

def run_simulation(total_queries=200, batch_size=20, do_touch_all=False):
    print("Loading dataset (may download if missing)...")
    ds, meta = load_or_fetch(URL, CACHE, use_mmap=True)
    print("Loaded:", meta.get('path', CACHE), "size:", fmt_bytes(meta.get('size_bytes')))
    parsed = ds.parsed
    R = parsed['R']
    print("Rows R=", R, "entries E=", parsed.get('E'), "unique U=", parsed.get('U'))
    # baseline RSS
    gc.collect()
    rss, vms = proc_rss()
    print("Baseline RSS:", fmt_bytes(rss) if rss else 'psutil missing')
    if sys.platform.startswith('linux'):
        print("Mapping RSS (smaps_rollup):", fmt_bytes(linux_mapping_rss()) if linux_mapping_rss() else 'n/a')

    # sample source keys spaced through R
    if R == 0:
        print("No rows in dataset")
        return

    # pick indices evenly spaced
    num_to_query = min(total_queries, R)
    step = max(1, R // num_to_query)
    indices = list(range(0, R, step))[:num_to_query]

    print(f"Will query {len(indices)} distinct rows (step={step}). Batch size {batch_size}.")

    start_total = now_ms()
    last_report_rss = rss
    queried = 0
    for i, idx in enumerate(indices):
        # get packed key from source_keys view
        pk = int(parsed['views']['source_keys'][idx])
        src_id = pk >> 1
        src_kind = 'tv' if (pk & 1) else 'movie'
        t0 = now_ms()
        res, timings = ds.query_similar(src_id, src_kind)
        t1 = now_ms()
        queried += 1

        # every batch, print rss and summary
        if (i + 1) % batch_size == 0 or (i + 1) == len(indices):
            gc.collect()
            rss_now, _ = proc_rss()
            linux_rss = linux_mapping_rss()
            print(f"[{i+1}/{len(indices)}] Queried distinct rows: {i+1}. Total runtime so far: {now_ms() - start_total:.1f} ms")
            print(f"  process RSS: {fmt_bytes(rss_now) if rss_now else 'n/a'} (delta {fmt_bytes((rss_now - (last_report_rss or 0))) if rss_now and last_report_rss else 'n/a'})")
            if linux_rss:
                print(f"  mapping_rss (smaps_rollup): {fmt_bytes(linux_rss)}")
            last_report_rss = rss_now
    total_time = now_ms() - start_total
    print(f"Finished {len(indices)} distinct queries in {total_time:.1f} ms")

    # optionally force-touch-all (worst-case)
    if do_touch_all:
        print("Now forcing worst-case: touching all pages to fault the entire file into working set...")
        before, _ = proc_rss()
        t0 = now_ms()
        touch_all(ds._mmap)
        t1 = now_ms()
        gc.collect()
        after, _ = proc_rss()
        print(f"  touch_all elapsed ms: {t1-t0:.1f}")
        print("  RSS before:", fmt_bytes(before), "after:", fmt_bytes(after), "delta:", fmt_bytes(after - before))

    # cleanup
    ds.close()
    print("Done. Final RSS:", fmt_bytes(proc_rss()[0]) if proc_rss()[0] else 'n/a')

if __name__ == '__main__':
    # tune params as desired
    run_simulation(total_queries=200, batch_size=20, do_touch_all=True)