#!/usr/bin/env python3
"""
gui_windows.py

Small Tkinter GUI that uses dataset_core.py.
Run: python gui_windows.py
"""

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk, messagebox
import json
import os
import threading
import time

# import dataset core functions + diagnostics + benchmark helper
from dataset_core import (
    load_or_fetch,
    now_ms,
    format_ms,
    describe_memory,
    benchmark_cold_vs_warm,
)

# --- Config ---
GITHUB_RAW_BIN_URL = 'https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin'
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.tmdb_similar_cache')
CACHE_PATH = os.path.join(CACHE_DIR, 'dataset.bin')

# --- GUI ---
class App:
    def __init__(self, root):
        self.root = root
        root.title("Similar Quick Viewer (Python)")
        root.geometry("760x520")

        top = ttk.Frame(root)
        top.pack(fill='x', padx=8, pady=8)

        ttk.Label(top, text="TMDB ID:").grid(row=0, column=0, sticky='w')
        self.id_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.id_var, width=12).grid(row=0, column=1, sticky='w')

        ttk.Label(top, text="Type:").grid(row=0, column=2, sticky='w', padx=(10,0))
        self.type_var = tk.StringVar(value="movie")
        ttk.Combobox(top, textvariable=self.type_var, values=('movie','tv'), width=6, state='readonly').grid(row=0, column=3, sticky='w')

        self.run_btn = ttk.Button(top, text="Run", command=self.on_run)
        self.run_btn.grid(row=0, column=4, padx=8)

        self.bench_btn = ttk.Button(top, text="Benchmark", command=self.on_benchmark)
        self.bench_btn.grid(row=0, column=5, padx=8)

        self.load_btn = ttk.Button(top, text="Load dataset", command=self.on_load)
        self.load_btn.grid(row=0, column=6, padx=8)

        self.clear_btn = ttk.Button(top, text="Clear Cache", command=self.on_clear)
        self.clear_btn.grid(row=0, column=7, padx=8)

        # ram_copy toggle (keep current default)
        self.ram_copy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="ram_copy (prefetch indices)", variable=self.ram_copy_var).grid(row=0, column=8, padx=(12,0))

        # status/meta
        meta_frame = ttk.Frame(root)
        meta_frame.pack(fill='x', padx=8, pady=(6,0))
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(meta_frame, text="Status:").pack(side='left')
        ttk.Label(meta_frame, textvariable=self.status_var, foreground='blue').pack(side='left', padx=(4,20))
        self.meta_label = ttk.Label(meta_frame, text="Meta: not loaded", foreground='gray')
        self.meta_label.pack(side='left')

        # result area
        self.result_box = ScrolledText(root, wrap='none', font=('Courier', 11))
        self.result_box.pack(fill='both', expand=True, padx=8, pady=8)

        # internal
        self.dataset = None
        self.meta = None

        # auto-start background load (uses ram_copy setting)
        threading.Thread(target=self.background_load, daemon=True).start()

    def set_status(self, s):
        self.status_var.set(s)
        self.root.update_idletasks()

    def set_meta(self, text):
        self.meta_label.config(text=text)
        self.root.update_idletasks()

    def set_result(self, text):
        if len(text) > 1_000_000:
            text = text[:1_000_000] + "\n\n[truncated large output]"
        self.result_box.delete('1.0', tk.END)
        self.result_box.insert(tk.END, text)
        self.root.update_idletasks()

    def _close_dataset_if_any(self):
        ds = self.dataset
        if not ds:
            return
        try:
            try:
                ds.close()
            except Exception:
                try:
                    if getattr(ds, '_mmap', None) is not None:
                        ds._mmap.close()
                except Exception:
                    pass
                try:
                    if getattr(ds, '_fileobj', None) is not None:
                        ds._fileobj.close()
                except Exception:
                    pass
        finally:
            self.dataset = None
            self.meta = None

    def background_load(self):
        """Auto-load at start; runs in background thread."""
        try:
            self.set_status('loading dataset (background)...')
            self._set_buttons_state(load=False, run=False, clear=False, bench=False)

            # Fetch (stream) and mmap the cached file (honor GUI checkbox)
            ds, meta = load_or_fetch(GITHUB_RAW_BIN_URL, CACHE_PATH, use_mmap=True, ram_copy=self.ram_copy_var.get())

            # replace previous dataset if any
            self._close_dataset_if_any()
            self.dataset = ds
            self.meta = meta

            # memory snapshot (include process info)
            try:
                meta['memory_snapshot'] = describe_memory(ds, include_process=True)
            except Exception:
                meta['memory_snapshot'] = {'error': 'could not describe memory'}

            s = f"Dataset ready. File: {meta.get('size_bytes', 'n/a')} bytes ({'cache' if meta.get('from_cache') else 'network'})"
            self.set_meta(s)
            self.set_status('idle')
        except Exception as e:
            self.set_meta(f"Load error: {e}")
            self.set_status('error')
        finally:
            self._set_buttons_state(load=True, run=True, clear=True, bench=True)

    def on_load(self):
        """Manual load triggered by button."""
        def task():
            try:
                self.set_status('loading...')
                self._set_buttons_state(load=False, run=False, clear=False, bench=False)
                ds, meta = load_or_fetch(GITHUB_RAW_BIN_URL, CACHE_PATH, use_mmap=True, ram_copy=self.ram_copy_var.get())
                self._close_dataset_if_any()
                self.dataset = ds
                self.meta = meta

                try:
                    meta['memory_snapshot'] = describe_memory(ds, include_process=True)
                except Exception:
                    meta['memory_snapshot'] = {'error': 'could not describe memory'}

                self.set_meta(f"Loaded. File: {meta.get('size_bytes', 'n/a')} bytes ({'cache' if meta.get('from_cache') else 'network'})")
                self.set_status('idle')
            except Exception as e:
                self.set_meta(f"Load error: {e}")
                self.set_status('error')
            finally:
                self._set_buttons_state(load=True, run=True, clear=True, bench=True)

        threading.Thread(target=task, daemon=True).start()

    def on_clear(self):
        """Remove cache file and unload dataset (if any)."""
        try:
            self._close_dataset_if_any()
            if os.path.exists(CACHE_PATH):
                try:
                    os.remove(CACHE_PATH)
                    self.set_meta("Cache cleared.")
                    self.set_result("Cache cleared. Click 'Load dataset' to fetch again.")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not clear cache file: {e}")
            else:
                self.set_meta("No cache file present.")
                self.set_result("No cache file present.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not clear cache: {e}")

    def on_run(self):
        if not self.dataset:
            messagebox.showinfo("No dataset", "Dataset not loaded yet. Click 'Load dataset' or wait for background load.")
            return
        idtxt = self.id_var.get().strip()
        if not idtxt.isdigit():
            messagebox.showerror("Invalid id", "Please enter a numeric tmdb id.")
            return
        tmdb_id = int(idtxt)
        kind = self.type_var.get()

        # run query in worker to avoid UI freeze (fast though)
        def task():
            try:
                self.set_status('running query...')
                self._set_buttons_state(load=False, run=False, clear=False, bench=False)

                # === Timing: EXACTLY like your original script ===
                # Start timer immediately before the dataset.query_similar() call,
                # stop immediately after it returns. Nothing else is included.
                t0 = time.perf_counter()
                results, timings = self.dataset.query_similar(tmdb_id, kind)
                total_ms = (time.perf_counter() - t0) * 1000.0
                # === End timing (query_similar only) ===

                # immediate memory diagnostics snapshot for this query (include process info)
                try:
                    mem_diag = describe_memory(self.dataset, include_process=True)
                except Exception:
                    mem_diag = {'error': 'could not describe memory'}

                info = {
                    'source_file': 'file' if self.meta and self.meta.get('from_cache') else 'network',
                    'file_byteLength': self.meta.get('size_bytes') if self.meta else None,
                    'parsed_sizes': self.dataset.sizes if self.dataset else None,
                    'memory_diagnostics': mem_diag,
                    'runtimes_ms': {
                        'total_ms': round(total_ms, 3),
                        'search_ms': timings.get('search_ms'),
                        'query_ms': timings.get('query_ms')
                    },
                    'result_count': len(results),
                    'result': results
                }

                # include load-time snapshot if present
                if self.meta:
                    if 'memory_snapshot' in self.meta:
                        info['memory_snapshot_on_load'] = self.meta['memory_snapshot']

                # UI update and JSON serialization happen AFTER the measured interval
                self.set_result(json.dumps(info, indent=2))
                self.set_meta(f"Found {len(results)}. Total {round(total_ms,3)} ms. File: {self.meta.get('size_bytes') if self.meta else 'n/a'} bytes")
            except Exception as e:
                self.set_result(f"Error during query: {e}")
                self.set_meta("error")
            finally:
                self._set_buttons_state(load=True, run=True, clear=True, bench=True)
                self.set_status('idle')

        threading.Thread(target=task, daemon=True).start()

    def on_benchmark(self):
        """Run benchmark_cold_vs_warm helper (background thread)."""
        if not self.dataset:
            messagebox.showinfo("No dataset", "Dataset not loaded yet. Click 'Load dataset' first.")
            return
        idtxt = self.id_var.get().strip()
        if not idtxt.isdigit():
            messagebox.showerror("Invalid id", "Please enter a numeric tmdb id for benchmark.")
            return
        tmdb_id = int(idtxt)
        kind = self.type_var.get()

        def task():
            try:
                self.set_status('benchmarking...')
                self._set_buttons_state(load=False, run=False, clear=False, bench=False)
                bench = benchmark_cold_vs_warm(self.dataset, tmdb_id, kind, n_warm=200)
                self.set_result(json.dumps(bench, indent=2))
                self.set_meta(f"Benchmark complete. ram_copy_active={bench.get('ram_copy_active')}")
            except Exception as e:
                self.set_result(f"Benchmark error: {e}")
                self.set_meta("error")
            finally:
                self._set_buttons_state(load=True, run=True, clear=True, bench=True)
                self.set_status('idle')

        threading.Thread(target=task, daemon=True).start()

    def _set_buttons_state(self, load=True, run=True, clear=True, bench=True):
        try:
            self.load_btn.config(state='normal' if load else 'disabled')
            self.run_btn.config(state='normal' if run else 'disabled')
            self.clear_btn.config(state='normal' if clear else 'disabled')
            self.bench_btn.config(state='normal' if bench else 'disabled')
            self.root.update_idletasks()
        except Exception:
            pass


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()