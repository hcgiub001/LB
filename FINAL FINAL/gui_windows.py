# gui_windows.py
"""
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

# import dataset core functions + diagnostics
from dataset_core import load_or_fetch, parse_dataset_file, now_ms, format_ms, describe_memory

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

        self.load_btn = ttk.Button(top, text="Load dataset", command=self.on_load)
        self.load_btn.grid(row=0, column=5, padx=8)

        self.clear_btn = ttk.Button(top, text="Clear Cache", command=self.on_clear)
        self.clear_btn.grid(row=0, column=6, padx=8)

        # status/meta
        meta_frame = ttk.Frame(root)
        meta_frame.pack(fill='x', padx=8)
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

        # auto-start background load
        threading.Thread(target=self.background_load, daemon=True).start()

    def set_status(self, s):
        self.status_var.set(s)
        self.root.update_idletasks()

    def set_meta(self, text):
        self.meta_label.config(text=text)
        self.root.update_idletasks()

    def set_result(self, text):
        self.result_box.delete('1.0', tk.END)
        self.result_box.insert(tk.END, text)
        self.root.update_idletasks()

    # watcher thread to record memory after background expansion finishes
    def _start_postexpand_watcher(self, ds, meta):
        def watcher():
            # If expansion in progress, wait until it's finished
            # Otherwise if values_idx appears later, capture then.
            waited = 0.0
            timeout = 10.0  # seconds max to avoid waiting forever
            poll = 0.02
            # If expansion flag exists and is True, wait until it clears
            while getattr(ds, '_expansion_in_progress', False) and waited < timeout:
                time.sleep(poll)
                waited += poll
            # if values_idx present (expanded), record snapshot
            if getattr(ds, 'values_idx', None) is not None:
                try:
                    meta['memory_after_expand'] = describe_memory(ds)
                except Exception:
                    meta['memory_after_expand'] = {'error': 'failed to describe after expand'}
                # update UI meta text to reflect post-expand snapshot
                self.set_meta(f"Loaded. File: {meta.get('size_bytes', 'n/a')} bytes (cache). Expanded values_idx.")
            else:
                # no expansion happened (either disabled or failed)
                meta['memory_after_expand'] = None
        threading.Thread(target=watcher, daemon=True).start()

    def background_load(self):
        try:
            self.set_status('loading dataset...')
            # allow dataset_core to start background expansion (default True)
            ds, meta = load_or_fetch(GITHUB_RAW_BIN_URL, CACHE_PATH, use_mmap=True, preexpand_3byte=True)
            self.dataset = ds
            self.meta = meta

            # take immediate memory snapshot (quick)
            try:
                meta['memory_snapshot'] = describe_memory(ds)
            except Exception:
                meta['memory_snapshot'] = {'error': 'could not describe memory'}

            # if background expansion is active, start watcher to capture post-expansion state
            try:
                self._start_postexpand_watcher(ds, meta)
            except Exception:
                pass

            s = f"Dataset ready. File: {meta.get('size_bytes', 'n/a')} bytes ({'cache' if meta.get('from_cache') else 'network'})"
            self.set_meta(s)
            self.set_status('idle')
        except Exception as e:
            self.set_meta(f"Load error: {e}")
            self.set_status('error')

    def on_load(self):
        def task():
            try:
                self.set_status('loading...')
                ds, meta = load_or_fetch(GITHUB_RAW_BIN_URL, CACHE_PATH, use_mmap=True, preexpand_3byte=True)
                self.dataset = ds
                self.meta = meta

                try:
                    meta['memory_snapshot'] = describe_memory(ds)
                except Exception:
                    meta['memory_snapshot'] = {'error': 'could not describe memory'}

                try:
                    self._start_postexpand_watcher(ds, meta)
                except Exception:
                    pass

                self.set_meta(f"Loaded. File: {meta.get('size_bytes', 'n/a')} bytes ({'cache' if meta.get('from_cache') else 'network'})")
                self.set_status('idle')
            except Exception as e:
                self.set_meta(f"Load error: {e}")
                self.set_status('error')
        threading.Thread(target=task, daemon=True).start()

    def on_clear(self):
        try:
            if os.path.exists(CACHE_PATH):
                os.remove(CACHE_PATH)
            self.dataset = None
            self.set_meta("Cache cleared.")
            self.set_result("Cache cleared. Click 'Load dataset' to fetch again.")
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
                t0 = time.perf_counter()
                results, timings = self.dataset.query_similar(tmdb_id, kind)
                total_ms = (time.perf_counter() - t0) * 1000.0

                # immediate memory diagnostics snapshot for this query
                try:
                    mem_diag = describe_memory(self.dataset)
                except Exception:
                    mem_diag = {'error': 'could not describe memory'}

                info = {
                    'source_file': 'file' if self.meta and self.meta.get('from_cache') else 'network',
                    'file_byteLength': self.meta.get('size_bytes') if self.meta else None,
                    'parsed_sizes': self.dataset.sizes,
                    'memory_diagnostics': mem_diag,
                    'runtimes_ms': {
                        'total_ms': round(total_ms, 3),
                        'search_ms': timings.get('search_ms'),
                        'query_ms': timings.get('query_ms')
                    },
                    'result_count': len(results),
                    'result': results
                }

                # if load meta included snapshots, attach them as well
                if self.meta:
                    if 'memory_snapshot' in self.meta:
                        info['memory_snapshot_on_load'] = self.meta['memory_snapshot']
                    if 'memory_after_expand' in self.meta:
                        info['memory_after_expand'] = self.meta.get('memory_after_expand')

                self.set_result(json.dumps(info, indent=2))
                self.set_meta(f"Found {len(results)}. Total {round(total_ms,3)} ms. File: {self.meta.get('size_bytes')} bytes")
            except Exception as e:
                self.set_result(f"Error during query: {e}")
                self.set_meta("error")
            finally:
                self.set_status('idle')

        threading.Thread(target=task, daemon=True).start()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()