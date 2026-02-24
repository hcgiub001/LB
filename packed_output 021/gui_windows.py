#!/usr/bin/env python3
"""
gui_windows.py
Minimal Tkinter GUI for dataset_core.py.
"""

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk, messagebox
import json
import os
import threading
import time

from dataset_core import (
    load_or_fetch,
    describe_memory,
    get_persisted_mode,
    set_persisted_mode,
)

# ---- Config ----
GITHUB_RAW_BIN_URL = (
    'https://raw.githubusercontent.com/hcgiub001/LB/main/'
    'packed_output%20007/dataset.bin'
)
CACHE_DIR  = os.path.join(os.path.expanduser('~'), '.tmdb_similar_cache')
CACHE_PATH = os.path.join(CACHE_DIR, 'dataset.bin')


class App:
    def __init__(self, root: tk.Tk):
        self.root    = root
        self.dataset = None
        self.meta    = None

        root.title("Similar Quick Viewer")
        root.geometry("860x560")

        # ---- Top bar ----
        top = ttk.Frame(root)
        top.pack(fill='x', padx=8, pady=8)

        ttk.Label(top, text="TMDB ID:").grid(row=0, column=0, sticky='w')
        self.id_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.id_var, width=12).grid(row=0, column=1, sticky='w')

        ttk.Label(top, text="Type:").grid(row=0, column=2, sticky='w', padx=(10, 0))
        self.type_var = tk.StringVar(value="movie")
        ttk.Combobox(top, textvariable=self.type_var,
                     values=('movie', 'tv'), width=8,
                     state='readonly').grid(row=0, column=3, sticky='w')

        ttk.Label(top, text="Mode:").grid(row=0, column=4, sticky='w', padx=(10, 0))
        self.mode_var   = tk.StringVar(value="auto")
        self.mode_combo = ttk.Combobox(top, textvariable=self.mode_var,
                                       values=('auto', 'air', 'mmap'),
                                       width=10, state='readonly')
        self.mode_combo.grid(row=0, column=5, sticky='w', padx=(4, 0))
        self.mode_combo.bind('<<ComboboxSelected>>', self.on_mode_selected)

        # Initialise from persisted setting
        try:
            persisted = get_persisted_mode(CACHE_DIR)
            self.mode_var.set(persisted if persisted in ('auto', 'air', 'mmap') else 'auto')
        except Exception:
            self.mode_var.set('auto')

        self.load_btn  = ttk.Button(top, text="Load dataset",  command=self.on_load)
        self.run_btn   = ttk.Button(top, text="Run",           command=self.on_run)
        self.clear_btn = ttk.Button(top, text="Clear Cache",   command=self.on_clear)
        self.load_btn.grid(row=0,  column=6, padx=8)
        self.run_btn.grid(row=0,   column=7, padx=8)
        self.clear_btn.grid(row=0, column=8, padx=8)

        # ---- Status / meta bar ----
        meta_frame = ttk.Frame(root)
        meta_frame.pack(fill='x', padx=8, pady=(6, 0))

        self.status_var = tk.StringVar(value="idle")
        ttk.Label(meta_frame, text="Status:").pack(side='left')
        ttk.Label(meta_frame, textvariable=self.status_var,
                  foreground='blue').pack(side='left', padx=(4, 20))
        self.meta_label = ttk.Label(meta_frame, text="Meta: not loaded",
                                    foreground='gray')
        self.meta_label.pack(side='left')

        # ---- Result area ----
        self.result_box = ScrolledText(root, wrap='none', font=('Courier', 11))
        self.result_box.pack(fill='both', expand=True, padx=8, pady=8)

        # Auto-load on startup
        threading.Thread(target=self._do_load, daemon=True).start()

    # ---- helpers ----

    def _set_status(self, s: str):
        self.status_var.set(s)
        self.root.update_idletasks()

    def _set_meta(self, text: str):
        self.meta_label.config(text=text)
        self.root.update_idletasks()

    def _set_result(self, text: str):
        if len(text) > 1_000_000:
            text = text[:1_000_000] + "\n\n[truncated]"
        self.result_box.delete('1.0', tk.END)
        self.result_box.insert(tk.END, text)
        self.root.update_idletasks()

    def _set_buttons(self, *, load=True, run=True, clear=True):
        try:
            self.load_btn.config( state='normal' if load  else 'disabled')
            self.run_btn.config(  state='normal' if run   else 'disabled')
            self.clear_btn.config(state='normal' if clear else 'disabled')
            self.root.update_idletasks()
        except Exception:
            pass

    def _close_dataset(self):
        ds = self.dataset
        if not ds:
            return
        try:
            ds.close()
        except Exception:
            pass
        self.dataset = None
        self.meta    = None

    # ---- load logic (shared by background auto-load and button) ----

    def _do_load(self):
        try:
            self._set_status('loading dataset…')
            self._set_buttons(load=False, run=False, clear=False)
            self.mode_combo.config(state='disabled')

            mode = self.mode_var.get()
            ds, meta = load_or_fetch(GITHUB_RAW_BIN_URL, CACHE_PATH, mode=mode)

            self._close_dataset()
            self.dataset = ds
            self.meta    = meta

            try:
                meta['memory_snapshot'] = describe_memory(ds, include_process=True)
            except Exception:
                meta['memory_snapshot'] = {'error': 'could not describe memory'}

            chosen     = meta.get('mode_chosen', '?')
            air_active = meta.get('air_active', False)
            air_info   = (
                f" AIR active ({meta.get('air_bytes_copied')} bytes "
                f"in {meta.get('air_copy_ms'):.1f} ms)."
                if air_active else ""
            )
            load_ms = meta.get('load_ms', 0.0)
            self._set_meta(
                f"Loaded. mode={chosen}.{air_info} "
                f"file={meta.get('size_bytes', 'n/a')} bytes  load={load_ms:.1f} ms"
            )
            self._set_status('idle')

        except Exception as e:
            self._set_meta(f"Load error: {e}")
            self._set_status('error')
        finally:
            try:
                self.mode_combo.config(state='readonly')
            except Exception:
                pass
            self._set_buttons(load=True, run=True, clear=True)

    def background_load(self):
        self._do_load()

    def on_load(self):
        threading.Thread(target=self._do_load, daemon=True).start()

    def on_clear(self):
        try:
            self._close_dataset()
            if os.path.exists(CACHE_PATH):
                try:
                    os.remove(CACHE_PATH)
                    self._set_meta("Cache cleared.")
                    self._set_result("Cache cleared. Click 'Load dataset' to fetch again.")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not clear cache: {e}")
            else:
                self._set_meta("No cache file present.")
                self._set_result("No cache file present.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not clear cache: {e}")

    def on_run(self):
        if not self.dataset:
            messagebox.showinfo("No dataset",
                                "Dataset not loaded yet. Click 'Load dataset' or wait.")
            return
        idtxt = self.id_var.get().strip()
        if not idtxt.isdigit():
            messagebox.showerror("Invalid id", "Please enter a numeric TMDB id.")
            return
        tmdb_id = int(idtxt)
        kind    = self.type_var.get()

        def task():
            try:
                self._set_status('running query…')
                self._set_buttons(load=False, run=False, clear=False)

                t0 = time.perf_counter()
                results, timings = self.dataset.query_similar(tmdb_id, kind)
                total_ms = (time.perf_counter() - t0) * 1000.0

                try:
                    mem_diag = describe_memory(self.dataset, include_process=True)
                except Exception:
                    mem_diag = {'error': 'could not describe memory'}

                info = {
                    'file_byteLength':    self.meta.get('size_bytes') if self.meta else None,
                    'parsed_sizes':       self.dataset.sizes,
                    'memory_diagnostics': mem_diag,
                    'runtimes_ms': {
                        'total_ms':  round(total_ms, 4),
                        'search_ms': round(timings.get('search_ms', 0), 4),
                        'query_ms':  round(timings.get('query_ms',  0), 4),
                    },
                    'result_count': len(results),
                    'result':       results,
                }
                self._set_result(json.dumps(info, indent=2))
                self._set_meta(
                    f"Found {len(results)} results. "
                    f"Total {round(total_ms, 3)} ms  "
                    f"(search {round(timings.get('search_ms',0),3)} ms  "
                    f"decode {round(timings.get('query_ms',0),3)} ms)"
                )
            except Exception as e:
                self._set_result(f"Error during query: {e}")
                self._set_meta("error")
            finally:
                self._set_buttons(load=True, run=True, clear=True)
                self._set_status('idle')

        threading.Thread(target=task, daemon=True).start()

    def on_mode_selected(self, _event=None):
        val = self.mode_var.get()
        try:
            ok = set_persisted_mode(CACHE_DIR, val)
            if not ok:
                messagebox.showwarning("Warning",
                                       f"Could not persist mode choice '{val}'.")
        except Exception:
            messagebox.showwarning("Warning", "Persisting mode failed.")
        messagebox.showinfo("Mode selected",
                            f"Mode set to '{val}'. Press 'Load dataset' to apply.")


if __name__ == '__main__':
    root = tk.Tk()
    app  = App(root)
    root.mainloop()