#!/usr/bin/env python3
"""
gui_windows.py

Minimal Tkinter GUI for dataset_core.py:
 - Mode selector (auto / air / mmap) — persisted
 - Load dataset (uses selected mode)
 - Clear cache
 - Enter TMDB id + type and Run query
 - Shows metadata and JSON result
No ram_copy, no prefetch buttons, no benchmark button — minimal for testing modes.
"""

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk, messagebox
import json
import os
import threading
import time

# import minimal dataset_core functions
from dataset_core import (
    load_or_fetch,
    describe_memory,
    get_persisted_mode,
    set_persisted_mode,
)

# --- Config ---
GITHUB_RAW_BIN_URL = 'https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin'
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.tmdb_similar_cache')
CACHE_PATH = os.path.join(CACHE_DIR, 'dataset.bin')

# --- GUI ---
class App:
    def __init__(self, root):
        self.root = root
        root.title("Similar Quick Viewer (minimal)")
        root.geometry("820x520")

        top = ttk.Frame(root)
        top.pack(fill='x', padx=8, pady=8)

        # TMDB id + type
        ttk.Label(top, text="TMDB ID:").grid(row=0, column=0, sticky='w')
        self.id_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.id_var, width=12).grid(row=0, column=1, sticky='w')

        ttk.Label(top, text="Type:").grid(row=0, column=2, sticky='w', padx=(10,0))
        self.type_var = tk.StringVar(value="movie")
        ttk.Combobox(top, textvariable=self.type_var, values=('movie','tv'), width=8, state='readonly').grid(row=0, column=3, sticky='w')

        # Mode selector
        ttk.Label(top, text="Mode:").grid(row=0, column=4, sticky='w', padx=(10,0))
        self.mode_var = tk.StringVar(value="auto")
        self.mode_combo = ttk.Combobox(top, textvariable=self.mode_var,
                                       values=('auto','air','mmap'),
                                       width=10, state='readonly')
        self.mode_combo.grid(row=0, column=5, sticky='w', padx=(4,0))
        self.mode_combo.bind('<<ComboboxSelected>>', self.on_mode_selected)

        # initialize mode from persisted setting if any
        try:
            persisted = get_persisted_mode(CACHE_DIR)
            if persisted in ('auto','air','mmap'):
                self.mode_var.set(persisted)
            else:
                self.mode_var.set('auto')
        except Exception:
            self.mode_var.set('auto')

        # Buttons: Load, Run, Clear
        self.load_btn = ttk.Button(top, text="Load dataset", command=self.on_load)
        self.load_btn.grid(row=0, column=6, padx=8)

        self.run_btn = ttk.Button(top, text="Run", command=self.on_run)
        self.run_btn.grid(row=0, column=7, padx=8)

        self.clear_btn = ttk.Button(top, text="Clear Cache", command=self.on_clear)
        self.clear_btn.grid(row=0, column=8, padx=8)

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

        # internals
        self.dataset = None
        self.meta = None

        # auto-start background load (runs once at startup)
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
        """Auto-load at start; runs in background thread. Uses selected mode."""
        try:
            self.set_status('loading dataset (background)...')
            self._set_buttons_state(load=False, run=False, clear=False)
            self.mode_combo.config(state='disabled')

            mode = self.mode_var.get()
            ds, meta = load_or_fetch(
                GITHUB_RAW_BIN_URL,
                CACHE_PATH,
                mode=mode
            )

            self._close_dataset_if_any()
            self.dataset = ds
            self.meta = meta

            try:
                meta['memory_snapshot'] = describe_memory(ds, include_process=True)
            except Exception:
                meta['memory_snapshot'] = {'error': 'could not describe memory'}

            chosen = meta.get('mode_chosen')
            air_active = meta.get('air_active', False)
            air_info = ""
            if air_active:
                air_info = f" AIR active (copied {meta.get('air_bytes_copied')} bytes in {meta.get('air_copy_ms')} ms)."
            s = f"Loaded. mode={chosen}.{air_info} file={meta.get('size_bytes','n/a')} bytes"
            self.set_meta(s)
            self.set_status('idle')
        except Exception as e:
            self.set_meta(f"Load error: {e}")
            self.set_status('error')
        finally:
            try:
                self.mode_combo.config(state='readonly')
            except Exception:
                pass
            self._set_buttons_state(load=True, run=True, clear=True)

    def on_load(self):
        """Manual load triggered by button."""
        def task():
            try:
                self.set_status('loading...')
                self._set_buttons_state(load=False, run=False, clear=False)
                self.mode_combo.config(state='disabled')

                mode = self.mode_var.get()
                ds, meta = load_or_fetch(
                    GITHUB_RAW_BIN_URL,
                    CACHE_PATH,
                    mode=mode
                )
                self._close_dataset_if_any()
                self.dataset = ds
                self.meta = meta

                try:
                    meta['memory_snapshot'] = describe_memory(ds, include_process=True)
                except Exception:
                    meta['memory_snapshot'] = {'error': 'could not describe memory'}

                chosen = meta.get('mode_chosen')
                air_active = meta.get('air_active', False)
                air_info = ""
                if air_active:
                    air_info = f" AIR active (copied {meta.get('air_bytes_copied')} bytes in {meta.get('air_copy_ms')} ms)."
                self.set_meta(f"Loaded. mode={chosen}.{air_info} file={meta.get('size_bytes','n/a')} bytes")
                self.set_status('idle')
            except Exception as e:
                self.set_meta(f"Load error: {e}")
                self.set_status('error')
            finally:
                try:
                    self.mode_combo.config(state='readonly')
                except Exception:
                    pass
                self._set_buttons_state(load=True, run=True, clear=True)

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

        def task():
            try:
                self.set_status('running query...')
                self._set_buttons_state(load=False, run=False, clear=False)

                t0 = time.perf_counter()
                results, timings = self.dataset.query_similar(tmdb_id, kind)
                total_ms = (time.perf_counter() - t0) * 1000.0

                try:
                    mem_diag = describe_memory(self.dataset, include_process=True)
                except Exception:
                    mem_diag = {'error': 'could not describe memory'}

                info = {
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

                self.set_result(json.dumps(info, indent=2))
                self.set_meta(f"Found {len(results)}. Total {round(total_ms,3)} ms.")
            except Exception as e:
                self.set_result(f"Error during query: {e}")
                self.set_meta("error")
            finally:
                self._set_buttons_state(load=True, run=True, clear=True)
                self.set_status('idle')

        threading.Thread(target=task, daemon=True).start()

    def on_mode_selected(self, event=None):
        val = self.mode_var.get()
        # persist user's explicit selection (so next runs remember it)
        try:
            ok = set_persisted_mode(CACHE_DIR, val)
            if not ok:
                messagebox.showwarning("Warning", f"Could not persist mode choice '{val}'.")
        except Exception:
            messagebox.showwarning("Warning", "Persisting mode failed (no Kodi addon or file access).")

        # Inform user that explicit selections are persisted (no restart strictly required for testing)
        messagebox.showinfo("Mode selected", f"Mode set to '{val}'. This choice is persisted; press Load dataset to apply now.")

    def _set_buttons_state(self, load=True, run=True, clear=True):
        try:
            self.load_btn.config(state='normal' if load else 'disabled')
            self.run_btn.config(state='normal' if run else 'disabled')
            self.clear_btn.config(state='normal' if clear else 'disabled')
            self.root.update_idletasks()
        except Exception:
            pass


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()