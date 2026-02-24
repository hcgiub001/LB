#!/usr/bin/env python3
"""
gui_windows.py — minimal Tkinter GUI that uses dataset_core.query_similar_pairs

Behavior:
 - Mode selector (auto / air / mmap) — NOT persisted (dataset_core no longer provides persistence)
 - Load dataset (uses selected mode)
 - Clear cache
 - Enter TMDB id + type and Run query
 - Shows compact JSON pairs output: {"count":N,"results":[[id,type_bit],...]}
"""
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk, messagebox
import json
import os
import threading
import time
import tempfile
import shutil

# import minimal dataset_core functions
from dataset_core import (
    load_or_fetch,
)

# --- Config ---
GITHUB_RAW_BIN_URL = 'https://raw.githubusercontent.com/hcgiub001/LB/main/packed_output%20007/dataset.bin'
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.tmdb_similar_cache')
CACHE_PATH = os.path.join(CACHE_DIR, 'dataset.bin')


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

        # Mode selector (not persisted)
        ttk.Label(top, text="Mode:").grid(row=0, column=4, sticky='w', padx=(10,0))
        self.mode_var = tk.StringVar(value="auto")
        self.mode_combo = ttk.Combobox(top, textvariable=self.mode_var,
                                       values=('auto','air','mmap'),
                                       width=10, state='readonly')
        self.mode_combo.grid(row=0, column=5, sticky='w', padx=(4,0))

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
                # try best-effort cleanup
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

            chosen = meta.get('mode_chosen')
            s = f"Loaded. mode={chosen} file={meta.get('size_bytes','n/a')} bytes"
            self.set_meta(s)
            self.set_status('idle')
            # show compact info in result box to confirm load (brief)
            self.set_result(f'{{"loaded":true,"mode":"{chosen}","file_bytes":{meta.get("size_bytes",0)}}}')
        except Exception as e:
            self.set_meta(f"Load error: {e}")
            self.set_status('error')
            self.set_result(f"Load error: {e}")
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

                chosen = meta.get('mode_chosen')
                self.set_meta(f"Loaded. mode={chosen} file={meta.get('size_bytes','n/a')} bytes")
                self.set_result(f'{{"loaded":true,"mode":"{chosen}","file_bytes":{meta.get("size_bytes",0)}}}')
                self.set_status('idle')
            except Exception as e:
                self.set_meta(f"Load error: {e}")
                self.set_status('error')
                self.set_result(f"Load error: {e}")
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
                    self.set_result('{"cache_cleared":true}')
                except Exception as e:
                    messagebox.showerror("Error", f"Could not clear cache file: {e}")
            else:
                self.set_meta("No cache file present.")
                self.set_result('{"cache_present":false}')
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

                # query_similar_pairs returns compact shape {"count":N,"results":[[id,type],...]}
                out = self.dataset.query_similar_pairs(tmdb_id, kind)

                # show compact JSON (no extras) — developer can copy/paste or pipe this output
                compact = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
                # also show a pretty version for readability — keep both: first pretty, then compact below
                pretty = json.dumps(out, indent=2, ensure_ascii=False)
                display_text = f"// compact JSON (ready for addon consumption)\n{compact}\n\n// pretty (readable)\n{pretty}"
                self.set_result(display_text)

                self.set_meta(f"Found {out.get('count',0)} results.")
            except Exception as e:
                self.set_result(f"Error during query: {e}")
                self.set_meta("error")
            finally:
                self._set_buttons_state(load=True, run=True, clear=True)
                self.set_status('idle')

        threading.Thread(target=task, daemon=True).start()

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