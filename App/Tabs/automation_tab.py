#!/usr/bin/env python3
"""
AutomationTab — GUI front-end for the LHS sweep pipeline + NN training.

Stages:
  1. Generate Samples  — runs generate_lhs_samples.py in a background thread
  2. Run Sweep         — runs run_sweep.py in a background subprocess
  3. Pause / Resume    — writes/removes the STOP_SWEEP sentinel file
  4. Train Surrogate   — runs NeuralNetwork/train_surrogate.py in a subprocess
"""

import os
import sys
import json
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk

_HERE         = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT     = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_APP_ROOT)
_SIMDATA_DIR  = os.path.join(_APP_ROOT, "SimulationData")
_MODULES_DIR  = os.path.join(_APP_ROOT, "Modules")
_NN_DIR       = os.path.join(_PROJECT_ROOT, "NeuralNetwork")
_GEN_SCRIPT   = os.path.join(_MODULES_DIR, "generate_lhs_samples.py")
_SWEEP_SCRIPT = os.path.join(_MODULES_DIR, "run_sweep.py")
_TRAIN_SCRIPT = os.path.join(_NN_DIR, "train_surrogate.py")
_SAMPLES_FILE = os.path.join(_SIMDATA_DIR, "lhs_samples.json")
_RESULTS_FILE = os.path.join(_SIMDATA_DIR, "sweep_results.json")
_STOP_FLAG    = os.path.join(_SIMDATA_DIR, "STOP_SWEEP")
_MODEL_FILE   = os.path.join(_NN_DIR, "surrogate_model.pth")
_LOSS_PLOT    = os.path.join(_NN_DIR, "loss_curve.png")

for _p in (_MODULES_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POLL_MS = 400


class AutomationTab(ttk.Frame):
    def __init__(self, parent, app=None, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self._log_queue        = queue.Queue()
        self._train_log_queue  = queue.Queue()
        self._gen_thread       = None   # thread for sample generation
        self._sweep_proc       = None   # subprocess for sweep
        self._sweep_thread     = None   # thread draining sweep stdout
        self._train_proc       = None   # subprocess for NN training
        self._train_thread     = None   # thread draining train stdout
        self._covered_indices = set()
        self._tag_to_idx      = {}
        self._n_total_samples = 0
        self._build()
        self._refresh_status()
        self.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ build
    def _build(self):
        # ---- top: two config columns -----------------------------------
        cfg = ttk.Frame(self)
        cfg.pack(fill="x", padx=10, pady=(10, 4))
        cfg.columnconfigure(0, weight=1, uniform="col")
        cfg.columnconfigure(1, weight=0)
        cfg.columnconfigure(2, weight=1, uniform="col")

        # Left column — Sample generation config
        gen_frm = ttk.LabelFrame(cfg, text="1 · Generate LHS Samples")
        gen_frm.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        r = ttk.Frame(gen_frm); r.pack(fill="x", padx=8, pady=3)
        ttk.Label(r, text="Samples per branch:", width=22, anchor="w").pack(side="left")
        self._n_var = tk.StringVar(value="1500")
        ttk.Entry(r, textvariable=self._n_var, width=8).pack(side="left", padx=4)

        r = ttk.Frame(gen_frm); r.pack(fill="x", padx=8, pady=3)
        ttk.Label(r, text="Random seed:", width=22, anchor="w").pack(side="left")
        self._seed_var = tk.StringVar(value="42")
        ttk.Entry(r, textvariable=self._seed_var, width=8).pack(side="left", padx=4)

        r = ttk.Frame(gen_frm); r.pack(fill="x", padx=8, pady=(3, 8))
        self._gen_btn = ttk.Button(r, text="Generate Samples",
                                   command=self._on_generate)
        self._gen_btn.pack(side="left")
        self._gen_status = tk.StringVar(value="—")
        ttk.Label(r, textvariable=self._gen_status,
                  foreground="gray").pack(side="left", padx=8)

        # Separator
        tk.Frame(cfg, bg="#b0b0b0", width=2).grid(
            row=0, column=1, sticky="ns", padx=4)

        # Right column — Sweep config
        sw_frm = ttk.LabelFrame(cfg, text="2 · Run FastHenry Sweep")
        sw_frm.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        r = ttk.Frame(sw_frm); r.pack(fill="x", padx=8, pady=3)
        ttk.Label(r, text="Parallel workers:", width=20, anchor="w").pack(side="left")
        self._workers_var = tk.StringVar(value="16")
        ttk.Entry(r, textvariable=self._workers_var, width=6).pack(side="left", padx=4)

        r = ttk.Frame(sw_frm); r.pack(fill="x", padx=8, pady=3)
        ttk.Label(r, text="Per-sim timeout (s):", width=20, anchor="w").pack(side="left")
        self._timeout_var = tk.StringVar(value="240")
        ttk.Entry(r, textvariable=self._timeout_var, width=6).pack(side="left", padx=4)

        r = ttk.Frame(sw_frm); r.pack(fill="x", padx=8, pady=3)
        ttk.Label(r, text="Checkpoint every N:", width=20, anchor="w").pack(side="left")
        self._ckpt_var = tk.StringVar(value="50")
        ttk.Entry(r, textvariable=self._ckpt_var, width=6).pack(side="left", padx=4)

        r = ttk.Frame(sw_frm); r.pack(fill="x", padx=8, pady=3)
        ttk.Label(r, text="From index:", width=20, anchor="w").pack(side="left")
        self._from_var = tk.StringVar(value="0")
        self._from_var.trace_add("write", lambda *_: self._redraw_progress())
        ttk.Entry(r, textvariable=self._from_var, width=7).pack(side="left", padx=4)
        ttk.Label(r, text="To index:", width=10, anchor="w").pack(side="left")
        self._to_var = tk.StringVar(value="")
        self._to_var.trace_add("write", lambda *_: self._redraw_progress())
        ttk.Entry(r, textvariable=self._to_var, width=7).pack(side="left", padx=4)
        ttk.Label(r, text="(blank = end)", foreground="gray").pack(side="left")

        btn_row = ttk.Frame(sw_frm); btn_row.pack(fill="x", padx=8, pady=(3, 8))
        self._sweep_btn = ttk.Button(btn_row, text="Start Sweep",
                                     command=self._on_sweep_start)
        self._sweep_btn.pack(side="left")
        self._pause_btn = ttk.Button(btn_row, text="Pause",
                                     command=self._on_pause, state="disabled")
        self._pause_btn.pack(side="left", padx=(6, 0))
        self._sweep_status = tk.StringVar(value="—")
        ttk.Label(btn_row, textvariable=self._sweep_status,
                  foreground="gray").pack(side="left", padx=8)

        # ---- progress bar + stats row -----------------------------------
        prog_frm = ttk.LabelFrame(self, text="Progress")
        prog_frm.pack(fill="x", padx=10, pady=4)

        # Canvas-based segmented progress bar (green = done in range, gray = rest)
        self._prog_canvas = tk.Canvas(prog_frm, height=18, bg="#3c3c3c",
                                      highlightthickness=1,
                                      highlightbackground="#606060")
        self._prog_canvas.pack(fill="x", padx=8, pady=(6, 2))
        self._prog_canvas.bind("<Configure>", lambda e: self._redraw_progress())
        self._prog_range_bar  = None   # green filled rect for active range
        self._prog_done_bar   = None   # brighter green for completed portion
        self._n_total_samples = 0      # total in samples file
        self._run_from = 0
        self._run_to   = 0

        stats_row = ttk.Frame(prog_frm)
        stats_row.pack(fill="x", padx=8, pady=(0, 2))
        self._stat_done  = tk.StringVar(value="Done: —")
        self._stat_ok    = tk.StringVar(value="OK: —")
        self._stat_fail  = tk.StringVar(value="Failed: —")
        self._stat_total = tk.StringVar(value="Total: —")
        for sv in (self._stat_total, self._stat_done,
                   self._stat_ok, self._stat_fail):
            ttk.Label(stats_row, textvariable=sv).pack(side="left", padx=12)

        # Coverage text — shows which index ranges exist in the results file
        self._coverage_var = tk.StringVar(value="Coverage: —")
        ttk.Label(prog_frm, textvariable=self._coverage_var,
                  foreground="#80c8ff", font=("Consolas", 8)).pack(
                      anchor="w", padx=10, pady=(0, 2))

        ttk.Button(prog_frm, text="Refresh",
                   command=self._refresh_status).pack(anchor="e", padx=8, pady=(0, 4))

        # ---- bottom pane: console (left) + NN trainer (right) ----------
        paned = tk.PanedWindow(self, orient="horizontal",
                               sashwidth=5, sashrelief="flat",
                               bg="#3c3c3c")
        paned.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        # --- left: console ---
        con_frm = ttk.LabelFrame(paned, text="Console")
        paned.add(con_frm, stretch="always")

        btn_bar = ttk.Frame(con_frm)
        btn_bar.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Button(btn_bar, text="Clear", command=self._clear_console).pack(side="right")

        self._console = tk.Text(con_frm, height=16, state="disabled",
                                wrap="word", font=("Consolas", 9),
                                bg="#1e1e1e", fg="#d4d4d4",
                                insertbackground="white",
                                relief="flat", bd=0)
        vsb = ttk.Scrollbar(con_frm, orient="vertical",
                             command=self._console.yview)
        self._console.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y", padx=(0, 4), pady=4)
        self._console.pack(fill="both", expand=True, padx=(6, 0), pady=4)

        # --- right: train surrogate NN ---
        nn_frm = ttk.LabelFrame(paned, text="3 · Train Surrogate Neural Network")
        paned.add(nn_frm, stretch="always")

        # Hyperparameter fields
        hp_frm = ttk.Frame(nn_frm)
        hp_frm.pack(fill="x", padx=8, pady=(6, 0))

        r = ttk.Frame(hp_frm); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Epochs:", width=18, anchor="w").pack(side="left")
        self._nn_epochs_var = tk.StringVar(value="800")
        ttk.Entry(r, textvariable=self._nn_epochs_var, width=8).pack(side="left", padx=4)

        r = ttk.Frame(hp_frm); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Batch size:", width=18, anchor="w").pack(side="left")
        self._nn_batch_var = tk.StringVar(value="128")
        ttk.Entry(r, textvariable=self._nn_batch_var, width=8).pack(side="left", padx=4)

        r = ttk.Frame(hp_frm); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Learning rate:", width=18, anchor="w").pack(side="left")
        self._nn_lr_var = tk.StringVar(value="0.001")
        ttk.Entry(r, textvariable=self._nn_lr_var, width=8).pack(side="left", padx=4)

        r = ttk.Frame(hp_frm); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Val split:", width=18, anchor="w").pack(side="left")
        self._nn_val_var = tk.StringVar(value="0.20")
        ttk.Entry(r, textvariable=self._nn_val_var, width=8).pack(side="left", padx=4)

        # Separator
        ttk.Separator(nn_frm, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # Data source indicator
        ds_row = ttk.Frame(nn_frm); ds_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(ds_row, text="Data source:", anchor="w").pack(side="left")
        self._nn_data_status = tk.StringVar(value="checking…")
        ttk.Label(ds_row, textvariable=self._nn_data_status,
                  foreground="#80c8ff", font=("Consolas", 8)).pack(side="left", padx=6)

        # Train button + status
        btn_row = ttk.Frame(nn_frm); btn_row.pack(fill="x", padx=8, pady=(0, 4))
        self._train_btn = ttk.Button(btn_row, text="Start Training",
                                     command=self._on_train_start)
        self._train_btn.pack(side="left")
        self._stop_train_btn = ttk.Button(btn_row, text="Stop",
                                          command=self._on_train_stop,
                                          state="disabled")
        self._stop_train_btn.pack(side="left", padx=(6, 0))
        self._train_status = tk.StringVar(value="—")
        ttk.Label(btn_row, textvariable=self._train_status,
                  foreground="gray").pack(side="left", padx=8)

        # Epoch progress label (updated by log parsing)
        ep_row = ttk.Frame(nn_frm); ep_row.pack(fill="x", padx=8, pady=(0, 2))
        self._train_epoch_var = tk.StringVar(value="")
        ttk.Label(ep_row, textvariable=self._train_epoch_var,
                  foreground="#4ec94e", font=("Consolas", 8)).pack(side="left")

        # Mini log (train-only output)
        ttk.Separator(nn_frm, orient="horizontal").pack(fill="x", padx=8, pady=(4, 2))
        log_hdr = ttk.Frame(nn_frm); log_hdr.pack(fill="x", padx=6, pady=(2, 0))
        ttk.Label(log_hdr, text="Training log", foreground="gray",
                  font=("Consolas", 8)).pack(side="left")
        ttk.Button(log_hdr, text="Clear",
                   command=self._clear_train_log).pack(side="right")

        self._train_log = tk.Text(nn_frm, height=8, state="disabled",
                                  wrap="word", font=("Consolas", 8),
                                  bg="#1e1e1e", fg="#d4d4d4",
                                  insertbackground="white",
                                  relief="flat", bd=0)
        tvsb = ttk.Scrollbar(nn_frm, orient="vertical",
                             command=self._train_log.yview)
        self._train_log.configure(yscrollcommand=tvsb.set)
        tvsb.pack(side="right", fill="y", padx=(0, 4), pady=4)
        self._train_log.pack(fill="both", expand=True, padx=(6, 0), pady=4)

        # Artifact status at the very bottom
        art_row = ttk.Frame(nn_frm); art_row.pack(fill="x", padx=8, pady=(0, 6))
        self._artifact_status = tk.StringVar(value="")
        ttk.Label(art_row, textvariable=self._artifact_status,
                  foreground="#80c8ff", font=("Consolas", 8),
                  wraplength=260, justify="left").pack(side="left")

        # Set the sash position after window is rendered
        self.after(100, lambda: paned.sash_place(0, self.winfo_width() // 2, 0))

        self._refresh_nn_status()

    # ------------------------------------------------------------------ log
    def _log(self, text: str):
        self._log_queue.put(text)

    def _flush_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._console.configure(state="normal")
                self._console.insert("end", msg + "\n")
                self._console.see("end")
                self._console.configure(state="disabled")
        except queue.Empty:
            pass

    def _clear_console(self):
        self._console.configure(state="normal")
        self._console.delete("1.0", "end")
        self._console.configure(state="disabled")

    # ------------------------------------------------------------------ poll
    def _poll(self):
        self._flush_log()
        self._flush_train_log()
        self._check_sweep_done()
        self._check_train_done()
        self.after(POLL_MS, self._poll)

    def _check_sweep_done(self):
        if self._sweep_proc is not None and self._sweep_proc.poll() is not None:
            rc = self._sweep_proc.returncode
            self._sweep_proc  = None
            self._sweep_thread = None
            if rc == 0:
                self._sweep_status.set("Finished")
                self._log("[sweep] Process exited OK.")
            else:
                self._sweep_status.set(f"Exited (code {rc})")
                self._log(f"[sweep] Process exited with code {rc}.")
            self._sweep_btn.config(state="normal")
            self._pause_btn.config(state="disabled")
            self._refresh_status()

    # ------------------------------------------------------------------ status
    def _refresh_status(self):
        os.makedirs(_SIMDATA_DIR, exist_ok=True)

        # Samples file → total sample count
        n_total_samples = 0
        if os.path.exists(_SAMPLES_FILE):
            try:
                with open(_SAMPLES_FILE) as f:
                    d = json.load(f)
                samples = d.get("samples", [])
                n_total_samples = len(samples)
                self._gen_status.set(f"{n_total_samples} samples on disk")
                self._stat_total.set(f"Total: {n_total_samples}")
                # Build tag->index map for coverage calculation
                self._tag_to_idx = {s["tag"]: i for i, s in enumerate(samples)
                                    if "tag" in s}
            except Exception:
                self._gen_status.set("samples file unreadable")
                self._tag_to_idx = {}
        else:
            self._gen_status.set("no samples file")
            self._stat_total.set("Total: —")
            self._tag_to_idx = {}

        self._n_total_samples = n_total_samples

        # Results file
        n_done = 0
        if os.path.exists(_RESULTS_FILE):
            try:
                with open(_RESULTS_FILE) as f:
                    d = json.load(f)
                results = d.get("results", [])
                meta    = d.get("meta", {})
                n_done  = len(results)
                n_range = meta.get("total", n_done)   # samples in the run range
                n_ok    = sum(1 for r in results if r.get("ok"))
                n_fail  = n_done - n_ok
                pct     = int(100 * n_done / n_range) if n_range else 0
                self._stat_done.set(f"Done: {n_done}/{n_range}  ({pct}%)")
                self._stat_ok.set(f"OK: {n_ok}")
                self._stat_fail.set(f"Failed: {n_fail}")
                if self._sweep_proc is None:
                    self._sweep_status.set(
                        "Complete" if n_done >= n_range else f"Paused at {pct}%")

                # Compute which global indices are covered
                self._update_coverage(results)
            except Exception:
                self._sweep_status.set("results file unreadable")
                self._coverage_var.set("Coverage: (unreadable)")
        else:
            self._stat_done.set("Done: —")
            self._stat_ok.set("OK: —")
            self._stat_fail.set("Failed: —")
            self._coverage_var.set("Coverage: none")

        self._redraw_progress(n_done=n_done)
        self._refresh_nn_status()

    def _update_coverage(self, results: list):
        """Compute contiguous covered index ranges and update label + canvas data."""
        tag_map = getattr(self, "_tag_to_idx", {})
        if not tag_map or not results:
            self._coverage_var.set("Coverage: —")
            self._covered_indices = set()
            self._redraw_progress()
            return

        indices = sorted(tag_map[r["tag"]] for r in results if r.get("tag") in tag_map)
        self._covered_indices = set(indices)

        if not indices:
            self._coverage_var.set("Coverage: —")
            return

        # Merge into contiguous runs
        runs = []
        lo = hi = indices[0]
        for idx in indices[1:]:
            if idx == hi + 1:
                hi = idx
            else:
                runs.append((lo, hi))
                lo = hi = idx
        runs.append((lo, hi))

        parts = "  |  ".join(f"{lo}-{hi}" for lo, hi in runs)
        total_covered = len(indices)
        n_total = self._n_total_samples
        pct = f"{100*total_covered/n_total:.1f}%" if n_total else ""
        self._coverage_var.set(f"Coverage ({total_covered}/{n_total} {pct}):  {parts}")
        self._redraw_progress()

    def _redraw_progress(self, n_done: int = None):
        """Draw the segmented canvas bar."""
        c = self._prog_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 2:
            return

        n_total = self._n_total_samples
        if n_total == 0:
            return

        # Parse current from/to from UI fields
        try:
            run_from = int(self._from_var.get())
        except ValueError:
            run_from = 0
        try:
            run_to = int(self._to_var.get())
        except ValueError:
            run_to = n_total

        run_from = max(0, min(run_from, n_total))
        run_to   = max(run_from, min(run_to, n_total))

        def _x(idx):
            return int(w * idx / n_total)

        # Background (unselected region) — dark gray already from canvas bg
        # Active range — slightly lighter trough
        x0, x1 = _x(run_from), _x(run_to)
        if x1 > x0:
            c.create_rectangle(x0, 0, x1, h, fill="#555555", outline="")

        # Covered indices inside the active range — bright green
        covered = getattr(self, "_covered_indices", set())
        # Draw covered pixels in green (batch by contiguous runs for speed)
        if covered and n_total:
            in_range = sorted(i for i in covered if run_from <= i < run_to)
            if in_range:
                lo = hi = in_range[0]
                for idx in in_range[1:]:
                    if idx == hi + 1:
                        hi = idx
                    else:
                        c.create_rectangle(_x(lo), 1, _x(hi + 1), h - 1,
                                           fill="#4ec94e", outline="")
                        lo = hi = idx
                c.create_rectangle(_x(lo), 1, _x(hi + 1), h - 1,
                                   fill="#4ec94e", outline="")

            # Covered outside range — dimmer green
            outside = sorted(i for i in covered if not (run_from <= i < run_to))
            if outside:
                lo = hi = outside[0]
                for idx in outside[1:]:
                    if idx == hi + 1:
                        hi = idx
                    else:
                        c.create_rectangle(_x(lo), 1, _x(hi + 1), h - 1,
                                           fill="#2a6e2a", outline="")
                        lo = hi = idx
                c.create_rectangle(_x(lo), 1, _x(hi + 1), h - 1,
                                   fill="#2a6e2a", outline="")

        # Range boundary tick marks
        if x0 > 0:
            c.create_line(x0, 0, x0, h, fill="#ffcc44", width=2)
        if x1 < w:
            c.create_line(x1, 0, x1, h, fill="#ffcc44", width=2)

    # ------------------------------------------------------------------ generate
    def _on_generate(self):
        if self._gen_thread and self._gen_thread.is_alive():
            return
        try:
            n    = int(self._n_var.get())
            seed = int(self._seed_var.get())
        except ValueError:
            self._log("[gen] ERROR: n and seed must be integers.")
            return

        self._gen_btn.config(state="disabled")
        self._gen_status.set("Running…")
        self._log(f"[gen] Generating {n} samples/branch, seed={seed} …")

        def _run():
            try:
                cmd = [sys.executable, _GEN_SCRIPT,
                       "--n",    str(n),
                       "--seed", str(seed),
                       "--out",  _SAMPLES_FILE]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=_MODULES_DIR,
                )
                for line in proc.stdout:
                    self._log(line.rstrip())
                proc.wait()
                if proc.returncode == 0:
                    self._log("[gen] Done.")
                    self._gen_status.set("Done")
                else:
                    self._log(f"[gen] ERROR: exit code {proc.returncode}")
                    self._gen_status.set(f"Error (code {proc.returncode})")
            except Exception as exc:
                self._log(f"[gen] EXCEPTION: {exc}")
                self._gen_status.set("Error")
            finally:
                self.after(0, lambda: self._gen_btn.config(state="normal"))
                self.after(0, self._refresh_status)

        self._gen_thread = threading.Thread(target=_run, daemon=True)
        self._gen_thread.start()

    # ------------------------------------------------------------------ sweep
    def _on_sweep_start(self):
        if self._sweep_proc is not None:
            return
        if not os.path.exists(_SAMPLES_FILE):
            self._log("[sweep] ERROR: No samples file found. Generate samples first.")
            return
        try:
            workers  = int(self._workers_var.get())
            timeout  = int(self._timeout_var.get())
            ckpt     = int(self._ckpt_var.get())
            from_idx = int(self._from_var.get()) if self._from_var.get().strip() else 0
            to_idx   = int(self._to_var.get())   if self._to_var.get().strip()   else None
        except ValueError:
            self._log("[sweep] ERROR: workers / timeout / checkpoint / from / to must be integers.")
            return

        # Remove any stale stop flag.
        if os.path.exists(_STOP_FLAG):
            os.remove(_STOP_FLAG)

        cmd = [sys.executable, _SWEEP_SCRIPT,
               "--samples",          _SAMPLES_FILE,
               "--out",              _RESULTS_FILE,
               "--workers",          str(workers),
               "--timeout",          str(timeout),
               "--checkpoint-every", str(ckpt),
               "--from-idx",         str(from_idx)]
        if to_idx is not None:
            cmd += ["--to-idx", str(to_idx)]

        range_str = f"{from_idx} -> {to_idx if to_idx is not None else 'end'}"
        self._log(f"[sweep] Starting: workers={workers}, timeout={timeout}s, "
                  f"ckpt={ckpt}, range={range_str}")

        try:
            self._sweep_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=_PROJECT_ROOT,
            )
        except Exception as exc:
            self._log(f"[sweep] Failed to launch: {exc}")
            return

        self._sweep_status.set("Running…")
        self._sweep_btn.config(state="disabled")
        self._pause_btn.config(state="normal")

        def _drain():
            for line in self._sweep_proc.stdout:
                stripped = line.rstrip()
                self._log(stripped)
                if "[checkpoint]" in stripped:
                    self.after(0, self._on_checkpoint)

        self._sweep_thread = threading.Thread(target=_drain, daemon=True)
        self._sweep_thread.start()

    def _on_checkpoint(self):
        self._refresh_status()
        if not os.path.exists(_RESULTS_FILE):
            return
        try:
            with open(_RESULTS_FILE) as f:
                data = json.load(f)
            results = [r for r in data.get("results", []) if r.get("ok")]
            if not results:
                return
            best = max(results, key=lambda r: r.get("k", 0.0))
            k    = best.get("k", 0.0)
            fom  = k * (best.get("Q_tx", 0.0) * best.get("Q_rx", 0.0)) ** 0.5
            self._log(
                f"[best-k] tag={best.get('tag','')}  "
                f"k={k:.4f}  FOM=k*sqrt(Q_tx*Q_rx)={fom:.2f}  |  "
                f"TX: {best.get('tx_turns','')}t  w={best.get('tx_width',''):.3f}mm  "
                f"OD={best.get('tx_od_mm',''):.1f}mm  "
                f"L={best.get('L_tx_uH',0):.2f}uH  Q={best.get('Q_tx',0):.1f}  |  "
                f"RX: {best.get('rx_turns','')}t  w={best.get('rx_width',''):.3f}mm  "
                f"OD={best.get('rx_od_mm',''):.1f}mm  "
                f"L={best.get('L_rx_uH',0):.2f}uH  Q={best.get('Q_rx',0):.1f}  "
                f"topo={best.get('rx_topology','')}  M={best.get('M_uH',0):.3f}uH"
            )
        except Exception as exc:
            self._log(f"[best-k] scan failed: {exc}")

    def _on_pause(self):
        if self._sweep_proc is None:
            return
        if os.path.exists(_STOP_FLAG):
            # Already pausing — cancel the pause request.
            try:
                os.remove(_STOP_FLAG)
            except OSError:
                pass
            self._pause_btn.config(text="Pause")
            self._sweep_status.set("Running…")
            self._log("[sweep] Pause cancelled.")
        else:
            open(_STOP_FLAG, "w").close()
            self._pause_btn.config(text="Cancel Pause")
            self._sweep_status.set("Pausing after batch…")
            self._log("[sweep] Stop flag set — will pause after current batch.")

    # ------------------------------------------------------------------ train
    def _log_train(self, text: str):
        self._train_log_queue.put(text)

    def _flush_train_log(self):
        try:
            while True:
                msg = self._train_log_queue.get_nowait()
                self._train_log.configure(state="normal")
                self._train_log.insert("end", msg + "\n")
                self._train_log.see("end")
                self._train_log.configure(state="disabled")
                # Parse "Epoch  800/800  |  Train MSE: …  |  Val MSE: …"
                if msg.startswith("Epoch"):
                    self._train_epoch_var.set(msg.strip())
        except queue.Empty:
            pass

    def _clear_train_log(self):
        self._train_log.configure(state="normal")
        self._train_log.delete("1.0", "end")
        self._train_log.configure(state="disabled")

    def _refresh_nn_status(self):
        """Update the data-source and artifact indicators."""
        if os.path.exists(_RESULTS_FILE):
            try:
                with open(_RESULTS_FILE) as f:
                    d = json.load(f)
                n = sum(1 for r in d.get("results", []) if r.get("ok"))
                self._nn_data_status.set(f"{n} OK simulations in sweep_results.json")
            except Exception:
                self._nn_data_status.set("sweep_results.json unreadable")
        else:
            self._nn_data_status.set("No sweep_results.json found — run sweep first")

        parts = []
        if os.path.exists(_MODEL_FILE):
            parts.append("model.pth ✓")
        if os.path.exists(os.path.join(_NN_DIR, "x_scaler.pkl")):
            parts.append("x_scaler ✓")
        if os.path.exists(os.path.join(_NN_DIR, "y_scaler.pkl")):
            parts.append("y_scaler ✓")
        if os.path.exists(_LOSS_PLOT):
            parts.append("loss_curve.png ✓")
        self._artifact_status.set("  ".join(parts) if parts else "No trained model yet")

    def _check_train_done(self):
        if self._train_proc is not None and self._train_proc.poll() is not None:
            rc = self._train_proc.returncode
            self._train_proc   = None
            self._train_thread = None
            if rc == 0:
                self._train_status.set("Done")
                self._log_train("[train] Training finished successfully.")
                self._log("[train] Training complete — model saved to NeuralNetwork/")
            else:
                self._train_status.set(f"Error (code {rc})")
                self._log_train(f"[train] Process exited with code {rc}.")
            self._train_btn.config(state="normal")
            self._stop_train_btn.config(state="disabled")
            self._refresh_nn_status()

    def _on_train_start(self):
        if self._train_proc is not None:
            return
        if not os.path.exists(_RESULTS_FILE):
            self._log_train("[train] ERROR: sweep_results.json not found. Run the sweep first.")
            return
        if not os.path.exists(_TRAIN_SCRIPT):
            self._log_train(f"[train] ERROR: train_surrogate.py not found at {_TRAIN_SCRIPT}")
            return

        # Validate hyperparameter fields
        try:
            epochs = int(self._nn_epochs_var.get())
            batch  = int(self._nn_batch_var.get())
            lr     = float(self._nn_lr_var.get())
            val    = float(self._nn_val_var.get())
            assert 0 < val < 1
        except (ValueError, AssertionError):
            self._log_train("[train] ERROR: invalid hyperparameter values.")
            return

        self._train_btn.config(state="disabled")
        self._stop_train_btn.config(state="normal")
        self._train_status.set("Running…")
        self._train_epoch_var.set("")
        self._log_train(
            f"[train] Starting — epochs={epochs}  batch={batch}  lr={lr}  val={val}"
        )
        self._log(f"[train] Launched train_surrogate.py  (epochs={epochs}, batch={batch})")

        env = os.environ.copy()
        env["SURROGATE_EPOCHS"]     = str(epochs)
        env["SURROGATE_BATCH_SIZE"] = str(batch)
        env["SURROGATE_LR"]         = str(lr)
        env["SURROGATE_VAL_SPLIT"]  = str(val)

        try:
            self._train_proc = subprocess.Popen(
                [sys.executable, "-u", _TRAIN_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=_NN_DIR,
                env=env,
            )
        except Exception as exc:
            self._log_train(f"[train] Failed to launch: {exc}")
            self._train_btn.config(state="normal")
            self._stop_train_btn.config(state="disabled")
            self._train_status.set("Error")
            return

        def _drain():
            for line in self._train_proc.stdout:
                self._log_train(line.rstrip())

        self._train_thread = threading.Thread(target=_drain, daemon=True)
        self._train_thread.start()

    def _on_train_stop(self):
        if self._train_proc is None:
            return
        try:
            self._train_proc.terminate()
        except Exception:
            pass
        self._train_status.set("Stopped")
        self._log_train("[train] Training stopped by user.")
        self._train_btn.config(state="normal")
        self._stop_train_btn.config(state="disabled")
