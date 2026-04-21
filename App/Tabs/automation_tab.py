#!/usr/bin/env python3
"""
AutomationTab — GUI front-end for the LHS sweep pipeline.

Stages:
  1. Generate Samples  — runs generate_lhs_samples.py in a background thread
  2. Run Sweep         — runs run_sweep.py in a background subprocess
  3. Pause / Resume    — writes/removes the STOP_SWEEP sentinel file
"""

import os
import sys
import json
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk

_HERE        = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT    = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_APP_ROOT)
_SIMDATA_DIR = os.path.join(_APP_ROOT, "SimulationData")
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")
_GEN_SCRIPT  = os.path.join(_PROJECT_ROOT, "generate_lhs_samples.py")
_SWEEP_SCRIPT = os.path.join(_PROJECT_ROOT, "run_sweep.py")
_SAMPLES_FILE = os.path.join(_SIMDATA_DIR, "lhs_samples.json")
_RESULTS_FILE = os.path.join(_SIMDATA_DIR, "sweep_results.json")
_STOP_FLAG   = os.path.join(_SIMDATA_DIR, "STOP_SWEEP")

for _p in (_MODULES_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POLL_MS = 400


class AutomationTab(ttk.Frame):
    def __init__(self, parent, app=None, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self._log_queue  = queue.Queue()
        self._gen_thread = None   # thread for sample generation
        self._sweep_proc = None   # subprocess for sweep
        self._sweep_thread = None # thread draining sweep stdout
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

        self._progress = ttk.Progressbar(prog_frm, mode="determinate",
                                          maximum=100, value=0)
        self._progress.pack(fill="x", padx=8, pady=(6, 2))

        stats_row = ttk.Frame(prog_frm)
        stats_row.pack(fill="x", padx=8, pady=(0, 6))
        self._stat_done  = tk.StringVar(value="Done: —")
        self._stat_ok    = tk.StringVar(value="OK: —")
        self._stat_fail  = tk.StringVar(value="Failed: —")
        self._stat_total = tk.StringVar(value="Total: —")
        for sv in (self._stat_total, self._stat_done,
                   self._stat_ok, self._stat_fail):
            ttk.Label(stats_row, textvariable=sv).pack(side="left", padx=12)

        ttk.Button(prog_frm, text="Refresh",
                   command=self._refresh_status).pack(anchor="e", padx=8, pady=(0, 4))

        # ---- console output ---------------------------------------------
        con_frm = ttk.LabelFrame(self, text="Console")
        con_frm.pack(fill="both", expand=True, padx=10, pady=(4, 10))

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
        self._check_sweep_done()
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

        # Samples file
        if os.path.exists(_SAMPLES_FILE):
            try:
                with open(_SAMPLES_FILE) as f:
                    d = json.load(f)
                n = len(d.get("samples", []))
                self._gen_status.set(f"{n} samples on disk")
                self._stat_total.set(f"Total: {n}")
            except Exception:
                self._gen_status.set("samples file unreadable")
        else:
            self._gen_status.set("no samples file")
            self._stat_total.set("Total: —")

        # Results file
        if os.path.exists(_RESULTS_FILE):
            try:
                with open(_RESULTS_FILE) as f:
                    d = json.load(f)
                results = d.get("results", [])
                meta    = d.get("meta", {})
                n_done  = len(results)
                n_total = meta.get("total", n_done)
                n_ok    = sum(1 for r in results if r.get("ok"))
                n_fail  = n_done - n_ok
                pct     = int(100 * n_done / n_total) if n_total else 0
                self._progress["value"] = pct
                self._stat_done.set(f"Done: {n_done}/{n_total}  ({pct}%)")
                self._stat_ok.set(f"OK: {n_ok}")
                self._stat_fail.set(f"Failed: {n_fail}")
                if self._sweep_proc is None:
                    self._sweep_status.set(
                        "Complete" if n_done >= n_total else f"Paused at {pct}%")
            except Exception:
                self._sweep_status.set("results file unreadable")
        else:
            self._progress["value"] = 0
            self._stat_done.set("Done: —")
            self._stat_ok.set("OK: —")
            self._stat_fail.set("Failed: —")

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
                    cwd=_PROJECT_ROOT,
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
            workers = int(self._workers_var.get())
            timeout = int(self._timeout_var.get())
            ckpt    = int(self._ckpt_var.get())
        except ValueError:
            self._log("[sweep] ERROR: workers / timeout / checkpoint must be integers.")
            return

        # Remove any stale stop flag.
        if os.path.exists(_STOP_FLAG):
            os.remove(_STOP_FLAG)

        cmd = [sys.executable, _SWEEP_SCRIPT,
               "--samples",          _SAMPLES_FILE,
               "--out",              _RESULTS_FILE,
               "--workers",          str(workers),
               "--timeout",          str(timeout),
               "--checkpoint-every", str(ckpt)]

        self._log(f"[sweep] Starting: workers={workers}, timeout={timeout}s, "
                  f"checkpoint_every={ckpt}")

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
                self._log(line.rstrip())

        self._sweep_thread = threading.Thread(target=_drain, daemon=True)
        self._sweep_thread.start()

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
