#!/usr/bin/env python3
"""
AutomationTab — GUI front-end for the LHS sweep pipeline + NN training.

Stages:
  1. Configure & Generate LHS Samples
  2. Run FastHenry Sweep (solver_output.json)
  3. Append to Global Progress (global_results.json)
  4. Train Surrogate NN
"""

import os
import sys
import json
import queue
import threading
import subprocess
import glob
import tkinter as tk
from tkinter import ttk

_HERE         = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT     = os.path.dirname(_HERE)
_SIMDATA_DIR  = os.path.join(_APP_ROOT, "SimulationData")
_NN_DIR       = os.path.join(_APP_ROOT, "NeuralNetwork")
_MODULES_DIR  = os.path.join(_APP_ROOT, "Modules")

_GEN_SCRIPT    = os.path.join(_NN_DIR, "generate_lhs_samples.py")
_SWEEP_SCRIPT  = os.path.join(_NN_DIR, "run_sweep.py")
_APPEND_SCRIPT = os.path.join(_NN_DIR, "append_results.py")
_TRAIN_SCRIPT  = os.path.join(_NN_DIR, "train_surrogate.py")

_DOMAIN_FILE  = os.path.join(_SIMDATA_DIR, "domain_master.json")
_GLOBAL_FILE  = os.path.join(_SIMDATA_DIR, "global_results.json")
_STOP_FLAG    = os.path.join(_SIMDATA_DIR, "STOP_SWEEP")
_MODEL_FILE   = os.path.join(_NN_DIR, "surrogate_model.pth")
_LOSS_PLOT    = os.path.join(_NN_DIR, "loss_curve.png")

def _batch_samples_path(n: int) -> str:
    return os.path.join(_SIMDATA_DIR, f"lhs_batch_{n}.json")

def _batch_results_path(n: int) -> str:
    return os.path.join(_SIMDATA_DIR, f"results_batch_{n}.json")

for _p in (_NN_DIR, _MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POLL_MS = 400

_LAYER_LABELS = ["L1 (outer)", "L2 (inner)", "L3 (inner)", "L4 (outer)"]

# Fixed pixel width for TX and RX panels
_SIDE_WIDTH = 260

# ---------------------------------------------------------------------------
# Default values for all input fields  — edit here to change startup defaults
# ---------------------------------------------------------------------------

# TX coil
TX_LAYERS_DEFAULT        = [True, True, True, False]   # L1, L2, L3, L4 active
TX_OUTER_OZ_LO           = "1.0"
TX_OUTER_OZ_HI           = "1.0"
TX_INNER_OZ_LO           = "0.5"
TX_INNER_OZ_HI           = "1.0"
TX_OUTER_GAP_LO          = "0.2"
TX_OUTER_GAP_HI          = "0.2"
TX_INNER_GAP_LO          = "0.8"
TX_INNER_GAP_HI          = "1.3"
TX_ID_MIN                = "20.0"
TX_OD_MAX                = "56.0"
TX_SPACING_LO            = "0.16"
TX_SPACING_HI            = "0.16"
TX_PORT_INSIDE_DEFAULT   = False
TX_PORT_OUTSIDE_DEFAULT  = True

# RX coil
RX_LAYERS_DEFAULT        = [True, True, True, True]
RX_OUTER_OZ_LO           = "1.0"
RX_OUTER_OZ_HI           = "1.0"
RX_INNER_OZ_LO           = "0.5"
RX_INNER_OZ_HI           = "1.0"
RX_OUTER_GAP_LO          = "0.2"
RX_OUTER_GAP_HI          = "0.2"
RX_INNER_GAP_LO          = "0.6"
RX_INNER_GAP_HI          = "1.0"
RX_ID_MIN                = "20.0"
RX_OD_MAX                = "55.0"
RX_SPACING_LO            = "0.16"
RX_SPACING_HI            = "0.16"
RX_PORT_INSIDE_DEFAULT   = True
RX_PORT_OUTSIDE_DEFAULT  = False

# Shared / global
RESOLUTION_MM            = "1.5"
NHINC                    = "1"
NWINC                    = "3"
PCB_GAP_LO               = "2.4"
PCB_GAP_HI               = "2.8"
FREQ_MIN_HZ              = "110000"
FREQ_MAX_HZ              = "140000"

# Sample generation
TOTAL_SAMPLES            = "64000"

# FastHenry sweep
SWEEP_WORKERS            = "6"
SWEEP_TIMEOUT_S          = "360"
SWEEP_CKPT_EVERY         = "50"
SWEEP_FROM_IDX           = "0"
SWEEP_TO_IDX             = ""

# NN training  (tuned for large dataset ~30k–60k rows)
NN_EPOCHS                = "300"
NN_BATCH                 = "512"
NN_LR                    = "0.0005"
NN_VAL_SPLIT             = "0.15"


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

def _range_row(parent, label, lo_var, hi_var, label_width=14, entry_width=6, pady=2):
    r = ttk.Frame(parent)
    r.pack(fill="x", padx=6, pady=pady)
    ttk.Label(r, text=label, width=label_width, anchor="w").pack(side="left")
    ttk.Entry(r, textvariable=lo_var, width=entry_width).pack(side="left", padx=(4, 2))
    ttk.Label(r, text="–", foreground="gray").pack(side="left")
    ttk.Entry(r, textvariable=hi_var, width=entry_width).pack(side="left", padx=(2, 0))
    return r


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

class AutomationTab(ttk.Frame):
    def __init__(self, parent, app=None, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self._log_queue   = queue.Queue()
        self._gen_thread  = None
        self._sweep_proc  = None
        self._sweep_thread = None
        self._train_proc  = None
        self._train_thread = None
        self._append_thread = None
        self._covered_global = set()
        self._n_target       = 0
        self._n_total_samples = 0
        self._active_batch_num = None
        self._batch_var = tk.StringVar()
        self._build()
        self._refresh_status()
        self.after(POLL_MS, self._poll)

    # ===================================================================== build
    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        # ---- scrollable canvas for the top 3-column config section ----
        cfg_canvas = tk.Canvas(outer, highlightthickness=0, bg="#2d2d2d")
        cfg_vsb    = ttk.Scrollbar(outer, orient="vertical", command=cfg_canvas.yview)
        cfg_canvas.configure(yscrollcommand=cfg_vsb.set)

        cfg_vsb.pack(side="right", fill="y")
        cfg_canvas.pack(side="top", fill="x")

        cfg_inner = ttk.Frame(cfg_canvas)
        _win_id = cfg_canvas.create_window((0, 0), window=cfg_inner, anchor="nw")

        def _on_cfg_resize(event):
            cfg_canvas.itemconfig(_win_id, width=event.width)

        def _on_cfg_frame(_):
            cfg_canvas.configure(scrollregion=cfg_canvas.bbox("all"))
            h = min(cfg_inner.winfo_reqheight(), 520)
            cfg_canvas.configure(height=h)

        cfg_canvas.bind("<Configure>", _on_cfg_resize)
        cfg_inner.bind("<Configure>", _on_cfg_frame)

        self._build_config(cfg_inner)

        # ---- progress bars + NN trainer below the scrollable section ----
        bottom = ttk.Frame(outer)
        bottom.pack(fill="both", expand=True, padx=0, pady=0)
        self._build_bottom(bottom)

    def _build_config(self, parent):
        """Three-column layout: TX (fixed) | RX (fixed) | right column."""

        cols = ttk.Frame(parent)
        cols.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        # TX column — fixed width
        tx_frm = ttk.LabelFrame(cols, text="TX Coil", width=_SIDE_WIDTH)
        tx_frm.pack(side="left", fill="y", padx=(0, 2), anchor="n")
        tx_frm.pack_propagate(False)

        # Separator
        tk.Frame(cols, bg="#606060", width=2).pack(side="left", fill="y", padx=2)

        # RX column — fixed width
        rx_frm = ttk.LabelFrame(cols, text="RX Coil", width=_SIDE_WIDTH)
        rx_frm.pack(side="left", fill="y", padx=(2, 4), anchor="n")
        rx_frm.pack_propagate(False)

        # Right column — expands
        right_col = ttk.Frame(cols)
        right_col.pack(side="left", fill="both", expand=True, anchor="n")

        self._tx = self._build_side(tx_frm, "tx")
        self._rx = self._build_side(rx_frm, "rx")
        self._build_right_column(right_col)

    def _build_side(self, parent, side: str) -> dict:
        """Build TX or RX config panel. Returns dict of control variables."""
        v = {}

        # Layer checkboxes
        lyr_frm = ttk.LabelFrame(parent, text="Active Layers")
        lyr_frm.pack(fill="x", padx=4, pady=(4, 2))

        _layers_def  = TX_LAYERS_DEFAULT        if side == "tx" else RX_LAYERS_DEFAULT
        _oo_lo       = TX_OUTER_OZ_LO           if side == "tx" else RX_OUTER_OZ_LO
        _oo_hi       = TX_OUTER_OZ_HI           if side == "tx" else RX_OUTER_OZ_HI
        _io_lo       = TX_INNER_OZ_LO           if side == "tx" else RX_INNER_OZ_LO
        _io_hi       = TX_INNER_OZ_HI           if side == "tx" else RX_INNER_OZ_HI
        _og_lo       = TX_OUTER_GAP_LO          if side == "tx" else RX_OUTER_GAP_LO
        _og_hi       = TX_OUTER_GAP_HI          if side == "tx" else RX_OUTER_GAP_HI
        _ig_lo       = TX_INNER_GAP_LO          if side == "tx" else RX_INNER_GAP_LO
        _ig_hi       = TX_INNER_GAP_HI          if side == "tx" else RX_INNER_GAP_HI
        _id_min      = TX_ID_MIN                if side == "tx" else RX_ID_MIN
        _od_max      = TX_OD_MAX                if side == "tx" else RX_OD_MAX
        _sp_lo       = TX_SPACING_LO            if side == "tx" else RX_SPACING_LO
        _sp_hi       = TX_SPACING_HI            if side == "tx" else RX_SPACING_HI
        _p_in_def    = TX_PORT_INSIDE_DEFAULT   if side == "tx" else RX_PORT_INSIDE_DEFAULT
        _p_out_def   = TX_PORT_OUTSIDE_DEFAULT  if side == "tx" else RX_PORT_OUTSIDE_DEFAULT

        v["layers"] = []
        for i, lbl in enumerate(_LAYER_LABELS):
            bv = tk.BooleanVar(value=_layers_def[i])
            ttk.Checkbutton(lyr_frm, text=lbl, variable=bv).pack(anchor="w", padx=6)
            v["layers"].append(bv)

        # Copper weight ranges
        cu_frm = ttk.LabelFrame(parent, text="Copper Weight (oz)")
        cu_frm.pack(fill="x", padx=4, pady=2)

        v["outer_oz_lo"] = tk.StringVar(value=_oo_lo)
        v["outer_oz_hi"] = tk.StringVar(value=_oo_hi)
        v["inner_oz_lo"] = tk.StringVar(value=_io_lo)
        v["inner_oz_hi"] = tk.StringVar(value=_io_hi)
        _range_row(cu_frm, "Outer layers:", v["outer_oz_lo"], v["outer_oz_hi"])
        _range_row(cu_frm, "Inner layers:", v["inner_oz_lo"], v["inner_oz_hi"])

        # Stackup spacing ranges
        stk_frm = ttk.LabelFrame(parent, text="Stackup Spacing (mm)")
        stk_frm.pack(fill="x", padx=4, pady=2)

        v["outer_gap_lo"] = tk.StringVar(value=_og_lo)
        v["outer_gap_hi"] = tk.StringVar(value=_og_hi)
        v["inner_gap_lo"] = tk.StringVar(value=_ig_lo)
        v["inner_gap_hi"] = tk.StringVar(value=_ig_hi)
        _range_row(stk_frm, "Outer gap:", v["outer_gap_lo"], v["outer_gap_hi"])
        _range_row(stk_frm, "Inner gap:", v["inner_gap_lo"], v["inner_gap_hi"])

        # Geometry ranges
        geo_frm = ttk.LabelFrame(parent, text="Geometry")
        geo_frm.pack(fill="x", padx=4, pady=2)

        v["id_min"] = tk.StringVar(value=_id_min)
        v["od_max"] = tk.StringVar(value=_od_max)
        _range_row(geo_frm, "Diameter (mm):", v["id_min"], v["od_max"], label_width=14)

        v["spacing_lo"] = tk.StringVar(value=_sp_lo)
        v["spacing_hi"] = tk.StringVar(value=_sp_hi)
        _range_row(geo_frm, "Spacing (mm):", v["spacing_lo"], v["spacing_hi"], label_width=14)

        # Port selection
        port_frm = ttk.LabelFrame(parent, text="Port Location")
        port_frm.pack(fill="x", padx=4, pady=2)

        v["port_inside"]  = tk.BooleanVar(value=_p_in_def)
        v["port_outside"] = tk.BooleanVar(value=_p_out_def)

        def _guard_port(changed_var, other_var):
            def _cb(*_):
                if not changed_var.get() and not other_var.get():
                    other_var.set(True)
            return _cb

        ttk.Checkbutton(port_frm, text="Port outside",
                        variable=v["port_outside"]).pack(anchor="w", padx=6)
        ttk.Checkbutton(port_frm, text="Port inside",
                        variable=v["port_inside"]).pack(anchor="w", padx=6)

        v["port_outside"].trace_add("write",
            _guard_port(v["port_outside"], v["port_inside"]))
        v["port_inside"].trace_add("write",
            _guard_port(v["port_inside"], v["port_outside"]))

        return v

    def _build_right_column(self, parent):
        """Shared params + Generate LHS + Sweep + Console — all in the right column."""

        # ---- Shared Parameters ----
        shared = ttk.LabelFrame(parent, text="Shared Parameters")
        shared.pack(fill="x", padx=4, pady=(4, 2))

        row1 = ttk.Frame(shared)
        row1.pack(fill="x", padx=4, pady=(4, 2))

        for label, attr, default, width in [
            ("Resolution (mm):", "_res_var",   RESOLUTION_MM, 6),
            ("nhinc:",           "_nhinc_var", NHINC,         4),
            ("nwinc:",           "_nwinc_var", NWINC,         4),
        ]:
            f = ttk.Frame(row1); f.pack(side="left", padx=(0, 10))
            ttk.Label(f, text=label, anchor="w").pack(anchor="w")
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(f, textvariable=var, width=width).pack(anchor="w")

        # PCB gap as a range
        row2 = ttk.Frame(shared)
        row2.pack(fill="x", padx=4, pady=(0, 2))
        self._pcb_gap_lo_var = tk.StringVar(value=PCB_GAP_LO)
        self._pcb_gap_hi_var = tk.StringVar(value=PCB_GAP_HI)
        _range_row(row2, "PCB gap (mm):", self._pcb_gap_lo_var, self._pcb_gap_hi_var,
                   label_width=14, entry_width=6, pady=0)

        # Frequency range
        row3 = ttk.Frame(shared)
        row3.pack(fill="x", padx=4, pady=(0, 4))
        self._fmin_var = tk.StringVar(value=FREQ_MIN_HZ)
        self._fmax_var = tk.StringVar(value=FREQ_MAX_HZ)
        _range_row(row3, "Freq range (Hz):", self._fmin_var, self._fmax_var,
                   label_width=14, entry_width=10, pady=0)

        # ---- Generate LHS Samples ----
        gen_frm = ttk.LabelFrame(parent, text="1 · Generate LHS Samples")
        gen_frm.pack(fill="x", padx=4, pady=2)

        gr = ttk.Frame(gen_frm)
        gr.pack(fill="x", padx=6, pady=4)

        f = ttk.Frame(gr); f.pack(side="left", padx=(0, 10))
        ttk.Label(f, text="Total samples:", anchor="w").pack(anchor="w")
        self._n_var = tk.StringVar(value=TOTAL_SAMPLES)
        ttk.Entry(f, textvariable=self._n_var, width=8).pack(anchor="w")

        self._gen_btn = ttk.Button(gr, text="Generate Samples",
                                   command=self._on_generate)
        self._gen_btn.pack(side="left", padx=(0, 8))
        self._gen_status = tk.StringVar(value="—")
        ttk.Label(gr, textvariable=self._gen_status,
                  foreground="gray").pack(side="left")

        # ---- FastHenry Sweep ----
        sw_frm = ttk.LabelFrame(parent, text="2 · Run FastHenry Sweep")
        sw_frm.pack(fill="x", padx=4, pady=2)

        # Batch selector row
        batch_row = ttk.Frame(sw_frm)
        batch_row.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Label(batch_row, text="Batch:", width=10, anchor="w").pack(side="left")
        self._batch_combobox = ttk.Combobox(batch_row, textvariable=self._batch_var,
                                             state="readonly", width=30)
        self._batch_combobox.pack(side="left", padx=(0, 6))
        self._batch_combobox.bind("<<ComboboxSelected>>", lambda _: self._on_batch_selected())
        self._batch_status = tk.StringVar(value="—")
        ttk.Label(batch_row, textvariable=self._batch_status,
                  foreground="gray", font=("Consolas", 8)).pack(side="left")

        sw_row = ttk.Frame(sw_frm)
        sw_row.pack(fill="x", padx=6, pady=(0, 2))

        for label, var_name, default, width in [
            ("Workers:",     "_workers_var", SWEEP_WORKERS,   5),
            ("Timeout (s):", "_timeout_var", SWEEP_TIMEOUT_S, 5),
            ("Ckpt every:",  "_ckpt_var",    SWEEP_CKPT_EVERY, 5),
        ]:
            f = ttk.Frame(sw_row); f.pack(side="left", padx=(0, 10))
            ttk.Label(f, text=label, anchor="w").pack(anchor="w")
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(f, textvariable=var, width=width).pack(anchor="w")

        idx_row = ttk.Frame(sw_frm)
        idx_row.pack(fill="x", padx=6, pady=(0, 2))

        f = ttk.Frame(idx_row); f.pack(side="left", padx=(0, 10))
        ttk.Label(f, text="From idx:", anchor="w").pack(anchor="w")
        self._from_var = tk.StringVar(value=SWEEP_FROM_IDX)
        self._from_var.trace_add("write", lambda *_: self._redraw_sweep_bar())
        ttk.Entry(f, textvariable=self._from_var, width=7).pack(anchor="w")

        f = ttk.Frame(idx_row); f.pack(side="left", padx=(0, 10))
        ttk.Label(f, text="To idx:", anchor="w").pack(anchor="w")
        self._to_var = tk.StringVar(value=SWEEP_TO_IDX)
        self._to_var.trace_add("write", lambda *_: self._redraw_sweep_bar())
        ttk.Entry(f, textvariable=self._to_var, width=7).pack(anchor="w")

        btn_row = ttk.Frame(sw_frm)
        btn_row.pack(fill="x", padx=6, pady=(2, 4))
        self._sweep_btn = ttk.Button(btn_row, text="Start Solving",
                                     command=self._on_sweep_start)
        self._sweep_btn.pack(side="left")
        self._pause_btn = ttk.Button(btn_row, text="Pause",
                                     command=self._on_pause, state="disabled")
        self._pause_btn.pack(side="left", padx=(6, 0))
        self._sweep_status = tk.StringVar(value="—")
        ttk.Label(btn_row, textvariable=self._sweep_status,
                  foreground="gray").pack(side="left", padx=8)

        # ---- Console ----
        con_frm = ttk.LabelFrame(parent, text="Console")
        con_frm.pack(fill="both", expand=True, padx=4, pady=2)

        btn_bar = ttk.Frame(con_frm)
        btn_bar.pack(fill="x", padx=4, pady=(2, 0))
        ttk.Button(btn_bar, text="Clear", command=self._clear_console).pack(side="right")

        self._console = tk.Text(con_frm, height=7, state="disabled",
                                wrap="none", font=("Consolas", 8),
                                bg="#1e1e1e", fg="#d4d4d4",
                                insertbackground="white", relief="flat", bd=0)
        vsb = ttk.Scrollbar(con_frm, orient="vertical", command=self._console.yview)
        hsb = ttk.Scrollbar(con_frm, orient="horizontal", command=self._console.xview)
        self._console.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._console.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 2))

    def _build_bottom(self, parent):
        # ---- Sweep progress bar ----
        sp_frm = ttk.LabelFrame(parent, text="Sweep Progress  (current index range)")
        sp_frm.pack(fill="x", padx=10, pady=(4, 2))

        self._sweep_bar = tk.Canvas(sp_frm, height=14, bg="#3c3c3c",
                                    highlightthickness=1,
                                    highlightbackground="#606060")
        self._sweep_bar.pack(fill="x", padx=6, pady=(4, 2))
        self._sweep_bar.bind("<Configure>", lambda _: self._redraw_sweep_bar())

        sstat_row = ttk.Frame(sp_frm)
        sstat_row.pack(fill="x", padx=6, pady=(0, 4))
        self._sstat_done  = tk.StringVar(value="Done: —")
        self._sstat_ok    = tk.StringVar(value="OK: —")
        for sv in (self._sstat_done, self._sstat_ok):
            ttk.Label(sstat_row, textvariable=sv).pack(side="left", padx=10)
        ttk.Button(sstat_row, text="Refresh",
                   command=self._refresh_status).pack(side="right", padx=6)

        # ---- Global progress bar ----
        gp_frm = ttk.LabelFrame(parent, text="Global Progress  (global_results.json)")
        gp_frm.pack(fill="x", padx=10, pady=2)

        self._global_bar = tk.Canvas(gp_frm, height=14, bg="#3c3c3c",
                                     highlightthickness=1,
                                     highlightbackground="#606060")
        self._global_bar.pack(fill="x", padx=6, pady=(4, 2))
        self._global_bar.bind("<Configure>", lambda _: self._redraw_global_bar())

        gstat_row = ttk.Frame(gp_frm)
        gstat_row.pack(fill="x", padx=6, pady=(0, 2))
        self._gstat_ok    = tk.StringVar(value="Global OK: —")
        self._gstat_total = tk.StringVar(value="Target: —")
        for sv in (self._gstat_ok, self._gstat_total):
            ttk.Label(gstat_row, textvariable=sv).pack(side="left", padx=10)

        self._append_btn = ttk.Button(gstat_row, text="Append to Global Progress",
                                      command=self._on_append)
        self._append_btn.pack(side="right", padx=6)
        self._append_status = tk.StringVar(value="")
        ttk.Label(gstat_row, textvariable=self._append_status,
                  foreground="#80c8ff", font=("Consolas", 8)).pack(side="right", padx=4)

        # ---- NN trainer — horizontal inline ----
        self._build_nn_panel(parent)

    def _build_nn_panel(self, parent):
        nn_frm = ttk.LabelFrame(parent, text="3 · Train Surrogate NN")
        nn_frm.pack(fill="x", padx=10, pady=(4, 6))

        # Single horizontal row: hyperparams | buttons | status | artifacts
        row = ttk.Frame(nn_frm)
        row.pack(fill="x", padx=6, pady=4)

        # Hyperparameters inline
        for label, attr, default, width in [
            ("Epochs:",    "_nn_epochs_var", NN_EPOCHS,    6),
            ("Batch:",     "_nn_batch_var",  NN_BATCH,     5),
            ("LR:",        "_nn_lr_var",     NN_LR,        7),
            ("Val split:", "_nn_val_var",    NN_VAL_SPLIT, 5),
        ]:
            f = ttk.Frame(row); f.pack(side="left", padx=(0, 8))
            ttk.Label(f, text=label, anchor="w").pack(anchor="w")
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(f, textvariable=var, width=width).pack(anchor="w")

        # Separator
        ttk.Separator(row, orient="vertical").pack(side="left", fill="y", padx=8)

        # Dataset status
        f = ttk.Frame(row); f.pack(side="left", padx=(0, 8))
        ttk.Label(f, text="Dataset:", anchor="w").pack(anchor="w")
        self._nn_data_status = tk.StringVar(value="checking…")
        ttk.Label(f, textvariable=self._nn_data_status,
                  foreground="#80c8ff", font=("Consolas", 8)).pack(anchor="w")

        # Separator
        ttk.Separator(row, orient="vertical").pack(side="left", fill="y", padx=8)

        # Buttons
        f = ttk.Frame(row); f.pack(side="left", padx=(0, 8))
        self._train_btn = ttk.Button(f, text="Start Training",
                                     command=self._on_train_start)
        self._train_btn.pack(anchor="w", pady=(0, 2))
        self._stop_train_btn = ttk.Button(f, text="Stop",
                                          command=self._on_train_stop,
                                          state="disabled")
        self._stop_train_btn.pack(anchor="w")

        # Status + epoch inline
        f = ttk.Frame(row); f.pack(side="left", padx=(0, 8))
        self._train_status = tk.StringVar(value="—")
        ttk.Label(f, textvariable=self._train_status, foreground="gray").pack(anchor="w")
        self._train_epoch_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self._train_epoch_var,
                  foreground="#4ec94e", font=("Consolas", 8)).pack(anchor="w")

        # Separator
        ttk.Separator(row, orient="vertical").pack(side="left", fill="y", padx=8)

        # Artifact status
        f = ttk.Frame(row); f.pack(side="left", fill="x", expand=True)
        self._artifact_status = tk.StringVar(value="")
        ttk.Label(f, textvariable=self._artifact_status,
                  foreground="#80c8ff", font=("Consolas", 8),
                  wraplength=300, justify="left").pack(anchor="w")

        # Training log below the inline row
        log_hdr = ttk.Frame(nn_frm)
        log_hdr.pack(fill="x", padx=4, pady=(0, 0))
        ttk.Label(log_hdr, text="Training log", foreground="gray",
                  font=("Consolas", 8)).pack(side="left")
        ttk.Button(log_hdr, text="Clear",
                   command=self._clear_train_log).pack(side="right")

        self._train_log = tk.Text(nn_frm, height=6, state="disabled",
                                  wrap="word", font=("Consolas", 8),
                                  bg="#1e1e1e", fg="#d4d4d4",
                                  insertbackground="white", relief="flat", bd=0)
        tvsb = ttk.Scrollbar(nn_frm, orient="vertical", command=self._train_log.yview)
        self._train_log.configure(yscrollcommand=tvsb.set)
        tvsb.pack(side="right", fill="y", padx=(0, 2), pady=2)
        self._train_log.pack(fill="x", padx=(4, 0), pady=(0, 4))

    # ================================================================ domain config
    def _build_domain_config(self) -> dict:
        def _flist(lo_var, hi_var):
            return [float(lo_var.get()), float(hi_var.get())]

        def _side_cfg(v: dict) -> dict:
            layers = [bv.get() for bv in v["layers"]]
            return {
                "layers_selected":      layers,
                "od_max_mm":            float(v["od_max"].get()),
                "id_min_mm":            float(v["id_min"].get()),
                "trace_spacing_mm":     _flist(v["spacing_lo"], v["spacing_hi"]),
                "outer_cu_oz":          _flist(v["outer_oz_lo"], v["outer_oz_hi"]),
                "inner_cu_oz":          _flist(v["inner_oz_lo"], v["inner_oz_hi"]),
                "outer_gap_mm":         _flist(v["outer_gap_lo"], v["outer_gap_hi"]),
                "inner_gap_mm":         _flist(v["inner_gap_lo"], v["inner_gap_hi"]),
                "nhinc":                int(self._nhinc_var.get()),
                "nwinc":                int(self._nwinc_var.get()),
                "port_outside_allowed": bool(v["port_outside"].get()),
                "port_inside_allowed":  bool(v["port_inside"].get()),
            }

        cfg = {
            "tx": _side_cfg(self._tx),
            "rx": _side_cfg(self._rx),
            "global": {
                "pcb_gap_mm":    _flist(self._pcb_gap_lo_var, self._pcb_gap_hi_var),
                "resolution_mm": float(self._res_var.get()),
                "freq_hz": [float(self._fmin_var.get()),
                             float(self._fmax_var.get())],
            },
            "n_total": int(self._n_var.get()),
        }
        return cfg

    # ================================================================ log helpers
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

    def _log_train(self, text: str):
        self._train_log_queue = getattr(self, "_train_log_queue", queue.Queue())
        self._train_log_queue.put(text)

    def _flush_train_log(self):
        q = getattr(self, "_train_log_queue", None)
        if q is None:
            return
        try:
            while True:
                msg = q.get_nowait()
                self._train_log.configure(state="normal")
                self._train_log.insert("end", msg + "\n")
                self._train_log.see("end")
                self._train_log.configure(state="disabled")
                if msg.startswith("Epoch"):
                    self._train_epoch_var.set(msg.strip())
        except queue.Empty:
            pass

    def _clear_train_log(self):
        self._train_log.configure(state="normal")
        self._train_log.delete("1.0", "end")
        self._train_log.configure(state="disabled")

    # ================================================================ poll
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
            self._pause_btn.config(state="disabled", text="Pause")
            self._refresh_status()

    def _check_train_done(self):
        if self._train_proc is not None and self._train_proc.poll() is not None:
            rc = self._train_proc.returncode
            self._train_proc   = None
            self._train_thread = None
            if rc == 0:
                self._train_status.set("Done")
                self._log_train("[train] Finished successfully.")
                self._log("[train] Training complete — model saved to NeuralNetwork/")
            else:
                self._train_status.set(f"Error ({rc})")
                self._log_train(f"[train] Exited with code {rc}.")
            self._train_btn.config(state="normal")
            self._stop_train_btn.config(state="disabled")
            self._refresh_nn_status()

    # ================================================================ batch selection
    def _scan_available_batches(self) -> list:
        """Return list of available batch numbers (sorted ascending)."""
        pattern = os.path.join(_SIMDATA_DIR, "lhs_batch_*.json")
        files = glob.glob(pattern)
        batches = []
        for f in files:
            basename = os.path.basename(f)
            if basename.startswith("lhs_batch_") and basename.endswith(".json"):
                try:
                    n = int(basename[10:-5])
                    batches.append(n)
                except (ValueError, IndexError):
                    pass
        return sorted(batches)

    def _update_batch_selector(self):
        """Scan available batches and update the combobox."""
        batches = self._scan_available_batches()
        if not batches:
            self._batch_combobox["values"] = []
            self._batch_var.set("")
            self._active_batch_num = None
            self._batch_status.set("No batches available")
            return

        options = []
        for b in batches:
            samples_path = _batch_samples_path(b)
            try:
                with open(samples_path) as f:
                    data = json.load(f)
                n_samples = data.get("meta", {}).get("n_samples", 0)
                options.append(f"Batch {b} ({n_samples:,} samples)")
            except Exception:
                options.append(f"Batch {b} (error reading)")

        self._batch_combobox["values"] = options

        if len(batches) == 1:
            self._active_batch_num = batches[0]
            self._batch_var.set(options[0])
        elif batches:
            self._active_batch_num = batches[-1]
            self._batch_var.set(options[-1])
            self._batch_status.set(f"Auto-selected batch {self._active_batch_num}")

    def _on_batch_selected(self):
        """Called when user selects a batch from the combobox."""
        selected = self._batch_var.get()
        if not selected:
            self._active_batch_num = None
            return
        try:
            batch_num = int(selected.split()[1])
            self._active_batch_num = batch_num
            self._batch_status.set(f"Batch {batch_num} loaded")
            self._refresh_sweep_status()
            self._refresh_samples_status()
        except (ValueError, IndexError):
            pass

    # ================================================================ status
    def _refresh_status(self):
        os.makedirs(_SIMDATA_DIR, exist_ok=True)
        self._update_batch_selector()
        self._refresh_samples_status()
        self._refresh_sweep_status()
        self._refresh_global_status()
        self._refresh_nn_status()

    def _refresh_samples_status(self):
        if self._active_batch_num is None:
            self._gen_status.set("no batch selected")
            self._n_total_samples = 0
            return

        samples_path = _batch_samples_path(self._active_batch_num)
        if os.path.exists(samples_path):
            try:
                with open(samples_path) as f:
                    d = json.load(f)
                n = len(d.get("samples", []))
                self._n_total_samples = n
                self._gen_status.set(f"{n} samples (batch {self._active_batch_num})")
            except Exception:
                self._gen_status.set("file unreadable")
                self._n_total_samples = 0
        else:
            self._gen_status.set("sample file not found")
            self._n_total_samples = 0

    def _refresh_sweep_status(self):
        self._sweep_covered = set()
        self._sweep_n_range = 0
        self._sweep_n_ok    = 0

        if self._active_batch_num is None:
            self._sstat_done.set("Done: —")
            self._sstat_ok.set("OK: —")
            self._redraw_sweep_bar()
            return

        results_path = _batch_results_path(self._active_batch_num)
        if not os.path.exists(results_path):
            self._sstat_done.set("Done: —")
            self._sstat_ok.set("OK: —")
            self._redraw_sweep_bar()
            return

        try:
            with open(results_path) as f:
                d = json.load(f)
        except Exception:
            self._sstat_done.set("Done: (unreadable)")
            self._redraw_sweep_bar()
            return

        results = d.get("results", [])
        meta    = d.get("meta", {})
        from_idx = int(meta.get("from_idx", 0))
        to_idx   = int(meta.get("to_idx",   len(results)))
        n_range  = to_idx - from_idx

        covered = set()
        n_ok = 0
        for r in results:
            sid = r.get("sample_num")
            if not isinstance(sid, int):
                continue
            if from_idx < sid <= to_idx:
                covered.add(sid)
                if r.get("ok"):
                    n_ok += 1

        n_done = len(covered)
        self._sweep_covered  = covered
        self._sweep_from_idx = from_idx
        self._sweep_to_idx   = to_idx
        self._sweep_n_range  = n_range
        self._sweep_n_ok     = n_ok

        pct = f"{100*n_done//n_range}%" if n_range else "—"
        self._sstat_done.set(f"Done: {n_done}/{n_range}  ({pct})")
        self._sstat_ok.set(f"OK: {n_ok}")
        self._redraw_sweep_bar()

    def _refresh_global_status(self):
        self._covered_global = set()
        n_ok = 0
        n_target = 0

        if os.path.exists(_DOMAIN_FILE):
            try:
                with open(_DOMAIN_FILE) as f:
                    dm = json.load(f)
                n_target = int(dm.get("n_total", 0))
            except Exception:
                pass

        if n_target == 0 and self._n_total_samples:
            n_target = self._n_total_samples

        self._n_target = n_target

        if os.path.exists(_GLOBAL_FILE):
            try:
                with open(_GLOBAL_FILE) as f:
                    d = json.load(f)
                for r in d.get("results", []):
                    sid = r.get("sample_id")
                    if isinstance(sid, int) and sid >= 0:
                        self._covered_global.add(sid)
                        if r.get("ok"):
                            n_ok += 1
            except Exception:
                pass

        self._gstat_ok.set(f"Global OK: {n_ok}")
        self._gstat_total.set(f"Target: {n_target}")
        self._redraw_global_bar()

    def _refresh_nn_status(self):
        if os.path.exists(_GLOBAL_FILE):
            try:
                with open(_GLOBAL_FILE) as f:
                    d = json.load(f)
                n = sum(1 for r in d.get("results", []) if r.get("ok"))
                self._nn_data_status.set(f"{n} OK rows in global_results.json")
            except Exception:
                self._nn_data_status.set("global_results.json unreadable")
        else:
            self._nn_data_status.set("No global_results.json — run append first")

        parts = []
        if os.path.exists(_MODEL_FILE):
            parts.append("model.pth ✓")
        if os.path.exists(os.path.join(_NN_DIR, "x_scaler.pkl")):
            parts.append("x_scaler ✓")
        if os.path.exists(os.path.join(_NN_DIR, "y_scaler.pkl")):
            parts.append("y_scaler ✓")
        if os.path.exists(_LOSS_PLOT):
            parts.append("loss_curve ✓")
        self._artifact_status.set("  ".join(parts) if parts else "No trained model yet")

    # ================================================================ progress bars
    def _redraw_global_bar(self):
        c = self._global_bar
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 2 or self._n_target == 0:
            return

        n = self._n_target

        def _x(sid):
            return int(w * sid / n)

        covered = self._covered_global
        if not covered:
            return

        ids = sorted(covered)
        lo = hi = ids[0]
        for sid in ids[1:]:
            if sid == hi + 1:
                hi = sid
            else:
                c.create_rectangle(_x(lo), 1, _x(hi + 1), h - 1,
                                   fill="#3ec9a7", outline="")
                lo = hi = sid
        c.create_rectangle(_x(lo), 1, _x(hi + 1), h - 1,
                           fill="#3ec9a7", outline="")

    def _redraw_sweep_bar(self):
        c = self._sweep_bar
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 2:
            return

        n_total = self._n_total_samples
        if n_total == 0:
            return

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

        x0, x1 = _x(run_from), _x(run_to)
        if x1 > x0:
            c.create_rectangle(x0, 0, x1, h, fill="#555555", outline="")

        covered = getattr(self, "_sweep_covered", set())
        if covered:
            in_range = sorted(i for i in covered if run_from < i <= run_to)
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

            outside = sorted(i for i in covered if not (run_from < i <= run_to))
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

        if x0 > 0:
            c.create_line(x0, 0, x0, h, fill="#ffcc44", width=2)
        if x1 < w:
            c.create_line(x1, 0, x1, h, fill="#ffcc44", width=2)

    # ================================================================ generate
    def _on_generate(self):
        if self._gen_thread and self._gen_thread.is_alive():
            return
        try:
            cfg = self._build_domain_config()
        except (ValueError, KeyError) as exc:
            self._log(f"[gen] ERROR building config: {exc}")
            return

        self._gen_btn.config(state="disabled")
        self._gen_status.set("Running…")
        self._log(f"[gen] Generating {cfg['n_total']} samples …")

        cfg_path = os.path.join(_SIMDATA_DIR, "_gen_config.json")
        os.makedirs(_SIMDATA_DIR, exist_ok=True)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        def _run():
            try:
                cmd = [sys.executable, _GEN_SCRIPT,
                       "--config", cfg_path,
                       "--master", _DOMAIN_FILE,
                       "--global", _GLOBAL_FILE,
                       "--n",      str(cfg["n_total"])]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=_NN_DIR)
                for line in proc.stdout:
                    self._log(line.rstrip())
                proc.wait()
                if proc.returncode == 0:
                    self._log("[gen] Done.")
                    self._gen_status.set("Done")
                else:
                    self._log(f"[gen] ERROR: exit {proc.returncode}")
                    self._gen_status.set(f"Error ({proc.returncode})")
            except Exception as exc:
                self._log(f"[gen] EXCEPTION: {exc}")
                self._gen_status.set("Error")
            finally:
                self.after(0, lambda: self._gen_btn.config(state="normal"))
                self.after(0, self._refresh_status)

        self._gen_thread = threading.Thread(target=_run, daemon=True)
        self._gen_thread.start()

    # ================================================================ sweep
    def _on_sweep_start(self):
        if self._sweep_proc is not None:
            return
        if self._active_batch_num is None:
            self._log("[sweep] ERROR: No batch selected — generate first.")
            return
        samples_path = _batch_samples_path(self._active_batch_num)
        if not os.path.exists(samples_path):
            self._log("[sweep] ERROR: Sample file not found — select a valid batch.")
            return
        try:
            workers  = int(self._workers_var.get())
            timeout  = int(self._timeout_var.get())
            ckpt     = int(self._ckpt_var.get())
            from_idx = int(self._from_var.get()) if self._from_var.get().strip() else 0
            to_idx   = int(self._to_var.get())   if self._to_var.get().strip()   else None
        except ValueError:
            self._log("[sweep] ERROR: invalid numeric parameter.")
            return

        if os.path.exists(_STOP_FLAG):
            os.remove(_STOP_FLAG)

        results_path = _batch_results_path(self._active_batch_num)
        cmd = [sys.executable, _SWEEP_SCRIPT,
               "--samples",          samples_path,
               "--out",              results_path,
               "--batch",            str(self._active_batch_num),
               "--workers",          str(workers),
               "--timeout",          str(timeout),
               "--checkpoint-every", str(ckpt),
               "--from-idx",         str(from_idx)]
        if to_idx is not None:
            cmd += ["--to-idx", str(to_idx)]

        self._log(f"[sweep] workers={workers} timeout={timeout}s "
                  f"range={from_idx}->{to_idx or 'end'}")
        try:
            self._sweep_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=_NN_DIR)
        except Exception as exc:
            self._log(f"[sweep] Failed to launch: {exc}")
            return

        self._sweep_status.set("Running…")
        self._sweep_btn.config(state="disabled")
        self._pause_btn.config(state="normal", text="Pause")

        def _drain():
            for line in self._sweep_proc.stdout:
                stripped = line.rstrip()
                self._log(stripped)
                if "[checkpoint]" in stripped.lower():
                    self.after(0, self._refresh_status)

        self._sweep_thread = threading.Thread(target=_drain, daemon=True)
        self._sweep_thread.start()

    def _on_pause(self):
        if self._sweep_proc is None:
            return
        if os.path.exists(_STOP_FLAG):
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
            self._sweep_status.set("Stopping…")
            self._log("[sweep] Stop flag set — terminating workers immediately.")

    # ================================================================ append
    def _on_append(self):
        if getattr(self, "_append_thread", None) and self._append_thread.is_alive():
            self._log("[append] Already running.")
            return

        if self._active_batch_num is None:
            self._log("[append] ERROR: No batch selected.")
            return

        results_path = _batch_results_path(self._active_batch_num)
        if not os.path.exists(results_path):
            self._log(f"[append] ERROR: No results for batch {self._active_batch_num}.")
            return

        self._append_btn.config(state="disabled")
        self._append_status.set("Appending…")
        self._log(f"[append] Merging batch {self._active_batch_num} → global_results.json …")

        def _run():
            try:
                cmd = [sys.executable, _APPEND_SCRIPT,
                       "--local", results_path,
                       "--global", _GLOBAL_FILE,
                       "--domain", _DOMAIN_FILE]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=_NN_DIR)
                for line in proc.stdout:
                    self._log(line.rstrip())
                proc.wait()
                if proc.returncode == 0:
                    self._append_status.set("Done")
                    self._log("[append] Done.")
                else:
                    self._append_status.set(f"Error ({proc.returncode})")
                    self._log(f"[append] Error: exit {proc.returncode}")
            except Exception as exc:
                self._log(f"[append] EXCEPTION: {exc}")
                self._append_status.set("Error")
            finally:
                self.after(0, lambda: self._append_btn.config(state="normal"))
                self.after(0, self._refresh_status)

        self._append_thread = threading.Thread(target=_run, daemon=True)
        self._append_thread.start()

    # ================================================================ train
    def _on_train_start(self):
        if self._train_proc is not None:
            return
        if not os.path.exists(_GLOBAL_FILE):
            self._log_train("[train] ERROR: global_results.json not found — run append first.")
            return
        if not os.path.exists(_TRAIN_SCRIPT):
            self._log_train(f"[train] ERROR: train_surrogate.py not at {_TRAIN_SCRIPT}")
            return
        try:
            epochs = int(self._nn_epochs_var.get())
            batch  = int(self._nn_batch_var.get())
            lr     = float(self._nn_lr_var.get())
            val    = float(self._nn_val_var.get())
            assert 0 < val < 1
        except (ValueError, AssertionError):
            self._log_train("[train] ERROR: invalid hyperparameters.")
            return

        self._train_btn.config(state="disabled")
        self._stop_train_btn.config(state="normal")
        self._train_status.set("Running…")
        self._train_epoch_var.set("")
        self._log_train(f"[train] epochs={epochs} batch={batch} lr={lr} val={val}")
        self._log("[train] Launched train_surrogate.py")

        env = os.environ.copy()
        env["SURROGATE_EPOCHS"]     = str(epochs)
        env["SURROGATE_BATCH_SIZE"] = str(batch)
        env["SURROGATE_LR"]         = str(lr)
        env["SURROGATE_VAL_SPLIT"]  = str(val)

        try:
            self._train_proc = subprocess.Popen(
                [sys.executable, "-u", _TRAIN_SCRIPT],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=_NN_DIR, env=env)
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
        self._log_train("[train] Stopped by user.")
        self._train_btn.config(state="normal")
        self._stop_train_btn.config(state="disabled")
