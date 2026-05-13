#!/usr/bin/env python3
"""
NN Training tab — folder-centric state machine.

The tab shows one of four panels depending on folder contents:
  EMPTY   → LHS generation form
  SAMPLES → Sweep controls + progress
  RESULTS → Training controls + log
  TRAINED → Training plot + auto-redirect to next tab

Fixed (not configurable here, never NN inputs):
  TX: series, port outside, L1+L2, 1oz/0.5oz
  RX: parallel_pairs_ser or series (user selects), port inside, all 4 layers, 1oz/0.5oz
"""

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk

_HERE        = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT    = os.path.dirname(_HERE)
_NN_DIR      = os.path.join(_APP_ROOT, "NeuralNetwork")
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")
_SCRIPTS_DIR = os.path.join(_NN_DIR, "scripts")

_GEN_SCRIPT      = os.path.join(_SCRIPTS_DIR, "generate_lhs_samples.py")
_SWEEP_SCRIPT    = os.path.join(_SCRIPTS_DIR, "run_sweep.py")
_TRAIN_SCRIPT    = os.path.join(_SCRIPTS_DIR, "train_surrogate.py")
_HP_SWEEP_SCRIPT = os.path.join(_SCRIPTS_DIR, "hp_sweep.py")

for _p in (_NN_DIR, _MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POLL_MS      = 600
FOLDER_POLL  = 3000

_GUI_SCALE  = float(os.environ.get("COIL_GUI_SCALE", "1.0"))
_SIDE_WIDTH = int(324 * _GUI_SCALE)


# ---------------------------------------------------------------------------
# Folder-state detection
# ---------------------------------------------------------------------------

def _samples_path(folder):   return os.path.join(folder, "lhs_samples.json")
def _domain_path(folder):    return os.path.join(folder, "domain.json")
def _results_path(folder):   return os.path.join(folder, "results.json")
def _model_path(folder):     return os.path.join(folder, "surrogate_model.pth")
def _loss_plot_path(folder): return os.path.join(folder, "loss_curve.png")
def _stop_flag(folder):      return os.path.join(folder, "STOP_SWEEP")
def _hp_plots_dir(folder):   return os.path.join(folder, "hp_sweep_plots")

ST_EMPTY   = "empty"
ST_SAMPLES = "samples"
ST_RESULTS = "results"
ST_TRAINED = "trained"


def _folder_state(folder: str) -> str:
    if not folder or not os.path.isdir(folder):
        return ST_EMPTY
    if os.path.exists(_model_path(folder)):
        return ST_TRAINED
    if os.path.exists(_results_path(folder)) and _results_complete(folder):
        return ST_RESULTS
    if os.path.exists(_samples_path(folder)):
        return ST_SAMPLES
    return ST_EMPTY


def _results_complete(folder: str) -> bool:
    sp = _samples_path(folder)
    rp = _results_path(folder)
    if not os.path.exists(sp) or not os.path.exists(rp):
        return False
    try:
        with open(sp) as f:
            n_samples = json.load(f).get("meta", {}).get("n_samples", 0)
        with open(rp) as f:
            d = json.load(f)
        n_done = len(d.get("results", []))
        return n_samples > 0 and n_done >= n_samples
    except Exception:
        return False


def _n_samples(folder: str) -> int:
    try:
        with open(_samples_path(folder)) as f:
            return json.load(f).get("meta", {}).get("n_samples", 0)
    except Exception:
        return 0


def _n_ok_results(folder: str) -> int:
    try:
        with open(_results_path(folder)) as f:
            d = json.load(f)
        return len(d.get("results", []))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Default domain values
# ---------------------------------------------------------------------------

# TX — geometry DOF only (copper/topology/port/layers are fixed)
TX_OD_LO        = "50.0";  TX_OD_HI        = "54.0"
TX_ID_MIN       = "39.0"
TX_W_LO         = "0.2";   TX_W_HI         = "1.2"
TX_SPACING      = "0.16"
TX_TURNS_LO     = "6";     TX_TURNS_HI     = "18"

# RX
RX_OD_LO        = "50.0";  RX_OD_HI        = "54.0"
RX_ID_MIN       = "35.0"
RX_W_LO         = "0.2";   RX_W_HI         = "1.2"
RX_SPACING      = "0.16"
RX_TURNS_LO     = "4";     RX_TURNS_HI     = "25"
RX_GROUND_DISC  = "20.0"   # Ø of passive copper disc on both RX inner layers

# Shared
RESOLUTION_MM  = "1.2"
PCB_GAP_LO     = "2.6";    PCB_GAP_HI     = "2.6"
FREQ_MIN_KHZ   = "280";    FREQ_MAX_KHZ   = "350"
TOTAL_SAMPLES  = "8000"

SWEEP_WORKERS  = "15"; SWEEP_TIMEOUT_S = "360"; SWEEP_CKPT_EVERY = "45"
NN_EPOCHS = "300"; NN_BATCH = "256"; NN_LR = "0.0005"; NN_VAL_SPLIT = "0.2"


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------

def _range_row(parent, label, lo_var, hi_var, label_width=16, entry_width=6, pady=2):
    r = ttk.Frame(parent)
    r.pack(fill="x", padx=6, pady=pady)
    ttk.Label(r, text=label, width=label_width, anchor="w").pack(side="left")
    ttk.Entry(r, textvariable=lo_var, width=entry_width).pack(side="left", padx=(4, 2))
    ttk.Label(r, text="–", foreground="gray").pack(side="left")
    ttk.Entry(r, textvariable=hi_var, width=entry_width).pack(side="left", padx=(2, 0))
    return r


def _fixed_row(parent, label, value_str, label_width=16):
    r = ttk.Frame(parent)
    r.pack(fill="x", padx=6, pady=1)
    ttk.Label(r, text=label, width=label_width, anchor="w",
              foreground="gray").pack(side="left")
    ttk.Label(r, text=value_str, foreground="#808080",
              font=("TkDefaultFont", 8, "italic")).pack(side="left")
    return r


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

class AutomationTab(ttk.Frame):

    def __init__(self, parent, app=None, **kw):
        super().__init__(parent, **kw)
        self.app = app

        self._folder    = tk.StringVar()
        self._state     = None
        self._log_queue = queue.Queue()
        self._train_log_queue = queue.Queue()

        self._gen_thread   = None
        self._sweep_proc   = None
        self._sweep_thread = None
        self._train_proc   = None
        self._train_thread = None

        self._n_total_samples     = 0
        self._sweep_covered_uuids = set()
        self._sweep_n_ok          = 0

        self._build_header()
        self._content_frame = ttk.Frame(self)
        self._content_frame.pack(fill="both", expand=True)

        self._current_panel  = None
        self._folder_poll_id = None
        self._last_folder    = None

        self._do_state_check()
        self.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ header

    def _build_header(self):
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(hdr, text="Model folder:").pack(side="left")
        self._folder_entry = ttk.Entry(hdr, textvariable=self._folder, width=50)
        self._folder_entry.pack(side="left", padx=(4, 4), fill="x", expand=True)
        ttk.Button(hdr, text="Browse…", command=self._browse_folder).pack(side="left")
        ttk.Button(hdr, text="New…",    command=self._new_folder).pack(side="left", padx=(4, 0))
        self._folder.trace_add("write", lambda *_: self.after(100, self._on_folder_changed))

        if self.app is not None:
            saved = self.app.load_nn_setup_folder()
            if saved and os.path.isdir(saved):
                self._folder.set(saved)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=8, pady=(4, 0))

    def _browse_folder(self):
        initial = self._folder.get() or _NN_DIR
        path = filedialog.askdirectory(initialdir=initial, title="Select model folder")
        if path:
            self._folder.set(path)

    def _new_folder(self):
        initial = self._folder.get() or _NN_DIR
        path = filedialog.askdirectory(initialdir=initial,
                                       title="Create / select new model folder",
                                       mustexist=False)
        if path:
            os.makedirs(path, exist_ok=True)
            self._folder.set(path)

    # ---------------------------------------------------------------- state machine

    def _on_folder_changed(self):
        folder = self._folder.get()
        self._last_folder = folder
        self._state = _folder_state(folder)
        self._rebuild_content()
        self._schedule_folder_poll()
        if self.app is not None:
            self.app.persist_nn_setup_folder(folder)
            has_model = bool(folder) and os.path.exists(_model_path(folder))
            if hasattr(self.app, "set_nn_optim_tab_visible"):
                self.app.set_nn_optim_tab_visible(has_model)
            auto_nn = getattr(self.app, "nn_optim_tab", None)
            if auto_nn is not None and hasattr(auto_nn, "load_domain_from_model"):
                try:
                    auto_nn.load_domain_from_model(folder)
                except Exception:
                    pass

    def _schedule_folder_poll(self):
        if self._folder_poll_id is not None:
            self.after_cancel(self._folder_poll_id)
        self._folder_poll_id = self.after(FOLDER_POLL, self._do_state_check)

    def _do_state_check(self):
        folder = self._folder.get()
        new_st = _folder_state(folder)
        folder_changed = (folder != self._last_folder)
        self._last_folder = folder
        if new_st != self._state or folder_changed:
            self._state = new_st
            self._rebuild_content()
            if self.app is not None and hasattr(self.app, "set_nn_optim_tab_visible"):
                has_model = bool(folder) and os.path.exists(_model_path(folder))
                self.app.set_nn_optim_tab_visible(has_model)
        self._schedule_folder_poll()

    def _rebuild_content(self):
        for w in self._content_frame.winfo_children():
            w.destroy()
        self._current_panel = self._state
        if self._state == ST_EMPTY:
            self._build_panel_empty(self._content_frame)
        elif self._state == ST_SAMPLES:
            self._build_panel_samples(self._content_frame)
        elif self._state == ST_RESULTS:
            self._build_panel_results(self._content_frame)
        elif self._state == ST_TRAINED:
            self._build_panel_trained(self._content_frame)

    def get_model_folder(self) -> str:
        return self._folder.get()

    def _go_next_tab(self):
        if self.app is None:
            return
        try:
            nb  = self.app._nb
            cur = nb.index("current")
            nb.select(cur + 1)
        except Exception:
            pass

    # ================================================================ PANEL: EMPTY

    def _build_panel_empty(self, parent):
        ttk.Label(parent,
                  text="Folder is empty — configure and generate LHS samples.",
                  foreground="gray").pack(anchor="w", padx=10, pady=(8, 4))

        cols = ttk.Frame(parent)
        cols.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        tx_frm = ttk.LabelFrame(cols, text="TX Coil", width=_SIDE_WIDTH)
        tx_frm.pack(side="left", fill="y", padx=(0, 2), anchor="n")
        tx_frm.pack_propagate(False)

        tk.Frame(cols, bg="#606060", width=2).pack(side="left", fill="y", padx=2)

        rx_frm = ttk.LabelFrame(cols, text="RX Coil", width=_SIDE_WIDTH)
        rx_frm.pack(side="left", fill="y", padx=(2, 4), anchor="n")
        rx_frm.pack_propagate(False)

        right = ttk.Frame(cols)
        right.pack(side="left", fill="both", expand=True, anchor="n")

        self._build_tx_side(tx_frm)
        self._build_rx_side(rx_frm)
        self._build_right_column(right)

        # Restore saved inputs, then wire up auto-save on every change.
        if self.app is not None:
            self._restore_empty_inputs(self.app.load_nn_setup_inputs())
        self._attach_empty_input_traces()

    def _build_tx_side(self, parent):
        # Fixed configuration notice
        info = ttk.LabelFrame(parent, text="Fixed (hardcoded)")
        info.pack(fill="x", padx=4, pady=(4, 2))
        _fixed_row(info, "Topology:",    "series, port outside")
        _fixed_row(info, "Layers:",      "L1 + L2 active")
        _fixed_row(info, "Copper:",      "L1=1oz  L2=0.5oz")
        _fixed_row(info, "L1–L2 gap:",   "0.2104 mm (7628×1 prepreg)")
        _fixed_row(info, "L3 ground:",   "fixed pour (see ground_plane.py)")

        # Geometry DOF
        geo = ttk.LabelFrame(parent, text="Geometry (sampled)")
        geo.pack(fill="x", padx=4, pady=2)

        self._tx_od_lo_var = tk.StringVar(value=TX_OD_LO)
        self._tx_od_hi_var = tk.StringVar(value=TX_OD_HI)
        _range_row(geo, "OD range (mm):", self._tx_od_lo_var, self._tx_od_hi_var)

        r_id = ttk.Frame(geo); r_id.pack(fill="x", padx=6, pady=2)
        ttk.Label(r_id, text="ID min (mm):", width=16, anchor="w").pack(side="left")
        self._tx_id_min_var = tk.StringVar(value=TX_ID_MIN)
        ttk.Entry(r_id, textvariable=self._tx_id_min_var, width=6).pack(side="left", padx=4)

        self._tx_w_lo_var = tk.StringVar(value=TX_W_LO)
        self._tx_w_hi_var = tk.StringVar(value=TX_W_HI)
        _range_row(geo, "Trace width (mm):", self._tx_w_lo_var, self._tx_w_hi_var)

        r = ttk.Frame(geo); r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="Spacing (mm):", width=16, anchor="w").pack(side="left")
        self._tx_spacing_var = tk.StringVar(value=TX_SPACING)
        ttk.Entry(r, textvariable=self._tx_spacing_var, width=6).pack(side="left", padx=4)

        self._tx_turns_lo_var = tk.StringVar(value=TX_TURNS_LO)
        self._tx_turns_hi_var = tk.StringVar(value=TX_TURNS_HI)
        _range_row(geo, "L1 turns:", self._tx_turns_lo_var, self._tx_turns_hi_var)

        self._tx_l2_turns_lo_var = tk.StringVar(value=TX_TURNS_LO)
        r_l2 = ttk.Frame(geo); r_l2.pack(fill="x", padx=6, pady=2)
        ttk.Label(r_l2, text="L2 turns:", width=16, anchor="w").pack(side="left")
        ttk.Entry(r_l2, textvariable=self._tx_l2_turns_lo_var, width=6).pack(side="left", padx=(4, 2))
        ttk.Label(r_l2, text="–", foreground="gray").pack(side="left")
        self._tx_l2_turns_hi_label = ttk.Label(r_l2, text=self._tx_turns_hi_var.get(),
                                                foreground="#808080", width=6, anchor="w")
        self._tx_l2_turns_hi_label.pack(side="left", padx=(2, 0))
        ttk.Label(r_l2, text="(= L1 max)", foreground="gray",
                  font=("TkDefaultFont", 7)).pack(side="left", padx=(4, 0))
        self._tx_turns_hi_var.trace_add("write",
            lambda *_: self._tx_l2_turns_hi_label.config(
                text=self._tx_turns_hi_var.get()))

    def _build_rx_side(self, parent):
        info = ttk.LabelFrame(parent, text="Fixed (hardcoded)")
        info.pack(fill="x", padx=4, pady=(4, 2))
        _fixed_row(info, "Port:",       "inside")
        _fixed_row(info, "Layers:",     "all 4 active")
        _fixed_row(info, "Copper:",     "outer=1oz  inner=0.5oz")
        _fixed_row(info, "Outer gap:",  "0.2104 mm (7628×1 prepreg)")
        _fixed_row(info, "Inner gap:",  "0.6 mm (core)")

        geo = ttk.LabelFrame(parent, text="Geometry (sampled)")
        geo.pack(fill="x", padx=4, pady=2)

        self._rx_od_lo_var = tk.StringVar(value=RX_OD_LO)
        self._rx_od_hi_var = tk.StringVar(value=RX_OD_HI)
        _range_row(geo, "OD range (mm):", self._rx_od_lo_var, self._rx_od_hi_var)

        r_id = ttk.Frame(geo); r_id.pack(fill="x", padx=6, pady=2)
        ttk.Label(r_id, text="ID min (mm):", width=16, anchor="w").pack(side="left")
        self._rx_id_min_var = tk.StringVar(value=RX_ID_MIN)
        ttk.Entry(r_id, textvariable=self._rx_id_min_var, width=6).pack(side="left", padx=4)

        self._rx_w_lo_var = tk.StringVar(value=RX_W_LO)
        self._rx_w_hi_var = tk.StringVar(value=RX_W_HI)
        _range_row(geo, "Trace width (mm):", self._rx_w_lo_var, self._rx_w_hi_var)

        r = ttk.Frame(geo); r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="Spacing (mm):", width=16, anchor="w").pack(side="left")
        self._rx_spacing_var = tk.StringVar(value=RX_SPACING)
        ttk.Entry(r, textvariable=self._rx_spacing_var, width=6).pack(side="left", padx=4)

        self._rx_turns_lo_var = tk.StringVar(value=RX_TURNS_LO)
        self._rx_turns_hi_var = tk.StringVar(value=RX_TURNS_HI)
        _range_row(geo, "Turns:", self._rx_turns_lo_var, self._rx_turns_hi_var)

        r_gd = ttk.Frame(geo); r_gd.pack(fill="x", padx=6, pady=2)
        ttk.Label(r_gd, text="Ground disc Ø (mm):", width=20, anchor="w").pack(side="left")
        self._rx_gdisc_dia_var = tk.StringVar(value=RX_GROUND_DISC)
        ttk.Entry(r_gd, textvariable=self._rx_gdisc_dia_var, width=6).pack(side="left", padx=4)
        ttk.Label(r_gd, text="(both inner layers; 0 disables)",
                  foreground="#808080", font=("TkDefaultFont", 8)).pack(side="left", padx=6)

        topo = ttk.LabelFrame(parent, text="Topology (sampled)")
        topo.pack(fill="x", padx=4, pady=2)
        ttk.Label(topo, text="series  |  parallel_pairs_ser",
                  foreground="#808080", font=("TkDefaultFont", 8)).pack(
                      anchor="w", padx=6, pady=4)

    def _build_right_column(self, parent):
        shared = ttk.LabelFrame(parent, text="Shared Parameters")
        shared.pack(fill="x", padx=4, pady=(4, 2))

        row1 = ttk.Frame(shared); row1.pack(fill="x", padx=4, pady=(4, 2))
        for label, attr, default, width in [
            ("Resolution (mm):", "_res_var", RESOLUTION_MM, 6),
        ]:
            f = ttk.Frame(row1); f.pack(side="left", padx=(0, 10))
            ttk.Label(f, text=label, anchor="w").pack(anchor="w")
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(f, textvariable=var, width=width).pack(anchor="w")

        row2 = ttk.Frame(shared); row2.pack(fill="x", padx=4, pady=(0, 2))
        self._pcb_gap_lo_var = tk.StringVar(value=PCB_GAP_LO)
        self._pcb_gap_hi_var = tk.StringVar(value=PCB_GAP_HI)
        _range_row(row2, "PCB gap (mm):", self._pcb_gap_lo_var, self._pcb_gap_hi_var,
                   label_width=16, entry_width=6, pady=0)

        row3 = ttk.Frame(shared); row3.pack(fill="x", padx=4, pady=(0, 4))
        self._fmin_var = tk.StringVar(value=FREQ_MIN_KHZ)
        self._fmax_var = tk.StringVar(value=FREQ_MAX_KHZ)
        _range_row(row3, "Freq range (kHz):", self._fmin_var, self._fmax_var,
                   label_width=17, entry_width=7, pady=0)

        gen_frm = ttk.LabelFrame(parent, text="Generate LHS Samples")
        gen_frm.pack(fill="x", padx=4, pady=2)

        gr = ttk.Frame(gen_frm); gr.pack(fill="x", padx=6, pady=4)

        f = ttk.Frame(gr); f.pack(side="left", padx=(0, 10))
        ttk.Label(f, text="Total samples:", anchor="w").pack(anchor="w")
        self._n_var = tk.StringVar(value=TOTAL_SAMPLES)
        ttk.Entry(f, textvariable=self._n_var, width=8).pack(anchor="w")

        self._gen_btn = ttk.Button(gr, text="Generate Samples", command=self._on_generate)
        self._gen_btn.pack(side="left", padx=(0, 8))
        self._gen_status = tk.StringVar(value="—")
        ttk.Label(gr, textvariable=self._gen_status, foreground="gray").pack(side="left")

        con_frm = ttk.LabelFrame(parent, text="Console")
        con_frm.pack(fill="both", expand=True, padx=4, pady=2)
        btn_bar = ttk.Frame(con_frm); btn_bar.pack(fill="x", padx=4, pady=(2, 0))
        ttk.Button(btn_bar, text="Clear", command=self._clear_console).pack(side="right")
        self._console = tk.Text(con_frm, height=7, state="disabled",
                                wrap="none", font=("Consolas", 8),
                                bg="#1e1e1e", fg="#d4d4d4",
                                insertbackground="white", relief="flat", bd=0)
        vsb = ttk.Scrollbar(con_frm, orient="vertical",   command=self._console.yview)
        hsb = ttk.Scrollbar(con_frm, orient="horizontal", command=self._console.xview)
        self._console.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._console.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 2))

    # ================================================================ domain config builder

    def _build_domain_config(self) -> dict:
        def fv(var): return float(var.get())

        return {
            "tx": {
                "od_mm":            [fv(self._tx_od_lo_var), fv(self._tx_od_hi_var)],
                "id_min_mm":        fv(self._tx_id_min_var),
                "trace_width_mm":   [fv(self._tx_w_lo_var),     fv(self._tx_w_hi_var)],
                "trace_spacing_mm": [fv(self._tx_spacing_var),  fv(self._tx_spacing_var)],
                "turns":            [int(self._tx_turns_lo_var.get()),
                                     int(self._tx_turns_hi_var.get())],
                "l2_turns":         [int(self._tx_l2_turns_lo_var.get()),
                                     int(self._tx_turns_hi_var.get())],
                "l1l2_gap_mm":      [0.2104, 0.2104],
            },
            "rx": {
                "od_mm":            [fv(self._rx_od_lo_var), fv(self._rx_od_hi_var)],
                "id_min_mm":        fv(self._rx_id_min_var),
                "trace_width_mm":   [fv(self._rx_w_lo_var),     fv(self._rx_w_hi_var)],
                "trace_spacing_mm": [fv(self._rx_spacing_var),  fv(self._rx_spacing_var)],
                "turns":            [int(self._rx_turns_lo_var.get()),
                                     int(self._rx_turns_hi_var.get())],
                "outer_gap_mm":     [0.2104, 0.2104],
                "inner_gap_mm":     [0.6, 0.6],
                "ground_disc_dia_mm": fv(self._rx_gdisc_dia_var),
            },
            "global": {
                "pcb_gap_mm":    [fv(self._pcb_gap_lo_var), fv(self._pcb_gap_hi_var)],
                "resolution_mm": fv(self._res_var),
                "freq_hz":       [float(self._fmin_var.get()) * 1000.0,
                                  float(self._fmax_var.get()) * 1000.0],
            },
            "n_total": int(self._n_var.get()),
        }

    # ================================================================ empty-panel input persistence

    _EMPTY_INPUT_VARS = [
        "tx_od_lo", "tx_od_hi", "tx_id_min", "tx_w_lo", "tx_w_hi",
        "tx_spacing", "tx_turns_lo", "tx_turns_hi", "tx_l2_turns_lo",
        "rx_od_lo", "rx_od_hi", "rx_id_min", "rx_w_lo", "rx_w_hi",
        "rx_spacing", "rx_turns_lo", "rx_turns_hi", "rx_gdisc_dia",
        "res", "pcb_gap_lo", "pcb_gap_hi", "fmin", "fmax", "n",
    ]

    def _collect_empty_inputs(self) -> dict:
        out = {}
        for name in self._EMPTY_INPUT_VARS:
            attr = f"_{name}_var"
            var = getattr(self, attr, None)
            if var is not None:
                out[name] = var.get()
        return out

    def _restore_empty_inputs(self, d: dict):
        if not d:
            return
        for name, value in d.items():
            attr = f"_{name}_var"
            var = getattr(self, attr, None)
            if var is not None:
                try:
                    var.set(value)
                except Exception:
                    pass

    def _attach_empty_input_traces(self):
        if self.app is None:
            return
        def _save(*_):
            if self._state == ST_EMPTY:
                self.app.persist_nn_setup_inputs(self._collect_empty_inputs())
        for name in self._EMPTY_INPUT_VARS:
            var = getattr(self, f"_{name}_var", None)
            if var is not None:
                var.trace_add("write", _save)

    # ================================================================ PANEL: SAMPLES

    def _build_panel_samples(self, parent):
        folder = self._folder.get()
        n_sam  = _n_samples(folder)
        n_ok   = _n_ok_results(folder)

        info = ttk.LabelFrame(parent, text="Domain")
        info.pack(fill="x", padx=10, pady=(8, 4))
        self._populate_domain_info(info, folder)

        sw_frm = ttk.LabelFrame(parent, text="Run FastHenry Sweep")
        sw_frm.pack(fill="x", padx=10, pady=(2, 4))

        sw_row = ttk.Frame(sw_frm); sw_row.pack(fill="x", padx=6, pady=(4, 2))
        for label, var_name, default, width in [
            ("Workers:",     "_workers_var", SWEEP_WORKERS,    5),
            ("Timeout (s):", "_timeout_var", SWEEP_TIMEOUT_S,  5),
            ("Ckpt every:",  "_ckpt_var",    SWEEP_CKPT_EVERY, 5),
        ]:
            f = ttk.Frame(sw_row); f.pack(side="left", padx=(0, 10))
            ttk.Label(f, text=label, anchor="w").pack(anchor="w")
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(f, textvariable=var, width=width).pack(anchor="w")

        btn_row = ttk.Frame(sw_frm); btn_row.pack(fill="x", padx=6, pady=(2, 4))
        self._sweep_btn = ttk.Button(btn_row, text="Start Solving",
                                     command=self._on_sweep_start)
        self._sweep_btn.pack(side="left")
        self._pause_btn = ttk.Button(btn_row, text="Pause",
                                     command=self._on_pause, state="disabled")
        self._pause_btn.pack(side="left", padx=(6, 0))
        self._sweep_status = tk.StringVar(value="—")
        ttk.Label(btn_row, textvariable=self._sweep_status,
                  foreground="gray").pack(side="left", padx=8)

        prog_frm = ttk.LabelFrame(parent, text="Sweep Progress")
        prog_frm.pack(fill="x", padx=10, pady=(2, 4))

        self._sweep_bar = tk.Canvas(prog_frm, height=14, bg="#3c3c3c",
                                    highlightthickness=1,
                                    highlightbackground="#606060")
        self._sweep_bar.pack(fill="x", padx=6, pady=(4, 2))
        self._sweep_bar.bind("<Configure>", lambda _: self._redraw_sweep_bar())

        sstat_row = ttk.Frame(prog_frm); sstat_row.pack(fill="x", padx=6, pady=(0, 4))
        self._sstat_done = tk.StringVar(value=f"Done: {n_ok}/{n_sam}")
        self._sstat_ok   = tk.StringVar(value=f"OK: {n_ok}")
        for sv in (self._sstat_done, self._sstat_ok):
            ttk.Label(sstat_row, textvariable=sv).pack(side="left", padx=10)
        ttk.Button(sstat_row, text="Refresh",
                   command=self._refresh_sweep_status).pack(side="right", padx=6)

        con_frm = ttk.LabelFrame(parent, text="Console")
        con_frm.pack(fill="both", expand=True, padx=10, pady=(2, 6))
        btn_bar = ttk.Frame(con_frm); btn_bar.pack(fill="x", padx=4, pady=(2, 0))
        ttk.Button(btn_bar, text="Clear", command=self._clear_console).pack(side="right")
        self._console = tk.Text(con_frm, height=10, state="disabled",
                                wrap="none", font=("Consolas", 8),
                                bg="#1e1e1e", fg="#d4d4d4",
                                insertbackground="white", relief="flat", bd=0)
        vsb = ttk.Scrollbar(con_frm, orient="vertical",   command=self._console.yview)
        hsb = ttk.Scrollbar(con_frm, orient="horizontal", command=self._console.xview)
        self._console.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._console.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 2))

        self._n_total_samples     = n_sam
        self._sweep_covered_uuids = set()
        self._sweep_n_ok          = n_ok
        self._refresh_sweep_status()

    def _populate_domain_info(self, parent, folder):
        dp   = _domain_path(folder)
        n    = _n_samples(folder)
        n_ok = _n_ok_results(folder)

        row = ttk.Frame(parent); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=f"Samples: {n:,}", foreground="#80c8ff").pack(side="left", padx=8)
        ttk.Label(row, text=f"OK results: {n_ok:,}", foreground="#4ec94e").pack(side="left", padx=8)

        if not os.path.exists(dp):
            ttk.Label(row, text="domain.json not found", foreground="red").pack(side="left", padx=8)
            return
        try:
            with open(dp) as f:
                d = json.load(f)
        except Exception:
            ttk.Label(row, text="domain.json unreadable", foreground="red").pack(side="left", padx=8)
            return

        g = d.get("global", {})
        freq = g.get("freq_hz", [])
        gap  = g.get("pcb_gap_mm", [])
        res  = g.get("resolution_mm", "?")
        pieces = []
        if freq:
            pieces.append(f"Freq: {freq[0]/1000:.0f}–{freq[-1]/1000:.0f} kHz")
        if gap:
            pieces.append(f"PCB gap: {gap[0]}–{gap[-1]} mm")
        pieces.append(f"Res: {res} mm")
        ttk.Label(row, text="  |  ".join(pieces),
                  foreground="gray", font=("Consolas", 8)).pack(side="left", padx=8)

    # ================================================================ PANEL: RESULTS

    def _build_panel_results(self, parent):
        folder = self._folder.get()
        n_sam  = _n_samples(folder)
        n_ok   = _n_ok_results(folder)

        info = ttk.LabelFrame(parent, text="Dataset")
        info.pack(fill="x", padx=10, pady=(8, 4))
        row = ttk.Frame(info); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=f"{n_ok:,} OK rows", foreground="#4ec94e").pack(side="left", padx=8)
        ttk.Label(row, text=f"of {n_sam:,} samples", foreground="gray").pack(side="left")
        ttk.Label(row, text=f"→ {os.path.basename(folder)}/results.json",
                  foreground="#80c8ff", font=("Consolas", 8)).pack(side="left", padx=12)

        nn_frm = ttk.LabelFrame(parent, text="Train Surrogate NN")
        nn_frm.pack(fill="x", padx=10, pady=(2, 4))

        hp_row = ttk.Frame(nn_frm); hp_row.pack(fill="x", padx=6, pady=4)
        for label, attr, default, width in [
            ("Epochs:",    "_nn_epochs_var", NN_EPOCHS,    6),
            ("Batch:",     "_nn_batch_var",  NN_BATCH,     5),
            ("LR:",        "_nn_lr_var",     NN_LR,        7),
            ("Val split:", "_nn_val_var",    NN_VAL_SPLIT, 5),
        ]:
            f = ttk.Frame(hp_row); f.pack(side="left", padx=(0, 8))
            ttk.Label(f, text=label, anchor="w").pack(anchor="w")
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(f, textvariable=var, width=width).pack(anchor="w")

        ttk.Separator(hp_row, orient="vertical").pack(side="left", fill="y", padx=8)

        f = ttk.Frame(hp_row); f.pack(side="left", padx=(0, 8))
        self._train_btn = ttk.Button(f, text="Start Training",
                                     command=self._on_train_start)
        self._train_btn.pack(anchor="w", pady=(0, 2))
        self._stop_train_btn = ttk.Button(f, text="Stop",
                                          command=self._on_train_stop,
                                          state="disabled")
        self._stop_train_btn.pack(anchor="w")

        f = ttk.Frame(hp_row); f.pack(side="left", padx=(0, 8))
        self._hp_sweep_btn = ttk.Button(
            f, text="Run HP Sweep",
            command=self._on_hp_sweep_start)
        self._hp_sweep_btn.pack(anchor="w", pady=(0, 2))
        ttk.Label(f, text="batch × lr × lr-drop",
                  foreground="gray",
                  font=("TkDefaultFont", 7)).pack(anchor="w")

        f = ttk.Frame(hp_row); f.pack(side="left")
        self._train_status    = tk.StringVar(value="—")
        self._train_epoch_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self._train_status, foreground="gray").pack(anchor="w")
        ttk.Label(f, textvariable=self._train_epoch_var,
                  foreground="#4ec94e", font=("Consolas", 8)).pack(anchor="w")

        log_hdr = ttk.Frame(nn_frm); log_hdr.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(log_hdr, text="Training log", foreground="gray",
                  font=("Consolas", 8)).pack(side="left")
        ttk.Button(log_hdr, text="Clear", command=self._clear_train_log).pack(side="right")

        self._train_log = tk.Text(nn_frm, height=28, state="disabled",
                                  wrap="word", font=("Consolas", 8),
                                  bg="#1e1e1e", fg="#d4d4d4",
                                  insertbackground="white", relief="flat", bd=0)
        tvsb = ttk.Scrollbar(nn_frm, orient="vertical", command=self._train_log.yview)
        self._train_log.configure(yscrollcommand=tvsb.set)
        tvsb.pack(side="right", fill="y", padx=(0, 2), pady=2)
        self._train_log.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 4))

    # ================================================================ PANEL: TRAINED

    def _build_panel_trained(self, parent):
        folder = self._folder.get()

        top_row = ttk.Frame(parent); top_row.pack(fill="x", padx=12, pady=(12, 4))
        ttk.Label(top_row,
                  text=f"Model trained — {os.path.basename(folder)}",
                  foreground="#4ec94e",
                  font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Button(top_row, text="Next Tab →", command=self._go_next_tab).pack(side="right")
        ttk.Button(top_row, text="Delete Model",
                   command=self._on_delete_model).pack(side="right", padx=(0, 6))

        plots_dir = _hp_plots_dir(folder)
        sweep_plots = []
        if os.path.isdir(plots_dir):
            sweep_plots = sorted(
                os.path.join(plots_dir, f)
                for f in os.listdir(plots_dir)
                if f.lower().endswith(".png")
            )

        if sweep_plots:
            self._build_sweep_grid(parent, folder, sweep_plots)
        else:
            plot = _loss_plot_path(folder)
            if os.path.exists(plot):
                try:
                    from PIL import Image, ImageTk
                    img_pil = Image.open(plot)
                    img_pil.thumbnail((700, 350))
                    img_tk  = ImageTk.PhotoImage(img_pil)
                    lbl = ttk.Label(parent, image=img_tk)
                    lbl.image = img_tk
                    lbl.pack(pady=8)
                except ImportError:
                    ttk.Label(parent, text=f"(install Pillow to preview {plot})",
                              foreground="gray").pack(pady=4)
                except Exception as e:
                    ttk.Label(parent, text=f"Could not load plot: {e}",
                              foreground="red").pack(pady=4)

        art_row = ttk.Frame(parent); art_row.pack(pady=(4, 2))
        for name, fn in [("surrogate_model.pth", _model_path),
                          ("x_scaler.pkl",  lambda f: os.path.join(f, "x_scaler.pkl")),
                          ("y_scaler.pkl",  lambda f: os.path.join(f, "y_scaler.pkl")),
                          ("loss_curve.png", _loss_plot_path)]:
            color = "#4ec94e" if os.path.exists(fn(folder)) else "red"
            ttk.Label(art_row, text=name, foreground=color,
                      font=("Consolas", 8)).pack(side="left", padx=6)

        ttk.Label(parent, text=f"Folder: {folder}",
                  foreground="gray", font=("Consolas", 8)).pack(pady=(4, 0))

    def _build_sweep_grid(self, parent, folder, plot_paths):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            ttk.Label(parent,
                      text="(install Pillow to preview sweep plots)",
                      foreground="gray").pack(pady=8)
            return

        hdr = ttk.Frame(parent); hdr.pack(fill="x", padx=12, pady=(4, 2))
        ttk.Label(hdr,
                  text=f"HP sweep — {len(plot_paths)} trials  "
                       f"(click a thumbnail to open full size)",
                  foreground="#80c8ff").pack(side="left")

        # Scrollable canvas hosting the grid.
        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        canvas = tk.Canvas(outer, highlightthickness=0,
                           background=self._get_bg())
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        canvas_win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_win, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        # Build thumbnail grid — 4 columns, tightly packed.
        thumb_w, thumb_h = 480, 270
        cols = 5
        self._sweep_thumb_refs = []  # prevent GC

        for i, path in enumerate(plot_paths):
            try:
                pil = Image.open(path)
                pil.thumbnail((thumb_w, thumb_h))
                tkimg = ImageTk.PhotoImage(pil)
            except Exception as exc:
                ttk.Label(inner, text=f"(load failed: {os.path.basename(path)}: {exc})",
                          foreground="red").grid(row=i // cols, column=i % cols,
                                                 padx=2, pady=2)
                continue
            self._sweep_thumb_refs.append(tkimg)

            cell = ttk.Frame(inner)
            cell.grid(row=i // cols, column=i % cols, padx=1, pady=1, sticky="n")

            btn = ttk.Button(cell, image=tkimg, padding=0,
                             command=lambda p=path: self._open_path(p))
            btn.image = tkimg
            btn.pack()
            ttk.Label(cell, text=os.path.basename(path),
                      foreground="gray",
                      font=("Consolas", 8)).pack(pady=(0, 0))

    def _get_bg(self):
        try:
            return ttk.Style().lookup("TFrame", "background") or "#2b2b2b"
        except Exception:
            return "#2b2b2b"

    def _open_path(self, path):
        try:
            os.startfile(path)  # Windows: open with default image viewer
        except Exception as exc:
            self._log(f"[open] {path}: {exc}")

    # ================================================================ log helpers

    def _log(self, text: str):
        self._log_queue.put(text)

    def _flush_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if not hasattr(self, "_console"):
                    break
                self._console.configure(state="normal")
                self._console.insert("end", msg + "\n")
                self._console.see("end")
                self._console.configure(state="disabled")
        except queue.Empty:
            pass

    def _clear_console(self):
        if not hasattr(self, "_console"):
            return
        self._console.configure(state="normal")
        self._console.delete("1.0", "end")
        self._console.configure(state="disabled")

    def _log_train(self, text: str):
        self._train_log_queue.put(text)

    def _flush_train_log(self):
        try:
            while True:
                msg = self._train_log_queue.get_nowait()
                if not hasattr(self, "_train_log"):
                    continue
                self._train_log.configure(state="normal")
                self._train_log.insert("end", msg + "\n")
                self._train_log.see("end")
                self._train_log.configure(state="disabled")
                if msg.startswith("Epoch") and hasattr(self, "_train_epoch_var"):
                    self._train_epoch_var.set(msg.strip())
        except queue.Empty:
            pass

    def _clear_train_log(self):
        if not hasattr(self, "_train_log"):
            return
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
            if hasattr(self, "_sweep_status"):
                self._sweep_status.set("Finished" if rc == 0 else f"Exited ({rc})")
            self._log(f"[sweep] Process exited {'OK' if rc == 0 else f'code {rc}'}.")
            if hasattr(self, "_sweep_btn"):
                self._sweep_btn.config(state="normal")
            if hasattr(self, "_pause_btn"):
                self._pause_btn.config(state="disabled", text="Pause")
            self._refresh_sweep_status()
            self.after(1000, self._do_state_check)

    def _check_train_done(self):
        if self._train_proc is not None and self._train_proc.poll() is not None:
            rc = self._train_proc.returncode
            self._train_proc   = None
            self._train_thread = None
            if hasattr(self, "_train_status"):
                self._train_status.set("Done" if rc == 0 else f"Error ({rc})")
            if rc == 0:
                self._log_train("[train] Finished successfully.")
            else:
                self._log_train(f"[train] Exited with code {rc}.")
            if hasattr(self, "_train_btn"):
                self._train_btn.config(state="normal")
            if hasattr(self, "_hp_sweep_btn"):
                self._hp_sweep_btn.config(state="normal")
            if hasattr(self, "_stop_train_btn"):
                self._stop_train_btn.config(state="disabled")
            self.after(500, self._do_state_check)

    # ================================================================ sweep status

    def _refresh_sweep_status(self):
        folder = self._folder.get()
        if not folder or not hasattr(self, "_sweep_bar"):
            return
        n_total = _n_samples(folder)
        self._n_total_samples = n_total

        rp = _results_path(folder)
        if not os.path.exists(rp):
            self._sweep_covered_uuids = set()
            self._sweep_n_ok = 0
            if hasattr(self, "_sstat_done"):
                self._sstat_done.set(f"Done: 0/{n_total}")
                self._sstat_ok.set("OK: 0")
            self._redraw_sweep_bar()
            return
        try:
            with open(rp) as f:
                d = json.load(f)
        except Exception:
            return
        covered = {r.get("uuid") for r in d.get("results", []) if isinstance(r, dict) and r.get("uuid")}
        self._sweep_covered_uuids = covered
        self._sweep_n_ok = len(covered)
        pct = f"{100*len(covered)//n_total}%" if n_total else "—"
        if hasattr(self, "_sstat_done"):
            self._sstat_done.set(f"Done: {len(covered)}/{n_total}  ({pct})")
            self._sstat_ok.set(f"OK: {len(covered)}")
        self._redraw_sweep_bar()

    def _redraw_sweep_bar(self):
        if not hasattr(self, "_sweep_bar"):
            return
        c = self._sweep_bar
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 2 or self._n_total_samples == 0:
            return
        n_total = self._n_total_samples
        n_ok    = self._sweep_n_ok
        if n_ok > 0:
            x_ok = max(1, int(w * n_ok / n_total))
            c.create_rectangle(0, 1, x_ok, h - 1, fill="#4ec94e", outline="")

    # ================================================================ generate

    def _on_generate(self):
        if self._gen_thread and self._gen_thread.is_alive():
            return
        folder = self._folder.get()
        if not folder:
            self._log("[gen] ERROR: No folder selected.")
            return
        try:
            cfg = self._build_domain_config()
        except (ValueError, KeyError) as exc:
            self._log(f"[gen] ERROR building config: {exc}")
            return

        self._gen_btn.config(state="disabled")
        self._gen_status.set("Running…")
        self._log(f"[gen] Generating {cfg['n_total']} samples into {folder} …")

        cfg_path = os.path.join(folder, "_gen_config.json")
        os.makedirs(folder, exist_ok=True)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        def _run():
            try:
                cmd = [sys.executable, _GEN_SCRIPT,
                       "--config",  cfg_path,
                       "--out-dir", folder,
                       "--n",       str(cfg["n_total"])]
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
                self.after(0, self._do_state_check)

        self._gen_thread = threading.Thread(target=_run, daemon=True)
        self._gen_thread.start()

    # ================================================================ sweep

    def _on_sweep_start(self):
        if self._sweep_proc is not None:
            return
        folder = self._folder.get()
        if not folder:
            self._log("[sweep] ERROR: No folder selected.")
            return
        samples = _samples_path(folder)
        if not os.path.exists(samples):
            self._log("[sweep] ERROR: lhs_samples.json not found.")
            return
        try:
            workers = int(self._workers_var.get())
            timeout = int(self._timeout_var.get())
            ckpt    = int(self._ckpt_var.get())
        except ValueError:
            self._log("[sweep] ERROR: invalid numeric parameter.")
            return

        sf = _stop_flag(folder)
        if os.path.exists(sf):
            os.remove(sf)

        out = _results_path(folder)
        cmd = [sys.executable, _SWEEP_SCRIPT,
               "--samples",          samples,
               "--out",              out,
               "--workers",          str(workers),
               "--timeout",          str(timeout),
               "--checkpoint-every", str(ckpt)]

        self._log(f"[sweep] workers={workers} timeout={timeout}s")
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
                    self.after(0, self._refresh_sweep_status)

        self._sweep_thread = threading.Thread(target=_drain, daemon=True)
        self._sweep_thread.start()

    def _on_pause(self):
        folder = self._folder.get()
        if not folder or self._sweep_proc is None:
            return
        sf = _stop_flag(folder)
        if os.path.exists(sf):
            try:
                os.remove(sf)
            except OSError:
                pass
            self._pause_btn.config(text="Pause")
            self._sweep_status.set("Running…")
            self._log("[sweep] Pause cancelled.")
        else:
            open(sf, "w").close()
            self._pause_btn.config(text="Cancel Pause")
            self._sweep_status.set("Stopping…")
            self._log("[sweep] Stop flag set — terminating workers.")

    # ================================================================ train

    def _on_train_start(self):
        if self._train_proc is not None:
            return
        folder = self._folder.get()
        if not folder:
            self._log_train("[train] ERROR: No folder selected.")
            return
        data_file = _results_path(folder)
        if not os.path.exists(data_file):
            self._log_train("[train] ERROR: results.json not found — run sweep first.")
            return
        if not os.path.exists(_TRAIN_SCRIPT):
            self._log_train(f"[train] ERROR: train_surrogate.py not found at {_TRAIN_SCRIPT}")
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
        if hasattr(self, "_hp_sweep_btn"):
            self._hp_sweep_btn.config(state="disabled")
        self._stop_train_btn.config(state="normal")
        self._train_status.set("Running…")
        self._train_epoch_var.set("")
        self._log_train(f"[train] epochs={epochs} batch={batch} lr={lr} val={val}")

        env = os.environ.copy()
        env["SURROGATE_DATA"]       = data_file
        env["SURROGATE_OUTPUT_DIR"] = folder
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

    def _on_hp_sweep_start(self):
        if self._train_proc is not None:
            return
        folder = self._folder.get()
        if not folder:
            self._log_train("[hp-sweep] ERROR: No folder selected.")
            return
        data_file = _results_path(folder)
        if not os.path.exists(data_file):
            self._log_train("[hp-sweep] ERROR: results.json not found.")
            return
        if not os.path.exists(_HP_SWEEP_SCRIPT):
            self._log_train(f"[hp-sweep] ERROR: hp_sweep.py not found at {_HP_SWEEP_SCRIPT}")
            return
        try:
            epochs = int(self._nn_epochs_var.get())
            val    = float(self._nn_val_var.get())
            assert 0 < val < 1
        except (ValueError, AssertionError):
            self._log_train("[hp-sweep] ERROR: invalid epochs/val split.")
            return

        self._train_btn.config(state="disabled")
        self._hp_sweep_btn.config(state="disabled")
        self._stop_train_btn.config(state="normal")
        self._train_status.set("HP sweep…")
        self._train_epoch_var.set("")
        self._log_train(f"[hp-sweep] epochs={epochs} val={val}  "
                        f"(batches=[1024,2048,4096] lrs=[0.01,0.005,0.001,0.0005] "
                        f"drops=[0,700])")

        env = os.environ.copy()
        env["SURROGATE_DATA"]       = data_file
        env["SURROGATE_OUTPUT_DIR"] = folder
        env["SWEEP_EPOCHS"]         = str(epochs)
        env["SWEEP_VAL_SPLIT"]      = str(val)
        # Defaults for batches/lrs/lr_drops are baked into hp_sweep.py.

        try:
            self._train_proc = subprocess.Popen(
                [sys.executable, "-u", _HP_SWEEP_SCRIPT],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=_NN_DIR, env=env)
        except Exception as exc:
            self._log_train(f"[hp-sweep] Failed to launch: {exc}")
            self._train_btn.config(state="normal")
            self._hp_sweep_btn.config(state="normal")
            self._stop_train_btn.config(state="disabled")
            self._train_status.set("Error")
            return

        def _drain():
            for line in self._train_proc.stdout:
                self._log_train(line.rstrip())

        self._train_thread = threading.Thread(target=_drain, daemon=True)
        self._train_thread.start()

    def _on_delete_model(self):
        import shutil
        folder = self._folder.get()
        if not folder or not os.path.isdir(folder):
            return
        targets = [
            _model_path(folder),
            os.path.join(folder, "x_scaler.pkl"),
            os.path.join(folder, "y_scaler.pkl"),
            _loss_plot_path(folder),
        ]
        for p in targets:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        shutil.rmtree(_hp_plots_dir(folder), ignore_errors=True)
        self._do_state_check()

    def _on_train_stop(self):
        if self._train_proc is None:
            return
        try:
            self._train_proc.terminate()
        except Exception:
            pass
        if hasattr(self, "_train_status"):
            self._train_status.set("Stopped")
        self._log_train("[train] Stopped by user.")
        if hasattr(self, "_train_btn"):
            self._train_btn.config(state="normal")
        if hasattr(self, "_hp_sweep_btn"):
            self._hp_sweep_btn.config(state="normal")
        if hasattr(self, "_stop_train_btn"):
            self._stop_train_btn.config(state="disabled")
