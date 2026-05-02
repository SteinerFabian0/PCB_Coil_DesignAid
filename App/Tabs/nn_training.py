#!/usr/bin/env python3
"""
NN Training tab — folder-centric state machine.

The tab shows one of four panels depending on folder contents:
  EMPTY   → LHS generation form
  SAMPLES → Sweep controls + progress
  RESULTS → Training controls + log
  TRAINED → Training plot + auto-redirect to next tab
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

_GEN_SCRIPT   = os.path.join(_SCRIPTS_DIR, "generate_lhs_samples.py")
_SWEEP_SCRIPT = os.path.join(_SCRIPTS_DIR, "run_sweep.py")
_TRAIN_SCRIPT = os.path.join(_SCRIPTS_DIR, "train_surrogate.py")

for _p in (_NN_DIR, _MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POLL_MS      = 600
FOLDER_POLL  = 3000  # ms between folder-state re-checks

_GUI_SCALE   = float(os.environ.get("COIL_GUI_SCALE", "1.0"))
_SIDE_WIDTH  = int(260 * _GUI_SCALE)
_LAYER_LABELS = ["L1 (outer)", "L2 (inner)", "L3 (inner)", "L4 (outer)"]

# ---------------------------------------------------------------------------
# Folder-state detection
# ---------------------------------------------------------------------------

def _samples_path(folder):   return os.path.join(folder, "lhs_samples.json")
def _domain_path(folder):    return os.path.join(folder, "domain.json")
def _results_path(folder):   return os.path.join(folder, "results.json")
def _model_path(folder):     return os.path.join(folder, "surrogate_model.pth")
def _loss_plot_path(folder): return os.path.join(folder, "loss_curve.png")
def _stop_flag(folder):      return os.path.join(folder, "STOP_SWEEP")

# States
ST_EMPTY    = "empty"
ST_SAMPLES  = "samples"
ST_RESULTS  = "results"
ST_TRAINED  = "trained"


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
    """True if results.json has at least as many OK rows as samples."""
    sp = _samples_path(folder)
    rp = _results_path(folder)
    if not os.path.exists(sp) or not os.path.exists(rp):
        return False
    try:
        with open(sp) as f:
            n_samples = json.load(f).get("meta", {}).get("n_samples", 0)
        with open(rp) as f:
            d = json.load(f)
        n_ok = sum(1 for r in d.get("results", []) if r.get("ok"))
        return n_samples > 0 and n_ok >= n_samples
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
        return sum(1 for r in d.get("results", []) if r.get("ok"))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------

TX_LAYERS_DEFAULT       = [True, True, True, False]
TX_OUTER_OZ_LO          = "1.0";  TX_OUTER_OZ_HI = "1.0"
TX_INNER_OZ_LO          = "0.5";  TX_INNER_OZ_HI = "1.0"
TX_OUTER_GAP_LO         = "0.2";  TX_OUTER_GAP_HI = "0.2"
TX_INNER_GAP_LO         = "0.8";  TX_INNER_GAP_HI = "1.3"
TX_ID_MIN               = "20.0"; TX_OD_MAX = "56.0"
TX_SPACING_LO           = "0.16"; TX_SPACING_HI = "0.16"
TX_PORT_INSIDE_DEFAULT  = False;  TX_PORT_OUTSIDE_DEFAULT = True

RX_LAYERS_DEFAULT       = [True, True, True, True]
RX_OUTER_OZ_LO          = "1.0";  RX_OUTER_OZ_HI = "1.0"
RX_INNER_OZ_LO          = "0.5";  RX_INNER_OZ_HI = "1.0"
RX_OUTER_GAP_LO         = "0.2";  RX_OUTER_GAP_HI = "0.2"
RX_INNER_GAP_LO         = "0.6";  RX_INNER_GAP_HI = "1.0"
RX_ID_MIN               = "20.0"; RX_OD_MAX = "55.0"
RX_SPACING_LO           = "0.16"; RX_SPACING_HI = "0.16"
RX_PORT_INSIDE_DEFAULT  = True;   RX_PORT_OUTSIDE_DEFAULT = False

RESOLUTION_MM  = "1.5";  NHINC = "1"; NWINC = "3"
PCB_GAP_LO     = "2.4";  PCB_GAP_HI = "2.8"
FREQ_MIN_HZ    = "110000"; FREQ_MAX_HZ = "140000"
TOTAL_SAMPLES  = "64000"

SWEEP_WORKERS  = "6"; SWEEP_TIMEOUT_S = "360"; SWEEP_CKPT_EVERY = "50"
NN_EPOCHS      = "300"; NN_BATCH = "512"; NN_LR = "0.0005"; NN_VAL_SPLIT = "0.15"


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

        self._folder       = tk.StringVar()
        self._state        = None
        self._log_queue    = queue.Queue()
        self._train_log_queue = queue.Queue()

        self._gen_thread   = None
        self._sweep_proc   = None
        self._sweep_thread = None
        self._train_proc   = None
        self._train_thread = None

        self._n_total_samples    = 0
        self._sweep_covered_uuids = set()
        self._sweep_n_ok         = 0

        self._build_header()
        self._content_frame = ttk.Frame(self)
        self._content_frame.pack(fill="both", expand=True)

        self._current_panel  = None
        self._folder_poll_id = None
        self._last_folder    = None  # track folder changes independently of state

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
        ttk.Button(hdr, text="New…", command=self._new_folder).pack(side="left", padx=(4, 0))

        self._folder.trace_add("write", lambda *_: self.after(100, self._on_folder_changed))

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
        """Called when the folder entry changes — always force a rebuild."""
        folder = self._folder.get()
        self._last_folder = folder
        new_st = _folder_state(folder)
        self._state = new_st
        self._rebuild_content()
        self._schedule_folder_poll()

    def _schedule_folder_poll(self):
        if self._folder_poll_id is not None:
            self.after_cancel(self._folder_poll_id)
        self._folder_poll_id = self.after(FOLDER_POLL, self._do_state_check)

    def _do_state_check(self):
        folder  = self._folder.get()
        new_st  = _folder_state(folder)
        folder_changed = (folder != self._last_folder)
        self._last_folder = folder
        if new_st != self._state or folder_changed:
            self._state = new_st
            self._rebuild_content()
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
        """Return the currently selected model folder path (may be empty string)."""
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
        ttk.Label(parent, text="Folder is empty — configure and generate LHS samples.",
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

        self._tx = self._build_side(tx_frm, "tx")
        self._rx = self._build_side(rx_frm, "rx")
        self._build_right_column(right)

    def _build_side(self, parent, side: str) -> dict:
        v = {}
        _layers_def = TX_LAYERS_DEFAULT if side == "tx" else RX_LAYERS_DEFAULT
        _oo_lo  = TX_OUTER_OZ_LO  if side == "tx" else RX_OUTER_OZ_LO
        _oo_hi  = TX_OUTER_OZ_HI  if side == "tx" else RX_OUTER_OZ_HI
        _io_lo  = TX_INNER_OZ_LO  if side == "tx" else RX_INNER_OZ_LO
        _io_hi  = TX_INNER_OZ_HI  if side == "tx" else RX_INNER_OZ_HI
        _og_lo  = TX_OUTER_GAP_LO if side == "tx" else RX_OUTER_GAP_LO
        _og_hi  = TX_OUTER_GAP_HI if side == "tx" else RX_OUTER_GAP_HI
        _ig_lo  = TX_INNER_GAP_LO if side == "tx" else RX_INNER_GAP_LO
        _ig_hi  = TX_INNER_GAP_HI if side == "tx" else RX_INNER_GAP_HI
        _id_min = TX_ID_MIN        if side == "tx" else RX_ID_MIN
        _od_max = TX_OD_MAX        if side == "tx" else RX_OD_MAX
        _sp_lo  = TX_SPACING_LO   if side == "tx" else RX_SPACING_LO
        _sp_hi  = TX_SPACING_HI   if side == "tx" else RX_SPACING_HI
        _p_in   = TX_PORT_INSIDE_DEFAULT  if side == "tx" else RX_PORT_INSIDE_DEFAULT
        _p_out  = TX_PORT_OUTSIDE_DEFAULT if side == "tx" else RX_PORT_OUTSIDE_DEFAULT

        lyr_frm = ttk.LabelFrame(parent, text="Active Layers")
        lyr_frm.pack(fill="x", padx=4, pady=(4, 2))
        v["layers"] = []
        for i, lbl in enumerate(_LAYER_LABELS):
            bv = tk.BooleanVar(value=_layers_def[i])
            ttk.Checkbutton(lyr_frm, text=lbl, variable=bv).pack(anchor="w", padx=6)
            v["layers"].append(bv)

        cu_frm = ttk.LabelFrame(parent, text="Copper Weight (oz)")
        cu_frm.pack(fill="x", padx=4, pady=2)
        v["outer_oz_lo"] = tk.StringVar(value=_oo_lo)
        v["outer_oz_hi"] = tk.StringVar(value=_oo_hi)
        v["inner_oz_lo"] = tk.StringVar(value=_io_lo)
        v["inner_oz_hi"] = tk.StringVar(value=_io_hi)
        _range_row(cu_frm, "Outer layers:", v["outer_oz_lo"], v["outer_oz_hi"])
        _range_row(cu_frm, "Inner layers:", v["inner_oz_lo"], v["inner_oz_hi"])

        stk_frm = ttk.LabelFrame(parent, text="Stackup Spacing (mm)")
        stk_frm.pack(fill="x", padx=4, pady=2)
        v["outer_gap_lo"] = tk.StringVar(value=_og_lo)
        v["outer_gap_hi"] = tk.StringVar(value=_og_hi)
        v["inner_gap_lo"] = tk.StringVar(value=_ig_lo)
        v["inner_gap_hi"] = tk.StringVar(value=_ig_hi)
        _range_row(stk_frm, "Outer gap:", v["outer_gap_lo"], v["outer_gap_hi"])
        _range_row(stk_frm, "Inner gap:", v["inner_gap_lo"], v["inner_gap_hi"])

        geo_frm = ttk.LabelFrame(parent, text="Geometry")
        geo_frm.pack(fill="x", padx=4, pady=2)
        v["id_min"] = tk.StringVar(value=_id_min)
        v["od_max"] = tk.StringVar(value=_od_max)
        _range_row(geo_frm, "Diameter (mm):", v["id_min"], v["od_max"], label_width=14)
        v["spacing_lo"] = tk.StringVar(value=_sp_lo)
        v["spacing_hi"] = tk.StringVar(value=_sp_hi)
        _range_row(geo_frm, "Spacing (mm):", v["spacing_lo"], v["spacing_hi"], label_width=14)

        port_frm = ttk.LabelFrame(parent, text="Port Location")
        port_frm.pack(fill="x", padx=4, pady=2)
        v["port_inside"]  = tk.BooleanVar(value=_p_in)
        v["port_outside"] = tk.BooleanVar(value=_p_out)

        def _guard(cv, ov):
            def _cb(*_):
                if not cv.get() and not ov.get():
                    ov.set(True)
            return _cb

        ttk.Checkbutton(port_frm, text="Port outside",
                        variable=v["port_outside"]).pack(anchor="w", padx=6)
        ttk.Checkbutton(port_frm, text="Port inside",
                        variable=v["port_inside"]).pack(anchor="w", padx=6)
        v["port_outside"].trace_add("write", _guard(v["port_outside"], v["port_inside"]))
        v["port_inside"].trace_add("write",  _guard(v["port_inside"],  v["port_outside"]))
        return v

    def _build_right_column(self, parent):
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

        row2 = ttk.Frame(shared); row2.pack(fill="x", padx=4, pady=(0, 2))
        self._pcb_gap_lo_var = tk.StringVar(value=PCB_GAP_LO)
        self._pcb_gap_hi_var = tk.StringVar(value=PCB_GAP_HI)
        _range_row(row2, "PCB gap (mm):", self._pcb_gap_lo_var, self._pcb_gap_hi_var,
                   label_width=14, entry_width=6, pady=0)

        row3 = ttk.Frame(shared); row3.pack(fill="x", padx=4, pady=(0, 4))
        self._fmin_var = tk.StringVar(value=FREQ_MIN_HZ)
        self._fmax_var = tk.StringVar(value=FREQ_MAX_HZ)
        _range_row(row3, "Freq range (Hz):", self._fmin_var, self._fmax_var,
                   label_width=14, entry_width=10, pady=0)

        gen_frm = ttk.LabelFrame(parent, text="Generate LHS Samples")
        gen_frm.pack(fill="x", padx=4, pady=2)

        gr = ttk.Frame(gen_frm); gr.pack(fill="x", padx=6, pady=4)

        f = ttk.Frame(gr); f.pack(side="left", padx=(0, 10))
        ttk.Label(f, text="Total samples:", anchor="w").pack(anchor="w")
        self._n_var = tk.StringVar(value=TOTAL_SAMPLES)
        ttk.Entry(f, textvariable=self._n_var, width=8).pack(anchor="w")

        f = ttk.Frame(gr); f.pack(side="left", padx=(0, 10))
        self._gp_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Add ground circles",
                        variable=self._gp_enabled_var,
                        command=self._update_gp_state).pack(anchor="w")
        gp_row = ttk.Frame(f); gp_row.pack(anchor="w", pady=(2, 0))
        ttk.Label(gp_row, text="Min (mm):").pack(side="left", padx=(20, 2))
        self._gp_min_var = tk.StringVar(value="18.0")
        self._gp_min_entry = ttk.Entry(gp_row, textvariable=self._gp_min_var, width=6)
        self._gp_min_entry.pack(side="left")
        ttk.Label(gp_row, text="Max (mm):").pack(side="left", padx=(6, 2))
        self._gp_max_var = tk.StringVar(value="24.0")
        self._gp_max_entry = ttk.Entry(gp_row, textvariable=self._gp_max_var, width=6)
        self._gp_max_entry.pack(side="left")
        self._update_gp_state()

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
        vsb2 = ttk.Scrollbar(con_frm, orient="vertical", command=self._console.yview)
        hsb2 = ttk.Scrollbar(con_frm, orient="horizontal", command=self._console.xview)
        self._console.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)
        vsb2.pack(side="right", fill="y")
        hsb2.pack(side="bottom", fill="x")
        self._console.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 2))

    def _update_gp_state(self):
        state = "normal" if self._gp_enabled_var.get() else "disabled"
        if hasattr(self, "_gp_min_entry"):
            self._gp_min_entry.config(state=state)
            self._gp_max_entry.config(state=state)

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
        vsb = ttk.Scrollbar(con_frm, orient="vertical", command=self._console.yview)
        hsb = ttk.Scrollbar(con_frm, orient="horizontal", command=self._console.xview)
        self._console.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._console.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 2))

        self._n_total_samples = n_sam
        self._sweep_covered_uuids = set()
        self._sweep_n_ok = n_ok
        self._refresh_sweep_status()

    def _populate_domain_info(self, parent, folder):
        dp = _domain_path(folder)
        n  = _n_samples(folder)
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
        self._train_status    = tk.StringVar(value="—")
        self._train_epoch_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self._train_status, foreground="gray").pack(anchor="w")
        ttk.Label(f, textvariable=self._train_epoch_var,
                  foreground="#4ec94e", font=("Consolas", 8)).pack(anchor="w")

        log_hdr = ttk.Frame(nn_frm); log_hdr.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(log_hdr, text="Training log", foreground="gray",
                  font=("Consolas", 8)).pack(side="left")
        ttk.Button(log_hdr, text="Clear", command=self._clear_train_log).pack(side="right")

        self._train_log = tk.Text(nn_frm, height=14, state="disabled",
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

        top_row = ttk.Frame(parent)
        top_row.pack(fill="x", padx=12, pady=(12, 4))
        ttk.Label(top_row,
                  text=f"Model trained — {os.path.basename(folder)}",
                  foreground="#4ec94e", font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Button(top_row, text="Next Tab →", command=self._go_next_tab).pack(side="right")

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

        art_row = ttk.Frame(parent); art_row.pack(pady=4)
        for name, fn in [("surrogate_model.pth", _model_path),
                          ("x_scaler.pkl",  lambda f: os.path.join(f, "x_scaler.pkl")),
                          ("y_scaler.pkl",  lambda f: os.path.join(f, "y_scaler.pkl")),
                          ("loss_curve.png", _loss_plot_path)]:
            color = "#4ec94e" if os.path.exists(fn(folder)) else "red"
            ttk.Label(art_row, text=name, foreground=color,
                      font=("Consolas", 8)).pack(side="left", padx=6)

        ttk.Label(parent, text=f"Folder: {folder}",
                  foreground="gray", font=("Consolas", 8)).pack(pady=(4, 0))

    # ================================================================ domain config
    def _build_domain_config(self) -> dict:
        def _flist(lo_var, hi_var):
            return [float(lo_var.get()), float(hi_var.get())]

        def _side_cfg(v: dict) -> dict:
            return {
                "layers_selected":      [bv.get() for bv in v["layers"]],
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

        return {
            "tx": _side_cfg(self._tx),
            "rx": _side_cfg(self._rx),
            "global": {
                "pcb_gap_mm":    _flist(self._pcb_gap_lo_var, self._pcb_gap_hi_var),
                "resolution_mm": float(self._res_var.get()),
                "freq_hz":       [float(self._fmin_var.get()),
                                  float(self._fmax_var.get())],
            },
            "n_total": int(self._n_var.get()),
        }

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
            if rc == 0:
                if hasattr(self, "_sweep_status"):
                    self._sweep_status.set("Finished")
                self._log("[sweep] Process exited OK.")
            else:
                if hasattr(self, "_sweep_status"):
                    self._sweep_status.set(f"Exited (code {rc})")
                self._log(f"[sweep] Process exited with code {rc}.")
            if hasattr(self, "_sweep_btn"):
                self._sweep_btn.config(state="normal")
            if hasattr(self, "_pause_btn"):
                self._pause_btn.config(state="disabled", text="Pause")
            self._refresh_sweep_status()
            # Let folder-state poll promote to RESULTS if sweep is complete
            self.after(1000, self._do_state_check)

    def _check_train_done(self):
        if self._train_proc is not None and self._train_proc.poll() is not None:
            rc = self._train_proc.returncode
            self._train_proc   = None
            self._train_thread = None
            if rc == 0:
                if hasattr(self, "_train_status"):
                    self._train_status.set("Done")
                self._log_train("[train] Finished successfully.")
                self._log("[train] Training complete — model saved.")
            else:
                if hasattr(self, "_train_status"):
                    self._train_status.set(f"Error ({rc})")
                self._log_train(f"[train] Exited with code {rc}.")
            if hasattr(self, "_train_btn"):
                self._train_btn.config(state="normal")
            if hasattr(self, "_stop_train_btn"):
                self._stop_train_btn.config(state="disabled")
            # Trigger state check — will advance to ST_TRAINED if model file appeared
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
        n_ok    = 0
        covered = set()
        for r in d.get("results", []):
            uid = r.get("uuid")
            if uid:
                covered.add(uid)
                if r.get("ok"):
                    n_ok += 1
        self._sweep_covered_uuids = covered
        self._sweep_n_ok = n_ok
        pct = f"{100*len(covered)//n_total}%" if n_total else "—"
        if hasattr(self, "_sstat_done"):
            self._sstat_done.set(f"Done: {len(covered)}/{n_total}  ({pct})")
            self._sstat_ok.set(f"OK: {n_ok}")
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
        n_done  = len(self._sweep_covered_uuids)
        n_ok    = self._sweep_n_ok
        if n_done > 0:
            x_done = max(1, int(w * n_done / n_total))
            c.create_rectangle(0, 1, x_done, h - 1, fill="#4ec94e", outline="")
        if n_ok > 0 and n_ok < n_done:
            x_ok = max(1, int(w * n_ok / n_total))
            c.create_rectangle(x_ok, 1, x_done, h - 1, fill="#2a6e2a", outline="")

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
                if self._gp_enabled_var.get():
                    cmd += ["--ground-circle-enabled", "1",
                            "--ground-circle-min", str(self._gp_min_var.get()),
                            "--ground-circle-max", str(self._gp_max_var.get())]
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
        if hasattr(self, "_stop_train_btn"):
            self._stop_train_btn.config(state="disabled")
