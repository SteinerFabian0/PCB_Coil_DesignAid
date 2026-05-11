#!/usr/bin/env python3
"""
SimulationNN tab — fast coil prediction using the trained surrogate neural network.

Inputs mirror the NN feature set (TX / RX settings + drive frequency).
Outputs mirror the Simulation tab layout (TX results, RX results, coupling, insights).

RX cap is determined automatically via the cap combinator: after the NN predicts
L_rx, the best capacitor combination that resonates closest to the drive frequency
is found and displayed.

Range validation: if any input is outside the LHS-covered range derived from
sweep_results.json, the field is highlighted red and the Run button is disabled.
"""

import os
import sys
import json
import math
import threading
import tkinter as tk
from tkinter import ttk, messagebox

_HERE      = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT  = os.path.dirname(_HERE)
_MODULES   = os.path.join(_APP_ROOT, "Modules")
_NN_DIR     = os.path.join(_APP_ROOT, "NeuralNetwork")
_NN_SCRIPTS = os.path.join(_NN_DIR, "scripts")
_SIMDATA    = os.path.join(_APP_ROOT, "SimulationData")

for _p in (_MODULES, _NN_DIR, _NN_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cap_combinator
import domain_lookup as _dl

_NN_V2_DIR = os.path.join(_NN_DIR, "NN_V2")

TOPOLOGY_CHOICES = [
    ("All parallel",  "parallel"),
    ("All series",    "series"),
    ("1p2 -|- 3p4",   "parallel_pairs_ser"),
]

# Stable topology vocabulary — must match TOPOLOGY_VOCAB in train_surrogate.py
TOPOLOGY_VOCAB = ["parallel", "series", "parallel_pairs_ser"]

# NN output order must match train_surrogate.py OUTPUT_COLS
# ["L_tx_uH", "L_rx_uH", "M_uH", "R_tx_ac", "R_rx_ac"]
OUT_L_TX  = 0
OUT_L_RX  = 1
OUT_M     = 2
OUT_R_TX  = 3
OUT_R_RX  = 4

# Defaults for the new input fields (mirror NN Optimisation defaults).
DEFAULT_PCB_GAP_MM   = "2.4"
DEFAULT_TX_OUTER_GAP = "0.2"
DEFAULT_TX_INNER_GAP = "1.0"
DEFAULT_RX_OUTER_GAP = "0.2"
DEFAULT_RX_INNER_GAP = "0.6"
DEFAULT_SPACING_MM   = "0.16"
DEFAULT_TX_LAYER_OZ  = ["1.0", "1.0", "1.0", "0"]
DEFAULT_RX_LAYER_OZ  = ["1.0", "0.5", "0.5", "1.0"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_nn(model_dir):
    """Import torch/joblib lazily and load the trained model + scalers.
    Returns (model, x_scaler, y_scaler, feature_columns) or raises."""
    import torch
    import joblib
    import numpy as np

    mf = os.path.join(model_dir, "surrogate_model.pth")
    xf = os.path.join(model_dir, "x_scaler.pkl")
    yf = os.path.join(model_dir, "y_scaler.pkl")
    if not os.path.exists(mf):
        raise FileNotFoundError(f"Model not found: {mf}")
    if not os.path.exists(xf):
        raise FileNotFoundError(f"x_scaler not found: {xf}")
    if not os.path.exists(yf):
        raise FileNotFoundError(f"y_scaler not found: {yf}")

    x_scaler = joblib.load(xf)
    y_scaler = joblib.load(yf)

    # Reconstruct model architecture — must match train_surrogate.py
    import torch.nn as nn

    class _CoilNN(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.GELU(),
                nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(),
                nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.GELU(),
                nn.Linear(64, 5),
            )
        def forward(self, x):
            return self.net(x)

    n_in = x_scaler.n_features_in_
    model = _CoilNN(n_in)
    model.load_state_dict(torch.load(mf, map_location="cpu"))
    model.eval()

    # Reconstruct feature column names from the scaler
    try:
        feature_cols = list(x_scaler.feature_names_in_)
    except AttributeError:
        feature_cols = None

    return model, x_scaler, y_scaler, feature_cols, torch, np




def _resonant_cap_nf(L_uH, freq_hz):
    """Ideal capacitance for series resonance at freq_hz given L in µH."""
    if L_uH <= 0 or freq_hz <= 0:
        return None
    L_h = L_uH * 1e-6
    w = 2 * math.pi * freq_hz
    C_f = 1.0 / (w * w * L_h)
    return C_f * 1e9   # → nF


def _series_resonant_freq(L_uH, C_nf):
    if L_uH <= 0 or C_nf <= 0:
        return None
    return 1.0 / (2 * math.pi * math.sqrt(L_uH * 1e-6 * C_nf * 1e-9))


def _q_factor(L_uH, R_ac_ohm, freq_hz):
    if R_ac_ohm <= 0 or freq_hz <= 0:
        return None
    return (2 * math.pi * freq_hz * L_uH * 1e-6) / R_ac_ohm


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

class SimNNTab(ttk.Frame):

    def __init__(self, parent, app=None, **kw):
        super().__init__(parent, **kw)
        self.app = app

        self._ranges  = {}     # field → (min, max) or list for topology
        self._domains = []     # [(batch_num, domain_dict), ...] from domain files
        self._run_btn = None   # set in _build
        self._field_widgets = {}  # field_key → Entry widget (for red highlight)
        self._result_vars = {}    # key → StringVar (output labels)
        self._topo_var = tk.StringVar(value="parallel")
        self._gc_enabled = tk.BooleanVar(value=False)

        self._build()
        self.after(300, self._refresh_ranges)

    # -----------------------------------------------------------------------
    # Build UI
    # -----------------------------------------------------------------------

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # ---- drive frequency (global) -----------------------------------
        freq_frm = ttk.LabelFrame(body, text="Drive Frequency")
        freq_frm.pack(fill="x", padx=8, pady=(8, 4))
        fr = ttk.Frame(freq_frm); fr.pack(fill="x", padx=6, pady=6)
        ttk.Label(fr, text="Drive frequency (Hz):", width=24, anchor="w").pack(side="left")
        self._freq_var = tk.StringVar(value="125000")
        freq_entry = ttk.Entry(fr, textvariable=self._freq_var, width=12)
        freq_entry.pack(side="left", padx=4)
        self._field_widgets["freq_hz"] = freq_entry
        self._range_label_freq = ttk.Label(fr, text="", foreground="#808080")
        self._range_label_freq.pack(side="left", padx=6)
        self._freq_var.trace_add("write", lambda *_: self._on_input_change())

        ttk.Label(fr, text="   PCB gap (mm):").pack(side="left", padx=(20, 2))
        self._pcb_gap_var = tk.StringVar(value=DEFAULT_PCB_GAP_MM)
        pcb_gap_entry = ttk.Entry(fr, textvariable=self._pcb_gap_var, width=8)
        pcb_gap_entry.pack(side="left", padx=2)
        self._field_widgets["pcb_gap_mm"] = pcb_gap_entry
        self._pcb_gap_var.trace_add("write", lambda *_: self._on_input_change())

        # ---- TX / RX split -----------------------------------------------
        split = ttk.Frame(body)
        split.pack(fill="x", padx=8, pady=4)
        split.columnconfigure(0, weight=1, uniform="col")
        split.columnconfigure(1, weight=0)
        split.columnconfigure(2, weight=1, uniform="col")

        sep = tk.Frame(split, bg="#b0b0b0", width=2)
        sep.grid(row=0, column=1, sticky="ns", padx=4)

        # TX inputs
        tx_frm = ttk.LabelFrame(split, text="TX Settings")
        tx_frm.grid(row=0, column=0, sticky="nsew", pady=4)
        self._tx_od_var    = self._input_row(tx_frm, "TX OD (mm):",         "53.0",  "tx_od_mm")
        self._tx_turns_var = self._input_row(tx_frm, "TX Turns:",            "9",     "tx_turns")
        self._tx_width_var = self._input_row(tx_frm, "TX Trace width (mm):", "0.4",   "tx_width")
        self._tx_spacing_var   = self._input_row(tx_frm, "TX Spacing (mm):",  DEFAULT_SPACING_MM,   "tx_spacing_mm")
        self._tx_outer_gap_var = self._input_row(tx_frm, "TX Outer gap (mm):", DEFAULT_TX_OUTER_GAP, "tx_outer_gap_mm")
        self._tx_inner_gap_var = self._input_row(tx_frm, "TX Inner gap (mm):", DEFAULT_TX_INNER_GAP, "tx_inner_gap_mm")
        # TX topology
        tx_topo_row = ttk.Frame(tx_frm); tx_topo_row.pack(fill="x", padx=4, pady=2)
        ttk.Label(tx_topo_row, text="TX Topology:", width=24, anchor="w").pack(side="left")
        self._tx_topo_var = tk.StringVar(value="parallel")
        self._tx_topo_combo = ttk.Combobox(
            tx_topo_row, textvariable=self._tx_topo_var,
            values=[v for _, v in TOPOLOGY_CHOICES], width=20, state="readonly")
        self._tx_topo_combo.pack(side="left", padx=2)
        self._tx_topo_var.trace_add("write", lambda *_: self._on_input_change())
        # TX port inside
        tx_port_row = ttk.Frame(tx_frm); tx_port_row.pack(fill="x", padx=4, pady=2)
        ttk.Label(tx_port_row, text="TX Port inside:", width=24, anchor="w").pack(side="left")
        self._tx_port_inside_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tx_port_row, variable=self._tx_port_inside_var,
                        command=self._on_input_change).pack(side="left")
        # TX layer copper oz (4 layers)
        tx_lyr_frm = ttk.LabelFrame(tx_frm, text="TX Copper oz per layer (0 = inactive)")
        tx_lyr_frm.pack(fill="x", padx=4, pady=(4, 2))
        self._tx_oz_vars = []
        for i, default in enumerate(DEFAULT_TX_LAYER_OZ):
            r = ttk.Frame(tx_lyr_frm); r.pack(fill="x", padx=4, pady=1)
            ttk.Label(r, text=f"L{i+1}:", width=4).pack(side="left")
            v = tk.StringVar(value=default)
            ttk.Entry(r, textvariable=v, width=6).pack(side="left", padx=2)
            v.trace_add("write", lambda *_: self._on_input_change())
            self._tx_oz_vars.append(v)
        # TX cap — manual user input with Find Cap Combo button
        cap_row = ttk.Frame(tx_frm); cap_row.pack(fill="x", padx=4, pady=2)
        ttk.Label(cap_row, text="C_TX (nF):", width=24, anchor="w").pack(side="left")
        self._tx_cap_var = tk.StringVar(value="")
        tx_cap_entry = ttk.Entry(cap_row, textvariable=self._tx_cap_var, width=9)
        tx_cap_entry.pack(side="left", padx=2)
        ttk.Button(cap_row, text="Find Cap Combo", width=16,
                   command=self._find_tx_cap).pack(side="left", padx=(4, 2))
        self._tx_cap_combo_var = tk.StringVar(value="")
        ttk.Label(cap_row, textvariable=self._tx_cap_combo_var,
                  foreground="gray").pack(side="left", padx=4)
        self._tx_cap_var.trace_add("write", lambda *_: self._on_input_change())

        # RX inputs
        rx_frm = ttk.LabelFrame(split, text="RX Settings")
        rx_frm.grid(row=0, column=2, sticky="nsew", pady=4)
        self._rx_od_var    = self._input_row(rx_frm, "RX OD (mm):",         "50.0",  "rx_od_mm")
        self._rx_turns_var = self._input_row(rx_frm, "RX Turns:",            "10",    "rx_turns")
        self._rx_width_var = self._input_row(rx_frm, "RX Trace width (mm):", "0.4",   "rx_width")
        self._rx_spacing_var   = self._input_row(rx_frm, "RX Spacing (mm):",  DEFAULT_SPACING_MM,   "rx_spacing_mm")
        self._rx_outer_gap_var = self._input_row(rx_frm, "RX Outer gap (mm):", DEFAULT_RX_OUTER_GAP, "rx_outer_gap_mm")
        self._rx_inner_gap_var = self._input_row(rx_frm, "RX Inner gap (mm):", DEFAULT_RX_INNER_GAP, "rx_inner_gap_mm")
        # RX topology
        topo_row = ttk.Frame(rx_frm); topo_row.pack(fill="x", padx=4, pady=2)
        ttk.Label(topo_row, text="RX Topology:", width=24, anchor="w").pack(side="left")
        self._topo_combo = ttk.Combobox(
            topo_row, textvariable=self._topo_var,
            values=[v for _, v in TOPOLOGY_CHOICES], width=20, state="readonly")
        self._topo_combo.pack(side="left", padx=2)
        self._topo_range_label = ttk.Label(topo_row, text="", foreground="#808080")
        self._topo_range_label.pack(side="left", padx=6)
        self._topo_var.trace_add("write", lambda *_: self._on_input_change())
        # RX port inside
        rx_port_row = ttk.Frame(rx_frm); rx_port_row.pack(fill="x", padx=4, pady=2)
        ttk.Label(rx_port_row, text="RX Port inside:", width=24, anchor="w").pack(side="left")
        self._rx_port_inside_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rx_port_row, variable=self._rx_port_inside_var,
                        command=self._on_input_change).pack(side="left")
        # RX layer copper oz
        rx_lyr_frm = ttk.LabelFrame(rx_frm, text="RX Copper oz per layer (0 = inactive)")
        rx_lyr_frm.pack(fill="x", padx=4, pady=(4, 2))
        self._rx_oz_vars = []
        for i, default in enumerate(DEFAULT_RX_LAYER_OZ):
            r = ttk.Frame(rx_lyr_frm); r.pack(fill="x", padx=4, pady=1)
            ttk.Label(r, text=f"L{i+1}:", width=4).pack(side="left")
            v = tk.StringVar(value=default)
            ttk.Entry(r, textvariable=v, width=6).pack(side="left", padx=2)
            v.trace_add("write", lambda *_: self._on_input_change())
            self._rx_oz_vars.append(v)
        # RX cap — auto-computed (read-only display)
        rx_cap_row = ttk.Frame(rx_frm); rx_cap_row.pack(fill="x", padx=4, pady=2)
        ttk.Label(rx_cap_row, text="C_RX (nF) [auto]:", width=24, anchor="w").pack(side="left")
        self._rx_cap_display = tk.StringVar(value="—")
        ttk.Label(rx_cap_row, textvariable=self._rx_cap_display,
                  foreground="#208020").pack(side="left", padx=2)
        self._rx_cap_combo_display = tk.StringVar(value="")
        ttk.Label(rx_cap_row, textvariable=self._rx_cap_combo_display,
                  foreground="gray").pack(side="left", padx=4)

        # ---- NN Model (from NN Setup tab) -----------------------------------
        model_frm = ttk.LabelFrame(body, text="NN Model")
        model_frm.pack(fill="x", padx=8, pady=(4, 2))
        mrow = ttk.Frame(model_frm); mrow.pack(fill="x", padx=6, pady=4)
        ttk.Label(mrow, text="Folder:", foreground="gray").pack(side="left")
        self._model_label_var = tk.StringVar(value="(select in NN Setup tab)")
        ttk.Label(mrow, textvariable=self._model_label_var,
                  foreground="#80c8ff", font=("Consolas", 8),
                  wraplength=500, justify="left").pack(side="left", padx=6)

        # ---- Ground Circle ----------------------------------------------
        gc_frm = ttk.LabelFrame(body, text="Ground Circle")
        gc_frm.pack(fill="x", padx=8, pady=(4, 2))
        gc_row = ttk.Frame(gc_frm); gc_row.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(gc_row, text="Enable ground circle",
                        variable=self._gc_enabled,
                        command=self._on_gc_toggle).pack(side="left")
        ttk.Label(gc_row, text="  Diameter (mm):").pack(side="left")
        self._gc_dia_var = tk.StringVar(value="21.0")
        self._gc_dia_entry = ttk.Entry(gc_row, textvariable=self._gc_dia_var, width=8)
        self._gc_dia_entry.pack(side="left", padx=4)
        self._on_gc_toggle()

        # ---- Run button --------------------------------------------------
        btn_row = ttk.Frame(body); btn_row.pack(fill="x", padx=8, pady=(4, 2))
        self._run_btn = ttk.Button(btn_row, text="Run Simulation NN",
                                   command=self._on_run, width=22)
        self._run_btn.pack(side="left")
        self._status_var = tk.StringVar(value="")
        ttk.Label(btn_row, textvariable=self._status_var,
                  foreground="gray").pack(side="left", padx=10)

        # Domain coverage indicator
        dom_row = ttk.Frame(body); dom_row.pack(fill="x", padx=8, pady=(0, 4))
        self._domain_var = tk.StringVar(value="")
        self._domain_label = ttk.Label(dom_row, textvariable=self._domain_var,
                                       font=("TkDefaultFont", 8))
        self._domain_label.pack(side="left")

        # ---- Coupling results -------------------------------------------
        coup = ttk.LabelFrame(body, text="Coupling (NN estimate)")
        coup.pack(fill="x", padx=8, pady=4)
        cg = ttk.Frame(coup); cg.pack(fill="x", padx=6, pady=6)
        self._res_M = tk.StringVar(value="—")
        self._res_k = tk.StringVar(value="—")
        self._put_result(cg, 0, 0, "Mutual inductance:", self._res_M)
        self._put_result(cg, 0, 2, "Coupling coeff. k:", self._res_k)

        # ---- TX / RX result columns -------------------------------------
        res_split = ttk.Frame(body)
        res_split.pack(fill="x", padx=8, pady=4)
        res_split.columnconfigure(0, weight=1, uniform="col")
        res_split.columnconfigure(1, weight=0)
        res_split.columnconfigure(2, weight=1, uniform="col")
        tk.Frame(res_split, bg="#b0b0b0", width=2).grid(
            row=0, column=1, sticky="ns", padx=4)

        tx_res = ttk.LabelFrame(res_split, text="TX coil results")
        tx_res.grid(row=0, column=0, sticky="nsew", pady=4)
        rx_res = ttk.LabelFrame(res_split, text="RX coil results")
        rx_res.grid(row=0, column=2, sticky="nsew", pady=4)

        self._build_coil_results(tx_res, "tx")
        self._build_coil_results(rx_res, "rx")

        # ---- System insights -------------------------------------------
        ins = ttk.LabelFrame(body, text="Derived system insights")
        ins.pack(fill="x", padx=8, pady=(4, 8))
        ig = ttk.Frame(ins); ig.pack(fill="x", padx=6, pady=6)
        insight_items = [
            ("Q_TX / Q_RX:",            "q_tx_rx"),
            ("|Z_TX| at drive freq:",   "z_tx_drive"),
            ("TX f0 (resonance):",      "f0_tx"),
            ("RX f0 (resonance):",      "f0_rx"),
            ("Δf TX (fc - f0_TX):",     "df_tx"),
            ("Δf RX (fc - f0_RX):",     "df_rx"),
        ]
        for i, (lbl, key) in enumerate(insight_items):
            r, c = divmod(i, 2)
            ttk.Label(ig, text=lbl).grid(row=r, column=c*2, sticky="w", padx=4, pady=2)
            v = tk.StringVar(value="—")
            self._result_vars[key] = v
            ttk.Label(ig, textvariable=v, font=("", 9, "bold")).grid(
                row=r, column=c*2+1, sticky="w", padx=4, pady=2)

    def _input_row(self, parent, label, default, field_key):
        row = ttk.Frame(parent); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label, width=24, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        entry = ttk.Entry(row, textvariable=var, width=9)
        entry.pack(side="left", padx=2)
        self._field_widgets[field_key] = entry
        range_lbl = ttk.Label(row, text="", foreground="#808080")
        range_lbl.pack(side="left", padx=4)
        setattr(self, f"_range_label_{field_key}", range_lbl)
        var.trace_add("write", lambda *_: self._on_input_change())
        return var

    def _build_coil_results(self, parent, role):
        grid = ttk.Frame(parent); grid.pack(fill="x", padx=6, pady=6)
        items = [
            ("Inductance (µH):",        f"{role}_L"),
            ("AC resistance (Ω):",      f"{role}_Rac"),
            ("Q at drive freq:",        f"{role}_Q"),
            ("f0 resonance (kHz):",     f"{role}_f0"),
            ("Cap (nF):",               f"{role}_cap"),
            ("Cap combination:",        f"{role}_cap_combo"),
        ]
        for i, (lbl, key) in enumerate(items):
            ttk.Label(grid, text=lbl).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            v = tk.StringVar(value="—")
            self._result_vars[key] = v
            ttk.Label(grid, textvariable=v, font=("", 9, "bold")).grid(
                row=i, column=1, sticky="w", padx=4, pady=2)

    @staticmethod
    def _put_result(parent, r, c, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=4, pady=2)
        ttk.Label(parent, textvariable=var, font=("", 9, "bold")).grid(
            row=r, column=c+1, sticky="w", padx=4, pady=2)

    # -----------------------------------------------------------------------
    # Range loading & validation
    # -----------------------------------------------------------------------

    def _refresh_ranges(self):
        model_dir = self._get_model_dir()
        domain_path = os.path.join(model_dir, "domain.json") if model_dir else None
        if domain_path and os.path.exists(domain_path):
            try:
                import json
                with open(domain_path) as f:
                    d = json.load(f)
                self._domains = [(0, d)]
            except Exception:
                self._domains = []
        else:
            self._domains = []
        self._ranges  = _dl.union_ranges(self._domains)
        self._update_range_labels()
        self._validate_inputs()

    def _update_range_labels(self):
        field_map = {
            "tx_od_mm":  "_range_label_tx_od_mm",
            "tx_turns":  "_range_label_tx_turns",
            "tx_width":  "_range_label_tx_width",
            "rx_od_mm":  "_range_label_rx_od_mm",
            "rx_turns":  "_range_label_rx_turns",
            "rx_width":  "_range_label_rx_width",
            "freq_hz":   "_range_label_freq",
        }
        for key, attr in field_map.items():
            lbl = getattr(self, attr, None)
            if lbl is None:
                continue
            rng = self._ranges.get(key)
            if rng:
                lo, hi = rng
                lbl.configure(text=f"[{lo:g} – {hi:g}]")
            else:
                lbl.configure(text="")

        topo_rng = self._ranges.get("rx_topology", [])
        if topo_rng:
            self._topo_range_label.configure(
                text="[" + ", ".join(topo_rng) + "]")
        else:
            self._topo_range_label.configure(text="")

    def _get_numeric_inputs(self):
        """Return dict of field→value (float) or raise ValueError."""
        return {
            "tx_od_mm":  float(self._tx_od_var.get()),
            "tx_turns":  float(self._tx_turns_var.get()),
            "tx_width":  float(self._tx_width_var.get()),
            "rx_od_mm":  float(self._rx_od_var.get()),
            "rx_turns":  float(self._rx_turns_var.get()),
            "rx_width":  float(self._rx_width_var.get()),
            "freq_hz":   float(self._freq_var.get()),
        }

    def _validate_inputs(self):
        """Highlight out-of-range fields red; enable/disable Run button."""
        any_bad = False

        # numeric fields
        try:
            vals = self._get_numeric_inputs()
        except ValueError:
            # Can't parse — mark everything normal (can't tell which is bad)
            for entry in self._field_widgets.values():
                entry.configure(style="TEntry")
            if self._run_btn:
                self._run_btn.configure(state="disabled")
            self._domain_var.set("")
            return

        for key, entry in self._field_widgets.items():
            if key == "freq_hz":
                continue   # handled below
            rng = self._ranges.get(key)
            val = vals.get(key)
            if rng and val is not None:
                lo, hi = rng
                bad = not (lo <= val <= hi)
            else:
                bad = False
            self._set_entry_color(entry, bad)
            if bad:
                any_bad = True

        # freq entry
        freq_entry = self._field_widgets.get("freq_hz")
        rng = self._ranges.get("freq_hz")
        try:
            fval = float(self._freq_var.get())
            bad_f = bool(rng and not (rng[0] <= fval <= rng[1]))
        except ValueError:
            bad_f = False
        if freq_entry:
            self._set_entry_color(freq_entry, bad_f)
        if bad_f:
            any_bad = True

        # topology
        topo_ok = self._ranges.get("rx_topology", None)
        topo_val = self._topo_var.get()
        if topo_ok and topo_val not in topo_ok:
            self._topo_combo.configure(style="Red.TCombobox")
            any_bad = True
        else:
            self._topo_combo.configure(style="TCombobox")

        if self._run_btn:
            self._run_btn.configure(
                state="normal" if not any_bad else "disabled")

        # Domain coverage status
        vals_all = dict(vals)
        vals_all["rx_topology"] = self._topo_var.get()
        matching = _dl.find_matching_domains(vals_all, self._domains)
        if matching:
            batches = ", ".join(f"batch {n}" for n in matching)
            self._domain_var.set(f"In training domain: {batches}")
            self._domain_label.configure(foreground="#208020")
        elif self._domains:
            self._domain_var.set("Outside all training domains")
            self._domain_label.configure(foreground="#cc4400")
        else:
            self._domain_var.set("No domain files found — range hints unavailable")
            self._domain_label.configure(foreground="#808080")

    @staticmethod
    def _set_entry_color(entry, bad):
        if bad:
            entry.configure(style="Red.TEntry")
        else:
            entry.configure(style="TEntry")

    def _on_input_change(self):
        self._validate_inputs()

    def _get_model_dir(self) -> str:
        if self.app is not None and hasattr(self.app, "auto_tab"):
            return self.app.auto_tab.get_model_folder()
        return _NN_V2_DIR

    def _on_gc_toggle(self):
        state = "normal" if self._gc_enabled.get() else "disabled"
        self._gc_dia_entry.configure(state=state)

    @staticmethod
    def _short_path(p):
        try:
            return os.path.relpath(p)
        except ValueError:
            return p

    # -----------------------------------------------------------------------
    # Public: receive parameters from parametric tab
    # -----------------------------------------------------------------------

    def receive_parametric_params(self, role, params):
        """Called by parametric tab's _on_send_to_sim.
        params: dict with keys matching field names."""
        mapping = {
            "TX": {
                "od":    self._tx_od_var,
                "turns": self._tx_turns_var,
                "w":     self._tx_width_var,
            },
            "RX": {
                "od":       self._rx_od_var,
                "turns":    self._rx_turns_var,
                "w":        self._rx_width_var,
                "topology": self._topo_var,
            },
        }
        m = mapping.get(role, {})
        for key, var in m.items():
            if key in params:
                var.set(str(params[key]))
        self._on_input_change()

    # -----------------------------------------------------------------------
    # Cap helpers
    # -----------------------------------------------------------------------

    def _find_tx_cap(self):
        s = self._tx_cap_var.get().strip()
        if not s:
            return
        try:
            val = float(s)
        except ValueError:
            messagebox.showerror("Cap", "Numeric nF value required.")
            return
        if val <= 0:
            messagebox.showerror("Cap", "Must be > 0.")
            return
        achieved, desc = cap_combinator.find_best_cap(val)
        if desc is None:
            self._tx_cap_combo_var.set(f"(single {achieved:g} nF)")
        else:
            self._tx_cap_var.set(f"{achieved:g}")
            self._tx_cap_combo_var.set(f"({desc})")

    def _auto_rx_cap(self, L_rx_uH, freq_hz):
        """Find best E-value cap combo for RX resonance at freq_hz."""
        target_nf = _resonant_cap_nf(L_rx_uH, freq_hz)
        if target_nf is None or target_nf <= 0:
            return None, None, None
        achieved_nf, desc = cap_combinator.find_best_cap(target_nf)
        return target_nf, achieved_nf, desc

    # -----------------------------------------------------------------------
    # NN inference
    # -----------------------------------------------------------------------

    def _on_run(self):
        try:
            vals = self._get_numeric_inputs()
        except ValueError:
            messagebox.showerror("NN Simulation", "All inputs must be numeric.")
            return

        topo = self._topo_var.get()
        freq_hz = vals["freq_hz"]

        gc_dia = 0.0
        if self._gc_enabled.get():
            try:
                gc_dia = float(self._gc_dia_var.get())
                if gc_dia <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Ground Circle", "GC diameter must be a positive number.")
                return

        try:
            c_tx_nf = float(self._tx_cap_var.get())
            if c_tx_nf <= 0:
                raise ValueError
        except (ValueError, TypeError):
            c_tx_nf = None

        model_dir = self._get_model_dir()
        self._model_label_var.set(self._short_path(model_dir))
        self._run_btn.configure(state="disabled")
        self._status_var.set("Running NN…")

        thread = threading.Thread(
            target=self._run_worker,
            args=(vals, topo, freq_hz, c_tx_nf, gc_dia, model_dir),
            daemon=True)
        thread.start()

    def _run_worker(self, vals, topo, freq_hz, c_tx_nf, gc_dia, model_dir):
        try:
            model, x_scaler, y_scaler, feat_cols, torch, np = _try_load_nn(model_dir)
        except Exception as e:
            self.after(0, lambda: self._on_run_error(f"Cannot load NN model:\n{e}"))
            return

        if not feat_cols:
            self.after(0, lambda: self._on_run_error(
                "Surrogate scaler has no feature_names_in_; retrain with the new "
                "trainer so the inference path can match the training schema."))
            return

        # Pull the rest of the trainer-required inputs from the UI.
        try:
            extras = {
                "tx_spacing_mm":   float(self._tx_spacing_var.get()),
                "tx_outer_gap_mm": float(self._tx_outer_gap_var.get()),
                "tx_inner_gap_mm": float(self._tx_inner_gap_var.get()),
                "rx_spacing_mm":   float(self._rx_spacing_var.get()),
                "rx_outer_gap_mm": float(self._rx_outer_gap_var.get()),
                "rx_inner_gap_mm": float(self._rx_inner_gap_var.get()),
                "pcb_gap_mm":      float(self._pcb_gap_var.get()),
            }
            tx_oz = [float(v.get()) for v in self._tx_oz_vars]
            rx_oz = [float(v.get()) for v in self._rx_oz_vars]
        except ValueError:
            self.after(0, lambda: self._on_run_error(
                "All numeric inputs (spacings, gaps, copper oz, pcb gap) must be valid numbers."))
            return

        tx_topo = self._tx_topo_var.get() or "parallel"
        rx_topo = topo or "parallel"

        # Build a row that populates every column the trainer's scaler expects.
        # Topology one-hots use the stable TOPOLOGY_VOCAB order.
        row = {
            "tx_turns":  vals["tx_turns"],
            "tx_width":  vals["tx_width"],
            "tx_od_mm":  vals["tx_od_mm"],
            "rx_turns":  vals["rx_turns"],
            "rx_width":  vals["rx_width"],
            "rx_od_mm":  vals["rx_od_mm"],
            "freq_hz":   vals["freq_hz"],
            "rx_ground_disc_dia_mm": gc_dia,
            "tx_port_inside": float(bool(self._tx_port_inside_var.get())),
            "rx_port_inside": float(bool(self._rx_port_inside_var.get())),
        }
        row.update(extras)
        for i in range(4):
            row[f"tx_layer{i+1}_oz"] = tx_oz[i]
            row[f"rx_layer{i+1}_oz"] = rx_oz[i]
        for t in TOPOLOGY_VOCAB:
            row[f"tx_topo_{t}"] = 1.0 if tx_topo == t else 0.0
            row[f"rx_topo_{t}"] = 1.0 if rx_topo == t else 0.0

        # Hard-fail on schema mismatch — silent zero-fill caused the inference
        # distribution to drift from the training distribution.
        missing = [c for c in feat_cols if c not in row]
        if missing:
            self.after(0, lambda: self._on_run_error(
                "Trained scaler expects feature columns that are not produced "
                "by this tab. Retrain or extend the input form. "
                f"Missing: {missing}"))
            return

        x_vec = np.array([[row[c] for c in feat_cols]], dtype=np.float32)
        x_scaled = x_scaler.transform(x_vec).astype(np.float32)
        with torch.no_grad():
            y_scaled = model(torch.tensor(x_scaled)).numpy()
        y_out = y_scaler.inverse_transform(y_scaled)[0]

        L_tx = float(y_out[OUT_L_TX])
        L_rx = float(y_out[OUT_L_RX])
        M    = float(y_out[OUT_M])
        R_tx = float(y_out[OUT_R_TX])
        R_rx = float(y_out[OUT_R_RX])

        self.after(0, lambda: self._on_run_done(
            L_tx, L_rx, M, R_tx, R_rx, freq_hz, c_tx_nf))

    def _on_run_error(self, msg):
        messagebox.showerror("NN Simulation", msg)
        self._status_var.set("Error")
        self._validate_inputs()   # re-enables button if still valid

    def _on_run_done(self, L_tx, L_rx, M, R_tx, R_rx, freq_hz, c_tx_nf):
        rv = self._result_vars

        # ---- TX results ----
        rv["tx_L"].set(f"{L_tx:.4f} µH")
        rv["tx_Rac"].set(f"{R_tx:.4f} Ω  ({R_tx*1000:.2f} mΩ)")
        q_tx = _q_factor(L_tx, R_tx, freq_hz)
        rv["tx_Q"].set(f"{q_tx:.2f}" if q_tx else "—")

        # TX cap / resonance
        if c_tx_nf and c_tx_nf > 0:
            f0_tx = _series_resonant_freq(L_tx, c_tx_nf)
            rv["tx_cap"].set(f"{c_tx_nf:g} nF")
            rv["tx_cap_combo"].set(self._tx_cap_combo_var.get())
            rv["tx_f0"].set(f"{f0_tx/1000:.3f} kHz" if f0_tx else "—")
            self._result_vars["f0_tx"].set(f"{f0_tx/1000:.3f} kHz" if f0_tx else "—")
            self._result_vars["df_tx"].set(
                f"{(freq_hz - f0_tx)/1000:+.3f} kHz" if f0_tx else "—")
        else:
            rv["tx_cap"].set("—"); rv["tx_cap_combo"].set("")
            rv["tx_f0"].set("—")
            self._result_vars["f0_tx"].set("—"); self._result_vars["df_tx"].set("—")

        # ---- RX results — auto cap ----
        target_nf, achieved_nf, desc = self._auto_rx_cap(L_rx, freq_hz)
        rv["rx_L"].set(f"{L_rx:.4f} µH")
        rv["rx_Rac"].set(f"{R_rx:.4f} Ω  ({R_rx*1000:.2f} mΩ)")
        q_rx = _q_factor(L_rx, R_rx, freq_hz)
        rv["rx_Q"].set(f"{q_rx:.2f}" if q_rx else "—")

        if achieved_nf is not None:
            self._rx_cap_display.set(f"{achieved_nf:g} nF")
            combo_str = f"({desc})" if desc else f"(single {achieved_nf:g} nF)"
            self._rx_cap_combo_display.set(combo_str)
            rv["rx_cap"].set(f"{achieved_nf:g} nF")
            rv["rx_cap_combo"].set(combo_str)
            f0_rx = _series_resonant_freq(L_rx, achieved_nf)
            rv["rx_f0"].set(f"{f0_rx/1000:.3f} kHz" if f0_rx else "—")
            self._result_vars["f0_rx"].set(f"{f0_rx/1000:.3f} kHz" if f0_rx else "—")
            self._result_vars["df_rx"].set(
                f"{(freq_hz - f0_rx)/1000:+.3f} kHz" if f0_rx else "—")
        else:
            self._rx_cap_display.set("—"); self._rx_cap_combo_display.set("")
            rv["rx_cap"].set("—"); rv["rx_cap_combo"].set("")
            rv["rx_f0"].set("—")
            self._result_vars["f0_rx"].set("—"); self._result_vars["df_rx"].set("—")

        # ---- Coupling ----
        self._res_M.set(f"{M:.4f} µH")
        if L_tx > 0 and L_rx > 0:
            k = M / math.sqrt(L_tx * L_rx)
            self._res_k.set(f"{k:.4f}")
        else:
            self._res_k.set("—")

        # ---- Insights ----
        if q_tx and q_rx:
            self._result_vars["q_tx_rx"].set(f"{q_tx:.1f} / {q_rx:.1f}")
        else:
            self._result_vars["q_tx_rx"].set("—")

        # |Z_TX| approximation: series RLC at drive frequency
        if c_tx_nf and c_tx_nf > 0:
            w = 2 * math.pi * freq_hz
            X_L = w * L_tx * 1e-6
            X_C = 1.0 / (w * c_tx_nf * 1e-9)
            Z_tx = complex(R_tx, X_L - X_C)
            self._result_vars["z_tx_drive"].set(
                f"{abs(Z_tx):.3f} Ω  (arg {math.degrees(math.atan2(Z_tx.imag, Z_tx.real)):+.1f}°)")
        else:
            self._result_vars["z_tx_drive"].set("—")

        self._status_var.set("Done")
        self._validate_inputs()

    # -----------------------------------------------------------------------
    # Style setup (called once after app is built)
    # -----------------------------------------------------------------------

    @staticmethod
    def configure_styles():
        s = ttk.Style()
        s.configure("Red.TEntry",     fieldbackground="#ffcccc")
        s.configure("Red.TCombobox",  fieldbackground="#ffcccc")
