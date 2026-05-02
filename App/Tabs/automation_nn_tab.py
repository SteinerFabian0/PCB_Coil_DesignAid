#!/usr/bin/env python3
"""
AutomationNN tab — bulk NN inference over a user-specified coil geometry domain.

Samples random geometry combinations, runs surrogate NN to predict L/R/M, and
holds ALL results in RAM.  No circuit math here — ranking and evaluation happens
in the NN Analysis tab so the math can be tweaked without re-running.

Two passes when "Include Ground Circle" is checked:
  Pass 1 (batch-1 domain): gc_dia_mm = 0   (no ground circle)
  Pass 2 (batch-2 domain): gc_dia_mm sampled from [gc_min, gc_max]
"""

import os, sys, threading, traceback
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

_HERE       = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT   = os.path.dirname(_HERE)
_NN_DIR     = os.path.join(_APP_ROOT, "NeuralNetwork")
_NN_SCRIPTS = os.path.join(_NN_DIR, "scripts")
_MODULES    = os.path.join(_APP_ROOT, "Modules")
for _p in (_NN_DIR, _NN_SCRIPTS, _MODULES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parametric_coil as pc


# ── Domain defaults (mirror generate_lhs_samples.py DEFAULT_DOMAIN) ──────────
TX_OD_MAX_DEF   = 53.0
TX_ID_MIN_DEF   = 35.0
TX_TURNS_MIN_DEF = 6
TX_TURNS_MAX_DEF = 18
TX_WIDTH_MIN_DEF = 0.2
TX_WIDTH_MAX_DEF = 1.2

RX_OD_MAX_DEF   = 53.0
RX_ID_MIN_DEF   = 35.0
RX_TURNS_MIN_DEF = 4
RX_TURNS_MAX_DEF = 25
RX_WIDTH_MIN_DEF = 0.2
RX_WIDTH_MAX_DEF = 1.2

GC_MIN_DEF = 18.0
GC_MAX_DEF = 24.0

# NN evaluated at the midpoint of the user freq range.
# L/M are essentially constant with freq for PCB spirals;
# R_ac(f) = DCR + R_skin(F0)·√(f/F0) is analytic and done in the Analysis tab.
F0_HZ      = 125_000.0
BATCH_SIZE = 500_000

# Fixed PCB stackup constants (must match training data)
_SPACING_MM  = 0.16
_OZ_MM       = 0.035
_RHO_30C     = 1.724e-8 * (1 + 0.00393 * 10)

_TX_N_LAYERS = 3
_TX_H_M      = _OZ_MM * 1e-3

_RX_H_OZ     = [1.0, 0.5, 0.5, 1.0]
_RX_H_M      = [h * _OZ_MM * 1e-3 for h in _RX_H_OZ]
_RX_HSUM_PAR = sum(_RX_H_M)
_RX_HSUM_A   = _RX_H_M[0] + _RX_H_M[1]
_RX_HSUM_B   = _RX_H_M[2] + _RX_H_M[3]
_RX_HINV_SER = sum(1.0 / h for h in _RX_H_M)

# One-hot column order (must match train_surrogate.py)
_TOPO_OH_IDX = {"parallel": 0, "parallel_pairs_ser": 1, "series": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spiral_length_m(od_mm, w_mm, turns):
    pitch   = w_mm + _SPACING_MM
    R0      = od_mm / 2.0 - w_mm / 2.0
    R_inner = R0 - (2.0 * turns - 1.0) * pitch / 2.0
    return np.pi * turns * (R0 + R_inner) * 1e-3


def _inner_diameter_mm(od_mm, w_mm, turns):
    return od_mm - 2.0 * w_mm - (2.0 * turns - 1.0) * (w_mm + _SPACING_MM)


# ─────────────────────────────────────────────────────────────────────────────
# NN loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_nn(model_dir):
    import torch, joblib
    import torch.nn as nn

    mf = os.path.join(model_dir, "surrogate_model.pth")
    xf = os.path.join(model_dir, "x_scaler.pkl")
    yf = os.path.join(model_dir, "y_scaler.pkl")
    for p in (mf, xf, yf):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_scaler = joblib.load(xf)
    y_scaler = joblib.load(yf)

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

    model = _CoilNN(x_scaler.n_features_in_)
    model.load_state_dict(torch.load(mf, map_location=device))
    model.to(device)
    model.eval()

    feat_cols = (list(x_scaler.feature_names_in_)
                 if hasattr(x_scaler, "feature_names_in_") else None)
    return model, x_scaler, y_scaler, feat_cols, torch, device


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer core  (background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_optimizer(params, progress_cb, log_cb, done_cb, cancel_flag):
    try:
        model_dir = params["model_dir"]
        log_cb(f"Loading model from: {model_dir}")
        model, x_scaler, y_scaler, feat_cols, torch, device = _load_nn(model_dir)
        log_cb(f"Device: {device}" +
               (f"  ({torch.cuda.get_device_name(device)})"
                if device.type == "cuda" else "  (CPU — no GPU detected)"))

        tx_od_max    = params["tx_od_max"]
        tx_id_min    = params["tx_id_min"]
        tx_width_min = params["tx_width_min"]
        tx_width_max = params["tx_width_max"]

        rx_od_max    = params["rx_od_max"]
        rx_id_min    = params["rx_id_min"]
        rx_width_min = params["rx_width_min"]
        rx_width_max = params["rx_width_max"]

        topologies  = params["topologies"]
        n_topos     = len(topologies)

        gc_enabled  = params["gc_enabled"]
        gc_min_mm   = params["gc_min_mm"]
        gc_max_mm   = params["gc_max_mm"]

        freq_min_hz = params["freq_min_hz"]
        freq_max_hz = params["freq_max_hz"]
        freq_ref_hz = 0.5 * (freq_min_hz + freq_max_hz)

        N = params["n_combos"]  # per pass

        # Resolve feature columns
        _base     = ["tx_turns", "tx_width", "tx_od_mm",
                     "rx_od_mm", "rx_turns", "rx_width", "freq_hz"]
        _topo_oh  = ["topo_parallel", "topo_parallel_pairs_ser", "topo_series"]
        _gc_col   = "ground_circle_dia_mm"
        all_cols  = feat_cols if feat_cols else (_base + sorted(_topo_oh))
        has_gc    = (_gc_col in all_cols)

        log_cb(f"NN features ({len(all_cols)}): {all_cols}")
        if gc_enabled and not has_gc:
            log_cb("  ! Model has no ground_circle_dia_mm feature — "
                   "GC will be 0 in NN input for both passes.")

        passes = [("no_gc", False)]
        if gc_enabled:
            passes.append(("gc", True))

        accs = {k: [] for k in (
            "tx_turns", "tx_width", "tx_od_mm",
            "rx_od_mm", "rx_turns", "rx_width",
            "rx_topo", "gc_dia_mm",
            "L_tx", "L_rx", "M",
            "R_tx", "R_rx", "DCR_tx", "DCR_rx",
            "tx_id_mm", "rx_id_mm",
        )}

        n_total = 0
        rng = np.random.default_rng(42)

        for p_idx, (p_label, sample_gc) in enumerate(passes):
            log_cb(f"\n--- Pass {p_idx+1}/{len(passes)}: {p_label}"
                   f"  ({'GC sampled' if sample_gc else 'no GC'}) ---")
            n_id_ok  = 0
            n_id_rej = 0
            n_pass   = 0
            b        = 0

            while n_id_ok < N:
                if cancel_flag.is_set():
                    log_cb("Cancelled.")
                    done_cb(None)
                    return

                b += 1
                progress_cb(((p_idx + min(n_id_ok / N, 1.0)) / len(passes)) * 0.95)
                if b % 10 == 0 or b == 1:
                    log_cb(f"  Batch {b}: {n_id_ok:,}/{N:,} valid, "
                           f"{n_id_rej:,} ID-rejected")

                # ── Sample geometry ──────────────────────────────────────
                # Turns are derived from geometry: given OD, ID_min, and width,
                # the max turns that keeps ID >= ID_min is computed analytically.
                # id = od + spacing - width - 2*T*(width+spacing) >= id_min
                # → T_max = floor((od + s - w - id_min) / (2*(w+s)))
                bs = BATCH_SIZE

                tx_od_s    = rng.uniform(tx_id_min + 2.0, tx_od_max,
                                         size=bs).astype(np.float32)
                tx_width_s = rng.uniform(tx_width_min, tx_width_max,
                                         size=bs).astype(np.float32)
                tx_t_max   = np.maximum(
                    np.floor((tx_od_s + _SPACING_MM - tx_width_s - tx_id_min)
                             / (2.0 * (tx_width_s + _SPACING_MM))).astype(np.int32),
                    TX_TURNS_MIN_DEF)
                tx_t_min   = np.minimum(
                    np.full(bs, TX_TURNS_MIN_DEF, dtype=np.int32), tx_t_max)
                tx_turns_s = rng.integers(tx_t_min, tx_t_max + 1).astype(np.float32)

                rx_od_s    = rng.uniform(rx_id_min + 2.0, rx_od_max,
                                         size=bs).astype(np.float32)
                rx_width_s = rng.uniform(rx_width_min, rx_width_max,
                                         size=bs).astype(np.float32)
                rx_t_max   = np.maximum(
                    np.floor((rx_od_s + _SPACING_MM - rx_width_s - rx_id_min)
                             / (2.0 * (rx_width_s + _SPACING_MM))).astype(np.int32),
                    RX_TURNS_MIN_DEF)
                rx_t_min   = np.minimum(
                    np.full(bs, RX_TURNS_MIN_DEF, dtype=np.int32), rx_t_max)
                rx_turns_s = rng.integers(rx_t_min, rx_t_max + 1).astype(np.float32)

                topo_idx_s = rng.integers(0, n_topos, size=bs)

                gc_s = (rng.uniform(gc_min_mm, gc_max_mm, size=bs).astype(np.float32)
                        if sample_gc
                        else np.zeros(bs, dtype=np.float32))

                # ── ID check ─────────────────────────────────────────────
                tx_id_s = _inner_diameter_mm(tx_od_s, tx_width_s, tx_turns_s)
                rx_id_s = _inner_diameter_mm(rx_od_s, rx_width_s, rx_turns_s)
                id_ok   = (tx_id_s >= tx_id_min) & (rx_id_s >= rx_id_min)
                n_id_rej += int((~id_ok).sum())
                n_id_ok  += int(id_ok.sum())
                if not id_ok.any():
                    continue

                tx_turns_s = tx_turns_s[id_ok]; tx_od_s    = tx_od_s[id_ok]
                tx_width_s = tx_width_s[id_ok]; rx_od_s    = rx_od_s[id_ok]
                rx_turns_s = rx_turns_s[id_ok]; rx_width_s = rx_width_s[id_ok]
                topo_idx_s = topo_idx_s[id_ok]; gc_s       = gc_s[id_ok]
                tx_id_s    = tx_id_s[id_ok];    rx_id_s    = rx_id_s[id_ok]
                bs_v = int(id_ok.sum())

                # ── One-hot topology ─────────────────────────────────────
                topo_oh = np.zeros((bs_v, 3), dtype=np.float32)
                for ti, tname in enumerate(topologies):
                    col = _TOPO_OH_IDX.get(tname, ti)
                    topo_oh[topo_idx_s == ti, col] = 1.0

                # ── Analytical DCR ────────────────────────────────────────
                tx_len_m  = _spiral_length_m(tx_od_s, tx_width_s, tx_turns_s)
                DCR_tx_np = (_RHO_30C * tx_len_m
                             / (tx_width_s * 1e-3 * _TX_H_M * _TX_N_LAYERS)
                             ).astype(np.float32)

                rx_len_m   = _spiral_length_m(rx_od_s, rx_width_s, rx_turns_s)
                rx_w_m     = rx_width_s * 1e-3
                DCR_rx_par = (_RHO_30C * rx_len_m / (rx_w_m * _RX_HSUM_PAR))
                DCR_rx_ser = (_RHO_30C * rx_len_m * _RX_HINV_SER / rx_w_m)
                DCR_rx_pps = (_RHO_30C * rx_len_m
                              * (1.0 / _RX_HSUM_A + 1.0 / _RX_HSUM_B) / rx_w_m)
                DCR_rx_np  = np.empty(bs_v, dtype=np.float32)
                for ti, tname in enumerate(topologies):
                    m = (topo_idx_s == ti)
                    if not m.any():
                        continue
                    if tname == "parallel":
                        DCR_rx_np[m] = DCR_rx_par[m]
                    elif tname == "series":
                        DCR_rx_np[m] = DCR_rx_ser[m]
                    else:
                        DCR_rx_np[m] = DCR_rx_pps[m]

                # ── NN feature matrix ─────────────────────────────────────
                row_map = {
                    "tx_turns":                tx_turns_s,
                    "tx_width":                tx_width_s,
                    "tx_od_mm":                tx_od_s,
                    "rx_od_mm":                rx_od_s,
                    "rx_turns":                rx_turns_s,
                    "rx_width":                rx_width_s,
                    "freq_hz":                 np.full(bs_v, freq_ref_hz, dtype=np.float32),
                    "topo_parallel":           topo_oh[:, 0],
                    "topo_parallel_pairs_ser": topo_oh[:, 1],
                    "topo_series":             topo_oh[:, 2],
                    "ground_circle_dia_mm":    gc_s,
                }
                X = np.column_stack(
                    [row_map.get(c, np.zeros(bs_v, dtype=np.float32))
                     for c in all_cols]
                ).astype(np.float32)

                # ── NN inference ──────────────────────────────────────────
                X_sc = x_scaler.transform(X).astype(np.float32)
                with torch.no_grad():
                    Y_sc = model(torch.tensor(X_sc, device=device))
                Y = y_scaler.inverse_transform(Y_sc.cpu().numpy())

                # ── Accumulate ────────────────────────────────────────────
                accs["tx_turns"].append(tx_turns_s)
                accs["tx_width"].append(tx_width_s)
                accs["tx_od_mm"].append(tx_od_s)
                accs["rx_od_mm"].append(rx_od_s)
                accs["rx_turns"].append(rx_turns_s)
                accs["rx_width"].append(rx_width_s)
                accs["rx_topo"].append(topo_idx_s.astype(np.uint8))
                accs["gc_dia_mm"].append(gc_s)
                accs["L_tx"].append(Y[:, 0].astype(np.float32))
                accs["L_rx"].append(Y[:, 1].astype(np.float32))
                accs["M"].append(Y[:, 2].astype(np.float32))
                accs["R_tx"].append(Y[:, 3].astype(np.float32))
                accs["R_rx"].append(Y[:, 4].astype(np.float32))
                accs["DCR_tx"].append(DCR_tx_np)
                accs["DCR_rx"].append(DCR_rx_np)
                accs["tx_id_mm"].append(tx_id_s)
                accs["rx_id_mm"].append(rx_id_s)
                n_pass += bs_v

            log_cb(f"  Pass done: {n_pass:,} records  ({n_id_rej:,} ID-rejected)")
            n_total += n_pass

        log_cb(f"\nConcatenating {n_total:,} total records…")
        progress_cb(0.97)

        result = {k: np.concatenate(v) for k, v in accs.items()}
        result["_topologies"]  = np.array(topologies)
        result["_freq_ref_hz"] = np.float64(freq_ref_hz)
        result["_freq_min_hz"] = np.float64(freq_min_hz)
        result["_freq_max_hz"] = np.float64(freq_max_hz)

        raw_mb = sum(v.nbytes for v in result.values()
                     if isinstance(v, np.ndarray)) / 1e6
        log_cb(f"Raw arrays: {raw_mb:.0f} MB  |  {n_total:,} records")
        progress_cb(1.0)
        done_cb(result)

    except Exception as e:
        log_cb(f"Error: {e}\n{traceback.format_exc()}")
        done_cb(None)


# ─────────────────────────────────────────────────────────────────────────────
# Tab widget
# ─────────────────────────────────────────────────────────────────────────────

class AutomationNNTab(ttk.Frame):

    def __init__(self, parent, app=None, on_next_tab=None, **kw):
        super().__init__(parent, **kw)
        self.app          = app
        self._on_next_tab = on_next_tab
        self._cancel_flag = threading.Event()
        self._running     = False
        self._results         = {}
        self._last_run_params = {}
        self._build()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(0, weight=1, uniform="col")
        self.columnconfigure(1, weight=1, uniform="col")
        self.rowconfigure(0, weight=1)

        col_l = ttk.Frame(self)
        col_l.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        col_r = ttk.Frame(self)
        col_r.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        # ── TX domain ────────────────────────────────────────────────────────
        tx_f = ttk.LabelFrame(col_l, text="Domain — TX")
        tx_f.pack(fill="x", pady=(0, 6))
        self._tx_od_max   = self._row(tx_f, "OD max (mm):",    str(TX_OD_MAX_DEF))
        self._tx_id_min   = self._row(tx_f, "ID min (mm):",    str(TX_ID_MIN_DEF))
        ttk.Label(tx_f,
                  text=f"Trace width: {TX_WIDTH_MIN_DEF}–{TX_WIDTH_MAX_DEF} mm  "
                       f"(fixed to training domain)",
                  foreground="gray", font=("TkDefaultFont", 8)
                  ).pack(anchor="w", padx=6, pady=(0, 4))

        # ── RX domain ────────────────────────────────────────────────────────
        rx_f = ttk.LabelFrame(col_l, text="Domain — RX")
        rx_f.pack(fill="x", pady=(0, 6))
        self._rx_od_max   = self._row(rx_f, "OD max (mm):",    str(RX_OD_MAX_DEF))
        self._rx_id_min   = self._row(rx_f, "ID min (mm):",    str(RX_ID_MIN_DEF))
        ttk.Label(rx_f,
                  text=f"Trace width: {RX_WIDTH_MIN_DEF}–{RX_WIDTH_MAX_DEF} mm  "
                       f"(fixed to training domain)",
                  foreground="gray", font=("TkDefaultFont", 8)
                  ).pack(anchor="w", padx=6, pady=(0, 4))

        # ── Topologies ───────────────────────────────────────────────────────
        topo_f = ttk.LabelFrame(col_l, text="RX Topologies")
        topo_f.pack(fill="x", pady=(0, 6))
        topo_row = ttk.Frame(topo_f)
        topo_row.pack(fill="x", padx=6, pady=4)
        self._topo_par  = tk.BooleanVar(value=True)
        self._topo_ser  = tk.BooleanVar(value=True)
        self._topo_pps  = tk.BooleanVar(value=True)
        ttk.Checkbutton(topo_row, text="Parallel",
                        variable=self._topo_par).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(topo_row, text="Series",
                        variable=self._topo_ser).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(topo_row, text="Parallel-pairs-series",
                        variable=self._topo_pps).pack(side="left")

        # ── Frequency (NN reference) ─────────────────────────────────────────
        ff = ttk.LabelFrame(col_l, text="Frequency Range")
        ff.pack(fill="x", pady=(0, 6))
        self._freq_min, self._freq_max = self._row_range(
            ff, "Min / Max (kHz):", "120", "130")
        ttk.Label(ff, text="NN queried at midpoint; range saved for Analysis tab.",
                  foreground="gray", font=("TkDefaultFont", 8)
                  ).pack(anchor="w", padx=6, pady=(0, 4))

        # ── Ground circle ────────────────────────────────────────────────────
        gc_f = ttk.LabelFrame(col_l, text="Ground Circle")
        gc_f.pack(fill="x", pady=(0, 6))
        self._gc_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(gc_f, text="Include Ground Circle  (adds a 2nd pass)",
                        variable=self._gc_enabled,
                        command=self._on_gc_toggle).pack(anchor="w", padx=6, pady=(4, 2))
        gc_range_row = ttk.Frame(gc_f)
        gc_range_row.pack(fill="x")
        self._gc_min, self._gc_max = self._row_range(
            gc_range_row, "GC dia min / max (mm):",
            str(GC_MIN_DEF), str(GC_MAX_DEF))
        # Collect all widgets inside gc_range_row for enable/disable
        self._gc_range_widgets = gc_range_row.winfo_children()
        # Defer child collection until after mainloop starts
        gc_range_row.bind("<Map>", lambda _: self._collect_gc_widgets(gc_range_row))
        self._on_gc_toggle()   # set initial enabled state

        # ── Sweep settings ───────────────────────────────────────────────────
        sf = ttk.LabelFrame(col_l, text="Sweep Settings")
        sf.pack(fill="x", pady=(0, 6))
        self._n_combos = self._row(sf, "Combinations per pass (M):", "10")

        # ── NN Model (from NN Setup tab) ──────────────────────────────────────
        nn_f = ttk.LabelFrame(col_l, text="NN Model")
        nn_f.pack(fill="x", pady=(0, 6))
        mrow = ttk.Frame(nn_f); mrow.pack(fill="x", padx=6, pady=4)
        ttk.Label(mrow, text="Folder:", foreground="gray").pack(side="left")
        self._model_label_var = tk.StringVar(value="(select in NN Setup tab)")
        ttk.Label(mrow, textvariable=self._model_label_var,
                  foreground="#80c8ff", font=("Consolas", 8),
                  wraplength=340, justify="left").pack(side="left", padx=6)

        ttk.Button(col_l, text="Next Tab →  (NN Analysis)",
                   command=self._on_next_tab_click).pack(fill="x")

        # ── RIGHT COLUMN ─────────────────────────────────────────────────────
        col_r.rowconfigure(2, weight=1)

        run_lf = ttk.LabelFrame(col_r, text="Optimizer Control")
        run_lf.pack(fill="x", pady=(0, 8))
        btn_row = ttk.Frame(run_lf)
        btn_row.pack(fill="x", padx=6, pady=6)
        self._run_btn = ttk.Button(btn_row, text="Run Sweep",
                                   command=self._on_run)
        self._run_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self._cancel_btn = ttk.Button(btn_row, text="Cancel",
                                      command=self._on_cancel,
                                      state="disabled")
        self._cancel_btn.pack(side="left", expand=True, fill="x")

        prog_lf = ttk.LabelFrame(col_r, text="Progress")
        prog_lf.pack(fill="x", pady=(0, 8))
        self._progress = ttk.Progressbar(prog_lf, mode="determinate", maximum=100)
        self._progress.pack(fill="x", padx=6, pady=(6, 4))
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(prog_lf, textvariable=self._status_var,
                  foreground="gray", wraplength=600, justify="left"
                  ).pack(fill="x", padx=6, pady=(0, 6))

        log_lf = ttk.LabelFrame(col_r, text="Log")
        log_lf.pack(fill="both", expand=True)
        log_lf.rowconfigure(0, weight=1)
        log_lf.columnconfigure(0, weight=1)
        self._log_text = tk.Text(log_lf, state="disabled",
                                  font=("Consolas", 9), wrap="word",
                                  background="#f8f8f8")
        log_sb = ttk.Scrollbar(log_lf, orient="vertical",
                                command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_sb.set)
        log_sb.grid(row=0, column=1, sticky="ns", pady=4)
        self._log_text.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _row(self, parent, label, default, label_width=24):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text=label, width=label_width, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var, width=10).pack(side="left", padx=4)
        return var

    def _row_range(self, parent, label, default_lo, default_hi, label_width=24):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text=label, width=label_width, anchor="w").pack(side="left")
        var_lo = tk.StringVar(value=default_lo)
        var_hi = tk.StringVar(value=default_hi)
        ttk.Entry(row, textvariable=var_lo, width=7).pack(side="left", padx=(4, 2))
        ttk.Label(row, text="/").pack(side="left")
        ttk.Entry(row, textvariable=var_hi, width=7).pack(side="left", padx=(2, 4))
        return var_lo, var_hi

    @staticmethod
    def _short_path(p):
        try:
            return os.path.relpath(p)
        except ValueError:
            return p

    # ─────────────────────────────────────────────────────────────────────────
    # Event handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_gc_widgets(self, frame):
        self._gc_range_widgets = []
        for child in frame.winfo_children():
            self._gc_range_widgets.append(child)
            if hasattr(child, "winfo_children"):
                self._gc_range_widgets.extend(child.winfo_children())
        self._on_gc_toggle()

    def _on_gc_toggle(self):
        state = "normal" if self._gc_enabled.get() else "disabled"
        for w in getattr(self, "_gc_range_widgets", []):
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def _get_model_dir(self) -> str:
        if self.app is not None and hasattr(self.app, "auto_tab"):
            return self.app.auto_tab.get_model_folder()
        return _NN_DIR

    def _on_next_tab_click(self):
        if self._on_next_tab:
            self._on_next_tab()

    def _on_run(self):
        try:
            params = self._parse_params()
        except ValueError as e:
            self._set_status(str(e), color="red")
            return

        self._last_run_params = params
        self._cancel_flag.clear()
        self._running = True
        self._run_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self._progress["value"] = 0
        self._log_clear()
        self._set_status("Running…")

        threading.Thread(
            target=_run_optimizer,
            args=(params, self._progress_cb, self._log_cb,
                  self._done_cb, self._cancel_flag),
            daemon=True,
        ).start()

    def _on_cancel(self):
        self._cancel_flag.set()
        self._set_status("Cancelling…", color="orange")

    def _progress_cb(self, frac):
        self.after(0, lambda: self._progress.configure(value=frac * 100))

    def _log_cb(self, msg):
        self.after(0, lambda m=msg: self._log_append(m))

    def _done_cb(self, result):
        self.after(0, lambda: self._on_done(result))

    def _on_done(self, result):
        self._running = False
        self._run_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")

        if result is None:
            self._set_status("Error — see log.", color="red")
            return

        self._results = result
        n = int(len(result.get("L_tx", [])))
        self._set_status(
            f"{n:,} records in RAM — switching to Analysis tab…",
            color="green")
        self.after(2000, self._on_next_tab_click)

    # ─────────────────────────────────────────────────────────────────────────
    # Param parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_params(self):
        def flt(var, name, lo=None, hi=None):
            try:
                v = float(var.get())
            except ValueError:
                raise ValueError(f"'{name}' must be a number.")
            if lo is not None and v < lo:
                raise ValueError(f"'{name}' must be ≥ {lo}.")
            if hi is not None and v > hi:
                raise ValueError(f"'{name}' must be ≤ {hi}.")
            return v

        tx_od_max    = flt(self._tx_od_max, "TX OD max", lo=10.0)
        tx_id_min    = flt(self._tx_id_min, "TX ID min", lo=0.0, hi=tx_od_max)

        rx_od_max    = flt(self._rx_od_max, "RX OD max", lo=10.0)
        rx_id_min    = flt(self._rx_id_min, "RX ID min", lo=0.0, hi=rx_od_max)

        topos = []
        if self._topo_par.get():  topos.append("parallel")
        if self._topo_ser.get():  topos.append("series")
        if self._topo_pps.get():  topos.append("parallel_pairs_ser")
        if not topos:
            raise ValueError("At least one topology must be selected.")

        freq_min = flt(self._freq_min, "Freq min (kHz)", lo=10.0) * 1e3
        freq_max = flt(self._freq_max, "Freq max (kHz)", lo=freq_min / 1e3) * 1e3

        gc_enabled = self._gc_enabled.get()
        gc_min = flt(self._gc_min, "GC min",  lo=0.0) if gc_enabled else 0.0
        gc_max = flt(self._gc_max, "GC max",  lo=gc_min) if gc_enabled else 0.0

        n_combos = int(flt(self._n_combos, "Combinations per pass (M)",
                           lo=0.001) * 1_000_000)

        model_dir = self._get_model_dir()
        if not os.path.isdir(model_dir):
            raise ValueError(f"Model folder not found:\n{model_dir}\n\nSelect a folder in the NN Setup tab.")
        self._model_label_var.set(self._short_path(model_dir))

        return dict(
            model_dir=model_dir,
            tx_od_max=tx_od_max, tx_id_min=tx_id_min,
            tx_width_min=TX_WIDTH_MIN_DEF, tx_width_max=TX_WIDTH_MAX_DEF,
            rx_od_max=rx_od_max, rx_id_min=rx_id_min,
            rx_width_min=RX_WIDTH_MIN_DEF, rx_width_max=RX_WIDTH_MAX_DEF,
            topologies=topos,
            freq_min_hz=freq_min, freq_max_hz=freq_max,
            gc_enabled=gc_enabled, gc_min_mm=gc_min, gc_max_mm=gc_max,
            n_combos=n_combos,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Send to Simulation tab  (called by Analysis tab with a result-row dict)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_send_to_sim(self, r):
        if self.app is None:
            messagebox.showerror("Send to Sim", "No app reference.")
            return
        temp_dir = (getattr(self.app.sim_tab, "temp_dir", None)
                    or getattr(self.app, "temp_dir", None)
                    or os.path.join(_APP_ROOT, "..", "temp"))
        try:
            self._build_and_register(r, temp_dir)
        except Exception as e:
            messagebox.showerror("Send to Sim",
                                 f"Failed to build coils:\n{e}\n\n{traceback.format_exc()}")
            return

        sim = self.app.sim_tab
        freq_hz = r["freq_hz"]
        sim.freq_var.set(str(int(freq_hz)))
        sim.fc_var.set(str(int(freq_hz)))
        sim.cap_tx.var.set(f"{r['C_tx_nf']:g}")
        sim.cap_rx.var.set(f"{r['C_rx_nf']:g}")

        def _sim_done_cb(result, _r=r):
            try:
                sim._done_callbacks.remove(_sim_done_cb)
            except ValueError:
                pass
            analysis = getattr(self.app, "nn_analysis_tab", None)
            if analysis:
                self.after(0, lambda: analysis.on_sim_done(_r, result))

        sim._done_callbacks.append(_sim_done_cb)
        try:
            self.app._nb.select(sim)
        except Exception:
            pass
        sim._on_start()

    def _build_and_register(self, r, temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        tx_sp = pc.SpiralParams(
            od_mm=r["tx_od_mm"], trace_width_mm=r["tx_width"],
            spacing_mm=_SPACING_MM, turns=r["tx_turns"], resolution_mm=0.6)
        tx_stackup = pc.StackUp(
            slots=[pc.LayerSlot(active=True,  copper_oz=1.0),
                   pc.LayerSlot(active=True,  copper_oz=1.0),
                   pc.LayerSlot(active=True,  copper_oz=1.0),
                   pc.LayerSlot(active=False, copper_oz=1.0)],
            outer_gap_mm=0.2, inner_gap_mm=1.3)
        ok, msg = pc.validate_spiral(tx_sp)
        if not ok:
            raise ValueError(f"TX spiral invalid: {msg}")
        tx_layers = pc.active_layer_data(tx_sp, tx_stackup)
        tx_path = os.path.join(temp_dir, "nn_auto_tx.inp")
        fmin = r["freq_hz"]; fmax = fmin + 15000.0
        pc.write_topology_inp("parallel", tx_layers, tx_path,
                              w_mm=r["tx_width"], fmin=fmin, fmax=fmax)
        tx_meta = {"role": "TX", "topology": "parallel",
                   "layer_params": [(r["tx_width"], ld["h_mm"], len(ld["nodes"]))
                                    for ld in tx_layers],
                   "nodes_by_layer": [list(ld["nodes"]) for ld in tx_layers]}

        rx_topo = r["rx_topology"]
        rx_sp = pc.SpiralParams(
            od_mm=r["rx_od_mm"], trace_width_mm=r["rx_width"],
            spacing_mm=_SPACING_MM, turns=r["rx_turns"], resolution_mm=0.6)
        rx_stackup = pc.StackUp(
            slots=[pc.LayerSlot(active=True, copper_oz=1.0),
                   pc.LayerSlot(active=True, copper_oz=0.5),
                   pc.LayerSlot(active=True, copper_oz=0.5),
                   pc.LayerSlot(active=True, copper_oz=1.0)],
            outer_gap_mm=0.2, inner_gap_mm=1.3)
        ok, msg = pc.validate_spiral(rx_sp)
        if not ok:
            raise ValueError(f"RX spiral invalid: {msg}")
        rx_layers = pc.active_layer_data(rx_sp, rx_stackup)
        rx_path = os.path.join(temp_dir, "nn_auto_rx.inp")
        pc.write_topology_inp(rx_topo, rx_layers, rx_path,
                              w_mm=r["rx_width"], fmin=fmin, fmax=fmax)
        rx_flags  = pc.series_reverse_flags_for_topology(rx_topo, len(rx_layers))
        rx_native = pc.reverse_nodes_for_series_flow(rx_layers, rx_flags)
        rx_meta = {"role": "RX", "topology": rx_topo,
                   "layer_params": [(r["rx_width"], ld["h_mm"], len(ld["nodes"]))
                                    for ld in rx_native],
                   "nodes_by_layer": [list(ld["nodes"]) for ld in rx_native]}

        sim = self.app.sim_tab
        sim.register_coil("TX", "Automation NN", tx_path, tx_meta)
        sim.register_coil("RX", "Automation NN", rx_path, rx_meta)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _log_clear(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _log_append(self, msg):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _set_status(self, msg, color="gray"):
        self._status_var.set(msg)
