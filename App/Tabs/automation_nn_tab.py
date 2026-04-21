#!/usr/bin/env python3
"""
AutomationNN tab — system-level optimizer using the surrogate neural network.

Generates a large random grid of coil geometries + frequencies, predicts
L/R/M/R via the PyTorch model, then runs vectorized circuit math to find
the optimal TX/RX configuration that meets the user's power delivery
requirements with duty cycle as close as possible to the target.
"""

import os, sys, math, threading, traceback
import tkinter as tk
from tkinter import ttk

_HERE     = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
_NN_DIR   = os.path.join(_APP_ROOT, "NeuralNetwork")
_MODULES  = os.path.join(_APP_ROOT, "Modules")
for _p in (_NN_DIR, _MODULES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cap_combinator import E_VALUES_NF

_SIMDATA_DIR = os.path.join(_APP_ROOT, "SimulationData")
_SAMPLES_FILE = os.path.join(_SIMDATA_DIR, "lhs_samples.json")
_MODEL_FILE = os.path.join(_NN_DIR, "surrogate_model.pth")
_X_SCALER   = os.path.join(_NN_DIR, "x_scaler.pkl")
_Y_SCALER   = os.path.join(_NN_DIR, "y_scaler.pkl")

# Design space boundaries (from sweep data)
TX_OD_MM         = 53.0
TX_TOPOLOGY      = "parallel"
TX_TURNS_MIN     = 6
TX_TURNS_MAX     = 18
TX_WIDTH_MIN     = 0.2
TX_WIDTH_MAX     = 1.2
RX_OD_MIN        = 48.0
RX_OD_MAX        = 53.0
RX_TURNS_MIN     = 4
RX_TURNS_MAX     = 25
RX_WIDTH_MIN     = 0.2
RX_WIDTH_MAX     = 1.2
FREQ_MIN_HZ      = 110_000.0
FREQ_MAX_HZ      = 140_000.0
RX_TOPOLOGIES    = ["parallel", "series", "parallel_pairs_ser"]

BATCH_SIZE    = 50_000
F0_HZ         = 125_000.0   # Frequency the NN was trained at; L/M are queried here, R_ac is then scaled
N_DRIVE_FREQS = 7            # Drive frequencies swept uniformly across FREQ_MIN_HZ–FREQ_MAX_HZ

# H-bridge fundamental voltage: V_fund = coeff * V_supply
_HBRIDGE_COEFF = 2.0 * math.sqrt(2.0) / math.pi   # ≈ 0.9003


# ---------------------------------------------------------------------------
# Cap table builders
# ---------------------------------------------------------------------------

def _build_cap_table(n_caps):
    """Return sorted list of (value_nF: float, label: str) for stock caps.
    n_caps=1: singles only.  n_caps=2: singles + parallel + series pairs."""
    e = list(E_VALUES_NF)
    entries = [(c, f"{c:g} nF") for c in e]
    if n_caps >= 2:
        for i, c1 in enumerate(e):
            for c2 in e[i:]:
                entries.append((c1 + c2, f"{c1:g}+{c2:g} nF ||"))
                s = (c1 * c2) / (c1 + c2)
                entries.append((s, f"{c1:g}+{c2:g} nF ser"))
    seen, out = set(), []
    for v, lbl in entries:
        key = round(v, 8)
        if key not in seen:
            seen.add(key)
            out.append((v, lbl))
    return out


# Pre-build TX cap table (always 2-cap combos available for TX)
_TX_CAP_TABLE = _build_cap_table(2)


# ---------------------------------------------------------------------------
# NN loader
# ---------------------------------------------------------------------------

def _load_nn():
    import torch, joblib
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_scaler = joblib.load(_X_SCALER)
    y_scaler = joblib.load(_Y_SCALER)

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
    model.load_state_dict(torch.load(_MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()
    feat_cols = (list(x_scaler.feature_names_in_)
                 if hasattr(x_scaler, "feature_names_in_") else None)
    return model, x_scaler, y_scaler, feat_cols, torch, device


# ---------------------------------------------------------------------------
# Optimizer core (runs in background thread)
# ---------------------------------------------------------------------------

def _run_optimizer(params, progress_cb, log_cb, done_cb, cancel_flag):
    try:
        import numpy as np, json

        log_cb("Loading surrogate model…")
        model, x_scaler, y_scaler, feat_cols, torch, device = _load_nn()
        log_cb(f"Running on: {device}  (PyTorch {torch.__version__})")
        if device.type == "cpu":
            log_cb("  → No CUDA GPU detected. To enable GPU acceleration:")
            log_cb("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            log_cb(f"  → GPU: {torch.cuda.get_device_name(device)}")

        N          = params["n_combos"]
        V_min      = params["v_min"]
        V_max      = params["v_max"]
        V_rx_min   = params["v_rx_min"]
        P_target_W = params["p_target_mw"] * 1e-3
        Duty_tgt   = params["duty_target"]
        rx_ncaps   = params["rx_ncaps"]

        P_inst_W   = P_target_W / Duty_tgt           # instantaneous power (W) when ON
        R_load_eq  = (V_rx_min ** 2) / P_inst_W      # equivalent AC load resistance (Ω)
        log_cb(f"R_load_eq = {R_load_eq:.3f} Ω  |  P_inst = {P_inst_W*1e3:.1f} mW")

        # ---- Derive geometry bounds from the actual LHS samples file ----
        try:
            with open(_SAMPLES_FILE, "r") as _f:
                _lhs = json.load(_f)
            if isinstance(_lhs, dict):
                _samples  = _lhs.get("samples", [])
                _rng_dict = (_lhs.get("ranges") or _lhs.get("param_ranges")
                             or _lhs.get("bounds") or {})
            else:
                _samples, _rng_dict = _lhs, {}
            if _samples:
                log_cb(f"LHS sample keys: {list(_samples[0].keys())}")
            # Key aliases — handles naming variations across generator versions
            _aliases = {
                            "tx_turns":    ["tx_turns",          "TX_turns",    "n_turns_tx"],
                            "tx_width":    ["tx_trace_width_mm",  "tx_width",    "tx_width_mm", "TX_width"],
                            "rx_od_mm":    ["rx_od_mm",           "rx_od",       "RX_od_mm"],
                            "rx_turns":    ["rx_turns",           "RX_turns",    "n_turns_rx"],
                            "rx_width":    ["rx_trace_width_mm",  "rx_width",    "rx_width_mm", "RX_width"],
                            "rx_topology": ["rx_topology",        "topology",    "rx_topo"],
            }
            def _resolve(canonical):
                for a in _aliases[canonical]:
                    if (_samples and a in _samples[0]) or a in _rng_dict:
                        return a
                return None
            def _bound(canonical):
                if _rng_dict:
                    k = next((a for a in _aliases[canonical] if a in _rng_dict), None)
                    if k:
                        v = _rng_dict[k]
                        if isinstance(v, (list, tuple)) and len(v) == 2:
                            return float(v[0]), float(v[1])
                k = _resolve(canonical)
                if k is None or not _samples:
                    raise KeyError(f"No key found for '{canonical}'")
                vals = [s[k] for s in _samples if k in s]
                if not vals: raise ValueError(f"Empty values for '{k}'")
                return float(min(vals)), float(max(vals))
            def _ibound(c):
                lo, hi = _bound(c); return int(round(lo)), int(round(hi))
            tx_turns_min, tx_turns_max = _ibound("tx_turns")
            tx_width_min, tx_width_max = _bound("tx_width")
            rx_od_min,    rx_od_max    = _bound("rx_od_mm")
            rx_turns_min, rx_turns_max = _ibound("rx_turns")
            rx_width_min, rx_width_max = _bound("rx_width")
            _tk    = _resolve("rx_topology")
            _topos = sorted({s[_tk] for s in _samples if _tk and _tk in s})
            rx_topologies = _topos if _topos else RX_TOPOLOGIES
            log_cb(f"Bounds from LHS: tx_turns [{tx_turns_min}–{tx_turns_max}], "
                   f"tx_width [{tx_width_min:.2f}–{tx_width_max:.2f} mm], "
                   f"rx_od [{rx_od_min:.1f}–{rx_od_max:.1f} mm], "
                   f"rx_turns [{rx_turns_min}–{rx_turns_max}]")
        except Exception as _e:
            log_cb(f"Warning: could not read LHS bounds ({_e}), using hardcoded defaults.")
            tx_turns_min, tx_turns_max = TX_TURNS_MIN, TX_TURNS_MAX
            tx_width_min, tx_width_max = TX_WIDTH_MIN, TX_WIDTH_MAX
            rx_od_min,    rx_od_max    = RX_OD_MIN,    RX_OD_MAX
            rx_turns_min, rx_turns_max = RX_TURNS_MIN,  RX_TURNS_MAX
            rx_width_min, rx_width_max = RX_WIDTH_MIN,  RX_WIDTH_MAX
            rx_topologies              = RX_TOPOLOGIES

        # Drive frequencies — always sweep full 110–140 kHz band regardless of training data
        drive_freqs = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, N_DRIVE_FREQS)
        log_cb(f"Drive freqs: {[f'{f/1e3:.1f}' for f in drive_freqs]} kHz "
               f"(R_ac scaled from F0={F0_HZ/1e3:.0f} kHz via √f)")

        # Cap tables as numpy arrays
        tx_cap_table  = _TX_CAP_TABLE
        tx_cap_nf     = np.array([v for v, _ in tx_cap_table], dtype=np.float64)
        tx_cap_lbls   = [lbl for _, lbl in tx_cap_table]

        rx_cap_table  = _build_cap_table(rx_ncaps)
        rx_cap_nf     = np.array([v for v, _ in rx_cap_table], dtype=np.float64)
        rx_cap_lbls   = [lbl for _, lbl in rx_cap_table]

        log_cb(f"TX cap options: {len(tx_cap_nf)}  |  RX cap options: {len(rx_cap_nf)}")

        # NN feature column order (must match train_surrogate.py)
        base_cols  = ["tx_turns", "tx_width", "tx_od_mm",
                      "rx_od_mm", "rx_turns", "rx_width", "freq_hz"]
        topo_cols  = ["topo_parallel", "topo_parallel_pairs_ser", "topo_series"]
        all_cols   = feat_cols if feat_cols else (base_cols + sorted(topo_cols))

        n_batches  = math.ceil(N / BATCH_SIZE)
        log_cb(f"Sampling {N:,} combinations across {n_batches} batch(es)…")

        rng        = np.random.default_rng(42)
        valid_rows = []

        for b in range(n_batches):
            if cancel_flag.is_set():
                log_cb("Cancelled by user.")
                done_cb(None)
                return

            bs = min(BATCH_SIZE, N - b * BATCH_SIZE)
            progress_cb(b / n_batches * 0.80)
            if b % 5 == 0:
                log_cb(f"Batch {b+1}/{n_batches}  ({len(valid_rows):,} valid so far)…")

            # ---- Sample geometry (within actual training bounds) --------
            n_topos    = len(rx_topologies)
            tx_turns_s = rng.integers(tx_turns_min, tx_turns_max + 1, size=bs).astype(np.float64)
            tx_width_s = rng.uniform(tx_width_min, tx_width_max, size=bs)
            rx_od_s    = rng.uniform(rx_od_min, rx_od_max, size=bs)
            rx_turns_s = rng.integers(rx_turns_min, rx_turns_max + 1, size=bs).astype(np.float64)
            rx_width_s = rng.uniform(rx_width_min, rx_width_max, size=bs)
            topo_idx_s = rng.integers(0, n_topos, size=bs)

            topo_oh    = np.zeros((bs, 3), dtype=np.float64)
            _topo_key  = ["topo_parallel", "topo_parallel_pairs_ser", "topo_series"]
            for ti, tname in enumerate(rx_topologies):
                col = _topo_key.index(f"topo_{tname}") if f"topo_{tname}" in _topo_key else ti
                topo_oh[topo_idx_s == ti, col] = 1.0

            row_map = {
                "tx_turns":              tx_turns_s,
                "tx_width":              tx_width_s,
                "tx_od_mm":              np.full(bs, TX_OD_MM),
                "rx_od_mm":              rx_od_s,
                "rx_turns":              rx_turns_s,
                "rx_width":              rx_width_s,
                "freq_hz":               np.full(bs, F0_HZ),   # always query NN at training freq
                "topo_parallel":         topo_oh[:, 0],
                "topo_parallel_pairs_ser": topo_oh[:, 1],
                "topo_series":           topo_oh[:, 2],
            }
            X = np.column_stack([row_map[c] for c in all_cols]).astype(np.float32)

            # ---- NN inference (on GPU if available) ---------------------
            X_sc = x_scaler.transform(X).astype(np.float32)
            with torch.no_grad():
                X_tensor = torch.tensor(X_sc, device=device)
                Y_sc = model(X_tensor).cpu().numpy()
            Y = y_scaler.inverse_transform(Y_sc)

            L_tx    = Y[:, 0]   # µH — geometric, essentially freq-independent in this band
            L_rx    = Y[:, 1]   # µH
            M       = Y[:, 2]   # µH
            R_tx_f0 = Y[:, 3]   # Ω at F0_HZ (skin-effect dominated → scales as √f)
            R_rx_f0 = Y[:, 4]   # Ω at F0_HZ

            # ---- Sweep drive frequencies; scale R_ac ∝ √f --------------
            # For each geometry sample, keep the best (drive_freq, C_tx) combo.
            best_fitness_batch = np.full(bs, np.inf)
            best_result_batch  = [None] * bs

            for f_drive in drive_freqs:
                w          = 2.0 * math.pi * f_drive
                freq_scale = math.sqrt(f_drive / F0_HZ)
                R_tx       = R_tx_f0 * freq_scale   # (bs,)
                R_rx       = R_rx_f0 * freq_scale   # (bs,)

                # RX tank: tune cap to the actual drive frequency
                C_rx_ideal_nf = 1e9 / (w**2 * L_rx * 1e-6)
                diff_rx       = np.abs(C_rx_ideal_nf[:, None] - rx_cap_nf[None, :])
                rx_ci         = np.argmin(diff_rx, axis=1)
                C_rx          = rx_cap_nf[rx_ci]

                X_L_rx   = w * L_rx * 1e-6
                X_C_rx   = 1.0 / (w * C_rx * 1e-9)
                Zrx_re   = R_rx + R_load_eq
                Zrx_im   = X_L_rx - X_C_rx
                Zrx_abs2 = Zrx_re**2 + Zrx_im**2
                Zrx_abs  = np.sqrt(Zrx_abs2)

                wM      = w * M * 1e-6
                wM2     = wM**2
                Zref_re =  wM2 * Zrx_re / Zrx_abs2
                Zref_im = -wM2 * Zrx_im / Zrx_abs2

                X_L_tx        = w * L_tx * 1e-6
                Zin_re        = R_tx + Zref_re
                Zin_im_no_cap = X_L_tx + Zref_im
                X_C_tx_all    = 1.0 / (w * tx_cap_nf * 1e-9)              # (n_tx,)
                Zin_im_all    = Zin_im_no_cap[:, None] - X_C_tx_all        # (bs, n_tx)
                Zin_abs_all   = np.sqrt(Zin_re[:, None]**2 + Zin_im_all**2)

                Vf_min = _HBRIDGE_COEFF * V_min
                Vf_max = _HBRIDGE_COEFF * V_max
                Vf_mid = _HBRIDGE_COEFF * (V_min + V_max) * 0.5

                It_min      = Vf_min / Zin_abs_all
                It_max      = Vf_max / Zin_abs_all
                It_mid      = Vf_mid / Zin_abs_all
                wM_over_Zrx = wM / Zrx_abs

                Ir_min    = It_min * wM_over_Zrx[:, None]
                Ir_max    = It_max * wM_over_Zrx[:, None]
                Ir_mid    = It_mid * wM_over_Zrx[:, None]
                V_ind_min = wM[:, None] * It_min

                Pload_min = Ir_min**2 * R_load_eq
                Pload_max = Ir_max**2 * R_load_eq
                Pload_mid = Ir_mid**2 * R_load_eq
                Duty_vmin = P_target_W / np.maximum(Pload_min, 1e-30)
                Duty_vmax = P_target_W / np.maximum(Pload_max, 1e-30)
                P_tx_mid  = It_mid**2 * Zin_re[:, None]
                eff_mid   = np.where(P_tx_mid > 0, Pload_mid / P_tx_mid, 0.0)

                valid = (
                    (Zin_im_all > 0.0)        &
                    (V_ind_min  >= V_rx_min)  &
                    (Duty_vmin  <= 1.0)       &
                    (Duty_vmax  <= 1.0)
                )
                duty_err    = (np.abs(Duty_vmin - Duty_tgt) +
                               np.abs(Duty_vmax - Duty_tgt))
                fitness_all = np.where(valid, duty_err - eff_mid * 0.1, np.inf)

                best_ti      = np.argmin(fitness_all, axis=1)
                best_fitness = fitness_all[np.arange(bs), best_ti]

                improved = np.isfinite(best_fitness) & (best_fitness < best_fitness_batch)
                if not np.any(improved):
                    continue
                best_fitness_batch[improved] = best_fitness[improved]

                for idx in np.where(improved)[0]:
                    ti   = int(best_ti[idx])
                    rx_c = int(rx_ci[idx])
                    best_result_batch[idx] = {
                        "fitness":     float(best_fitness[idx]),
                        "freq_hz":     f_drive,
                        "tx_turns":    int(round(float(tx_turns_s[idx]))),
                        "tx_width":    float(tx_width_s[idx]),
                        "tx_od_mm":    TX_OD_MM,
                        "rx_od_mm":    float(rx_od_s[idx]),
                        "rx_turns":    int(round(float(rx_turns_s[idx]))),
                        "rx_width":    float(rx_width_s[idx]),
                        "rx_topology": rx_topologies[int(topo_idx_s[idx])],
                        "L_tx_uH":    float(L_tx[idx]),
                        "L_rx_uH":    float(L_rx[idx]),
                        "M_uH":       float(M[idx]),
                        "R_tx_ohm":   float(R_tx[idx]),
                        "R_rx_ohm":   float(R_rx[idx]),
                        "C_tx_nf":    float(tx_cap_nf[ti]),
                        "C_tx_label": tx_cap_lbls[ti],
                        "C_rx_nf":    float(rx_cap_nf[rx_c]),
                        "C_rx_label": rx_cap_lbls[rx_c],
                        "Duty_vmin":  float(Duty_vmin[idx, ti]),
                        "Duty_vmax":  float(Duty_vmax[idx, ti]),
                        "V_ind_min_V": float(V_ind_min[idx, ti]),
                        "eff_mid":    float(eff_mid[idx, ti]),
                        "Zin_re":     float(Zin_re[idx]),
                        "Zin_im":     float(Zin_im_all[idx, ti]),
                    }

            valid_rows.extend(r for r in best_result_batch if r is not None)

        log_cb(f"Sorting {len(valid_rows):,} valid configurations…")
        progress_cb(0.95)

        if not valid_rows:
            done_cb([])
            return

        valid_rows.sort(key=lambda r: r["fitness"])
        top = valid_rows[:10]

        progress_cb(1.0)
        log_cb(f"Done — {len(valid_rows):,} valid / {N:,} sampled. Top 10 shown.")
        done_cb(top)

    except Exception as e:
        log_cb(f"Error: {e}\n{traceback.format_exc()}")
        done_cb(None)


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class AutomationNNTab(ttk.Frame):

    def __init__(self, parent, app=None, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self._cancel_flag = threading.Event()
        self._running     = False
        self._results     = []
        self._build()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # ---- Left: inputs + controls ------------------------------------
        left_outer = ttk.Frame(paned, width=310)
        left_outer.pack_propagate(False)
        paned.add(left_outer, weight=0)

        canvas = tk.Canvas(left_outer, highlightthickness=0)
        vsb    = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        left = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Power requirements
        pf = ttk.LabelFrame(left, text="Power Requirements")
        pf.pack(fill="x", padx=8, pady=(8, 4))
        self._v_min    = self._row(pf, "TX V_min  (V):",         "3.2")
        self._v_max    = self._row(pf, "TX V_max  (V):",         "4.2")
        self._v_rx_min = self._row(pf, "Min RX V_induced  (V):", "5.0")
        self._p_target = self._row(pf, "Target RX Power  (mW):", "40.0")
        self._duty_tgt = self._row(pf, "Target Duty Cycle:",     "0.60")

        # Optimizer settings
        of = ttk.LabelFrame(left, text="Optimizer Settings")
        of.pack(fill="x", padx=8, pady=4)
        self._n_combos = self._row(of, "Combinations:",          "1000000")

        rx_cap_row = ttk.Frame(of)
        rx_cap_row.pack(fill="x", padx=4, pady=3)
        ttk.Label(rx_cap_row, text="RX cap count (1 or 2):",
                  width=26, anchor="w").pack(side="left")
        self._rx_ncaps_var = tk.StringVar(value="1")
        ttk.Spinbox(rx_cap_row, textvariable=self._rx_ncaps_var,
                    from_=1, to=2, width=4, state="readonly").pack(side="left", padx=2)

        # Buttons
        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", padx=8, pady=6)
        self._run_btn = ttk.Button(btn_row, text="Run Optimizer",
                                   command=self._on_run, width=15)
        self._run_btn.pack(side="left")
        self._cancel_btn = ttk.Button(btn_row, text="Cancel",
                                      command=self._on_cancel,
                                      state="disabled", width=8)
        self._cancel_btn.pack(side="left", padx=4)

        # Progress
        prog_frm = ttk.Frame(left)
        prog_frm.pack(fill="x", padx=8, pady=(0, 4))
        self._progress   = ttk.Progressbar(prog_frm, mode="determinate", maximum=100)
        self._progress.pack(fill="x")
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(prog_frm, textvariable=self._status_var,
                  foreground="gray", wraplength=280, justify="left").pack(
                  fill="x", pady=(2, 0))

        # Log
        lf = ttk.LabelFrame(left, text="Log")
        lf.pack(fill="x", padx=8, pady=4)
        self._log_text = tk.Text(lf, height=8, state="disabled",
                                  font=("Consolas", 8), wrap="word",
                                  background="#f8f8f8")
        log_sb = ttk.Scrollbar(lf, orient="vertical",
                                command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # ---- Right: results table + detail ------------------------------
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        res_lf = ttk.LabelFrame(right, text="Top 10 Configurations")
        res_lf.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self._build_tree(res_lf)

        det_lf = ttk.LabelFrame(right, text="Detail (click a row)")
        det_lf.pack(fill="x", padx=8, pady=(0, 8))
        self._detail_text = tk.Text(det_lf, height=9, state="disabled",
                                     font=("Consolas", 9), wrap="none",
                                     background="#f8f8f8")
        det_sb = ttk.Scrollbar(det_lf, orient="vertical",
                                command=self._detail_text.yview)
        det_xsb = ttk.Scrollbar(det_lf, orient="horizontal",
                                 command=self._detail_text.xview)
        self._detail_text.configure(yscrollcommand=det_sb.set,
                                     xscrollcommand=det_xsb.set)
        det_xsb.pack(side="bottom", fill="x")
        det_sb.pack(side="right", fill="y")
        self._detail_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _row(self, parent, label, default):
        """Labelled entry row; returns StringVar."""
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label, width=26, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var, width=12).pack(side="left", padx=2)
        return var

    def _build_tree(self, parent):
        cols = (
            "#", "Fitness", "Freq kHz",
            "TX Turns", "TX W mm", "RX OD mm", "RX Turns", "RX W mm", "RX Topo",
            "L_tx µH", "L_rx µH", "M µH", "R_tx Ω", "R_rx Ω",
            "C_tx", "C_rx",
            "Duty@Vmin", "Duty@Vmax", "V_ind_min", "Eff%",
        )
        widths = {
            "#": 28, "Fitness": 72, "Freq kHz": 72,
            "TX Turns": 62, "TX W mm": 60,
            "RX OD mm": 62, "RX Turns": 65, "RX W mm": 60, "RX Topo": 105,
            "L_tx µH": 68, "L_rx µH": 68, "M µH": 58,
            "R_tx Ω": 58, "R_rx Ω": 58,
            "C_tx": 120, "C_rx": 120,
            "Duty@Vmin": 78, "Duty@Vmax": 78,
            "V_ind_min": 72, "Eff%": 52,
        }
        tree_f = ttk.Frame(parent)
        tree_f.pack(fill="both", expand=True, padx=4, pady=4)

        xsb = ttk.Scrollbar(tree_f, orient="horizontal")
        ysb = ttk.Scrollbar(tree_f, orient="vertical")
        self._tree = ttk.Treeview(tree_f, columns=cols, show="headings",
                                   yscrollcommand=ysb.set,
                                   xscrollcommand=xsb.set, height=12)
        xsb.configure(command=self._tree.xview)
        ysb.configure(command=self._tree.yview)
        xsb.pack(side="bottom", fill="x")
        ysb.pack(side="right", fill="y")
        self._tree.pack(side="left", fill="both", expand=True)

        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=widths.get(c, 80), minwidth=36, stretch=False)

        self._tree.tag_configure("rank1", background="#d4edda")
        self._tree.tag_configure("rank2", background="#e8f4fd")
        self._tree.tag_configure("rank3", background="#fff3cd")
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    # -----------------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------------

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

        v_min    = flt(self._v_min,    "TX V_min",             lo=0.1)
        v_max    = flt(self._v_max,    "TX V_max",             lo=v_min)
        v_rx_min = flt(self._v_rx_min, "Min RX V_induced",     lo=0.1)
        p_target = flt(self._p_target, "Target RX Power (mW)", lo=0.1)
        duty_tgt = flt(self._duty_tgt, "Target Duty Cycle",    lo=0.01, hi=1.0)
        n_combos = int(flt(self._n_combos, "Combinations",     lo=1000))
        rx_ncaps = int(self._rx_ncaps_var.get())
        return dict(v_min=v_min, v_max=v_max, v_rx_min=v_rx_min,
                    p_target_mw=p_target, duty_target=duty_tgt,
                    n_combos=n_combos, rx_ncaps=rx_ncaps)

    def _on_run(self):
        try:
            params = self._parse_params()
        except ValueError as e:
            self._set_status(str(e), color="red")
            return

        self._cancel_flag.clear()
        self._running = True
        self._run_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self._progress["value"] = 0
        self._clear_tree()
        self._clear_detail()
        self._log_clear()
        self._set_status("Running…")

        thread = threading.Thread(
            target=_run_optimizer,
            args=(params,
                  self._progress_cb,
                  self._log_cb,
                  self._done_cb,
                  self._cancel_flag),
            daemon=True)
        thread.start()

    def _on_cancel(self):
        self._cancel_flag.set()
        self._set_status("Cancelling…", color="orange")

    def _progress_cb(self, frac):
        self.after(0, lambda: self._progress.configure(value=frac * 100))

    def _log_cb(self, msg):
        self.after(0, lambda m=msg: self._log_append(m))

    def _done_cb(self, results):
        self.after(0, lambda: self._on_done(results))

    def _on_done(self, results):
        self._running = False
        self._run_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")

        if results is None:
            self._set_status("Optimizer encountered an error — see log.", color="red")
            return
        if not results:
            self._set_status("No valid configurations found. Try relaxing constraints.",
                             color="orange")
            return

        self._results = results
        self._populate_tree(results)
        self._set_status(f"Top {len(results)} configurations loaded.", color="green")

    # -----------------------------------------------------------------------
    # Tree population
    # -----------------------------------------------------------------------

    def _clear_tree(self):
        for iid in self._tree.get_children():
            self._tree.delete(iid)

    def _populate_tree(self, rows):
        self._clear_tree()
        tag_map = {0: "rank1", 1: "rank2", 2: "rank3"}
        for i, r in enumerate(rows):
            tag = tag_map.get(i, "")
            vals = (
                i + 1,
                f"{r['fitness']:.4f}",
                f"{r['freq_hz']/1e3:.2f}",
                r["tx_turns"],
                f"{r['tx_width']:.2f}",
                f"{r['rx_od_mm']:.1f}",
                r["rx_turns"],
                f"{r['rx_width']:.2f}",
                r["rx_topology"],
                f"{r['L_tx_uH']:.3f}",
                f"{r['L_rx_uH']:.3f}",
                f"{r['M_uH']:.4f}",
                f"{r['R_tx_ohm']:.3f}",
                f"{r['R_rx_ohm']:.3f}",
                r["C_tx_label"],
                r["C_rx_label"],
                f"{r['Duty_vmin']:.3f}",
                f"{r['Duty_vmax']:.3f}",
                f"{r['V_ind_min_V']:.2f}",
                f"{r['eff_mid']*100:.1f}",
            )
            self._tree.insert("", "end", values=vals, tags=(tag,) if tag else ())

    # -----------------------------------------------------------------------
    # Detail panel
    # -----------------------------------------------------------------------

    def _clear_detail(self):
        self._detail_text.configure(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.configure(state="disabled")

    def _on_tree_select(self, _event=None):
        sel = self._tree.selection()
        if not sel:
            return
        iid   = sel[0]
        items = self._tree.get_children()
        idx   = list(items).index(iid)
        if idx >= len(self._results):
            return
        r = self._results[idx]
        self._show_detail(idx + 1, r)

    def _show_detail(self, rank, r):
        lines = [
            f"{'='*60}",
            f"  Rank #{rank}  —  Fitness score: {r['fitness']:.6f}",
            f"{'='*60}",
            f"",
            f"  GEOMETRY",
            f"    TX: OD={r['tx_od_mm']:.1f} mm  turns={r['tx_turns']}  "
            f"width={r['tx_width']:.3f} mm  topology={TX_TOPOLOGY}",
            f"    RX: OD={r['rx_od_mm']:.1f} mm  turns={r['rx_turns']}  "
            f"width={r['rx_width']:.3f} mm  topology={r['rx_topology']}",
            f"",
            f"  DRIVE",
            f"    Frequency : {r['freq_hz']/1e3:.3f} kHz",
            f"    C_TX      : {r['C_tx_nf']:.4f} nF  ({r['C_tx_label']})",
            f"    C_RX      : {r['C_rx_nf']:.4f} nF  ({r['C_rx_label']})",
            f"",
            f"  PREDICTED EM PARAMETERS",
            f"    L_TX  = {r['L_tx_uH']:.4f} µH",
            f"    L_RX  = {r['L_rx_uH']:.4f} µH",
            f"    M     = {r['M_uH']:.5f} µH",
            f"    k     = {r['M_uH'] / math.sqrt(r['L_tx_uH']*r['L_rx_uH']):.4f}",
            f"    R_TX  = {r['R_tx_ohm']:.4f} Ω",
            f"    R_RX  = {r['R_rx_ohm']:.4f} Ω",
            f"",
            f"  CIRCUIT",
            f"    Z_in  = {r['Zin_re']:.4f} + j{r['Zin_im']:.4f} Ω  "
            f"({'inductive ✓' if r['Zin_im'] > 0 else 'CAPACITIVE ✗'})",
            f"    V_induced_min = {r['V_ind_min_V']:.3f} V",
            f"",
            f"  PERFORMANCE",
            f"    Duty @ V_min  = {r['Duty_vmin']:.4f}  "
            f"({r['Duty_vmin']*100:.1f}%)",
            f"    Duty @ V_max  = {r['Duty_vmax']:.4f}  "
            f"({r['Duty_vmax']*100:.1f}%)",
            f"    Efficiency    = {r['eff_mid']*100:.2f}%  (at mid supply voltage)",
        ]
        text = "\n".join(lines)
        self._detail_text.configure(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.insert("end", text)
        self._detail_text.configure(state="disabled")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

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
