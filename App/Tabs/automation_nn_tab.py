#!/usr/bin/env python3
"""
AutomationNN tab — system-level optimizer using the surrogate neural network.

Generates a large random grid of coil geometries, predicts L/R/M via PyTorch,
then runs fully-GPU vectorised circuit math across every drive frequency and
cap combo to find the best TX/RX configuration.
"""

import os, sys, math, threading, traceback
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

_HERE     = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
_NN_DIR   = os.path.join(_APP_ROOT, "NeuralNetwork")
_MODULES  = os.path.join(_APP_ROOT, "Modules")
for _p in (_NN_DIR, _MODULES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cap_combinator import E_VALUES_NF
import parametric_coil as pc

_SIMDATA_DIR  = os.path.join(_APP_ROOT, "SimulationData")
_SAMPLES_FILE = os.path.join(_SIMDATA_DIR, "lhs_samples.json")
_MODEL_FILE   = os.path.join(_NN_DIR, "surrogate_model.pth")
_X_SCALER     = os.path.join(_NN_DIR, "x_scaler.pkl")
_Y_SCALER     = os.path.join(_NN_DIR, "y_scaler.pkl")

# Design space boundaries (fallback if LHS file unavailable)
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
RX_TOPOLOGIES    = ["parallel", "series", "parallel_pairs_ser"]

# NN was trained at 125 kHz; L/M are queried here, R_ac is then scaled ∝ √f
F0_HZ      = 125_000.0
# Large batch to saturate GPU (500 K × 188 TX caps × float32 ≈ 376 MB/tensor — fits 4070 Ti)
BATCH_SIZE = 500_000

# H-bridge fundamental voltage: V_fund = coeff * V_supply
_HBRIDGE_COEFF = 2.0 * math.sqrt(2.0) / math.pi   # ≈ 0.9003

# PCB / copper constants fixed in training data
_SPACING_MM   = 0.16                        # trace-to-trace gap (fixed in LHS sweep)
_OZ_MM        = 0.035                       # 1 oz copper = 35 µm
_RHO_30C      = 1.724e-8 * (1 + 0.00393 * 10)  # Ω·m at 30 °C

# TX stackup: 3 active layers, all 1 oz, topology always "parallel"
_TX_N_LAYERS  = 3
_TX_H_M       = _OZ_MM * 1e-3              # 1 oz in metres

# RX stackup: 4 active layers, copper weights [1, 0.5, 0.5, 1] oz
_RX_H_OZ      = [1.0, 0.5, 0.5, 1.0]
_RX_H_M       = [h * _OZ_MM * 1e-3 for h in _RX_H_OZ]   # metres
_RX_HSUM_PAR  = sum(_RX_H_M)              # effective height for parallel wiring
_RX_HSUM_A    = _RX_H_M[0] + _RX_H_M[1]  # pair A (slots 0+1)
_RX_HSUM_B    = _RX_H_M[2] + _RX_H_M[3]  # pair B (slots 2+3)
# Effective 1/R factor for series wiring (sum of 1/h_i)
_RX_HINV_SER  = sum(1.0 / h for h in _RX_H_M)

# Minimum inner-edge diameter for both TX and RX coils
MIN_ID_MM     = 35.0


# ---------------------------------------------------------------------------
# Cap table builders
# ---------------------------------------------------------------------------

def _build_cap_table(n_caps):
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


_TX_CAP_TABLE = _build_cap_table(2)


# ---------------------------------------------------------------------------
# NN loader
# ---------------------------------------------------------------------------

def _load_nn():
    import torch, joblib
    import torch.nn as nn

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# Geometry helpers
# ---------------------------------------------------------------------------

def _spiral_length_m(od_mm, w_mm, turns):
    """Analytical planar-spiral path length in metres (numpy-vectorised).

    Derivation: 2N half-arcs with radii R0, R0-p/2, R0-p, …
    Total = π * N * (R0 + R_inner)  where R0 = OD/2 - W/2,
    R_inner = R0 - (2N-1)*pitch/2,  pitch = W + spacing.
    """
    pitch   = w_mm + _SPACING_MM
    R0      = od_mm / 2.0 - w_mm / 2.0
    R_inner = R0 - (2.0 * turns - 1.0) * pitch / 2.0
    return np.pi * turns * (R0 + R_inner) * 1e-3   # metres


def _inner_diameter_mm(od_mm, w_mm, turns):
    """Inner edge diameter of the spiral (numpy-vectorised)."""
    pitch = w_mm + _SPACING_MM
    return od_mm - 2.0 * w_mm - (2.0 * turns - 1.0) * pitch


def _rx_effective_turns(rx_turns, topology):
    """Electrical turns seen at the port, given spiral turns-per-layer."""
    if topology == "series":            return rx_turns * 4
    if topology == "parallel_pairs_ser": return rx_turns * 2
    return rx_turns   # "parallel": all layers share the same winding


# ---------------------------------------------------------------------------
# Optimizer core (runs in background thread)
# ---------------------------------------------------------------------------

def _run_optimizer(params, progress_cb, log_cb, done_cb, cancel_flag):
    try:
        import json

        log_cb("Loading surrogate model…")
        model, x_scaler, y_scaler, feat_cols, torch, device = _load_nn()
        log_cb(f"Running on: {device}  (PyTorch {torch.__version__})")
        if device.type == "cpu":
            log_cb("  → No CUDA GPU detected.")
        else:
            log_cb(f"  → GPU: {torch.cuda.get_device_name(device)}")

        N           = params["n_combos"]
        V_min       = params["v_min"]
        V_max       = params["v_max"]
        V_rx_min    = params["v_rx_min"]
        P_target_W  = params["p_target_mw"] * 1e-3
        Duty_tgt    = params["duty_target"]
        rx_ncaps    = params["rx_ncaps"]
        freq_min_hz = params["freq_min_hz"]
        freq_max_hz = params["freq_max_hz"]

        P_inst_W  = P_target_W / Duty_tgt
        R_load_eq = (V_rx_min ** 2) / P_inst_W
        log_cb(f"R_load_eq = {R_load_eq:.3f} Ω  |  P_inst = {P_inst_W*1e3:.1f} mW")

        # Drive frequencies at 1 kHz resolution across user-specified range
        drive_freqs = np.arange(freq_min_hz, freq_max_hz + 500.0, 1000.0)
        drive_freqs = drive_freqs[drive_freqs <= freq_max_hz + 1.0]
        log_cb(f"Drive freqs: {len(drive_freqs)} steps  "
               f"[{drive_freqs[0]/1e3:.0f}–{drive_freqs[-1]/1e3:.0f} kHz, 1 kHz res]  "
               f"(R_ac scaled from F0={F0_HZ/1e3:.0f} kHz via √f)")

        # ---- Derive geometry bounds from LHS samples ----------------------
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

        # Cap tables as numpy + GPU tensors
        tx_cap_table = _TX_CAP_TABLE
        tx_cap_nf    = np.array([v for v, _ in tx_cap_table], dtype=np.float32)
        tx_cap_lbls  = [lbl for _, lbl in tx_cap_table]
        rx_cap_table = _build_cap_table(rx_ncaps)
        rx_cap_nf    = np.array([v for v, _ in rx_cap_table], dtype=np.float32)
        rx_cap_lbls  = [lbl for _, lbl in rx_cap_table]
        log_cb(f"TX cap options: {len(tx_cap_nf)}  |  RX cap options: {len(rx_cap_nf)}")

        tx_cap_nf_g = torch.tensor(tx_cap_nf, device=device)   # (n_tx,)
        rx_cap_nf_g = torch.tensor(rx_cap_nf, device=device)   # (n_rx,)

        # NN feature column order (must match train_surrogate.py)
        base_cols = ["tx_turns", "tx_width", "tx_od_mm",
                     "rx_od_mm", "rx_turns", "rx_width", "freq_hz"]
        topo_cols = ["topo_parallel", "topo_parallel_pairs_ser", "topo_series"]
        all_cols  = feat_cols if feat_cols else (base_cols + sorted(topo_cols))
        _topo_key = ["topo_parallel", "topo_parallel_pairs_ser", "topo_series"]

        log_cb(f"Targeting {N:,} ID-valid geometry evaluations (batch size {BATCH_SIZE:,})…")

        rng = np.random.default_rng(42)

        # Accumulators — flat numpy arrays (no per-sample dicts until the very end)
        acc_fitness   = []
        acc_freq      = []
        acc_tx_turns  = []
        acc_tx_width  = []
        acc_rx_od     = []
        acc_rx_turns  = []
        acc_rx_width  = []
        acc_topo_idx  = []
        acc_L_tx      = []
        acc_L_rx      = []
        acc_M         = []
        acc_R_tx      = []
        acc_R_rx      = []
        acc_DCR_tx    = []
        acc_DCR_rx    = []
        acc_tx_ci     = []
        acc_rx_ci     = []
        acc_duty_vmin = []
        acc_duty_vmax = []
        acc_V_ind_min = []
        acc_eff       = []
        acc_Zin_re    = []
        acc_Zin_im    = []
        acc_tx_id     = []
        acc_rx_id     = []

        n_valid_total  = 0
        n_id_rejected  = 0
        n_id_passed    = 0
        b              = 0

        while n_id_passed < N:
            if cancel_flag.is_set():
                log_cb("Cancelled by user.")
                done_cb(None)
                return

            b   += 1
            bs   = BATCH_SIZE
            progress_cb(min(n_id_passed / N, 1.0) * 0.80)
            if b % 5 == 0 or b == 1:
                log_cb(f"Batch {b}  ({n_id_passed:,}/{N:,} ID-valid, "
                       f"{n_valid_total:,} circuit-valid)…")

            # ---- Sample geometry with constrained widths (CPU) -----------
            # Width upper-bound per sample enforces ID >= MIN_ID_MM exactly,
            # so virtually no combinations are wasted on ID rejection.
            # Derivation: id = OD - (2N+1)*w - (2N-1)*spacing >= MIN_ID
            #             => w <= (OD - MIN_ID - (2N-1)*spacing) / (2N+1)
            n_topos    = len(rx_topologies)
            tx_turns_s = rng.integers(tx_turns_min, tx_turns_max + 1, size=bs).astype(np.float32)
            tx_w_max_id = ((TX_OD_MM - MIN_ID_MM
                            - (2*tx_turns_s - 1) * _SPACING_MM)
                           / (2*tx_turns_s + 1)).astype(np.float32)
            tx_w_max   = np.minimum(tx_w_max_id, tx_width_max).clip(min=tx_width_min)
            tx_width_s = (tx_width_min
                          + rng.random(bs).astype(np.float32) * (tx_w_max - tx_width_min))

            rx_od_s    = rng.uniform(rx_od_min, rx_od_max, size=bs).astype(np.float32)
            rx_turns_s = rng.integers(rx_turns_min, rx_turns_max + 1, size=bs).astype(np.float32)
            rx_w_max_id = ((rx_od_s - MIN_ID_MM
                            - (2*rx_turns_s - 1) * _SPACING_MM)
                           / (2*rx_turns_s + 1)).astype(np.float32)
            rx_w_max   = np.minimum(rx_w_max_id, rx_width_max).clip(min=rx_width_min)
            rx_width_s = (rx_width_min
                          + rng.random(bs).astype(np.float32) * (rx_w_max - rx_width_min))

            topo_idx_s = rng.integers(0, n_topos, size=bs)

            # ---- ID sanity check (should be near-zero rejections now) ----
            tx_id_s = _inner_diameter_mm(TX_OD_MM, tx_width_s, tx_turns_s)
            rx_id_s = _inner_diameter_mm(rx_od_s,  rx_width_s, rx_turns_s)
            id_ok   = (tx_id_s >= MIN_ID_MM) & (rx_id_s >= MIN_ID_MM)
            n_id_rejected += int((~id_ok).sum())
            n_id_passed   += int(id_ok.sum())
            if not id_ok.any():
                continue

            # Trim to ID-valid samples
            tx_turns_s = tx_turns_s[id_ok];  tx_width_s = tx_width_s[id_ok]
            rx_od_s    = rx_od_s[id_ok];     rx_turns_s = rx_turns_s[id_ok]
            rx_width_s = rx_width_s[id_ok];  topo_idx_s = topo_idx_s[id_ok]
            tx_id_s    = tx_id_s[id_ok];     rx_id_s    = rx_id_s[id_ok]
            bs_v = int(id_ok.sum())

            topo_oh = np.zeros((bs_v, 3), dtype=np.float32)
            for ti, tname in enumerate(rx_topologies):
                col = _topo_key.index(f"topo_{tname}") if f"topo_{tname}" in _topo_key else ti
                topo_oh[topo_idx_s == ti, col] = 1.0

            # ---- Analytical DCR (CPU, vectorised) -----------------------
            # TX: 3 parallel 1-oz layers (fixed in training data)
            tx_len_m  = _spiral_length_m(TX_OD_MM, tx_width_s, tx_turns_s)
            DCR_tx_np = (_RHO_30C * tx_len_m
                         / (tx_width_s * 1e-3 * _TX_H_M * _TX_N_LAYERS)).astype(np.float32)

            # RX: 4 layers [1, 0.5, 0.5, 1] oz — system DCR depends on topology
            rx_len_m    = _spiral_length_m(rx_od_s, rx_width_s, rx_turns_s)
            rx_w_m      = rx_width_s * 1e-3
            DCR_rx_par  = _RHO_30C * rx_len_m / (rx_w_m * _RX_HSUM_PAR)
            DCR_rx_ser  = _RHO_30C * rx_len_m * _RX_HINV_SER / rx_w_m
            DCR_rx_pps  = _RHO_30C * rx_len_m * (1.0/_RX_HSUM_A + 1.0/_RX_HSUM_B) / rx_w_m
            DCR_rx_np   = np.empty(bs_v, dtype=np.float32)
            for ti, tname in enumerate(rx_topologies):
                mask_t = (topo_idx_s == ti)
                if not mask_t.any():
                    continue
                if tname == "parallel":
                    DCR_rx_np[mask_t] = DCR_rx_par[mask_t]
                elif tname == "series":
                    DCR_rx_np[mask_t] = DCR_rx_ser[mask_t]
                elif tname == "parallel_pairs_ser":
                    DCR_rx_np[mask_t] = DCR_rx_pps[mask_t]
                else:
                    DCR_rx_np[mask_t] = DCR_rx_par[mask_t]

            row_map = {
                "tx_turns":                tx_turns_s,
                "tx_width":                tx_width_s,
                "tx_od_mm":                np.full(bs_v, TX_OD_MM, dtype=np.float32),
                "rx_od_mm":                rx_od_s,
                "rx_turns":                rx_turns_s,
                "rx_width":                rx_width_s,
                "freq_hz":                 np.full(bs_v, F0_HZ,    dtype=np.float32),
                "topo_parallel":           topo_oh[:, 0],
                "topo_parallel_pairs_ser": topo_oh[:, 1],
                "topo_series":             topo_oh[:, 2],
            }
            X = np.column_stack([row_map[c] for c in all_cols]).astype(np.float32)

            # ---- NN inference (GPU) -------------------------------------
            X_sc = x_scaler.transform(X).astype(np.float32)
            with torch.no_grad():
                X_g  = torch.tensor(X_sc, device=device)
                Y_sc = model(X_g)
            Y = y_scaler.inverse_transform(Y_sc.cpu().numpy())

            # Predicted EM params at F0 — move to GPU
            L_tx_g    = torch.tensor(Y[:, 0], dtype=torch.float32, device=device)  # µH
            L_rx_g    = torch.tensor(Y[:, 1], dtype=torch.float32, device=device)  # µH
            M_g       = torch.tensor(Y[:, 2], dtype=torch.float32, device=device)  # µH
            R_tx_f0_g = torch.tensor(Y[:, 3], dtype=torch.float32, device=device)  # Ω @ F0
            R_rx_f0_g = torch.tensor(Y[:, 4], dtype=torch.float32, device=device)  # Ω @ F0

            # Pre-split R_ac(F0) into skin-effect part (scales ∝ √f) and DC part (constant).
            # R_ac(f) = DCR + R_skin(F0) * √(f/F0)  — physically correct, not naive scaling.
            DCR_tx_g      = torch.tensor(DCR_tx_np, device=device)
            DCR_rx_g      = torch.tensor(DCR_rx_np, device=device)
            R_skin_tx_f0  = (R_tx_f0_g - DCR_tx_g).clamp(min=0.0)
            R_skin_rx_f0  = (R_rx_f0_g - DCR_rx_g).clamp(min=0.0)

            # Per-batch best trackers (GPU tensors)
            INF = float('inf')
            best_fit_g   = torch.full((bs_v,), INF, dtype=torch.float32, device=device)
            best_freq_g  = torch.zeros(bs_v,        dtype=torch.float32, device=device)
            best_tx_ci_g = torch.zeros(bs_v,        dtype=torch.long,    device=device)
            best_rx_ci_g = torch.zeros(bs_v,        dtype=torch.long,    device=device)
            best_R_tx_g  = torch.zeros(bs_v,        dtype=torch.float32, device=device)
            best_R_rx_g  = torch.zeros(bs_v,        dtype=torch.float32, device=device)
            best_dvmin_g = torch.zeros(bs_v,        dtype=torch.float32, device=device)
            best_dvmax_g = torch.zeros(bs_v,        dtype=torch.float32, device=device)
            best_Vind_g  = torch.zeros(bs_v,        dtype=torch.float32, device=device)
            best_eff_g   = torch.zeros(bs_v,        dtype=torch.float32, device=device)
            best_Zre_g   = torch.zeros(bs_v,        dtype=torch.float32, device=device)
            best_Zim_g   = torch.zeros(bs_v,        dtype=torch.float32, device=device)

            arange_bs = torch.arange(bs_v, device=device)

            # ---- GPU circuit math for each drive frequency --------------
            for f_drive in drive_freqs:
                w          = 2.0 * math.pi * f_drive
                freq_scale = math.sqrt(f_drive / F0_HZ)

                # Physically correct R_ac: DC part fixed, skin part scales ∝ √f
                R_tx_g = DCR_tx_g + R_skin_tx_f0 * freq_scale
                R_rx_g = DCR_rx_g + R_skin_rx_f0 * freq_scale

                # RX: pick nearest stock cap to tune at this frequency
                C_rx_ideal_g = 1e9 / (w * w * L_rx_g * 1e-6)
                diff_rx      = (C_rx_ideal_g.unsqueeze(1)
                                - rx_cap_nf_g.unsqueeze(0)).abs()
                rx_ci_g  = diff_rx.argmin(dim=1)
                C_rx_g   = rx_cap_nf_g[rx_ci_g]

                X_L_rx   = w * L_rx_g * 1e-6
                X_C_rx   = 1.0 / (w * C_rx_g * 1e-9)
                Zrx_re   = R_rx_g + R_load_eq
                Zrx_im   = X_L_rx - X_C_rx
                Zrx_abs2 = Zrx_re ** 2 + Zrx_im ** 2
                Zrx_abs  = Zrx_abs2.sqrt()

                wM      = w * M_g * 1e-6
                Zref_re =  wM ** 2 * Zrx_re / Zrx_abs2
                Zref_im = -wM ** 2 * Zrx_im / Zrx_abs2

                X_L_tx        = w * L_tx_g * 1e-6
                Zin_re        = R_tx_g + Zref_re
                Zin_im_no_cap = X_L_tx + Zref_im

                X_C_tx_all  = 1.0 / (w * tx_cap_nf_g * 1e-9)                       # (n_tx,)
                Zin_im_all  = Zin_im_no_cap.unsqueeze(1) - X_C_tx_all.unsqueeze(0)  # (bs_v, n_tx)
                Zin_abs_all = (Zin_re.unsqueeze(1) ** 2
                               + Zin_im_all ** 2).sqrt()

                Vf_min = _HBRIDGE_COEFF * V_min
                Vf_max = _HBRIDGE_COEFF * V_max
                Vf_mid = _HBRIDGE_COEFF * (V_min + V_max) * 0.5

                It_min = Vf_min / Zin_abs_all
                It_max = Vf_max / Zin_abs_all
                It_mid = Vf_mid / Zin_abs_all

                wM_Zrx    = (wM / Zrx_abs).unsqueeze(1)
                V_ind_min = wM.unsqueeze(1) * It_min

                Pload_min = (It_min * wM_Zrx) ** 2 * R_load_eq
                Pload_max = (It_max * wM_Zrx) ** 2 * R_load_eq
                Pload_mid = (It_mid * wM_Zrx) ** 2 * R_load_eq

                Duty_vmin = P_target_W / Pload_min.clamp(min=1e-30)
                Duty_vmax = P_target_W / Pload_max.clamp(min=1e-30)
                P_tx_mid  = It_mid ** 2 * Zin_re.unsqueeze(1)
                eff_mid   = torch.where(P_tx_mid > 0,
                                        Pload_mid / P_tx_mid.clamp(min=1e-30),
                                        torch.zeros_like(P_tx_mid))

                valid = (
                    (Zin_im_all > 0.0)      &
                    (V_ind_min  >= V_rx_min) &
                    (Duty_vmin  <= 1.0)      &
                    (Duty_vmax  <= 1.0)
                )

                duty_err    = (Duty_vmin - Duty_tgt).abs() + (Duty_vmax - Duty_tgt).abs()
                # Efficiency is the primary objective: minimise (1 - eff).
                # Duty-cycle matching is a soft secondary term (weight 0.15).
                # DCR is already captured through eff_mid — no separate penalty.
                fitness_all = torch.where(valid,
                                          (1.0 - eff_mid) + duty_err * 0.15,
                                          torch.full_like(duty_err, INF))

                best_ti  = fitness_all.argmin(dim=1)
                best_fit = fitness_all[arange_bs, best_ti]

                improved = torch.isfinite(best_fit) & (best_fit < best_fit_g)
                if not improved.any():
                    continue

                def _upd1(acc, new):
                    return torch.where(improved, new, acc)
                def _gather(t2d):
                    return t2d[arange_bs, best_ti]

                best_fit_g   = _upd1(best_fit_g,   best_fit)
                best_freq_g  = _upd1(best_freq_g,  torch.full((bs_v,), f_drive, device=device))
                best_tx_ci_g = torch.where(improved, best_ti,  best_tx_ci_g)
                best_rx_ci_g = torch.where(improved, rx_ci_g,  best_rx_ci_g)
                best_R_tx_g  = _upd1(best_R_tx_g,  R_tx_g)
                best_R_rx_g  = _upd1(best_R_rx_g,  R_rx_g)
                best_dvmin_g = _upd1(best_dvmin_g, _gather(Duty_vmin))
                best_dvmax_g = _upd1(best_dvmax_g, _gather(Duty_vmax))
                best_Vind_g  = _upd1(best_Vind_g,  _gather(V_ind_min))
                best_eff_g   = _upd1(best_eff_g,   _gather(eff_mid))
                best_Zre_g   = _upd1(best_Zre_g,   Zin_re)
                best_Zim_g   = _upd1(best_Zim_g,   _gather(Zin_im_all))

            # ---- Collect valid samples from this batch (CPU transfer) ---
            valid_mask = torch.isfinite(best_fit_g).cpu().numpy()
            if not valid_mask.any():
                continue

            bf     = best_fit_g.cpu().numpy()
            bfreq  = best_freq_g.cpu().numpy()
            btxci  = best_tx_ci_g.cpu().numpy()
            brxci  = best_rx_ci_g.cpu().numpy()
            bRtx   = best_R_tx_g.cpu().numpy()
            bRrx   = best_R_rx_g.cpu().numpy()
            bdvmin = best_dvmin_g.cpu().numpy()
            bdvmax = best_dvmax_g.cpu().numpy()
            bVind  = best_Vind_g.cpu().numpy()
            beff   = best_eff_g.cpu().numpy()
            bZre   = best_Zre_g.cpu().numpy()
            bZim   = best_Zim_g.cpu().numpy()
            bL_tx  = L_tx_g.cpu().numpy()
            bL_rx  = L_rx_g.cpu().numpy()
            bM     = M_g.cpu().numpy()

            acc_fitness.append(bf[valid_mask])
            acc_freq.append(bfreq[valid_mask])
            acc_tx_turns.append(tx_turns_s[valid_mask])
            acc_tx_width.append(tx_width_s[valid_mask])
            acc_rx_od.append(rx_od_s[valid_mask])
            acc_rx_turns.append(rx_turns_s[valid_mask])
            acc_rx_width.append(rx_width_s[valid_mask])
            acc_topo_idx.append(topo_idx_s[valid_mask])
            acc_L_tx.append(bL_tx[valid_mask])
            acc_L_rx.append(bL_rx[valid_mask])
            acc_M.append(bM[valid_mask])
            acc_R_tx.append(bRtx[valid_mask])
            acc_R_rx.append(bRrx[valid_mask])
            acc_DCR_tx.append(DCR_tx_np[valid_mask])
            acc_DCR_rx.append(DCR_rx_np[valid_mask])
            acc_tx_ci.append(btxci[valid_mask])
            acc_rx_ci.append(brxci[valid_mask])
            acc_duty_vmin.append(bdvmin[valid_mask])
            acc_duty_vmax.append(bdvmax[valid_mask])
            acc_V_ind_min.append(bVind[valid_mask])
            acc_eff.append(beff[valid_mask])
            acc_Zin_re.append(bZre[valid_mask])
            acc_Zin_im.append(bZim[valid_mask])
            acc_tx_id.append(tx_id_s[valid_mask])
            acc_rx_id.append(rx_id_s[valid_mask])

            n_valid_total += int(valid_mask.sum())

        log_cb(f"Sorting {n_valid_total:,} circuit-valid configurations…  "
               f"({n_id_passed:,} ID-valid evaluated, {n_id_rejected:,} ID-rejected)")
        progress_cb(0.95)

        if n_valid_total == 0:
            done_cb([])
            return

        # Concatenate and find top 10 — only build dicts for those
        all_fitness = np.concatenate(acc_fitness)
        order       = np.argsort(all_fitness)[:10]

        def _cat(lst): return np.concatenate(lst)
        cat = {
            "fitness":   all_fitness,
            "freq":      _cat(acc_freq),
            "tx_turns":  _cat(acc_tx_turns),
            "tx_width":  _cat(acc_tx_width),
            "rx_od":     _cat(acc_rx_od),
            "rx_turns":  _cat(acc_rx_turns),
            "rx_width":  _cat(acc_rx_width),
            "topo_idx":  _cat(acc_topo_idx),
            "L_tx":      _cat(acc_L_tx),
            "L_rx":      _cat(acc_L_rx),
            "M":         _cat(acc_M),
            "R_tx":      _cat(acc_R_tx),
            "R_rx":      _cat(acc_R_rx),
            "DCR_tx":    _cat(acc_DCR_tx),
            "DCR_rx":    _cat(acc_DCR_rx),
            "tx_ci":     _cat(acc_tx_ci),
            "rx_ci":     _cat(acc_rx_ci),
            "dvmin":     _cat(acc_duty_vmin),
            "dvmax":     _cat(acc_duty_vmax),
            "V_ind_min": _cat(acc_V_ind_min),
            "eff":       _cat(acc_eff),
            "Zin_re":    _cat(acc_Zin_re),
            "Zin_im":    _cat(acc_Zin_im),
            "tx_id":     _cat(acc_tx_id),
            "rx_id":     _cat(acc_rx_id),
        }

        top = []
        for i in order:
            ti   = int(cat["tx_ci"][i])
            rxi  = int(cat["rx_ci"][i])
            topo = rx_topologies[int(cat["topo_idx"][i])]
            f_hz = float(cat["freq"][i])
            w_hz = 2.0 * math.pi * f_hz
            L_tx = float(cat["L_tx"][i])
            L_rx = float(cat["L_rx"][i])
            R_tx = float(cat["R_tx"][i])
            R_rx = float(cat["R_rx"][i])
            top.append({
                "fitness":       float(cat["fitness"][i]),
                "freq_hz":       f_hz,
                "tx_turns":      int(round(float(cat["tx_turns"][i]))),
                "tx_width":      float(cat["tx_width"][i]),
                "tx_od_mm":      TX_OD_MM,
                "rx_od_mm":      float(cat["rx_od"][i]),
                "rx_turns":      int(round(float(cat["rx_turns"][i]))),
                "rx_width":      float(cat["rx_width"][i]),
                "rx_topology":   topo,
                "rx_eff_turns":  _rx_effective_turns(int(round(float(cat["rx_turns"][i]))), topo),
                "L_tx_uH":       L_tx,
                "L_rx_uH":       L_rx,
                "M_uH":          float(cat["M"][i]),
                "R_tx_ohm":      R_tx,
                "R_rx_ohm":      R_rx,
                "DCR_tx_ohm":    float(cat["DCR_tx"][i]),
                "DCR_rx_ohm":    float(cat["DCR_rx"][i]),
                "Q_tx":          (w_hz * L_tx * 1e-6 / R_tx) if R_tx > 0 else 0.0,
                "Q_rx":          (w_hz * L_rx * 1e-6 / R_rx) if R_rx > 0 else 0.0,
                "C_tx_nf":       float(tx_cap_nf[ti]),
                "C_tx_label":    tx_cap_lbls[ti],
                "C_rx_nf":       float(rx_cap_nf[rxi]),
                "C_rx_label":    rx_cap_lbls[rxi],
                "Duty_vmin":     float(cat["dvmin"][i]),
                "Duty_vmax":     float(cat["dvmax"][i]),
                "V_ind_min_V":   float(cat["V_ind_min"][i]),
                "eff_mid":       float(cat["eff"][i]),
                "Zin_re":        float(cat["Zin_re"][i]),
                "Zin_im":        float(cat["Zin_im"][i]),
                "tx_id_mm":      float(cat["tx_id"][i]),
                "rx_id_mm":      float(cat["rx_id"][i]),
            })

        progress_cb(1.0)
        log_cb(f"Done — {n_id_passed:,} ID-valid evaluated, {n_valid_total:,} circuit-valid. Top {len(top)} shown.")
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
        self._results         = []
        self._last_run_params = {}   # p_target_mw etc. stored at run time
        self._selected_idx    = -1  # index of row shown in detail panels
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

        # Frequency sweep
        ff = ttk.LabelFrame(left, text="Frequency Sweep")
        ff.pack(fill="x", padx=8, pady=4)
        self._freq_min = self._row(ff, "Freq min (kHz):", "120")
        self._freq_max = self._row(ff, "Freq max (kHz):", "130")
        ttk.Label(ff, text="Resolution: 1 kHz steps", foreground="gray",
                  font=("TkDefaultFont", 8)).pack(anchor="w", padx=6, pady=(0, 4))

        # Optimizer settings
        of = ttk.LabelFrame(left, text="Optimizer Settings")
        of.pack(fill="x", padx=8, pady=4)
        self._n_combos = self._row(of, "Combinations (M):", "10")

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
        det_lf.pack(fill="x", padx=8, pady=(0, 4))
        self._build_detail_panels(det_lf)

        send_row = ttk.Frame(right)
        send_row.pack(fill="x", padx=8, pady=(0, 8))
        self._send_btn = ttk.Button(send_row, text="Send Selected to Simulation",
                                    command=self._on_send_to_sim, state="disabled",
                                    width=30)
        self._send_btn.pack(side="left")

    def _row(self, parent, label, default):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label, width=26, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var, width=12).pack(side="left", padx=2)
        return var

    def _build_tree(self, parent):
        cols = (
            "#", "Fitness", "Eff%", "f kHz",
            "TX T/Ly", "TX width", "RX OD", "RX T/Ly", "RX width", "RX Topo",
            "TX L µH", "RX L µH", "M L µH", "TX R Ω", "RX R Ω",
            "TX DCR mΩ", "RX DCR mΩ", "TX Q", "RX Q",
            "TX cap", "RX cap",
            "DutyVmin", "DutyVmax", "V_ind",
        )
        widths = {
            "#": 26, 
            "Fitness": 50, 
            "Eff%": 34, 
            "f kHz": 42,
            "TX T/Ly": 55, 
            "TX width": 58,
            "RX OD": 50, 
            "RX T/Ly": 55, 
            "RX width": 58, 
            "RX Topo": 76,
            "TX L µH": 58, 
            "RX L µH": 58, 
            "M L µH": 52,
            "TX R Ω": 52, 
            "RX R Ω": 52,
            "TX DCR mΩ": 75, 
            "RX DCR mΩ": 75, 
            "TX Q": 40, 
            "RX Q": 40,
            "TX cap": 85, 
            "RX cap": 50,
            "DutyVmin": 70, 
            "DutyVmax": 70,
            "V_ind": 40, 
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

        v_min      = flt(self._v_min,    "TX V_min",             lo=0.1)
        v_max      = flt(self._v_max,    "TX V_max",             lo=v_min)
        v_rx_min   = flt(self._v_rx_min, "Min RX V_induced",     lo=0.1)
        p_target   = flt(self._p_target, "Target RX Power (mW)", lo=0.1)
        duty_tgt   = flt(self._duty_tgt, "Target Duty Cycle",    lo=0.01, hi=1.0)
        n_combos   = int(flt(self._n_combos, "Combinations (M)",  lo=0.001) * 1_000_000)
        rx_ncaps   = int(self._rx_ncaps_var.get())
        freq_min   = flt(self._freq_min, "Freq min (kHz)",       lo=50.0, hi=500.0)
        freq_max   = flt(self._freq_max, "Freq max (kHz)",       lo=freq_min)
        return dict(v_min=v_min, v_max=v_max, v_rx_min=v_rx_min,
                    p_target_mw=p_target, duty_target=duty_tgt,
                    n_combos=n_combos, rx_ncaps=rx_ncaps,
                    freq_min_hz=freq_min * 1e3,
                    freq_max_hz=freq_max * 1e3)

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
            tag  = tag_map.get(i, "")
            vals = (
                i + 1,
                f"{r['fitness']:.4f}",
                f"{r['eff_mid']*100:.1f}",
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
                f"{r['DCR_tx_ohm']*1e3:.1f}",
                f"{r['DCR_rx_ohm']*1e3:.1f}",
                f"{r['Q_tx']:.1f}",
                f"{r['Q_rx']:.1f}",
                r["C_tx_label"],
                r["C_rx_label"],
                f"{r['Duty_vmin']:.3f}",
                f"{r['Duty_vmax']:.3f}",
                f"{r['V_ind_min_V']:.2f}",
            )
            self._tree.insert("", "end", values=vals, tags=(tag,) if tag else ())

    # -----------------------------------------------------------------------
    # Detail panel — 3-column (NN prediction | Sim result | Delta)
    # -----------------------------------------------------------------------

    def _build_detail_panels(self, parent):
        """Build three equal-width text columns with a shared vertical scrollbar."""
        FONT = ("Consolas", 9)
        BG   = "#f8f8f8"
        H    = 20  # line height

        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        # Shared vertical scrollbar on the far right
        vsb = ttk.Scrollbar(outer, orient="vertical")
        vsb.pack(side="right", fill="y")

        def _scroll_all(*args):
            for t in (self._det_nn, self._det_sim, self._det_delta):
                t.yview(*args)

        vsb.configure(command=_scroll_all)

        cols_frame = ttk.Frame(outer)
        cols_frame.pack(fill="both", expand=True)
        cols_frame.columnconfigure(0, weight=1, uniform="dc")
        cols_frame.columnconfigure(2, weight=1, uniform="dc")
        cols_frame.columnconfigure(4, weight=1, uniform="dc")

        def _make_col(col_idx, title):
            hdr = ttk.Frame(cols_frame)
            hdr.grid(row=0, column=col_idx, sticky="ew")
            ttk.Label(hdr, text=title, font=("TkDefaultFont", 9, "bold"),
                      anchor="center", background="#e8eef8"
                      ).pack(fill="x", padx=1, pady=(2, 0))
            t = tk.Text(cols_frame, height=H, state="disabled",
                        font=FONT, wrap="none", background=BG,
                        borderwidth=0, highlightthickness=0)
            t.grid(row=1, column=col_idx, sticky="nsew")
            t.configure(yscrollcommand=vsb.set)

            def _on_wheel(event):
                delta = -1 * (event.delta // 120) if event.delta else (1 if event.num == 5 else -1)
                _scroll_all("scroll", delta, "units")
                return "break"

            t.bind("<MouseWheel>", _on_wheel)
            t.bind("<Button-4>", _on_wheel)
            t.bind("<Button-5>", _on_wheel)
            return t

        self._det_nn    = _make_col(0, "NN Prediction")
        tk.Frame(cols_frame, bg="#b0b0b0", width=1).grid(row=0, column=1, rowspan=2, sticky="ns")
        self._det_sim   = _make_col(2, "Simulation Result")
        tk.Frame(cols_frame, bg="#b0b0b0", width=1).grid(row=0, column=3, rowspan=2, sticky="ns")
        self._det_delta = _make_col(4, "Delta  (Sim − NN)")

    def _write_col(self, widget, lines):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", "\n".join(lines))
        widget.configure(state="disabled")

    def _clear_detail(self):
        for w in (self._det_nn, self._det_sim, self._det_delta):
            w.configure(state="normal")
            w.delete("1.0", "end")
            w.configure(state="disabled")

    def _on_tree_select(self, _=None):
        sel = self._tree.selection()
        if not sel:
            self._send_btn.configure(state="disabled")
            self._selected_idx = -1
            return
        iid   = sel[0]
        items = self._tree.get_children()
        idx   = list(items).index(iid)
        if idx >= len(self._results):
            self._send_btn.configure(state="disabled")
            self._selected_idx = -1
            return
        self._send_btn.configure(state="normal")
        self._selected_idx = idx
        sim_res = getattr(self.app, "sim_tab", None)
        sim_res = sim_res.last_result if sim_res else None
        L, M, R = self._make_detail_columns(idx + 1, self._results[idx], sim_res)
        self._write_col(self._det_nn,    L)
        self._write_col(self._det_sim,   M)
        self._write_col(self._det_delta, R)

    def _populate_sim_panels(self, r_nn, sim_result):
        """Called via callback after FastHenry completes; refreshes mid + right cols."""
        if self._selected_idx < 0 or self._selected_idx >= len(self._results):
            return
        if self._results[self._selected_idx] is not r_nn:
            return
        L, M, R = self._make_detail_columns(self._selected_idx + 1, r_nn, sim_result)
        self._write_col(self._det_nn,    L)
        self._write_col(self._det_sim,   M)
        self._write_col(self._det_delta, R)

    # -----------------------------------------------------------------------
    # Column content builder
    # -----------------------------------------------------------------------

    def _make_detail_columns(self, rank, r, sim_res=None):
        """Build (left_lines, mid_lines, right_lines) with identical length."""
        topo      = r["rx_topology"]
        eff_turns = r["rx_eff_turns"]
        tx_l_mm   = (_spiral_length_m(r["tx_od_mm"], r["tx_width"], r["tx_turns"])
                     * 1e3 * _TX_N_LAYERS)
        rx_l_mm   = (_spiral_length_m(r["rx_od_mm"], r["rx_width"], r["rx_turns"])
                     * 1e3 * 4)

        # Extract sim values if available
        have_sim = (sim_res is not None and
                    sim_res.get("n_ports", 0) == 2)
        if have_sim:
            Zmat  = sim_res["Zmat"]
            f_sim = sim_res["frequency"]
            w_s   = 2.0 * math.pi * f_sim
            sL_tx = Zmat[0][0].imag / w_s * 1e6   # µH
            sL_rx = Zmat[1][1].imag / w_s * 1e6   # µH
            sM    = Zmat[0][1].imag / w_s * 1e6   # µH
            sR_tx = Zmat[0][0].real * 1e3          # mΩ
            sR_rx = Zmat[1][1].real * 1e3          # mΩ
            sk    = sM / math.sqrt(sL_tx * sL_rx) if sL_tx > 0 and sL_rx > 0 else 0.0
            sQ_tx = w_s * sL_tx * 1e-6 / Zmat[0][0].real if Zmat[0][0].real > 0 else 0.0
            sQ_rx = w_s * sL_rx * 1e-6 / Zmat[1][1].real if Zmat[1][1].real > 0 else 0.0
        else:
            f_sim = sL_tx = sL_rx = sM = sR_tx = sR_rx = sk = sQ_tx = sQ_rx = None

        nn_k = r["M_uH"] / math.sqrt(r["L_tx_uH"] * r["L_rx_uH"])

        SEP = "─" * 44

        def _pct(nn, s):
            return f" ({(s - nn) / nn * 100:+.1f}%)" if nn != 0 else ""

        def _d(nn_val, s_val, label, fmt, unit):
            if not have_sim:
                return ""
            d = s_val - nn_val
            return f"    {label:<14}{fmt.format(d)}{unit}{_pct(nn_val, s_val)}"

        # Precompute sim cell strings (empty when no sim)
        s_freq = f"    {'Frequency':<16}{f_sim/1e3:.3f} kHz"      if have_sim else "    Frequency        —"
        s_ltx  = f"    {'L_TX':<16}{sL_tx:.4f} µH"                if have_sim else "  —"
        s_lrx  = f"    {'L_RX':<16}{sL_rx:.4f} µH"                if have_sim else "  —"
        s_m    = f"    {'M':<16}{sM:.5f} µH"                       if have_sim else "  —"
        s_k    = f"    {'k':<16}{sk:.4f}"                          if have_sim else "  —"
        s_rtx  = f"    {'R_TX':<16}{sR_tx:.2f} mΩ"                 if have_sim else "  —"
        s_rrx  = f"    {'R_RX':<16}{sR_rx:.2f} mΩ"                 if have_sim else "  —"
        s_qtx  = f"    {'Q_TX':<16}{sQ_tx:.1f}"                    if have_sim else "  —"
        s_qrx  = f"    {'Q_RX':<16}{sQ_rx:.1f}"                    if have_sim else "  —"

        em_hdr_m = "  EM PARAMETERS  (Simulated)"    if have_sim else "  EM PARAMETERS  (run sim…)"
        em_hdr_r = "  DELTA  (Sim − NN)"             if have_sim else "  DELTA  (run sim…)"

        rows = [
            # ── header ────────────────────────────────────────────────────────
            (SEP,                                          SEP,                   SEP),
            (f"  Rank #{rank}  —  Fitness {r['fitness']:.6f}",
             "  FastHenry Sim Results",
             "  Delta  (Sim − NN)"),
            (SEP,                                          SEP,                   SEP),
            ("", "", ""),
            # ── geometry (left only) ──────────────────────────────────────────
            ("  GEOMETRY  (T/Ly = turns/layer)",           "",                    ""),
            (f"    TX: OD={r['tx_od_mm']:.1f}  T={r['tx_turns']}  W={r['tx_width']:.3f} mm",
             "", ""),
            (f"        ID={r['tx_id_mm']:.1f} mm  wire={tx_l_mm:.0f} mm  (3 layers ||)",
             "", ""),
            (f"    RX: OD={r['rx_od_mm']:.1f}  T={r['rx_turns']}  W={r['rx_width']:.3f} mm  topo={topo}",
             "", ""),
            (f"        ID={r['rx_id_mm']:.1f} mm  wire={rx_l_mm:.0f} mm"
             + (f"  eff.turns={eff_turns}" if topo != "parallel" else ""),
             "", ""),
            ("", "", ""),
            # ── drive ─────────────────────────────────────────────────────────
            ("  DRIVE",                                    "",                    ""),
            (f"    {'Frequency':<16}{r['freq_hz']/1e3:.3f} kHz",   s_freq,       ""),
            (f"    {'C_TX':<16}{r['C_tx_nf']:.4f} nF  ({r['C_tx_label']})", "", ""),
            (f"    {'C_RX':<16}{r['C_rx_nf']:.4f} nF  ({r['C_rx_label']})", "", ""),
            ("", "", ""),
            # ── EM parameters ─────────────────────────────────────────────────
            ("  EM PARAMETERS  (NN predicted)",            em_hdr_m,             em_hdr_r),
            (f"    {'L_TX':<16}{r['L_tx_uH']:.4f} µH",
             s_ltx,
             _d(r["L_tx_uH"], sL_tx, "ΔL_TX", "{:+.4f}", " µH")),
            (f"    {'L_RX':<16}{r['L_rx_uH']:.4f} µH",
             s_lrx,
             _d(r["L_rx_uH"], sL_rx, "ΔL_RX", "{:+.4f}", " µH")),
            (f"    {'M':<16}{r['M_uH']:.5f} µH",
             s_m,
             _d(r["M_uH"],    sM,    "ΔM",    "{:+.5f}", " µH")),
            (f"    {'k':<16}{nn_k:.4f}",
             s_k,
             _d(nn_k,         sk,    "Δk",    "{:+.4f}", "")),
            (f"    {'R_TX':<16}{r['R_tx_ohm']*1e3:.2f} mΩ"
             f"  (DCR={r['DCR_tx_ohm']*1e3:.2f}  skin={max(0,r['R_tx_ohm']-r['DCR_tx_ohm'])*1e3:.2f})",
             s_rtx,
             _d(r["R_tx_ohm"]*1e3, sR_tx, "ΔR_TX", "{:+.2f}", " mΩ")),
            (f"    {'R_RX':<16}{r['R_rx_ohm']*1e3:.2f} mΩ"
             f"  (DCR={r['DCR_rx_ohm']*1e3:.2f}  skin={max(0,r['R_rx_ohm']-r['DCR_rx_ohm'])*1e3:.2f})",
             s_rrx,
             _d(r["R_rx_ohm"]*1e3, sR_rx, "ΔR_RX", "{:+.2f}", " mΩ")),
            (f"    {'Q_TX':<16}{r['Q_tx']:.1f}",          s_qtx,                ""),
            (f"    {'Q_RX':<16}{r['Q_rx']:.1f}",          s_qrx,                ""),
            ("", "", ""),
            # ── circuit (left only) ───────────────────────────────────────────
            ("  CIRCUIT",                                  "",                    ""),
            (f"    Z_in = {r['Zin_re']*1e3:.2f}+j{r['Zin_im']*1e3:.2f} mΩ"
             f"  ({'inductive ✓' if r['Zin_im'] > 0 else 'CAPACITIVE ✗'})",
             "", ""),
            (f"    V_induced_min = {r['V_ind_min_V']:.3f} V", "", ""),
            ("", "", ""),
            # ── performance (left only) ───────────────────────────────────────
            ("  PERFORMANCE",                              "",                    ""),
            (f"    Duty @ V_min   {r['Duty_vmin']:.4f}  ({r['Duty_vmin']*100:.1f}%)", "", ""),
            (f"    Duty @ V_max   {r['Duty_vmax']:.4f}  ({r['Duty_vmax']*100:.1f}%)", "", ""),
            (f"    Efficiency     {r['eff_mid']*100:.2f}%  (at mid supply)", "", ""),
        ]

        L = [row[0] for row in rows]
        M = [row[1] for row in rows]
        R = [row[2] for row in rows]
        # Pad all to same length (they're already equal via zip, but be safe)
        n = max(len(L), len(M), len(R))
        L += [""] * (n - len(L))
        M += [""] * (n - len(M))
        R += [""] * (n - len(R))
        return L, M, R

    # -----------------------------------------------------------------------
    # Send selected result to Simulation tab
    # -----------------------------------------------------------------------

    def _on_send_to_sim(self):
        sel = self._tree.selection()
        if not sel:
            return
        items = self._tree.get_children()
        idx   = list(items).index(sel[0])
        if idx >= len(self._results):
            return
        r = self._results[idx]

        if self.app is None:
            messagebox.showerror("Send to Sim", "No app reference available.")
            return

        temp_dir = (getattr(self.app.sim_tab, "temp_dir", None)
                    or getattr(self.app, "temp_dir", None)
                    or os.path.join(os.path.dirname(_APP_ROOT), "..", "temp"))

        try:
            self._build_and_register(r, temp_dir)
        except Exception as e:
            messagebox.showerror("Send to Sim", f"Failed to build coils:\n{e}\n\n{traceback.format_exc()}")
            return

        # Push all relevant values into the sim tab
        sim = self.app.sim_tab
        freq_hz = r["freq_hz"]
        sim.freq_var.set(str(int(freq_hz)))
        sim.fc_var.set(str(int(freq_hz)))
        sim.cap_tx.var.set(f"{r['C_tx_nf']:g}")
        sim.cap_rx.var.set(f"{r['C_rx_nf']:g}")
        p_mw = self._last_run_params.get("p_target_mw", "")
        if p_mw:
            sim.p_avg_var.set(f"{p_mw:g}")

        # Register one-shot callback to populate sim panels when done
        _r_ref = r
        def _sim_done_cb(result, _r=_r_ref):
            try:
                sim._done_callbacks.remove(_sim_done_cb)
            except ValueError:
                pass
            self.after(0, lambda: self._populate_sim_panels(_r, result))

        sim._done_callbacks.append(_sim_done_cb)

        # Switch to Simulation tab and start
        try:
            self.app._nb.select(sim)
        except Exception:
            pass
        sim._on_start()

    def _build_and_register(self, r, temp_dir):
        """Generate TX and RX .inp files from result params and register with sim_tab."""
        os.makedirs(temp_dir, exist_ok=True)

        # --- TX: 53 mm OD, 3 parallel 1-oz layers, topology = parallel ---
        tx_sp = pc.SpiralParams(
            od_mm=r["tx_od_mm"],
            trace_width_mm=r["tx_width"],
            spacing_mm=_SPACING_MM,
            turns=r["tx_turns"],
            resolution_mm=0.6,
        )
        tx_stackup = pc.StackUp(
            slots=[
                pc.LayerSlot(active=True,  copper_oz=1.0),
                pc.LayerSlot(active=True,  copper_oz=1.0),
                pc.LayerSlot(active=True,  copper_oz=1.0),
                pc.LayerSlot(active=False, copper_oz=1.0),
            ],
            outer_gap_mm=0.2,
            inner_gap_mm=1.3,
        )
        ok, msg = pc.validate_spiral(tx_sp)
        if not ok:
            raise ValueError(f"TX spiral invalid: {msg}")
        tx_layers = pc.active_layer_data(tx_sp, tx_stackup)

        tx_path = os.path.join(temp_dir, "nn_auto_tx.inp")
        fmin = r["freq_hz"]
        fmax = fmin + 15000.0
        pc.write_topology_inp("parallel", tx_layers, tx_path,
                              w_mm=r["tx_width"], fmin=fmin, fmax=fmax)

        tx_meta = {
            "role": "TX",
            "topology": "parallel",
            "layer_params": [(r["tx_width"], ld["h_mm"], len(ld["nodes"]))
                             for ld in tx_layers],
            "nodes_by_layer": [list(ld["nodes"]) for ld in tx_layers],
        }

        # --- RX: variable OD, 4 layers [1, 0.5, 0.5, 1] oz, variable topo ---
        rx_topo = r["rx_topology"]
        rx_sp = pc.SpiralParams(
            od_mm=r["rx_od_mm"],
            trace_width_mm=r["rx_width"],
            spacing_mm=_SPACING_MM,
            turns=r["rx_turns"],
            resolution_mm=0.6,
        )
        rx_stackup = pc.StackUp(
            slots=[
                pc.LayerSlot(active=True, copper_oz=1.0),
                pc.LayerSlot(active=True, copper_oz=0.5),
                pc.LayerSlot(active=True, copper_oz=0.5),
                pc.LayerSlot(active=True, copper_oz=1.0),
            ],
            outer_gap_mm=0.2,
            inner_gap_mm=1.3,
        )
        ok, msg = pc.validate_spiral(rx_sp)
        if not ok:
            raise ValueError(f"RX spiral invalid: {msg}")
        rx_layers = pc.active_layer_data(rx_sp, rx_stackup)

        rx_path = os.path.join(temp_dir, "nn_auto_rx.inp")
        pc.write_topology_inp(rx_topo, rx_layers, rx_path,
                              w_mm=r["rx_width"], fmin=fmin, fmax=fmax)

        # Use writer-native node ordering so via_connections_for_topology
        # indices (which expect post-reversal nodes) are correct in the viewer.
        rx_flags = pc.series_reverse_flags_for_topology(rx_topo, len(rx_layers))
        rx_native = pc.reverse_nodes_for_series_flow(rx_layers, rx_flags)

        rx_meta = {
            "role": "RX",
            "topology": rx_topo,
            "layer_params": [(r["rx_width"], ld["h_mm"], len(ld["nodes"]))
                             for ld in rx_native],
            "nodes_by_layer": [list(ld["nodes"]) for ld in rx_native],
        }

        sim = self.app.sim_tab
        sim.register_coil("TX", "Automation NN", tx_path, tx_meta)
        sim.register_coil("RX", "Automation NN", rx_path, rx_meta)

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
