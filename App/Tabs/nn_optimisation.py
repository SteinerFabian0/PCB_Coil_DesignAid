#!/usr/bin/env python3
"""
NN Optimisation tab — iterative surrogate-guided coil optimisation.

Each iteration:
  1. Sample N combinations via NN → rank by system efficiency
  2. Take top-K winners → run through FastHenry (parallel)
  3. Append results to refined_results.json
  4. Retrain model (overwrites surrogate_model.pth etc.)
  5. Repeat until max_iters or same winner 3 times in a row.

Evaluation physics mirrors nn_analysis_tab: skin-effect AC resistance,
full-bridge rectifier losses, duty-cycle feasibility.

Domain bounds are read from the selected model's domain.json.
User inputs on the left narrow the domain further.
"""

import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import traceback
import uuid

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
from cap_combinator import E_VALUES_NF



# ─────────────────────────────────────────────────────────────────────────────
# UI Defaults  ← tweak these to change the tab's initial values
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_P_TARGET_MW    = "30"     # mW (post-LDO downstream load)
DEFAULT_V_MIN          = "3.2"    # V
DEFAULT_V_MAX          = "4.4"    # V
DEFAULT_V_CAP_TARGET   = "4.5"    # V — controller steady-state cap target
DEFAULT_RX_CAP_VRATING = "100"    # V — rated voltage of C_rx series resonant cap
DEFAULT_D_MIN_PCT      = "20"     # blank = no floor
DEFAULT_D_MAX_PCT      = "90"     # blank = no ceiling
DEFAULT_FREQ_MIN_KHZ   = "200"    # kHz — user operating range (min)
DEFAULT_FREQ_MAX_KHZ   = "360"    # kHz — user operating range (max)
DEFAULT_DITHER_AMP_KHZ = "5"      # frequency dither amplitude (kHz)
DEFAULT_ZVS_MARGIN_PCT = "50"     # f_drive_min must be this % above f0_tx
DEFAULT_N_COMBOS_M     = "80"     # millions
DEFAULT_MAX_ITERS      = "10"
DEFAULT_TOP_K          = "30"
DEFAULT_FH_WORKERS     = "15"
DEFAULT_FH_TIMEOUT_S   = "360"
DEFAULT_TR_EPOCHS    = "300"
DEFAULT_TR_BATCH     = "1024"   # bigger batches saturate the 4070 Ti and cut DataLoader overhead
DEFAULT_TR_LR        = "0.0005"
DEFAULT_TR_VAL_SPLIT = "0.2"

# Resolution at which sampled combinations are snapped to physical grid.
# Spurious sub-resolution variation contaminates the surrogate's training
# distribution and the FastHenry refinement budget. Match the values you
# can actually realise on a PCB.
_GRID_TURNS_STEP    = 1
_GRID_OD_MM_STEP    = 0.2     # all diameters
_GRID_WIDTH_MM_STEP = 0.01    # trace width
_GRID_FREQ_HZ_STEP  = 1000.0
_GRID_OZ_STEP       = 0.5

# ── Physical constants ────────────────────────────────────────────────────────
BATCH_SIZE   = 500_000
_SPACING_MM  = 0.16
_OZ_MM       = 0.035                    # 1 oz copper thickness in mm
_RHO_30C     = 1.724e-8 * (1 + 0.00393 * 10)

# On NN_V8, TX and RX topologies are both fixed to series and are NOT NN
# input features. RX has 4 active layers in series, with outer turns
# (slots 1 & 4) and inner turns (slots 2 & 3) as independent sampled
# integers. Inner trace width is auto-computed for endpoint match.
_TX_TOPOLOGY = "series"
_RX_TOPOLOGY = "series"

_V_DIODE = 0.35
_V_DROP  = 2.0 * _V_DIODE

# RX-side rail constraints (post-rectifier, on storage cap)
# The downstream chain is:  rectifier → cap → LDO → 3 V load.
_V_LDO_OUT        = 3.0   # 3 V LDO output rail
_V_CAP_TARGET_DEF = 4.5   # default controller setpoint (UI-overridable)
_V_ZENER_DC       = 6.8   # passive TVS clamp — safety only
_RX_CAP_V_RATING_DEF = 100.0  # V — default C_rx rated voltage
_V_MIN_INDUCED_DC = 3.6   # hard floor: cap must be able to reach this at V_min, D=1

_TRAIN_SCRIPT = os.path.join(_NN_SCRIPTS, "train_surrogate.py")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_domain(model_dir: str) -> dict:
    path = os.path.join(model_dir, "domain.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


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
# Vectorised physics evaluation (GPU broadcast — geometry × cap pair)
# ─────────────────────────────────────────────────────────────────────────────

def _score_torch(L_tx_uH, L_rx_uH, M_uH, R_tx_nn, R_rx_nn, DCR_tx, DCR_rx,
                 C_tx, C_rx, omega,
                 f_drive_min_hz,
                 V_min, V_max, P_target_w, D_min, D_max,
                 zvs_margin, V_min_induced_dc, V_zener_dc,
                 V_cap_target_dc, V_ldo_out, V_rx_cap_rating, torch):
    """
    GPU port of the controller-model scoring physics.

    Geometry-only quantities are (bs_v,); cap-pair-dependent quantities
    broadcast to (bs_v, n_pairs).

    Returns:
      score_2d           : (bs_v, n_pairs)  ranking metric (eta × soft penalties)
      eta_2d             : (bs_v, n_pairs)  true physical efficiency (displayed)
      feasible_2d        : (bs_v, n_pairs)  strict: hard gates + D_min preference
      feasible_relaxed_2d: (bs_v, n_pairs)  relaxed: hard gates only (D_min dropped)
      D_vmax             : (bs_v,)
      D_vmin             : (bs_v,)
      V_dc_min           : (bs_v,)  zener-clamped natural V_dc at V_min
      f0_tx_2d           : (bs_v, n_pairs)
      f0_rx_2d           : (bs_v, n_pairs)
      I_tx_pk_vmax       : (bs_v,)  TX peak current at V_max (EMI proxy)
      V_Crx_pk           : (bs_v,)  peak voltage across C_rx series cap
    """
    pi = math.pi

    # ── Geometry-only: link parameters ───────────────────────────────────────
    L_tx = torch.clamp(L_tx_uH * 1e-6, min=1e-9)
    L_rx = torch.clamp(L_rx_uH * 1e-6, min=1e-9)
    M    = torch.clamp(M_uH    * 1e-6, min=0.0)

    R_tx_ac = DCR_tx + torch.clamp(R_tx_nn - DCR_tx, min=0.0)
    R_rx_ac = DCR_rx + torch.clamp(R_rx_nn - DCR_rx, min=0.0)

    Q_tx = omega * L_tx / torch.clamp(R_tx_ac, min=1e-12)
    Q_rx = omega * L_rx / torch.clamp(R_rx_ac, min=1e-12)
    k    = torch.clamp(M / torch.sqrt(L_tx * L_rx), 0.0, 1.0)
    U    = k * torch.sqrt(torch.clamp(Q_tx * Q_rx, min=0.0))

    sq        = torch.sqrt(1.0 + U * U)
    eta_link  = U * U / torch.clamp((1.0 + sq) ** 2, min=1e-18)
    Z_tx_opt  = torch.clamp(R_tx_ac * sq, min=1e-12)
    sq_p1_inv = 1.0 / torch.clamp(1.0 + sq, min=1e-12)
    R_ratio   = torch.sqrt(torch.clamp(R_rx_ac / torch.clamp(R_tx_ac, min=1e-12), min=0.0))

    V_rms_min = V_min * (math.sqrt(2.0) / pi)
    V_rms_max = V_max * (math.sqrt(2.0) / pi)

    P_rx_min_m = eta_link * (V_rms_min ** 2) / Z_tx_opt   # (bs_v,)
    P_rx_max_m = eta_link * (V_rms_max ** 2) / Z_tx_opt

    V_pk_min_m = U * V_rms_min * R_ratio * sq_p1_inv * math.sqrt(2.0)  # (bs_v,)
    V_pk_max_m = U * V_rms_max * R_ratio * sq_p1_inv * math.sqrt(2.0)

    V_dc_min_m = torch.clamp(V_pk_min_m - _V_DROP, min=0.0)  # (bs_v,)

    eta_rect_min = torch.where(V_pk_min_m > _V_DROP,
                               V_dc_min_m / torch.clamp(V_pk_min_m, min=1e-12),
                               torch.zeros_like(V_pk_min_m))

    P_dc_min = P_rx_min_m * eta_rect_min  # (bs_v,)

    # ── Controller model: cap regulated to V_cap_target ──────────────────────
    V_cap_steady = torch.minimum(V_dc_min_m,
                                 torch.full_like(V_dc_min_m, V_cap_target_dc))

    throttled   = V_dc_min_m > V_cap_target_dc
    V_pk_steady = torch.where(throttled, V_cap_steady + _V_DROP, V_pk_min_m)
    eta_rect_ss = torch.where(V_pk_steady > _V_DROP,
                              V_cap_steady / torch.clamp(V_pk_steady, min=1e-12),
                              torch.zeros_like(V_pk_steady))

    eta_ldo = torch.clamp(V_ldo_out / torch.clamp(V_cap_steady, min=1e-12),
                          max=1.0)

    P_cap_required = P_target_w * V_cap_target_dc / V_ldo_out  # scalar

    # ── Power at V_max, full drive (natural, no detuning) ────────────────────
    zeros_v = torch.zeros_like(V_pk_max_m)
    P_dc_max_natural = P_rx_max_m * torch.where(
        V_pk_max_m > _V_DROP,
        (V_pk_max_m - _V_DROP) / torch.clamp(V_pk_max_m, min=1e-12),
        zeros_v)

    inf_t = torch.full_like(P_dc_min, float("inf"))
    D_vmax = torch.where(P_dc_max_natural > 1e-18,
                         torch.full_like(P_dc_min, P_cap_required) / P_dc_max_natural,
                         inf_t)
    D_vmin = torch.where(P_dc_min > 1e-18,
                         torch.full_like(P_dc_min, P_cap_required) / P_dc_min,
                         inf_t)

    V_dc_min_eff = torch.minimum(V_dc_min_m,
                                 torch.full_like(V_dc_min_m, V_zener_dc))

    # ── Hard gate checks (geometry-only) ─────────────────────────────────────
    pmin_ok  = P_dc_min >= P_cap_required
    if D_min > 0.0:
        dmin_ok = (D_vmax >= D_min) & (D_vmax <= 1.0)
    else:
        dmin_ok = torch.ones_like(pmin_ok)
    if D_max < 1.0:
        dmax_ok = D_vmin <= D_max
    else:
        dmax_ok = torch.ones_like(pmin_ok)
    vfloor_ok = V_dc_min_eff >= V_min_induced_dc

    # V_Crx_pk: peak voltage across series resonant C_rx at matched load
    V_Crx_pk  = V_pk_min_m * Q_rx
    vrxcap_ok = V_Crx_pk <= (V_rx_cap_rating * 0.8)

    # I_tx_pk at V_max (EMI proxy)
    I_tx_pk_vmax = V_rms_max * math.sqrt(2.0) / Z_tx_opt

    # ── Soft scoring penalties (geometry-only) ────────────────────────────────
    if D_min > 0.0:
        deficit       = torch.clamp((D_min - torch.minimum(D_vmax, torch.ones_like(D_vmax))) / D_min,
                                    0.0, 1.0)
        dmin_softness = 1.0 - 0.6 * deficit
    else:
        dmin_softness = torch.ones_like(D_vmax)

    I_ref_emi    = 0.5
    emi_softness = 0.7 + 0.3 * torch.exp(-I_tx_pk_vmax / I_ref_emi)

    reserve_softness = torch.clamp(
        V_dc_min_m / max(V_cap_target_dc, 1e-12), 0.0, 1.0) ** 2

    soft_factor = dmin_softness * emi_softness * reserve_softness  # (bs_v,)

    # ── Cap-pair dependent (broadcast to (bs_v, n_pairs)) ────────────────────
    L_tx_2 = L_tx.unsqueeze(1)
    L_rx_2 = L_rx.unsqueeze(1)
    C_tx_2 = C_tx.unsqueeze(0)
    C_rx_2 = C_rx.unsqueeze(0)
    f0_tx_2d = 1.0 / (2.0 * pi * torch.sqrt(L_tx_2 * C_tx_2))
    f0_rx_2d = 1.0 / (2.0 * pi * torch.sqrt(L_rx_2 * C_rx_2))

    zvs_ok = f_drive_min_hz >= f0_tx_2d * (1.0 + zvs_margin)

    f0_rx_safe = torch.clamp(f0_rx_2d, min=1.0)
    x_rx       = f_drive_min_hz / f0_rx_safe - f0_rx_safe / f_drive_min_hz
    detune_rx  = 1.0 / (1.0 + (Q_rx.unsqueeze(1) * x_rx) ** 2)

    # True physical efficiency: TX bridge → air gap → rectifier → cap → LDO → load
    eta_2d = ((eta_link * eta_rect_ss * eta_ldo).unsqueeze(1) * detune_rx)

    # Ranking score = efficiency × soft penalties
    score_2d = eta_2d * soft_factor.unsqueeze(1)

    # Feasibility masks
    base_ok_geom = (pmin_ok & dmax_ok & vfloor_ok & vrxcap_ok
                    & (R_tx_ac > 0) & (R_rx_ac > 0) & (M > 0))
    base_ok_2d          = base_ok_geom.unsqueeze(1) & zvs_ok
    feasible_relaxed_2d = base_ok_2d                          # D_min soft penalty only
    feasible_2d         = base_ok_2d & dmin_ok.unsqueeze(1)   # strict: D_min as hard gate

    return (score_2d, eta_2d,
            feasible_2d, feasible_relaxed_2d,
            D_vmax, D_vmin, V_dc_min_eff,
            f0_tx_2d, f0_rx_2d,
            I_tx_pk_vmax, V_Crx_pk)


# ─────────────────────────────────────────────────────────────────────────────
# Single-iteration NN sweep → returns sorted list of dicts (best first)
# ─────────────────────────────────────────────────────────────────────────────

def _snap(x, step):
    """Snap continuous values to a physical grid (vectorised)."""
    return np.round(x / step) * step


def _dcr_ohm(length_m, width_mm, h_total_m):
    """DC resistance of a planar conductor of given length, width, total height."""
    return _RHO_30C * length_m / np.maximum(width_mm * 1e-3 * h_total_m, 1e-15)


def _rx_h_total_for_topology(topology: str, h_per_layer_m: list) -> float:
    """
    Effective copper-thickness factor for a per-topology RX DCR estimate.

    Mirrors the original constants:
      parallel            : sum of h
      series              : 1 / sum(1/h)
      parallel_pairs_ser  : 1 / (1/(h0+h1) + 1/(h2+h3))    (two parallel pairs in series)
    Inactive layers (h == 0) are excluded.
    """
    h_active = [h for h in h_per_layer_m if h > 0]
    if not h_active:
        return 1e-9
    if topology == "parallel":
        return float(sum(h_active))
    if topology == "series":
        return 1.0 / sum(1.0 / h for h in h_active)
    # parallel_pairs_ser: requires 4 active layers grouped (0,1) || (2,3)
    if len(h_per_layer_m) == 4 and all(h > 0 for h in h_per_layer_m):
        a = h_per_layer_m[0] + h_per_layer_m[1]
        b = h_per_layer_m[2] + h_per_layer_m[3]
        return (a * b) / (a + b)
    # Fallback to parallel.
    return float(sum(h_active))


def _nn_sweep(params, log_cb, log_cb_ow, cancel_flag):
    """
    GPU-resident sweep:
      • Geometry sampling stays on CPU (numpy — fast at this size).
      • NN inference + scaling + feasibility scoring all run on GPU.
      • Per-batch torch.topk merges into a running K-buffer, so total memory
        stays O(K) instead of O(N) — no end-of-sweep concatenate over millions.
      • AMP autocast (fp16 forward, fp32 master) on CUDA.

    All sampled geometry / frequency values are snapped to the physical grid
    before NN evaluation so the surrogate, the FastHenry refinement step,
    and the saved refined entries all agree on the same realisable values.

    Returns a dict of numpy arrays in the same schema downstream code expects.
    """
    model_dir = params["model_dir"]
    model, x_scaler, y_scaler, feat_cols, torch, device = _load_nn(model_dir)

    if not feat_cols:
        raise RuntimeError(
            "Surrogate scaler has no feature_names_in_; retrain with the new "
            "trainer so the inference path can match the training schema.")

    # ── Pull params ───────────────────────────────────────────────────────
    tx_od_max       = params["tx_od_max"]
    tx_id_min       = params["tx_id_min"]
    tx_width_min    = params["tx_width_min"]
    tx_width_max    = params["tx_width_max"]
    tx_turns_min    = int(params["tx_turns_min"])
    tx_turns_max    = int(params.get("tx_turns_max", 99))
    tx_l2_turns_min = int(params.get("tx_l2_turns_min", 1))
    tx_l2_turns_max = int(params.get("tx_l2_turns_max", tx_turns_max))

    rx_od_max    = params["rx_od_max"]
    rx_id_min    = params["rx_id_min"]
    rx_width_min = params["rx_width_min"]
    rx_width_max = params["rx_width_max"]
    rx_turns_min = int(params["rx_turns_min"])
    rx_turns_max = int(params.get("rx_turns_max", 99))

    rx_inner_turns_min = int(params.get("rx_inner_turns_min", 1))
    rx_inner_turns_max = int(params.get("rx_inner_turns_max", rx_turns_max))

    gc_enabled = params["gc_enabled"]
    gc_dia_mm  = round(_snap(params["gc_dia_mm"], _GRID_OD_MM_STEP), 4)
    gc_val     = float(gc_dia_mm if gc_enabled else 0.0)

    freq_min_hz = params["freq_min_hz"]
    freq_max_hz = params["freq_max_hz"]
    freq_ref_hz = round(_snap(0.5 * (freq_min_hz + freq_max_hz),
                              _GRID_FREQ_HZ_STEP), 1)

    V_min           = params["v_min"]
    V_max           = params["v_max"]
    P_target        = params["p_target_w"]
    D_min           = params["d_min"]
    D_max           = params.get("d_max", 1.0)
    V_cap_target    = params.get("v_cap_target_dc", _V_CAP_TARGET_DEF)
    V_ldo_out       = params.get("v_ldo_out", _V_LDO_OUT)
    V_rx_cap_rating = params.get("v_rx_cap_rating", _RX_CAP_V_RATING_DEF)

    pcb_gap_mm   = float(params["pcb_gap_mm"])
    tx_oz_layers = list(params["tx_oz_per_layer"])
    rx_oz_layers = list(params["rx_oz_per_layer"])

    domain = params.get("domain", {}) or {}
    tx_dom = domain.get("tx", {})
    rx_dom = domain.get("rx", {})
    def _port_choices(side_dom):
        opts = []
        if side_dom.get("port_outside_allowed", True): opts.append(False)
        if side_dom.get("port_inside_allowed",  False): opts.append(True)
        return opts or [False]
    tx_port_choices = _port_choices(tx_dom)
    rx_port_choices = _port_choices(rx_dom)

    tx_h_layers_m = [oz * _OZ_MM * 1e-3 for oz in tx_oz_layers]
    rx_h_layers_m = [oz * _OZ_MM * 1e-3 for oz in rx_oz_layers]
    tx_h_total_m  = sum(h for h in tx_h_layers_m if h > 0) or 1e-9
    # RX is fixed to 4-layer series — equivalent to series-stack DCR.
    rx_h_eff_m    = _rx_h_total_for_topology(_RX_TOPOLOGY, rx_h_layers_m)

    N              = int(params["n_combos"])
    dither_amp_hz  = params.get("dither_amp_hz", 0.0)
    zvs_margin     = params.get("zvs_margin", 0.0)
    c_tx_options_f = params.get("c_tx_options_f", [])
    c_rx_options_f = params.get("c_rx_options_f", [])

    if not c_tx_options_f or not c_rx_options_f:
        raise ValueError("Cap option lists must be non-empty for stratified sampling.")

    ctx_arr      = np.array(c_tx_options_f, dtype=np.float32)
    crx_arr      = np.array(c_rx_options_f, dtype=np.float32)
    n_caps_tx    = ctx_arr.size
    n_caps_rx    = crx_arr.size
    n_pairs      = n_caps_tx * n_caps_rx
    ctx_pair_arr = np.tile(ctx_arr,   n_caps_rx)
    crx_pair_arr = np.repeat(crx_arr, n_caps_tx)

    f_drive_min_hz = float(max(freq_min_hz - dither_amp_hz, 1.0))
    f_drive_max_hz = float(freq_max_hz + dither_amp_hz)
    omega          = 2.0 * math.pi * float(freq_ref_hz)

    if params.get("log_features", True):
        log_cb(f"  NN features ({len(feat_cols)}): {feat_cols}")
        log_cb(f"  Fixed: pcb_gap={pcb_gap_mm:.2f}mm  "
               f"TX oz={tx_oz_layers}  RX oz={rx_oz_layers}")
        log_cb(f"  Device: {device}")
    log_cb(f"  Topology: TX={_TX_TOPOLOGY} (fixed)  |  RX={_RX_TOPOLOGY} 4-layer (fixed)")
    log_cb(f"  Cap pairs per geometry: {n_caps_tx} TX × {n_caps_rx} RX = {n_pairs}")
    log_cb(f"  Drive sweep: {f_drive_min_hz/1e3:.1f}–{f_drive_max_hz/1e3:.1f} kHz "
           f"(ZVS margin {zvs_margin*100:.0f}%)")

    # ── GPU scaler tensors (replace sklearn CPU transform) ────────────────
    x_mean_g  = torch.as_tensor(x_scaler.mean_,  dtype=torch.float32, device=device)
    x_scale_g = torch.as_tensor(x_scaler.scale_, dtype=torch.float32, device=device)
    y_mean_g  = torch.as_tensor(y_scaler.mean_,  dtype=torch.float32, device=device)
    y_scale_g = torch.as_tensor(y_scaler.scale_, dtype=torch.float32, device=device)

    # ── Feature schema: only freq_hz is per-run constant; everything else
    # the trainer treated as fixed has been dropped from the input columns.
    const_vals = {
        "freq_hz": float(freq_ref_hz),
    }

    var_cols = {
        "tx_turns", "tx_l2_turns", "tx_width", "tx_od_mm",
        "rx_turns", "rx_inner_turns", "rx_width", "rx_od_mm",
    }

    covered = set(const_vals) | var_cols
    missing = [c for c in feat_cols if c not in covered]
    if missing:
        raise RuntimeError(
            "Trained scaler expects feature columns the optimiser does not "
            f"produce: {missing}. Retrain the model with the new trainer "
            "or extend the const/var lists explicitly.")

    col_idx = {c: feat_cols.index(c) for c in feat_cols}
    n_feat  = len(feat_cols)

    idx_tx_turns       = col_idx["tx_turns"]
    idx_tx_width       = col_idx["tx_width"]
    idx_tx_od          = col_idx["tx_od_mm"]
    idx_tx_l2_turns    = col_idx["tx_l2_turns"]
    idx_rx_turns       = col_idx["rx_turns"]
    idx_rx_inner_turns = col_idx["rx_inner_turns"]
    idx_rx_width       = col_idx["rx_width"]
    idx_rx_od          = col_idx["rx_od_mm"]

    geom_per_batch = max(1, BATCH_SIZE // n_pairs)
    X_buf = np.zeros((geom_per_batch, n_feat), dtype=np.float32)
    for c, val in const_vals.items():
        X_buf[:, col_idx[c]] = val

    # ── GPU cap-pair tensors ──────────────────────────────────────────────
    ctx_g = torch.as_tensor(ctx_pair_arr, dtype=torch.float32, device=device)
    crx_g = torch.as_tensor(crx_pair_arr, dtype=torch.float32, device=device)

    # ── Running top-K buffer (GPU) ────────────────────────────────────────
    # Over-keep by ~200× the user-requested top-K so post-dedup we have
    # plenty of unique winners; K=5000 floor keeps memory trivial (<1 MB).
    K_keep = max(int(params["top_k"]) * 200, 5000)
    inf_neg = float("-inf")

    def _kbuf(dtype=torch.float32):
        return torch.zeros(K_keep, dtype=dtype, device=device)

    keep_score = torch.full((K_keep,), inf_neg, dtype=torch.float32, device=device)
    keep = {
        "tx_turns":       _kbuf(),
        "tx_l2_turns":    _kbuf(),
        "tx_width":       _kbuf(),
        "tx_od_mm":       _kbuf(),
        "rx_turns":       _kbuf(),
        "rx_inner_turns": _kbuf(),
        "rx_width":       _kbuf(),
        "rx_od_mm":       _kbuf(),
        "tx_port_inside": _kbuf(),
        "rx_port_inside": _kbuf(),
        "gc_dia_mm":      _kbuf(),
        "L_tx":           _kbuf(),
        "L_rx":           _kbuf(),
        "M":              _kbuf(),
        "R_tx":           _kbuf(),
        "R_rx":           _kbuf(),
        "DCR_tx":         _kbuf(),
        "DCR_rx":         _kbuf(),
        "tx_id_mm":       _kbuf(),
        "rx_id_mm":       _kbuf(),
        "eta_sys":        _kbuf(),
        "D_vmax":         _kbuf(),
        "D_vmin":         _kbuf(),
        "V_dc_min":       _kbuf(),
        "I_tx_pk_vmax":   _kbuf(),
        "V_Crx_pk":       _kbuf(),
        "f0_tx_hz":       _kbuf(),
        "f0_rx_hz":       _kbuf(),
        "f_op_lo_hz":     _kbuf(),
        "c_tx_nf":        _kbuf(),
        "c_rx_nf":        _kbuf(),
    }

    n_feasible_counter = torch.zeros(1, dtype=torch.int64, device=device)
    n_strict_counter   = torch.zeros(1, dtype=torch.int64, device=device)

    use_amp = (device.type == "cuda")
    def _amp_ctx():
        return (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp else contextlib.nullcontext())

    n_ok = 0
    n_rej = 0
    b = 0
    rng = np.random.default_rng(params.get("rng_seed", 42))
    log_cb(f"    batch 0: 0/{N:,} combos, 0 rejected")  # seed line to overwrite

    while n_ok < N:
        if cancel_flag.is_set():
            return None
        b += 1
        if b % 5 == 0:
            log_cb_ow(f"    batch {b}: {n_ok:,}/{N:,} combos, {n_rej:,} rejected")

        bs = geom_per_batch

        # ── Sample geometry on CPU (numpy is plenty fast here) ───────────
        tx_od_s    = _snap(rng.uniform(tx_id_min + 2.0, tx_od_max, size=bs),
                           _GRID_OD_MM_STEP).astype(np.float32)
        tx_width_s = _snap(rng.uniform(tx_width_min, tx_width_max, size=bs),
                           _GRID_WIDTH_MM_STEP).astype(np.float32)
        tx_t_max   = np.maximum(
            np.floor((tx_od_s + _SPACING_MM - tx_width_s - tx_id_min)
                     / (2.0 * (tx_width_s + _SPACING_MM))).astype(np.int32),
            tx_turns_min)
        tx_t_min   = np.minimum(np.full(bs, tx_turns_min, dtype=np.int32), tx_t_max)
        tx_turns_s = rng.integers(tx_t_min, tx_t_max + 1).astype(np.float32)

        # tx_l2_turns ∈ [max(_MIN_L2, ceil(tx_turns/2)), min(tx_turns, l2_max)]
        # Mirrors the LHS generator's feasibility constraint so the surrogate
        # sees the same input distribution at inference time.
        _l2_floor_from_l1 = ((tx_turns_s.astype(np.int32) + 1) // 2)
        l2_lo_arr = np.maximum(tx_l2_turns_min, _l2_floor_from_l1).astype(np.int32)
        l2_hi_arr = np.minimum(tx_turns_s.astype(np.int32),
                               np.full(bs, tx_l2_turns_max, dtype=np.int32))
        l2_hi_arr = np.maximum(l2_hi_arr, l2_lo_arr)
        tx_l2_turns_s = rng.integers(l2_lo_arr, l2_hi_arr + 1).astype(np.float32)

        rx_od_lo_s = np.full(bs, rx_id_min + 2.0, dtype=np.float32)
        u_rx       = rng.random(size=bs).astype(np.float32)
        rx_od_s    = _snap(rx_od_lo_s + u_rx * np.maximum(rx_od_max - rx_od_lo_s, 0.0),
                           _GRID_OD_MM_STEP).astype(np.float32)
        rx_width_s = _snap(rng.uniform(rx_width_min, rx_width_max, size=bs),
                           _GRID_WIDTH_MM_STEP).astype(np.float32)
        rx_t_max   = np.maximum(
            np.floor((rx_od_s + _SPACING_MM - rx_width_s - rx_id_min)
                     / (2.0 * (rx_width_s + _SPACING_MM))).astype(np.int32),
            rx_turns_min)
        rx_t_min   = np.minimum(np.full(bs, rx_turns_min, dtype=np.int32), rx_t_max)
        rx_turns_s = rng.integers(rx_t_min, rx_t_max + 1).astype(np.float32)

        # rx_inner_turns ∈ [max(1, inner_min), min(rx_turns, inner_max)] —
        # mirrors the LHS generator and feasibility check (endpoint-match
        # width requires inner_turns <= outer_turns).
        _rx_outer_int = rx_turns_s.astype(np.int32)
        rx_inner_lo_arr = np.maximum(rx_inner_turns_min,
                                     np.ones(bs, dtype=np.int32))
        rx_inner_hi_arr = np.minimum(_rx_outer_int,
                                     np.full(bs, rx_inner_turns_max, dtype=np.int32))
        rx_inner_hi_arr = np.maximum(rx_inner_hi_arr, rx_inner_lo_arr)
        rx_inner_turns_s = rng.integers(rx_inner_lo_arr,
                                        rx_inner_hi_arr + 1).astype(np.float32)

        tx_port_idx_s = rng.integers(0, len(tx_port_choices), size=bs)
        rx_port_idx_s = rng.integers(0, len(rx_port_choices), size=bs)

        tx_id_s = _inner_diameter_mm(tx_od_s, tx_width_s, tx_turns_s)
        rx_id_s = _inner_diameter_mm(rx_od_s, rx_width_s, rx_turns_s)
        id_ok   = (tx_id_s >= tx_id_min) & (rx_id_s >= rx_id_min)
        n_rej  += int((~id_ok).sum()) * n_pairs
        if not id_ok.any():
            continue

        tx_turns_s = tx_turns_s[id_ok]; tx_od_s    = tx_od_s[id_ok]
        tx_width_s = tx_width_s[id_ok]; rx_od_s    = rx_od_s[id_ok]
        rx_turns_s = rx_turns_s[id_ok]; rx_width_s = rx_width_s[id_ok]
        tx_l2_turns_s    = tx_l2_turns_s[id_ok]
        rx_inner_turns_s = rx_inner_turns_s[id_ok]
        tx_port_idx_s = tx_port_idx_s[id_ok]; rx_port_idx_s = rx_port_idx_s[id_ok]
        tx_id_s    = tx_id_s[id_ok];    rx_id_s    = rx_id_s[id_ok]
        bs_v = int(id_ok.sum())

        tx_port_in_s = np.asarray(tx_port_choices, dtype=np.float32)[tx_port_idx_s]
        rx_port_in_s = np.asarray(rx_port_choices, dtype=np.float32)[rx_port_idx_s]

        # ── DCR (TX lumped active-layer; RX fixed 4-layer series) ────────
        # The RX series-stack DCR uses the per-layer width on each layer.
        # Outer layers carry rx_width; inner layers carry the auto-computed
        # endpoint-match width. For the surrogate DCR estimate we
        # approximate the series sum using the *outer* width — same model
        # the previous code applied (length × width × h_eff). The
        # endpoint-match inner width differs by a small factor that the
        # NN itself absorbs through R_rx_ac.
        tx_len_m  = _spiral_length_m(tx_od_s, tx_width_s, tx_turns_s)
        DCR_tx_np = _dcr_ohm(tx_len_m, tx_width_s, tx_h_total_m).astype(np.float32)
        rx_len_m  = _spiral_length_m(rx_od_s, rx_width_s, rx_turns_s)
        DCR_rx_np = _dcr_ohm(rx_len_m, rx_width_s, rx_h_eff_m).astype(np.float32)

        # ── Write variable cols into X_buf (constants already filled) ────
        X = X_buf[:bs_v]
        X[:, idx_tx_turns]       = tx_turns_s
        X[:, idx_tx_width]       = tx_width_s
        X[:, idx_tx_od]          = tx_od_s
        X[:, idx_rx_turns]       = rx_turns_s
        X[:, idx_rx_inner_turns] = rx_inner_turns_s
        X[:, idx_rx_width]       = rx_width_s
        X[:, idx_rx_od]          = rx_od_s
        X[:, idx_tx_l2_turns]    = tx_l2_turns_s

        # ── Inference: scale + forward + unscale, all on GPU ─────────────
        X_t  = torch.from_numpy(X).to(device, non_blocking=True)
        X_sc = (X_t - x_mean_g) / x_scale_g
        with torch.no_grad(), _amp_ctx():
            Y_sc = model(X_sc)
        Y = Y_sc.float() * y_scale_g + y_mean_g     # (bs_v, 5)

        L_tx_g    = Y[:, 0]
        L_rx_g    = Y[:, 1]
        M_g       = Y[:, 2]
        R_tx_nn_g = Y[:, 3]
        R_rx_nn_g = Y[:, 4]
        DCR_tx_g  = torch.from_numpy(DCR_tx_np).to(device, non_blocking=True)
        DCR_rx_g  = torch.from_numpy(DCR_rx_np).to(device, non_blocking=True)

        # ── Score (broadcast geometry × cap pair on GPU) ─────────────────
        (score_2d, eta_2d,
         feas_2d, feas_rel_2d,
         D_vmax_g, D_vmin_g, V_dc_min_g,
         f0_tx_2d, f0_rx_2d,
         I_tx_pk_g, V_Crx_pk_g) = _score_torch(
            L_tx_g, L_rx_g, M_g, R_tx_nn_g, R_rx_nn_g, DCR_tx_g, DCR_rx_g,
            ctx_g, crx_g, omega,
            f_drive_min_hz,
            V_min, V_max, P_target, D_min, D_max,
            zvs_margin, _V_MIN_INDUCED_DC, _V_ZENER_DC,
            V_cap_target, V_ldo_out, V_rx_cap_rating, torch)

        n_strict_counter   += feas_2d.sum()
        n_feasible_counter += feas_rel_2d.sum()

        # Mask relaxed-infeasible to -inf; D_min preference is already a soft
        # penalty inside score_2d so strictly-infeasible designs still rank low.
        inf_t = torch.tensor(inf_neg, dtype=score_2d.dtype, device=device)
        score_2d = torch.where(feas_rel_2d, score_2d, inf_t)
        eta_2d   = torch.where(feas_rel_2d, eta_2d,   inf_t)

        score_flat = score_2d.reshape(-1)   # (bs_v * n_pairs,)
        eta_flat   = eta_2d.reshape(-1)
        n_ok += bs_v * n_pairs

        # ── Merge with running top-K ─────────────────────────────────────
        combined = torch.cat([keep_score, score_flat])
        K_take   = min(K_keep, combined.numel())
        top_vals, top_idx = torch.topk(combined, K_take)

        from_old = top_idx < K_keep
        from_new = ~from_old
        old_sel  = top_idx[from_old]                  # idx into keep buffers
        new_sel  = top_idx[from_new] - K_keep         # idx into score_flat

        geom_idx = new_sel // n_pairs
        cap_idx  = new_sel %  n_pairs

        # Upload per-geometry arrays once (lazy — only when there are winners)
        tx_turns_t       = torch.from_numpy(tx_turns_s).to(device, non_blocking=True)
        tx_l2_turns_t    = torch.from_numpy(tx_l2_turns_s).to(device, non_blocking=True)
        tx_width_t       = torch.from_numpy(tx_width_s).to(device, non_blocking=True)
        tx_od_t          = torch.from_numpy(tx_od_s).to(device, non_blocking=True)
        rx_turns_t       = torch.from_numpy(rx_turns_s).to(device, non_blocking=True)
        rx_inner_turns_t = torch.from_numpy(rx_inner_turns_s).to(device, non_blocking=True)
        rx_width_t       = torch.from_numpy(rx_width_s).to(device, non_blocking=True)
        rx_od_t          = torch.from_numpy(rx_od_s).to(device, non_blocking=True)
        tx_port_t        = torch.from_numpy(tx_port_in_s).to(device, non_blocking=True)
        rx_port_t        = torch.from_numpy(rx_port_in_s).to(device, non_blocking=True)
        tx_id_t          = torch.from_numpy(tx_id_s).to(device, non_blocking=True)
        rx_id_t          = torch.from_numpy(rx_id_s).to(device, non_blocking=True)

        f0_tx_flat_t = f0_tx_2d.reshape(-1)
        f0_rx_flat_t = f0_rx_2d.reshape(-1)

        def _merge(field, new_vals):
            out = torch.empty(K_take, dtype=field.dtype, device=device)
            out[from_old] = field[old_sel]
            out[from_new] = new_vals.to(field.dtype)
            return out

        keep_score = top_vals
        keep = {
            "tx_turns":       _merge(keep["tx_turns"],       tx_turns_t[geom_idx]),
            "tx_l2_turns":    _merge(keep["tx_l2_turns"],    tx_l2_turns_t[geom_idx]),
            "tx_width":       _merge(keep["tx_width"],       tx_width_t[geom_idx]),
            "tx_od_mm":       _merge(keep["tx_od_mm"],       tx_od_t[geom_idx]),
            "rx_turns":       _merge(keep["rx_turns"],       rx_turns_t[geom_idx]),
            "rx_inner_turns": _merge(keep["rx_inner_turns"], rx_inner_turns_t[geom_idx]),
            "rx_width":       _merge(keep["rx_width"],       rx_width_t[geom_idx]),
            "rx_od_mm":       _merge(keep["rx_od_mm"],       rx_od_t[geom_idx]),
            "tx_port_inside": _merge(keep["tx_port_inside"], tx_port_t[geom_idx]),
            "rx_port_inside": _merge(keep["rx_port_inside"], rx_port_t[geom_idx]),
            "tx_id_mm":       _merge(keep["tx_id_mm"],       tx_id_t[geom_idx]),
            "rx_id_mm":       _merge(keep["rx_id_mm"],       rx_id_t[geom_idx]),
            "L_tx":           _merge(keep["L_tx"],           L_tx_g[geom_idx]),
            "L_rx":           _merge(keep["L_rx"],           L_rx_g[geom_idx]),
            "M":              _merge(keep["M"],              M_g[geom_idx]),
            "R_tx":           _merge(keep["R_tx"],           R_tx_nn_g[geom_idx]),
            "R_rx":           _merge(keep["R_rx"],           R_rx_nn_g[geom_idx]),
            "DCR_tx":         _merge(keep["DCR_tx"],         DCR_tx_g[geom_idx]),
            "DCR_rx":         _merge(keep["DCR_rx"],         DCR_rx_g[geom_idx]),
            "eta_sys":        _merge(keep["eta_sys"],        eta_flat[new_sel]),
            "D_vmax":         _merge(keep["D_vmax"],         D_vmax_g[geom_idx]),
            "D_vmin":         _merge(keep["D_vmin"],         D_vmin_g[geom_idx]),
            "V_dc_min":       _merge(keep["V_dc_min"],       V_dc_min_g[geom_idx]),
            "I_tx_pk_vmax":   _merge(keep["I_tx_pk_vmax"],  I_tx_pk_g[geom_idx]),
            "V_Crx_pk":       _merge(keep["V_Crx_pk"],      V_Crx_pk_g[geom_idx]),
            "f0_tx_hz":       _merge(keep["f0_tx_hz"],       f0_tx_flat_t[new_sel]),
            "f0_rx_hz":       _merge(keep["f0_rx_hz"],       f0_rx_flat_t[new_sel]),
            "c_tx_nf":        _merge(keep["c_tx_nf"],        ctx_g[cap_idx] * 1e9),
            "c_rx_nf":        _merge(keep["c_rx_nf"],        crx_g[cap_idx] * 1e9),
            # gc_dia_mm + f_op_lo_hz are run-constants — just refill.
            "gc_dia_mm":      torch.full((K_take,), gc_val, dtype=torch.float32, device=device),
            "f_op_lo_hz":     torch.full((K_take,), f_drive_min_hz,
                                         dtype=torch.float32, device=device),
        }

    # ── Gather final result to CPU ───────────────────────────────────────
    def _to_np(t): return t.detach().cpu().numpy()

    score_np = _to_np(keep_score)
    score_np = np.where(score_np > inf_neg, score_np, np.float32(0.0)).astype(np.float32)

    concat = {
        "tx_turns":       _to_np(keep["tx_turns"]),
        "tx_l2_turns":    _to_np(keep["tx_l2_turns"]),
        "tx_width":       _to_np(keep["tx_width"]),
        "tx_od_mm":       _to_np(keep["tx_od_mm"]),
        "rx_turns":       _to_np(keep["rx_turns"]),
        "rx_inner_turns": _to_np(keep["rx_inner_turns"]),
        "rx_width":       _to_np(keep["rx_width"]),
        "rx_od_mm":       _to_np(keep["rx_od_mm"]),
        "tx_port_inside": _to_np(keep["tx_port_inside"]),
        "rx_port_inside": _to_np(keep["rx_port_inside"]),
        "gc_dia_mm":      _to_np(keep["gc_dia_mm"]),
        "L_tx":           _to_np(keep["L_tx"]),
        "L_rx":           _to_np(keep["L_rx"]),
        "M":              _to_np(keep["M"]),
        "R_tx":           _to_np(keep["R_tx"]),
        "R_rx":           _to_np(keep["R_rx"]),
        "DCR_tx":         _to_np(keep["DCR_tx"]),
        "DCR_rx":         _to_np(keep["DCR_rx"]),
        "tx_id_mm":       _to_np(keep["tx_id_mm"]),
        "rx_id_mm":       _to_np(keep["rx_id_mm"]),
        "score":          score_np,
        "eta_sys":        _to_np(keep["eta_sys"]),
        "D_vmax":         _to_np(keep["D_vmax"]),
        "D_vmin":         _to_np(keep["D_vmin"]),
        "V_dc_min":       _to_np(keep["V_dc_min"]),
        "I_tx_pk_vmax":   _to_np(keep["I_tx_pk_vmax"]),
        "V_Crx_pk":       _to_np(keep["V_Crx_pk"]),
        "f0_tx_hz":       _to_np(keep["f0_tx_hz"]),
        "f0_rx_hz":       _to_np(keep["f0_rx_hz"]),
        "f_op_lo_hz":     _to_np(keep["f_op_lo_hz"]),
        "c_tx_nf":        _to_np(keep["c_tx_nf"]),
        "c_rx_nf":        _to_np(keep["c_rx_nf"]),
        "_freq_ref_hz":   np.float64(freq_ref_hz),
        "_freq_min_hz":   np.float64(freq_min_hz),
        "_freq_max_hz":   np.float64(freq_max_hz),
    }

    n_feasible = int(n_feasible_counter.item())
    n_strict   = int(n_strict_counter.item())
    relaxed_note = "" if n_strict > 0 else "  [D_min preference unmet by all designs — best-effort fallback]"
    log_cb(f"  Sweep done: {n_ok:,} combos scored "
           f"({n_feasible:,} feasible incl. {n_strict:,} strict, {n_rej:,} geometry-rejected; "
           f"top-{K_keep} retained){relaxed_note}")
    return concat


# ─────────────────────────────────────────────────────────────────────────────
# Pick top-K unique configurations from sweep result
# ─────────────────────────────────────────────────────────────────────────────

def _top_k(result, k=12):
    """
    Pick top-K unique configurations from sweep results.

    Dedup key includes port_inside booleans; topology is fixed
    (TX=series, RX=4-layer series) on NN_V8 so it's not part of the key.
    Sorted by `score` (efficiency × soft penalties) so the D_min preference
    is respected without hard-gating.
    """
    score = result["score"]
    eta   = result["eta_sys"]
    order = np.argsort(score)[::-1]

    winners = []
    seen    = set()
    for i in order:
        if score[i] <= 0.0:
            break
        tx_t  = int(result["tx_turns"][i])
        tx_od = round(float(result["tx_od_mm"][i]), 1)
        tx_w  = round(float(result["tx_width"][i]), 2)
        rx_t  = int(result["rx_turns"][i])
        rx_in = int(result["rx_inner_turns"][i])
        rx_od = round(float(result["rx_od_mm"][i]), 1)
        rx_w  = round(float(result["rx_width"][i]), 2)
        tx_port  = bool(result["tx_port_inside"][i])
        rx_port  = bool(result["rx_port_inside"][i])
        gc       = round(float(result["gc_dia_mm"][i]), 1)
        key = (tx_t, tx_od, tx_w, rx_t, rx_in, rx_od, rx_w,
               tx_port, rx_port, gc)
        if key in seen:
            continue
        seen.add(key)
        winners.append({
            "tx_turns":       tx_t,
            "tx_l2_turns":    int(result["tx_l2_turns"][i]) if "tx_l2_turns" in result else 0,
            "tx_width":       float(result["tx_width"][i]),
            "tx_od_mm":       float(result["tx_od_mm"][i]),
            "tx_topology":    _TX_TOPOLOGY,
            "tx_port_inside": tx_port,
            "rx_turns":       rx_t,
            "rx_inner_turns": rx_in,
            "rx_width":       float(result["rx_width"][i]),
            "rx_od_mm":       float(result["rx_od_mm"][i]),
            "rx_topology":    _RX_TOPOLOGY,
            "rx_port_inside": rx_port,
            "gc_dia_mm":      float(result["gc_dia_mm"][i]),
            "eta_sys":        float(eta[i]),
            "score":          float(score[i]),
            "I_tx_pk_vmax":   float(result["I_tx_pk_vmax"][i]),
            "V_Crx_pk":       float(result["V_Crx_pk"][i]),
            "L_tx_uH":        float(result["L_tx"][i]),
            "L_rx_uH":        float(result["L_rx"][i]),
            "c_tx_nf":        float(result["c_tx_nf"][i]),
            "c_rx_nf":        float(result["c_rx_nf"][i]),
            "D_vmax":         float(result["D_vmax"][i]),
            "D_vmin":         float(result["D_vmin"][i]),
            "V_dc_min":       float(result["V_dc_min"][i]),
            "f0_tx_hz":       float(result["f0_tx_hz"][i]),
            "f0_rx_hz":       float(result["f0_rx_hz"][i]),
            "f_op_lo_hz":     float(result["f_op_lo_hz"][i]),
            "M_uH":           float(result["M"][i]),
            "k":              float(result["M"][i]) / max(
                                  math.sqrt(float(result["L_tx"][i]) *
                                            float(result["L_rx"][i])), 1e-12),
        })
        if len(winners) >= k:
            break
    return winners


# ─────────────────────────────────────────────────────────────────────────────
# Build SimParams list for FastHenry batch
# ─────────────────────────────────────────────────────────────────────────────

def _build_sim_params(winners, params, timeout_sec):
    """
    Build FastHenry `SimParams` from winning combos (NN_V8 schema).

    NN_V8 hard-fixes:
      • TX: series, port_outside, L1+L2 active
      • RX: series across all 4 active layers, port_inside,
        with independent outer / inner turn counts
        (inner trace width auto-computed inside `parallel_sim`).
    """
    from parallel_sim import SimParams

    domain     = params["domain"]
    glb        = domain.get("global", {})
    freq_range = glb.get("freq_hz", [340000.0, 380000.0])
    freq_hz    = 0.5 * (float(freq_range[0]) + float(freq_range[1]))
    freq_hz    = round(_snap(freq_hz, _GRID_FREQ_HZ_STEP), 1)

    tx_d   = domain.get("tx", {})
    rx_d   = domain.get("rx", {})

    def _rng_mid(rng, fallback):
        if isinstance(rng, (list, tuple)) and len(rng) >= 2:
            return 0.5 * (float(rng[0]) + float(rng[1]))
        return float(fallback)

    tx_l1l2_gap = _rng_mid(tx_d.get("l1l2_gap_mm",
                                    tx_d.get("outer_gap_mm")), 0.2104)
    rx_outer    = _rng_mid(rx_d.get("outer_gap_mm"), 0.2104)
    rx_inner    = _rng_mid(rx_d.get("inner_gap_mm"), 0.6)

    pcb_gap_mm  = float(params["pcb_gap_mm"])

    params_list = []
    for w in winners:
        p = SimParams(
            tx_turns        = int(w["tx_turns"]),
            tx_od_mm        = round(float(w["tx_od_mm"]), 4),
            tx_w_mm         = round(float(w["tx_width"]), 4),
            tx_spacing_mm   = _SPACING_MM,
            tx_l1l2_gap_mm  = tx_l1l2_gap,
            tx_l2_turns     = int(w.get("tx_l2_turns", 0)),
            tx_nwinc        = int(tx_d.get("nwinc", 3)),

            rx_turns        = int(w["rx_turns"]),
            rx_inner_turns  = int(w.get("rx_inner_turns", 0)),
            rx_od_mm        = round(float(w["rx_od_mm"]), 4),
            rx_w_mm         = round(float(w["rx_width"]), 4),
            rx_spacing_mm   = _SPACING_MM,
            rx_outer_gap_mm = rx_outer,
            rx_inner_gap_mm = rx_inner,
            rx_nwinc        = int(rx_d.get("nwinc", 3)),

            pcb_gap_mm      = pcb_gap_mm,
            freq_hz         = freq_hz,
            resolution_mm   = float(glb.get("resolution_mm", 1.2)),
            timeout_sec     = float(timeout_sec),

            rx_ground_disc_dia_mm = round(float(w.get("gc_dia_mm", 20.0)), 1),
            tx_ground_enabled     = True,
            tag                   = "opt",
        )
        params_list.append(p)
    return params_list


# ─────────────────────────────────────────────────────────────────────────────
# Append sim results to refined_results.json
# ─────────────────────────────────────────────────────────────────────────────

def _append_to_refined(sim_results, winners, refined_path, log_cb):
    """
    Append OK FastHenry rows to refined_results.json.

    Each row passes through the simulator's full geometry / electrical echo
    plus a fresh uuid and hasGroundCircle flag. The schema is structurally
    identical to a results.json entry — same fields, same rounding, no
    duplicate {tx,rx}_trace_width_mm keys.
    """
    if os.path.exists(refined_path):
        try:
            with open(refined_path) as f:
                data = json.load(f)
        except Exception:
            data = {"meta": {}, "results": []}
    else:
        data = {"meta": {}, "results": []}

    existing_uuids = {r.get("uuid") for r in data["results"] if r.get("uuid")}
    added = 0

    _INPUT_FIELDS  = ["tx_turns", "tx_l2_turns", "tx_width", "tx_od_mm",
                      "rx_turns", "rx_inner_turns", "rx_width", "rx_od_mm",
                      "freq_hz"]
    _OUTPUT_FIELDS = ["L_tx_uH", "L_rx_uH", "M_uH", "R_tx_ac", "R_rx_ac"]

    for i, (sr, w) in enumerate(zip(sim_results, winners)):
        if sr is None or "error" in sr:
            log_cb(f"    sim {i+1}: FAIL — {(sr or {}).get('error','?')}")
            continue

        row = {k: sr[k] for k in _INPUT_FIELDS + _OUTPUT_FIELDS if k in sr}
        row["uuid"] = str(uuid.uuid4())

        if row["uuid"] not in existing_uuids:
            data["results"].append(row)
            existing_uuids.add(row["uuid"])
            added += 1
        eta_pct = w.get("eta_sys", 0.0) * 100
        log_cb(f"    sim {i+1}: OK  L_tx={sr.get('L_tx_uH',0):.2f}µH  "
               f"L_rx={sr.get('L_rx_uH',0):.2f}µH  "
               f"M={sr.get('M_uH',0):.3f}µH  k={sr.get('k',0):.4f}  "
               f"η={eta_pct:.1f}%")

    data["meta"]["n_results"] = len(data["results"])
    tmp = refined_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, refined_path)
    log_cb(f"    → {added} rows appended to refined_results.json "
           f"(total {len(data['results'])})")
    return added


# ─────────────────────────────────────────────────────────────────────────────
# Retrain the model on refined_results.json
# ─────────────────────────────────────────────────────────────────────────────

def _retrain(model_dir, refined_path, train_params, log_cb, log_cb_ow, cancel_flag):
    env = os.environ.copy()
    env["SURROGATE_DATA"]       = refined_path
    env["SURROGATE_OUTPUT_DIR"] = model_dir
    env["SURROGATE_EPOCHS"]     = str(train_params["epochs"])
    env["SURROGATE_BATCH_SIZE"] = str(train_params["batch"])
    env["SURROGATE_LR"]         = str(train_params["lr"])
    env["SURROGATE_VAL_SPLIT"]  = str(train_params["val_split"])

    log_cb("  Retraining model…")
    log_cb("    ")  # seed a blank line for epoch overwrites
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", _TRAIN_SCRIPT],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=_NN_DIR, env=env)
        for line in proc.stdout:
            if cancel_flag.is_set():
                proc.terminate()
                return False
            stripped = line.rstrip()
            # Epoch progress lines are overwritten in-place; others appended normally
            if stripped.startswith("Epoch "):
                log_cb_ow("    " + stripped)
            else:
                log_cb("    " + stripped)
        proc.wait()
        if proc.returncode != 0:
            log_cb(f"  Training exited with code {proc.returncode}")
            return False
        log_cb("  Retraining complete.")
        return True
    except Exception as e:
        log_cb(f"  Retrain error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation loop (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_optimisation(params, progress_cb, log_cb, log_cb_ow, done_cb, cancel_flag):
    try:
        model_dir     = params["model_dir"]
        max_iters     = params["max_iters"]
        top_k         = params["top_k"]
        fh_workers    = params["fh_workers"]
        fh_timeout    = params["fh_timeout"]
        train_params  = params["train_params"]

        # Step 1 — ensure refined_results.json exists (copy from results.json)
        results_path  = os.path.join(model_dir, "results.json")
        refined_path  = os.path.join(model_dir, "refined_results.json")

        if not os.path.exists(refined_path):
            if os.path.exists(results_path):
                shutil.copy2(results_path, refined_path)
                log_cb(f"Created refined_results.json from results.json "
                       f"({os.path.getsize(refined_path)//1024} kB)")
            else:
                log_cb("WARNING: results.json not found — refined_results.json will start empty.")
                with open(refined_path, "w") as f:
                    json.dump({"meta": {}, "results": []}, f)

        prev_winners  = []
        win_streak    = 0

        for iteration in range(1, max_iters + 1):
            if cancel_flag.is_set():
                log_cb("Cancelled.")
                done_cb(None)
                return

            log_cb(f"\n{'='*60}")
            log_cb(f"ITERATION {iteration}/{max_iters}")
            log_cb(f"{'='*60}")

            progress_cb((iteration - 1) / max_iters * 0.9)

            # --- NN sweep ---
            log_cb(f"Phase 1: NN sweep ({params['n_combos']:,} combinations)…")
            sweep_seed_params = dict(params)
            sweep_seed_params["rng_seed"] = 42 + iteration
            sweep_seed_params["log_features"] = (iteration == 1)
            sweep_result = _nn_sweep(sweep_seed_params, log_cb, log_cb_ow, cancel_flag)
            if sweep_result is None:
                log_cb("Cancelled during sweep.")
                done_cb(None)
                return

            # --- Pick top-K ---
            winners = _top_k(sweep_result, k=top_k)
            if not winners:
                log_cb("  No feasible combinations found — stopping.")
                break

            top1      = winners[0]
            _f0_tx    = top1.get("f0_tx_hz", 0.0)
            _f0_rx    = top1.get("f0_rx_hz", 0.0)
            _f_op_lo  = top1.get("f_op_lo_hz", 0.0)
            _zvs_x    = (_f_op_lo / _f0_tx) if _f0_tx > 0 else 0.0
            _rx_off   = ((_f0_rx - _f_op_lo) / 1e3) if (_f0_rx > 0 and _f_op_lo > 0) else 0.0
            _M_uH     = top1.get("M_uH", 0.0)
            _k        = top1.get("k", 0.0)
            log_cb(f"\nTop winner: TX {top1['tx_od_mm']:.1f}mm {top1['tx_turns']}t "
                   f"w={top1['tx_width']:.2f}  |  "
                   f"RX {top1['rx_od_mm']:.1f}mm {top1['rx_turns']}t "
                   f"w={top1['rx_width']:.2f} {top1['rx_topology']}")
            log_cb(f"  η={top1['eta_sys']*100:.1f}%  "
                   f"D@Vmax={min(top1.get('D_vmax', 1.0), 1.0)*100:.0f}%  "
                   f"D@Vmin={min(top1.get('D_vmin', 1.0), 1.0)*100:.0f}%  "
                   f"I_tx_pk@Vmax={top1.get('I_tx_pk_vmax', 0.0)*1000:.0f}mA  "
                   f"V_dc(Vmin,D=1)={top1.get('V_dc_min', 0.0):.2f}V  "
                   f"V_Crx_pk={top1.get('V_Crx_pk', 0.0):.0f}V")
            log_cb(f"  L_tx={top1.get('L_tx_uH',0):.2f}µH  "
                   f"L_rx={top1.get('L_rx_uH',0):.2f}µH  "
                   f"C_tx={top1.get('c_tx_nf',0):.1f}nF  "
                   f"C_rx={top1.get('c_rx_nf',0):.1f}nF")
            log_cb(f"  M={_M_uH:.3f}µH  k={_k:.4f}")
            log_cb(f"  f0_tx={_f0_tx/1e3:.1f}kHz  f_op_lo={_f_op_lo/1e3:.1f}kHz "
                   f"(ZVS×{_zvs_x:.2f})  |  "
                   f"f0_rx={_f0_rx/1e3:.1f}kHz  Δf_rx={_rx_off:+.1f}kHz")
            log_cb(f"  Top {len(winners)} sent to FastHenry.")

            # Check convergence: same winner 3 times in a row
            if prev_winners and _same_config(top1, prev_winners[0]):
                win_streak += 1
            else:
                win_streak = 1
            prev_winners = winners

            if win_streak >= 3:
                log_cb("\nSame winner 3 iterations in a row — converged.")
                break

            # --- FastHenry ---
            log_cb(f"\nPhase 2: FastHenry ({len(winners)} sims, {fh_workers} workers)…")
            sim_params_list = _build_sim_params(winners, params, fh_timeout)

            from parallel_sim import run_batch as _run_batch
            sim_counter = [0]
            log_cb(f"    FastHenry: 0/{len(winners)}")  # seed the line to overwrite

            def _prog(done, total, _i=iteration, _mi=max_iters):
                sim_counter[0] = done
                log_cb_ow(f"    FastHenry: {done}/{total}")
                frac = ((_i - 1) + (done / max(total, 1)) * 0.5) / _mi
                progress_cb(frac * 0.9)

            sim_results = _run_batch(sim_params_list,
                                     max_workers=fh_workers,
                                     progress_cb=_prog)

            ok_count = sum(1 for r in sim_results if r and "error" not in r)
            log_cb_ow(f"    FastHenry: {ok_count}/{len(sim_results)} OK")

            # --- Append to refined ---
            log_cb("\nPhase 3: Appending to refined_results.json…")
            _append_to_refined(sim_results, winners, refined_path, log_cb)

            # --- Retrain ---
            log_cb("\nPhase 4: Retraining model…")
            ok = _retrain(model_dir, refined_path, train_params, log_cb, log_cb_ow, cancel_flag)
            if not ok or cancel_flag.is_set():
                log_cb("Retrain failed or cancelled.")
                done_cb(None)
                return

        # --- Final sweep with the fully refined model ---
        log_cb(f"\n{'='*60}")
        log_cb("FINAL SWEEP (refined model)")
        log_cb(f"{'='*60}")
        final_seed_params = dict(params)
        final_seed_params["rng_seed"] = 42 + max_iters + 1
        final_seed_params["log_features"] = False
        final_result = _nn_sweep(final_seed_params, log_cb, log_cb_ow, cancel_flag)
        if final_result is not None and not cancel_flag.is_set():
            final_winners = _top_k(final_result, k=top_k)
            if final_winners:
                prev_winners = final_winners
                top1 = final_winners[0]
                _f_op    = top1.get("freq_hz", 0.0)
                _f0_tx   = top1.get("f0_tx_hz", 0.0)
                _f0_rx   = top1.get("f0_rx_hz", 0.0)
                _f_op_lo = top1.get("f_op_lo_hz", 0.0)
                _zvs_x   = (_f_op_lo / _f0_tx) if _f0_tx > 0 else 0.0
                _rx_off  = ((_f0_rx - _f_op) / 1e3) if (_f0_rx > 0 and _f_op > 0) else 0.0
                _M_uH    = top1.get("M_uH", 0.0)
                _k       = top1.get("k", 0.0)
                log_cb(f"\nFinal top winner: TX {top1['tx_od_mm']:.1f}mm {top1['tx_turns']}t "
                       f"w={top1['tx_width']:.2f}  |  "
                       f"RX {top1['rx_od_mm']:.1f}mm {top1['rx_turns']}t "
                       f"w={top1['rx_width']:.2f} {top1['rx_topology']}")
                log_cb(f"  η={top1['eta_sys']*100:.1f}%  "
                       f"D@Vmax={min(top1.get('D_vmax', 1.0), 1.0)*100:.0f}%  "
                       f"D@Vmin={min(top1.get('D_vmin', 1.0), 1.0)*100:.0f}%  "
                       f"I_tx_pk@Vmax={top1.get('I_tx_pk_vmax', 0.0)*1000:.0f}mA  "
                       f"V_dc(Vmin,D=1)={top1.get('V_dc_min', 0.0):.2f}V  "
                       f"V_Crx_pk={top1.get('V_Crx_pk', 0.0):.0f}V")
                log_cb(f"  L_tx={top1.get('L_tx_uH',0):.2f}µH  "
                       f"L_rx={top1.get('L_rx_uH',0):.2f}µH  "
                       f"C_tx={top1.get('c_tx_nf',0):.1f}nF  "
                       f"C_rx={top1.get('c_rx_nf',0):.1f}nF")
                log_cb(f"  M={_M_uH:.3f}µH  k={_k:.4f}")
                log_cb(f"  f_op={_f_op/1e3:.1f}kHz  "
                       f"f0_tx={_f0_tx/1e3:.1f}kHz  f_op_lo={_f_op_lo/1e3:.1f}kHz "
                       f"(ZVS×{_zvs_x:.2f})  |  "
                       f"f0_rx={_f0_rx/1e3:.1f}kHz  Δf_rx(vs f_op)={_rx_off:+.1f}kHz")

        progress_cb(1.0)
        log_cb(f"\nOptimisation complete after {min(iteration, max_iters)} iteration(s).")
        done_cb({"winner": prev_winners[0] if prev_winners else None,
                 "iterations": iteration})

    except Exception as e:
        log_cb(f"\nFATAL ERROR: {e}\n{traceback.format_exc()}")
        done_cb(None)


def _same_config(a, b):
    return (int(a["tx_turns"]) == int(b["tx_turns"])
            and int(a.get("tx_l2_turns", 0)) == int(b.get("tx_l2_turns", 0))
            and int(a["rx_turns"]) == int(b["rx_turns"])
            and int(a.get("rx_inner_turns", 0)) == int(b.get("rx_inner_turns", 0))
            and round(a["tx_od_mm"], 1) == round(b["tx_od_mm"], 1)
            and round(a["rx_od_mm"], 1) == round(b["rx_od_mm"], 1)
            and round(a["tx_width"], 2) == round(b["tx_width"], 2)
            and round(a["rx_width"], 2) == round(b["rx_width"], 2))


# ─────────────────────────────────────────────────────────────────────────────
# Tab widget
# ─────────────────────────────────────────────────────────────────────────────

class NNOptimisationTab(ttk.Frame):

    def __init__(self, parent, app=None, on_next_tab=None, **kw):
        super().__init__(parent, **kw)
        self.app          = app
        self._on_next_tab = on_next_tab
        self._cancel_flag = threading.Event()
        self._running     = False
        self._results         = {}
        self._last_run_params = {}
        self._domain          = None
        self._build()
        self.after(100, self._post_build_init)

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

        # ── Domain info banner ────────────────────────────────────────────────
        dom_f = ttk.LabelFrame(col_l, text="Active Domain")
        dom_f.pack(fill="x", pady=(0, 6))
        self._domain_info_var = tk.StringVar(value="No model loaded.")
        self._domain_info_lbl = ttk.Label(dom_f, textvariable=self._domain_info_var,
                                          foreground="#1a6fcc", font=("Consolas", 8),
                                          wraplength=340, justify="left")
        self._domain_info_lbl.pack(anchor="w", padx=6, pady=(4, 4))

        # ── TX / RX narrowing ────────────────────────────────────────────────
        tx_f = ttk.LabelFrame(col_l, text="Narrow Down Domain — TX")
        tx_f.pack(fill="x", pady=(0, 6))
        self._tx_od_max = self._row(tx_f, "OD max (mm):", "")
        self._tx_id_min = self._row(tx_f, "ID min (mm):", "")

        rx_f = ttk.LabelFrame(col_l, text="Narrow Down Domain — RX")
        rx_f.pack(fill="x", pady=(0, 6))
        self._rx_od_max = self._row(rx_f, "OD max (mm):", "")
        self._rx_id_min = self._row(rx_f, "ID min (mm):", "")

        # Ground disc, PCB gap and per-layer stackup were tunable in earlier
        # versions, but the V7 LHS generator and FastHenry pipeline hard-code
        # all three (rx.ground_disc_dia_mm fixed, _TX_LAYERS/_RX_LAYERS baked
        # into parallel_sim, pcb_gap range degenerate).  The surrogate has no
        # training data outside those values, so the optimiser reads them
        # straight from domain.json now — see `load_domain_from_model`.

        # ── NN model folder (from NN Setup tab) ───────────────────────────────
        nn_f = ttk.LabelFrame(col_l, text="NN Model")
        nn_f.pack(fill="x", pady=(0, 6))
        mrow = ttk.Frame(nn_f); mrow.pack(fill="x", padx=6, pady=4)
        ttk.Label(mrow, text="Folder:", foreground="gray").pack(side="left")
        self._model_label_var = tk.StringVar(value="(select in NN Setup tab)")
        ttk.Label(mrow, textvariable=self._model_label_var,
                  foreground="#1a6fcc", font=("Consolas", 8),
                  wraplength=340, justify="left").pack(side="left", padx=6)

        ttk.Button(col_l, text="Next Tab →  (NN Analysis)",
                   command=self._on_next_tab_click).pack(fill="x", pady=(0, 6))

        log_lf = ttk.LabelFrame(col_l, text="Log")
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

        # ── RIGHT COLUMN ──────────────────────────────────────────────────────

        # ── Evaluation parameters ─────────────────────────────────────────────
        ev_f = ttk.LabelFrame(col_r, text="Evaluation Parameters")
        ev_f.pack(fill="x", pady=(0, 6))
        self._p_target_mw  = self._row_inline(ev_f, "Target RX power (mW):",
                                              DEFAULT_P_TARGET_MW, "Post-LDO load")
        self._v_cap_target = self._row(ev_f, "Cap target voltage (V):", DEFAULT_V_CAP_TARGET)
        ttk.Separator(ev_f, orient="horizontal").pack(fill="x", padx=6, pady=(2, 2))
        self._v_min        = self._row(ev_f, "TX V_min (V):", DEFAULT_V_MIN)
        self._v_max        = self._row(ev_f, "TX V_max (V):", DEFAULT_V_MAX)
        ttk.Separator(ev_f, orient="horizontal").pack(fill="x", padx=6, pady=(2, 2))
        self._d_min_pct    = self._row(ev_f, "Min duty cycle (%):", DEFAULT_D_MIN_PCT)
        self._d_max_pct    = self._row(ev_f, "Max duty cycle (%):", DEFAULT_D_MAX_PCT)
        ttk.Label(ev_f,
                  text="Min@Vmax: power-limiting floor. Max@Vmin: envelope-mod headroom. Blank = no limit.",
                  foreground="gray", font=("TkDefaultFont", 8),
                  wraplength=340, justify="left"
                  ).pack(anchor="w", padx=6, pady=(0, 2))
        self._freq_min_khz = self._row(ev_f, "Freq range min (kHz):", DEFAULT_FREQ_MIN_KHZ)
        self._freq_max_khz = self._row(ev_f, "Freq range max (kHz):", DEFAULT_FREQ_MAX_KHZ)
        ttk.Label(ev_f,
                  text="Operating freq range swept at 1 kHz steps. NN queries at domain centre.",
                  foreground="gray", font=("TkDefaultFont", 8),
                  wraplength=340, justify="left"
                  ).pack(anchor="w", padx=6, pady=(0, 2))
        self._dither_amp_khz = self._row(ev_f, "Dither amplitude ±(kHz):", DEFAULT_DITHER_AMP_KHZ)
        self._zvs_margin_pct = self._row(ev_f, "ZVS margin (%):", DEFAULT_ZVS_MARGIN_PCT)
        ttk.Label(ev_f,
                  text="TX: ZVS gate (f_drive_min > f₀·(1+margin)). "
                       "RX: closest to series resonance. 0 dither = ZVS disabled.",
                  foreground="gray", font=("TkDefaultFont", 8),
                  wraplength=340, justify="left"
                  ).pack(anchor="w", padx=6, pady=(0, 2))
        _cap_default = ", ".join(f"{c:g}" for c in E_VALUES_NF)
        for _cap_label, _cap_attr in [("TX caps (nF):", "_tx_caps_nf"),
                                       ("RX caps (nF):", "_rx_caps_nf")]:
            _cap_row = ttk.Frame(ev_f)
            _cap_row.pack(fill="x", padx=6, pady=3)
            ttk.Label(_cap_row, text=_cap_label, width=24, anchor="w").pack(side="left")
            _cap_var = tk.StringVar(value=_cap_default)
            setattr(self, _cap_attr, _cap_var)
            ttk.Entry(_cap_row, textvariable=_cap_var).pack(side="left", padx=4, fill="x", expand=True)
        self._rx_cap_vrating = self._row(ev_f, "C_rx rated voltage (V):", DEFAULT_RX_CAP_VRATING)

        # ── Iteration & sweep settings ────────────────────────────────────────
        it_f = ttk.LabelFrame(col_r, text="Optimisation Settings")
        it_f.pack(fill="x", pady=(0, 6))
        self._n_combos  = self._row(it_f, "Combinations / iter (M):", DEFAULT_N_COMBOS_M)
        self._max_iters = self._row(it_f, "Max iterations:", DEFAULT_MAX_ITERS)
        self._top_k     = self._row(it_f, "Top-K for FastHenry:", DEFAULT_TOP_K)
        self._fh_workers   = self._row(it_f, "FH workers:", DEFAULT_FH_WORKERS)
        self._fh_timeout   = self._row(it_f, "FH timeout (s):", DEFAULT_FH_TIMEOUT_S)

        # ── Training hyperparams ──────────────────────────────────────────────
        tr_f = ttk.LabelFrame(col_r, text="Retrain Hyperparameters")
        tr_f.pack(fill="x", pady=(0, 6))
        self._tr_epochs    = self._row(tr_f, "Epochs:", DEFAULT_TR_EPOCHS)
        self._tr_batch     = self._row(tr_f, "Batch size:", DEFAULT_TR_BATCH)
        self._tr_lr        = self._row(tr_f, "LR:", DEFAULT_TR_LR)
        self._tr_val_split = self._row(tr_f, "Val split:", DEFAULT_TR_VAL_SPLIT)

        # ── Control / progress / log ──────────────────────────────────────────
        run_lf = ttk.LabelFrame(col_r, text="Optimisation Control")
        run_lf.pack(fill="x", pady=(0, 6))
        btn_row = ttk.Frame(run_lf); btn_row.pack(fill="x", padx=6, pady=6)
        self._run_btn = ttk.Button(btn_row, text="▶  Start Optimisation",
                                   command=self._on_run)
        self._run_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self._cancel_btn = ttk.Button(btn_row, text="Cancel",
                                      command=self._on_cancel,
                                      state="disabled")
        self._cancel_btn.pack(side="left", expand=True, fill="x")

        prog_lf = ttk.LabelFrame(col_r, text="Progress")
        prog_lf.pack(fill="x", pady=(0, 6))
        self._progress = ttk.Progressbar(prog_lf, mode="determinate", maximum=100)
        self._progress.pack(fill="x", padx=6, pady=(6, 4))
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(prog_lf, textvariable=self._status_var,
                  foreground="gray", wraplength=600, justify="left"
                  ).pack(fill="x", padx=6, pady=(0, 6))


    def _post_build_init(self):
        self._restore_from_savestate()
        for _, _var in self._savestate_vars():
            _var.trace_add("write", lambda *_, s=self: s._save_state())

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _row(self, parent, label, default, label_width=24):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text=label, width=label_width, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var, width=10).pack(side="left", padx=4)
        return var

    def _row_inline(self, parent, label, default, inline_text, label_width=24):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text=label, width=label_width, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var, width=10).pack(side="left", padx=4)
        ttk.Label(row, text=inline_text,
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side="left", padx=(4, 0))
        return var

    def _bind_cap(self, var, coil, key, is_min=False):
        def _cap(*_):
            dom = self._domain.get(coil, {})
            bound = dom.get(key)
            if bound is None:
                return
            try:
                v = float(var.get())
            except ValueError:
                return
            clamped = max(v, bound) if is_min else min(v, bound)
            if abs(clamped - v) > 1e-9:
                var.set(f"{clamped:.1f}")
        var.trace_add("write", _cap)

    def _bind_od_id_guard(self, od_var, id_var):
        """Ensure OD > ID: clamp whichever field was just edited."""
        def _guard_od(*_):
            try:
                od = float(od_var.get())
                id_ = float(id_var.get())
            except ValueError:
                return
            if od <= id_:
                od_var.set(f"{id_ + 1.0:.1f}")

        def _guard_id(*_):
            try:
                od = float(od_var.get())
                id_ = float(id_var.get())
            except ValueError:
                return
            if id_ >= od:
                id_var.set(f"{od - 1.0:.1f}")

        od_var.trace_add("write", _guard_od)
        id_var.trace_add("write", _guard_id)

    @staticmethod
    def _short_path(p):
        try:
            return os.path.relpath(p)
        except ValueError:
            return p

    # ─────────────────────────────────────────────────────────────────────────
    # Domain loading (called by NN Setup tab when model folder changes)
    # ─────────────────────────────────────────────────────────────────────────

    def load_domain_from_model(self, model_dir: str):
        """
        Read a NN_V8-style `domain.json`:
          • tx/rx OD given as a range `od_mm: [lo, hi]`.
          • Both topologies are fixed: `fixed.tx_topology` and
            `fixed.rx_topology` — TX series (L1+L2), RX series across all
            4 active layers with independent outer/inner turn counts.
          • Stackup fixed per side under `fixed.tx_layers`/`fixed.rx_layers`.
          • Ground disc: `rx.ground_disc_dia_mm` (0 disables).
          • `tx.l2_turns` and `rx.inner_turns` are separate sampling dims.
        """
        domain = _load_domain(model_dir)
        self._domain = domain
        self._model_label_var.set(self._short_path(model_dir))

        # Reset fixed-run snapshot — _parse_params reads it instead of UI vars.
        self._fixed_gc_dia_mm    = 0.0
        self._fixed_pcb_gap_mm   = 0.0
        self._fixed_tx_oz_layers = [0.0, 0.0, 0.0, 0.0]
        self._fixed_rx_oz_layers = [0.0, 0.0, 0.0, 0.0]

        if domain is None:
            self._domain_info_var.set(
                "ERROR: domain.json missing or corrupt — model is unusable.")
            self._domain_info_lbl.configure(foreground="red")
            for v in (self._tx_od_max, self._tx_id_min,
                      self._rx_od_max, self._rx_id_min):
                v.set("")
            return

        self._domain_info_lbl.configure(foreground="#1a6fcc")

        tx    = domain.get("tx", {})
        rx    = domain.get("rx", {})
        glb   = domain.get("global", {})
        fixed = domain.get("fixed", {})

        def _hi(rng, fallback):
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                return float(rng[1])
            return float(fallback)

        tx_od_hi = _hi(tx.get("od_mm"), tx.get("od_max_mm", 54.0))
        rx_od_hi = _hi(rx.get("od_mm"), rx.get("od_max_mm", 54.0))
        tx_id_lo = float(tx.get("id_min_mm", 0.0))
        rx_id_lo = float(rx.get("id_min_mm", 0.0))

        self._tx_od_max.set(f"{tx_od_hi:.1f}")
        self._tx_id_min.set(f"{tx_id_lo:.1f}")
        self._rx_od_max.set(f"{rx_od_hi:.1f}")
        self._rx_id_min.set(f"{rx_id_lo:.1f}")

        # Stash all V7-fixed values; _parse_params will consume them directly.
        self._fixed_gc_dia_mm = float(rx.get("ground_disc_dia_mm", 0.0))

        pcb_rng = glb.get("pcb_gap_mm")
        if isinstance(pcb_rng, (list, tuple)) and len(pcb_rng) >= 2:
            self._fixed_pcb_gap_mm = 0.5 * (float(pcb_rng[0]) + float(pcb_rng[1]))
        elif pcb_rng is not None:
            self._fixed_pcb_gap_mm = float(pcb_rng)

        def _slot_oz(slots):
            out = [0.0, 0.0, 0.0, 0.0]
            for i, slot in enumerate(slots[:4]):
                try:
                    out[i] = (float(slot.get("copper_oz", 1.0))
                              if slot.get("active") else 0.0)
                except (TypeError, AttributeError):
                    pass
            return out
        self._fixed_tx_oz_layers = _slot_oz(fixed.get("tx_layers", []))
        self._fixed_rx_oz_layers = _slot_oz(fixed.get("rx_layers", []))

        freq     = glb.get("freq_hz", [0, 0])
        tx_topo  = fixed.get("tx_topology", "series")
        rx_topo  = fixed.get("rx_topology", "series")
        tx_turns = tx.get("turns", ["-", "-"])
        rx_turns = rx.get("turns", ["-", "-"])
        l2_turns = tx.get("l2_turns", ["-", "-"])
        rx_inner = rx.get("inner_turns", ["-", "-"])

        gc_str   = (f"{self._fixed_gc_dia_mm:.1f} mm"
                    if self._fixed_gc_dia_mm > 0 else "off")
        tx_oz_s  = " / ".join(f"{o:g}" for o in self._fixed_tx_oz_layers)
        rx_oz_s  = " / ".join(f"{o:g}" for o in self._fixed_rx_oz_layers)

        info = (
            f"TX: OD {tx.get('od_mm', '?')} mm | ID ≥ {tx_id_lo:.1f} mm | "
            f"turns {tx_turns[0]}–{tx_turns[1]} | L2 {l2_turns[0]}–{l2_turns[1]} | "
            f"topo: {tx_topo} (fixed)\n"
            f"RX: OD {rx.get('od_mm', '?')} mm | ID ≥ {rx_id_lo:.1f} mm | "
            f"outer {rx_turns[0]}–{rx_turns[1]} | inner {rx_inner[0]}–{rx_inner[1]} | "
            f"topo: {rx_topo} 4-layer (fixed)\n"
            f"Freq: {freq[0]/1e3:.0f}–{freq[1]/1e3:.0f} kHz  |  "
            f"PCB gap {self._fixed_pcb_gap_mm:.2f} mm  |  GC: {gc_str}\n"
            f"Stackup oz  TX: {tx_oz_s}   RX: {rx_oz_s}   (all fixed by trainer)"
        )
        self._domain_info_var.set(info)

    # ─────────────────────────────────────────────────────────────────────────
    # Savestate
    # ─────────────────────────────────────────────────────────────────────────

    def _savestate_vars(self):
        """Return list of (key, StringVar) pairs for all persistent inputs."""
        return [
            ("p_target_mw",    self._p_target_mw),
            ("v_cap_target",   self._v_cap_target),
            ("rx_cap_vrating", self._rx_cap_vrating),
            ("v_min",          self._v_min),
            ("v_max",          self._v_max),
            ("d_min_pct",      self._d_min_pct),
            ("d_max_pct",      self._d_max_pct),
            ("freq_min_khz",   self._freq_min_khz),
            ("freq_max_khz",   self._freq_max_khz),
            ("dither_amp_khz", self._dither_amp_khz),
            ("zvs_margin_pct", self._zvs_margin_pct),
            ("tx_caps_nf",     self._tx_caps_nf),
            ("rx_caps_nf",     self._rx_caps_nf),
            ("n_combos",       self._n_combos),
            ("max_iters",      self._max_iters),
            ("top_k",          self._top_k),
            ("fh_workers",     self._fh_workers),
            ("fh_timeout",     self._fh_timeout),
            ("tr_epochs",      self._tr_epochs),
            ("tr_batch",       self._tr_batch),
            ("tr_lr",          self._tr_lr),
            ("tr_val_split",   self._tr_val_split),
            ("tx_od_max",      self._tx_od_max),
            ("tx_id_min",      self._tx_id_min),
            ("rx_od_max",      self._rx_od_max),
            ("rx_id_min",      self._rx_id_min),
        ]

    def _save_state(self):
        if self.app is None:
            return
        st = {key: var.get() for key, var in self._savestate_vars()}
        self.app.persist_nn_optim_tab(st)

    def _restore_from_savestate(self):
        if self.app is None:
            return
        st = self.app.load_nn_optim_tab_state()
        if not st:
            return
        try:
            for key, var in self._savestate_vars():
                if key in st:
                    var.set(st[key])
        except Exception:
            pass

    def _on_reset_defaults(self):
        _cap_default = ", ".join(f"{c:g}" for c in E_VALUES_NF)
        defaults = {
            "p_target_mw":    DEFAULT_P_TARGET_MW,
            "v_cap_target":   DEFAULT_V_CAP_TARGET,
            "rx_cap_vrating": DEFAULT_RX_CAP_VRATING,
            "v_min":          DEFAULT_V_MIN,
            "v_max":          DEFAULT_V_MAX,
            "d_min_pct":      DEFAULT_D_MIN_PCT,
            "d_max_pct":      DEFAULT_D_MAX_PCT,
            "freq_min_khz":   DEFAULT_FREQ_MIN_KHZ,
            "freq_max_khz":   DEFAULT_FREQ_MAX_KHZ,
            "dither_amp_khz": DEFAULT_DITHER_AMP_KHZ,
            "zvs_margin_pct": DEFAULT_ZVS_MARGIN_PCT,
            "tx_caps_nf":     _cap_default,
            "rx_caps_nf":     _cap_default,
            "n_combos":       DEFAULT_N_COMBOS_M,
            "max_iters":      DEFAULT_MAX_ITERS,
            "top_k":          DEFAULT_TOP_K,
            "fh_workers":     DEFAULT_FH_WORKERS,
            "fh_timeout":     DEFAULT_FH_TIMEOUT_S,
            "tr_epochs":      DEFAULT_TR_EPOCHS,
            "tr_batch":       DEFAULT_TR_BATCH,
            "tr_lr":          DEFAULT_TR_LR,
            "tr_val_split":   DEFAULT_TR_VAL_SPLIT,
            "tx_od_max":      "",
            "tx_id_min":      "",
            "rx_od_max":      "",
            "rx_id_min":      "",
        }
        for key, var in self._savestate_vars():
            if key in defaults:
                var.set(defaults[key])
        self._save_state()

    # ─────────────────────────────────────────────────────────────────────────
    # Event handlers
    # ─────────────────────────────────────────────────────────────────────────

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
        self._set_status("Running optimisation…")

        threading.Thread(
            target=_run_optimisation,
            args=(params, self._progress_cb, self._log_cb,
                  self._log_cb_overwrite, self._done_cb, self._cancel_flag),
            daemon=True,
        ).start()

    def _on_cancel(self):
        self._cancel_flag.set()
        self._set_status("Cancelling…", color="orange")

    def _progress_cb(self, frac):
        self.after(0, lambda: self._progress.configure(value=frac * 100))

    def _log_cb(self, msg):
        self.after(0, lambda m=msg: self._log_append(m))

    def _log_cb_overwrite(self, msg):
        self.after(0, lambda m=msg: self._log_overwrite(m))

    def _done_cb(self, result):
        self.after(0, lambda: self._on_done(result))

    def _on_done(self, result):
        self._running = False
        self._run_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")

        if result is None:
            self._set_status("Error or cancelled — see log.", color="red")
            return

        winner = result.get("winner")
        iters  = result.get("iterations", "?")
        if winner:
            self._set_status(
                f"Done after {iters} iteration(s).  "
                f"Winner: TX {winner['tx_od_mm']:.1f}mm {winner['tx_turns']}t  "
                f"RX {winner['rx_od_mm']:.1f}mm {winner['rx_turns']}t "
                f"{winner['rx_topology']}  η={winner['eta_sys']*100:.1f}%",
                color="green")
        else:
            self._set_status(f"Done after {iters} iteration(s).", color="green")

    # ─────────────────────────────────────────────────────────────────────────
    # Param parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_params(self):
        def flt(var, name, lo=None, hi=None, allow_empty=False):
            raw = var.get().strip()
            if allow_empty and raw == "":
                return 0.0
            try:
                v = float(raw)
            except ValueError:
                raise ValueError(f"'{name}' must be a number.")
            if lo is not None and v < lo:
                raise ValueError(f"'{name}' must be ≥ {lo}.")
            if hi is not None and v > hi:
                raise ValueError(f"'{name}' must be ≤ {hi}.")
            return v

        if self._domain is None:
            raise ValueError(
                "No domain loaded.\n\nSelect a valid model folder in the NN Setup tab.")

        tx    = self._domain.get("tx", {})
        rx    = self._domain.get("rx", {})
        glb   = self._domain.get("global", {})

        def _hi(rng, fallback):
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                return float(rng[1])
            return float(fallback)

        dom_tx_od_max = _hi(tx.get("od_mm"), tx.get("od_max_mm", 999.0))
        dom_tx_id_min = float(tx.get("id_min_mm", 0.0))
        dom_rx_od_max = _hi(rx.get("od_mm"), rx.get("od_max_mm", 999.0))
        dom_rx_id_min = float(rx.get("id_min_mm", 0.0))
        dom_tx_w      = tx.get("trace_width_mm", [0.2, 1.2])
        dom_rx_w      = rx.get("trace_width_mm", [0.2, 1.2])
        dom_tx_turns  = tx.get("turns", [3, 99])
        dom_rx_turns  = rx.get("turns", [3, 99])
        dom_tx_l2     = tx.get("l2_turns", [1, int(dom_tx_turns[1])])
        dom_rx_inner  = rx.get("inner_turns", [1, int(dom_rx_turns[1])])
        dom_freq      = glb.get("freq_hz", [340000.0, 380000.0])

        tx_od_max = min(flt(self._tx_od_max, "TX OD max", lo=10.0), dom_tx_od_max)
        tx_id_min = max(flt(self._tx_id_min, "TX ID min", lo=0.0), dom_tx_id_min)
        if tx_id_min >= tx_od_max:
            raise ValueError("TX ID min must be < TX OD max.")

        rx_od_max = min(flt(self._rx_od_max, "RX OD max", lo=10.0), dom_rx_od_max)
        rx_id_min = max(flt(self._rx_id_min, "RX ID min", lo=0.0), dom_rx_id_min)
        if rx_id_min >= rx_od_max:
            raise ValueError("RX ID min must be < RX OD max.")

        # Ground disc, PCB gap, stackup: snapshot taken in load_domain_from_model
        # straight from domain.json (V7 hard-fixes all three).
        gc_dia_mm  = float(getattr(self, "_fixed_gc_dia_mm", 0.0))
        gc_enabled = gc_dia_mm > 0.0

        p_target_w      = flt(self._p_target_mw,    "Target RX power",    lo=0.001) / 1000.0
        v_cap_target_dc = flt(self._v_cap_target,  "Cap target voltage",
                              lo=_V_MIN_INDUCED_DC, hi=_V_ZENER_DC)
        v_rx_cap_rating = flt(self._rx_cap_vrating, "C_rx rated voltage", lo=5.0, hi=1000.0)
        v_min      = flt(self._v_min, "V_min", lo=0.1)
        v_max      = flt(self._v_max, "V_max", lo=v_min)
        d_min_pct  = flt(self._d_min_pct, "Min duty %", lo=0.0, allow_empty=True)
        d_min      = d_min_pct / 100.0
        d_max_pct  = flt(self._d_max_pct, "Max duty %", lo=0.0, hi=100.0, allow_empty=True)
        d_max      = (d_max_pct / 100.0) if d_max_pct < 100.0 else 1.0

        freq_min_hz = flt(self._freq_min_khz, "Freq min", lo=1.0) * 1e3
        freq_max_hz = flt(self._freq_max_khz, "Freq max", lo=freq_min_hz / 1e3) * 1e3
        if freq_max_hz <= freq_min_hz:
            raise ValueError("Freq range max must be > min.")

        def _parse_cap_list(var, name):
            opts = []
            for _s in var.get().strip().split(","):
                _s = _s.strip()
                if _s:
                    try:
                        _v = float(_s)
                        if _v > 0:
                            opts.append(_v * 1e-9)
                    except ValueError:
                        pass
            if not opts:
                raise ValueError(f"{name} must be a comma-separated list in nF (e.g. '100, 200').")
            return sorted(opts)

        c_tx_options_f = _parse_cap_list(self._tx_caps_nf, "TX cap options")
        c_rx_options_f = _parse_cap_list(self._rx_caps_nf, "RX cap options")
        dither_amp_hz  = flt(self._dither_amp_khz, "Dither amplitude", lo=0.0) * 1e3
        zvs_margin     = flt(self._zvs_margin_pct, "ZVS margin",       lo=0.0, hi=50.0) / 100.0

        # PCB gap + stackup come from the domain snapshot (V7-fixed).
        pcb_gap_mm = float(getattr(self, "_fixed_pcb_gap_mm", 2.4))
        tx_oz_per_layer = list(getattr(self, "_fixed_tx_oz_layers",
                                       [0.0, 0.0, 0.0, 0.0]))
        rx_oz_per_layer = list(getattr(self, "_fixed_rx_oz_layers",
                                       [0.0, 0.0, 0.0, 0.0]))
        if not any(o > 0 for o in tx_oz_per_layer):
            raise ValueError(
                "TX stackup has no active layers in the loaded domain.json.")
        if not any(o > 0 for o in rx_oz_per_layer):
            raise ValueError(
                "RX stackup has no active layers in the loaded domain.json.")

        n_combos  = int(flt(self._n_combos, "Combinations (M)", lo=0.001) * 1_000_000)
        max_iters = int(flt(self._max_iters, "Max iterations", lo=1))
        top_k     = int(flt(self._top_k, "Top-K", lo=1, hi=50))
        fh_workers  = int(flt(self._fh_workers, "FH workers", lo=1))
        fh_timeout  = flt(self._fh_timeout, "FH timeout", lo=10.0)

        epochs    = int(flt(self._tr_epochs,    "Epochs",    lo=1))
        batch     = int(flt(self._tr_batch,     "Batch",     lo=1))
        lr        = flt(self._tr_lr,        "LR",        lo=1e-7)
        val_split = flt(self._tr_val_split, "Val split", lo=0.01, hi=0.99)

        model_dir = self._get_model_dir()
        if not os.path.isdir(model_dir):
            raise ValueError(
                f"Model folder not found:\n{model_dir}\n\n"
                "Select a folder in the NN Setup tab.")
        self._model_label_var.set(self._short_path(model_dir))

        return dict(
            model_dir=model_dir,
            tx_od_max=tx_od_max, tx_id_min=tx_id_min,
            tx_width_min=dom_tx_w[0], tx_width_max=dom_tx_w[1],
            tx_turns_min=int(dom_tx_turns[0]),
            tx_turns_max=int(dom_tx_turns[1]),
            tx_l2_turns_min=int(dom_tx_l2[0]),
            tx_l2_turns_max=int(dom_tx_l2[1]),
            rx_od_max=rx_od_max, rx_id_min=rx_id_min,
            rx_width_min=dom_rx_w[0], rx_width_max=dom_rx_w[1],
            rx_turns_min=int(dom_rx_turns[0]),
            rx_turns_max=int(dom_rx_turns[1]),
            rx_inner_turns_min=int(dom_rx_inner[0]),
            rx_inner_turns_max=int(dom_rx_inner[1]),
            freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
            gc_enabled=gc_enabled, gc_dia_mm=gc_dia_mm,
            p_target_w=p_target_w, v_min=v_min, v_max=v_max,
            d_min=d_min, d_max=d_max,
            v_cap_target_dc=v_cap_target_dc,
            v_ldo_out=_V_LDO_OUT,
            v_rx_cap_rating=v_rx_cap_rating,
            c_tx_options_f=c_tx_options_f,
            c_rx_options_f=c_rx_options_f,
            dither_amp_hz=dither_amp_hz,
            zvs_margin=zvs_margin,
            n_combos=n_combos, max_iters=max_iters, top_k=top_k,
            fh_workers=fh_workers, fh_timeout=fh_timeout,
            # Fixed-stackup constants snapshotted from domain.json.
            pcb_gap_mm=pcb_gap_mm,
            tx_oz_per_layer=tx_oz_per_layer,
            rx_oz_per_layer=rx_oz_per_layer,
            train_params=dict(epochs=epochs, batch=batch,
                              lr=lr, val_split=val_split),
            domain=self._domain,
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

        sim     = self.app.sim_tab
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
        tx_path   = os.path.join(temp_dir, "nn_auto_tx.inp")
        fmin = r["freq_hz"]; fmax = fmin + 15000.0
        pc.write_topology_inp("parallel", tx_layers, tx_path,
                              w_mm=r["tx_width"], fmin=fmin, fmax=fmax)
        tx_meta = {"role": "TX", "topology": "parallel",
                   "layer_params": [(r["tx_width"], ld["h_mm"], len(ld["nodes"]))
                                    for ld in tx_layers],
                   "nodes_by_layer": [list(ld["nodes"]) for ld in tx_layers]}

        rx_topo = _RX_TOPOLOGY
        rx_inner_t = int(r.get("rx_inner_turns", 0)) or int(r["rx_turns"])
        rx_sp   = pc.SpiralParams(
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
        rx_layers, _ = pc.active_layer_data_rx_independent(rx_sp, rx_stackup, rx_inner_t)
        rx_path   = os.path.join(temp_dir, "nn_auto_rx.inp")
        pc.write_topology_inp(rx_topo, rx_layers, rx_path,
                              w_mm=r["rx_width"], fmin=fmin, fmax=fmax)
        rx_flags  = pc.series_reverse_flags_for_topology(rx_topo, len(rx_layers))
        rx_native = pc.reverse_nodes_for_series_flow(rx_layers, rx_flags)
        rx_meta = {"role": "RX", "topology": rx_topo,
                   "layer_params": [(ld.get("w_mm", r["rx_width"]),
                                     ld["h_mm"], len(ld["nodes"]))
                                    for ld in rx_native],
                   "nodes_by_layer": [list(ld["nodes"]) for ld in rx_native]}

        sim = self.app.sim_tab
        sim.register_coil("TX", "Optimisation", tx_path, tx_meta)
        sim.register_coil("RX", "Optimisation", rx_path, rx_meta)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _log_clear(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _log_append(self, msg):
        at_bottom = self._log_text.yview()[1] >= 0.999
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        if at_bottom:
            self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _log_overwrite(self, msg):
        """Replace the last line in the log (for progress updates)."""
        at_bottom = self._log_text.yview()[1] >= 0.999
        self._log_text.configure(state="normal")
        self._log_text.delete("end-1l linestart", "end-1c")
        self._log_text.insert("end", msg)
        if at_bottom:
            self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _set_status(self, msg, color="gray"):
        self._status_var.set(msg)
