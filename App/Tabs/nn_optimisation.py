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
DEFAULT_GC_ENABLED     = True
DEFAULT_P_TARGET_MW    = "30"   # mW (post-LDO downstream load)
DEFAULT_V_MIN          = "3.2"  # V
DEFAULT_V_MAX          = "4.4"  # V
DEFAULT_V_CAP_TARGET   = "4.5"  # V — controller steady-state cap target
DEFAULT_RX_CAP_VRATING = "100"  # V — rated voltage of C_rx series resonant cap
DEFAULT_D_MIN_PCT      = "20"   # blank = no floor
DEFAULT_D_MAX_PCT      = "90"   # blank = no ceiling
DEFAULT_FREQ_MIN_KHZ   = "200"  # kHz — user operating range (min)
DEFAULT_FREQ_MAX_KHZ   = "360"  # kHz — user operating range (max)
DEFAULT_DITHER_AMP_KHZ = "5"    # frequency dither amplitude (kHz)
DEFAULT_ZVS_MARGIN_PCT = "50"   # f_drive_min must be this % above f0_tx
DEFAULT_N_COMBOS_M     = "30"   # millions
DEFAULT_MAX_ITERS      = "50"
DEFAULT_TOP_K          = "6"
DEFAULT_FH_WORKERS     = "6"
DEFAULT_FH_TIMEOUT_S   = "360"
DEFAULT_TR_EPOCHS      = "300"
DEFAULT_TR_BATCH       = "256"
DEFAULT_TR_LR          = "0.0005"
DEFAULT_TR_VAL_SPLIT   = "0.2"

# Fixed-stackup defaults for one optimisation run (user-editable in the tab).
# 0 oz = layer inactive. Re-run the optimiser for other stackups / pcb gaps.
DEFAULT_PCB_GAP_MM     = "2.5"
DEFAULT_TX_LAYER_OZ    = ["1.0", "0.5", "0.5", "1.0"]
DEFAULT_RX_LAYER_OZ    = ["1.0", "0.5", "0.5", "1.0"]

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

# Stable topology vocabulary — must match TOPOLOGY_VOCAB in train_surrogate.py
TOPOLOGY_VOCAB = ["parallel", "series", "parallel_pairs_ser"]

_V_DIODE = 0.35
_V_DROP  = 2.0 * _V_DIODE

# RX-side rail constraints (post-rectifier, on storage cap)
# The downstream chain is:  rectifier → cap → LDO → 3 V load.
# The controller actively regulates the cap to V_CAP_TARGET so the tag has
# headroom to keep running through the WPT-off window during optical transfer.
# The Zener is a passive safety clamp; it should rarely engage in normal
# operation because the controller throttles before reaching it.
_V_LDO_OUT        = 3.0   # 3 V LDO output rail (downstream of cap)
_V_CAP_TARGET_DEF = 4.5   # default controller setpoint (UI-overridable)
_V_ZENER_DC       = 6.8   # passive TVS clamp — safety only
_RX_CAP_V_RATING_DEF = 100.0  # V — default C_rx rated voltage (series resonant cap)
_V_MIN_INDUCED_DC = 4.0   # hard floor: cap must be able to reach this at V_min, D=1
                          #   (LDO needs ≥ ~0.5 V dropout, plus reserve for the
                          #    optical-transfer WPT-off window)

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
# Vectorised physics evaluation (mirrors nn_analysis_tab logic)
# ─────────────────────────────────────────────────────────────────────────────

def _score_batch(L_tx, L_rx, M, R_tx_nn, R_rx_nn, DCR_tx, DCR_rx,
                 C_tx, C_rx,
                 freq_ref_hz, freq_hz, f_drive_min_hz,
                 V_min, V_max, P_target_w, D_min, D_max,
                 zvs_margin, V_min_induced_dc, V_zener_dc,
                 V_cap_target_dc, V_ldo_out, V_rx_cap_rating):
    """
    Returns (eta_sys, score, feasible, feasible_relaxed,
             D_vmax, D_vmin, V_dc_min,
             f0_tx_hz, f0_rx_hz, f_op_lo_hz, I_tx_pk_vmax, V_Crx_pk).

    eta_sys          = TRUE physical efficiency end-to-end (TX bridge → 3 V load).
    score            = ranking metric = eta_sys × dmin_softness × emi_softness × reserve_softness.
    feasible         = strict (all hard gates AND D_vmax ≥ D_min preference).
    feasible_relaxed = D_min preference dropped (fallback when strict is empty).
    I_tx_pk_vmax     = TX coil peak on-time current at V_max (EMI proxy).
    V_Crx_pk         = Peak voltage across the series resonant cap C_rx at matched load / V_min
                       = V_pk_matched × Q_rx.  Hard gate: must be ≤ V_rx_cap_rating × 0.8.

    System model (downstream of the air-gap link):
        rectifier → storage cap → LDO → 3 V load
      • The user-specified P_target_w is the *post-LDO* load power (what the tag
        actually consumes downstream of the 3 V rail).
      • A closed-loop controller throttles drive (duty / detune) to hold the cap
        at V_cap_target_dc — that gives the tag a reserve voltage so it can keep
        running through the brief WPT-off window during optical transfer.
      • Cap-side power demand:  P_cap = P_target_w · V_cap / V_ldo_out
      • LDO efficiency:         eta_ldo = V_ldo_out / V_cap   (linear regulator)
      • If the natural full-drive V_dc at V_min exceeds V_cap_target the
        controller throttles down — no zener heat dump in normal operation
        (the zener is a passive safety clamp only).
      • If the natural V_dc is below V_cap_target the cap sags to that lower
        value (still feasible if ≥ V_min_induced_dc) and eta_ldo improves
        slightly, at the cost of less reserve for the WPT-off window.

    Control model (link side):
      • At heavy load (V_min) the controller drives near f0_tx, just enough above
        to maintain ZVS. Gates and ranking use matched-load values at this op
        point.
      • At light load (V_max) the controller can detune to f_drive_max to throttle.
        TX-tank power gain G(f) = 1 / (1 + (Q_tx · (f/f0 - f0/f))²) reduces P,
        giving the system room to back off without violating duty floor.

    Hard gates:
      • ZVS:       f_drive_min ≥ f0_tx · (1 + zvs_margin)
      • P_min:     P_dc_natural(V_min) ≥ P_cap_required   (P_target × V_cap_target / V_ldo)
      • D_min:     D_vmax ≥ D_min_user             (coil weak enough at Vmax that ≥D_min% duty is required → power-limiting / EMI headroom)
      • D_max:     D_vmin ≤ D_max_user             (Vmin needs ≤ D_max% duty → envelope-mod gap preserved)
      • V_floor:   V_dc_natural(V_min) ≥ V_min_induced_dc  (cap can reach the floor at full drive)

    eta_sys (true physical efficiency, displayed):
      eta_link · eta_rect_steady · eta_ldo · detune_rx(f_drive_min)

    where eta_rect_steady is the rectifier efficiency at the *throttled* op point
    (V_pk = V_cap_steady + V_drop), and eta_ldo accounts for the linear-regulator
    drop from V_cap_steady down to V_ldo_out.

    score (ranking only, NOT displayed):
      eta_sys · dmin_softness · emi_softness · reserve_softness

    Soft preferences:
      • dmin_softness    — D_vmax ≥ D_min (power-limit / EMI headroom at V_max).
      • emi_softness     — low TX peak on-time current at V_max (radiated H-field).
      • reserve_softness — natural V_dc at V_min full-drive reaches V_cap_target.
                           A coil that only just clears V_min_induced_dc lets the
                           cap sag, eating burst-handling margin during the
                           WPT-off IR-transmit window. Quadratic ramp 0→1 across
                           [0, V_cap_target].
    """
    # freq_hz may be a per-row array when operating frequency is a sweep dim.
    freq_hz_arr = np.asarray(freq_hz, dtype=np.float64)
    omega       = 2.0 * math.pi * freq_hz_arr
    skin_sc     = np.sqrt(freq_hz_arr / freq_ref_hz)

    L_tx = np.maximum(L_tx.astype(np.float64) * 1e-6, 1e-9)
    L_rx = np.maximum(L_rx.astype(np.float64) * 1e-6, 1e-9)
    M    = np.maximum(M.astype(np.float64)    * 1e-6, 0.0)
    C_tx = np.maximum(np.asarray(C_tx, dtype=np.float64), 1e-15)
    C_rx = np.maximum(np.asarray(C_rx, dtype=np.float64), 1e-15)

    R_tx_ac = DCR_tx.astype(np.float64) + np.maximum(
        0.0, R_tx_nn.astype(np.float64) - DCR_tx.astype(np.float64)) * skin_sc
    R_rx_ac = DCR_rx.astype(np.float64) + np.maximum(
        0.0, R_rx_nn.astype(np.float64) - DCR_rx.astype(np.float64)) * skin_sc

    Q_tx = omega * L_tx / np.maximum(R_tx_ac, 1e-12)
    Q_rx = omega * L_rx / np.maximum(R_rx_ac, 1e-12)
    k    = np.clip(M / np.sqrt(L_tx * L_rx), 0.0, 1.0)
    U    = k * np.sqrt(np.maximum(Q_tx * Q_rx, 0.0))

    sq       = np.sqrt(1.0 + U * U)
    eta_link = U * U / np.maximum((1.0 + sq) ** 2, 1e-18)
    Z_tx_opt = np.maximum(R_tx_ac * sq, 1e-12)

    # ── Resonances (per-row, given the sampled C_tx, C_rx) ────────────────
    f0_tx = 1.0 / (2.0 * math.pi * np.sqrt(L_tx * C_tx))
    f0_rx = 1.0 / (2.0 * math.pi * np.sqrt(L_rx * C_rx))

    # ── ZVS feasibility: f_drive_min = freq - dither must clear f0_tx ─────
    # f_drive_min/max may be per-row arrays when freq is a sweep dimension.
    f_drive_min_arr = np.asarray(f_drive_min_hz, dtype=np.float64)
    zvs_ok  = f_drive_min_arr >= f0_tx * (1.0 + zvs_margin)


    # Heavy-load op point = minimum drive frequency (per-row).
    f_op_lo = f_drive_min_arr

    V_rms_min = V_min * math.sqrt(2.0) / math.pi
    V_rms_max = V_max * math.sqrt(2.0) / math.pi

    # Matched-load reference (heavy-load assumption).
    P_rx_min_m = eta_link * (V_rms_min ** 2) / Z_tx_opt
    P_rx_max_m = eta_link * (V_rms_max ** 2) / Z_tx_opt

    R_ratio = np.sqrt(np.maximum(R_rx_ac / np.maximum(R_tx_ac, 1e-12), 0.0))

    def _v_pk_matched(V_rms):
        return (U * V_rms * R_ratio
                / np.maximum(1.0 + sq, 1e-12) * math.sqrt(2.0))

    V_pk_min_m = _v_pk_matched(V_rms_min)        # heavy load, matched
    V_pk_max_m = _v_pk_matched(V_rms_max)        # light load, matched
    V_dc_min_m = np.maximum(V_pk_min_m - _V_DROP, 0.0)

    eta_rect_min = np.where(V_pk_min_m > _V_DROP,
                            V_dc_min_m / np.maximum(V_pk_min_m, 1e-12), 0.0)

    P_dc_min = P_rx_min_m * eta_rect_min   # natural full-drive DC power at V_min

    # ── Steady-state operating point (controller throttles to V_cap_target) ──
    # The controller drives the cap to V_cap_target_dc whenever the link can
    # support it. If the natural V_dc at full drive falls short, the cap
    # sags to V_dc_min_m (still feasible if ≥ V_min_induced_dc).
    V_cap_steady = np.minimum(V_dc_min_m, V_cap_target_dc)

    # Throttled rectifier op point: when the controller throttles (natural V_dc
    # would exceed V_cap_target), the AC swing is reduced until the rectifier
    # outputs V_cap_target. So V_pk_steady = V_cap_target + V_drop (throttled)
    # or V_pk_min_m (unthrottled — coil running flat-out).
    throttled    = V_dc_min_m > V_cap_target_dc
    V_pk_steady  = np.where(throttled, V_cap_steady + _V_DROP, V_pk_min_m)
    eta_rect_ss  = np.where(V_pk_steady > _V_DROP,
                            V_cap_steady / np.maximum(V_pk_steady, 1e-12), 0.0)

    # ── LDO loss (linear regulator: V_cap → 3 V at constant load current) ──
    # Power at cap = P_target_w · V_cap / V_ldo_out, so eta_ldo = V_ldo / V_cap.
    eta_ldo = np.minimum(V_ldo_out / np.maximum(V_cap_steady, 1e-12), 1.0)

    # Required cap-side power = P_target × V_cap_target / V_ldo_out.
    # Use V_cap_target (controller's target) for gates so the budget is
    # consistent with the assumed steady-state operating point.
    P_cap_required = P_target_w * V_cap_target_dc / V_ldo_out

    # ── P_dc at V_max (light load), full drive, no zener clip, no detuning ──
    # Used for duty-cycle gates — represents the natural maximum the coil
    # would deliver if the controller didn't throttle.
    P_dc_max_natural = P_rx_max_m * np.where(V_pk_max_m > _V_DROP,
                                              (V_pk_max_m - _V_DROP)
                                              / np.maximum(V_pk_max_m, 1e-12), 0.0)

    # D_vmax = duty needed at V_max (no detuning) to deliver P_cap_required.
    # Gate: D_vmax >= D_min means the coil is weak enough at Vmax that we can't
    # over-deliver — gives power-limiting / EMI headroom.
    D_vmax = np.where(P_dc_max_natural > 1e-18,
                      P_cap_required / P_dc_max_natural, np.inf)

    # D_vmin = duty needed at V_min (no detuning) to deliver P_cap_required.
    # Gate: D_vmin <= D_max means we never exceed D_max%, preserving envelope-mod gap.
    D_vmin = np.where(P_dc_min > 1e-18, P_cap_required / P_dc_min, np.inf)

    # ── Hard gates ────────────────────────────────────────────────────────
    # pmin_ok: at V_min with 100% duty, coil delivers at least P_cap_required.
    pmin_ok = P_dc_min >= P_cap_required
    if D_min > 0.0:
        # D_vmax must be in [D_min, 1.0]: ≥ D_min for power-limiting headroom,
        # ≤ 1.0 because we can't exceed 100% duty (coil too weak otherwise).
        dmin_ok = (D_vmax >= D_min) & (D_vmax <= 1.0)
    else:
        dmin_ok = np.ones_like(P_dc_min, dtype=bool)
    # dmax_ok: at V_min, duty needed <= D_max (preserves envelope-mod gap).
    if D_max < 1.0:
        dmax_ok = D_vmin <= D_max
    else:
        dmax_ok = np.ones_like(P_dc_min, dtype=bool)
    # V_floor: natural full-drive cap voltage at V_min must clear the floor.
    # The zener (V_zener_dc) is irrelevant here since V_min_induced_dc << V_zener.
    vfloor_ok = V_dc_min_m >= V_min_induced_dc
    V_dc_min_eff = np.minimum(V_dc_min_m, V_zener_dc)   # for display/output

    # ── Series resonant C_rx voltage stress ───────────────────────────────
    # In a series-resonant RX (L_rx — C_rx — bridge), at matched load:
    #   I_circ_pk = V_pk_matched / R_rx_ac
    #   V_Crx_pk  = I_circ_pk × X_Crx = V_pk_matched × Q_rx
    #               (Q_rx = ω·L_rx / R_rx_ac = X_Lrx / R_rx_ac = X_Crx / R_rx_ac at f0_rx)
    # 20 % derating applied: gate passes when V_Crx_pk ≤ V_rx_cap_rating × 0.8.
    V_Crx_pk  = V_pk_min_m * Q_rx
    vrxcap_ok = V_Crx_pk <= (V_rx_cap_rating * 0.8)

    # Base hard gates (D_min is intentionally excluded — it's a soft preference).
    base_ok = (zvs_ok & pmin_ok & dmax_ok & vfloor_ok & vrxcap_ok
               & (R_tx_ac > 0) & (R_rx_ac > 0) & (M > 0))
    feasible        = base_ok & dmin_ok   # strict: includes D_min preference
    feasible_relaxed = base_ok            # relaxed: D_min dropped if nothing passes strict

    # ── RX detune at the actual operating frequency ──────────────────────
    # C_rx is snapped to resonate near freq_hz_arr (the operating freq),
    # so detune is measured relative to that, not freq_ref_hz.
    f0_rx_safe = np.maximum(f0_rx, 1.0)
    x_rx       = freq_hz_arr / f0_rx_safe - f0_rx_safe / freq_hz_arr
    detune_rx  = 1.0 / (1.0 + (Q_rx * x_rx) ** 2)

    # ── True physical system efficiency (this is what we display) ────────
    # eta_sys is the actual end-to-end electrical efficiency at the heavy-load
    # op point — TX bridge → tank → air gap → rectifier → cap → LDO → 3 V load.
    # It is NEVER multiplied by soft preference penalties; those go into a
    # separate ranking score below.
    eta_sys = eta_link * eta_rect_ss * eta_ldo * detune_rx

    # ── EMI proxy: TX peak on-time current at V_max ─────────────────────
    # The TX coil current during the duty-on portion of the envelope is what
    # radiates. At matched (heavy) load the TX sees Z_tx_opt impedance, so
    #   I_tx_pk_on(V) = V_rms · sqrt(2) / Z_tx_opt
    # We use V_max because that's the worst-case EMI scenario (full battery).
    # Lower I_tx_pk_on → less radiated H-field → better EMI.
    I_tx_pk_vmax = V_rms_max * math.sqrt(2.0) / Z_tx_opt

    # ── Ranking score (used for sorting, NOT displayed) ──────────────────
    # score = eta_sys
    #         × dmin_softness   (soft preference: D_vmax ≥ D_min)
    #         × emi_softness    (prefer low TX peak current)
    #
    # Both soft factors are gentle (~10–30 % impact at typical deviations) so
    # a moderately better design with a slightly violated preference can still
    # win over a worse strictly-compliant design — but a heavily-violated
    # design loses to any in-band one.
    score = eta_sys.copy()

    # D_min softness: linear ramp.  In-band → 1.0.  Below band → 1 − α·(D_min−D_vmax)/D_min.
    # α = 0.6 means D_vmax = 0 docks score by 60 %; D_vmax = D_min/2 docks by 30 %.
    if D_min > 0.0:
        deficit = np.clip((D_min - np.minimum(D_vmax, 1.0)) / D_min, 0.0, 1.0)
        dmin_softness = 1.0 - 0.6 * deficit
        score = score * dmin_softness.astype(score.dtype)

    # EMI softness: rank-relative — designs with lower TX peak current score better.
    # We don't know the absolute "good" current ahead of time, so we use a soft
    # exp-decay penalty: exp(-I_tx_pk / I_ref).  I_ref = 0.5 A is a reasonable
    # PCB-coil scale; tweak if needed.  This contributes a smooth 0–~30 % preference.
    I_ref_emi = 0.5
    emi_softness = 0.7 + 0.3 * np.exp(-I_tx_pk_vmax / I_ref_emi)
    score = score * emi_softness.astype(score.dtype)

    # Reserve-headroom softness: prefer designs whose natural full-drive V_dc at
    # V_min reaches V_cap_target. Below that the cap sags and the tag has reduced
    # reserve for the WPT-off IR-transmit window (a 480 B / 1 Mbaud burst pulls
    # ~10.5 mA from cap for 4.8 ms ⇒ ~0.55 V droop on a 100 µF cap; 4.5 V start
    # → 3.95 V end keeps the 3 V LDO out of dropout). Quadratic ramp from floor
    # to target so a design at 4.0 V (just at floor) loses ~21 % vs one at 4.5 V.
    reserve_softness = np.clip(V_dc_min_m / np.maximum(V_cap_target_dc, 1e-12),
                               0.0, 1.0) ** 2
    score = score * reserve_softness.astype(score.dtype)

    return (eta_sys.astype(np.float32),
            score.astype(np.float32),
            feasible,
            feasible_relaxed,
            D_vmax.astype(np.float32),
            D_vmin.astype(np.float32),
            V_dc_min_eff.astype(np.float32),
            f0_tx.astype(np.float32),
            f0_rx.astype(np.float32),
            f_op_lo.astype(np.float32),
            I_tx_pk_vmax.astype(np.float32),
            V_Crx_pk.astype(np.float32))


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


def _nn_sweep(params, log_cb, cancel_flag):
    """
    Sample N combinations, evaluate via NN, score with physics kernel.
    Returns dict of accumulator arrays for downstream ranking.

    All sampled geometry / frequency values are snapped to the physical grid
    before NN evaluation so the surrogate, the FastHenry refinement step,
    and the saved refined entries all agree on the same realisable values.
    """
    model_dir    = params["model_dir"]
    model, x_scaler, y_scaler, feat_cols, torch, device = _load_nn(model_dir)

    if not feat_cols:
        raise RuntimeError(
            "Surrogate scaler has no feature_names_in_; retrain with the new "
            "trainer so the inference path can match the training schema.")

    tx_od_max    = params["tx_od_max"]
    tx_id_min    = params["tx_id_min"]
    tx_width_min = params["tx_width_min"]
    tx_width_max = params["tx_width_max"]
    tx_turns_min = params["tx_turns_min"]

    rx_od_max    = params["rx_od_max"]
    rx_id_min    = params["rx_id_min"]
    rx_width_min = params["rx_width_min"]
    rx_width_max = params["rx_width_max"]
    rx_turns_min = params["rx_turns_min"]

    tx_topologies = params["tx_topologies"]
    rx_topologies = params["rx_topologies"]
    n_tx_topos    = len(tx_topologies)
    n_rx_topos    = len(rx_topologies)

    gc_enabled  = params["gc_enabled"]
    gc_dia_mm   = round(_snap(params["gc_dia_mm"], _GRID_OD_MM_STEP), 4)

    freq_min_hz = params["freq_min_hz"]   # user operating range (sweep start)
    freq_max_hz = params["freq_max_hz"]   # user operating range (sweep end)
    # freq_ref_hz = centre of training domain (fixed NN query point).
    # Passed explicitly so it stays anchored to dom_freq, not the user sweep range.
    freq_ref_hz = params.get("freq_ref_hz") or round(
        _snap(0.5 * (freq_min_hz + freq_max_hz), _GRID_FREQ_HZ_STEP), 1)

    V_min          = params["v_min"]
    V_max          = params["v_max"]
    P_target       = params["p_target_w"]
    D_min          = params["d_min"]
    D_max          = params["d_max"]
    V_cap_target    = params.get("v_cap_target_dc", _V_CAP_TARGET_DEF)
    V_ldo_out       = params.get("v_ldo_out", _V_LDO_OUT)
    V_rx_cap_rating = params.get("v_rx_cap_rating", _RX_CAP_V_RATING_DEF)

    # Run-fixed stackup constants (also used as NN inputs).
    pcb_gap_mm     = float(params["pcb_gap_mm"])
    tx_oz_layers   = list(params["tx_oz_per_layer"])
    rx_oz_layers   = list(params["rx_oz_per_layer"])
    tx_layers_full = list(params["tx_layers"])
    rx_layers_full = list(params["rx_layers"])

    # Pull port_inside choices from domain (boolean field; sample if both allowed).
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

    # Per-side per-layer copper thickness in metres (zero where inactive).
    tx_h_layers_m = [oz * _OZ_MM * 1e-3 for oz in tx_oz_layers]
    rx_h_layers_m = [oz * _OZ_MM * 1e-3 for oz in rx_oz_layers]
    tx_h_total_m  = sum(h for h in tx_h_layers_m if h > 0) or 1e-9

    # Per-RX-topology effective copper height for DCR.
    rx_h_eff_per_topo = {
        t: _rx_h_total_for_topology(t, rx_h_layers_m) for t in rx_topologies
    }

    N              = params["n_combos"]
    dither_amp_hz  = params.get("dither_amp_hz", 0.0)
    zvs_margin     = params.get("zvs_margin", 0.0)
    c_tx_options_f = params.get("c_tx_options_f", [])
    c_rx_options_f = params.get("c_rx_options_f", [])

    if not c_tx_options_f or not c_rx_options_f:
        raise ValueError("Cap option lists must be non-empty for stratified sampling.")

    ctx_arr   = np.array(c_tx_options_f, dtype=np.float64)
    crx_arr   = np.array(c_rx_options_f, dtype=np.float64)
    n_caps_tx = ctx_arr.size

    # ── Frequency sweep dimension (1 kHz steps across user-specified range) ──
    # The NN is always queried at freq_ref_hz (the training frequency).
    # For each operating frequency f_op in the sweep, AC resistance is scaled
    # by sqrt(f_op / freq_ref_hz) before the efficiency calculation.
    # This handles the fact that training data was collected at a fixed freq.
    freq_sweep_hz = np.arange(freq_min_hz, freq_max_hz + 0.5,
                              _GRID_FREQ_HZ_STEP, dtype=np.float64)
    n_freqs = freq_sweep_hz.size

    # C_rx is snapped per-geometry-per-freq to the nearest E-series cap that
    # resonates closest to f_op — it is NOT a free sweep variable.
    # Total combos per geometry = n_caps_tx × n_freqs.
    n_pairs = n_caps_tx * n_freqs

    if params.get("log_header", False):
        log_cb(f"  NN features ({len(feat_cols)}): {feat_cols}")
        log_cb(f"  TX topologies: {tx_topologies}  |  RX topologies: {rx_topologies}")
        log_cb(f"  Fixed: pcb_gap={pcb_gap_mm:.2f}mm  "
               f"TX oz={tx_oz_layers}  RX oz={rx_oz_layers}")
        log_cb(f"  C_tx options: {n_caps_tx}  |  C_rx: snapped to f0_rx≈f_op per geometry")
        log_cb(f"  Freq sweep: {freq_min_hz/1e3:.0f}–{freq_max_hz/1e3:.0f} kHz "
               f"({n_freqs} steps)  |  Dither ±{dither_amp_hz/1e3:.1f} kHz  "
               f"ZVS margin {zvs_margin*100:.0f}%")

    acc_keys = [
        "tx_turns", "tx_width", "tx_od_mm",
        "rx_od_mm", "rx_turns", "rx_width",
        "tx_topo", "rx_topo", "tx_port_inside", "rx_port_inside",
        "gc_dia_mm",
        "L_tx", "L_rx", "M",
        "R_tx", "R_rx", "DCR_tx", "DCR_rx",
        "tx_id_mm", "rx_id_mm",
        "eta_sys", "score", "feasible", "feasible_relaxed",
        "D_vmax", "D_vmin", "V_dc_min", "f0_tx_hz", "f0_rx_hz", "f_op_lo_hz",
        "I_tx_pk_vmax", "V_Crx_pk",
        "c_tx_nf", "c_rx_nf", "freq_hz",
    ]
    accs = {k: [] for k in acc_keys}

    n_ok  = 0
    n_rej = 0
    b     = 0
    rng   = np.random.default_rng(params.get("rng_seed", 42))

    # Pre-compute scaler stats so we can validate that every feature column is
    # actually populated by row_map (catches the silent-zero-fill bug).
    feat_cols_set = set(feat_cols)
    geom_per_batch = max(1, BATCH_SIZE // n_pairs)

    while n_ok < N:
        if cancel_flag.is_set():
            return None
        b += 1
        if b % 100 == 0:
            log_cb(f"    batch {b}: {n_ok:,}/{N:,} combos, {n_rej:,} rejected")

        bs = geom_per_batch

        # ── Sample geometry, then snap to physical grid ───────────────────
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

        tx_topo_idx_s = rng.integers(0, n_tx_topos, size=bs)
        rx_topo_idx_s = rng.integers(0, n_rx_topos, size=bs)

        tx_port_idx_s = rng.integers(0, len(tx_port_choices), size=bs)
        rx_port_idx_s = rng.integers(0, len(rx_port_choices), size=bs)

        gc_s = (np.full(bs, gc_dia_mm, dtype=np.float32)
                if gc_enabled
                else np.zeros(bs, dtype=np.float32))

        tx_id_s = _inner_diameter_mm(tx_od_s, tx_width_s, tx_turns_s)
        rx_id_s = _inner_diameter_mm(rx_od_s, rx_width_s, rx_turns_s)
        id_ok   = (tx_id_s >= tx_id_min) & (rx_id_s >= rx_id_min)
        n_rej  += int((~id_ok).sum()) * n_pairs
        if not id_ok.any():
            continue

        tx_turns_s = tx_turns_s[id_ok]; tx_od_s    = tx_od_s[id_ok]
        tx_width_s = tx_width_s[id_ok]; rx_od_s    = rx_od_s[id_ok]
        rx_turns_s = rx_turns_s[id_ok]; rx_width_s = rx_width_s[id_ok]
        tx_topo_idx_s = tx_topo_idx_s[id_ok]; rx_topo_idx_s = rx_topo_idx_s[id_ok]
        tx_port_idx_s = tx_port_idx_s[id_ok]; rx_port_idx_s = rx_port_idx_s[id_ok]
        gc_s       = gc_s[id_ok]
        tx_id_s    = tx_id_s[id_ok];    rx_id_s    = rx_id_s[id_ok]
        bs_v = int(id_ok.sum())

        tx_port_in_s = np.array([float(tx_port_choices[i]) for i in tx_port_idx_s],
                                dtype=np.float32)
        rx_port_in_s = np.array([float(rx_port_choices[i]) for i in rx_port_idx_s],
                                dtype=np.float32)

        # One-hot per fixed TOPOLOGY_VOCAB order so column names line up with
        # the trainer regardless of which subset the domain allows here.
        tx_topo_oh = {f"tx_topo_{name}": np.zeros(bs_v, dtype=np.float32)
                      for name in TOPOLOGY_VOCAB}
        rx_topo_oh = {f"rx_topo_{name}": np.zeros(bs_v, dtype=np.float32)
                      for name in TOPOLOGY_VOCAB}
        for ti, tname in enumerate(tx_topologies):
            mask = (tx_topo_idx_s == ti)
            if mask.any():
                tx_topo_oh[f"tx_topo_{tname}"][mask] = 1.0
        for ti, tname in enumerate(rx_topologies):
            mask = (rx_topo_idx_s == ti)
            if mask.any():
                rx_topo_oh[f"rx_topo_{tname}"][mask] = 1.0

        # ── DCR (TX uses lumped active-layer height; RX is per-topology) ──
        tx_len_m  = _spiral_length_m(tx_od_s, tx_width_s, tx_turns_s)
        DCR_tx_np = _dcr_ohm(tx_len_m, tx_width_s, tx_h_total_m).astype(np.float32)

        rx_len_m  = _spiral_length_m(rx_od_s, rx_width_s, rx_turns_s)
        DCR_rx_np = np.empty(bs_v, dtype=np.float32)
        for ti, tname in enumerate(rx_topologies):
            m = (rx_topo_idx_s == ti)
            if not m.any():
                continue
            h_eff = rx_h_eff_per_topo[tname]
            DCR_rx_np[m] = _dcr_ohm(rx_len_m[m], rx_width_s[m], h_eff)

        # ── Build feature row map covering EVERY trainer column ───────────
        # Run-fixed scalars become full-length vectors for stacking.
        row_map = {
            # TX geometry
            "tx_turns":       tx_turns_s,
            "tx_width":       tx_width_s,
            "tx_od_mm":       tx_od_s,
            "tx_spacing_mm":  np.full(bs_v, _SPACING_MM,                dtype=np.float32),
            "tx_outer_gap_mm": np.full(bs_v, float(tx_dom.get("outer_gap_mm", [0.2, 0.2])[0]),
                                       dtype=np.float32),
            "tx_inner_gap_mm": np.full(bs_v, float(tx_dom.get("inner_gap_mm", [1.0, 1.0])[0]),
                                       dtype=np.float32),
            # RX geometry
            "rx_turns":       rx_turns_s,
            "rx_width":       rx_width_s,
            "rx_od_mm":       rx_od_s,
            "rx_spacing_mm":  np.full(bs_v, _SPACING_MM,                dtype=np.float32),
            "rx_outer_gap_mm": np.full(bs_v, float(rx_dom.get("outer_gap_mm", [0.2, 0.2])[0]),
                                       dtype=np.float32),
            "rx_inner_gap_mm": np.full(bs_v, float(rx_dom.get("inner_gap_mm", [0.6, 0.6])[0]),
                                       dtype=np.float32),
            # Global
            "freq_hz":             np.full(bs_v, freq_ref_hz,  dtype=np.float32),
            "pcb_gap_mm":          np.full(bs_v, pcb_gap_mm,   dtype=np.float32),
            "ground_circle_dia_mm": gc_s,
            # Booleans
            "tx_port_inside": tx_port_in_s,
            "rx_port_inside": rx_port_in_s,
        }
        for i in range(4):
            row_map[f"tx_layer{i+1}_oz"] = np.full(bs_v, tx_oz_layers[i], dtype=np.float32)
            row_map[f"rx_layer{i+1}_oz"] = np.full(bs_v, rx_oz_layers[i], dtype=np.float32)
        row_map.update(tx_topo_oh)
        row_map.update(rx_topo_oh)

        # Validate up front that no scaler column gets a silent zero-fill.
        missing = [c for c in feat_cols if c not in row_map]
        if missing:
            raise RuntimeError(
                "Trained scaler expects feature columns the optimiser does not "
                f"produce: {missing}. Retrain the model with the new trainer "
                "or extend row_map to populate these columns explicitly.")

        X    = np.column_stack([row_map[c] for c in feat_cols]).astype(np.float32)
        X_sc = x_scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            Y_sc = model(torch.tensor(X_sc, device=device))
        Y = y_scaler.inverse_transform(Y_sc.cpu().numpy())

        # ── Stratified expansion: geometry × C_tx × freq_sweep ──────────────
        # Layout per geometry: n_freqs blocks of n_caps_tx rows.
        # For each (geometry, freq) pair, C_rx is snapped to the nearest
        # E-series cap that resonates closest to f_op (the operating freq).
        # NN was queried at freq_ref_hz; ACR is scaled by sqrt(f_op/freq_ref).

        L_rx_pred = Y[:, 1].astype(np.float64) * 1e-6   # µH → H, shape (bs_v,)
        crx_arr_nf = crx_arr * 1e9                        # nF for distance calc

        # Pre-build freq broadcast arrays (shape: bs_v × n_freqs).
        # C_rx ideal = 1 / ((2π f_op)² L_rx).
        freq_2pi_sq = (2.0 * math.pi * freq_sweep_hz) ** 2  # (n_freqs,)
        # C_rx ideal per (geom, freq): shape (bs_v, n_freqs)
        C_rx_ideal_mat = 1.0 / (freq_2pi_sq[None, :] *
                                 np.maximum(L_rx_pred[:, None], 1e-15))  # F
        C_rx_ideal_nf_mat = C_rx_ideal_mat * 1e9
        # Snap to nearest E-series: argmin over crx_arr dim → shape (bs_v, n_freqs)
        crx_idx_mat = np.argmin(
            np.abs(C_rx_ideal_nf_mat[:, :, None] - crx_arr_nf[None, None, :]),
            axis=2
        )
        crx_mat = crx_arr[crx_idx_mat]   # F, shape (bs_v, n_freqs)

        # Expand all arrays to shape (bs_v * n_freqs * n_caps_tx,).
        # Order: for each geometry, iterate freqs, then caps_tx.
        n_combos_batch = bs_v * n_pairs   # n_pairs = n_freqs * n_caps_tx

        # Geometry fields: repeat each geom value n_pairs times.
        def _rep(arr): return np.repeat(arr, n_pairs)

        L_tx_full   = _rep(Y[:, 0]).astype(np.float32)
        L_rx_full   = _rep(Y[:, 1]).astype(np.float32)
        M_full      = _rep(Y[:, 2]).astype(np.float32)
        R_tx_full   = _rep(Y[:, 3]).astype(np.float32)
        R_rx_full   = _rep(Y[:, 4]).astype(np.float32)
        DCR_tx_full = _rep(DCR_tx_np)
        DCR_rx_full = _rep(DCR_rx_np)

        # freq and C_rx: for each geometry, tile (freq × cap) in row-major order.
        # crx_mat row i → repeat each element n_caps_tx times, then tile.
        # Shape target: (bs_v, n_freqs, n_caps_tx) → flatten.
        # For each freq, n_caps_tx cap values: repeat each freq entry n_caps_tx times.
        freq_tile   = np.repeat(freq_sweep_hz, n_caps_tx)        # (n_freqs*n_caps_tx,)
        ctx_tile    = np.tile(ctx_arr, n_freqs)                   # (n_freqs*n_caps_tx,)
        # crx per (geom,freq): crx_mat[g, f] tiled n_caps_tx times each freq block.
        # crx_mat shape (bs_v, n_freqs) → repeat each freq entry n_caps_tx times
        crx_per_geom_freq = np.repeat(crx_mat, n_caps_tx, axis=1)  # (bs_v, n_freqs*n_caps_tx)

        freq_full = np.tile(freq_tile, bs_v).astype(np.float64)          # (n_combos_batch,)
        ctx_full  = np.tile(ctx_tile, bs_v).astype(np.float64)           # (n_combos_batch,)
        crx_full  = crx_per_geom_freq.ravel().astype(np.float64)         # (n_combos_batch,)

        f_drive_min_full = np.maximum(freq_full - dither_amp_hz, 1.0)

        eta_sys, score_arr, feasible, feasible_relaxed, D_vmax, D_vmin, V_dc_min, f0_tx_arr, f0_rx_arr, f_op_lo_arr, I_tx_pk_arr, V_Crx_pk_arr = _score_batch(
            L_tx_full, L_rx_full, M_full, R_tx_full, R_rx_full,
            DCR_tx_full, DCR_rx_full,
            ctx_full, crx_full,
            freq_ref_hz, freq_full, f_drive_min_full,
            V_min, V_max, P_target, D_min, D_max,
            zvs_margin, _V_MIN_INDUCED_DC, _V_ZENER_DC,
            V_cap_target, V_ldo_out, V_rx_cap_rating,
        )
        # Don't zero eta_sys here — store raw, and apply strict-or-relaxed mask
        # after the full sweep so the D_min preference can be relaxed globally.

        n_ok += n_combos_batch

        accs["tx_turns"].append(_rep(tx_turns_s))
        accs["tx_width"].append(_rep(tx_width_s))
        accs["tx_od_mm"].append(_rep(tx_od_s))
        accs["rx_od_mm"].append(_rep(rx_od_s))
        accs["rx_turns"].append(_rep(rx_turns_s))
        accs["rx_width"].append(_rep(rx_width_s))
        accs["tx_topo"].append(_rep(tx_topo_idx_s.astype(np.uint8)))
        accs["rx_topo"].append(_rep(rx_topo_idx_s.astype(np.uint8)))
        accs["tx_port_inside"].append(_rep(tx_port_in_s))
        accs["rx_port_inside"].append(_rep(rx_port_in_s))
        accs["gc_dia_mm"].append(_rep(gc_s))
        accs["L_tx"].append(L_tx_full)
        accs["L_rx"].append(L_rx_full)
        accs["M"].append(M_full)
        accs["R_tx"].append(R_tx_full)
        accs["R_rx"].append(R_rx_full)
        accs["DCR_tx"].append(DCR_tx_full)
        accs["DCR_rx"].append(DCR_rx_full)
        accs["tx_id_mm"].append(_rep(tx_id_s))
        accs["rx_id_mm"].append(_rep(rx_id_s))
        accs["eta_sys"].append(eta_sys)
        accs["score"].append(score_arr)
        accs["feasible"].append(feasible)
        accs["feasible_relaxed"].append(feasible_relaxed)
        accs["D_vmax"].append(D_vmax)
        accs["D_vmin"].append(D_vmin)
        accs["V_dc_min"].append(V_dc_min)
        accs["f0_tx_hz"].append(f0_tx_arr)
        accs["f0_rx_hz"].append(f0_rx_arr)
        accs["f_op_lo_hz"].append(f_op_lo_arr.astype(np.float32))
        accs["I_tx_pk_vmax"].append(I_tx_pk_arr)
        accs["V_Crx_pk"].append(V_Crx_pk_arr)
        accs["c_tx_nf"].append((ctx_full * 1e9).astype(np.float32))
        accs["c_rx_nf"].append((crx_full * 1e9).astype(np.float32))
        accs["freq_hz"].append(freq_full.astype(np.float32))

    concat = {k: np.concatenate(v) for k, v in accs.items()}
    concat["_tx_topologies"] = np.array(tx_topologies)
    concat["_rx_topologies"] = np.array(rx_topologies)
    concat["_freq_ref_hz"]   = np.float64(freq_ref_hz)
    concat["_freq_min_hz"]   = np.float64(freq_min_hz)
    concat["_freq_max_hz"]   = np.float64(freq_max_hz)

    # Hard gates (P_min, D_max, ZVS, V_floor) define the relaxed mask. The D_min
    # preference is folded into `score` as a soft penalty, so we mask `score`
    # (the ranking metric) by the relaxed mask and leave `eta_sys` (the displayed
    # physical efficiency) untouched.
    relaxed_mask = concat["feasible_relaxed"]
    strict_mask  = concat["feasible"]
    n_strict     = int(np.count_nonzero(strict_mask))
    n_feasible   = int(np.count_nonzero(relaxed_mask))
    concat["score"]   = np.where(relaxed_mask, concat["score"],   np.float32(0.0))
    concat["eta_sys"] = np.where(relaxed_mask, concat["eta_sys"], np.float32(0.0))
    del concat["feasible"], concat["feasible_relaxed"]

    relaxed_note = "" if n_strict > 0 else "  [D_min preference unmet by every design — best-effort fallback]"
    log_cb(f"  Sweep done: {n_ok:,} combos scored "
           f"({n_feasible:,} feasible incl. {n_strict:,} strict, {n_rej:,} geometry-rejected){relaxed_note}")
    return concat


# ─────────────────────────────────────────────────────────────────────────────
# Pick top-K unique configurations from sweep result
# ─────────────────────────────────────────────────────────────────────────────

def _top_k(result, tx_topologies, rx_topologies, k=12):
    """
    Pick top-K unique configurations from sweep results.

    Dedup key includes both topologies and port_inside booleans, since these
    materially affect FastHenry results and the surrogate sees them as inputs.
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
        rx_od = round(float(result["rx_od_mm"][i]), 1)
        rx_w  = round(float(result["rx_width"][i]), 2)
        tx_t_idx = int(result["tx_topo"][i])
        rx_t_idx = int(result["rx_topo"][i])
        tx_topo  = tx_topologies[tx_t_idx] if tx_t_idx < len(tx_topologies) else "parallel"
        rx_topo  = rx_topologies[rx_t_idx] if rx_t_idx < len(rx_topologies) else "parallel"
        tx_port  = bool(result["tx_port_inside"][i])
        rx_port  = bool(result["rx_port_inside"][i])
        gc       = round(float(result["gc_dia_mm"][i]), 1)
        f_op_khz = round(float(result["freq_hz"][i]) / 1e3, 1)
        key = (tx_t, tx_od, tx_w, rx_t, rx_od, rx_w,
               tx_topo, rx_topo, tx_port, rx_port, gc, f_op_khz)
        if key in seen:
            continue
        seen.add(key)
        winners.append({
            "tx_turns":       tx_t,
            "tx_width":       float(result["tx_width"][i]),
            "tx_od_mm":       float(result["tx_od_mm"][i]),
            "tx_topology":    tx_topo,
            "tx_port_inside": tx_port,
            "rx_turns":       rx_t,
            "rx_width":       float(result["rx_width"][i]),
            "rx_od_mm":       float(result["rx_od_mm"][i]),
            "rx_topology":    rx_topo,
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
            "freq_hz":        float(result["freq_hz"][i]),
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
    Build FastHenry SimParams from winning combos.

    pcb_gap_mm and per-layer copper stackup come from the run's fixed config
    (the user picks them once per optimisation run); gaps come from domain
    (constant per training set); other geometry / topology / port_inside
    fields come from the winning combination so each FastHenry sim is run
    on the exact configuration the surrogate scored.
    """
    from parallel_sim import SimParams

    domain     = params["domain"]
    glb        = domain.get("global", {})

    tx_d = domain.get("tx", {})
    rx_d = domain.get("rx", {})

    pcb_gap_mm = float(params["pcb_gap_mm"])
    tx_layers  = list(params["tx_layers"])
    rx_layers  = list(params["rx_layers"])

    params_list = []
    for w in winners:
        # Use the winner's own operating frequency for FastHenry so the sim
        # runs at the actual frequency the optimizer chose, not the training centre.
        w_freq_hz = round(_snap(float(w.get("freq_hz", glb.get("freq_hz", [360000.0])[0])),
                                _GRID_FREQ_HZ_STEP), 1)
        p = SimParams(
            tx_turns          = int(w["tx_turns"]),
            tx_trace_width_mm = round(float(w["tx_width"]), 4),
            tx_od_mm          = round(float(w["tx_od_mm"]), 4),
            tx_spacing_mm     = _SPACING_MM,
            tx_outer_gap_mm   = float(tx_d.get("outer_gap_mm", [0.2, 0.2])[0]),
            tx_inner_gap_mm   = float(tx_d.get("inner_gap_mm", [1.0, 1.0])[0]),
            tx_topology       = w["tx_topology"],
            tx_layers         = tx_layers,
            tx_port_inside    = bool(w["tx_port_inside"]),

            rx_turns          = int(w["rx_turns"]),
            rx_trace_width_mm = round(float(w["rx_width"]), 4),
            rx_od_mm          = round(float(w["rx_od_mm"]), 4),
            rx_spacing_mm     = _SPACING_MM,
            rx_outer_gap_mm   = float(rx_d.get("outer_gap_mm", [0.2, 0.2])[0]),
            rx_inner_gap_mm   = float(rx_d.get("inner_gap_mm", [0.6, 0.6])[0]),
            rx_topology       = w["rx_topology"],
            rx_layers         = rx_layers,
            rx_port_inside    = bool(w["rx_port_inside"]),

            tx_nhinc          = int(tx_d.get("nhinc", 1)),
            tx_nwinc          = int(tx_d.get("nwinc", 3)),
            rx_nhinc          = int(rx_d.get("nhinc", 1)),
            rx_nwinc          = int(rx_d.get("nwinc", 3)),

            pcb_gap_mm        = pcb_gap_mm,
            freq_hz           = w_freq_hz,
            fmin_hz           = w_freq_hz,
            fmax_hz           = w_freq_hz,
            resolution_mm     = float(glb.get("resolution_mm", 1.5)),
            timeout_sec       = timeout_sec,
            ground_circle_dia_mm = round(float(w.get("gc_dia_mm", 0.0)), 1),
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

    for i, (sr, _w) in enumerate(zip(sim_results, winners)):
        if sr is None or "error" in sr:
            log_cb(f"    sim {i+1}: FAIL — {(sr or {}).get('error','?')}")
            continue

        row = dict(sr)  # passthrough: freq_hz, L_*, R_*, Q_*, M, k, geometry,
                        # layers, port_inside, pcb_gap_mm, ground_circle_dia_mm,
                        # pid, elapsed_sec — all already rounded by the sim.
        row["uuid"]             = str(uuid.uuid4())
        row["hasGroundCircle"]  = bool(round(float(sr.get("ground_circle_dia_mm", 0.0)), 3) > 0.0)

        if row["uuid"] not in existing_uuids:
            data["results"].append(row)
            existing_uuids.add(row["uuid"])
            added += 1
        log_cb(f"    sim {i+1}: OK  L_tx={sr.get('L_tx_uH',0):.2f}µH  "
               f"L_rx={sr.get('L_rx_uH',0):.2f}µH  "
               f"M={sr.get('M_uH',0):.3f}µH  k={sr.get('k',0):.4f}")

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

def _retrain(model_dir, refined_path, train_params, log_cb, cancel_flag):
    env = os.environ.copy()
    env["SURROGATE_DATA"]       = refined_path
    env["SURROGATE_OUTPUT_DIR"] = model_dir
    env["SURROGATE_EPOCHS"]     = str(train_params["epochs"])
    env["SURROGATE_BATCH_SIZE"] = str(train_params["batch"])
    env["SURROGATE_LR"]         = str(train_params["lr"])
    env["SURROGATE_VAL_SPLIT"]  = str(train_params["val_split"])

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
            if stripped.startswith("Using device:"):
                continue
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

def _run_optimisation(params, progress_cb, log_cb, done_cb, cancel_flag):
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

        tx_topologies = params["tx_topologies"]
        rx_topologies = params["rx_topologies"]
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
            sweep_seed_params["log_header"] = (iteration == 1)
            sweep_result = _nn_sweep(sweep_seed_params, log_cb, cancel_flag)
            if sweep_result is None:
                log_cb("Cancelled during sweep.")
                done_cb(None)
                return

            # --- Pick top-K ---
            winners = _top_k(sweep_result, tx_topologies, rx_topologies, k=top_k)
            if not winners:
                log_cb("  No feasible combinations found — stopping.")
                break

            top1     = winners[0]
            _f_op     = top1.get("freq_hz", 0.0)
            _f0_tx    = top1.get("f0_tx_hz", 0.0)
            _f0_rx    = top1.get("f0_rx_hz", 0.0)
            _f_op_lo  = top1.get("f_op_lo_hz", 0.0)
            _zvs_x    = (_f_op_lo / _f0_tx) if _f0_tx > 0 else 0.0
            _rx_off   = ((_f0_rx - _f_op) / 1e3) if (_f0_rx > 0 and _f_op > 0) else 0.0
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
            log_cb(f"  f_op={_f_op/1e3:.1f}kHz  "
                   f"f0_tx={_f0_tx/1e3:.1f}kHz  f_op_lo={_f_op_lo/1e3:.1f}kHz "
                   f"(ZVS×{_zvs_x:.2f})  |  "
                   f"f0_rx={_f0_rx/1e3:.1f}kHz  Δf_rx(vs f_op)={_rx_off:+.1f}kHz")
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
            log_cb(f"\nPhase 2: Running physics simulation on top {len(winners)} results")
            sim_params_list = _build_sim_params(winners, params, fh_timeout)

            from parallel_sim import run_batch as _run_batch
            sim_counter = [0]

            def _prog(done, total, _i=iteration, _mi=max_iters):
                sim_counter[0] = done
                log_cb(f"    FastHenry: {done}/{total}")
                frac = ((_i - 1) + (done / max(total, 1)) * 0.5) / _mi
                progress_cb(frac * 0.9)

            sim_results = _run_batch(sim_params_list,
                                     max_workers=fh_workers,
                                     progress_cb=_prog)

            ok_count = sum(1 for r in sim_results if r and "error" not in r)
            log_cb(f"  FastHenry: {ok_count}/{len(sim_results)} OK")

            # --- Append to refined ---
            log_cb("\nPhase 3: Appending to refined_results.json…")
            _append_to_refined(sim_results, winners, refined_path, log_cb)

            # --- Retrain ---
            log_cb("\nPhase 4: Retraining model…")
            ok = _retrain(model_dir, refined_path, train_params, log_cb, cancel_flag)
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
        final_result = _nn_sweep(final_seed_params, log_cb, cancel_flag)
        if final_result is not None and not cancel_flag.is_set():
            final_winners = _top_k(final_result, tx_topologies, rx_topologies, k=top_k)
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
            and int(a["rx_turns"]) == int(b["rx_turns"])
            and round(a["tx_od_mm"], 1) == round(b["tx_od_mm"], 1)
            and round(a["rx_od_mm"], 1) == round(b["rx_od_mm"], 1)
            and round(a["tx_width"], 2) == round(b["tx_width"], 2)
            and round(a["rx_width"], 2) == round(b["rx_width"], 2)
            and a["rx_topology"] == b["rx_topology"]
            and round(a.get("freq_hz", 0.0) / 1e3, 1) == round(b.get("freq_hz", 0.0) / 1e3, 1))


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

        # ── RX topology selection ─────────────────────────────────────────────
        rx_topo_f = ttk.LabelFrame(col_l, text="RX Topologies to Consider")
        rx_topo_f.pack(fill="x", pady=(0, 6))
        self._rx_topo_parallel     = tk.BooleanVar(value=True)
        self._rx_topo_series       = tk.BooleanVar(value=True)
        self._rx_topo_par_ser_pairs = tk.BooleanVar(value=True)
        rx_topo_row = ttk.Frame(rx_topo_f)
        rx_topo_row.pack(anchor="w", padx=6, pady=(4, 4))
        ttk.Checkbutton(rx_topo_row, text="All Parallel",
                        variable=self._rx_topo_parallel).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(rx_topo_row, text="All Series",
                        variable=self._rx_topo_series).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(rx_topo_row, text="Parallel-Series Pairs",
                        variable=self._rx_topo_par_ser_pairs).pack(side="left")

        # ── Ground circle ────────────────────────────────────────────────────
        gc_f = ttk.LabelFrame(col_l, text="Ground Circle")
        gc_f.pack(fill="x", pady=(0, 6))
        self._gc_enabled = tk.BooleanVar(value=DEFAULT_GC_ENABLED)
        gc_row = ttk.Frame(gc_f)
        gc_row.pack(fill="x", padx=6, pady=(4, 4))
        ttk.Checkbutton(gc_row, text="Enable Ground Circle (fixed diameter)",
                        variable=self._gc_enabled,
                        command=self._on_gc_toggle).pack(side="left")
        self._gc_dia = tk.StringVar(value="")
        ttk.Label(gc_row, text="  GC diameter (mm):", anchor="w").pack(side="left")
        self._gc_dia_entry = ttk.Entry(gc_row, textvariable=self._gc_dia, width=8)
        self._gc_dia_entry.pack(side="left", padx=(2, 0))
        self._on_gc_toggle()

        # ── Fixed stackup & PCB gap for this run ─────────────────────────────
        st_f = ttk.LabelFrame(col_l, text="PCB Gap & Layer Stackup (fixed for run)")
        st_f.pack(fill="x", pady=(0, 6))

        gap_row = ttk.Frame(st_f); gap_row.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Label(gap_row, text="PCB gap (mm):", width=22, anchor="w").pack(side="left")
        self._pcb_gap_mm = tk.StringVar(value=DEFAULT_PCB_GAP_MM)
        ttk.Entry(gap_row, textvariable=self._pcb_gap_mm, width=8).pack(side="left")

        ttk.Label(st_f, text="Copper oz per layer (0 = layer inactive):",
                  foreground="gray", font=("TkDefaultFont", 8)
                  ).pack(anchor="w", padx=6, pady=(4, 2))

        self._tx_oz_vars = []
        tx_row = ttk.Frame(st_f); tx_row.pack(fill="x", padx=6, pady=2)
        ttk.Label(tx_row, text="TX  L1 / L2 / L3 / L4:", width=22, anchor="w").pack(side="left")
        for i in range(4):
            v = tk.StringVar(value=DEFAULT_TX_LAYER_OZ[i])
            self._tx_oz_vars.append(v)
            ttk.Entry(tx_row, textvariable=v, width=5).pack(side="left", padx=1)

        self._rx_oz_vars = []
        rx_row = ttk.Frame(st_f); rx_row.pack(fill="x", padx=6, pady=2)
        ttk.Label(rx_row, text="RX  L1 / L2 / L3 / L4:", width=22, anchor="w").pack(side="left")
        for i in range(4):
            v = tk.StringVar(value=DEFAULT_RX_LAYER_OZ[i])
            self._rx_oz_vars.append(v)
            ttk.Entry(rx_row, textvariable=v, width=5).pack(side="left", padx=1)

        # ── NN model folder (from NN Setup tab) ───────────────────────────────
        nn_f = ttk.LabelFrame(col_l, text="NN Model")
        nn_f.pack(fill="x", pady=(0, 6))
        mrow = ttk.Frame(nn_f); mrow.pack(fill="x", padx=6, pady=4)
        ttk.Label(mrow, text="Folder:", foreground="gray").pack(side="left")
        self._model_label_var = tk.StringVar(value="(select in NN Setup tab)")
        ttk.Label(mrow, textvariable=self._model_label_var,
                  foreground="#1a6fcc", font=("Consolas", 8),
                  wraplength=340, justify="left").pack(side="left", padx=6)

        nav_row = ttk.Frame(col_l); nav_row.pack(fill="x", pady=(0, 6))
        ttk.Button(nav_row, text="Reset to Defaults",
                   command=self._on_reset_defaults).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(nav_row, text="Next Tab →  (NN Analysis)",
                   command=self._on_next_tab_click).pack(side="left", expand=True, fill="x", padx=(3, 0))

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
        self._p_target_mw = self._row_inline(ev_f, "Target RX power (mW):",
                                             DEFAULT_P_TARGET_MW, "Post Figure LDO Power")
        self._v_cap_target = self._row(ev_f, "Cap target voltage (V):", DEFAULT_V_CAP_TARGET)
        ttk.Separator(ev_f, orient="horizontal").pack(fill="x", padx=6, pady=(2, 2))
        self._v_min       = self._row(ev_f, "TX V_min (V):", DEFAULT_V_MIN)
        self._v_max       = self._row(ev_f, "TX V_max (V):", DEFAULT_V_MAX)
        ttk.Separator(ev_f, orient="horizontal").pack(fill="x", padx=6, pady=(2, 2))
        self._d_min_pct   = self._row(ev_f, "Min duty cycle (%):", DEFAULT_D_MIN_PCT)
        self._d_max_pct   = self._row(ev_f, "Max duty cycle (%):", DEFAULT_D_MAX_PCT)
        ttk.Label(ev_f, text="Min@Vmax: power-limiting floor. Max@Vmin: envelope-mod headroom. Blank = no limit.",
                  foreground="gray", font=("TkDefaultFont", 8),
                  wraplength=340, justify="left"
                  ).pack(anchor="w", padx=6, pady=(0, 2))
        self._freq_min_khz   = self._row(ev_f, "Freq range min (kHz):", DEFAULT_FREQ_MIN_KHZ)
        self._freq_max_khz   = self._row(ev_f, "Freq range max (kHz):", DEFAULT_FREQ_MAX_KHZ)
        ttk.Label(ev_f, text="Operating freq range swept at 1 kHz steps. NN queries at domain centre.",
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
        tr_row1 = ttk.Frame(tr_f); tr_row1.pack(fill="x", padx=6, pady=3)
        ttk.Label(tr_row1, text="Epochs:", width=12, anchor="w").pack(side="left")
        self._tr_epochs = tk.StringVar(value=DEFAULT_TR_EPOCHS)
        ttk.Entry(tr_row1, textvariable=self._tr_epochs, width=8).pack(side="left", padx=(0, 12))
        ttk.Label(tr_row1, text="Batch size:", width=12, anchor="w").pack(side="left")
        self._tr_batch = tk.StringVar(value=DEFAULT_TR_BATCH)
        ttk.Entry(tr_row1, textvariable=self._tr_batch, width=8).pack(side="left")
        tr_row2 = ttk.Frame(tr_f); tr_row2.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(tr_row2, text="LR:", width=12, anchor="w").pack(side="left")
        self._tr_lr = tk.StringVar(value=DEFAULT_TR_LR)
        ttk.Entry(tr_row2, textvariable=self._tr_lr, width=8).pack(side="left", padx=(0, 12))
        ttk.Label(tr_row2, text="Val split:", width=12, anchor="w").pack(side="left")
        self._tr_val_split = tk.StringVar(value=DEFAULT_TR_VAL_SPLIT)
        ttk.Entry(tr_row2, textvariable=self._tr_val_split, width=8).pack(side="left")

        # ── Control / progress / log ──────────────────────────────────────────
        run_lf = ttk.LabelFrame(col_r, text="Optimisation Control")
        run_lf.pack(fill="x", pady=(0, 6))
        btn_row = ttk.Frame(run_lf); btn_row.pack(fill="x", padx=6, pady=(6, 2))
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
        domain = _load_domain(model_dir)
        self._domain = domain
        self._model_label_var.set(self._short_path(model_dir))

        if domain is None:
            self._domain_info_var.set(
                "ERROR: domain.json missing or corrupt — model is unusable.")
            self._domain_info_lbl.configure(foreground="red")
            for v in (self._tx_od_max, self._tx_id_min,
                      self._rx_od_max, self._rx_id_min, self._gc_dia):
                v.set("")
            return

        self._domain_info_lbl.configure(foreground="#1a6fcc")

        tx  = domain.get("tx", {})
        rx  = domain.get("rx", {})
        glb = domain.get("global", {})

        self._tx_od_max.set(f"{tx.get('od_max_mm'):.1f}")
        self._tx_id_min.set(f"{tx.get('id_min_mm'):.1f}")
        self._rx_od_max.set(f"{rx.get('od_max_mm'):.1f}")
        self._rx_id_min.set(f"{rx.get('id_min_mm'):.1f}")

        gc_en  = domain.get("ground_circle_enabled", False)
        gc_min = domain.get("ground_circle_min_mm", 18.0)
        gc_max = domain.get("ground_circle_max_mm", 24.0)
        gc_def = 20.0 if gc_min <= 20.0 <= gc_max else gc_min
        self._gc_dia.set(f"{gc_def:.1f}")
        self._gc_enabled.set(gc_en)
        self._on_gc_toggle()

        freq      = glb.get("freq_hz", [0, 0])
        tx_topos  = tx.get("allowed_topologies", [])
        rx_topos  = rx.get("allowed_topologies", [])
        tx_turns  = tx.get("turns", ["-", "-"])
        rx_turns  = rx.get("turns", ["-", "-"])
        info = (
            f"TX: OD ≤ {tx.get('od_max_mm')} mm | ID ≥ {tx.get('id_min_mm')} mm | "
            f"turns {tx_turns[0]}–{tx_turns[1]} | topos: {', '.join(tx_topos)}\n"
            f"RX: OD ≤ {rx.get('od_max_mm')} mm | ID ≥ {rx.get('id_min_mm')} mm | "
            f"turns {rx_turns[0]}–{rx_turns[1]} | topos: {', '.join(rx_topos)}\n"
            f"Freq: {freq[0]/1e3:.0f}–{freq[1]/1e3:.0f} kHz"
            + (f"  |  GC: {gc_min}–{gc_max} mm" if gc_en else "")
        )
        self._domain_info_var.set(info)

    # ─────────────────────────────────────────────────────────────────────────
    # Event handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_gc_toggle(self):
        state = "normal" if self._gc_enabled.get() else "disabled"
        try:
            self._gc_dia_entry.configure(state=state)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Savestate
    # ─────────────────────────────────────────────────────────────────────────

    def _savestate_vars(self):
        """Return list of (key, StringVar/BooleanVar) pairs for all persistent inputs."""
        return [
            ("p_target_mw",     self._p_target_mw),
            ("v_cap_target",    self._v_cap_target),
            ("rx_cap_vrating",  self._rx_cap_vrating),
            ("v_min",           self._v_min),
            ("v_max",           self._v_max),
            ("d_min_pct",       self._d_min_pct),
            ("d_max_pct",       self._d_max_pct),
            ("freq_min_khz",    self._freq_min_khz),
            ("freq_max_khz",    self._freq_max_khz),
            ("dither_amp_khz",  self._dither_amp_khz),
            ("zvs_margin_pct",  self._zvs_margin_pct),
            ("tx_caps_nf",      self._tx_caps_nf),
            ("rx_caps_nf",      self._rx_caps_nf),
            ("n_combos",        self._n_combos),
            ("max_iters",       self._max_iters),
            ("top_k",           self._top_k),
            ("fh_workers",      self._fh_workers),
            ("fh_timeout",      self._fh_timeout),
            ("tr_epochs",       self._tr_epochs),
            ("tr_batch",        self._tr_batch),
            ("tr_lr",           self._tr_lr),
            ("tr_val_split",    self._tr_val_split),
            ("pcb_gap_mm",      self._pcb_gap_mm),
            ("gc_enabled",      self._gc_enabled),
            ("gc_dia",          self._gc_dia),
            ("tx_od_max",       self._tx_od_max),
            ("tx_id_min",       self._tx_id_min),
            ("rx_od_max",       self._rx_od_max),
            ("rx_id_min",       self._rx_id_min),
            ("rx_topo_parallel",      self._rx_topo_parallel),
            ("rx_topo_series",        self._rx_topo_series),
            ("rx_topo_par_ser_pairs", self._rx_topo_par_ser_pairs),
        ] + [(f"tx_oz_{i}", self._tx_oz_vars[i]) for i in range(4)] \
          + [(f"rx_oz_{i}", self._rx_oz_vars[i]) for i in range(4)]

    def _save_state(self):
        if self.app is None:
            return
        st = {}
        for key, var in self._savestate_vars():
            st[key] = var.get()
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
        self._on_gc_toggle()

    def _on_reset_defaults(self):
        _cap_default = ", ".join(f"{c:g}" for c in E_VALUES_NF)
        defaults = {
            "p_target_mw":     DEFAULT_P_TARGET_MW,
            "v_cap_target":    DEFAULT_V_CAP_TARGET,
            "rx_cap_vrating":  DEFAULT_RX_CAP_VRATING,
            "v_min":           DEFAULT_V_MIN,
            "v_max":           DEFAULT_V_MAX,
            "d_min_pct":       DEFAULT_D_MIN_PCT,
            "d_max_pct":       DEFAULT_D_MAX_PCT,
            "freq_min_khz":    DEFAULT_FREQ_MIN_KHZ,
            "freq_max_khz":    DEFAULT_FREQ_MAX_KHZ,
            "dither_amp_khz":  DEFAULT_DITHER_AMP_KHZ,
            "zvs_margin_pct":  DEFAULT_ZVS_MARGIN_PCT,
            "tx_caps_nf":      _cap_default,
            "rx_caps_nf":      _cap_default,
            "n_combos":        DEFAULT_N_COMBOS_M,
            "max_iters":       DEFAULT_MAX_ITERS,
            "top_k":           DEFAULT_TOP_K,
            "fh_workers":      DEFAULT_FH_WORKERS,
            "fh_timeout":      DEFAULT_FH_TIMEOUT_S,
            "tr_epochs":       DEFAULT_TR_EPOCHS,
            "tr_batch":        DEFAULT_TR_BATCH,
            "tr_lr":           DEFAULT_TR_LR,
            "tr_val_split":    DEFAULT_TR_VAL_SPLIT,
            "pcb_gap_mm":      DEFAULT_PCB_GAP_MM,
            "gc_enabled":      DEFAULT_GC_ENABLED,
            "gc_dia":          "",
            "tx_od_max":       "",
            "tx_id_min":       "",
            "rx_od_max":       "",
            "rx_id_min":       "",
            "rx_topo_parallel":      True,
            "rx_topo_series":        True,
            "rx_topo_par_ser_pairs": True,
        }
        for i in range(4):
            defaults[f"tx_oz_{i}"] = DEFAULT_TX_LAYER_OZ[i]
            defaults[f"rx_oz_{i}"] = DEFAULT_RX_LAYER_OZ[i]
        for key, var in self._savestate_vars():
            if key in defaults:
                var.set(defaults[key])
        self._on_gc_toggle()
        self._save_state()

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

        tx  = self._domain.get("tx", {})
        rx  = self._domain.get("rx", {})
        glb = self._domain.get("global", {})

        dom_tx_od_max = tx.get("od_max_mm", 999.0)
        dom_tx_id_min = tx.get("id_min_mm", 0.0)
        dom_rx_od_max = rx.get("od_max_mm", 999.0)
        dom_rx_id_min = rx.get("id_min_mm", 0.0)
        dom_tx_w      = tx.get("trace_width_mm", [0.2, 1.2])
        dom_rx_w      = rx.get("trace_width_mm", [0.2, 1.2])
        dom_tx_turns  = tx.get("turns", [3, 99])
        dom_rx_turns  = rx.get("turns", [3, 99])
        dom_freq      = glb.get("freq_hz", [340000.0, 380000.0])
        rx_topos_dom  = rx.get("allowed_topologies") or ["parallel", "series", "parallel_pairs_ser"]
        _rx_topo_map  = [
            ("parallel",          self._rx_topo_parallel),
            ("series",            self._rx_topo_series),
            ("parallel_pairs_ser", self._rx_topo_par_ser_pairs),
        ]
        rx_topos = [t for t, var in _rx_topo_map if var.get() and t in rx_topos_dom]
        if not rx_topos:
            raise ValueError(
                "At least one RX topology must be selected.")

        tx_od_max = min(flt(self._tx_od_max, "TX OD max", lo=10.0), dom_tx_od_max)
        tx_id_min = max(flt(self._tx_id_min, "TX ID min", lo=0.0), dom_tx_id_min)
        if tx_id_min >= tx_od_max:
            raise ValueError("TX ID min must be < TX OD max.")

        rx_od_max = min(flt(self._rx_od_max, "RX OD max", lo=10.0), dom_rx_od_max)
        rx_id_min = max(flt(self._rx_id_min, "RX ID min", lo=0.0), dom_rx_id_min)
        if rx_id_min >= rx_od_max:
            raise ValueError("RX ID min must be < RX OD max.")

        gc_enabled = self._gc_enabled.get()
        gc_dia_mm  = flt(self._gc_dia, "GC diameter", lo=0.0) if gc_enabled else 0.0

        p_target_w     = flt(self._p_target_mw, "Target RX power", lo=0.001) / 1000.0
        v_cap_target_dc  = flt(self._v_cap_target,   "Cap target voltage",
                               lo=_V_MIN_INDUCED_DC, hi=_V_ZENER_DC)
        v_rx_cap_rating  = flt(self._rx_cap_vrating, "C_rx rated voltage", lo=5.0, hi=1000.0)
        v_min          = flt(self._v_min, "V_min", lo=0.1)
        v_max          = flt(self._v_max, "V_max", lo=v_min)
        d_min_pct  = flt(self._d_min_pct, "Min duty %", lo=0.0, allow_empty=True)
        d_min      = d_min_pct / 100.0
        d_max_pct  = flt(self._d_max_pct, "Max duty %", lo=0.0, hi=100.0, allow_empty=True)
        d_max      = (d_max_pct / 100.0) if d_max_pct < 100.0 else 1.0

        # User-specified operating frequency range (independent of training domain).
        # The NN is always queried at freq_ref_hz = centre of dom_freq (training freq).
        # The sweep runs at 1 kHz steps between freq_min_hz and freq_max_hz.
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

        # PCB gap + per-layer copper-oz stackup (fixed for this run; the
        # surrogate still receives them as inputs so it stays consistent
        # with the training data).
        pcb_gap_mm = flt(self._pcb_gap_mm, "PCB gap", lo=0.1, hi=10.0)
        # Snap to 0.5 oz; 0 oz means inactive layer.
        def _parse_oz(var, name):
            try:
                v = float(var.get().strip())
            except ValueError:
                raise ValueError(f"'{name}' must be a number (0, 0.5, 1.0, ...).")
            if v < 0:
                raise ValueError(f"'{name}' cannot be negative.")
            v_snap = round(v / _GRID_OZ_STEP) * _GRID_OZ_STEP
            return round(v_snap, 4)

        tx_oz_per_layer = [_parse_oz(self._tx_oz_vars[i], f"TX L{i+1}") for i in range(4)]
        rx_oz_per_layer = [_parse_oz(self._rx_oz_vars[i], f"RX L{i+1}") for i in range(4)]
        if not any(o > 0 for o in tx_oz_per_layer):
            raise ValueError("At least one TX layer must be active (oz > 0).")
        if not any(o > 0 for o in rx_oz_per_layer):
            raise ValueError("At least one RX layer must be active (oz > 0).")

        tx_layers = [{"active": (o > 0), "copper_oz": (o if o > 0 else 1.0)}
                     for o in tx_oz_per_layer]
        rx_layers = [{"active": (o > 0), "copper_oz": (o if o > 0 else 1.0)}
                     for o in rx_oz_per_layer]

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

        tx_topos_dom = tx.get("allowed_topologies") or ["parallel", "series", "parallel_pairs_ser"]

        return dict(
            model_dir=model_dir,
            tx_od_max=tx_od_max, tx_id_min=tx_id_min,
            tx_width_min=dom_tx_w[0], tx_width_max=dom_tx_w[1],
            tx_turns_min=dom_tx_turns[0],
            rx_od_max=rx_od_max, rx_id_min=rx_id_min,
            rx_width_min=dom_rx_w[0], rx_width_max=dom_rx_w[1],
            rx_turns_min=dom_rx_turns[0],
            tx_topologies=tx_topos_dom,
            rx_topologies=rx_topos,
            freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
            freq_ref_hz=round(_snap(0.5 * (dom_freq[0] + dom_freq[1]),
                                   _GRID_FREQ_HZ_STEP), 1),
            gc_enabled=gc_enabled, gc_dia_mm=gc_dia_mm,
            p_target_w=p_target_w, v_min=v_min, v_max=v_max,
            v_cap_target_dc=v_cap_target_dc,
            v_ldo_out=_V_LDO_OUT,
            v_rx_cap_rating=v_rx_cap_rating,
            d_min=d_min, d_max=d_max,
            c_tx_options_f=c_tx_options_f,
            c_rx_options_f=c_rx_options_f,
            dither_amp_hz=dither_amp_hz,
            zvs_margin=zvs_margin,
            n_combos=n_combos, max_iters=max_iters, top_k=top_k,
            fh_workers=fh_workers, fh_timeout=fh_timeout,
            # Fixed-stackup constants for this run
            pcb_gap_mm=pcb_gap_mm,
            tx_layers=tx_layers,
            rx_layers=rx_layers,
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

        rx_topo = r["rx_topology"]
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
        rx_layers = pc.active_layer_data(rx_sp, rx_stackup)
        rx_path   = os.path.join(temp_dir, "nn_auto_rx.inp")
        pc.write_topology_inp(rx_topo, rx_layers, rx_path,
                              w_mm=r["rx_width"], fmin=fmin, fmax=fmax)
        rx_flags  = pc.series_reverse_flags_for_topology(rx_topo, len(rx_layers))
        rx_native = pc.reverse_nodes_for_series_flow(rx_layers, rx_flags)
        rx_meta = {"role": "RX", "topology": rx_topo,
                   "layer_params": [(r["rx_width"], ld["h_mm"], len(ld["nodes"]))
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
        self._log_text.configure(state="normal")
        at_end = self._log_text.yview()[1] >= 0.999
        self._log_text.insert("end", msg + "\n")
        if at_end:
            self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _set_status(self, msg, color="gray"):
        self._status_var.set(msg)
