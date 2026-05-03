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

import copy
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


# ── Fallback domain ───────────────────────────────────────────────────────────
_FALLBACK_DOMAIN = {
    "tx": {
        "od_max_mm": 53.0, "id_min_mm": 35.0,
        "trace_width_mm": [0.2, 1.2], "turns": [6, 18],
        "allowed_topologies": ["parallel", "series"],
    },
    "rx": {
        "od_max_mm": 53.0, "id_min_mm": 35.0,
        "trace_width_mm": [0.2, 1.2], "turns": [4, 25],
        "allowed_topologies": ["parallel", "series", "parallel_pairs_ser"],
    },
    "global": {"freq_hz": [110000.0, 135000.0]},
    "ground_circle_enabled": False,
    "ground_circle_min_mm": 18.0,
    "ground_circle_max_mm": 24.0,
}

# ── Physical constants ────────────────────────────────────────────────────────
BATCH_SIZE   = 500_000
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

_TOPO_OH_IDX = {"parallel": 0, "parallel_pairs_ser": 1, "series": 2}

_V_DIODE = 0.35
_V_DROP  = 2.0 * _V_DIODE

_TRAIN_SCRIPT = os.path.join(_NN_SCRIPTS, "train_surrogate.py")

# ── SimParams stackup defaults (must match training data) ─────────────────────
_TX_LAYERS_DEFAULT = [
    {"active": True,  "copper_oz": 1.0},
    {"active": True,  "copper_oz": 1.0},
    {"active": True,  "copper_oz": 1.0},
    {"active": False, "copper_oz": 1.0},
]
_RX_LAYERS_DEFAULT = [
    {"active": True, "copper_oz": 1.0},
    {"active": True, "copper_oz": 0.5},
    {"active": True, "copper_oz": 0.5},
    {"active": True, "copper_oz": 1.0},
]


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
            pass
    return copy.deepcopy(_FALLBACK_DOMAIN)


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
                 freq_ref_hz, freq_hz, V_min, V_max, P_target_w, D_min):
    """
    Returns (eta_sys, feasible) arrays.  eta_sys is system efficiency (link×rect)
    at worst-case V_min.  feasible = True where combo can deliver P_target.
    """
    omega   = 2.0 * math.pi * freq_hz
    skin_sc = math.sqrt(freq_hz / freq_ref_hz)

    L_tx = np.maximum(L_tx.astype(np.float64) * 1e-6, 1e-9)
    L_rx = np.maximum(L_rx.astype(np.float64) * 1e-6, 1e-9)
    M    = np.maximum(M.astype(np.float64)    * 1e-6, 0.0)

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

    V_rms_min = V_min * math.sqrt(2.0) / math.pi
    V_rms_max = V_max * math.sqrt(2.0) / math.pi

    P_rx_min = eta_link * (V_rms_min ** 2) / Z_tx_opt
    P_rx_max = eta_link * (V_rms_max ** 2) / Z_tx_opt

    R_ratio = np.sqrt(np.maximum(R_rx_ac / np.maximum(R_tx_ac, 1e-12), 0.0))

    def _eta_rect(V_rms):
        V_load_pk = (U * V_rms * R_ratio
                     / np.maximum(1.0 + sq, 1e-12) * math.sqrt(2.0))
        return np.where(V_load_pk > _V_DROP,
                        (V_load_pk - _V_DROP) / np.maximum(V_load_pk, 1e-12), 0.0)

    eta_rect_min = _eta_rect(V_rms_min)
    eta_sys      = eta_link * eta_rect_min
    P_dc_min     = P_rx_min * eta_rect_min
    P_dc_max     = P_rx_max * _eta_rect(V_rms_max)

    D_vmin = np.where(P_dc_min > 1e-18, P_target_w / P_dc_min, np.inf)

    if D_min > 0.0:
        feasible = (P_dc_min * D_min >= P_target_w)
    else:
        feasible = (D_vmin <= 1.0)
    feasible &= (R_tx_ac > 0) & (R_rx_ac > 0) & (M > 0)

    return eta_sys.astype(np.float32), feasible, D_vmin.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Single-iteration NN sweep → returns sorted list of dicts (best first)
# ─────────────────────────────────────────────────────────────────────────────

def _nn_sweep(params, log_cb, cancel_flag):
    """
    Sample N combinations, evaluate via NN, score with physics kernel.
    Returns list of result dicts sorted by eta_sys descending.
    """
    model_dir    = params["model_dir"]
    model, x_scaler, y_scaler, feat_cols, torch, device = _load_nn(model_dir)

    tx_od_max    = params["tx_od_max"]
    tx_id_min    = params["tx_id_min"]
    tx_width_min = params["tx_width_min"]
    tx_width_max = params["tx_width_max"]
    tx_turns_min = params["tx_turns_min"]

    rx_od_max    = min(params["rx_od_max"], tx_od_max)
    rx_id_min    = params["rx_id_min"]
    rx_width_min = params["rx_width_min"]
    rx_width_max = params["rx_width_max"]
    rx_turns_min = params["rx_turns_min"]

    topologies  = params["topologies"]
    n_topos     = len(topologies)

    gc_enabled  = params["gc_enabled"]
    gc_dia_mm   = params["gc_dia_mm"]

    freq_min_hz = params["freq_min_hz"]
    freq_max_hz = params["freq_max_hz"]
    freq_ref_hz = 0.5 * (freq_min_hz + freq_max_hz)

    V_min      = params["v_min"]
    V_max      = params["v_max"]
    P_target   = params["p_target_w"]
    D_min      = params["d_min"]

    N   = params["n_combos"]

    _base    = ["tx_turns", "tx_width", "tx_od_mm",
                "rx_od_mm", "rx_turns", "rx_width", "freq_hz"]
    _topo_oh = ["topo_parallel", "topo_parallel_pairs_ser", "topo_series"]
    _gc_col  = "ground_circle_dia_mm"
    all_cols = feat_cols if feat_cols else (_base + sorted(_topo_oh))
    has_gc   = (_gc_col in all_cols)

    log_cb(f"  NN features: {all_cols}")
    log_cb(f"  Topologies: {topologies}")

    # Accumulators
    acc_keys = [
        "tx_turns", "tx_width", "tx_od_mm",
        "rx_od_mm", "rx_turns", "rx_width",
        "rx_topo", "gc_dia_mm",
        "L_tx", "L_rx", "M",
        "R_tx", "R_rx", "DCR_tx", "DCR_rx",
        "tx_id_mm", "rx_id_mm",
        "eta_sys", "D_vmin",
    ]
    accs = {k: [] for k in acc_keys}

    n_ok  = 0
    n_rej = 0
    b     = 0
    rng   = np.random.default_rng(params.get("rng_seed", 42))

    while n_ok < N:
        if cancel_flag.is_set():
            return None
        b += 1
        if b % 20 == 1:
            log_cb(f"    batch {b}: {n_ok:,}/{N:,} valid, {n_rej:,} rejected")

        bs = BATCH_SIZE

        tx_od_s    = rng.uniform(tx_id_min + 2.0, tx_od_max, size=bs).astype(np.float32)
        tx_width_s = rng.uniform(tx_width_min, tx_width_max, size=bs).astype(np.float32)
        tx_t_max   = np.maximum(
            np.floor((tx_od_s + _SPACING_MM - tx_width_s - tx_id_min)
                     / (2.0 * (tx_width_s + _SPACING_MM))).astype(np.int32),
            tx_turns_min)
        tx_t_min   = np.minimum(np.full(bs, tx_turns_min, dtype=np.int32), tx_t_max)
        tx_turns_s = rng.integers(tx_t_min, tx_t_max + 1).astype(np.float32)

        rx_od_s    = rng.uniform(rx_id_min + 2.0, rx_od_max, size=bs).astype(np.float32)
        rx_width_s = rng.uniform(rx_width_min, rx_width_max, size=bs).astype(np.float32)
        rx_t_max   = np.maximum(
            np.floor((rx_od_s + _SPACING_MM - rx_width_s - rx_id_min)
                     / (2.0 * (rx_width_s + _SPACING_MM))).astype(np.int32),
            rx_turns_min)
        rx_t_min   = np.minimum(np.full(bs, rx_turns_min, dtype=np.int32), rx_t_max)
        rx_turns_s = rng.integers(rx_t_min, rx_t_max + 1).astype(np.float32)

        topo_idx_s = rng.integers(0, n_topos, size=bs)

        gc_s = (np.full(bs, gc_dia_mm, dtype=np.float32)
                if gc_enabled and has_gc
                else np.zeros(bs, dtype=np.float32))

        tx_id_s = _inner_diameter_mm(tx_od_s, tx_width_s, tx_turns_s)
        rx_id_s = _inner_diameter_mm(rx_od_s, rx_width_s, rx_turns_s)
        id_ok   = (tx_id_s >= tx_id_min) & (rx_id_s >= rx_id_min)
        n_rej  += int((~id_ok).sum())
        n_ok   += int(id_ok.sum())
        if not id_ok.any():
            continue

        tx_turns_s = tx_turns_s[id_ok]; tx_od_s    = tx_od_s[id_ok]
        tx_width_s = tx_width_s[id_ok]; rx_od_s    = rx_od_s[id_ok]
        rx_turns_s = rx_turns_s[id_ok]; rx_width_s = rx_width_s[id_ok]
        topo_idx_s = topo_idx_s[id_ok]; gc_s       = gc_s[id_ok]
        tx_id_s    = tx_id_s[id_ok];    rx_id_s    = rx_id_s[id_ok]
        bs_v = int(id_ok.sum())

        topo_oh = np.zeros((bs_v, 3), dtype=np.float32)
        for ti, tname in enumerate(topologies):
            col = _TOPO_OH_IDX.get(tname, ti)
            topo_oh[topo_idx_s == ti, col] = 1.0

        tx_len_m  = _spiral_length_m(tx_od_s, tx_width_s, tx_turns_s)
        DCR_tx_np = (_RHO_30C * tx_len_m
                     / (tx_width_s * 1e-3 * _TX_H_M * _TX_N_LAYERS)).astype(np.float32)

        rx_len_m   = _spiral_length_m(rx_od_s, rx_width_s, rx_turns_s)
        rx_w_m     = rx_width_s * 1e-3
        DCR_rx_par = _RHO_30C * rx_len_m / (rx_w_m * _RX_HSUM_PAR)
        DCR_rx_ser = _RHO_30C * rx_len_m * _RX_HINV_SER / rx_w_m
        DCR_rx_pps = _RHO_30C * rx_len_m * (1.0 / _RX_HSUM_A + 1.0 / _RX_HSUM_B) / rx_w_m
        DCR_rx_np  = np.empty(bs_v, dtype=np.float32)
        for ti, tname in enumerate(topologies):
            m = (topo_idx_s == ti)
            if not m.any(): continue
            if tname == "parallel":
                DCR_rx_np[m] = DCR_rx_par[m]
            elif tname == "series":
                DCR_rx_np[m] = DCR_rx_ser[m]
            else:
                DCR_rx_np[m] = DCR_rx_pps[m]

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
        X    = np.column_stack([row_map.get(c, np.zeros(bs_v, dtype=np.float32))
                                for c in all_cols]).astype(np.float32)
        X_sc = x_scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            Y_sc = model(torch.tensor(X_sc, device=device))
        Y = y_scaler.inverse_transform(Y_sc.cpu().numpy())

        eta_sys, feasible, D_vmin = _score_batch(
            Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4],
            DCR_tx_np, DCR_rx_np,
            freq_ref_hz, freq_ref_hz,
            V_min, V_max, P_target, D_min,
        )

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
        accs["eta_sys"].append(eta_sys)
        accs["D_vmin"].append(D_vmin)

    concat = {k: np.concatenate(v) for k, v in accs.items()}
    concat["_topologies"]  = np.array(topologies)
    concat["_freq_ref_hz"] = np.float64(freq_ref_hz)
    concat["_freq_min_hz"] = np.float64(freq_min_hz)
    concat["_freq_max_hz"] = np.float64(freq_max_hz)

    log_cb(f"  Sweep done: {n_ok:,} valid ({n_rej:,} rejected)")
    return concat


# ─────────────────────────────────────────────────────────────────────────────
# Pick top-K unique configurations from sweep result
# ─────────────────────────────────────────────────────────────────────────────

def _top_k(result, topologies, k=12):
    eta   = result["eta_sys"]
    order = np.argsort(eta)[::-1]

    winners = []
    seen    = set()
    for i in order:
        tx_t  = int(result["tx_turns"][i])
        tx_od = round(float(result["tx_od_mm"][i]), 1)
        tx_w  = round(float(result["tx_width"][i]), 2)
        rx_t  = int(result["rx_turns"][i])
        rx_od = round(float(result["rx_od_mm"][i]), 1)
        rx_w  = round(float(result["rx_width"][i]), 2)
        t_idx = int(result["rx_topo"][i])
        topo  = topologies[t_idx] if t_idx < len(topologies) else "parallel"
        key   = (tx_t, tx_od, tx_w, rx_t, rx_od, rx_w, topo)
        if key in seen:
            continue
        seen.add(key)
        gc = float(result["gc_dia_mm"][i])
        winners.append({
            "tx_turns":   tx_t,
            "tx_width":   float(result["tx_width"][i]),
            "tx_od_mm":   float(result["tx_od_mm"][i]),
            "rx_turns":   rx_t,
            "rx_width":   float(result["rx_width"][i]),
            "rx_od_mm":   float(result["rx_od_mm"][i]),
            "rx_topology": topo,
            "gc_dia_mm":  gc,
            "eta_sys":    float(eta[i]),
        })
        if len(winners) >= k:
            break
    return winners


# ─────────────────────────────────────────────────────────────────────────────
# Build SimParams list for FastHenry batch
# ─────────────────────────────────────────────────────────────────────────────

def _build_sim_params(winners, domain, timeout_sec):
    from parallel_sim import SimParams
    glb = domain.get("global", {})
    freq_range = glb.get("freq_hz", [110000.0, 135000.0])
    freq_hz  = 0.5 * (freq_range[0] + freq_range[1])
    pcb_gap  = glb.get("pcb_gap_mm", [2.4, 2.8])
    pcb_gap_mm = 0.5 * (pcb_gap[0] + pcb_gap[1]) if isinstance(pcb_gap, list) else float(pcb_gap)

    tx_d = domain.get("tx", {})
    rx_d = domain.get("rx", {})

    params_list = []
    for w in winners:
        p = SimParams(
            tx_turns          = w["tx_turns"],
            tx_trace_width_mm = w["tx_width"],
            tx_od_mm          = w["tx_od_mm"],
            tx_spacing_mm     = 0.16,
            tx_outer_gap_mm   = float(tx_d.get("outer_gap_mm", [0.2, 0.2])[0]),
            tx_inner_gap_mm   = float(tx_d.get("inner_gap_mm", [1.0, 1.0])[0]),
            tx_topology       = "parallel",
            tx_layers         = list(_TX_LAYERS_DEFAULT),
            tx_port_inside    = bool(tx_d.get("port_inside_allowed", False)),

            rx_turns          = w["rx_turns"],
            rx_trace_width_mm = w["rx_width"],
            rx_od_mm          = w["rx_od_mm"],
            rx_spacing_mm     = 0.16,
            rx_outer_gap_mm   = float(rx_d.get("outer_gap_mm", [0.21, 0.21])[0]),
            rx_inner_gap_mm   = float(rx_d.get("inner_gap_mm", [0.6, 0.6])[0]),
            rx_topology       = w["rx_topology"],
            rx_layers         = list(_RX_LAYERS_DEFAULT),
            rx_port_inside    = bool(rx_d.get("port_inside_allowed", True)),

            tx_nhinc          = int(tx_d.get("nhinc", 1)),
            tx_nwinc          = int(tx_d.get("nwinc", 3)),
            rx_nhinc          = int(rx_d.get("nhinc", 1)),
            rx_nwinc          = int(rx_d.get("nwinc", 3)),

            pcb_gap_mm        = pcb_gap_mm,
            freq_hz           = freq_hz,
            fmin_hz           = freq_range[0],
            fmax_hz           = freq_range[1],
            resolution_mm     = float(glb.get("resolution_mm", 1.5)),
            timeout_sec       = timeout_sec,
            ground_circle_dia_mm = w.get("gc_dia_mm", 0.0),
        )
        params_list.append(p)
    return params_list


# ─────────────────────────────────────────────────────────────────────────────
# Append sim results to refined_results.json
# ─────────────────────────────────────────────────────────────────────────────

def _append_to_refined(sim_results, winners, refined_path, domain, log_cb):
    if os.path.exists(refined_path):
        try:
            with open(refined_path) as f:
                data = json.load(f)
        except Exception:
            data = {"meta": {}, "results": []}
    else:
        data = {"meta": {}, "results": []}

    existing_uuids = {r.get("uuid") for r in data["results"] if r.get("uuid")}
    freq_range = domain.get("global", {}).get("freq_hz", [110000.0, 135000.0])
    freq_hz    = 0.5 * (freq_range[0] + freq_range[1])
    added      = 0

    for i, (sr, w) in enumerate(zip(sim_results, winners)):
        if sr is None or "error" in sr:
            log_cb(f"    sim {i+1}: FAIL — {(sr or {}).get('error','?')}")
            continue
        uid = str(uuid.uuid4())
        row = {
            "uuid":              uid,
            "ok":                True,
            "tx_turns":          w["tx_turns"],
            "tx_width":          w["tx_width"],
            "tx_trace_width_mm": w["tx_width"],
            "tx_od_mm":          w["tx_od_mm"],
            "tx_spacing_mm":     0.16,
            "tx_topology":       "parallel",
            "rx_turns":          w["rx_turns"],
            "rx_width":          w["rx_width"],
            "rx_trace_width_mm": w["rx_width"],
            "rx_od_mm":          w["rx_od_mm"],
            "rx_spacing_mm":     0.16,
            "rx_topology":       w["rx_topology"],
            "ground_circle_dia_mm": w.get("gc_dia_mm", 0.0),
            "freq_hz":           sr.get("freq_hz", freq_hz),
            "L_tx_uH":           sr.get("L_tx_uH", 0.0),
            "L_rx_uH":           sr.get("L_rx_uH", 0.0),
            "M_uH":              sr.get("M_uH", 0.0),
            "R_tx_ac":           sr.get("R_tx_ac", 0.0),
            "R_rx_ac":           sr.get("R_rx_ac", 0.0),
            "k":                 sr.get("k", 0.0),
            "elapsed_sec":       sr.get("elapsed_sec", 0.0),
        }
        if uid not in existing_uuids:
            data["results"].append(row)
            existing_uuids.add(uid)
            added += 1
        log_cb(f"    sim {i+1}: OK  L_tx={sr.get('L_tx_uH',0):.2f}µH  "
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

    log_cb("  Retraining model…")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", _TRAIN_SCRIPT],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=_NN_DIR, env=env)
        for line in proc.stdout:
            if cancel_flag.is_set():
                proc.terminate()
                return False
            log_cb("    " + line.rstrip())
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
        domain        = params["domain"]

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

        topologies = params["topologies"]
        prev_winners = []
        win_streak   = 0

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
            sweep_result = _nn_sweep(sweep_seed_params, log_cb, cancel_flag)
            if sweep_result is None:
                log_cb("Cancelled during sweep.")
                done_cb(None)
                return

            # --- Pick top-K ---
            winners = _top_k(sweep_result, topologies, k=top_k)
            if not winners:
                log_cb("  No feasible combinations found — stopping.")
                break

            top1 = winners[0]
            log_cb(f"\nTop winner: TX {top1['tx_od_mm']:.1f}mm {top1['tx_turns']}t "
                   f"w={top1['tx_width']:.2f}  |  "
                   f"RX {top1['rx_od_mm']:.1f}mm {top1['rx_turns']}t "
                   f"w={top1['rx_width']:.2f} {top1['rx_topology']}  "
                   f"η={top1['eta_sys']*100:.1f}%  D={top1.get('D_vmin',1)*100:.0f}%")
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
            sim_params_list = _build_sim_params(winners, domain, fh_timeout)

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
            _append_to_refined(sim_results, winners, refined_path, domain, log_cb)

            # --- Retrain ---
            log_cb("\nPhase 4: Retraining model…")
            ok = _retrain(model_dir, refined_path, train_params, log_cb, cancel_flag)
            if not ok or cancel_flag.is_set():
                log_cb("Retrain failed or cancelled.")
                done_cb(None)
                return

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
            and a["rx_topology"] == b["rx_topology"])


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
        self._domain          = copy.deepcopy(_FALLBACK_DOMAIN)
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

        # ── Domain info banner ────────────────────────────────────────────────
        dom_f = ttk.LabelFrame(col_l, text="Active Domain")
        dom_f.pack(fill="x", pady=(0, 6))
        self._domain_info_var = tk.StringVar(value="No model loaded.")
        ttk.Label(dom_f, textvariable=self._domain_info_var,
                  foreground="#1a6fcc", font=("Consolas", 8),
                  wraplength=340, justify="left").pack(anchor="w", padx=6, pady=(4, 4))

        # ── TX / RX narrowing ────────────────────────────────────────────────
        tx_f = ttk.LabelFrame(col_l, text="Narrow Down Domain — TX")
        tx_f.pack(fill="x", pady=(0, 6))
        self._tx_od_max = self._row(tx_f, "OD max (mm):", "53.0")
        self._tx_id_min = self._row(tx_f, "ID min (mm):", "35.0")
        self._bind_cap(self._tx_od_max, "tx", "od_max_mm")
        self._bind_cap(self._tx_id_min, "tx", "id_min_mm", is_min=True)

        rx_f = ttk.LabelFrame(col_l, text="Narrow Down Domain — RX")
        rx_f.pack(fill="x", pady=(0, 6))
        self._rx_od_max = self._row(rx_f, "OD max (mm):", "53.0")
        self._rx_id_min = self._row(rx_f, "ID min (mm):", "35.0")
        self._bind_cap(self._rx_od_max, "rx", "od_max_mm")
        self._bind_cap(self._rx_id_min, "rx", "id_min_mm", is_min=True)

        # ── Ground circle ────────────────────────────────────────────────────
        gc_f = ttk.LabelFrame(col_l, text="Ground Circle")
        gc_f.pack(fill="x", pady=(0, 6))
        self._gc_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(gc_f, text="Enable Ground Circle (fixed diameter)",
                        variable=self._gc_enabled,
                        command=self._on_gc_toggle).pack(anchor="w", padx=6, pady=(4, 2))
        gc_row = ttk.Frame(gc_f)
        gc_row.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(gc_row, text="GC diameter (mm):", width=22, anchor="w").pack(side="left")
        self._gc_dia = tk.StringVar(value="20.0")
        self._gc_dia_entry = ttk.Entry(gc_row, textvariable=self._gc_dia, width=8)
        self._gc_dia_entry.pack(side="left")
        self._on_gc_toggle()

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
                   command=self._on_next_tab_click).pack(fill="x")

        # ── RIGHT COLUMN ──────────────────────────────────────────────────────

        # ── Evaluation parameters ─────────────────────────────────────────────
        ev_f = ttk.LabelFrame(col_r, text="Evaluation Parameters")
        ev_f.pack(fill="x", pady=(0, 6))
        self._p_target_mw = self._row(ev_f, "Target RX power (mW):", "50")
        self._v_min       = self._row(ev_f, "TX V_min (V):", "3.2")
        self._v_max       = self._row(ev_f, "TX V_max (V):", "4.4")
        self._d_min_pct   = self._row(ev_f, "Min duty cycle (%):", "")
        ttk.Label(ev_f, text="Blank = no minimum duty floor.",
                  foreground="gray", font=("TkDefaultFont", 8)
                  ).pack(anchor="w", padx=6, pady=(0, 2))
        self._rx_max_caps = self._row(ev_f, "RX caps (1 or 2):", "2")

        # ── Iteration & sweep settings ────────────────────────────────────────
        it_f = ttk.LabelFrame(col_r, text="Optimisation Settings")
        it_f.pack(fill="x", pady=(0, 6))
        self._n_combos  = self._row(it_f, "Combinations / iter (M):", "10")
        self._max_iters = self._row(it_f, "Max iterations:", "20")
        self._top_k     = self._row(it_f, "Top-K for FastHenry:", "12")
        self._fh_workers   = self._row(it_f, "FH workers:", "6")
        self._fh_timeout   = self._row(it_f, "FH timeout (s):", "360")

        # ── Training hyperparams ──────────────────────────────────────────────
        tr_f = ttk.LabelFrame(col_r, text="Retrain Hyperparameters")
        tr_f.pack(fill="x", pady=(0, 6))
        self._tr_epochs    = self._row(tr_f, "Epochs:", "300")
        self._tr_batch     = self._row(tr_f, "Batch size:", "512")
        self._tr_lr        = self._row(tr_f, "LR:", "0.0005")
        self._tr_val_split = self._row(tr_f, "Val split:", "0.15")

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

        tx  = domain.get("tx", {})
        rx  = domain.get("rx", {})
        glb = domain.get("global", {})

        self._tx_od_max.set(f"{tx.get('od_max_mm', 53.0):.1f}")
        self._tx_id_min.set(f"{tx.get('id_min_mm', 35.0):.1f}")
        self._rx_od_max.set(f"{rx.get('od_max_mm', 53.0):.1f}")
        self._rx_id_min.set(f"{rx.get('id_min_mm', 35.0):.1f}")

        gc_en  = domain.get("ground_circle_enabled", False)
        gc_min = domain.get("ground_circle_min_mm", 18.0)
        gc_max = domain.get("ground_circle_max_mm", 24.0)
        mid_gc = 0.5 * (gc_min + gc_max)
        self._gc_dia.set(f"{mid_gc:.1f}")
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
        self._model_label_var.set(self._short_path(model_dir))

    # ─────────────────────────────────────────────────────────────────────────
    # Event handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_gc_toggle(self):
        state = "normal" if self._gc_enabled.get() else "disabled"
        try:
            self._gc_dia_entry.configure(state=state)
        except Exception:
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
        dom_freq      = glb.get("freq_hz", [110000.0, 135000.0])
        rx_topos      = rx.get("allowed_topologies") or ["parallel", "series", "parallel_pairs_ser"]

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

        p_target_w = flt(self._p_target_mw, "Target RX power", lo=0.001) / 1000.0
        v_min      = flt(self._v_min, "V_min", lo=0.1)
        v_max      = flt(self._v_max, "V_max", lo=v_min)
        d_min_pct  = flt(self._d_min_pct, "Min duty %", lo=0.0, allow_empty=True)
        d_min      = d_min_pct / 100.0
        rx_caps    = max(1, min(2, int(flt(self._rx_max_caps, "RX caps", lo=1, hi=2))))

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
            tx_turns_min=dom_tx_turns[0],
            rx_od_max=rx_od_max, rx_id_min=rx_id_min,
            rx_width_min=dom_rx_w[0], rx_width_max=dom_rx_w[1],
            rx_turns_min=dom_rx_turns[0],
            topologies=rx_topos,
            freq_min_hz=dom_freq[0], freq_max_hz=dom_freq[1],
            gc_enabled=gc_enabled, gc_dia_mm=gc_dia_mm,
            p_target_w=p_target_w, v_min=v_min, v_max=v_max,
            d_min=d_min, rx_max_caps=rx_caps,
            n_combos=n_combos, max_iters=max_iters, top_k=top_k,
            fh_workers=fh_workers, fh_timeout=fh_timeout,
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
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _set_status(self, msg, color="gray"):
        self._status_var.set(msg)
