#!/usr/bin/env python3
"""
Parallel FastHenry simulation worker — combined TX+RX 2-port sims.

Fixed stackup (not NN inputs):
  TX: L1+L2 active, series, port outside, outer=1oz inner=0.5oz
  RX: all 4 layers, port inside, outer=1oz inner=0.5oz

Free variables (sampled, become NN features):
  TX: od, turns (L1), l2_turns, trace_width, spacing, l1l2_gap
  RX: od, turns, trace_width, spacing, outer_gap, inner_gap, rx_topo (0/1)
  Global: pcb_gap, freq
"""

import math
import os
import sys
import tempfile
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional

_HERE        = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT    = os.path.dirname(os.path.dirname(_HERE))
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")

for _p in (_HERE, _MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parametric_coil as pc
import zc_parser

# Fixed copper weights — never change, never passed as parameters.
_TX_OUTER_OZ = 1.0
_TX_INNER_OZ = 0.5
_RX_OUTER_OZ = 1.0
_RX_INNER_OZ = 0.5

_TX_LAYERS = [
    {"active": True,  "copper_oz": _TX_OUTER_OZ},
    {"active": True,  "copper_oz": _TX_INNER_OZ},
    {"active": False, "copper_oz": 1.0},
    {"active": False, "copper_oz": 1.0},
]
_RX_LAYERS = [
    {"active": True, "copper_oz": _RX_OUTER_OZ},
    {"active": True, "copper_oz": _RX_INNER_OZ},
    {"active": True, "copper_oz": _RX_INNER_OZ},
    {"active": True, "copper_oz": _RX_OUTER_OZ},
]

# RX topology index → string name used by parametric_coil.
_RX_TOPO_NAMES = ["series", "parallel_pairs_ser"]


@dataclass
class SimParams:
    """All inputs for one combined TX+RX FastHenry simulation."""

    # TX free variables
    tx_turns:       float
    tx_od_mm:       float
    tx_w_mm:        float
    tx_spacing_mm:  float = 0.16
    tx_l1l2_gap_mm: float = 0.2104
    tx_l2_turns:    int   = 0      # 0 → same as tx_turns (no independent L2)
    tx_nwinc:       int   = 3

    # RX free variables
    rx_turns:       float = 10.0
    rx_od_mm:       float = 50.0
    rx_w_mm:        float = 0.5
    rx_spacing_mm:  float = 0.16
    rx_outer_gap_mm: float = 0.2104
    rx_inner_gap_mm: float = 0.6
    rx_topo:        int   = 0      # 0=series, 1=parallel_pairs_ser
    rx_nwinc:       int   = 3

    # Global
    pcb_gap_mm:    float = 2.6
    freq_hz:       float = 360_000.0
    resolution_mm: float = 1.2
    timeout_sec:   float = 240.0

    # Passive ground copper
    rx_ground_disc_dia_mm: float = 20.0     # 0 disables; on both RX inner layers
    tx_ground_enabled:     bool  = True     # fixed-shape TX layer-3 pour

    tag:       str = ""
    sample_id: int = -1


# ---------------------------------------------------------------------------
# Feasibility pre-check
# ---------------------------------------------------------------------------

def _check_feasibility(p: SimParams) -> Optional[str]:
    # TX
    sp_tx = pc.SpiralParams(od_mm=p.tx_od_mm, trace_width_mm=p.tx_w_mm,
                            spacing_mm=p.tx_spacing_mm, turns=p.tx_turns,
                            resolution_mm=p.resolution_mm)
    ok, msg = pc.validate_spiral(sp_tx)
    if not ok:
        return f"TX: {msg}"
    stackup_tx = pc.StackUp(
        slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
               for l in _TX_LAYERS],
        outer_gap_mm=p.tx_l1l2_gap_mm, inner_gap_mm=0.0,
    )
    ok, msg = pc.validate_stackup(stackup_tx, min_active=2, max_active=2)
    if not ok:
        return f"TX stackup: {msg}"
    l2_t = p.tx_l2_turns if p.tx_l2_turns > 0 else int(round(p.tx_turns))
    _, w2 = pc.compute_layer2_width(sp_tx, l2_t)
    if w2 <= 0:
        return f"TX L2 width {w2:.4f} mm <= 0"

    # RX
    sp_rx = pc.SpiralParams(od_mm=p.rx_od_mm, trace_width_mm=p.rx_w_mm,
                            spacing_mm=p.rx_spacing_mm, turns=p.rx_turns,
                            resolution_mm=p.resolution_mm)
    ok, msg = pc.validate_spiral(sp_rx)
    if not ok:
        return f"RX: {msg}"
    stackup_rx = pc.StackUp(
        slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
               for l in _RX_LAYERS],
        outer_gap_mm=p.rx_outer_gap_mm, inner_gap_mm=p.rx_inner_gap_mm,
    )
    ok, msg = pc.validate_stackup(stackup_rx)
    if not ok:
        return f"RX stackup: {msg}"
    return None


# ---------------------------------------------------------------------------
# .inp generation
# ---------------------------------------------------------------------------

def _write_inp(p: SimParams, inp_path: str) -> None:
    # TX: independent L2 turns via active_layer_data_tx_independent.
    sp_tx = pc.SpiralParams(od_mm=p.tx_od_mm, trace_width_mm=p.tx_w_mm,
                            spacing_mm=p.tx_spacing_mm, turns=p.tx_turns,
                            resolution_mm=p.resolution_mm)
    stackup_tx = pc.StackUp(
        slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
               for l in _TX_LAYERS],
        outer_gap_mm=p.tx_l1l2_gap_mm, inner_gap_mm=0.0,
    )
    l2_t = p.tx_l2_turns if p.tx_l2_turns > 0 else int(round(p.tx_turns))
    tx_ld, _ = pc.active_layer_data_tx_independent(sp_tx, stackup_tx, l2_t)

    # RX: standard 4-layer.
    sp_rx = pc.SpiralParams(od_mm=p.rx_od_mm, trace_width_mm=p.rx_w_mm,
                            spacing_mm=p.rx_spacing_mm, turns=p.rx_turns,
                            resolution_mm=p.resolution_mm)
    stackup_rx = pc.StackUp(
        slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
               for l in _RX_LAYERS],
        outer_gap_mm=p.rx_outer_gap_mm, inner_gap_mm=p.rx_inner_gap_mm,
    )
    rx_ld = pc.active_layer_data(sp_rx, stackup_rx)

    rx_topo_name = _RX_TOPO_NAMES[p.rx_topo] if 0 <= p.rx_topo < len(_RX_TOPO_NAMES) else "series"

    pc.write_combined_tx_rx_inp(
        tx_ld, rx_ld, inp_path,
        tx_w_mm=p.tx_w_mm,
        rx_w_mm=p.rx_w_mm,
        tx_topology="series",
        rx_topology=rx_topo_name,
        tx_port_inside=False,
        rx_port_inside=True,
        pcb_gap_mm=p.pcb_gap_mm,
        tx_nhinc=1, tx_nwinc=p.tx_nwinc,
        rx_nhinc=1, rx_nwinc=p.rx_nwinc,
        fmin=p.freq_hz, fmax=p.freq_hz, freq_ndec=0,
        rx_ground_disc_dia_mm=p.rx_ground_disc_dia_mm,
        tx_ground_enabled=p.tx_ground_enabled,
    )


# ---------------------------------------------------------------------------
# FastHenry execution
# ---------------------------------------------------------------------------

def _run_fasthenry(inp_path: str, p: SimParams) -> str:
    work_dir = os.path.dirname(os.path.abspath(inp_path))

    if os.environ.get("FASTHENRY_BACKEND", "").lower() == "linux":
        import subprocess
        bin_name = os.environ.get("FASTHENRY_BIN", "fasthenry")
        proc = subprocess.Popen(
            [bin_name, inp_path],
            cwd=work_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        t0 = time.time()
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            if time.time() - t0 > p.timeout_sec:
                proc.terminate()
                raise RuntimeError(f"FastHenry timed out after {p.timeout_sec:.0f}s")
            time.sleep(0.25)
    else:
        import pythoncom
        import win32com.client as w32c

        pythoncom.CoInitialize()
        fh = w32c.Dispatch("FastHenry2.Document")
        try:
            started = fh.Run(f'"{inp_path}"')
            if not started:
                raise RuntimeError("FastHenry2.Run() returned False")
            t0 = time.time()
            while fh.IsRunning:
                if time.time() - t0 > p.timeout_sec:
                    try:
                        fh.Stop()
                    except Exception:
                        pass
                    raise RuntimeError(f"FastHenry timed out after {p.timeout_sec:.0f}s")
                time.sleep(0.25)
        finally:
            try:
                fh.Quit
            except Exception:
                pass

    zc_path = os.path.join(work_dir, "Zc.mat")
    if not os.path.exists(zc_path):
        raise FileNotFoundError("Zc.mat not produced by FastHenry")
    return zc_path


# ---------------------------------------------------------------------------
# Result parsing — verbose dict schema, compatible with train_surrogate.py
# ---------------------------------------------------------------------------

def _parse_result(zc_path: str, p: SimParams, sample: dict) -> dict:
    blocks = zc_parser.parse_zc_mat(zc_path)

    n_ports = zc_parser.port_count(blocks)
    if n_ports != 2:
        raise ValueError(f"Expected 2-port Zc.mat but got {n_ports}-port")

    freq_actual, matrix = zc_parser.matrix_at(blocks, p.freq_hz)

    z11 = matrix[0][0]
    z22 = matrix[1][1]
    z12 = matrix[0][1]

    for name, z in [("Z11", z11), ("Z22", z22), ("Z12", z12)]:
        if math.isnan(z.real) or math.isnan(z.imag):
            raise ValueError(f"NaN in {name}")

    omega = 2.0 * math.pi * p.freq_hz
    L_tx  = z11.imag / omega
    L_rx  = z22.imag / omega
    M     = z12.imag / omega
    R_tx  = z11.real
    R_rx  = z22.real
    denom = math.sqrt(L_tx * L_rx) if L_tx > 0 and L_rx > 0 else 0.0
    k     = M / denom if denom > 0 else 0.0

    Q_tx  = (omega * L_tx / R_tx) if R_tx > 0 else 0.0
    Q_rx  = (omega * L_rx / R_rx) if R_rx > 0 else 0.0

    rx_topo_idx  = int(sample.get("rx_topo", p.rx_topo))
    rx_topo_name = (_RX_TOPO_NAMES[rx_topo_idx]
                    if 0 <= rx_topo_idx < len(_RX_TOPO_NAMES) else "series")

    def r5(v): return round(v, 5)
    def r4(v): return round(v, 4)

    return {
        # ---- electrical outputs ----
        "freq_hz": freq_actual,
        "L_tx_uH": r5(L_tx * 1e6),
        "R_tx_ac": r5(R_tx),
        "Q_tx":    r5(Q_tx),
        "L_rx_uH": r5(L_rx * 1e6),
        "R_rx_ac": r5(R_rx),
        "Q_rx":    r5(Q_rx),
        "M_uH":    r5(M * 1e6),
        "k":       r5(k),
        # ---- TX geometry / stackup ----
        "tx_turns":        float(sample.get("tx_turns",    p.tx_turns)),
        "tx_l2_turns":     int(sample.get("tx_l2_turns",   p.tx_l2_turns)),
        "tx_width":        r4(float(sample.get("tx_w",     p.tx_w_mm))),
        "tx_od_mm":        r4(float(sample.get("tx_od",    p.tx_od_mm))),
        "tx_spacing_mm":   r4(float(sample.get("tx_s",     p.tx_spacing_mm))),
        "tx_outer_gap_mm": r4(float(sample.get("tx_l1l2_gap", p.tx_l1l2_gap_mm))),
        "tx_inner_gap_mm": 0.0,           # TX uses L1+L2 only
        "tx_topology":     "series",
        "tx_port_inside":  False,
        "tx_layers":       [dict(l) for l in _TX_LAYERS],
        # ---- RX geometry / stackup ----
        "rx_turns":        float(sample.get("rx_turns",      p.rx_turns)),
        "rx_width":        r4(float(sample.get("rx_w",       p.rx_w_mm))),
        "rx_od_mm":        r4(float(sample.get("rx_od",      p.rx_od_mm))),
        "rx_spacing_mm":   r4(float(sample.get("rx_s",       p.rx_spacing_mm))),
        "rx_outer_gap_mm": r4(float(sample.get("rx_outer_gap", p.rx_outer_gap_mm))),
        "rx_inner_gap_mm": r4(float(sample.get("rx_inner_gap", p.rx_inner_gap_mm))),
        "rx_topology":     rx_topo_name,
        "rx_port_inside":  True,
        "rx_layers":       [dict(l) for l in _RX_LAYERS],
        # ---- global ----
        "pcb_gap_mm":            r4(float(sample.get("pcb_gap", p.pcb_gap_mm))),
        "rx_ground_disc_dia_mm": r4(float(sample.get("rx_ground_disc_dia",
                                                     p.rx_ground_disc_dia_mm))),
        "tx_ground_enabled":     bool(p.tx_ground_enabled),
    }


# ---------------------------------------------------------------------------
# Public worker
# ---------------------------------------------------------------------------

def run_single_sim(p: SimParams, sample: dict = None) -> dict:
    """
    Complete 2-port FastHenry simulation.

    Returns a verbose result dict (see _parse_result schema) on success,
    or {"error": "<reason>"} on failure. Never raises.
    """
    if sample is None:
        sample = {}
    overall_start = time.time()
    pid = os.getpid()

    reason = _check_feasibility(p)
    if reason is not None:
        return {"error": f"infeasible: {reason}",
                "pid": pid, "elapsed_sec": round(time.time() - overall_start, 2)}

    last_error = "unknown error"
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2.0 * attempt)

        attempt_start = time.time()
        tmp_dir = tempfile.mkdtemp(prefix="fh_sim_")
        try:
            inp_path = os.path.join(tmp_dir, "coil.inp")
            _write_inp(p, inp_path)
            zc_path = _run_fasthenry(inp_path, p)
            result  = _parse_result(zc_path, p, sample)
            result["pid"]         = pid
            result["elapsed_sec"] = round(time.time() - overall_start, 2)
            return result

        except Exception as exc:
            last_error = str(exc)
            if time.time() - attempt_start >= 30.0:
                break

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"error": last_error,
            "pid": pid, "elapsed_sec": round(time.time() - overall_start, 2)}


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(params_list: list, samples_list: list = None,
              max_workers: int = 4, progress_cb=None) -> list:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if samples_list is None:
        samples_list = [{}] * len(params_list)

    total   = len(params_list)
    results = [None] * total
    futures = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, (p, s) in enumerate(zip(params_list, samples_list)):
            fut = pool.submit(run_single_sim, p, s)
            futures[fut] = idx

        done_count = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = {"error": f"executor error: {exc}"}
            done_count += 1
            if progress_cb is not None:
                try:
                    progress_cb(done_count, total)
                except Exception:
                    pass

    return results
