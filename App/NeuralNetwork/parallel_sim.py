#!/usr/bin/env python3
"""
Parallel FastHenry simulation worker — combined TX+RX 2-port sims.

Each SimParams describes a complete TX+RX coil pair. run_single_sim():
  - Validates both coil geometries before touching disk.
  - Writes a single combined .inp (TX at z=0, RX shifted up by pcb_gap_mm).
  - Runs FastHenry via COM automation with a configurable timeout.
  - Parses the 2×2 Zc.mat and returns L_tx, L_rx, M, R_tx, R_rx, k, Q_tx, Q_rx.
  - Deletes the temp directory on exit (success or failure).

Usage
-----
    from App.Modules.parallel_sim import SimParams, run_batch

    params = SimParams(...)
    results = run_batch([params, ...], max_workers=4)
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
_APP_ROOT    = os.path.dirname(_HERE)
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")

for _p in (_HERE, _MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parametric_coil as pc
import zc_parser


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """All inputs for one combined TX+RX FastHenry simulation."""

    # --- TX geometry (fixed OD, spacing, layers; variable turns & width) ---
    tx_turns:          float
    tx_trace_width_mm: float
    tx_od_mm:          float  = 52.0
    tx_spacing_mm:     float  = 0.16
    tx_outer_gap_mm:   float  = 0.2
    tx_inner_gap_mm:   float  = 1.3
    tx_topology:       str    = "parallel"
    # L1, L2, L3 active (1 oz each); L4 inactive
    tx_layers: list = field(default_factory=lambda: [
        {"active": True,  "copper_oz": 1.0},
        {"active": True,  "copper_oz": 1.0},
        {"active": True,  "copper_oz": 1.0},
        {"active": False, "copper_oz": 1.0},
    ])

    # --- RX geometry (variable OD, turns, width, topology) ---
    rx_turns:          float  = 10.0
    rx_trace_width_mm: float  = 0.5
    rx_od_mm:          float  = 50.0
    rx_spacing_mm:     float  = 0.16
    rx_outer_gap_mm:   float  = 0.2
    rx_inner_gap_mm:   float  = 0.6
    rx_topology:       str    = "parallel"
    # All 4 layers active: outer 1 oz, inner 0.5 oz
    rx_layers: list = field(default_factory=lambda: [
        {"active": True, "copper_oz": 1.0},
        {"active": True, "copper_oz": 0.5},
        {"active": True, "copper_oz": 0.5},
        {"active": True, "copper_oz": 1.0},
    ])

    # --- Per-side meshing / port choice ---
    tx_port_inside: bool = False
    rx_port_inside: bool = True
    tx_nhinc:       int  = 1
    tx_nwinc:       int  = 3
    rx_nhinc:       int  = 1
    rx_nwinc:       int  = 3
 
    # --- Global / sim params ---
    pcb_gap_mm:    float = 2.6
    freq_hz:       float = 130_000.0
    fmin_hz:       float = 110_000.0
    fmax_hz:       float = 140_000.0
    freq_ndec:     int   = 1
    resolution_mm: float = 1.0
    timeout_sec:   float = 240.0   # 4 minutes — 2-port sims are heavier

    tag:     str = ""
    sample_id: int = -1

# ---------------------------------------------------------------------------
# Geometry feasibility pre-check
# ---------------------------------------------------------------------------

def _check_feasibility(p: SimParams) -> Optional[str]:
    for label, od, w, s, t, og, ig, layers in [
        ("TX", p.tx_od_mm, p.tx_trace_width_mm, p.tx_spacing_mm,
         p.tx_turns, p.tx_outer_gap_mm, p.tx_inner_gap_mm, p.tx_layers),
        ("RX", p.rx_od_mm, p.rx_trace_width_mm, p.rx_spacing_mm,
         p.rx_turns, p.rx_outer_gap_mm, p.rx_inner_gap_mm, p.rx_layers),
    ]:
        sp = pc.SpiralParams(od_mm=od, trace_width_mm=w, spacing_mm=s,
                             turns=t, resolution_mm=p.resolution_mm)
        ok, msg = pc.validate_spiral(sp)
        if not ok:
            return f"{label}: {msg}"

        stackup = pc.StackUp(
            slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
                   for l in layers],
            outer_gap_mm=og,
            inner_gap_mm=ig,
        )
        ok, msg = pc.validate_stackup(stackup)
        if not ok:
            return f"{label}: {msg}"

    return None


# ---------------------------------------------------------------------------
# .inp generation — combined 2-port file
# ---------------------------------------------------------------------------

def _write_inp(p: SimParams, inp_path: str) -> None:
    def _layer_data(od, w, s, t, og, ig, layers):
        sp = pc.SpiralParams(od_mm=od, trace_width_mm=w, spacing_mm=s,
                             turns=t, resolution_mm=p.resolution_mm)
        stackup = pc.StackUp(
            slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
                   for l in layers],
            outer_gap_mm=og, inner_gap_mm=ig,
        )
        return pc.active_layer_data(sp, stackup)

    tx_ld = _layer_data(p.tx_od_mm, p.tx_trace_width_mm, p.tx_spacing_mm,
                        p.tx_turns, p.tx_outer_gap_mm, p.tx_inner_gap_mm,
                        p.tx_layers)
    rx_ld = _layer_data(p.rx_od_mm, p.rx_trace_width_mm, p.rx_spacing_mm,
                        p.rx_turns, p.rx_outer_gap_mm, p.rx_inner_gap_mm,
                        p.rx_layers)

    pc.write_combined_tx_rx_inp(
        tx_ld, rx_ld, inp_path,
        tx_w_mm=p.tx_trace_width_mm,
        rx_w_mm=p.rx_trace_width_mm,
        tx_topology=p.tx_topology,
        rx_topology=p.rx_topology,
        tx_port_inside=p.tx_port_inside,
        rx_port_inside=p.rx_port_inside,
        pcb_gap_mm=p.pcb_gap_mm,
        tx_nhinc=p.tx_nhinc, tx_nwinc=p.tx_nwinc,
        rx_nhinc=p.rx_nhinc, rx_nwinc=p.rx_nwinc,
        fmin=p.fmin_hz, fmax=p.fmax_hz, freq_ndec=p.freq_ndec,
    )


# ---------------------------------------------------------------------------
# FastHenry execution (COM automation)
# ---------------------------------------------------------------------------

def _run_fasthenry(inp_path: str, p: SimParams) -> str:
    import pythoncom
    import win32com.client as w32c

    pythoncom.CoInitialize()
    fh = w32c.Dispatch("FastHenry2.Document")
    try:
        # Do NOT pass -i. FastHenry's default iteration cap is correct;
        # overriding it with any value (even a large one) triggers a C-style
        # argv bug in the solver that zeros the cap and produces all-NaN output.
        cmdline = f'"{inp_path}"'

        started = fh.Run(cmdline)
        if not started:
            raise RuntimeError("FastHenry2.Run() returned False")

        t0 = time.time()
        while fh.IsRunning:
            if time.time() - t0 > p.timeout_sec:
                try:
                    fh.Stop()
                except Exception:
                    pass
                raise RuntimeError(
                    f"FastHenry timed out after {p.timeout_sec:.0f}s"
                )
            time.sleep(0.25)
    finally:
        try:
            fh.Quit
        except Exception:
            pass

    zc_path = os.path.join(os.path.dirname(inp_path), "Zc.mat")
    if not os.path.exists(zc_path):
        raise FileNotFoundError("Zc.mat not produced by FastHenry")
    return zc_path


# ---------------------------------------------------------------------------
# Result parsing — 2×2 Z-matrix
# ---------------------------------------------------------------------------

def _parse_result(zc_path: str, p: SimParams) -> dict:
    blocks = zc_parser.parse_zc_mat(zc_path)

    n_ports = zc_parser.port_count(blocks)
    if n_ports != 2:
        raise ValueError(
            f"Expected 2-port Zc.mat (2×2 matrix) but got {n_ports}-port. "
            "Check that the .inp has exactly two .external statements."
        )

    freq_actual, matrix = zc_parser.matrix_at(blocks, p.freq_hz)

    z11 = matrix[0][0]   # TX self-impedance
    z22 = matrix[1][1]   # RX self-impedance
    z12 = matrix[0][1]   # mutual impedance  (= Z21 for reciprocal structure)

    for name, z in [("Z11", z11), ("Z22", z22), ("Z12", z12)]:
        if math.isnan(z.real) or math.isnan(z.imag):
            raise ValueError(f"FastHenry produced NaN in {name} — solver did not converge")

    omega = 2.0 * math.pi * p.freq_hz

    L_tx_H  = z11.imag / omega
    L_rx_H  = z22.imag / omega
    M_H     = z12.imag / omega       # mutual inductance

    R_tx    = z11.real
    R_rx    = z22.real

    # Coupling coefficient k = M / sqrt(L_tx * L_rx)
    denom = math.sqrt(L_tx_H * L_rx_H) if L_tx_H > 0 and L_rx_H > 0 else 0.0
    k     = M_H / denom if denom > 0 else 0.0

    Q_tx  = (omega * L_tx_H) / R_tx if R_tx > 0 else 0.0
    Q_rx  = (omega * L_rx_H) / R_rx if R_rx > 0 else 0.0

    # DC resistance: lowest-frequency block diagonal
    z_dc_block = min(blocks, key=lambda b: b["frequency"])
    R_tx_dc = z_dc_block["matrix"][0][0].real
    R_rx_dc = z_dc_block["matrix"][1][1].real

    return {
        "ok":         True,
        "tag":        p.tag,
        "sample_id":  p.sample_id,
        "freq_hz":    freq_actual,
        # TX
        "L_tx_uH":    L_tx_H  * 1e6,
        "R_tx_ac":    R_tx,
        "R_tx_dc":    R_tx_dc,
        "Q_tx":       Q_tx,
        # RX
        "L_rx_uH":    L_rx_H  * 1e6,
        "R_rx_ac":    R_rx,
        "R_rx_dc":    R_rx_dc,
        "Q_rx":       Q_rx,
        # Coupling
        "M_uH":       M_H     * 1e6,
        "k":          k,
        # Geometry echo (full — needed by the append step for NN features)
        "tx_turns":        p.tx_turns,
        "tx_width":        p.tx_trace_width_mm,
        "tx_od_mm":        p.tx_od_mm,
        "tx_spacing_mm":   p.tx_spacing_mm,
        "tx_outer_gap_mm": p.tx_outer_gap_mm,
        "tx_inner_gap_mm": p.tx_inner_gap_mm,
        "tx_topology":     p.tx_topology,
        "tx_port_inside":  p.tx_port_inside,
        "tx_layers":       p.tx_layers,
        "rx_turns":        p.rx_turns,
        "rx_width":        p.rx_trace_width_mm,
        "rx_od_mm":        p.rx_od_mm,
        "rx_spacing_mm":   p.rx_spacing_mm,
        "rx_outer_gap_mm": p.rx_outer_gap_mm,
        "rx_inner_gap_mm": p.rx_inner_gap_mm,
        "rx_topology":     p.rx_topology,
        "rx_port_inside":  p.rx_port_inside,
        "rx_layers":       p.rx_layers,
        "pcb_gap_mm":      p.pcb_gap_mm,
    }


# ---------------------------------------------------------------------------
# Public worker — one simulation end-to-end
# ---------------------------------------------------------------------------

def run_single_sim(p: SimParams) -> dict:
    """
    Complete 2-port FastHenry simulation for one TX+RX parameter point.

    Returns a dict with at minimum:
      {"ok": True,  "tag": p.tag, ...result fields...}
      {"ok": False, "tag": p.tag, "error": "<reason>"}

    Never raises — all exceptions are caught so a ProcessPoolExecutor
    worker never crashes the pool.

    Includes "pid" and "elapsed_sec" in every result.

    Fast failures (< 30 s) are retried up to 2 times with exponential backoff
    (2 s, then 4 s) to recover from transient Windows COM / process-limit
    errors that occur when many FastHenry instances run concurrently.
    """
    overall_start = time.time()
    pid = os.getpid()

    reason = _check_feasibility(p)
    if reason is not None:
        return {
            "ok": False, "tag": p.tag, "sample_id": p.sample_id,
            "error": f"infeasible: {reason}",
            "pid": pid, "elapsed_sec": round(time.time() - overall_start, 2),
        }

    last_error = "unknown error"
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2.0 * attempt)   # 2 s before attempt 2, 4 s before attempt 3

        attempt_start = time.time()
        tmp_dir = tempfile.mkdtemp(prefix="fh_sim_")
        try:
            inp_path = os.path.join(tmp_dir, "coil.inp")
            _write_inp(p, inp_path)
            zc_path = _run_fasthenry(inp_path, p)
            result  = _parse_result(zc_path, p)
            result["pid"]         = pid
            result["elapsed_sec"] = round(time.time() - overall_start, 2)
            return result

        except Exception as exc:
            last_error = str(exc)
            attempt_elapsed = time.time() - attempt_start
            # Long-running failures (timeout, NaN) won't improve with retries.
            if attempt_elapsed >= 30.0:
                break

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "ok": False, "tag": p.tag, "sample_id": p.sample_id, "error": last_error,
        "pid": pid, "elapsed_sec": round(time.time() - overall_start, 2),
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    params_list: list,
    max_workers: int = 4,
    progress_cb=None,
) -> list:
    """
    Run a list of SimParams in parallel via ProcessPoolExecutor.

    progress_cb(done: int, total: int) is called after each job finishes.
    Returns results in the same order as params_list.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    total   = len(params_list)
    results = [None] * total
    futures = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, p in enumerate(params_list):
            fut = pool.submit(run_single_sim, p)
            futures[fut] = idx

        done_count = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = {
                    "ok": False,
                    "tag": params_list[idx].tag,
                    "error": f"executor error: {exc}",
                }
            done_count += 1
            if progress_cb is not None:
                try:
                    progress_cb(done_count, total)
                except Exception:
                    pass

    return results
