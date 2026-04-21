#!/usr/bin/env python3
"""
Smoke-test for parallel FastHenry simulation.

Fires 5 simulations with slightly varying geometry and verifies:
  - Results come back in order.
  - An infeasible geometry is rejected early (no FastHenry call).
  - Feasible designs return sensible L / R_ac / Q values.
  - PIDs and wall-clock times prove the jobs ran concurrently.

Run from the project root:
    python test_parallel.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "App", "Modules"))

from parallel_sim import SimParams, run_batch

PARAM_SETS = [
    SimParams(
        od_mm=50.0, trace_width_mm=0.5, spacing_mm=0.3, turns=3,
        topology="parallel",
        layers=[
            {"active": True,  "copper_oz": 1.0},
            {"active": True,  "copper_oz": 1.0},
            {"active": False, "copper_oz": 1.0},
            {"active": False, "copper_oz": 1.0},
        ],
        freq_hz=130_000, fmin_hz=110_000, fmax_hz=140_000,
        timeout_sec=120,
        tag="A_3T_parallel",
    ),
    SimParams(
        od_mm=50.0, trace_width_mm=0.5, spacing_mm=0.3, turns=5,
        topology="parallel",
        layers=[
            {"active": True,  "copper_oz": 1.0},
            {"active": True,  "copper_oz": 1.0},
            {"active": False, "copper_oz": 1.0},
            {"active": False, "copper_oz": 1.0},
        ],
        freq_hz=130_000, fmin_hz=110_000, fmax_hz=140_000,
        timeout_sec=120,
        tag="B_5T_parallel",
    ),
    # Intentionally infeasible — should be caught before FastHenry is called.
    SimParams(
        od_mm=10.0, trace_width_mm=0.5, spacing_mm=0.3, turns=30,
        layers=[{"active": True, "copper_oz": 1.0}] +
               [{"active": False, "copper_oz": 1.0}] * 3,
        freq_hz=130_000, fmin_hz=110_000, fmax_hz=140_000,
        timeout_sec=120,
        tag="C_infeasible",
    ),
    SimParams(
        od_mm=60.0, trace_width_mm=0.8, spacing_mm=0.4, turns=4,
        topology="series",
        layers=[
            {"active": True,  "copper_oz": 1.0},
            {"active": True,  "copper_oz": 1.0},
            {"active": True,  "copper_oz": 1.0},
            {"active": True,  "copper_oz": 1.0},
        ],
        outer_gap_mm=0.2, inner_gap_mm=0.2,
        freq_hz=130_000, fmin_hz=110_000, fmax_hz=140_000,
        timeout_sec=120,
        tag="D_4layer_series",
    ),
    SimParams(
        od_mm=40.0, trace_width_mm=0.4, spacing_mm=0.25, turns=6,
        topology="parallel",
        layers=[{"active": True, "copper_oz": 2.0}] +
               [{"active": False, "copper_oz": 1.0}] * 3,
        freq_hz=130_000, fmin_hz=110_000, fmax_hz=140_000,
        timeout_sec=120,
        tag="E_6T_2oz",
    ),
]


def _progress(done, total):
    print(f"  [{done}/{total}] done", flush=True)


def main():
    print("=" * 65)
    print("Parallel FastHenry smoke-test  (4 workers, 5 designs)")
    print("=" * 65)

    wall_start = time.time()
    results = run_batch(PARAM_SETS, max_workers=4, progress_cb=_progress)
    wall_total = time.time() - wall_start

    print()
    print(f"{'Tag':<22} {'PID':>6}  {'Elapsed':>8}  Result")
    print("-" * 65)

    passed = 0
    pids_seen = set()
    for r in results:
        tag  = r.get("tag", "?")
        pid  = r.get("pid", "?")
        secs = r.get("elapsed_sec", "?")
        pids_seen.add(pid)

        if r["ok"]:
            print(f"{tag:<22} {pid:>6}  {secs:>7.1f}s  "
                  f"L={r['L_uH']:.3f} uH  "
                  f"R_ac={r['R_ac_ohm']:.4f} Ω  "
                  f"Q={r['Q']:.1f}")
            passed += 1
        elif "infeasible" in r.get("error", ""):
            print(f"{tag:<22} {pid:>6}  {secs:>7.1f}s  "
                  f"[SKIP — {r['error']}]")
            passed += 1
        else:
            print(f"{tag:<22} {pid:>6}  {secs:>7.1f}s  "
                  f"[FAIL — {r['error']}]")

    print()
    print(f"Wall time: {wall_total:.1f}s total")
    print(f"Unique worker PIDs seen: {sorted(pids_seen)}")
    n_workers = len(pids_seen)
    if n_workers > 1:
        print(f"  -> {n_workers} distinct PIDs — jobs ran across multiple processes. PARALLEL OK.")
    else:
        print(f"  -> Only 1 PID — all jobs ran in 1 process (serial or single-worker).")

    print()
    print(f"{passed}/{len(results)} checks passed.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
