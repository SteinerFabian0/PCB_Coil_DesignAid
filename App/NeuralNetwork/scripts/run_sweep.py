#!/usr/bin/env python3
"""
Run a FastHenry sweep over a pre-generated sample file.

Results file format (results.json):
  {
    "meta":    {"n_samples": int, "seed": int, "config_hash": str, ...},
    "results": [{"uuid": "...", "freq_hz": ..., "L_tx_uH": ..., ...}, ...]
  }

Each result is a verbose dict matching the schema train_surrogate.py expects:
  freq_hz, L_tx_uH, R_tx_ac, Q_tx, L_rx_uH, R_rx_ac, Q_rx, M_uH, k,
  tx_turns, tx_l2_turns, tx_width, tx_od_mm, tx_spacing_mm,
  tx_outer_gap_mm, tx_inner_gap_mm, tx_topology, tx_port_inside, tx_layers,
  rx_turns, rx_width, rx_od_mm, rx_spacing_mm,
  rx_outer_gap_mm, rx_inner_gap_mm, rx_topology, rx_port_inside, rx_layers,
  pcb_gap_mm, rx_ground_disc_dia_mm, tx_ground_enabled, uuid,
  pid, elapsed_sec.

Usage:
    python run_sweep.py --samples lhs_samples.json
                        --out     results.json
                        --workers 8 --timeout 360
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE     = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(os.path.dirname(_HERE))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parallel_sim import SimParams, run_single_sim  # noqa: E402


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def _sample_to_simparams(d: dict, timeout_sec: float) -> SimParams:
    uid = d.get("uuid", "")
    return SimParams(
        tx_turns       = float(d["tx_turns"]),
        tx_od_mm       = float(d["tx_od"]),
        tx_w_mm        = float(d["tx_w"]),
        tx_spacing_mm  = float(d.get("tx_s", 0.16)),
        tx_l1l2_gap_mm = float(d.get("tx_l1l2_gap", 0.2)),
        tx_l2_turns    = int(d.get("tx_l2_turns", 0)),
        tx_nwinc       = 2 if float(d.get("tx_w", 0.4)) < 0.4 else 3,
        rx_turns       = float(d["rx_turns"]),
        rx_od_mm       = float(d["rx_od"]),
        rx_w_mm        = float(d["rx_w"]),
        rx_spacing_mm  = float(d.get("rx_s", 0.16)),
        rx_outer_gap_mm = float(d.get("rx_outer_gap", 0.2)),
        rx_inner_gap_mm = float(d.get("rx_inner_gap", 0.6)),
        rx_topo        = int(d.get("rx_topo", 0)),
        rx_nwinc       = 2 if float(d.get("rx_w", 0.5)) < 0.4 else 3,
        pcb_gap_mm     = float(d.get("pcb_gap", 2.6)),
        freq_hz        = float(d.get("freq", 360_000.0)),
        resolution_mm  = float(d.get("resolution", 1.2)),
        rx_ground_disc_dia_mm = float(d.get("rx_ground_disc_dia", 20.0)),
        tx_ground_enabled     = True,
        timeout_sec    = float(timeout_sec),
        tag            = uid[:8] if uid else "????????",
    )


def _load_samples(path: str, timeout_sec: float):
    with open(path) as f:
        payload = json.load(f)
    samples = payload["samples"]
    params  = [_sample_to_simparams(d, timeout_sec) for d in samples]
    return params, samples, payload.get("meta", {})


# ---------------------------------------------------------------------------
# Results file I/O — dict-of-dicts format
# ---------------------------------------------------------------------------

def _load_existing(path: str) -> dict:
    """Return {uuid: result_dict} for already-written rows."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for r in data.get("results", []):
        uid = r.get("uuid") if isinstance(r, dict) else None
        if uid:
            out[uid] = r
    return out


def _flush(out_path: str, results: dict, meta: dict) -> None:
    payload = {
        "meta":    {**meta, "n_results": len(results)},
        "results": list(results.values()),
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))   # compact — no spaces
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Stop flag
# ---------------------------------------------------------------------------

def _stop_flag_path(out_path: str) -> str:
    return os.path.join(os.path.dirname(out_path) or ".", "STOP_SWEEP")


def _stop_requested(out_path: str) -> bool:
    return os.path.exists(_stop_flag_path(out_path))


# ---------------------------------------------------------------------------
# Progress line
# ---------------------------------------------------------------------------

def _one_line_report(result: dict, uuid_val: str, ok: bool) -> str:
    short = uuid_val[:8] if uuid_val else "????????"
    t = float(result.get("elapsed_sec", 0.0)) if isinstance(result, dict) else 0.0
    if not ok:
        err = (result.get("error", "") if isinstance(result, dict) else "")[:80]
        return f"{short}  FAIL  {t:6.1f}s  {err}"
    return f"{short}  OK    {t:6.1f}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="FastHenry sweep runner")
    parser.add_argument("--samples",          required=True)
    parser.add_argument("--out",              required=True)
    parser.add_argument("--workers",          type=int, default=8)
    parser.add_argument("--timeout",          type=int, default=360)
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        dest="ckpt_every")
    args = parser.parse_args()

    params_list, samples_list, sample_meta = _load_samples(args.samples, args.timeout)

    run_meta = {
        "n_samples":   sample_meta.get("n_samples", len(samples_list)),
        "seed":        sample_meta.get("seed"),
        "config_hash": sample_meta.get("config_hash"),
        "workers":     args.workers,
        "timeout":     args.timeout,
    }

    existing   = _load_existing(args.out)
    skip_uuids = set(existing)

    todo, todo_samples = [], []
    for p, s in zip(params_list, samples_list):
        uid = s.get("uuid")
        if uid and uid not in skip_uuids:
            todo.append(p)
            todo_samples.append(s)

    n_skip = len(params_list) - len(todo)

    print("=" * 65, flush=True)
    print(f"Samples     : {len(params_list)}", flush=True)
    print(f"Workers     : {args.workers}", flush=True)
    print(f"Timeout     : {args.timeout}s", flush=True)
    print(f"Skip        : {n_skip}", flush=True)
    print(f"To simulate : {len(todo)}", flush=True)
    print(f"Output file : {args.out}", flush=True)
    print("=" * 65, flush=True)

    if not todo:
        print("Nothing to do.", flush=True)
        _flush(args.out, existing, run_meta)
        return 0

    sf = _stop_flag_path(args.out)
    if os.path.exists(sf):
        try:
            os.remove(sf)
        except OSError:
            pass

    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = dict(existing)
    wall_start = time.time()
    since_ckpt = 0

    pool = ProcessPoolExecutor(max_workers=args.workers)
    stop_signaled = False
    try:
        fut_to_sample: dict = {}
        for p, s in zip(todo, todo_samples):
            fut = pool.submit(run_single_sim, p, s)
            fut_to_sample[fut] = (p, s)

        for fut in as_completed(fut_to_sample):
            if fut.cancelled():
                continue
            p, sample = fut_to_sample[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {"error": f"worker crash: {exc}", "elapsed_sec": 0.0}

            uid = sample.get("uuid", "")

            if "error" not in r and uid:
                r["uuid"] = uid
                results[uid] = r
                print(_one_line_report(r, uid, ok=True), flush=True)
            else:
                print(_one_line_report(r, uid, ok=False), flush=True)

            since_ckpt += 1
            if since_ckpt >= args.ckpt_every:
                _flush(args.out, results, run_meta)
                since_ckpt = 0
                print("[checkpoint]", flush=True)

            if not stop_signaled and _stop_requested(args.out):
                print("[STOP] Stop flag — terminating workers.", flush=True)
                stop_signaled = True
                _flush(args.out, results, run_meta)
                try:
                    for proc in pool._processes.values():
                        proc.terminate()
                except Exception:
                    pass
                break
    finally:
        pool.shutdown(wait=False)

    _flush(args.out, results, run_meta)

    wall = time.time() - wall_start
    ok   = len(results)
    print("=" * 65, flush=True)
    print(f"Done in {wall:.1f}s  |  OK: {ok}", flush=True)
    print(f"Output: {args.out}", flush=True)
    print("=" * 65, flush=True)

    if os.path.exists(sf):
        try:
            os.remove(sf)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
