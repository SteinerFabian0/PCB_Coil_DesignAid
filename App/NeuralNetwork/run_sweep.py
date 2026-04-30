#!/usr/bin/env python3
"""
Run a FastHenry sweep over a pre-generated sample file.
 
Reads ``lhs_samples.json`` (produced by generate_lhs_samples.py) and writes a
per-run ``solver_output.json`` file.  The global / training file is *not*
touched — merging happens in a separate ``append`` step that computes
derived NN input features.
 
Per-sample stdout is a single line::
    S000123 OK     12.4s
    S000124 FAIL  360.0s  (timeout)

Usage::
 
    python run_sweep.py --samples lhs_samples.json
                        --out     solver_output.json
                        --workers 8 --timeout 360
                        --from-idx 500 --to-idx 1000
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import sys
import time

_HERE        = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT    = os.path.dirname(_HERE)
_SIMDATA_DIR = os.path.join(_APP_ROOT, "SimulationData")

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
 
from parallel_sim import SimParams, run_single_sim  # noqa: E402
 
 
# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------
 
def sample_to_simparams(d: dict, timeout_sec: float) -> SimParams:
    return SimParams(
        tx_turns=d["tx_turns"],
        tx_trace_width_mm=d["tx_trace_width_mm"],
        tx_od_mm=d["tx_od_mm"],
        tx_spacing_mm=d.get("tx_spacing_mm", 0.16),
        tx_outer_gap_mm=d.get("tx_outer_gap_mm", 0.2),
        tx_inner_gap_mm=d.get("tx_inner_gap_mm", 1.3),
        tx_topology=d.get("tx_topology", "parallel"),
        tx_layers=d.get("tx_layers", [
            {"active": True,  "copper_oz": 1.0},
            {"active": True,  "copper_oz": 1.0},
            {"active": True,  "copper_oz": 1.0},
            {"active": False, "copper_oz": 1.0},
        ]),
        rx_turns=d["rx_turns"],
        rx_trace_width_mm=d["rx_trace_width_mm"],
        rx_od_mm=d["rx_od_mm"],
        rx_spacing_mm=d.get("rx_spacing_mm", 0.16),
        rx_outer_gap_mm=d.get("rx_outer_gap_mm", 0.2),
        rx_inner_gap_mm=d.get("rx_inner_gap_mm", 0.6),
        rx_topology=d.get("rx_topology", "parallel"),
        rx_layers=d.get("rx_layers", [
            {"active": True, "copper_oz": 1.0},
            {"active": True, "copper_oz": 0.5},
            {"active": True, "copper_oz": 0.5},
            {"active": True, "copper_oz": 1.0},
        ]),
        tx_port_inside=bool(d.get("tx_port_inside", False)),
        rx_port_inside=bool(d.get("rx_port_inside", True)),
        tx_nhinc=int(d.get("nhinc", 1)),
        tx_nwinc=int(d.get("nwinc", 3)),
        rx_nhinc=int(d.get("rx_nhinc", d.get("nhinc", 1))),
        rx_nwinc=int(d.get("rx_nwinc", d.get("nwinc", 3))),
        pcb_gap_mm=d.get("pcb_gap_mm", 2.6),
        freq_hz=d.get("freq_hz", 125_000.0),
        fmin_hz=d.get("fmin_hz", d.get("freq_hz", 125_000.0)),
        fmax_hz=d.get("fmax_hz", d.get("freq_hz", 125_000.0)),
        freq_ndec=d.get("freq_ndec", 0),
        resolution_mm=d.get("resolution_mm", 1.2),
        timeout_sec=float(timeout_sec),
        ground_circle_dia_mm=float(d.get("ground_circle_dia_mm", 0.0)),
        tag=d.get("tag", f"S{int(d.get('id', -1)):06d}"),
        sample_id=int(d.get("id", -1)),
    )
 
 
def _load_samples(path: str, timeout_sec: float):
    with open(path) as f:
        payload = json.load(f)
    samples = payload["samples"]
    params  = [sample_to_simparams(d, timeout_sec) for d in samples]
    return params, samples, payload.get("meta", {})


def _load_existing(path: str) -> dict:
    """Return {uuid: result_dict} for already-written local results."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for r in data.get("results", []):
        uuid_val = r.get("uuid")
        if uuid_val:
            out[uuid_val] = r
    return out


def _load_global_uuids(global_path: str) -> set:
    """Return the set of UUIDs already present in global_results.json."""
    if not os.path.exists(global_path):
        return set()
    try:
        with open(global_path) as f:
            data = json.load(f)
    except Exception:
        return set()
    return {r["uuid"] for r in data.get("results", []) if r.get("uuid")}


def _flush(out_path: str, meta: dict, results: dict) -> None:
    payload = {"meta": meta, "results": list(results.values())}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, out_path)


def _stop_flag_path(out_path: str) -> str:
    return os.path.join(os.path.dirname(out_path) or ".", "STOP_SWEEP")


def _stop_requested(out_path: str) -> bool:
    return os.path.exists(_stop_flag_path(out_path))


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------
 
def _one_line_report(r: dict) -> str:
    batch = r.get("batch")
    snum  = r.get("sample_num")
    t     = float(r.get("elapsed_sec", 0.0))
    if batch is not None and snum is not None:
        label = f"B{batch:02d}·S{snum:05d}"
    else:
        label = r.get("tag", "?")
    if r.get("ok"):
        return f"{label}  OK    {t:6.1f}s"
    err = r.get("error", "")[:80]
    return f"{label}  FAIL  {t:6.1f}s  {err}"

def main() -> int:
    parser = argparse.ArgumentParser(description="FastHenry sweep runner")
    parser.add_argument("--samples", default=os.path.join(_SIMDATA_DIR, "lhs_samples.json"))
    parser.add_argument("--out",     default=None)
    parser.add_argument("--batch",   type=int, default=None, dest="batch",
                        help="Batch number (derives --out if not specified)")
    parser.add_argument("--workers",          type=int, default=8)
    parser.add_argument("--timeout",          type=int, default=360)
    parser.add_argument("--checkpoint-every", type=int, default=25, dest="ckpt_every")
    parser.add_argument("--from-idx",         type=int, default=0,   dest="from_idx")
    parser.add_argument("--to-idx",           type=int, default=None, dest="to_idx")
    args = parser.parse_args()

    params_list, samples_list, sample_meta = _load_samples(args.samples, args.timeout)

    if args.out is None:
        batch_num = args.batch if args.batch is not None else sample_meta.get("batch")
        if batch_num is None:
            print("ERROR: --out or --batch must be specified", file=sys.stderr)
            return 1
        args.out = os.path.join(_SIMDATA_DIR, f"results_batch_{batch_num}.json")

    batch_num = args.batch if args.batch is not None else sample_meta.get("batch")

    from_idx = max(0, args.from_idx)
    to_idx   = args.to_idx if args.to_idx is not None else len(params_list)
    to_idx   = min(to_idx, len(params_list))
    target   = params_list[from_idx:to_idx]
    target_samples = samples_list[from_idx:to_idx]

    existing    = _load_existing(args.out)
    global_uuids = _load_global_uuids(os.path.join(_SIMDATA_DIR, "global_results.json"))
    skip_uuids  = set(existing) | global_uuids

    todo = []
    todo_samples = []
    for i, p in enumerate(target):
        sample = target_samples[i]
        uuid_val = sample.get("uuid")
        if uuid_val and uuid_val not in skip_uuids:
            todo.append(p)
            todo_samples.append(sample)

    n_in_local = sum(1 for s in target_samples if s.get("uuid") in existing)
    n_in_global = sum(1 for s in target_samples if s.get("uuid") in global_uuids and s.get("uuid") not in existing)

    print("=" * 65, flush=True)
    print(f"Sweep range [{from_idx}:{to_idx}]  ({len(target)} samples)", flush=True)
    print(f"Workers     : {args.workers}", flush=True)
    print(f"Timeout     : {args.timeout}s", flush=True)
    print(f"Skip (local) : {n_in_local}  Skip (global): {n_in_global}", flush=True)
    print(f"To simulate : {len(todo)}", flush=True)
    print(f"Output file : {args.out}", flush=True)
    print("=" * 65, flush=True)

    if not todo:
        print("Nothing to do.", flush=True)
        # Still refresh the meta so the UI sees the requested range.
        meta = {**sample_meta,
                "from_idx": from_idx, "to_idx": to_idx,
                "range_total": len(target)}
        _flush(args.out, meta, existing)
        return 0

    if os.path.exists(_stop_flag_path(args.out)):
        try:
            os.remove(_stop_flag_path(args.out))
        except OSError:
            pass
 
    from concurrent.futures import ProcessPoolExecutor, as_completed

    meta = {**sample_meta,
            "workers":    args.workers,
            "timeout":    args.timeout,
            "from_idx":   from_idx,
            "to_idx":     to_idx,
            "range_total": len(target)}

    results = dict(existing)
    sample_by_id = {p.sample_id: s for p, s in zip(target, target_samples)}

    wall_start = time.time()
    since_ckpt = 0

    pool = ProcessPoolExecutor(max_workers=args.workers)
    stop_signaled = False
    try:
        fut_map = {pool.submit(run_single_sim, p): p for p in todo}
        done_count = 0
        for fut in as_completed(fut_map):
            if fut.cancelled():
                continue
            p = fut_map[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {"ok": False, "tag": p.tag, "sample_id": p.sample_id,
                     "error": f"worker crash: {exc}", "elapsed_sec": 0.0}
            sid = r.get("sample_id", p.sample_id)
            r["sample_id"] = sid

            sample = sample_by_id.get(sid)
            if sample:
                r["uuid"] = sample.get("uuid")
                r["batch"] = sample.get("batch")
                r["sample_num"] = sample.get("sample_num")

            if r.get("ok"):
                uuid_val = r.get("uuid")
                if uuid_val:
                    results[uuid_val] = r

            done_count += 1
            since_ckpt += 1

            print(_one_line_report(r), flush=True)

            if since_ckpt >= args.ckpt_every:
                _flush(args.out, meta, results)
                since_ckpt = 0

            if not stop_signaled and _stop_requested(args.out):
                print("[STOP] Stop flag — terminating all workers immediately.", flush=True)
                stop_signaled = True
                _flush(args.out, meta, results)
                for proc in pool._processes.values():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                break
    finally:
        pool.shutdown(wait=False)

    _flush(args.out, meta, results)

    wall = time.time() - wall_start
    ok = len(results)
    failed = len(todo) - len(results)
    print("=" * 65, flush=True)
    print(f"Done in {wall:.1f}s  |  OK: {ok}  Failed: {failed}", flush=True)
    print(f"Output: {args.out}", flush=True)
    print("=" * 65, flush=True)
 
    # Remove any leftover stop flag so the next run starts clean.
    sf = _stop_flag_path(args.out)
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
