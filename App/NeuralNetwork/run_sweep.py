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
        tag=d.get("tag", f"S{int(d.get('id', -1)):06d}"),
        sample_id=int(d.get("id", -1)),
    )
 
 
def _load_samples(path: str, timeout_sec: float):
    with open(path) as f:
        payload = json.load(f)
    samples = payload["samples"]
    params  = [sample_to_simparams(d, timeout_sec) for d in samples]
    return params, payload.get("meta", {})


def _load_existing(path: str) -> dict:
    """Return {sample_id: result_dict} for already-written solver rows."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for r in data.get("results", []):
        sid = r.get("sample_id")
        if isinstance(sid, int) and sid >= 0:
            out[sid] = r
    return out


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
    sid = r.get("sample_id", -1)
    tag = r.get("tag", f"S{sid:06d}" if sid >= 0 else "S?")
    t   = float(r.get("elapsed_sec", 0.0))
    if r.get("ok"):
        return f"{tag} OK    {t:6.1f}s"
    err = r.get("error", "")[:80]
    return f"{tag} FAIL  {t:6.1f}s  {err}"

def main() -> int:
    parser = argparse.ArgumentParser(description="FastHenry sweep runner")
    parser.add_argument("--samples", default=os.path.join(_SIMDATA_DIR, "lhs_samples.json"))
    parser.add_argument("--out",     default=os.path.join(_SIMDATA_DIR, "solver_output.json"))
    parser.add_argument("--workers",          type=int, default=8)
    parser.add_argument("--timeout",          type=int, default=360)
    parser.add_argument("--checkpoint-every", type=int, default=25, dest="ckpt_every")
    parser.add_argument("--from-idx",         type=int, default=0,   dest="from_idx")
    parser.add_argument("--to-idx",           type=int, default=None, dest="to_idx")
    args = parser.parse_args()

    params_list, sample_meta = _load_samples(args.samples, args.timeout)

    from_idx = max(0, args.from_idx)
    to_idx   = args.to_idx if args.to_idx is not None else len(params_list)
    to_idx   = min(to_idx, len(params_list))
    target   = params_list[from_idx:to_idx]

    existing = _load_existing(args.out)
    todo     = [p for p in target if p.sample_id not in existing]

    print("=" * 65, flush=True)
    print(f"Sweep range [{from_idx}:{to_idx}]  ({len(target)} samples)", flush=True)
    print(f"Workers     : {args.workers}", flush=True)
    print(f"Timeout     : {args.timeout}s", flush=True)
    print(f"Already done: {len(target) - len(todo)}", flush=True)
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

    wall_start = time.time()
    since_ckpt = 0
 
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fut_map = {pool.submit(run_single_sim, p): p for p in todo}
        done_count = 0
        for fut in as_completed(fut_map):
            p = fut_map[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {"ok": False, "tag": p.tag, "sample_id": p.sample_id,
                     "error": f"worker crash: {exc}", "elapsed_sec": 0.0}
            sid = r.get("sample_id", p.sample_id)
            r["sample_id"] = sid
            results[sid] = r
            done_count += 1
            since_ckpt += 1
 
            print(_one_line_report(r), flush=True)
 
            if since_ckpt >= args.ckpt_every:
                _flush(args.out, meta, results)
                since_ckpt = 0
 
            if _stop_requested(args.out):
                print("[STOP] Stop flag detected — draining in-flight jobs.",
                      flush=True)
                # Cancel what we can and break; as_completed will still yield
                # already-running futures.
                for f in fut_map:
                    if not f.running() and not f.done():
                        f.cancel()
                # Don't break — let remaining complete naturally.
 
    _flush(args.out, meta, results)
 
    wall = time.time() - wall_start
    ok    = sum(1 for r in results.values() if r.get("ok")
                and from_idx <= r.get("sample_id", -1) < to_idx)
    fail  = sum(1 for r in results.values() if not r.get("ok")
                and from_idx <= r.get("sample_id", -1) < to_idx)
    print("=" * 65, flush=True)
    print(f"Done in {wall:.1f}s  |  OK: {ok}  Failed: {fail}", flush=True)
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
