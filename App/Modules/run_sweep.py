#!/usr/bin/env python3
"""
Run the full LHS design-space sweep through FastHenry2.

Loads SimParams from lhs_samples.json (produced by generate_lhs_samples.py),
feeds them to run_batch(), appends FastHenry results, and writes
sweep_results.json.

Usage:
    python run_sweep.py [--samples lhs_samples.json] [--out sweep_results.json]
                        [--workers 16] [--timeout 240]

Checkpointing:
    Results are flushed to disk every --checkpoint-every N completions so a
    crash doesn't lose everything. Re-running with the same --out file will
    skip already-completed tags and resume from where it left off.

Graceful stop:
    Create a file called STOP_SWEEP (in the same directory as --out) while the
    sweep is running.  The current batch will finish, results will be
    checkpointed, then the process exits cleanly.  Re-running will resume.
"""

import argparse
import json
import os
import sys
import time

_HERE        = os.path.dirname(os.path.abspath(__file__))  # App/Modules
_APP_ROOT    = os.path.dirname(_HERE)                       # App
_SIMDATA_DIR = os.path.join(_APP_ROOT, "SimulationData")

from parallel_sim import SimParams, run_batch
from generate_lhs_samples import dict_to_simparams


def _load_samples(path: str) -> list:
    with open(path) as f:
        payload = json.load(f)
    return [dict_to_simparams(d) for d in payload["samples"]], payload.get("meta", {})


def _load_existing_results(path: str) -> dict:
    """Return {tag: result_dict} for already-completed rows."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["tag"]: r for r in data.get("results", []) if r.get("tag")}


def _flush(out_path, meta, results):
    payload = {"meta": meta, "results": results}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)


def _stop_flag_path(out_path: str) -> str:
    return os.path.join(os.path.dirname(out_path), "STOP_SWEEP")


def _stop_requested(out_path: str) -> bool:
    return os.path.exists(_stop_flag_path(out_path))


def main():
    os.makedirs(_SIMDATA_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run LHS FastHenry sweep")
    parser.add_argument("--samples",
                        default=os.path.join(_SIMDATA_DIR, "lhs_samples.json"))
    parser.add_argument("--out",
                        default=os.path.join(_SIMDATA_DIR, "sweep_results.json"))
    parser.add_argument("--workers",          type=int, default=16)
    parser.add_argument("--timeout",          type=int, default=240,
                        help="Per-sim timeout (seconds)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        dest="ckpt_every")
    parser.add_argument("--from-idx",         type=int, default=0,
                        dest="from_idx",
                        help="0-based index of first sample to process (inclusive)")
    parser.add_argument("--to-idx",           type=int, default=None,
                        dest="to_idx",
                        help="0-based index of last sample to process (exclusive). "
                             "Defaults to end of list.")
    args = parser.parse_args()

    params_list, meta = _load_samples(args.samples)

    # Apply index range slice before anything else.
    from_idx = max(0, args.from_idx)
    to_idx   = args.to_idx if args.to_idx is not None else len(params_list)
    to_idx   = min(to_idx, len(params_list))
    params_list = params_list[from_idx:to_idx]
    range_str = f"[{from_idx}:{to_idx}]" if (from_idx or args.to_idx) else "[all]"

    existing = _load_existing_results(args.out)

    # Patch per-sim timeout from CLI.
    for p in params_list:
        p.timeout_sec = float(args.timeout)

    # Skip already-done tags.
    todo   = [p for p in params_list if p.tag not in existing]
    done   = list(existing.values())

    print("=" * 65)
    print(f"FastHenry sweep  ({args.workers} workers, {args.timeout}s timeout)")
    print(f"  Range         : {range_str}  ({len(params_list)} samples)")
    print(f"  Already done  : {len(done)}")
    print(f"  To simulate   : {len(todo)}")
    print(f"  Output        : {args.out}")
    print("=" * 65)

    if not todo:
        print("Nothing to do -- all samples already in results file.")
        return 0

    # Remove any stale stop flag from a previous run.
    stop_flag = _stop_flag_path(args.out)
    if os.path.exists(stop_flag):
        os.remove(stop_flag)

    wall_start = time.time()
    results_so_far = list(done)
    chunk_size = args.ckpt_every

    for chunk_start in range(0, len(todo), chunk_size):
        if _stop_requested(args.out):
            print("\n[STOP] Stop flag detected -- finishing current chunk and exiting.")
            break

        chunk = todo[chunk_start: chunk_start + chunk_size]
        n_done_before = len(results_so_far) - len(done)

        def _progress(n_done_in_chunk, _total_in_chunk):
            elapsed = time.time() - wall_start
            n_total_done = len(done) + n_done_before + n_done_in_chunk
            rate = (n_total_done - len(done)) / elapsed if elapsed > 0 else 0
            remaining = len(todo) - (n_done_before + n_done_in_chunk)
            eta = remaining / rate if rate > 0 else float("inf")
            print(f"  [{n_total_done}/{len(params_list)}]  "
                  f"{elapsed:.0f}s elapsed  "
                  f"ETA {eta:.0f}s",
                  flush=True)

        batch_results = run_batch(chunk, max_workers=args.workers,
                                  progress_cb=_progress)
        results_so_far.extend(batch_results)

        run_meta = {
            **meta,
            "workers":     args.workers,
            "timeout_sec": args.timeout,
            "completed":   len(results_so_far),
            "total":       len(params_list),
        }
        _flush(args.out, run_meta, results_so_far)
        print(f"  [checkpoint] {len(results_so_far)} results saved -> {args.out}")

    wall_total = time.time() - wall_start
    ok_count   = sum(1 for r in results_so_far if r.get("ok"))
    fail_count = len(results_so_far) - ok_count

    stopped_early = _stop_requested(args.out)
    if stopped_early:
        # Clean up the flag so the next run starts fresh.
        try:
            os.remove(stop_flag)
        except OSError:
            pass

    print()
    print("=" * 65)
    if stopped_early:
        print(f"Sweep paused after {wall_total:.1f}s  (resume by re-running)")
    else:
        print(f"Sweep complete in {wall_total:.1f}s")
    print(f"  OK      : {ok_count}")
    print(f"  Failed  : {fail_count}")
    print(f"  Output  : {args.out}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
