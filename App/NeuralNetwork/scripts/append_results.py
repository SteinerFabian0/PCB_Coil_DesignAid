#!/usr/bin/env python3
"""
Append valid solver outputs into the global training file.
 
Reads ``solver_output.json`` (written by run_sweep.py) and merges the OK
rows into ``global_results.json``.  Samples that already exist in the
global file (matched by ``sample_id``) are left untouched — we never
overwrite.
 
Each appended row is enriched with the analytical NN input features
(wire length, Wheeler L, fill factor, mean radius, inner diameter,
ln(N^2)) computed for both TX and RX.  These features are *not* written
back to the solver file — they live only in the global file that the
NN trainer consumes.
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import sys
 
_HERE        = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT    = os.path.dirname(os.path.dirname(_HERE))
_SIMDATA_DIR = os.path.join(_APP_ROOT, "SimulationData")
 
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
 
from coil_features import features_for_coil, FEATURE_KEYS  # noqa: E402
 
 
GLOBAL_FILE = os.path.join(_SIMDATA_DIR, "global_results.json")
SOLVER_FILE = os.path.join(_SIMDATA_DIR, "solver_output.json")
 
 
# ---------------------------------------------------------------------------
# Feature enrichment
# ---------------------------------------------------------------------------
 
def enrich_row(r: dict) -> dict:
    """Return a copy of *r* with TX / RX derived features attached."""
    out = dict(r)
 
    tx_feats = features_for_coil(
        od_mm=float(r["tx_od_mm"]),
        turns=float(r["tx_turns"]),
        trace_width_mm=float(r.get("tx_width", r.get("tx_trace_width_mm", 0.0))),
        spacing_mm=float(r.get("tx_spacing_mm", 0.16)),
    )
    rx_feats = features_for_coil(
        od_mm=float(r["rx_od_mm"]),
        turns=float(r["rx_turns"]),
        trace_width_mm=float(r.get("rx_width", r.get("rx_trace_width_mm", 0.0))),
        spacing_mm=float(r.get("rx_spacing_mm", 0.16)),
    )
    for k in FEATURE_KEYS:
        out[f"tx_{k}"] = tx_feats[k]
        out[f"rx_{k}"] = rx_feats[k]
    return out
 
 
# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
 
def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {"meta": {}, "results": []}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"meta": {}, "results": []}
 
 
def _save(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
 
 
# ---------------------------------------------------------------------------
# Public API (called by the GUI)
# ---------------------------------------------------------------------------
 
def append_solver_to_global(solver_path: str = SOLVER_FILE,
                            global_path: str = GLOBAL_FILE,
                            domain_master_path: str | None = None,
                            log=print) -> dict:
    """
    Merge OK rows from *solver_path* (local results file) into *global_path*.
    Uses UUID-based deduplication.

    Returns a summary dict::
        {"added": N, "skipped_existing": M, "total": T}
    """
    sol = _load(solver_path)
    glb = _load(global_path)

    existing_uuids = {r.get("uuid") for r in glb.get("results", []) if r.get("uuid")}

    added = 0
    new_rows = list(glb.get("results", []))

    for r in sol.get("results", []):
        uuid_val = r.get("uuid")
        if not uuid_val:
            continue
        if uuid_val in existing_uuids:
            continue

        enriched = enrich_row(r)
        new_rows.append(enriched)
        existing_uuids.add(uuid_val)
        added += 1

    new_rows.sort(key=lambda x: (x.get("batch", 0), x.get("sample_num", 0)))

    meta = dict(glb.get("meta", {}))
    if domain_master_path and os.path.exists(domain_master_path):
        try:
            with open(domain_master_path) as f:
                dm = json.load(f)
            meta["config_hash"] = dm.get("config_hash")
            meta["n_total"]     = dm.get("n_total")
        except Exception:
            pass
    meta["n_results"] = len(new_rows)

    _save(global_path, {"meta": meta, "results": new_rows})

    summary = {
        "added":            added,
        "skipped_existing": len(new_rows) - added - len(glb.get("results", [])),
        "total":            len(new_rows),
    }
    log(f"Appended {added} rows  "
        f"-> {global_path}  [total now {len(new_rows)}]")
    return summary
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append valid results into global_results.json")
    parser.add_argument("--local", default=None,
                        help="Local results file (results_batch_N.json)")
    parser.add_argument("--solver", default=None,
                        help="Deprecated: use --local")
    parser.add_argument("--global", dest="global_", default=GLOBAL_FILE)
    parser.add_argument("--domain", default=os.path.join(_SIMDATA_DIR, "domain_master.json"))
    args = parser.parse_args()

    local_path = args.local or args.solver or SOLVER_FILE
    summary = append_solver_to_global(local_path, args.global_, args.domain)
    print(json.dumps(summary, indent=2))
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())