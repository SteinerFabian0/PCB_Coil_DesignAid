#!/usr/bin/env python3
"""
Generate Latin Hypercube Samples for the combined TX+RX design-space sweep.

Domain configuration lives in a ``domain_master.json`` file that pins every
free / fixed variable for the run.  The same file is read by the GUI and by
the NN at training time so that the valid input domain is explicit and
reproducible.
 
The generator is deterministic: hard-coded seed + domain config + sample
count ⇒ identical sample list.  A small companion master file is written
with the full domain, so later code (bounds validation, NN training)
does not have to re-read ``lhs_samples.json``.
 
CLI usage:
    python generate_lhs_samples.py --config domain_master.json
                                    --out    lhs_samples.json
                                    --n      20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid as uuid_module

from scipy.stats import qmc

_HERE        = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT    = os.path.dirname(_HERE)
_SIMDATA_DIR = os.path.join(_APP_ROOT, "SimulationData")
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")

for _p in (_HERE, _MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parametric_coil as pc  # noqa: E402
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed seed — guarantees deterministic reproducibility.
HARDCODED_SEED = 42

DEFAULT_DOMAIN = {
    "tx": {
        "layers_selected":  [True, True, True, False],    # 3 active layers
        "od_max_mm":        53.0,
        "id_min_mm":        35.0,
        "trace_width_mm":   [0.2, 1.2],
        "trace_spacing_mm": [0.16, 0.16],
        "turns":            [6, 18],
        "outer_cu_oz":      [1.0, 1.0],    # min, max (step 0.5)
        "inner_cu_oz":      [0.5, 1.0],
        "outer_gap_mm":     [0.2, 0.2],
        "inner_gap_mm":     [1.3, 1.3],
        "nhinc":            1,
        "nwinc":            3,
        "port_outside_allowed": True,
        "port_inside_allowed":  False,
    },
    "rx": {
        "layers_selected":  [True, True, True, True],
        "od_max_mm":        53.0,
        "id_min_mm":        35.0,
        "trace_width_mm":   [0.2, 1.2],
        "trace_spacing_mm": [0.16, 0.16],
        "turns":            [4, 25],
        "outer_cu_oz":      [1.0, 1.0],
        "inner_cu_oz":      [0.5, 0.5],
        "outer_gap_mm":     [0.2, 0.2],
        "inner_gap_mm":     [0.6, 0.6],
        "nhinc":            1,
        "nwinc":            3,
        "port_outside_allowed": False,
        "port_inside_allowed":  True,
    },
    "global": {
        "pcb_gap_mm":    [2.6, 2.6],
        "resolution_mm": 1.2,
        "freq_hz":       [100_000.0, 150_000.0],
    },
    "n_total": 20_000,
}


# ---------------------------------------------------------------------------
# UUID generation for deduplication
# ---------------------------------------------------------------------------

def sample_uuid(sample: dict) -> str:
    """Deterministic UUID based on all simulation-relevant parameters."""
    SIM_KEYS = [
        "tx_turns", "tx_trace_width_mm", "tx_od_mm", "tx_spacing_mm",
        "tx_outer_gap_mm", "tx_inner_gap_mm", "tx_topology", "tx_port_inside",
        "tx_layers", "rx_turns", "rx_trace_width_mm", "rx_od_mm",
        "rx_spacing_mm", "rx_outer_gap_mm", "rx_inner_gap_mm", "rx_topology",
        "rx_port_inside", "rx_layers", "pcb_gap_mm", "resolution_mm",
        "freq_hz", "nhinc", "nwinc", "rx_nhinc", "rx_nwinc",
        "ground_circle_dia_mm",
    ]
    canonical = {k: sample[k] for k in SIM_KEYS if k in sample}
    payload = json.dumps(canonical, sort_keys=True, default=str).encode()
    h = hashlib.sha256(payload).digest()[:16]
    return str(uuid_module.UUID(bytes=h))


# ---------------------------------------------------------------------------
# Domain Helpers
# ---------------------------------------------------------------------------

def allowed_topologies(layers_selected) -> list:
    """Which topologies can be used given the active-layer mask.
 
    * 4 active layers → all three topologies (parallel / series / parallel_pairs_ser)
    * <4 active layers → only all-parallel and all-series (pairs_ser needs 4)
    """
    n_active = sum(1 for a in layers_selected if a)
    if n_active >= 4:
        return ["parallel", "series", "parallel_pairs_ser"]
    if n_active >= 2:
        return ["parallel", "series"]
    return ["parallel"]
 
 
def port_choices(side_cfg: dict) -> list:
    """Return list of allowed port_inside bools (at least one)."""
    out = []
    if side_cfg.get("port_outside_allowed", True):
        out.append(False)
    if side_cfg.get("port_inside_allowed", False):
        out.append(True)
    if not out:
        out = [False]
    return out
 
 
def quantize_oz(u: float, lo: float, hi: float, step: float = 0.5) -> float:
    """Map u ∈ [0,1) to a 0.5-oz-quantized value in [lo, hi]."""
    lo_q = round(lo / step) * step
    hi_q = round(hi / step) * step
    if hi_q <= lo_q:
        return lo_q
    n_steps = int(round((hi_q - lo_q) / step)) + 1
    idx = min(int(u * n_steps), n_steps - 1)
    return round(lo_q + idx * step, 4)

def _scale(u, lo, hi):
    return lo + u * (hi - lo)


def _round_int(v, lo, hi):
    return int(round(max(lo, min(hi, v))))


def _stackup_val(u: float, lo: float, hi: float) -> float:
    """Quantize u ∈ [0,1) to one of {lo, mid, hi} (or just {lo} when lo==hi)."""
    if hi <= lo:
        return round(lo, 4)
    options = [lo, (lo + hi) / 2.0, hi]
    idx = min(int(u * len(options)), len(options) - 1)
    return round(options[idx], 4)


# ---------------------------------------------------------------------------
# Config hashing / canonicalisation
# ---------------------------------------------------------------------------
 
def canonicalize(cfg: dict) -> dict:
    """Fill in missing keys, sort nested dicts — produces a stable JSON form."""
    out = {"tx": {}, "rx": {}, "global": {}, "n_total": int(cfg.get("n_total", 0))}
    for side in ("tx", "rx"):
        src = cfg.get(side) or {}
        dst = dict(DEFAULT_DOMAIN[side])
        dst.update(src)
        out[side] = dst
    src_g = cfg.get("global") or {}
    out["global"] = {**DEFAULT_DOMAIN["global"], **src_g}
    # Normalise pcb_gap_mm: old configs may store it as a scalar float.
    g = out["global"]
    if not isinstance(g.get("pcb_gap_mm"), list):
        v = float(g.get("pcb_gap_mm", 2.6))
        g["pcb_gap_mm"] = [v, v]
    return out
 
 
def config_hash(cfg: dict) -> str:
    """Stable short hash of the canonical domain (for sample provenance)."""
    payload = json.dumps(canonicalize(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:12]
 
 
# ---------------------------------------------------------------------------
# Feasibility check (matches FastHenry's own validators)
# ---------------------------------------------------------------------------
 
def _feasible(side_cfg: dict, od, w, s, t, og, ig, layers) -> tuple:
    sp = pc.SpiralParams(od_mm=od, trace_width_mm=w, spacing_mm=s,
                         turns=t, resolution_mm=1.0)
    ok, msg = pc.validate_spiral(sp)
    if not ok:
        return False, msg
    id_mm = 2.0 * (sp.r_inner_centerline - w / 2.0)
    if id_mm < side_cfg["id_min_mm"]:
        return False, f"ID {id_mm:.2f} mm < {side_cfg['id_min_mm']} mm minimum"
    stackup = pc.StackUp(
        slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
               for l in layers],
        outer_gap_mm=og, inner_gap_mm=ig,
    )
    ok, msg = pc.validate_stackup(stackup)
    return ok, msg

def _build_layers(selected: list, inner_oz: float, outer_oz: float) -> list:
    """Return 4-slot layer list with the requested oz weights."""
    oz = [outer_oz, inner_oz, inner_oz, outer_oz]
    return [{"active": bool(selected[i]), "copper_oz": float(oz[i])}
            for i in range(4)]
 

# ---------------------------------------------------------------------------
# LHS Sampling
# ---------------------------------------------------------------------------

# Dimension layout (LHS always fills every dim uniformly):
#   0  tx_turns            (int)
#   1  tx_width_mm
#   2  tx_od_mm
#   3  tx_spacing_mm       (stackup: {lo, mid, hi})
#   4  tx_outer_gap_mm     (stackup)
#   5  tx_inner_gap_mm     (stackup)
#   6  tx_outer_oz          (quantised 0.5)
#   7  tx_inner_oz          (quantised 0.5)
#   8  rx_turns             (int)
#   9  rx_width_mm
#   10 rx_od_mm
#   11 rx_spacing_mm        (stackup)
#   12 rx_outer_gap_mm      (stackup)
#   13 rx_inner_gap_mm      (stackup)
#   14 rx_outer_oz
#   15 rx_inner_oz
#   16 freq_hz              (stackup)
#   17 pcb_gap_mm           (stackup)
#   18 tx_topology          (categorical)
#   19 rx_topology          (categorical)
#   20 tx_port_inside       (categorical)
#   21 rx_port_inside       (categorical)

_LHS_DIMS = 22  # Note: ground_circle is expanded after LHS generation, not a 23rd dimension
 
 
def _decode(u_row, cfg):
    tx, rx, glob = cfg["tx"], cfg["rx"], cfg["global"]
    tx_topos = allowed_topologies(tx["layers_selected"])
    rx_topos = allowed_topologies(rx["layers_selected"])
    tx_ports = port_choices(tx)
    rx_ports = port_choices(rx)
 
    def _cat(u, options):
        idx = min(int(u * len(options)), len(options) - 1)
        return options[idx]
 
    tx_turns = _round_int(_scale(u_row[0], *tx["turns"]), *tx["turns"])
    tx_w     = round(_scale(u_row[1], *tx["trace_width_mm"]), 4)
    tx_od    = round(_scale(u_row[2], tx["id_min_mm"] + 2.0,
                                    tx["od_max_mm"]), 4)
    tx_s     = _stackup_val(u_row[3], *tx["trace_spacing_mm"])
    tx_og    = _stackup_val(u_row[4], *tx["outer_gap_mm"])
    tx_ig    = _stackup_val(u_row[5], *tx["inner_gap_mm"])
    tx_oz_o  = quantize_oz(u_row[6], *tx["outer_cu_oz"])
    tx_oz_i  = quantize_oz(u_row[7], *tx["inner_cu_oz"])
 
    rx_turns = _round_int(_scale(u_row[8], *rx["turns"]), *rx["turns"])
    rx_w     = round(_scale(u_row[9],  *rx["trace_width_mm"]), 4)
    rx_od    = round(_scale(u_row[10], rx["id_min_mm"] + 2.0,
                                     rx["od_max_mm"]), 4)
    rx_s     = _stackup_val(u_row[11], *rx["trace_spacing_mm"])
    rx_og    = _stackup_val(u_row[12], *rx["outer_gap_mm"])
    rx_ig    = _stackup_val(u_row[13], *rx["inner_gap_mm"])
    rx_oz_o  = quantize_oz(u_row[14], *rx["outer_cu_oz"])
    rx_oz_i  = quantize_oz(u_row[15], *rx["inner_cu_oz"])
 
    freq_hz     = _stackup_val(u_row[16], *glob["freq_hz"])
    pcb_gap_mm  = _stackup_val(u_row[17], *glob["pcb_gap_mm"])
    tx_topo     = _cat(u_row[18], tx_topos)
    rx_topo     = _cat(u_row[19], rx_topos)
    tx_p_in     = _cat(u_row[20], tx_ports)
    rx_p_in     = _cat(u_row[21], rx_ports)
 
    return {
        "tx_turns": tx_turns, "tx_width": tx_w, "tx_od_mm": tx_od,
        "tx_spacing_mm": tx_s, "tx_outer_gap_mm": tx_og,
        "tx_inner_gap_mm": tx_ig, "tx_outer_oz": tx_oz_o,
        "tx_inner_oz": tx_oz_i, "tx_topology": tx_topo,
        "tx_port_inside": tx_p_in,
        "rx_turns": rx_turns, "rx_width": rx_w, "rx_od_mm": rx_od,
        "rx_spacing_mm": rx_s, "rx_outer_gap_mm": rx_og,
        "rx_inner_gap_mm": rx_ig, "rx_outer_oz": rx_oz_o,
        "rx_inner_oz": rx_oz_i, "rx_topology": rx_topo,
        "rx_port_inside": rx_p_in,
        "freq_hz": freq_hz,
        "pcb_gap_mm": pcb_gap_mm,
    }
 
 
def generate_samples(cfg: dict, n_total: int, seed: int = HARDCODED_SEED,
                     oversample: int = 8, log=print,
                     ground_circle_enabled: bool = False,
                     ground_circle_min_mm: float = 10.0,
                     ground_circle_max_mm: float = 20.0) -> list:
    """Return up to *n_total* feasible sample dicts (deterministic).

    If ground_circle_enabled=True, each base sample is expanded into
    (max_mm - min_mm + 1) copies, one per 1mm diameter step.
    Total output = n_total * (max_mm - min_mm + 1) when enabled.
    """
    cfg = canonicalize(cfg)
    n_candidates = max(n_total * oversample, n_total + 100)
    sampler = qmc.LatinHypercube(d=_LHS_DIMS, seed=seed)
    raw = sampler.random(n_candidates)
 
    valid, rejected = [], 0
    for i, row in enumerate(raw):
        if len(valid) >= n_total:
            break
        c = _decode(row, cfg)

        tx_layers = _build_layers(cfg["tx"]["layers_selected"],
                                  c["tx_inner_oz"], c["tx_outer_oz"])
        
        rx_layers = _build_layers(cfg["rx"]["layers_selected"],
                                  c["rx_inner_oz"], c["rx_outer_oz"])
        
        ok_tx, msg_tx = _feasible(cfg["tx"],
                                  c["tx_od_mm"], c["tx_width"],
                                  c["tx_spacing_mm"], c["tx_turns"],
                                  c["tx_outer_gap_mm"], c["tx_inner_gap_mm"],
                                  tx_layers)
        if not ok_tx:
            rejected += 1
            continue

        ok_rx, msg_rx = _feasible(cfg["rx"],
                                  c["rx_od_mm"], c["rx_width"],
                                  c["rx_spacing_mm"], c["rx_turns"],
                                  c["rx_outer_gap_mm"], c["rx_inner_gap_mm"],
                                  rx_layers)
        if not ok_rx:
            rejected += 1
            continue

        sample_id = len(valid)

        tx_nwinc = 2 if c["tx_width"] < 0.4 else 3
        rx_nwinc = 2 if c["rx_width"] < 0.4 else 3

        sample = {
            "id":            sample_id,
            "tag":           f"S{sample_id:06d}",
            # TX
            "tx_turns":         c["tx_turns"],
            "tx_trace_width_mm": c["tx_width"],
            "tx_od_mm":         c["tx_od_mm"],
            "tx_spacing_mm":    c["tx_spacing_mm"],
            "tx_outer_gap_mm":  c["tx_outer_gap_mm"],
            "tx_inner_gap_mm":  c["tx_inner_gap_mm"],
            "tx_topology":      c["tx_topology"],
            "tx_port_inside":   c["tx_port_inside"],
            "tx_layers":        tx_layers,
            # RX
            "rx_turns":         c["rx_turns"],
            "rx_trace_width_mm": c["rx_width"],
            "rx_od_mm":         c["rx_od_mm"],
            "rx_spacing_mm":    c["rx_spacing_mm"],
            "rx_outer_gap_mm":  c["rx_outer_gap_mm"],
            "rx_inner_gap_mm":  c["rx_inner_gap_mm"],
            "rx_topology":      c["rx_topology"],
            "rx_port_inside":   c["rx_port_inside"],
            "rx_layers":        rx_layers,
            # Global
            "pcb_gap_mm":       c["pcb_gap_mm"],
            "resolution_mm":    cfg["global"]["resolution_mm"],
            "freq_hz":          c["freq_hz"],
            "fmin_hz":          c["freq_hz"],
            "fmax_hz":          c["freq_hz"],
            "freq_ndec":        0,
            "nhinc":            cfg["tx"]["nhinc"],
            "nwinc":            cfg["tx"]["nwinc"],
            "rx_nhinc":         cfg["rx"]["nhinc"],
            "tx_nwinc":         tx_nwinc,
            "rx_nwinc":         rx_nwinc,
            # Ground plane (always included; may be 0)
            "ground_circle_dia_mm": 0.0,
        }
        sample["uuid"] = sample_uuid(sample)
        valid.append(sample)
 
    log(f"LHS: {len(valid)} valid / {rejected} rejected "
        f"out of {n_candidates} candidates")

    # Expand samples by ground circle diameter if enabled
    if ground_circle_enabled:
        diameters = list(range(int(ground_circle_min_mm), int(ground_circle_max_mm) + 1))
        if not diameters:
            log(f"WARNING: ground_circle_min ({ground_circle_min_mm}) > ground_circle_max "
                f"({ground_circle_max_mm}) — ground circle expansion skipped")
        else:
            n_base = len(valid)
            expanded = []
            for sample in valid:
                for dia in diameters:
                    new_sample = sample.copy()
                    new_sample["ground_circle_dia_mm"] = float(dia)
                    new_sample["uuid"] = sample_uuid(new_sample)
                    expanded.append(new_sample)
            valid = expanded
            log(f"Ground circles enabled: expanded {n_base} base samples "
                f"× {len(diameters)} diameters = {len(valid)} total samples")

    return valid


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_domain_master(cfg: dict, out_path: str,
                        n_total: int, n_valid: int) -> str:
    """Write the canonical domain master file; return the config hash."""
    canon = canonicalize(cfg)
    canon["n_total"] = int(n_total)
    canon["n_valid"] = int(n_valid)
    canon["seed"]    = HARDCODED_SEED
    h = config_hash(canon)
    canon["config_hash"] = h
    canon["tx"]["allowed_topologies"] = allowed_topologies(canon["tx"]["layers_selected"])
    canon["rx"]["allowed_topologies"] = allowed_topologies(canon["rx"]["layers_selected"])
    with open(out_path, "w") as f:
        json.dump(canon, f, indent=2, default=str)
    return h
 
 
def write_samples_file(samples: list, cfg: dict, out_path: str,
                       config_hash_: str, batch_num: int = None) -> None:
    for i, sample in enumerate(samples, start=1):
        sample["sample_num"] = i
        if batch_num is not None:
            sample["batch"] = batch_num

    payload = {
        "meta": {
            "batch":         batch_num,
            "n_samples":     len(samples),
            "seed":          HARDCODED_SEED,
            "config_hash":   config_hash_,
            "domain":        canonicalize(cfg),
        },
        "samples": samples,
    }

    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Global results metadata
# ---------------------------------------------------------------------------

def _load_global_results(path: str) -> dict:
    """Load global_results.json, return empty structure if missing."""
    if not os.path.exists(path):
        return {"meta": {"n_lhs_batches": 0, "lhs_batches": {}}, "results": []}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"meta": {"n_lhs_batches": 0, "lhs_batches": {}}, "results": []}


def _next_batch_number(global_path: str) -> int:
    """Determine the next batch number by reading global_results metadata."""
    data = _load_global_results(global_path)
    n_batches = data.get("meta", {}).get("n_lhs_batches", 0)
    return n_batches + 1


def _update_global_batch_metadata(global_path: str, batch_num: int,
                                  batch_file: str, n_samples: int) -> None:
    """Update global_results.json with new batch metadata."""
    data = _load_global_results(global_path)
    meta = data.setdefault("meta", {})
    lhs_batches = meta.setdefault("lhs_batches", {})

    lhs_batches[str(batch_num)] = {
        "file": batch_file,
        "n_samples": n_samples,
    }
    meta["n_lhs_batches"] = max(meta.get("n_lhs_batches", 0), batch_num)

    tmp = global_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, global_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic LHS sampler")
    parser.add_argument("--config", required=True,
                        help="Path to the domain config JSON "
                             "(written by the GUI on Generate)")
    parser.add_argument("--master", default=os.path.join(_SIMDATA_DIR, "domain_master.json"))
    parser.add_argument("--global", dest="global_", default=os.path.join(_SIMDATA_DIR, "global_results.json"),
                        help="Path to global_results.json for batch tracking")
    parser.add_argument("--n", type=int, default=None,
                        help="Override total sample count from config")
    parser.add_argument("--ground-circle-enabled", type=int, default=0,
                        help="1 = expand samples by ground circle diameter, 0 = no ground circle")
    parser.add_argument("--ground-circle-min", type=float, default=10.0,
                        help="Minimum ground circle diameter (mm)")
    parser.add_argument("--ground-circle-max", type=float, default=20.0,
                        help="Maximum ground circle diameter (mm)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    n_total = args.n if args.n is not None else cfg.get("n_total", 0)
    if not n_total or n_total <= 0:
        print("ERROR: n_total must be > 0", file=sys.stderr)
        return 2

    batch_num = _next_batch_number(args.global_)
    samples = generate_samples(
        cfg, n_total,
        ground_circle_enabled=bool(args.ground_circle_enabled),
        ground_circle_min_mm=args.ground_circle_min,
        ground_circle_max_mm=args.ground_circle_max,
    )
    if not samples:
        print("ERROR: no feasible samples — check the domain config", file=sys.stderr)
        return 3

    h = write_domain_master(cfg, args.master, n_total, len(samples))

    out_path = os.path.join(_SIMDATA_DIR, f"lhs_batch_{batch_num}.json")
    write_samples_file(samples, cfg, out_path, h, batch_num=batch_num)

    _update_global_batch_metadata(args.global_, batch_num,
                                   f"lhs_batch_{batch_num}.json", len(samples))

    print(f"Wrote {len(samples)} samples -> {out_path}")
    print(f"Domain master       -> {args.master}  (hash {h})")
    print(f"Batch {batch_num} registered in global_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
