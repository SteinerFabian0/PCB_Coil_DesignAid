#!/usr/bin/env python3
"""
Generate Latin Hypercube Samples for the combined TX+RX design-space sweep.

Each sample is a complete TX+RX coil pair. FastHenry will simulate both
coils in one .inp file (2-port, 2×2 Z-matrix) so that L_tx, L_rx, and M
are extracted from the same field solution.

Three branches — one per RX topology:
    parallel, series, parallel_pairs_ser
Each branch targets N_TARGET valid samples (default 1500) → 4500 total.

TX fixed:   spacing=0.16, topology=parallel, OD=52, L1+L2+L3 active (1oz),
            outer_gap=0.2, inner_gap=1.3, port outside
TX free:    turns ∈ [6,18] (int), trace_width ∈ [0.2,1.2]

RX fixed:   spacing=0.16, all 4 layers (outer 1oz, inner 0.5oz),
            outer_gap=0.2, inner_gap=0.6, port inside
RX free:    OD ∈ [48,52], turns ∈ [4,25] (int), trace_width ∈ [0.2,1.2]

Global:     PCB gap = 2.6 mm, freq fixed at 125 000 Hz

Geometry constraint:  ID of both TX and RX >= 35 mm

LHS dimensions per sample (5D):
    [tx_turns, tx_width, rx_od, rx_turns, rx_width]

Run:
    python generate_lhs_samples.py [--n 1500] [--out lhs_samples.json] [--seed 42]
"""

import argparse
import json
import sys
import os

from scipy.stats import qmc

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIMDATA_DIR = os.path.join(_HERE, "App", "SimulationData")

sys.path.insert(0, os.path.join(_HERE, "App", "Modules"))
import parametric_coil as pc
from parallel_sim import SimParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TARGET     = 1500
OVERSAMPLE   = 4       # generate this × N_TARGET candidates, then filter
DEFAULT_SEED = 42
RESOLUTION_MM = 1.5    # FastHenry segment length — coarser = faster, finer = more accurate

PCB_GAP_MM   = 2.6
FREQ_HZ      = 125_000.0
MIN_ID_MM    = 35.0

# TX fixed
TX_OD         = 52.0
TX_SPACING    = 0.16
TX_OUTER_GAP  = 0.2
TX_INNER_GAP  = 1.3
TX_TOPOLOGY   = "parallel"
TX_LAYERS = [
    {"active": True,  "copper_oz": 1.0},
    {"active": True,  "copper_oz": 1.0},
    {"active": True,  "copper_oz": 1.0},
    {"active": False, "copper_oz": 1.0},
]

# RX fixed
RX_SPACING   = 0.16
RX_OUTER_GAP = 0.2
RX_INNER_GAP = 0.6
RX_LAYERS = [
    {"active": True, "copper_oz": 1.0},
    {"active": True, "copper_oz": 0.5},
    {"active": True, "copper_oz": 0.5},
    {"active": True, "copper_oz": 1.0},
]
RX_TOPOLOGIES = ["parallel", "series", "parallel_pairs_ser"]

# Variable ranges
TX_TURNS_RANGE = (6,   18)
TX_WIDTH_RANGE = (0.2, 1.2)
RX_OD_RANGE    = (48.0, 52.0)
RX_TURNS_RANGE = (4,   25)
RX_WIDTH_RANGE = (0.2, 1.2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale(u, lo, hi):
    return lo + u * (hi - lo)


def _round_int(v, lo, hi):
    return int(round(max(lo, min(hi, v))))


def _feasible(od, w, s, t, og, ig, layers):
    sp = pc.SpiralParams(od_mm=od, trace_width_mm=w, spacing_mm=s,
                         turns=t, resolution_mm=1.0)
    ok, msg = pc.validate_spiral(sp)
    if not ok:
        return False, msg
    # Inner-diameter constraint: innermost trace inner edge >= MIN_ID_MM
    id_mm = 2.0 * (sp.r_inner_centerline - w / 2.0)
    if id_mm < MIN_ID_MM:
        return False, f"ID {id_mm:.2f} mm < {MIN_ID_MM} mm minimum"
    stackup = pc.StackUp(
        slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
               for l in layers],
        outer_gap_mm=og, inner_gap_mm=ig,
    )
    ok, msg = pc.validate_stackup(stackup)
    return ok, msg


# ---------------------------------------------------------------------------
# Candidate generation + filtering
# ---------------------------------------------------------------------------

def _generate_candidates(n: int, rx_topology: str, seed: int) -> list:
    """
    LHS over 5 dimensions:
        0: tx_turns  1: tx_width  2: rx_od  3: rx_turns  4: rx_width
    Frequency is fixed at FREQ_HZ (125 kHz) — not swept.
    Returns list of raw candidate dicts (not yet filtered).
    """
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    raw = sampler.random(n)

    candidates = []
    for row in raw:
        tx_turns = _round_int(_scale(row[0], *TX_TURNS_RANGE),
                               TX_TURNS_RANGE[0], TX_TURNS_RANGE[1])
        tx_width = round(_scale(row[1], *TX_WIDTH_RANGE), 4)
        rx_od    = round(_scale(row[2], *RX_OD_RANGE),    4)
        rx_turns = _round_int(_scale(row[3], *RX_TURNS_RANGE),
                               RX_TURNS_RANGE[0], RX_TURNS_RANGE[1])
        rx_width = round(_scale(row[4], *RX_WIDTH_RANGE), 4)
        candidates.append({
            "tx_turns": tx_turns, "tx_width": tx_width,
            "rx_od":    rx_od,    "rx_turns": rx_turns,
            "rx_width": rx_width, "freq":     FREQ_HZ,
            "rx_topology": rx_topology,
        })
    return candidates


def _filter(candidates: list, n_target: int) -> tuple:
    """Return (valid_list, n_rejected)."""
    valid    = []
    rejected = 0
    for c in candidates:
        if len(valid) >= n_target:
            break
        tx_ok, tx_msg = _feasible(TX_OD, c["tx_width"], TX_SPACING, c["tx_turns"],
                                   TX_OUTER_GAP, TX_INNER_GAP, TX_LAYERS)
        if not tx_ok:
            rejected += 1
            continue
        rx_ok, rx_msg = _feasible(c["rx_od"], c["rx_width"], RX_SPACING, c["rx_turns"],
                                   RX_OUTER_GAP, RX_INNER_GAP, RX_LAYERS)
        if not rx_ok:
            rejected += 1
            continue
        valid.append(c)
    return valid, rejected


# ---------------------------------------------------------------------------
# SimParams builder
# ---------------------------------------------------------------------------

def _make_simparams(c: dict, branch_idx: int, sample_idx: int) -> SimParams:
    topo     = c["rx_topology"]
    topo_tag = {"parallel": "PAR", "series": "SER",
                "parallel_pairs_ser": "PPS"}[topo]
    tag = (f"{topo_tag}_{sample_idx:05d}"
           f"_txt{c['tx_turns']}_txw{c['tx_width']:.3f}"
           f"_rxt{c['rx_turns']}_rxw{c['rx_width']:.3f}"
           f"_od{c['rx_od']:.1f}")
    return SimParams(
        tx_turns=c["tx_turns"],
        tx_trace_width_mm=c["tx_width"],
        tx_od_mm=TX_OD,
        tx_spacing_mm=TX_SPACING,
        tx_outer_gap_mm=TX_OUTER_GAP,
        tx_inner_gap_mm=TX_INNER_GAP,
        tx_topology=TX_TOPOLOGY,
        tx_layers=TX_LAYERS,
        rx_turns=c["rx_turns"],
        rx_trace_width_mm=c["rx_width"],
        rx_od_mm=c["rx_od"],
        rx_spacing_mm=RX_SPACING,
        rx_outer_gap_mm=RX_OUTER_GAP,
        rx_inner_gap_mm=RX_INNER_GAP,
        rx_topology=topo,
        rx_layers=RX_LAYERS,
        pcb_gap_mm=PCB_GAP_MM,
        freq_hz=FREQ_HZ,
        fmin_hz=FREQ_HZ,
        fmax_hz=FREQ_HZ,
        freq_ndec=0,
        resolution_mm=RESOLUTION_MM,
        timeout_sec=240.0,
        tag=tag,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def simparams_to_dict(p: SimParams) -> dict:
    return {
        "tag":             p.tag,
        "tx_turns":        p.tx_turns,
        "tx_trace_width_mm": p.tx_trace_width_mm,
        "tx_od_mm":        p.tx_od_mm,
        "tx_spacing_mm":   p.tx_spacing_mm,
        "tx_outer_gap_mm": p.tx_outer_gap_mm,
        "tx_inner_gap_mm": p.tx_inner_gap_mm,
        "tx_topology":     p.tx_topology,
        "tx_layers":       p.tx_layers,
        "rx_turns":        p.rx_turns,
        "rx_trace_width_mm": p.rx_trace_width_mm,
        "rx_od_mm":        p.rx_od_mm,
        "rx_spacing_mm":   p.rx_spacing_mm,
        "rx_outer_gap_mm": p.rx_outer_gap_mm,
        "rx_inner_gap_mm": p.rx_inner_gap_mm,
        "rx_topology":     p.rx_topology,
        "rx_layers":       p.rx_layers,
        "pcb_gap_mm":      p.pcb_gap_mm,
        "freq_hz":         p.freq_hz,
        "fmin_hz":         p.fmin_hz,
        "fmax_hz":         p.fmax_hz,
        "freq_ndec":       p.freq_ndec,
        "resolution_mm":   p.resolution_mm,
        "timeout_sec":     p.timeout_sec,
    }


def dict_to_simparams(d: dict) -> SimParams:
    return SimParams(
        tx_turns=d["tx_turns"],
        tx_trace_width_mm=d["tx_trace_width_mm"],
        tx_od_mm=d.get("tx_od_mm",        52.0),
        tx_spacing_mm=d.get("tx_spacing_mm",  0.16),
        tx_outer_gap_mm=d.get("tx_outer_gap_mm", 0.2),
        tx_inner_gap_mm=d.get("tx_inner_gap_mm", 1.3),
        tx_topology=d.get("tx_topology",   "parallel"),
        tx_layers=d.get("tx_layers", [
            {"active": True,  "copper_oz": 1.0},
            {"active": True,  "copper_oz": 1.0},
            {"active": True,  "copper_oz": 1.0},
            {"active": False, "copper_oz": 1.0},
        ]),
        rx_turns=d["rx_turns"],
        rx_trace_width_mm=d["rx_trace_width_mm"],
        rx_od_mm=d.get("rx_od_mm",        50.0),
        rx_spacing_mm=d.get("rx_spacing_mm",  0.16),
        rx_outer_gap_mm=d.get("rx_outer_gap_mm", 0.2),
        rx_inner_gap_mm=d.get("rx_inner_gap_mm", 0.6),
        rx_topology=d.get("rx_topology",   "parallel"),
        rx_layers=d.get("rx_layers", [
            {"active": True, "copper_oz": 1.0},
            {"active": True, "copper_oz": 0.5},
            {"active": True, "copper_oz": 0.5},
            {"active": True, "copper_oz": 1.0},
        ]),
        pcb_gap_mm=d.get("pcb_gap_mm",     2.6),
        freq_hz=d.get("freq_hz",        125_000.0),
        fmin_hz=d.get("fmin_hz",        125_000.0),
        fmax_hz=d.get("fmax_hz",        125_000.0),
        freq_ndec=d.get("freq_ndec",       1),
        resolution_mm=d.get("resolution_mm",  1.0),
        timeout_sec=d.get("timeout_sec",    240.0),
        tag=d.get("tag", ""),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate LHS TX+RX design-space samples for WPT sweep")
    os.makedirs(_SIMDATA_DIR, exist_ok=True)
    parser.add_argument("--n",    type=int, default=N_TARGET,
                        help="Valid samples per RX-topology branch (default 1500)")
    parser.add_argument("--out",
                        default=os.path.join(_SIMDATA_DIR, "lhs_samples.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    n_target   = args.n
    n_oversamp = n_target * OVERSAMPLE
    seed       = args.seed

    all_params = []
    summary    = []

    print("=" * 70)
    print(f"LHS TX+RX sampler -- 3 branches x {n_target} valid samples each")
    print(f"5D LHS (freq fixed at {FREQ_HZ/1e3:.0f} kHz, ID >= {MIN_ID_MM} mm)")
    print(f"Oversampling {OVERSAMPLE}x -> filtering infeasible geometries")
    print("=" * 70)

    for b_idx, topo in enumerate(RX_TOPOLOGIES):
        branch_seed = seed + b_idx * 1000
        print(f"\n[{topo}] Generating {n_oversamp} candidates ...", flush=True)
        candidates = _generate_candidates(n_oversamp, topo, seed=branch_seed)
        valid, rejected = _filter(candidates, n_target)
        print(f"  Valid: {len(valid)}  |  Rejected: {rejected}")

        if len(valid) < n_target:
            print(f"  WARNING: only {len(valid)}/{n_target} valid -- "
                  f"increase --n or OVERSAMPLE constant")

        for i, c in enumerate(valid):
            all_params.append(_make_simparams(c, b_idx, i))

        summary.append({
            "branch":   topo,
            "valid":    len(valid),
            "rejected": rejected,
        })

    payload = {
        "meta": {
            "seed":          seed,
            "n_target":      n_target,
            "oversample":    OVERSAMPLE,
            "total_samples": len(all_params),
            "pcb_gap_mm":    PCB_GAP_MM,
            "freq_hz":       FREQ_HZ,
            "min_id_mm":     MIN_ID_MM,
            "summary":       summary,
        },
        "samples": [simparams_to_dict(p) for p in all_params],
    }

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print()
    print("=" * 70)
    print(f"Total valid samples: {len(all_params)}")
    for s in summary:
        print(f"  {s['branch']:<26}  valid={s['valid']:<6}  rejected={s['rejected']}")
    print(f"\nSaved -> {args.out}")
    print("=" * 70)
    print("\nNext: python run_sweep.py --workers 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
