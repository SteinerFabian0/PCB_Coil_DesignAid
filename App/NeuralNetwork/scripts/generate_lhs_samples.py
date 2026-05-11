#!/usr/bin/env python3
"""
Generate Latin Hypercube Samples for the combined TX+RX design-space sweep.

Fixed (not sampled, not NN inputs):
  TX: series topology, port outside, L1+L2 active, 1 oz outer / 0.5 oz inner
  RX: all 4 layers active, port inside, 1 oz outer / 0.5 oz inner

Free variables (sampled → NN inputs):
  TX: od, turns (L1), l2_turns, trace_width, spacing, l1l2_gap, pcb_gap
  RX: od (≥ TX od, within 1.4 mm), turns, trace_width, spacing, outer_gap, inner_gap, topology
  freq: linearly assigned across sample index (not an LHS dimension)

CLI usage:
    python generate_lhs_samples.py --config domain_master.json
                                    --out    lhs_samples.json
                                    --n      20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid as uuid_module

from scipy.stats import qmc

_HERE        = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT    = os.path.dirname(os.path.dirname(_HERE))
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")

for _p in (_HERE, _MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parametric_coil as pc  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARDCODED_SEED = 42

_MIN_TX_TURNS   = 3
_MIN_TX_L2_TURNS = 1
_MIN_RX_TURNS   = 3

# Fixed stackup — never sampled, never NN inputs.
_TX_OUTER_OZ  = 1.0
_TX_INNER_OZ  = 0.5
_TX_LAYERS    = [
    {"active": True,  "copper_oz": _TX_OUTER_OZ},
    {"active": True,  "copper_oz": _TX_INNER_OZ},
    {"active": False, "copper_oz": 1.0},
    {"active": False, "copper_oz": 1.0},
]
_RX_OUTER_OZ  = 1.0
_RX_INNER_OZ  = 0.5
_RX_LAYERS    = [
    {"active": True, "copper_oz": _RX_OUTER_OZ},
    {"active": True, "copper_oz": _RX_INNER_OZ},
    {"active": True, "copper_oz": _RX_INNER_OZ},
    {"active": True, "copper_oz": _RX_OUTER_OZ},
]

# RX topologies — encoded as int in samples (0=series, 1=parallel_pairs_ser).
_RX_TOPOS = ["series", "parallel_pairs_ser"]

DEFAULT_DOMAIN = {
    "tx": {
        "od_mm":            [50.0, 54.0],  # sampled range [min, max]
        "id_min_mm":        39.0,          # feasibility floor only
        "trace_width_mm":   [0.2, 1.2],
        "trace_spacing_mm": [0.16, 0.16],
        "turns":            [6, 18],
        "l2_turns_frac":    [0.4, 1.0],
        "l1l2_gap_mm":      [0.2104, 0.2104],
        "nhinc":            1,
        "nwinc":            3,
    },
    "rx": {
        "od_mm":            [50.0, 54.0],  # sampled; constrained to [tx_od, tx_od+1.4] at decode time
        "id_min_mm":        35.0,          # feasibility floor only
        "trace_width_mm":   [0.2, 1.2],
        "trace_spacing_mm": [0.16, 0.16],
        "turns":            [4, 25],
        "outer_gap_mm":     [0.2104, 0.2104],
        "inner_gap_mm":     [0.6, 0.6],
        "ground_disc_dia_mm": 20.0,        # passive copper on both inner layers; 0 disables
        "nhinc":            1,
        "nwinc":            3,
    },
    "global": {
        "pcb_gap_mm":    [2.6, 2.6],
        "resolution_mm": 1.2,
        "freq_hz":       [280_000.0, 350_000.0],  # linearly assigned per sample index
    },
    "n_total": 20_000,
}


# ---------------------------------------------------------------------------
# UUID for deduplication
# ---------------------------------------------------------------------------

def sample_uuid(sample: dict) -> str:
    SIM_KEYS = [
        "tx_od", "tx_turns", "tx_l2_turns", "tx_w", "tx_s", "tx_l1l2_gap",
        "rx_od", "rx_turns", "rx_w", "rx_s", "rx_outer_gap", "rx_inner_gap",
        "rx_topo", "rx_ground_disc_dia",
        "pcb_gap", "freq", "resolution",
    ]
    canonical = {k: sample[k] for k in SIM_KEYS if k in sample}
    payload = json.dumps(canonical, sort_keys=True, default=str).encode()
    h = hashlib.sha256(payload).digest()[:16]
    return str(uuid_module.UUID(bytes=h))


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def _scale(u, lo, hi):
    return lo + u * (hi - lo)


def _round_int(v, lo, hi):
    return int(round(max(lo, min(hi, v))))


def _stackup_val(u: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return round(lo, 4)
    options = [lo, (lo + hi) / 2.0, hi]
    idx = min(int(u * len(options)), len(options) - 1)
    return round(options[idx], 4)



# ---------------------------------------------------------------------------
# Config canonicalisation
# ---------------------------------------------------------------------------

def canonicalize(cfg: dict) -> dict:
    out = {"tx": {}, "rx": {}, "global": {}, "n_total": int(cfg.get("n_total", 0))}
    for side in ("tx", "rx"):
        src = cfg.get(side) or {}
        dst = dict(DEFAULT_DOMAIN[side])
        dst.update(src)
        # Backwards-compat: if old single od_max_mm supplied, convert to range.
        if "od_mm" not in dst and "od_max_mm" in dst:
            dst["od_mm"] = [dst["od_max_mm"], dst["od_max_mm"]]
        out[side] = dst
    src_g = cfg.get("global") or {}
    out["global"] = {**DEFAULT_DOMAIN["global"], **src_g}
    g = out["global"]
    if not isinstance(g.get("pcb_gap_mm"), list):
        v = float(g.get("pcb_gap_mm", 2.6))
        g["pcb_gap_mm"] = [v, v]
    # Remove legacy freq_step_hz if present (freq is now index-assigned).
    g.pop("freq_step_hz", None)
    return out


def config_hash(cfg: dict) -> str:
    payload = json.dumps(canonicalize(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def _feasible_tx(tx_cfg, od, w, s, t, l1l2_gap, l2_turns) -> tuple:
    sp = pc.SpiralParams(od_mm=od, trace_width_mm=w, spacing_mm=s,
                         turns=t, resolution_mm=1.0)
    ok, msg = pc.validate_spiral(sp)
    if not ok:
        return False, msg
    id_mm = 2.0 * (sp.r_inner_centerline - w / 2.0)
    if id_mm < tx_cfg.get("id_min_mm", 0.0):
        return False, f"TX ID {id_mm:.2f} mm < {tx_cfg.get('id_min_mm', 0.0)} mm"
    stackup = pc.StackUp(
        slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
               for l in _TX_LAYERS],
        outer_gap_mm=l1l2_gap, inner_gap_mm=0.0,
    )
    ok, msg = pc.validate_stackup(stackup, min_active=2, max_active=2)
    if not ok:
        return False, f"TX stackup: {msg}"
    try:
        _, w2 = pc.compute_layer2_width(sp, l2_turns)
        if w2 <= 0:
            return False, f"TX L2 width {w2:.4f} mm <= 0"
    except Exception as e:
        return False, str(e)
    return True, ""


def _feasible_rx(rx_cfg, od, w, s, t, og, ig) -> tuple:
    sp = pc.SpiralParams(od_mm=od, trace_width_mm=w, spacing_mm=s,
                         turns=t, resolution_mm=1.0)
    ok, msg = pc.validate_spiral(sp)
    if not ok:
        return False, msg
    id_mm = 2.0 * (sp.r_inner_centerline - w / 2.0)
    if id_mm < rx_cfg.get("id_min_mm", 0.0):
        return False, f"RX ID {id_mm:.2f} mm < {rx_cfg.get('id_min_mm', 0.0)} mm"
    stackup = pc.StackUp(
        slots=[pc.LayerSlot(active=l["active"], copper_oz=l["copper_oz"])
               for l in _RX_LAYERS],
        outer_gap_mm=og, inner_gap_mm=ig,
    )
    ok, msg = pc.validate_stackup(stackup)
    return ok, msg


# ---------------------------------------------------------------------------
# LHS decoding — 13 dimensions (freq is assigned linearly per sample index)
#   0  tx_turns
#   1  tx_w
#   2  tx_od         (sampled from tx["od_mm"] range)
#   3  tx_s
#   4  tx_l1l2_gap
#   5  tx_l2_turns / tx_l2_turns_frac
#   6  rx_turns
#   7  rx_w
#   8  rx_od_offset  (0..1 → [tx_od, min(tx_od+1.4, rx_od_max)])
#   9  rx_s
#  10  rx_outer_gap
#  11  rx_inner_gap
#  12  rx_topo  (categorical)
# ---------------------------------------------------------------------------

_LHS_DIMS = 13
_RX_OD_MAX_OFFSET_MM = 1.4  # RX OD always within this of TX OD, and RX OD >= TX OD


def _decode(u_row, cfg, sample_index: int = 0, n_total: int = 1):
    tx, rx, glob = cfg["tx"], cfg["rx"], cfg["global"]

    tx_turns = max(_round_int(_scale(u_row[0], *tx["turns"]), *tx["turns"]),
                   _MIN_TX_TURNS)
    tx_w     = round(_scale(u_row[1], *tx["trace_width_mm"]), 4)

    tx_od_range = tx.get("od_mm", [tx.get("od_max_mm", 54.0)] * 2)
    tx_od    = round(_scale(u_row[2], tx_od_range[0], tx_od_range[1]), 4)

    tx_s     = _stackup_val(u_row[3], *tx["trace_spacing_mm"])
    tx_gap   = round(_scale(u_row[4], *tx["l1l2_gap_mm"]), 4)

    _l2_min_from_l1 = -(-tx_turns // 2)  # ceil(tx_turns / 2)
    if "l2_turns" in tx:
        l2_lo, l2_hi = tx["l2_turns"]
        tx_l2_turns = max(_MIN_TX_L2_TURNS, _l2_min_from_l1,
                          min(tx_turns, _round_int(_scale(u_row[5], l2_lo, l2_hi), l2_lo, l2_hi)))
    else:
        frac_lo, frac_hi = tx.get("l2_turns_frac", [0.4, 1.0])
        l2_frac = _scale(u_row[5], frac_lo, frac_hi)
        tx_l2_turns = max(_MIN_TX_L2_TURNS, _l2_min_from_l1,
                          min(tx_turns, int(round(l2_frac * tx_turns))))

    rx_turns = max(_round_int(_scale(u_row[6], *rx["turns"]), *rx["turns"]),
                   _MIN_RX_TURNS)
    rx_w     = round(_scale(u_row[7], *rx["trace_width_mm"]), 4)

    # RX OD: always >= TX OD, and within _RX_OD_MAX_OFFSET_MM above TX OD.
    # u_row[8] spans the full [0,1] range so RX OD varies uniformly in the offset band.
    rx_od_range = rx.get("od_mm", [rx.get("od_max_mm", 54.0)] * 2)
    rx_od_hi = min(tx_od + _RX_OD_MAX_OFFSET_MM, rx_od_range[1])
    rx_od_lo = tx_od  # RX OD >= TX OD
    if rx_od_hi < rx_od_lo:
        rx_od_hi = rx_od_lo
    rx_od    = round(_scale(u_row[8], rx_od_lo, rx_od_hi), 4)

    rx_s     = _stackup_val(u_row[9],  *rx["trace_spacing_mm"])
    rx_og    = _stackup_val(u_row[10], *rx["outer_gap_mm"])
    rx_ig    = _stackup_val(u_row[11], *rx["inner_gap_mm"])

    n_topos  = len(_RX_TOPOS)
    rx_topo  = min(int(u_row[12] * n_topos), n_topos - 1)  # 0 or 1

    # Frequency: linearly distributed across sample index, independent of LHS.
    freq_lo, freq_hi = glob["freq_hz"]
    if n_total > 1:
        freq = freq_lo + (freq_hi - freq_lo) * sample_index / (n_total - 1)
    else:
        freq = freq_lo
    freq = round(freq, 1)

    pcb_gap  = _stackup_val(0.5, *glob["pcb_gap_mm"])  # fixed mid-point

    rx_gdisc = float(rx.get("ground_disc_dia_mm", 20.0))

    return {
        "tx_od": tx_od, "tx_turns": tx_turns, "tx_l2_turns": tx_l2_turns,
        "tx_w": tx_w, "tx_s": tx_s, "tx_l1l2_gap": tx_gap,
        "rx_od": rx_od, "rx_turns": rx_turns,
        "rx_w": rx_w, "rx_s": rx_s,
        "rx_outer_gap": rx_og, "rx_inner_gap": rx_ig,
        "rx_topo": rx_topo,
        "rx_ground_disc_dia": rx_gdisc,
        "pcb_gap": pcb_gap, "freq": freq,
        "resolution": glob["resolution_mm"],
    }


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_samples(cfg: dict, n_total: int, seed: int = HARDCODED_SEED,
                     oversample: int = 32, log=print) -> list:
    cfg = canonicalize(cfg)
    n_candidates = max(n_total * oversample, n_total + 100)
    sampler = qmc.LatinHypercube(d=_LHS_DIMS, seed=seed)
    raw = sampler.random(n_candidates)

    valid, rejected = [], 0
    for row in raw:
        if len(valid) >= n_total:
            break
        c = _decode(row, cfg, sample_index=len(valid), n_total=n_total)

        ok, _ = _feasible_tx(cfg["tx"],
                              c["tx_od"], c["tx_w"], c["tx_s"],
                              c["tx_turns"], c["tx_l1l2_gap"], c["tx_l2_turns"])
        if not ok:
            rejected += 1
            continue

        ok, _ = _feasible_rx(cfg["rx"],
                              c["rx_od"], c["rx_w"], c["rx_s"],
                              c["rx_turns"], c["rx_outer_gap"], c["rx_inner_gap"])
        if not ok:
            rejected += 1
            continue

        c["uuid"] = sample_uuid(c)
        valid.append(c)

    log(f"LHS: {len(valid)} valid / {rejected} rejected out of {n_candidates} candidates")
    return valid


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_domain_master(cfg: dict, out_path: str, n_total: int, n_valid: int) -> str:
    canon = canonicalize(cfg)
    canon["n_total"]    = int(n_total)
    canon["n_valid"]    = int(n_valid)
    canon["seed"]       = HARDCODED_SEED
    canon["fixed"] = {
        "tx_topology":   "series",
        "tx_port_inside": False,
        "tx_layers":     _TX_LAYERS,
        "rx_port_inside": True,
        "rx_layers":     _RX_LAYERS,
        "rx_topos":      _RX_TOPOS,
    }
    h = config_hash(cfg)
    canon["config_hash"] = h
    with open(out_path, "w") as f:
        json.dump(canon, f, indent=2, default=str)
    return h


# Results file (results.json) is written by run_sweep.py as a list of verbose
# dicts under the "results" key — see run_sweep.py docstring for the schema.


def write_samples_file(samples: list, cfg: dict, out_path: str,
                       config_hash_: str) -> None:
    payload = {
        "meta": {
            "n_samples":   len(samples),
            "seed":        HARDCODED_SEED,
            "config_hash": config_hash_,
        },
        "samples": samples,
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic LHS sampler")
    parser.add_argument("--config",  required=True)
    parser.add_argument("--out-dir", dest="out_dir", required=True)
    parser.add_argument("--n",       type=int, default=None)
    args = parser.parse_args()

    cfg     = _load_config(args.config)
    n_total = args.n if args.n is not None else cfg.get("n_total", 0)
    if not n_total or n_total <= 0:
        print("ERROR: n_total must be > 0", file=sys.stderr)
        return 2

    samples = generate_samples(cfg, n_total)
    if not samples:
        print("ERROR: no feasible samples — check the domain config", file=sys.stderr)
        return 3

    os.makedirs(args.out_dir, exist_ok=True)
    master_path = os.path.join(args.out_dir, "domain.json")
    h = write_domain_master(cfg, master_path, n_total, len(samples))

    out_path = os.path.join(args.out_dir, "lhs_samples.json")
    write_samples_file(samples, cfg, out_path, h)

    print(f"Wrote {len(samples)} samples -> {out_path}")
    print(f"Domain file         -> {master_path}  (hash {h})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
