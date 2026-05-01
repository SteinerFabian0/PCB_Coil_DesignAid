"""
domain_lookup.py — helpers for checking if NN inputs lie within training domains.

Each batch's parameter space is captured in domain_batch_N.json.  This module:
  - Loads all domain files from SimulationData/
  - Computes the union of valid ranges across all domains
  - Checks whether a specific input point is covered by any single domain

Field names in the vals dict passed to check_point_in_domain / find_matching_domains:
  tx_od_mm, tx_turns, tx_width, rx_od_mm, rx_turns, rx_width,
  freq_hz, rx_topology, [ground_circle_dia_mm — optional]

Absent fields are treated as unconstrained (always pass the check).
"""

import glob
import json
import os
import re


def load_all_domains(simdata_dir: str) -> list:
    """Return [(batch_num, domain_dict), ...] sorted by batch_num."""
    result = []
    for path in sorted(glob.glob(os.path.join(simdata_dir, "domain_batch_*.json"))):
        m = re.search(r"domain_batch_(\d+)\.json$", path)
        if not m:
            continue
        batch_num = int(m.group(1))
        try:
            with open(path) as f:
                result.append((batch_num, json.load(f)))
        except Exception:
            pass
    return result


def union_ranges(domains: list) -> dict:
    """
    Compute the union of valid input ranges across all domain files.

    Returns a dict with keys:
      tx_od_mm    → (lo, hi)   lo = min(id_min_mm), hi = max(od_max_mm)
      tx_turns    → (lo, hi)
      tx_width    → (lo, hi)
      rx_od_mm    → (lo, hi)
      rx_turns    → (lo, hi)
      rx_width    → (lo, hi)
      freq_hz     → (lo, hi)
      rx_topology → sorted list of all allowed topology strings across batches
    """
    if not domains:
        return {}

    tx_od_lo, tx_od_hi = [], []
    tx_t_lo,  tx_t_hi  = [], []
    tx_w_lo,  tx_w_hi  = [], []
    rx_od_lo, rx_od_hi = [], []
    rx_t_lo,  rx_t_hi  = [], []
    rx_w_lo,  rx_w_hi  = [], []
    f_lo,     f_hi     = [], []
    topos: set = set()

    for _, d in domains:
        tx = d.get("tx", {})
        rx = d.get("rx", {})
        gl = d.get("global", {})

        _accum_range(tx, "id_min_mm", "od_max_mm", tx_od_lo, tx_od_hi)
        _accum_list2(tx, "turns",          tx_t_lo, tx_t_hi)
        _accum_list2(tx, "trace_width_mm", tx_w_lo, tx_w_hi)

        _accum_range(rx, "id_min_mm", "od_max_mm", rx_od_lo, rx_od_hi)
        _accum_list2(rx, "turns",          rx_t_lo, rx_t_hi)
        _accum_list2(rx, "trace_width_mm", rx_w_lo, rx_w_hi)

        _accum_list2(gl, "freq_hz", f_lo, f_hi)

        for t in rx.get("allowed_topologies", []):
            topos.add(t)

    out = {}
    if tx_od_hi:
        out["tx_od_mm"]  = (min(tx_od_lo) if tx_od_lo else 0.0, max(tx_od_hi))
    if tx_t_lo and tx_t_hi:
        out["tx_turns"]  = (min(tx_t_lo), max(tx_t_hi))
    if tx_w_lo and tx_w_hi:
        out["tx_width"]  = (min(tx_w_lo), max(tx_w_hi))
    if rx_od_hi:
        out["rx_od_mm"]  = (min(rx_od_lo) if rx_od_lo else 0.0, max(rx_od_hi))
    if rx_t_lo and rx_t_hi:
        out["rx_turns"]  = (min(rx_t_lo), max(rx_t_hi))
    if rx_w_lo and rx_w_hi:
        out["rx_width"]  = (min(rx_w_lo), max(rx_w_hi))
    if f_lo and f_hi:
        out["freq_hz"]   = (min(f_lo), max(f_hi))
    if topos:
        out["rx_topology"] = sorted(topos)
    return out


def check_point_in_domain(vals: dict, domain: dict) -> bool:
    """
    Return True if all provided fields in vals fall within domain's ranges.

    Ground-circle check is only applied when the caller provides
    ground_circle_dia_mm AND the domain file records ground_circle_enabled.
    Absent fields are always considered valid.
    """
    tx = domain.get("tx", {})
    rx = domain.get("rx", {})
    gl = domain.get("global", {})

    if not _in_range(vals.get("tx_od_mm"), tx.get("id_min_mm"), tx.get("od_max_mm")):
        return False
    if not _in_list2(vals.get("tx_turns"),  tx.get("turns")):
        return False
    if not _in_list2(vals.get("tx_width"),  tx.get("trace_width_mm")):
        return False
    if not _in_range(vals.get("rx_od_mm"), rx.get("id_min_mm"), rx.get("od_max_mm")):
        return False
    if not _in_list2(vals.get("rx_turns"),  rx.get("turns")):
        return False
    if not _in_list2(vals.get("rx_width"),  rx.get("trace_width_mm")):
        return False
    if not _in_list2(vals.get("freq_hz"),   gl.get("freq_hz")):
        return False
    if not _topo_ok(vals.get("rx_topology"), rx.get("allowed_topologies")):
        return False

    # Optional ground-circle check — only when both the domain and the caller
    # record ground_circle intent.
    gc_enabled = domain.get("ground_circle_enabled")
    gc_dia     = vals.get("ground_circle_dia_mm")
    if gc_dia is not None and gc_enabled is not None:
        if gc_enabled:
            gc_min = float(domain.get("ground_circle_min_mm", 0.0))
            gc_max = float(domain.get("ground_circle_max_mm", 0.0))
            if not (gc_min <= gc_dia <= gc_max):
                return False
        else:
            if gc_dia != 0.0:
                return False

    return True


def find_matching_domains(vals: dict, domains: list) -> list:
    """Return sorted list of batch numbers whose domains cover vals."""
    return [n for n, d in domains if check_point_in_domain(vals, d)]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _accum_range(d: dict, lo_key: str, hi_key: str,
                 lo_list: list, hi_list: list) -> None:
    if lo_key in d:
        lo_list.append(float(d[lo_key]))
    if hi_key in d:
        hi_list.append(float(d[hi_key]))


def _accum_list2(d: dict, key: str, lo_list: list, hi_list: list) -> None:
    v = d.get(key)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        lo_list.append(float(v[0]))
        hi_list.append(float(v[1]))


def _in_range(val, lo, hi) -> bool:
    if val is None:
        return True
    if lo is not None and float(val) < float(lo):
        return False
    if hi is not None and float(val) > float(hi):
        return False
    return True


def _in_list2(val, rng) -> bool:
    if val is None or not isinstance(rng, (list, tuple)) or len(rng) < 2:
        return True
    return float(rng[0]) <= float(val) <= float(rng[1])


def _topo_ok(val, allowed) -> bool:
    if val is None or not allowed:
        return True
    return val in allowed
