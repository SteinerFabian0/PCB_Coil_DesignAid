#!/usr/bin/env python3
"""
E-value capacitor combination finder.

Given a target capacitance in nF, returns the best achievable value
using up to 2 caps (single, parallel pair, or series pair) from a
fixed E-value list.

If target is already an E-value exactly, returns (value, None) to
signal "no change needed". Otherwise returns (achieved_nF, desc_str).
"""

E_VALUES_NF = (1.0, 2.2, 3.3, 4.7, 6.8, 8.2, 10.0,
               22.0, 33.0, 47.0, 68.0, 82.0, 100.0)

EXACT_TOL_NF = 1e-6


def _fmt(c):
    return f"{c:g}"


def find_best_cap(target_nf, max_caps=2):
    """
    Args:
        target_nf: desired capacitance in nF (positive).
        max_caps: maximum number of capacitors to combine (1 or 2, default 2).

    Returns:
        (achieved_nf, description) where description is None if target
        is already achievable with a single E-value (no change signal).
    """
    if target_nf <= 0:
        return (target_nf, None)

    # Single E-value exact match → no change.
    for c in E_VALUES_NF:
        if abs(target_nf - c) < EXACT_TOL_NF:
            return (c, None)

    best_val = None
    best_desc = None
    best_err = float("inf")

    def consider(val, desc):
        nonlocal best_val, best_desc, best_err
        err = abs(val - target_nf)
        if err < best_err:
            best_val, best_desc, best_err = val, desc, err

    # Singles.
    for c in E_VALUES_NF:
        consider(c, f"{_fmt(c)} nF")

    # Pairs (i <= j includes same-value pairs).
    if max_caps >= 2:
        for i, c1 in enumerate(E_VALUES_NF):
            for c2 in E_VALUES_NF[i:]:
                consider(c1 + c2, f"{_fmt(c1)}+{_fmt(c2)} nF ||")
                consider((c1 * c2) / (c1 + c2),
                         f"{_fmt(c1)}+{_fmt(c2)} nF series")

    return (best_val, best_desc)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target_nF>")
        sys.exit(1)
    tgt = float(sys.argv[1])
    val, desc = find_best_cap(tgt)
    if desc is None:
        print(f"{tgt} nF: already an E-value ({val} nF)")
    else:
        print(f"{tgt} nF → {val:g} nF via {desc}")