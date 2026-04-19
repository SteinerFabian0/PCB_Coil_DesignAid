#!/usr/bin/env python3
"""
Parse FastHenry's Zc.mat output into Python-native structures.

Zc.mat format (single-port example):

    Row 1:  n0  to  n4485
    Impedance matrix for frequency = 130000 1 x 1
           1.70447      +4.37211j

Multi-frequency files repeat the "Impedance matrix..." block.
Multi-port files have larger matrices within each block.

For our use case (single-port two-layer coil with one frequency of
interest) we'll almost always hit the 1x1 case, but the parser handles
the general NxN case because it's not harder.
"""

import re
import math


_FREQ_HEADER_RE = re.compile(
    r"Impedance\s+matrix\s+for\s+frequency\s*=\s*([-\d.eE+]+)"
    r"\s+(\d+)\s*x\s*(\d+)",
    re.IGNORECASE,
)

# A matrix entry: "  1.70447      +4.37211j"  or  "  -nan(ind)     -nan(ind)j"
# The NaN/Inf variants appear when FastHenry doesn't converge.
_SCALAR_PAT = r"[-+]?(?:nan(?:\([^)]*\))?|inf|\d+\.?\d*(?:[eE][-+]?\d+)?)"
_SIGNED_PAT = r"[-+](?:nan(?:\([^)]*\))?|inf|\d+\.?\d*(?:[eE][-+]?\d+)?)"
_COMPLEX_RE = re.compile(
    rf"({_SCALAR_PAT})\s*({_SIGNED_PAT})j",
    re.IGNORECASE,
)


def _parse_scalar(s):
    """Parse a float string that may contain 'nan(ind)' or 'inf'."""
    sl = s.lower()
    if "nan" in sl:
        return float("nan")
    if "inf" in sl:
        return float("-inf") if sl.lstrip().startswith("-") else float("inf")
    return float(s)


def parse_zc_mat(filepath):
    """
    Parse a Zc.mat file.

    Returns:
        [
          {"frequency": 130000.0,
           "size": (rows, cols),
           "matrix": [[complex, complex, ...], ...]},
          ...
        ]

    One entry per frequency block in the file.
    """
    results = []
    current = None
    row_buffer = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = _FREQ_HEADER_RE.search(line)
            if m:
                # Flush previous block if any.
                if current is not None:
                    current["matrix"] = row_buffer
                    results.append(current)
                current = {
                    "frequency": float(m.group(1)),
                    "size": (int(m.group(2)), int(m.group(3))),
                    "matrix": None,
                }
                row_buffer = []
                continue

            if current is None:
                # Header line ("Row 1: n0 to n4485") or similar - skip.
                continue

            # Parse a row of complex numbers (NaN/Inf included).
            row = [
                complex(_parse_scalar(re_), _parse_scalar(im))
                for re_, im in _COMPLEX_RE.findall(line)
            ]
            if row:
                row_buffer.append(row)

        # Flush the trailing block.
        if current is not None:
            current["matrix"] = row_buffer
            results.append(current)

    return results


def impedance_at(blocks, target_freq_hz, port_row=0, port_col=0):
    """
    Pull the single complex impedance value at a given frequency for a
    specific port pair. Matches the frequency whose block value is closest
    to target_freq_hz (since FastHenry sweeps can slightly overshoot).

    Returns a complex number: Z = R + jX.
    """
    if not blocks:
        raise ValueError("Empty Zc.mat data")

    closest = min(blocks, key=lambda b: abs(b["frequency"] - target_freq_hz))
    matrix = closest["matrix"]
    return matrix[port_row][port_col]


def inductance_from_z(z_complex, frequency_hz):
    """
    Convert a complex impedance at a given frequency into inductance (H).

    L = imag(Z) / (2 * pi * f). Assumes series L-R model, which is what
    FastHenry outputs for a single-port extraction.
    """
    return z_complex.imag / (2.0 * math.pi * frequency_hz)


def resistance_from_z(z_complex):
    """AC resistance at the sweep frequency in ohms."""
    return z_complex.real


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} Zc.mat")
        sys.exit(1)

    blocks = parse_zc_mat(sys.argv[1])
    for b in blocks:
        f = b["frequency"]
        z = b["matrix"][0][0]
        L_uH = inductance_from_z(z, f) * 1e6
        print(f"f = {f:.0f} Hz:  Z = {z.real:.4f} + {z.imag:.4f}j Ohm"
              f"   L = {L_uH:.4f} uH")
        
def port_count(blocks):
    """Number of ports inferred from the first block's matrix size."""
    if not blocks:
        return 0
    rows, cols = blocks[0]["size"]
    if rows != cols:
        return 0
    return rows


def matrix_at(blocks, target_freq_hz):
    """
    Full NxN complex impedance matrix at the frequency closest to target.
    Returns (actual_freq_hz, list-of-rows-of-complex).
    """
    if not blocks:
        raise ValueError("Empty Zc.mat data")
    closest = min(blocks, key=lambda b: abs(b["frequency"] - target_freq_hz))
    return closest["frequency"], closest["matrix"]


if __name__ == "__main__":
    main()