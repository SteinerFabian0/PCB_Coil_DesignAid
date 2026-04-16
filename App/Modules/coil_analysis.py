#!/usr/bin/env python3
"""
Derived metrics from coil geometry and solver output.

All functions are pure (no I/O, no side effects). The GUI calls them
with numbers and displays the results.

Units convention:
    - Lengths in mm when they come from the node list (our .inp is mm).
    - Resistance in ohms.
    - Inductance in henries (convert to uH at the display layer).
    - Capacitance in farads (GUI converts from the user's nF input).
"""

import math


# Copper properties. These are "good enough for a rough sanity check"
# values, not a datasheet-accurate model.
COPPER_RHO_20C_OHM_M = 1.724e-8    # resistivity at 20 C in Ohm*m
COPPER_ALPHA = 0.00393             # temperature coefficient per C


# -------------------------------------------------------------------------
# Path length
# -------------------------------------------------------------------------

def path_length_mm(nodes):
    """
    Sum the 3D edge lengths along a sequential node list.

    `nodes` is a list of (x, y, z) tuples. Edges are implicit: node i
    connects to node i+1. This matches how our .inp files are written
    (one edge per consecutive pair, plus one via edge if merged — the
    via is just another consecutive pair in the combined list).
    """
    total = 0.0
    for i in range(len(nodes) - 1):
        dx = nodes[i + 1][0] - nodes[i][0]
        dy = nodes[i + 1][1] - nodes[i][1]
        dz = nodes[i + 1][2] - nodes[i][2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


# -------------------------------------------------------------------------
# DC resistance
# -------------------------------------------------------------------------

def copper_rho_at(temp_c):
    """Resistivity of copper in Ohm*m, linearly corrected to temp_c."""
    return COPPER_RHO_20C_OHM_M * (1.0 + COPPER_ALPHA * (temp_c - 20.0))


def dc_resistance_ohm(length_mm, width_mm, height_mm, temp_c=30.0):
    """
    Simple DC resistance estimate: R = rho * L / A.

    This ignores the via's smaller effective cross-section. For a
    two-layer coil the via contributes a negligible fraction of the
    total path length (maybe 0.05 mm vs. hundreds of mm of trace), so
    the error is well under 1%. Good enough for comparison.
    """
    rho = copper_rho_at(temp_c)
    length_m = length_mm * 1e-3
    area_m2 = (width_mm * 1e-3) * (height_mm * 1e-3)
    if area_m2 <= 0 or length_m <= 0:
        return 0.0
    return rho * length_m / area_m2


# -------------------------------------------------------------------------
# Design quality indicator
# -------------------------------------------------------------------------

def quality_ratio_uh_per_mohm(inductance_h, dc_resistance_ohm):
    """
    uH divided by mOhm of DC resistance. Higher = better design for a
    given coil footprint, because you get more stored magnetic energy
    per joule of ohmic loss. At a fixed drive frequency this ratio is
    proportional to the unloaded Q factor.

    Returns 0 if resistance is zero or negative (protects against
    division-by-zero when called with degenerate inputs).
    """
    if dc_resistance_ohm <= 0:
        return 0.0
    uH = inductance_h * 1e6
    mOhm = dc_resistance_ohm * 1e3
    return uH / mOhm


# -------------------------------------------------------------------------
# LC resonance
# -------------------------------------------------------------------------

def series_resonant_freq_hz(inductance_h, capacitance_f):
    """
    Natural resonance of a series L-C pair:  f0 = 1 / (2*pi*sqrt(L*C)).

    At f0 the reactive impedance is minimized — a series LC ideally has
    Z = R (pure resistance) at resonance. Returns 0 if either operand
    is non-positive so the GUI can display "-" instead of a NaN.
    """
    if inductance_h <= 0 or capacitance_f <= 0:
        return 0.0
    return 1.0 / (2.0 * math.pi * math.sqrt(inductance_h * capacitance_f))


def reactance_ohm(inductance_h, frequency_hz):
    """Inductive reactance X_L = 2*pi*f*L at a given frequency."""
    return 2.0 * math.pi * frequency_hz * inductance_h


def q_factor(inductance_h, ac_resistance_ohm, frequency_hz):
    """
    Unloaded Q = omega*L / R_ac.

    Uses the solver's AC resistance, not the DC approximation, because
    at 100+ kHz skin effect starts to matter. FastHenry accounts for
    that when we asked for nwinc=7 filament subdivision.
    """
    if ac_resistance_ohm <= 0:
        return 0.0
    return reactance_ohm(inductance_h, frequency_hz) / ac_resistance_ohm