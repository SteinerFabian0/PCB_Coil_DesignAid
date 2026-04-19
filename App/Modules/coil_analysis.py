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

# ---------------------------------------------------------------------------
# System-level circuit analysis (TX series-LC + RX parallel-LC + coupling)
# ---------------------------------------------------------------------------

def tx_series_branch_z(L_h, R_ac_ohm, C_f, f_hz):
    """TX branch: R_ac + jωL + 1/(jωC) = R_ac + j(ωL - 1/ωC)."""
    omega = 2.0 * math.pi * f_hz
    X_L = omega * L_h
    X_C = 1.0 / (omega * C_f) if C_f > 0 else 0.0
    return complex(R_ac_ohm, X_L - X_C)


def rx_parallel_tank_z(L_h, R_ac_ohm, C_f, R_load_ohm, f_hz):
    """
    RX tank: coil branch (R_ac + jωL) in parallel with C_f and R_load.
    Computed via Y_sum = 1/Z_coil + jωC + 1/R_load.
    """
    omega = 2.0 * math.pi * f_hz
    Z_coil = complex(R_ac_ohm, omega * L_h)
    Y = 1.0 / Z_coil
    if C_f > 0:
        Y += complex(0.0, omega * C_f)
    if R_load_ohm and R_load_ohm > 0:
        Y += 1.0 / R_load_ohm
    if Y == 0:
        return complex(float("inf"), 0)
    return 1.0 / Y


def reflected_impedance(M_h, Z_rx_secondary, f_hz):
    """
    Impedance reflected into TX primary from RX secondary:
    Z_ref = (ωM)² / Z_rx_secondary
    Z_rx_secondary is whatever TX "sees" through the RX loop (RX coil
    + tuning C + load), i.e. the full secondary-loop impedance.
    """
    omega = 2.0 * math.pi * f_hz
    if Z_rx_secondary == 0 or abs(Z_rx_secondary) < 1e-18:
        return complex(float("inf"), 0)
    return (omega * M_h) ** 2 / Z_rx_secondary


def coupling_k(L1_h, L2_h, M_h):
    if L1_h <= 0 or L2_h <= 0:
        return 0.0
    denom = math.sqrt(L1_h * L2_h)
    return M_h / denom if denom > 0 else 0.0


def square_wave_fundamental_rms(v_supply):
    """
    Fundamental (1st harmonic) RMS of a half-bridge 50% square wave
    swinging between 0 and V_supply:
    V_fund_peak = (2/π) * V_supply
    V_fund_rms  = V_fund_peak / √2 = V_supply * √2 / π
    Higher harmonics are rejected by the tuned LC and neglected here.
    """
    return v_supply * math.sqrt(2.0) / math.pi


def tx_rx_system_analysis(L_tx, R_tx, L_rx, R_rx, M, f_hz,
                          C_tx, C_rx, R_load,
                          V_supply_min, V_supply_max,
                          duty_fraction):
    """
    Pure math — returns a dict of scalars for the GUI to display.
    Assumes: TX driven by half-bridge square wave (fundamental only),
    TX topology = series LC, RX topology = parallel LC with R_load
    across the tank. Reflected impedance couples RX loading back to TX.
    """
    omega = 2.0 * math.pi * f_hz
    out = {}

    # Nominal LC resonant frequencies.
    out["f0_tx"] = series_resonant_freq_hz(L_tx, C_tx)
    out["f0_rx"] = series_resonant_freq_hz(L_rx, C_rx)

    # Coupling.
    k = coupling_k(L_tx, L_rx, M)
    out["k"] = k

    # Q factors at drive frequency (unloaded, coil-only).
    out["Q_tx"] = q_factor(L_tx, R_tx, f_hz)
    out["Q_rx"] = q_factor(L_rx, R_rx, f_hz)

    # RX-side impedance as seen from TX through mutual coupling:
    # The secondary loop is R_rx + jωL_rx + Z_tank_branch_excl_coil,
    # but with the RX-parallel-LC convention the load and C sit across
    # the coil, which reflects into TX via (ωM)² / Z_secondary where
    # Z_secondary = jωL_rx + R_rx in series with (C_rx || R_load).
    Z_rx_loadside = _parallel(1.0 / (1j * omega * C_rx) if C_rx > 0 else complex(1e18),
                              complex(R_load, 0) if R_load > 0 else complex(1e18))
    Z_rx_secondary = complex(R_rx, omega * L_rx) + Z_rx_loadside
    Z_ref = reflected_impedance(M, Z_rx_secondary, f_hz)
    out["Z_ref"] = Z_ref
    out["Z_rx_secondary"] = Z_rx_secondary

    # TX-side total impedance = TX series-LC branch + reflected Z.
    Z_tx_branch = tx_series_branch_z(L_tx, R_tx, C_tx, f_hz)
    Z_tx_total = Z_tx_branch + Z_ref
    out["Z_tx_branch"] = Z_tx_branch
    out["Z_tx_total"] = Z_tx_total

    # Drive model: fundamental RMS of 50% square wave.
    v_rms_min = square_wave_fundamental_rms(V_supply_min)
    v_rms_max = square_wave_fundamental_rms(V_supply_max)

    # TX-side instantaneous (ON-state) power draw = V²/|Z| * cos(φ)?
    # We want active (real) input power: P = V_rms² * Re(1/Z_tx_total).
    def p_in(v_rms):
        if abs(Z_tx_total) < 1e-18:
            return float("inf")
        return (v_rms ** 2) * (1.0 / Z_tx_total).real

    P_on_min = p_in(v_rms_min)
    P_on_max = p_in(v_rms_max)
    out["P_in_on_Vmin"] = P_on_min
    out["P_in_on_Vmax"] = P_on_max
    out["P_in_avg_Vmin"] = P_on_min * duty_fraction
    out["P_in_avg_Vmax"] = P_on_max * duty_fraction

    # Primary current RMS (fundamental) for sanity.
    def i_rms(v_rms):
        if abs(Z_tx_total) < 1e-18:
            return float("inf")
        return v_rms / abs(Z_tx_total)
    
    out["I_tx_rms_Vmin"] = i_rms(v_rms_min)
    out["I_tx_rms_Vmax"] = i_rms(v_rms_max)

    # Power dissipated in Z_ref → this is the real power crossing into
    # the secondary loop (before RX coil losses and load split).
    def p_ref(v_rms):
        I = i_rms(v_rms)
        return (I ** 2) * Z_ref.real
    
    P_sec_min = p_ref(v_rms_min)
    P_sec_max = p_ref(v_rms_max)

    # Of the secondary-loop power, the load fraction is
    # Re(Z_rx_loadside) / Re(Z_rx_secondary).
    re_sec = Z_rx_secondary.real
    re_load = Z_rx_loadside.real
    load_frac = (re_load / re_sec) if re_sec > 0 else 0.0
    out["P_rect_on_Vmin"] = P_sec_min * load_frac
    out["P_rect_on_Vmax"] = P_sec_max * load_frac
    out["P_rect_avg_Vmin"] = out["P_rect_on_Vmin"] * duty_fraction
    out["P_rect_avg_Vmax"] = out["P_rect_on_Vmax"] * duty_fraction

    # Estimated rectified DC voltage assuming full-bridge: V_dc ≈ V_ac_rms·√2
    # minus 2 diode drops; caller can treat 0.7 V per diode if desired.
    # Here we give raw peak: V_peak = I_load_rms * R_load * √2 if R_load.
    if R_load > 0:
        V_load_rms_min = math.sqrt(max(0.0, out["P_rect_on_Vmin"] * R_load))
        V_load_rms_max = math.sqrt(max(0.0, out["P_rect_on_Vmax"] * R_load))
        out["V_rect_peak_Vmin"] = V_load_rms_min * math.sqrt(2.0)
        out["V_rect_peak_Vmax"] = V_load_rms_max * math.sqrt(2.0)
    else:
        out["V_rect_peak_Vmin"] = 0.0
        out["V_rect_peak_Vmax"] = 0.0

    return out


def _parallel(z1, z2):
    """Helper: impedances in parallel."""
    if z1 == 0 or z2 == 0:
        return complex(0, 0)
    return (z1 * z2) / (z1 + z2)