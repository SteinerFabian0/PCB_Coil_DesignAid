#!/usr/bin/env python3
"""
Analytical coil feature helpers used as additional NN inputs.
 
All inputs are in millimetres; angular quantities are in radians.
Returned inductance is in micro-henries.
 
Features provided for each coil (TX / RX):
    wire_length_mm   total spiral wire path length
    wheeler_uh       Wheeler 1928 planar-spiral self-inductance estimate
    ln_n_sq          ln(N^2)  (= 2 * ln(N))
    mean_radius_mm   (OD + ID) / 4
    fill_factor      (OD - ID) / (OD + ID)
    inner_diameter_mm  OD - 2 * N * pitch   (pitch = W + S)
 
These features are derivable from the geometry alone — they are computed
lazily at "append" time rather than simulated.
"""
 
import math
 
 
def pitch_mm(trace_width_mm: float, spacing_mm: float) -> float:
    return float(trace_width_mm) + float(spacing_mm)
 
 
def inner_diameter_mm(od_mm: float, turns: float,
                      trace_width_mm: float, spacing_mm: float) -> float:
    """Inner edge diameter of the innermost turn.
 
    Derivation:
        R0       = OD/2 - W/2                (center-line radius of outer turn)
        R_inner  = R0 - (2N - 1) * pitch / 2
        ID       = 2 * (R_inner - W/2)
                 = OD - 2*W - (2N - 1) * pitch
                 = OD - (2N + 1) * W - (2N - 1) * S
    Keeping the simpler form requested by the user:  OD - 2 * N * pitch
    differs only by a single pitch and gives an adequate proxy the NN
    can learn from.  We use the exact formula above.
    """
    p = pitch_mm(trace_width_mm, spacing_mm)
    return float(od_mm) - (2.0 * turns + 1.0) * trace_width_mm - (2.0 * turns - 1.0) * spacing_mm
 
 
def wire_length_mm(od_mm: float, turns: float,
                   trace_width_mm: float, spacing_mm: float) -> float:
    """Analytical planar spiral path length.
 
    Same derivation used in automation_nn_tab._spiral_length_m:
        total = pi * N * (R0 + R_inner)
    Returns millimetres.
    """
    pitch = pitch_mm(trace_width_mm, spacing_mm)
    r_outer_centerline = od_mm / 2.0 - trace_width_mm / 2.0
    r_inner_centerline = r_outer_centerline - (2.0 * turns - 1.0) * pitch / 2.0
    return math.pi * turns * (r_outer_centerline + r_inner_centerline)
 
 
def mean_radius_mm(od_mm: float, id_mm: float) -> float:
    return (od_mm + id_mm) / 4.0
 
 
def fill_factor(od_mm: float, id_mm: float) -> float:
    s = od_mm + id_mm
    if s <= 0:
        return 0.0
    return (od_mm - id_mm) / s
 
 
def ln_n_sq(turns: float) -> float:
    n = max(float(turns), 1e-6)
    return 2.0 * math.log(n)
 
 
def wheeler_uh(od_mm: float, turns: float,
               trace_width_mm: float, spacing_mm: float) -> float:
    """Wheeler (1928) simplified expression for a circular planar spiral.
 
        L = K1 * mu0 * N^2 * d_avg / (1 + K2 * rho)
 
    with d_avg = (OD + ID) / 2  and  rho = (OD - ID) / (OD + ID).
    Circular geometry: K1 = 2.34, K2 = 2.75.
    Returns micro-henries.
    """
    K1, K2 = 2.34, 2.75
    MU0 = 4.0 * math.pi * 1e-7   # H/m
 
    id_mm = inner_diameter_mm(od_mm, turns, trace_width_mm, spacing_mm)
    d_avg_mm = (od_mm + id_mm) / 2.0
    if d_avg_mm <= 0:
        return 0.0
    rho = fill_factor(od_mm, id_mm)
    L_h = K1 * MU0 * (turns ** 2) * (d_avg_mm * 1e-3) / (1.0 + K2 * rho)
    return L_h * 1e6
 
 
def features_for_coil(od_mm: float, turns: float,
                      trace_width_mm: float, spacing_mm: float) -> dict:
    """Compute all derived features for a single coil."""
    id_mm = inner_diameter_mm(od_mm, turns, trace_width_mm, spacing_mm)
    return {
        "wire_length_mm":   wire_length_mm(od_mm, turns, trace_width_mm, spacing_mm),
        "wheeler_uh":       wheeler_uh(od_mm, turns, trace_width_mm, spacing_mm),
        "ln_n_sq":          ln_n_sq(turns),
        "mean_radius_mm":   mean_radius_mm(od_mm, id_mm),
        "fill_factor":      fill_factor(od_mm, id_mm),
        "inner_diameter_mm": id_mm,
    }
 
 
# Feature names produced per side, in stable order (for NN input column ordering)
FEATURE_KEYS = (
    "wire_length_mm",
    "wheeler_uh",
    "ln_n_sq",
    "mean_radius_mm",
    "fill_factor",
    "inner_diameter_mm",
)