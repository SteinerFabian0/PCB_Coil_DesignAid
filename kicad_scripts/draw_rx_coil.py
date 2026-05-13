"""
RX Coil drawing script for KiCad PCB Editor Scripting Console.

Paste into: Tools > Scripting Console
    exec(open(r"C:\\Users\\enots\\Desktop\\PCB_Coil_DesignAid\\kicad_scripts\\draw_rx_coil.py").read())

Topology:
  - 4 copper layers (F.Cu = L1, In1.Cu = L2, In2.Cu = L3, B.Cu = L4)
  - All four spirals share the SAME outer and SAME inner centerline radii,
    so every layer's outer endpoint sits at the same (x, y) and every layer's
    inner endpoint sits at the same (x, y). You stack a via at each endpoint
    to connect all four layers in series.
  - Outer layers (L1, L4): 12 turns, trace width 0.23 mm (the user-specified W)
  - Inner layers (L2, L3): 10 turns, trace width auto-computed so that the
    edge-to-edge gap stays 0.16 mm. With 10 turns instead of 12 across the
    SAME radial span, the traces are wider, and they stick out slightly past
    the outer-layer edges -- this is intentional.
  - All layers wind CCW when viewed from the +Z side. The user's TX script
    was already produced with "mirror=False" -> CW from above (because they
    flipped the board); for RX we use mirror=True throughout to get CCW.

The endpoint coordinates are fixed by the OUTER-layer geometry. The inner
layers are forced to share those same endpoints, and their pitch / trace
width fall out from that constraint.
"""

import pcbnew

board = pcbnew.GetBoard()

# ── Parameters ──────────────────────────────────────────────────────────────
CENTER_X_MM     = 100.0
CENTER_Y_MM     = 100.0
OD_MM           = 50.7        # outer EDGE diameter
W_OUTER_MM      = 0.23        # trace width on L1, L4 (the 12-turn layers)
GAP_MM          = 0.16        # edge-to-edge spacing between adjacent traces
TURNS_OUTER     = 12          # turns on L1 and L4
TURNS_INNER     = 10          # turns on L2 and L3

# Derived geometry for OUTER layers (L1, L4)
PITCH_OUTER     = W_OUTER_MM + GAP_MM           # 0.39 mm
HALF_PITCH_OUT  = PITCH_OUTER / 2.0              # 0.195 mm
R_OUTER_START   = (OD_MM / 2.0) - (W_OUTER_MM / 2.0)  # 25.235 mm
R_OUTER_END     = R_OUTER_START - (2 * TURNS_OUTER - 1) * HALF_PITCH_OUT
# R_OUTER_END = 20.750 mm

# Derived geometry for INNER layers (L2, L3): must span the same R_OUTER_START
# to R_OUTER_END with 10 turns. Solve for pitch and trace width.
HALF_PITCH_IN   = (R_OUTER_START - R_OUTER_END) / (2 * TURNS_INNER - 1)
PITCH_INNER     = 2.0 * HALF_PITCH_IN
W_INNER_MM      = PITCH_INNER - GAP_MM
# Expected: half_pitch_in = 4.485 / 19 = 0.23605 mm
#           pitch_inner   = 0.4721  mm
#           w_inner       = 0.3121  mm

def mm(val):
    return pcbnew.FromMM(val)

def add_arc(board, start_x, start_y, end_x, end_y, mid_x, mid_y, width_mm, layer):
    arc = pcbnew.PCB_ARC(board)
    arc.SetStart(pcbnew.VECTOR2I(mm(start_x), mm(start_y)))
    arc.SetEnd(pcbnew.VECTOR2I(mm(end_x), mm(end_y)))
    arc.SetMid(pcbnew.VECTOR2I(mm(mid_x), mm(mid_y)))
    arc.SetWidth(mm(width_mm))
    arc.SetLayer(layer)
    board.Add(arc)
    return arc

def draw_spiral_layer(board, cx, cy, r0, pitch, w_trace, turns, layer, mirror=True):
    """
    Draw an Archimedean spiral as 2*turns semicircular arcs.
    See draw_tx_coil.py for the geometry derivation.
    For RX we always use mirror=True (CCW when viewed from +Z).
    """
    arcs_drawn = []
    half_pitch = pitch / 2.0

    for i in range(2 * turns):
        r = r0 - i * half_pitch

        if not mirror:
            is_top = (i % 2 == 0)
        else:
            is_top = (i % 2 == 1)

        cx_arc = cx + (half_pitch if (i % 2 == 1) else 0.0)
        cy_arc = cy

        if is_top:
            sx, sy = cx_arc - r, cy_arc
            ex, ey = cx_arc + r, cy_arc
            mx, my = cx_arc, cy_arc - r
        else:
            sx, sy = cx_arc + r, cy_arc
            ex, ey = cx_arc - r, cy_arc
            mx, my = cx_arc, cy_arc + r

        if mirror:
            sx, ex = ex, sx
            sy, ey = ey, sy

        arc = add_arc(board, sx, sy, ex, ey, mx, my, w_trace, layer)
        arcs_drawn.append({
            'arc_num': i + 1,
            'center': (cx_arc, cy_arc),
            'radius': r,
            'start': (sx, sy),
            'end':   (ex, ey),
        })

    return arcs_drawn

# ── Print plan ──────────────────────────────────────────────────────────────
print("=== RX Coil Plan ===")
print(f"  Center: ({CENTER_X_MM}, {CENTER_Y_MM}) mm")
print(f"  Outer edge diameter: {OD_MM} mm")
print(f"  Outer layers (L1, L4): {TURNS_OUTER} turns,  W = {W_OUTER_MM:.4f} mm,  pitch = {PITCH_OUTER:.4f} mm")
print(f"  Inner layers (L2, L3): {TURNS_INNER} turns,  W = {W_INNER_MM:.4f} mm,  pitch = {PITCH_INNER:.4f} mm")
print(f"  Common outer centerline radius: {R_OUTER_START:.4f} mm")
print(f"  Common inner centerline radius: {R_OUTER_END:.4f} mm")
print(f"  Common outer endpoint: ({CENTER_X_MM - R_OUTER_START:.4f}, {CENTER_Y_MM:.4f})")

# Inner endpoint depends on number of arcs (parity). For both 12 turns and
# 10 turns we have an EVEN turn count, so the last arc is an odd-index (bottom
# in non-mirrored, top in mirrored) -- center offset by +half_pitch, ends on
# the left side of the coil.
inner_endpoint_x_outer = CENTER_X_MM + HALF_PITCH_OUT - R_OUTER_END
inner_endpoint_x_inner = CENTER_X_MM + HALF_PITCH_IN  - R_OUTER_END
print(f"  Inner endpoint (outer layers L1,L4): ({inner_endpoint_x_outer:.4f}, {CENTER_Y_MM:.4f})")
print(f"  Inner endpoint (inner layers L2,L3): ({inner_endpoint_x_inner:.4f}, {CENTER_Y_MM:.4f})")
print(f"  Inner endpoint mismatch: {abs(inner_endpoint_x_outer - inner_endpoint_x_inner)*1000:.2f} um")
print()

# NOTE: The inner endpoint x is slightly different between outer and inner
# layers because the LAST arc's center offset is half_pitch, which differs
# between layers (0.195 mm vs 0.23605 mm). The mismatch is ~41 um. A via
# pad easily covers this, but be aware when placing the inner via.

# ── Draw all four layers ────────────────────────────────────────────────────
# KiCad layer constants:
#   F.Cu     = pcbnew.F_Cu   (top, layer 1)
#   In1.Cu   = pcbnew.In1_Cu (layer 2)
#   In2.Cu   = pcbnew.In2_Cu (layer 3)
#   B.Cu     = pcbnew.B_Cu   (bottom, layer 4)

print("Drawing RX Layer 1 (F.Cu, 12 turns)...")
l1_arcs = draw_spiral_layer(
    board, CENTER_X_MM, CENTER_Y_MM,
    R_OUTER_START, PITCH_OUTER, W_OUTER_MM, TURNS_OUTER,
    layer=pcbnew.F_Cu, mirror=True
)

print("Drawing RX Layer 2 (In1.Cu, 10 turns)...")
l2_arcs = draw_spiral_layer(
    board, CENTER_X_MM, CENTER_Y_MM,
    R_OUTER_START, PITCH_INNER, W_INNER_MM, TURNS_INNER,
    layer=pcbnew.In1_Cu, mirror=True
)

print("Drawing RX Layer 3 (In2.Cu, 10 turns)...")
l3_arcs = draw_spiral_layer(
    board, CENTER_X_MM, CENTER_Y_MM,
    R_OUTER_START, PITCH_INNER, W_INNER_MM, TURNS_INNER,
    layer=pcbnew.In2_Cu, mirror=True
)

print("Drawing RX Layer 4 (B.Cu, 12 turns)...")
l4_arcs = draw_spiral_layer(
    board, CENTER_X_MM, CENTER_Y_MM,
    R_OUTER_START, PITCH_OUTER, W_OUTER_MM, TURNS_OUTER,
    layer=pcbnew.B_Cu, mirror=True
)

pcbnew.Refresh()

# ── Summary ─────────────────────────────────────────────────────────────────
def report(name, arcs):
    print(f"\n=== {name} ({len(arcs)} arcs) ===")
    for a in (arcs[0], arcs[-1]):
        print(f"  Arc {a['arc_num']:2d}  center=({a['center'][0]:.4f}, {a['center'][1]:.4f})"
              f"  r={a['radius']:.4f}  start={a['start']}  end={a['end']}")

report("Layer 1 F.Cu",   l1_arcs)
report("Layer 2 In1.Cu", l2_arcs)
report("Layer 3 In2.Cu", l3_arcs)
report("Layer 4 B.Cu",   l4_arcs)

print("\nDone.")
print("Place one via at the OUTER endpoint connecting all 4 layers.")
print("Place one via at the INNER endpoint connecting all 4 layers.")
print("Route your RX+ / RX- pads to the two via positions (one outer, one inner)")
print("  -- you'll need to break out one of them with a short trace on a free layer.")
