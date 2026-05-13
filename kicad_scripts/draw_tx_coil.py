"""
TX Coil drawing script for KiCad PCB Editor Scripting Console.

Paste into: Tools > Scripting Console

Coil parameters:
  OD = 50.3 mm  ->  outer edge radius = 25.15 mm
  Trace width W = 0.57 mm
  Trace spacing (gap) = 0.16 mm
  Pitch p = W + gap = 0.73 mm
  Turns total = 8  (4 per layer, 2 layers)
  Center on board = (100, 100) mm

Layer 1 (F.Cu)  : 4 turns, arcs 1-8,  spiral goes inward CW when viewed from top
Layer 2 (B.Cu)  : 4 turns, arcs 9-16, continues inward, mirrored in X so it winds
                  in the same physical direction as seen from the top.

Arc construction (Archimedean semicircles):
  - Each half-turn is one 180-degree arc.
  - Odd arcs  = top semicircle (start on left,  end on right, CCW)
  - Even arcs = bottom semicircle (start on right, end on left,  CW)
  - Each successive arc center shifts +p/2 in X (toward board center).
  - Each successive arc radius shrinks by p/2.
  - The endpoint of arc N is exactly the start point of arc N+1.

In KiCad pcbnew, PCB_ARC is defined by:
  SetStart()  - one endpoint
  SetEnd()    - other endpoint
  SetMid()    - a point on the arc (midpoint of the arc sweep)
  SetWidth()  - trace width (in nm)
  SetLayer()  - copper layer
"""

import pcbnew
import math

board = pcbnew.GetBoard()

# ── Parameters ──────────────────────────────────────────────────────────────
CENTER_X_MM   = 100.0
CENTER_Y_MM   = 100.0
OD_MM         = 50.3          # outer edge diameter
W_MM          = 0.57          # trace width
GAP_MM        = 0.16          # edge-to-edge spacing between adjacent traces
TURNS_PER_LAYER = 8
NUM_LAYERS    = 2

PITCH         = W_MM + GAP_MM               # 0.73 mm
HALF_PITCH    = PITCH / 2.0                  # 0.365 mm

# Outer trace centerline radius (first arc)
R0 = (OD_MM / 2.0) - (W_MM / 2.0)          # 25.15 - 0.285 = 24.865 mm

def mm(val):
    """Convert mm to KiCad internal units (nm)."""
    return pcbnew.FromMM(val)

def add_arc(board, cx, cy, start_x, start_y, end_x, end_y, mid_x, mid_y,
            width_mm, layer):
    """Add a PCB_ARC trace defined by start, end, and midpoint."""
    arc = pcbnew.PCB_ARC(board)
    arc.SetStart(pcbnew.VECTOR2I(mm(start_x), mm(start_y)))
    arc.SetEnd(pcbnew.VECTOR2I(mm(end_x), mm(end_y)))
    arc.SetMid(pcbnew.VECTOR2I(mm(mid_x), mm(mid_y)))
    arc.SetWidth(mm(width_mm))
    arc.SetLayer(layer)
    board.Add(arc)
    return arc

def draw_spiral_layer(board, cx, cy, r0, pitch, turns, layer, mirror=False):
    """
    Draw an Archimedean spiral as 2*turns semicircular arcs.

    mirror=False: arc-1 sweeps the TOP semicircle, arc-2 the BOTTOM, etc.
                  Spiral pitch is produced by alternating the bottom-arc
                  centers to +half_pitch in X.
    mirror=True : the spiral is reflected across the X-axis through cy.
                  Arc-1 now sweeps the BOTTOM, arc-2 the TOP, and the
                  pitch is produced by alternating the top-arc centers
                  to +half_pitch in X. The winding direction (rotational
                  sense, viewed from +Z) is reversed compared to mirror=False,
                  while every endpoint stays on the line Y = cy -- so the
                  inner termination of both layers lies at exactly the
                  same (x, y), perfect for a single via.
    """
    arcs_drawn = []
    half_pitch = pitch / 2.0

    for i in range(2 * turns):
        r = r0 - i * half_pitch

        # Determine which semicircle (top vs bottom) for this arc.
        # Without mirror: even i -> top, odd i -> bottom.
        # With    mirror: even i -> bottom, odd i -> top  (swap top/bottom).
        if not mirror:
            is_top = (i % 2 == 0)
        else:
            is_top = (i % 2 == 1)

        # Determine center X offset.
        # The arc whose center is offset by +half_pitch is the one that
        # "moves the spiral inward" by half a pitch. Without mirror these
        # are the bottom arcs (odd i). With mirror they are the top arcs
        # (also odd i in our re-mapped sense) -- i.e. odd i in both cases.
        cx_arc = cx + (half_pitch if (i % 2 == 1) else 0.0)
        cy_arc = cy

        if is_top:
            # TOP semicircle: left endpoint -> top -> right endpoint
            sx, sy = cx_arc - r, cy_arc
            ex, ey = cx_arc + r, cy_arc
            mx, my = cx_arc, cy_arc - r       # top of arc (-Y in KiCad screen)
        else:
            # BOTTOM semicircle: right endpoint -> bottom -> left endpoint
            sx, sy = cx_arc + r, cy_arc
            ex, ey = cx_arc - r, cy_arc
            mx, my = cx_arc, cy_arc + r       # bottom of arc (+Y in KiCad screen)

        # For mirror=True the start/end direction must also swap so that the
        # spiral is traversed in the reversed rotational sense.
        if mirror:
            sx, ex = ex, sx
            sy, ey = ey, sy

        arc = add_arc(board, cx_arc, cy_arc, sx, sy, ex, ey, mx, my, W_MM, layer)
        arcs_drawn.append({
            'arc_num': i + 1,
            'center': (cx_arc, cy_arc),
            'radius': r,
            'start': (sx, sy),
            'end':   (ex, ey),
        })

    return arcs_drawn

# ── Draw Layer 1: F.Cu ───────────────────────────────────────────────────────
print("Drawing TX Layer 1 (F.Cu)...")
layer1_arcs = draw_spiral_layer(
    board, CENTER_X_MM, CENTER_Y_MM,
    R0, PITCH, TURNS_PER_LAYER,
    layer=pcbnew.F_Cu,
    mirror=False
)

# ── Draw Layer 2: B.Cu ───────────────────────────────────────────────────────
# Disabled for now -- verify Layer 1 geometry first, then enable.
DRAW_LAYER_2 = True
layer2_arcs = []
if DRAW_LAYER_2:
    print("Drawing TX Layer 2 (B.Cu)...")
    layer2_arcs = draw_spiral_layer(
        board, CENTER_X_MM, CENTER_Y_MM,
        R0, PITCH, TURNS_PER_LAYER,
        layer=pcbnew.B_Cu,
        mirror=True
    )

# Innermost radius (same on both layers)
r_inner = R0 - (2 * TURNS_PER_LAYER - 1) * HALF_PITCH

# ── Refresh board view ───────────────────────────────────────────────────────
pcbnew.Refresh()

# ── Print summary ────────────────────────────────────────────────────────────
print("\n=== Layer 1 (F.Cu) arcs ===")
for a in layer1_arcs:
    print(f"  Arc {a['arc_num']:2d}  center=({a['center'][0]:.3f}, {a['center'][1]:.3f})"
          f"  r={a['radius']:.3f}  start={a['start']}  end={a['end']}")

print("\n=== Layer 2 (B.Cu) arcs ===")
for a in layer2_arcs:
    print(f"  Arc {a['arc_num']:2d}  center=({a['center'][0]:.3f}, {a['center'][1]:.3f})"
          f"  r={a['radius']:.3f}  start={a['start']}  end={a['end']}")

print(f"\nDone. Outer radius (both layers): {R0:.3f} mm")
print(f"      Inner radius (both layers): {r_inner:.3f} mm")
print("\nConnect the two layers with a single via at the inner end of the spiral,")
print("and route the two outer terminals to your pads.")
