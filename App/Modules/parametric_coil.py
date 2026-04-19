#!/usr/bin/env python3
"""
Parametric planar spiral coil generator.

Pure logic only — no Tk, no matplotlib. Consumed by:
  - parametric_tab.py (live preview + export)
  - automation/sweep engines (future)

Spiral geometry
---------------
Built from 2*N semicircular arcs (N = turn count). Upper halves use a
center at (0, 0); lower halves use a center at (-pitch/2, 0). Radius
drops by pitch/2 every half-turn. Pitch = trace_width + spacing.

Position-continuous at every half-turn junction; the radius of
curvature is discontinuous there, but that's fine for FastHenry and
matches how planar spirals are drawn in EDA tools.

    Start point: ( R0,          0)   where R0 = OD/2 - W/2
    End   point: ( R0 - N*p,    0)

Both endpoints sit on the +x axis. Terminal lead-outs are NOT added —
the user asked that they be left where they naturally land.

Stack-up
--------
Up to 4 layer slots at z-coordinates determined by two gap params:
    outer_gap: gap between slots 1&2 and between slots 3&4
    inner_gap: gap between slots 2&3

    z[1] = 0
    z[2] = outer_gap
    z[3] = outer_gap + inner_gap
    z[4] = 2*outer_gap + inner_gap

Parallel wiring (multi-layer)
-----------------------------
All active layers share the same x,y spiral and differ only in z and
copper thickness. write_parallel_multilayer_inp() emits a FastHenry
.inp where active slots are wired electrically in parallel through two
physical via ladders (at spiral start_xy and end_xy). The single
.external port spans the bottom slot's start/end nodes.
"""

from dataclasses import dataclass, field
import math


OZ_TO_MM = 0.035                # 1 oz Cu = 35 um
COPPER_SIGMA_PER_MM = 5.8e4     # S/mm for FastHenry with .Units mm


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass
class SpiralParams:
    od_mm: float                # outer diameter (trace outer edge)
    trace_width_mm: float       # W
    spacing_mm: float           # S (gap between adjacent trace edges)
    turns: float                # N; 2*N half-arcs emitted
    resolution_mm: float        # max chord length when tessellating arcs

    @property
    def pitch_mm(self):
        return self.trace_width_mm + self.spacing_mm

    @property
    def r_outer_centerline(self):
        # Centerline of the outermost trace so its outer edge sits at OD/2.
        return self.od_mm / 2.0 - self.trace_width_mm / 2.0

    @property
    def r_inner_centerline(self):
        # Innermost half-turn radius = R0 - (2N-1)*pitch/2
        return self.r_outer_centerline - (2 * self.turns - 1) * self.pitch_mm / 2.0


@dataclass
class LayerSlot:
    """One of 4 stack-up slots. active=False means the slot is skipped."""
    active: bool = False
    copper_oz: float = 1.0

    @property
    def copper_h_mm(self):
        return self.copper_oz * OZ_TO_MM


@dataclass
class StackUp:
    """4-slot PCB stackup. slots[0] is slot 1 (z=0), slots[3] is slot 4."""
    slots: list = field(default_factory=list)   # always len 4
    outer_gap_mm: float = 0.5
    inner_gap_mm: float = 0.5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_spiral(params):
    """Returns (ok, message). Message is '' on success."""
    if params.od_mm <= 0:
        return False, "OD must be > 0."
    if params.trace_width_mm <= 0:
        return False, "Trace width must be > 0."
    if params.spacing_mm < 0:
        return False, "Trace spacing cannot be negative."
    if params.turns < 0.5:
        return False, "Turns must be >= 0.5."
    if params.resolution_mm <= 0:
        return False, "Resolution must be > 0."

    # Inner-radius sanity: we require inner centerline >= W/2 so the
    # innermost trace doesn't bite its own outer edge at the center.
    inner = params.r_inner_centerline
    if inner < params.trace_width_mm / 2.0:
        return False, (f"Spiral collapses at center: inner R = "
                       f"{inner:.3f} mm < W/2 = "
                       f"{params.trace_width_mm/2.0:.3f} mm. "
                       f"Reduce turns or increase OD.")
    return True, ""


def validate_stackup(stackup, min_active=1, max_active=4):
    n_active = sum(1 for s in stackup.slots if s.active)
    if n_active < min_active:
        return False, f"Need at least {min_active} active layer(s)."
    if n_active > max_active:
        return False, f"At most {max_active} active layers supported."
    if n_active >= 2 and stackup.outer_gap_mm <= 0:
        return False, "Outer gap must be > 0 with 2+ layers."
    if (stackup.slots[1].active and stackup.slots[2].active
            and stackup.inner_gap_mm <= 0):
        return False, "Inner gap must be > 0 when slots 2 and 3 are both active."
    return True, ""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def generate_nodes(params, z=0.0):
    """
    Ordered list of (x, y, z) tracing the full spiral.

    Two-center construction:
      - even h: center (0, 0),        theta 0 -> pi     (upper half)
      - odd  h: center (-pitch/2, 0), theta pi -> 2*pi  (lower half)
      - radius for half-turn h: R0 - h*pitch/2

    Each half-turn shares its last point with the next half-turn's
    first point, so for h >= 1 we skip the starting sample to avoid
    a duplicated node.
    """
    R0 = params.r_outer_centerline
    p  = params.pitch_mm
    n_half_turns = int(round(params.turns * 2))

    points = []
    for h in range(n_half_turns):
        r = R0 - h * (p / 2.0)
        if r <= 0:
            break

        if h % 2 == 0:
            cx, cy = 0.0, 0.0
            theta_start, theta_end = 0.0, math.pi
        else:
            cx, cy = -p / 2.0, 0.0
            theta_start, theta_end = math.pi, 2.0 * math.pi

        arc_len = math.pi * r
        n_segs = max(2, int(math.ceil(arc_len / params.resolution_mm)))

        first_i = 1 if h > 0 else 0
        for i in range(first_i, n_segs + 1):
            t = theta_start + (theta_end - theta_start) * i / n_segs
            x = cx + r * math.cos(t)
            y = cy + r * math.sin(t)
            points.append((x, y, z))

    return points


def layer_z_positions(outer_gap_mm, inner_gap_mm):
    """Z coordinates for the four stack-up slots."""
    return [
        0.0,
        outer_gap_mm,
        outer_gap_mm + inner_gap_mm,
        2.0 * outer_gap_mm + inner_gap_mm,
    ]


def active_layer_data(spiral_params, stackup):
    """
    Per-active-slot geometry + copper, in ascending z order.

    Returns [{"slot": 1..4, "z": float, "h_mm": float,
              "nodes": [(x,y,z), ...]}, ...]
    """
    zs = layer_z_positions(stackup.outer_gap_mm, stackup.inner_gap_mm)
    out = []
    for idx, slot in enumerate(stackup.slots):
        if not slot.active:
            continue
        nodes = generate_nodes(spiral_params, z=zs[idx])
        out.append({
            "slot": idx + 1,
            "z": zs[idx],
            "h_mm": slot.copper_h_mm,
            "nodes": nodes,
        })
    return out


# ---------------------------------------------------------------------------
# .inp emission
# ---------------------------------------------------------------------------

def write_single_layer_inp(nodes, out_path,
                           w_mm, h_mm,
                           sigma=COPPER_SIGMA_PER_MM,
                           nhinc=1, nwinc=3,
                           fmin=1.35e5, fmax=1.50e5, freq_ndec=1,
                           header="Parametric spiral coil (single layer)"):
    """Mirrors dxf_to_inp_converter output style — rest of pipeline is oblivious."""
    n = len(nodes)
    with open(out_path, "w", newline="\n") as f:
        f.write(f"* {header}\n.Units mm\n\n")
        f.write(f".Default sigma={sigma} w={w_mm} h={h_mm}"
                f" nhinc={nhinc} nwinc={nwinc}\n\n")
        for i, (x, y, z) in enumerate(nodes):
            f.write(f"N{i} x={x:.6g} y={y:.6g} z={z:.6g}\n")
        f.write("\n")
        for i in range(n - 1):
            f.write(f"E{i} N{i} N{i+1}\n")
        f.write(f"\n.external N0 N{n-1}\n\n")
        f.write(f".freq fmin={fmin} fmax={fmax} ndec={freq_ndec}\n\n.end\n")


def write_parallel_multilayer_inp(layer_data, out_path,
                                  w_mm,
                                  sigma=COPPER_SIGMA_PER_MM,
                                  nhinc=1, nwinc=3,
                                  via_w_mm=None, via_h_mm=None,
                                  fmin=1.35e5, fmax=1.50e5, freq_ndec=1,
                                  header="Parametric parallel multilayer coil"):
    """
    Emit a .inp with all active layers wired electrically in parallel.

    Node indexing: concatenate slots' spirals, slot 0 first.
        slot 0 -> N0 .. N(n0-1)
        slot 1 -> N(n0) .. N(n0+n1-1)   ...
    Via ladders reuse the already-emitted slot-start / slot-end nodes;
    no new nodes are needed because slot k's start node is at
    (start_x, start_y, z_k) and slot (k+1)'s start is at the same x,y
    with z_(k+1), so a single vertical edge connects them.

    Current flows:
        N(slot0_start) -> start via ladder -> each slot's spiral ->
        end via ladder -> N(slot0_end)

    layer_data must be sorted by z ascending (active_layer_data does this).
    """
    if not layer_data:
        raise ValueError("layer_data is empty")

    if via_w_mm is None:
        via_w_mm = w_mm
    if via_h_mm is None:
        via_h_mm = layer_data[0]["h_mm"]

    offsets, cum = [], 0
    for ld in layer_data:
        offsets.append(cum)
        cum += len(ld["nodes"])

    with open(out_path, "w", newline="\n") as f:
        f.write(f"* {header}\n")
        f.write(f"* {len(layer_data)} layer(s) in parallel via real via segments\n")
        for ld in layer_data:
            f.write(f"* slot {ld['slot']}: z={ld['z']:.3f} mm, "
                    f"h={ld['h_mm']:.4f} mm "
                    f"({ld['h_mm']/OZ_TO_MM:.2f} oz)\n")
        f.write(".Units mm\n\n")

        # Initial .Default sets slot 0's copper thickness.
        f.write(f".Default sigma={sigma} w={w_mm} h={layer_data[0]['h_mm']}"
                f" nhinc={nhinc} nwinc={nwinc}\n\n")

        # --- Nodes ---
        for k, ld in enumerate(layer_data):
            for i, (x, y, z) in enumerate(ld["nodes"]):
                f.write(f"N{offsets[k] + i} "
                        f"x={x:.6g} y={y:.6g} z={z:.6g}\n")
        f.write("\n")

        # --- Spiral edges, re-emit .Default when copper thickness changes ---
        edge_counter = 0
        current_h = layer_data[0]["h_mm"]
        for k, ld in enumerate(layer_data):
            if ld["h_mm"] != current_h:
                current_h = ld["h_mm"]
                f.write(f".Default sigma={sigma} w={w_mm} h={current_h}"
                        f" nhinc={nhinc} nwinc={nwinc}\n")
            n_k = len(ld["nodes"])
            base = offsets[k]
            for i in range(n_k - 1):
                f.write(f"E{edge_counter} N{base + i} N{base + i + 1}\n")
                edge_counter += 1
        f.write("\n")

        # --- Start via ladder ---
        for k in range(len(layer_data) - 1):
            a = offsets[k]
            b = offsets[k + 1]
            f.write(f"E{edge_counter} N{a} N{b}"
                    f" w={via_w_mm} h={via_h_mm}\n")
            edge_counter += 1
        # --- End via ladder ---
        for k in range(len(layer_data) - 1):
            a = offsets[k]     + len(layer_data[k]["nodes"])     - 1
            b = offsets[k + 1] + len(layer_data[k + 1]["nodes"]) - 1
            f.write(f"E{edge_counter} N{a} N{b}"
                    f" w={via_w_mm} h={via_h_mm}\n")
            edge_counter += 1
        f.write("\n")

        # --- Port on bottom slot's endpoints ---
        start_idx = offsets[0]
        end_idx   = offsets[0] + len(layer_data[0]["nodes"]) - 1
        f.write(f".external N{start_idx} N{end_idx}\n\n")
        f.write(f".freq fmin={fmin} fmax={fmax} ndec={freq_ndec}\n\n.end\n")

    return {
        "total_nodes": cum,
        "external_start": start_idx,
        "external_end":   end_idx,
        "slot_offsets":   offsets,
    }