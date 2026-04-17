#!/usr/bin/env python3
"""
Merge two single-layer coil node lists into one combined FastHenry .inp
that represents two layers connected in series by a single vertical via.

The "via" is the node shared between the two DXF drawings. In the input
DXFs, both layers have a trace endpoint at (approximately) the same x,y.
This module:
  1. Finds the four endpoints (N0 and Nlast of each layer).
  2. Picks the pair with the smallest distance as the via location.
  3. Reorients each layer's node list so the via is the LAST node of
     layer 1 and the FIRST node of layer 2.
  4. Writes a combined .inp where:
       - Layer 1 nodes use z = 0.
       - The via segment is a single vertical E-element.
       - Layer 2 nodes use z = layer_spacing, with x,y pulled to match
         layer 1's via point exactly (snap).
       - .external runs from N0 (layer 1 start) to N<last> (layer 2 end).
"""

from math import sqrt


DEFAULT_VIA_MATCH_TOL_MM = 0.5   # loose match - DXFs are rarely perfect


# -------------------------------------------------------------------------
# Endpoint geometry
# -------------------------------------------------------------------------

def _dist(a, b):
    """Euclidean distance between two (x,y,z) tuples."""
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _dist_xy(a, b):
    """Planar distance, ignoring z. Useful for via matching across layers."""
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def find_via_match(nodes_a, nodes_b, tol=DEFAULT_VIA_MATCH_TOL_MM):
    """
    Pick which endpoint of A coincides with which endpoint of B.

    Both layers are single open traces, so each has exactly two endpoints:
    index 0 and index -1 in the node list. We test all four pairings and
    return the best one if it's within tolerance.

    Returns:
        (end_a, end_b, distance) where end_a/end_b are each either "start"
        or "end", indicating which endpoint of that layer is the via.
        Returns None if no pairing is within tol.
    """
    endpoints_a = {"start": nodes_a[0], "end": nodes_a[-1]}
    endpoints_b = {"start": nodes_b[0], "end": nodes_b[-1]}

    best = None
    for name_a, pt_a in endpoints_a.items():
        for name_b, pt_b in endpoints_b.items():
            d = _dist_xy(pt_a, pt_b)
            if best is None or d < best[2]:
                best = (name_a, name_b, d)

    if best is None or best[2] > tol:
        return None
    return best


# -------------------------------------------------------------------------
# Reorientation
# -------------------------------------------------------------------------


def orient_layers(nodes_a, nodes_b, match):
    """
    Given a via match from find_via_match, return reoriented copies of both
    node lists such that:
        - nodes_a's via endpoint is at index -1
        - nodes_b's via endpoint is at index 0

    If either layer's via is at the "wrong" end, we reverse that list.
    """
    end_a, end_b, _ = match

    a_out = list(nodes_a) if end_a == "end" else list(reversed(nodes_a))
    b_out = list(nodes_b) if end_b == "start" else list(reversed(nodes_b))

    return a_out, b_out


# -------------------------------------------------------------------------
# Combined .inp emission
# -------------------------------------------------------------------------

def write_combined_inp(nodes_layer1, nodes_layer2, out_path,
                       layer_spacing_mm,
                       w=0.52, h=0.035, sigma=5.8e4,
                       w2=None, h2=None,
                       nhinc=1, nwinc=3,
                       fmin=1.35e5, fmax=1.50e5, freq_ndec=1,
                       via_w=None, via_h=None,
                       header_comment="Combined 2-layer coil"):
    """
    Write a FastHenry .inp containing both layers connected by a via.

    nodes_layer1 must be oriented so its last node is the via.
    nodes_layer2 must be oriented so its first node is the via.

    Layer 1 is written at z = 0.
    Layer 2 is written at z = layer_spacing_mm, with its first node's
    x,y snapped to match layer 1's last node exactly.

    Node numbering (single monotonic sequence across both layers):
        N0 .. N(n1-1)       -> layer 1
        N(n1) .. N(n1+n2-1) -> layer 2
    The via edge is a single E-element connecting N(n1-1) to N(n1).

    via_w / via_h default to the trace w/h if not given. A PCB via is
    roughly a small cylinder of copper; modeling it as a single rect
    segment with the same cross-section as the trace is a reasonable
    first-order approximation and avoids a separate via-geometry input.

    Returns:
        A dict with indices useful to the GUI for colored highlighting:
          {"start_idx": 0,
           "via_idx": n1 - 1,          # layer 1's via node
           "via_idx_layer2": n1,       # layer 2's via node (same xy, z shifted)
           "end_idx":  n1 + n2 - 1}
    """
    if layer_spacing_mm <= 0:
        raise ValueError("layer_spacing_mm must be > 0 when merging two layers")

    if via_w is None:
        via_w = w
    if via_h is None:
        via_h = h

    n1 = len(nodes_layer1)
    n2 = len(nodes_layer2)

    # Snap layer 2's via node (index 0) onto layer 1's via node (index -1)
    # in x,y. This kills any residual <tol mismatch in the DXFs so the
    # via segment is purely vertical.
    via_xy = (nodes_layer1[-1][0], nodes_layer1[-1][1])

    # Build layer 2 with z-offset applied and the first point snapped.
    nodes_layer2_snapped = []
    for i, (x, y, z) in enumerate(nodes_layer2):
        if i == 0:
            nodes_layer2_snapped.append(
                (via_xy[0], via_xy[1], z + layer_spacing_mm))
        else:
            nodes_layer2_snapped.append((x, y, z + layer_spacing_mm))

    with open(out_path, "w", newline="\n") as f:
        f.write(f"* {header_comment}\n")
        f.write(f"* Two-layer series coil, via at (x,y) = "
                f"({via_xy[0]:.4f}, {via_xy[1]:.4f})\n")
        f.write(f"* Layer spacing: {layer_spacing_mm} mm\n")
        f.write(".Units mm\n\n")
        f.write(f".Default sigma={sigma} w={w} h={h}"
                f" nhinc={nhinc} nwinc={nwinc}\n\n")

        # Nodes: layer 1, then layer 2.
        for i, (x, y, z) in enumerate(nodes_layer1):
            f.write(f"N{i} x={x:.6g} y={y:.6g} z={z:.6g}\n")
        for j, (x, y, z) in enumerate(nodes_layer2_snapped):
            f.write(f"N{n1 + j} x={x:.6g} y={y:.6g} z={z:.6g}\n")
        f.write("\n")

# Edges within layer 1 (use layer-1 defaults).
        for i in range(n1 - 1):
            f.write(f"E{i} N{i} N{i + 1}\n")

        # If layer 2 has different trace dimensions, drop a second
        # .Default so subsequent E-lines pick up the new w/h.
        _w2 = w if w2 is None else w2
        _h2 = h if h2 is None else h2
        if _w2 != w or _h2 != h:
            f.write(f".Default sigma={sigma} w={_w2} h={_h2}"
                    f" nhinc={nhinc} nwinc={nwinc}\n")

        # Via edge: explicit w,h override regardless (via geometry differs).
        via_edge_idx = n1 - 1
        f.write(f"E{via_edge_idx} N{n1 - 1} N{n1}"
                f" w={via_w} h={via_h}\n")

        # Edges within layer 2 (use layer-2 defaults if we emitted them).
        for j in range(n2 - 1):
            edge_idx = n1 + j
            f.write(f"E{edge_idx} N{n1 + j} N{n1 + j + 1}\n")
        f.write("\n")

        # Single port spanning the whole structure.
        last_idx = n1 + n2 - 1
        f.write(f".external N0 N{last_idx}\n\n")
        f.write(f".freq fmin={fmin} fmax={fmax} ndec={freq_ndec}\n\n")
        f.write(".end\n")

    return {
        "start_idx": 0,
        "via_idx": n1 - 1,
        "via_idx_layer2": n1,
        "end_idx": last_idx,
        "layer1_count": n1,
        "layer2_count": n2,
    }