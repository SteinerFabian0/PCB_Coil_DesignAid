#!/usr/bin/env python3
"""
FastHenry .inp viewer.

Two modes:
  - Library: `build_figure(...)` returns a matplotlib Figure for embedding
    in the master GUI via FigureCanvasTkAgg.
  - CLI: `python inp_visualizer.py file.inp` opens a standalone window.
"""

import sys
import re

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Patterns shared by both entry points.
_NODE_RE = re.compile(
    r'^(N\d+)\s+'
    r'x=([-\d.eE+]+)\s+'
    r'y=([-\d.eE+]+)\s+'
    r'z=([-\d.eE+]+)',
    re.IGNORECASE,
)
_EDGE_RE = re.compile(r'^(E\d+)\s+(N\d+)\s+(N\d+)', re.IGNORECASE)


# -------------------------------------------------------------------------
# Parsing
# -------------------------------------------------------------------------

def parse_inp(filepath):
    """
    Extract nodes and edges from a FastHenry .inp.

    Returns:
        nodes: dict mapping "N0" -> (x, y, z)
        edges: list of (edge_name, from_node, to_node)

    Comments (*) and directives (.) are skipped.
    """
    nodes = {}
    edges = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*") or line.startswith("."):
                continue

            m = _NODE_RE.match(line)
            if m:
                nodes[m.group(1).upper()] = (
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                )
                continue

            m = _EDGE_RE.match(line)
            if m:
                edges.append((
                    m.group(1).upper(),
                    m.group(2).upper(),
                    m.group(3).upper(),
                ))

    return nodes, edges


def sorted_node_names(nodes):
    """Node dict -> names in numeric order (N0, N1, ..., N10, ...)."""
    return sorted(nodes.keys(), key=lambda n: int(n[1:]))


# -------------------------------------------------------------------------
# Rendering
# -------------------------------------------------------------------------

def _draw(ax, nodes, edges, highlight=None, title=None, view="xy"):
    """
    Shared drawing routine for both the standalone plot and the embedded
    Figure. Pulled out so we don't duplicate styling.

    `highlight` is an optional dict like:
        {"start": "N0", "via": "N123", "end": "N456"}
    Each value is a node name; "start" draws green, "via" blue, "end" red.
    Any key can be omitted.

    `view` is "xy" (top-down) or "iso" (pseudo-3D, used for the combined
    two-layer overlay to suggest layer separation). "iso" is cheap — we
    project (x, y, z) -> (x + 0.3*z, y + 0.3*z) just to offset the layers.
    """
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3)

    if view == "iso":
        project = lambda p: (p[0] + 0.3 * p[2], p[1] + 0.3 * p[2])
    else:
        project = lambda p: (p[0], p[1])

    # Edge lines: color layer 2 edges differently if we can infer which
    # layer a node is on by its z coordinate. Cheap heuristic: z != 0
    # means layer 2. Two-layer plots look a lot clearer this way.
    for _, nf, nt in edges:
        if nf not in nodes or nt not in nodes:
            continue
        p0 = nodes[nf]
        p1 = nodes[nt]
        x0, y0 = project(p0)
        x1, y1 = project(p1)
        # Color rule: if BOTH endpoints have z > 0, it's a layer-2 edge.
        # Otherwise use the default color (includes via edges, which
        # bridge z=0 to z>0).
        if p0[2] > 0 and p1[2] > 0:
            color = "#d06020"   # orange-red for layer 2
        elif p0[2] != p1[2]:
            color = "#9020c0"   # purple for via edges
        else:
            color = "#2080d0"   # default layer-1 blue-ish
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=0.7)

    # Highlight markers. Caller passes node names.
    if highlight:
        style = {
            "start": ("go", "Start"),
            "via":   ("bs", "Via"),
            "end":   ("rs", "End"),
        }
        for key, node_name in highlight.items():
            if key not in style or node_name not in nodes:
                continue
            marker, label = style[key]
            x, y = project(nodes[node_name])
            ax.plot(x, y, marker, markersize=10, label=label)
        ax.legend(loc="best")


def build_figure(filepath, highlight=None, title=None, view="xy",
                 figsize=(6, 6), dpi=100):
    """
    Build a matplotlib Figure for embedding in a Tk GUI.

    Caller should NOT call plt.show() — just hand this Figure to
    FigureCanvasTkAgg and pack the canvas.

    Returns: (Figure, num_nodes, num_edges)
    """
    nodes, edges = parse_inp(filepath)
    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    _draw(ax, nodes, edges,
          highlight=highlight,
          title=title or filepath,
          view=view)
    fig.tight_layout()
    return fig, len(nodes), len(edges)


def build_overlay_figure(filepath_combined, highlight=None,
                         figsize=(7, 7), dpi=100):
    """
    Build a Figure showing both layers of a combined .inp in one view,
    using pseudo-iso projection so the layers visibly separate.

    The combined .inp already has both layers; we just render it with
    view="iso".
    """
    return build_figure(
        filepath_combined,
        highlight=highlight,
        title="Combined (layer overlay)",
        view="iso",
        figsize=figsize,
        dpi=dpi,
    )


# -------------------------------------------------------------------------
# CLI (standalone viewer, kept for debugging)
# -------------------------------------------------------------------------

def _cli_plot(filepath):
    nodes, edges = parse_inp(filepath)
    if not nodes:
        print("No nodes found. Check the file format.")
        sys.exit(1)

    names = sorted_node_names(nodes)
    highlight = {"start": names[0], "end": names[-1]}

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    _draw(ax, nodes, edges, highlight=highlight, title=filepath)
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} file.inp")
        sys.exit(1)
    _cli_plot(sys.argv[1])


if __name__ == "__main__":
    main()