#!/usr/bin/env python3
"""
Simple GUI viewer for FastHenry .inp files.
Draws nodes and edge connections on a matplotlib canvas.

Usage: python inp_viewer.py coilV1.inp
"""

import sys
import re
import matplotlib.pyplot as plt


def parse_inp(filepath):
    """
    Parse a FastHenry .inp file and extract nodes and edges.

    Returns
    -------
    nodes : dict mapping node name -> (x, y, z)
    edges : list of (name, node_from, node_to)
    """
    nodes = {}
    edges = []

    # Patterns for node and edge lines:
    #   N0  x=1.234  y=5.678  z=0
    #   E0  N0  N1  w=1  h=1 ...
    node_re = re.compile(
        r'^(N\d+)\s+'
        r'x=([-\d.eE+]+)\s+'
        r'y=([-\d.eE+]+)\s+'
        r'z=([-\d.eE+]+)',
        re.IGNORECASE
    )
    edge_re = re.compile(
        r'^(E\d+)\s+(N\d+)\s+(N\d+)',
        re.IGNORECASE
    )

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*") or line.startswith("."):
                continue

            m = node_re.match(line)
            if m:
                name = m.group(1).upper()
                x = float(m.group(2))
                y = float(m.group(3))
                z = float(m.group(4))
                nodes[name] = (x, y, z)
                continue

            m = edge_re.match(line)
            if m:
                ename = m.group(1).upper()
                nfrom = m.group(2).upper()
                nto = m.group(3).upper()
                edges.append((ename, nfrom, nto))

    return nodes, edges


def plot_inp(nodes, edges, title="FastHenry .inp"):
    """Plot the nodes and edges on a 2D matplotlib figure (X-Y plane)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3)

    # Draw edges as lines
    for ename, nfrom, nto in edges:
        if nfrom in nodes and nto in nodes:
            x0, y0, _ = nodes[nfrom]
            x1, y1, _ = nodes[nto]
            ax.plot([x0, x1], [y0, y1], color="#2080d0", linewidth=0.5)

    # Draw start and end nodes as markers
    if nodes:
        all_names = sorted(nodes.keys(), key=lambda n: int(n[1:]))
        sx, sy, _ = nodes[all_names[0]]
        ex, ey, _ = nodes[all_names[-1]]
        ax.plot(sx, sy, "go", markersize=8, label="Start (N0)")
        ax.plot(ex, ey, "rs", markersize=8, label=f"End ({all_names[-1]})")
        ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} file.inp")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Parsing {filepath}...")
    nodes, edges = parse_inp(filepath)
    print(f"Found {len(nodes)} nodes and {len(edges)} edges.")

    if not nodes:
        print("No nodes found. Check the file format.")
        sys.exit(1)

    plot_inp(nodes, edges, title=filepath)


if __name__ == "__main__":
    main()