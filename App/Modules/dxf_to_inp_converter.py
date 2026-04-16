#!/usr/bin/env python3
"""
Convert a DXF file containing LINE and ARC entities into a FastHenry .inp file.

The script chains individual LINE and ARC segments into a continuous path
by matching endpoints, then tessellates arcs into straight segments.

Usage: python dxf_to_fasthenry.py input.dxf output.inp
"""

import sys
import math
import ezdxf


# Tolerance for considering two endpoints "the same point"
TOL = 1e-4


def pts_equal(a, b):
    """Check if two 2D/3D points are equal within tolerance."""
    return all(abs(ai - bi) < TOL for ai, bi in zip(a, b))


def tessellate_arc(center, radius, start_angle_deg, end_angle_deg, z=0.0, seg_len=0.4):
    """
    Break a circular arc into straight line segments.

    Angles are in degrees, CCW from the positive X axis (DXF convention).
    seg_len controls the maximum chord length of each small segment.
    Returns a list of (x, y, z) points INCLUDING the start and end points.
    """
    start_rad = math.radians(start_angle_deg)
    end_rad = math.radians(end_angle_deg)

    # DXF arcs always go CCW; if end < start, the arc wraps past 360
    if end_rad <= start_rad:
        end_rad += 2.0 * math.pi

    total_angle = end_rad - start_rad
    arc_length = radius * total_angle

    n_segs = max(2, math.ceil(arc_length / seg_len))

    points = []
    for i in range(n_segs + 1):
        angle = start_rad + total_angle * i / n_segs
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y, z))

    return points


def extract_segments(dxf_path):
    """
    Read all LINE and ARC entities from the DXF and return them as
    a list of (start_point, end_point, interior_points) tuples.

    For LINEs, interior_points is empty.
    For ARCs, interior_points holds the tessellated points between start and end.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    segments = []

    for entity in msp.query("LINE"):
        s = (entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z)
        e = (entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z)
        segments.append((s, e, []))

    for entity in msp.query("ARC"):
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        cz = entity.dxf.center.z
        r = entity.dxf.radius
        sa = entity.dxf.start_angle
        ea = entity.dxf.end_angle

        arc_pts = tessellate_arc((cx, cy), r, sa, ea, z=cz)

        # start and end are the first and last tessellated points
        start = arc_pts[0]
        end = arc_pts[-1]
        interior = arc_pts[1:-1]

        segments.append((start, end, interior))

    if not segments:
        raise RuntimeError("No LINE or ARC entities found in the DXF file.")

    return segments


def find_nearest_match(current_end, segments, remaining, tol):
    """
    Find the segment in remaining whose start or end is closest to current_end.
    Returns (index_in_remaining, forward_bool, distance) or None if nothing within tol.
    """
    best = None
    for i, idx in enumerate(remaining):
        s, e, _ = segments[idx]
        ds = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_end, s)))
        de = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_end, e)))

        if ds <= de and ds < tol:
            if best is None or ds < best[2]:
                best = (i, True, ds)
        elif de < tol:
            if best is None or de < best[2]:
                best = (i, False, de)

    return best


def chain_from(segments, start_idx, tol_tight, tol_loose):
    """
    Attempt to chain all segments starting from start_idx.
    Returns (path, n_unchained).
    """
    remaining = list(range(len(segments)))
    remaining.remove(start_idx)

    s, e, interior = segments[start_idx]
    path = [s] + interior + [e]

    while remaining:
        current_end = path[-1]

        match = find_nearest_match(current_end, segments, remaining, tol_tight)
        if match is None:
            match = find_nearest_match(current_end, segments, remaining, tol_loose)

        if match is None:
            break

        i_in_remaining, forward, dist = match
        idx = remaining.pop(i_in_remaining)
        s, e, interior = segments[idx]

        if forward:
            path.extend(interior + [e])
        else:
            path.extend(list(reversed(interior)) + [s])

    return path, len(remaining)


def chain_segments(segments):
    """
    Chain segments into a continuous path. Tries every segment as a
    potential starting point and picks the chain that captures the most
    segments. This brute-force approach handles cases where the DXF
    entity order is unhelpful.
    """
    tol_tight = TOL
    tol_loose = 0.5

    best_path = []
    best_unchained = len(segments)
    best_start = 0

    for trial_idx in range(len(segments)):
        path, n_unchained = chain_from(segments, trial_idx, tol_tight, tol_loose)
        if n_unchained < best_unchained:
            best_unchained = n_unchained
            best_path = path
            best_start = trial_idx
            if n_unchained == 0:
                break  # perfect chain, stop early

    s, e, _ = segments[best_start]
    print(f"Best chain starts from segment ({s[0]:.4f},{s[1]:.4f}) -> ({e[0]:.4f},{e[1]:.4f})")
    print(f"  Unchained segments: {best_unchained}")

    if best_unchained > 0:
        # Show what's left
        _, remaining_segs = chain_from(segments, best_start, tol_tight, tol_loose)
        print(f"  WARNING: {best_unchained} segments could not be chained.")

    return best_path


def write_fasthenry_inp(points, out_path, w=0.52, h=0.035,
                        nhinc=1, nwinc=7, freq_min=1e5, freq_max=1.4e5, freq_ndec=1):
    """
    Write a FastHenry .inp file connecting the given points as a single conductor.

    Parameters
    ----------
    points : list of (x, y, z) tuples
    out_path : output file path
    w : trace width in mm
    h : trace thickness (copper height) in mm (0.035 = 1oz copper)
    nhinc, nwinc : filament count in height and width directions
    freq_min, freq_max : sweep range in Hz
    freq_ndec : frequency points per decade
    """
    # sigma for copper in 1/(mm*Ohms) since .Units is mm
    sigma = 5.8e4

    n_pts = len(points)

    with open(out_path, "w", newline="\n") as f:
        f.write("* PCB Coil Simulation - generated from DXF\n")
        f.write(".Units mm\n\n")

        # Defaults for all segments
        f.write(f".Default sigma={sigma} w={w} h={h}"
                f" nhinc={nhinc} nwinc={nwinc}\n\n")

        # --- Nodes ---
        for i, (x, y, z) in enumerate(points):
            f.write(f"N{i} x={x:.6g} y={y:.6g} z={z:.6g}\n")

        f.write("\n")

        # --- Segments (w and h inherited from .Default) ---
        for i in range(n_pts - 1):
            f.write(f"E{i} N{i} N{i+1}\n")

        f.write("\n")

        # --- Port definition ---
        f.write(f".external N0 N{n_pts - 1}\n\n")

        # --- Frequency sweep ---
        f.write(f".freq fmin={freq_min} fmax={freq_max} ndec={freq_ndec}\n\n")

        f.write(".end\n")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.dxf output.inp")
        sys.exit(1)

    dxf_path = sys.argv[1]
    out_path = sys.argv[2]

    print("Reading DXF entities...")
    segments = extract_segments(dxf_path)
    print(f"Found {len(segments)} segments (LINE + ARC).")

    print("Chaining segments into continuous path...")
    path = chain_segments(segments)
    print(f"Path has {len(path)} points.")

    write_fasthenry_inp(path, out_path)
    print(f"FastHenry file written to: {out_path}")


if __name__ == "__main__":
    main()