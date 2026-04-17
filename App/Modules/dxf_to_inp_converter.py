#!/usr/bin/env python3
"""
DXF -> FastHenry .inp converter.

Reads LINE and ARC entities from a DXF, chains them into a continuous path
by endpoint matching, tessellates arcs into straight chord segments, and
emits a FastHenry .inp file.

Exposes both a CLI entry point and a library function `convert_dxf_to_inp`
that returns the node list so downstream tools (merger, analyzer) can
reuse it without re-parsing the .inp.

DXF files are ASSUMED to be in millimeters. The caller is responsible
for validating this upstream.
"""

import sys
import math
import ezdxf


# Tolerance for treating two endpoints as the same point, in mm.
# DXF exports from most EDA tools are clean enough for 1e-4 to work,
# but the chain_segments loose fallback tolerates up to 0.5 mm gaps.
TOL = 1e-4

# Default fabrication / simulation parameters. These are safe values for
# 1 oz copper on standard FR-4; the caller should override them.
DEFAULT_W_MM = 0.52
DEFAULT_H_MM = 0.035            # 1 oz copper = 35 um
DEFAULT_SIGMA_PER_MM = 5.8e4    # copper conductivity in 1/(mm*Ohm) for .Units mm
DEFAULT_NHINC = 1
DEFAULT_NWINC = 3
DEFAULT_FMIN_HZ = 1.35e5
DEFAULT_FMAX_HZ = 1.50e5
DEFAULT_FREQ_NDEC = 1
DEFAULT_ARC_SEG_LEN_MM = 0.6   # max chord length when tessellating an arc


# -------------------------------------------------------------------------
# Geometry primitives
# -------------------------------------------------------------------------

def tessellate_arc(center, radius, start_angle_deg, end_angle_deg,
                   z=0.0, seg_len=DEFAULT_ARC_SEG_LEN_MM):
    """
    Break a DXF arc into straight chord segments.

    DXF arcs are always CCW, so if end_angle < start_angle the arc wraps
    past 360 and we add 2pi. Returns a list of (x,y,z) INCLUDING both
    endpoints — the caller is expected to drop the first point when
    appending to an existing path.
    """
    start_rad = math.radians(start_angle_deg)
    end_rad = math.radians(end_angle_deg)

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


# -------------------------------------------------------------------------
# DXF ingestion
# -------------------------------------------------------------------------

def extract_segments(dxf_path):
    """
    Read LINE + ARC entities and return them as a list of
    (start, end, interior_points) tuples. For LINEs, interior_points is [].
    For ARCs, interior_points is everything between start and end.

    Raises RuntimeError if no usable entities exist.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    segments = []

    for e in msp.query("LINE"):
        s = (e.dxf.start.x, e.dxf.start.y, e.dxf.start.z)
        t = (e.dxf.end.x, e.dxf.end.y, e.dxf.end.z)
        segments.append((s, t, []))

    for e in msp.query("ARC"):
        cx, cy, cz = e.dxf.center.x, e.dxf.center.y, e.dxf.center.z
        pts = tessellate_arc((cx, cy), e.dxf.radius,
                             e.dxf.start_angle, e.dxf.end_angle, z=cz)
        segments.append((pts[0], pts[-1], pts[1:-1]))

    if not segments:
        raise RuntimeError("No LINE or ARC entities found in " + dxf_path)

    return segments


def check_dxf_is_mm(dxf_path):
    """
    Probe the DXF's $INSUNITS header variable.
    4 = millimeters. Returns True if mm or if header is absent/0 (unitless).
    The GUI layer will warn the user when this returns False.
    """
    doc = ezdxf.readfile(dxf_path)
    try:
        units = doc.header.get("$INSUNITS", 0)
    except Exception:
        return True
    # 0 = unitless (we assume mm and hope), 4 = mm explicitly
    return units in (0, 4)


# -------------------------------------------------------------------------
# Path chaining
# -------------------------------------------------------------------------

def find_nearest_match(current_end, segments, remaining, tol):
    """
    Among `remaining` segment indices, find the one whose start or end
    is closest to `current_end` within `tol`. Returns
    (position_in_remaining, forward_bool, distance) or None.
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
    Build the longest chain possible starting at segments[start_idx].
    Tries tight tolerance first, falls back to loose. Returns
    (path_points, num_unchained_segments).
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

        i_in_remaining, forward, _ = match
        idx = remaining.pop(i_in_remaining)
        s, e, interior = segments[idx]

        if forward:
            path.extend(interior + [e])
        else:
            path.extend(list(reversed(interior)) + [s])

    return path, len(remaining)


def chain_segments(segments, verbose=False):
    """
    Try every segment as a possible starting seed and pick the run that
    captures the most segments. O(N^2) and dumb, but reliable on DXFs
    whose entity ordering in the file is arbitrary.
    """
    tol_tight = TOL
    tol_loose = 0.5

    best_path = []
    best_unchained = len(segments)

    for trial_idx in range(len(segments)):
        path, n_unchained = chain_from(segments, trial_idx, tol_tight, tol_loose)
        if n_unchained < best_unchained:
            best_unchained = n_unchained
            best_path = path
            if n_unchained == 0:
                break

    if verbose and best_unchained > 0:
        print(f"WARNING: {best_unchained} DXF segments could not be chained")

    return best_path, best_unchained


# -------------------------------------------------------------------------
# .inp emission
# -------------------------------------------------------------------------

def _write_inp(points, out_path,
               w=DEFAULT_W_MM, h=DEFAULT_H_MM,
               sigma=DEFAULT_SIGMA_PER_MM,
               nhinc=DEFAULT_NHINC, nwinc=DEFAULT_NWINC,
               fmin=DEFAULT_FMIN_HZ, fmax=DEFAULT_FMAX_HZ,
               freq_ndec=DEFAULT_FREQ_NDEC,
               header_comment="PCB Coil - generated from DXF"):
    """Internal writer. See convert_dxf_to_inp for the public entry point."""
    n_pts = len(points)

    with open(out_path, "w", newline="\n") as f:
        f.write(f"* {header_comment}\n")
        f.write(".Units mm\n\n")
        f.write(f".Default sigma={sigma} w={w} h={h}"
                f" nhinc={nhinc} nwinc={nwinc}\n\n")

        for i, (x, y, z) in enumerate(points):
            f.write(f"N{i} x={x:.6g} y={y:.6g} z={z:.6g}\n")
        f.write("\n")

        for i in range(n_pts - 1):
            f.write(f"E{i} N{i} N{i+1}\n")
        f.write("\n")

        f.write(f".external N0 N{n_pts - 1}\n\n")
        f.write(f".freq fmin={fmin} fmax={fmax} ndec={freq_ndec}\n\n")
        f.write(".end\n")


def convert_dxf_to_inp(dxf_path, out_path,
                       w=DEFAULT_W_MM, h=DEFAULT_H_MM,
                       sigma=DEFAULT_SIGMA_PER_MM,
                       nhinc=DEFAULT_NHINC, nwinc=DEFAULT_NWINC,
                       fmin=DEFAULT_FMIN_HZ, fmax=DEFAULT_FMAX_HZ,
                       z_offset=0.0,
                       header_comment="PCB Coil - generated from DXF",
                       verbose=False):
    """
    Public library entry point. Returns the node list so callers don't
    have to re-parse the .inp they just wrote.

    z_offset is added to every node's z coordinate before writing. This
    lets the merger place a second layer at z = layer_spacing using the
    same converter code path.

    Returns: list of (x, y, z) tuples — the node positions written to disk.
    """
    segments = extract_segments(dxf_path)
    if verbose:
        print(f"{dxf_path}: {len(segments)} DXF segments")

    path, n_unchained = chain_segments(segments, verbose=verbose)
    if verbose:
        print(f"{dxf_path}: chained into {len(path)} points"
              f" ({n_unchained} unchained)")

    if z_offset != 0.0:
        path = [(x, y, z + z_offset) for (x, y, z) in path]

    _write_inp(path, out_path,
               w=w, h=h, sigma=sigma,
               nhinc=nhinc, nwinc=nwinc,
               fmin=fmin, fmax=fmax,
               header_comment=header_comment)

    return path


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.dxf output.inp")
        sys.exit(1)
    convert_dxf_to_inp(sys.argv[1], sys.argv[2], verbose=True)


if __name__ == "__main__":
    main()