"""Passive copper-obstruction mesh generation for FastHenry2.

Two ground-plane families are supported, both meshed as a uniform grid of
thin-conductor `N` nodes connected by `E` edges (passive — no .external):

  * Disc       — `DiscGroundPlaneParams`     (used for RX side, two inner layers)
  * Polygon    — `PolygonGroundPlaneParams`  (used for TX side, layer-3 pour)

Polygons are described as the union of axis-aligned ADD rectangles, with
SUBTRACT rectangles and SUBTRACT circles cut out. A grid cell is "active"
iff its center lies inside at least one ADD rect AND inside no SUB rect AND
inside no SUB circle.

The TX layer-3 ground geometry is fixed hardware on this branch and is
exported as the constant `TX_LAYER3_POLYGON_SPEC`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import math

GP_NODE_OFFSET = 100_000
GP_EDGE_OFFSET = 100_000
COPPER_SIGMA_PER_MM = 58000.0
OZ_TO_MM = 0.035


# ---------------------------------------------------------------------------
# Disc ground plane (RX 20 mm disc)
# ---------------------------------------------------------------------------

@dataclass
class DiscGroundPlaneParams:
    """Parameters for a circular passive ground disc."""
    dia_mm: float                       # 0 = disabled
    z_mm: float
    mesh_step_mm: float = 1.5
    copper_oz: float = 0.5
    sigma: float = COPPER_SIGMA_PER_MM

    @property
    def radius_mm(self) -> float:
        return self.dia_mm / 2.0

    @property
    def copper_h_mm(self) -> float:
        return self.copper_oz * OZ_TO_MM


# Backwards-compatible alias — older code/tests refer to GroundPlaneParams.
GroundPlaneParams = DiscGroundPlaneParams


# ---------------------------------------------------------------------------
# Polygon ground plane (TX layer-3 pour)
# ---------------------------------------------------------------------------

@dataclass
class PolygonGroundPlaneParams:
    """
    Parameters for a passive polygon ground pour built from rectangle ops.

    `add_rects` / `sub_rects` are tuples (x_min, y_min, x_max, y_max) in mm.
    `sub_circles` are tuples (cx, cy, radius_mm).
    """
    add_rects:   List[Tuple[float, float, float, float]] = field(default_factory=list)
    sub_rects:   List[Tuple[float, float, float, float]] = field(default_factory=list)
    sub_circles: List[Tuple[float, float, float]]        = field(default_factory=list)
    z_mm:         float = 0.0
    mesh_step_mm: float = 1.5
    copper_oz:    float = 0.5
    sigma:        float = COPPER_SIGMA_PER_MM

    @property
    def copper_h_mm(self) -> float:
        return self.copper_oz * OZ_TO_MM


# TX layer-3 ground pour spec, in coil-relative coordinates (mm).
# Origin = TX coil center. ADD = pour, SUB = cutouts. Approximating the two
# Ø3 mm via clearances as equivalent-area squares (side √π·1.5 ≈ 2.659 mm),
# implemented here as sub-circles with the FULL Ø3 radius for clean geometry —
# the mesh discretisation already has ~0.5 mm fidelity, so circle vs square is
# in the noise.
TX_LAYER3_POLYGON_SPEC = PolygonGroundPlaneParams(
    add_rects=[
        (-10.0, -3.0, 30.0, 3.0),    # 40×6 mm horizontal arm
        ( -4.0, -6.0,  4.0, -3.0),   # 8×3 mm tab below the arm
    ],
    sub_rects=[
        (  9.0, -1.5, 30.0,  1.5),   # 21×3 mm slot in right portion of arm
        (  3.6, -3.0,  6.5,  0.0),   # 2.9×3 mm notch in lower edge
    ],
    sub_circles=[
        ( 0.0, 0.0, 1.5),            # Ø3 mm via clearance at coil center
        (-9.4, 0.0, 1.5),            # Ø3 mm via clearance at (-9.4, 0)
    ],
    mesh_step_mm=1.5,
    copper_oz=0.5,
)


# ---------------------------------------------------------------------------
# Disc-mesh generation
# ---------------------------------------------------------------------------

def generate_disc_mesh(
    params: DiscGroundPlaneParams,
    node_offset: int = GP_NODE_OFFSET,
    edge_offset: int = GP_EDGE_OFFSET,
) -> dict:
    """Build a uniform-grid disc mesh. Empty result when dia_mm <= 0."""
    if params.dia_mm <= 0:
        return _empty_mesh(params)

    r    = params.radius_mm
    step = params.mesh_step_mm

    grid_points = set()
    ix_min = math.floor(-r / step) - 1
    ix_max = math.ceil( r / step) + 1
    for ix in range(ix_min, ix_max + 1):
        for iy in range(ix_min, ix_max + 1):
            x = ix * step
            y = iy * step
            if x * x + y * y <= r * r:
                grid_points.add((ix, iy))

    return _emit_mesh(grid_points, step, params.z_mm,
                      node_offset, edge_offset, params)


# Legacy name kept for any external callers.
generate_ground_plane_mesh = generate_disc_mesh


# ---------------------------------------------------------------------------
# Polygon-mesh generation
# ---------------------------------------------------------------------------

def generate_polygon_mesh(
    params: PolygonGroundPlaneParams,
    node_offset: int = GP_NODE_OFFSET,
    edge_offset: int = GP_EDGE_OFFSET,
) -> dict:
    """
    Build a uniform-grid polygon mesh from add/sub rectangles and sub circles.
    Empty result when no add_rects.
    """
    if not params.add_rects:
        return _empty_mesh(params)

    step = params.mesh_step_mm

    # Bounding box of all ADD rects.
    x_lo = min(r[0] for r in params.add_rects)
    y_lo = min(r[1] for r in params.add_rects)
    x_hi = max(r[2] for r in params.add_rects)
    y_hi = max(r[3] for r in params.add_rects)

    ix_min = math.floor(x_lo / step) - 1
    ix_max = math.ceil( x_hi / step) + 1
    iy_min = math.floor(y_lo / step) - 1
    iy_max = math.ceil( y_hi / step) + 1

    grid_points = set()
    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            cx = ix * step
            cy = iy * step
            if not _point_in_any_rect(cx, cy, params.add_rects):
                continue
            if _point_in_any_rect(cx, cy, params.sub_rects):
                continue
            if _point_in_any_circle(cx, cy, params.sub_circles):
                continue
            grid_points.add((ix, iy))

    return _emit_mesh(grid_points, step, params.z_mm,
                      node_offset, edge_offset, params)


# ---------------------------------------------------------------------------
# Shared mesh helpers
# ---------------------------------------------------------------------------

def _empty_mesh(params) -> dict:
    return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0,
            "params": params}


def _point_in_any_rect(x: float, y: float,
                       rects: List[Tuple[float, float, float, float]]) -> bool:
    for x0, y0, x1, y1 in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _point_in_any_circle(x: float, y: float,
                         circles: List[Tuple[float, float, float]]) -> bool:
    for cx, cy, r in circles:
        dx, dy = x - cx, y - cy
        if dx * dx + dy * dy <= r * r:
            return True
    return False


def _emit_mesh(grid_points, step: float, z_mm: float,
               node_offset: int, edge_offset: int, params) -> dict:
    if not grid_points:
        return _empty_mesh(params)

    sorted_points = sorted(grid_points, key=lambda p: (p[1], p[0]))
    node_map = {pt: node_offset + i for i, pt in enumerate(sorted_points)}

    nodes = []
    for (ix, iy), gid in node_map.items():
        nodes.append((f"N{gid}", ix * step, iy * step, z_mm))

    edges = []
    edge_count = 0
    for (ix, iy) in sorted_points:
        if (ix + 1, iy) in grid_points:
            edges.append((f"E{edge_offset + edge_count}",
                          f"N{node_map[(ix, iy)]}",
                          f"N{node_map[(ix + 1, iy)]}"))
            edge_count += 1
    for (ix, iy) in sorted_points:
        if (ix, iy + 1) in grid_points:
            edges.append((f"E{edge_offset + edge_count}",
                          f"N{node_map[(ix, iy)]}",
                          f"N{node_map[(ix, iy + 1)]}"))
            edge_count += 1

    return {"nodes": nodes, "edges": edges,
            "node_count": len(nodes), "edge_count": edge_count,
            "params": params}


# ---------------------------------------------------------------------------
# .inp text emission
# ---------------------------------------------------------------------------

def disc_inp_block(params: DiscGroundPlaneParams,
                   node_offset: int = GP_NODE_OFFSET,
                   edge_offset: int = GP_EDGE_OFFSET) -> str:
    if params.dia_mm <= 0:
        return ""
    mesh = generate_disc_mesh(params, node_offset, edge_offset)
    if mesh["node_count"] == 0:
        return ""
    header = (f"* Ground disc: dia={params.dia_mm:.1f}mm "
              f"step={params.mesh_step_mm}mm z={params.z_mm:.3f}mm")
    return _mesh_to_inp_text(mesh, header)


def polygon_inp_block(params: PolygonGroundPlaneParams,
                      node_offset: int = GP_NODE_OFFSET,
                      edge_offset: int = GP_EDGE_OFFSET) -> str:
    if not params.add_rects:
        return ""
    mesh = generate_polygon_mesh(params, node_offset, edge_offset)
    if mesh["node_count"] == 0:
        return ""
    header = (f"* Ground polygon: {len(params.add_rects)} add / "
              f"{len(params.sub_rects)} sub-rect / "
              f"{len(params.sub_circles)} sub-circle  "
              f"step={params.mesh_step_mm}mm z={params.z_mm:.3f}mm")
    return _mesh_to_inp_text(mesh, header)


# Legacy alias.
ground_plane_inp_block = disc_inp_block


def _mesh_to_inp_text(mesh: dict, header: str) -> str:
    p = mesh["params"]
    lines = [
        header,
        f"* {mesh['node_count']} nodes, {mesh['edge_count']} edges "
        f"(passive floating — no .external)",
        f".Default sigma={p.sigma:.0f} w={p.mesh_step_mm:.3f} "
        f"h={p.copper_h_mm:.6f} nhinc=1 nwinc=1",
    ]
    for name, x, y, z in mesh["nodes"]:
        lines.append(f"{name} x={x:.6g} y={y:.6g} z={z:.6g}")
    for name, n_from, n_to in mesh["edges"]:
        lines.append(f"{name} {n_from} {n_to}")
    return "\n".join(lines)
