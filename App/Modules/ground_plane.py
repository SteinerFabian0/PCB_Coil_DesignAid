"""Circular ground plane / passive copper obstruction mesh generation for FastHenry2."""

from dataclasses import dataclass
import math

GP_NODE_OFFSET = 100_000
GP_EDGE_OFFSET = 100_000
COPPER_SIGMA_PER_MM = 58000.0
OZ_TO_MM = 0.035


@dataclass
class GroundPlaneParams:
    """Parameters for circular ground plane mesh."""

    dia_mm: float  # 0 = disabled
    z_mm: float  # z-position
    mesh_step_mm: float = 2.0
    copper_oz: float = 1.0
    sigma: float = COPPER_SIGMA_PER_MM

    @property
    def radius_mm(self) -> float:
        return self.dia_mm / 2.0

    @property
    def copper_h_mm(self) -> float:
        return self.copper_oz * OZ_TO_MM


def generate_ground_plane_mesh(
    params: GroundPlaneParams,
    node_offset: int = GP_NODE_OFFSET,
    edge_offset: int = GP_EDGE_OFFSET,
) -> dict:
    """
    Generate circular ground plane mesh as rectangular grid cropped to circle.

    Returns:
        {
            "nodes": [(name, x, y, z), ...],
            "edges": [(name, n_from, n_to), ...],
            "node_count": int,
            "edge_count": int,
            "params": params,
        }
        or empty lists when dia_mm <= 0.
    """
    if params.dia_mm <= 0:
        return {
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
            "params": params,
        }

    r = params.radius_mm
    step = params.mesh_step_mm

    # Build set of active grid points
    grid_points = set()
    ix_min = math.floor(-r / step) - 1
    ix_max = math.ceil(r / step) + 1
    iy_min = ix_min
    iy_max = ix_max

    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            x = ix * step
            y = iy * step
            if x * x + y * y <= r * r:
                grid_points.add((ix, iy))

    # Sort points deterministically
    sorted_points = sorted(grid_points, key=lambda p: (p[1], p[0]))

    # Map grid index to global node index
    node_map = {}
    for local_idx, (ix, iy) in enumerate(sorted_points):
        global_idx = node_offset + local_idx
        node_map[(ix, iy)] = global_idx

    # Emit nodes
    nodes = []
    for (ix, iy), global_idx in node_map.items():
        x = ix * step
        y = iy * step
        z = params.z_mm
        node_name = f"N{global_idx}"
        nodes.append((node_name, x, y, z))

    # Emit edges (horizontal, then vertical)
    edges = []
    edge_count = 0

    # Horizontal edges
    for (ix, iy) in sorted_points:
        if (ix + 1, iy) in grid_points:
            n_from = node_map[(ix, iy)]
            n_to = node_map[(ix + 1, iy)]
            edge_name = f"E{edge_offset + edge_count}"
            edges.append((edge_name, f"N{n_from}", f"N{n_to}"))
            edge_count += 1

    # Vertical edges
    for (ix, iy) in sorted_points:
        if (ix, iy + 1) in grid_points:
            n_from = node_map[(ix, iy)]
            n_to = node_map[(ix, iy + 1)]
            edge_name = f"E{edge_offset + edge_count}"
            edges.append((edge_name, f"N{n_from}", f"N{n_to}"))
            edge_count += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": edge_count,
        "params": params,
    }


def ground_plane_inp_block(
    params: GroundPlaneParams,
    node_offset: int = GP_NODE_OFFSET,
    edge_offset: int = GP_EDGE_OFFSET,
) -> str:
    """
    Generate FastHenry .inp text block for ground plane (comment, .Default, N-lines, E-lines).

    Returns empty string when params.dia_mm <= 0.
    """
    if params.dia_mm <= 0:
        return ""

    mesh = generate_ground_plane_mesh(params, node_offset, edge_offset)

    if mesh["node_count"] == 0:
        return ""

    lines = []
    lines.append(
        f"* Ground plane: dia={params.dia_mm:.1f}mm step={params.mesh_step_mm}mm z={params.z_mm:.3f}mm"
    )
    lines.append(
        f"* {mesh['node_count']} nodes, {mesh['edge_count']} edges (passive floating — no .external)"
    )
    lines.append(
        f".Default sigma={params.sigma:.0f} w={params.mesh_step_mm:.3f} h={params.copper_h_mm:.6f} nhinc=1 nwinc=1"
    )

    # Nodes
    for node_name, x, y, z in mesh["nodes"]:
        lines.append(f"{node_name} x={x:.6g} y={y:.6g} z={z:.6g}")

    # Edges
    for edge_name, n_from, n_to in mesh["edges"]:
        lines.append(f"{edge_name} {n_from} {n_to}")

    return "\n".join(lines)
