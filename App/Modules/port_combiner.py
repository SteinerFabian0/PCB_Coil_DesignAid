#!/usr/bin/env python3
"""
Combine two single-port FastHenry .inp files into a two-port .inp.

Why a separate module: the sim tab needs this when both coils are
registered (mutual-inductance run). The DXF merger won't do it because
its job is merging two halves of ONE coil in series via a physical via.
Here we keep the coils electrically independent — FastHenry sees two
ports and returns the full 2x2 Z matrix.

Strategy
--------
Parse each input to structured form, tracking the effective w/h/sigma
of every edge through any .Default state changes. Re-emit as one file
with all per-edge parameters written explicitly — this sidesteps the
bug where coil 2's edges would otherwise inherit coil 1's trailing
.Default values.
"""

import re


_NODE_RE     = re.compile(r'^(N\d+)\s+x=([-\d.eE+]+)\s+y=([-\d.eE+]+)'
                          r'\s+z=([-\d.eE+]+)', re.IGNORECASE)
_EDGE_RE     = re.compile(r'^(E\d+)\s+(N\d+)\s+(N\d+)(.*)$', re.IGNORECASE)
_DEFAULT_RE  = re.compile(r'^\.Default\s+(.*)$', re.IGNORECASE)
_EXTERNAL_RE = re.compile(r'^\.external\s+(N\d+)\s+(N\d+)', re.IGNORECASE)
_FREQ_RE     = re.compile(r'^\.freq\s+(.*)$', re.IGNORECASE)
_KV_RE       = re.compile(r'(\w+)\s*=\s*([-\d.eE+]+)')


def _parse_kv(s):
    """'w=0.52 h=0.035 sigma=5.8e4' -> {'w':0.52,'h':0.035,'sigma':58000.0}"""
    return {k: float(v) for k, v in _KV_RE.findall(s)}


def parse_inp(path):
    """
    Structured parse. External is the FIRST .external found (we only
    support single-port inputs here). Returns a dict; raises on missing
    mandatory directives.
    """
    nodes, edges = [], []
    external = None
    freq = None
    current_default = {}

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*") or s.lower().startswith(".units"):
                continue
            if s.lower().startswith(".end"):
                break

            m = _DEFAULT_RE.match(s)
            if m:
                current_default = _parse_kv(m.group(1))
                continue

            m = _NODE_RE.match(s)
            if m:
                nodes.append((m.group(1),
                              float(m.group(2)),
                              float(m.group(3)),
                              float(m.group(4))))
                continue

            m = _EDGE_RE.match(s)
            if m:
                params = dict(current_default)          # inherit defaults
                params.update(_parse_kv(m.group(4) or ""))  # per-edge overrides
                edges.append({
                    "name":  m.group(1),
                    "from":  m.group(2),
                    "to":    m.group(3),
                    "params": params,
                })
                continue

            m = _EXTERNAL_RE.match(s)
            if m and external is None:
                external = (m.group(1), m.group(2))
                continue

            m = _FREQ_RE.match(s)
            if m:
                freq = m.group(1)
                continue

    if external is None:
        raise ValueError(f"{path}: no .external directive found")
    if freq is None:
        raise ValueError(f"{path}: no .freq directive found")

    return {"nodes": nodes, "edges": edges,
            "external": external, "freq": freq}


def _bump_name(name, offset):
    assert name.startswith("N"), name
    return f"N{int(name[1:]) + offset}"


def combine_two_port(inp_tx_path, inp_rx_path, out_path,
                     pcb_gap_mm=0.0,
                     header="Two-port coupled coil simulation"):
    """
    Merge two single-port .inp files (TX + RX) into a two-port .inp.

    Stacking convention:
      - RX nodes are written at their original z coordinates.
      - TX nodes are shifted UP in code-z so that TX's lowest z lands at
        (RX's max z + pcb_gap_mm). I.e. the PCB gap separates TX's bottom
        and RX's top. Port 0 = TX (first .external), port 1 = RX.

    In the user's mental model (where each coil's slot 1 is "topmost"),
    this places RX physically above TX with `pcb_gap_mm` separation.
    """
    p_tx = parse_inp(inp_tx_path)
    p_rx = parse_inp(inp_rx_path)

    # Z-shift computation.
    z_tx_min = min(n[3] for n in p_tx["nodes"])
    z_rx_max = max(n[3] for n in p_rx["nodes"])
    z_shift_tx = z_rx_max + pcb_gap_mm - z_tx_min

    # Renumber RX past TX's highest index; keeps port 0 = TX.
    max_n_tx = max(int(n[0][1:]) for n in p_tx["nodes"])
    offset = max_n_tx + 1

    e_counter = 0
    with open(out_path, "w", newline="\n") as f:
        f.write(f"* {header}\n")
        f.write(f"* TX source: {inp_tx_path}\n")
        f.write(f"* RX source: {inp_rx_path}\n")
        f.write(f"* PCB gap (TX-top to RX-bottom, code-z): {pcb_gap_mm} mm\n")
        f.write(f"* TX z-shift applied: {z_shift_tx:.4f} mm\n")
        if p_tx["freq"] != p_rx["freq"]:
            f.write(f"* NOTE: RX .freq '{p_rx['freq']}' ignored; "
                    f"using TX's '{p_tx['freq']}'.\n")
        f.write(".Units mm\n\n")

        # TX nodes first (original N-indices), z-shifted.
        for name, x, y, z in p_tx["nodes"]:
            f.write(f"{name} x={x:.6g} y={y:.6g} z={z + z_shift_tx:.6g}\n")
        # RX nodes with renumbered indices, original z.
        for name, x, y, z in p_rx["nodes"]:
            f.write(f"{_bump_name(name, offset)} "
                    f"x={x:.6g} y={y:.6g} z={z:.6g}\n")
        f.write("\n")

        # Edges with all per-edge params explicit (no .Default inheritance).
        for e in p_tx["edges"]:
            kv = " ".join(f"{k}={v:g}" for k, v in e["params"].items())
            f.write(f"E{e_counter} {e['from']} {e['to']} {kv}\n")
            e_counter += 1
        for e in p_rx["edges"]:
            kv = " ".join(f"{k}={v:g}" for k, v in e["params"].items())
            f.write(f"E{e_counter} {_bump_name(e['from'], offset)} "
                    f"{_bump_name(e['to'], offset)} {kv}\n")
            e_counter += 1
        f.write("\n")

        # TX external first → port 0 = TX.
        ext_tx = p_tx["external"]
        ext_rx = (_bump_name(p_rx["external"][0], offset),
                  _bump_name(p_rx["external"][1], offset))
        f.write(f".external {ext_tx[0]} {ext_tx[1]}\n")
        f.write(f".external {ext_rx[0]} {ext_rx[1]}\n\n")
        f.write(f".freq {p_tx['freq']}\n\n.end\n")

    return {
        "external_tx": ext_tx,
        "external_rx": ext_rx,
        "offset": offset,
        "z_shift_tx": z_shift_tx,
    }