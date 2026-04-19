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


def combine_two_port(inp1_path, inp2_path, out_path,
                     header="Two-port coupled coil simulation"):
    """
    Merge two single-port .inp files. Returns summary dict with port indices.
    """
    p1 = parse_inp(inp1_path)
    p2 = parse_inp(inp2_path)

    # Shift coil 2's node indices past coil 1's highest.
    max_n1 = max(int(n[0][1:]) for n in p1["nodes"])
    offset = max_n1 + 1

    # Edge index counter resets to zero and spans both coils, avoiding E# collisions.
    e_counter = 0

    with open(out_path, "w", newline="\n") as f:
        f.write(f"* {header}\n")
        f.write(f"* Coil 1 source: {inp1_path}\n")
        f.write(f"* Coil 2 source: {inp2_path}\n")
        if p1["freq"] != p2["freq"]:
            f.write(f"* NOTE: coil 2 had freq '{p2['freq']}', "
                    f"using coil 1's for the combined sweep.\n")
        f.write(".Units mm\n\n")

        # Nodes, coil 1 then coil 2 with offset.
        for name, x, y, z in p1["nodes"]:
            f.write(f"{name} x={x:.6g} y={y:.6g} z={z:.6g}\n")
        for name, x, y, z in p2["nodes"]:
            f.write(f"{_bump_name(name, offset)} "
                    f"x={x:.6g} y={y:.6g} z={z:.6g}\n")
        f.write("\n")

        # All edges written with fully explicit params (no .Default inheritance
        # across the coil boundary).
        for e in p1["edges"]:
            kv = " ".join(f"{k}={v:g}" for k, v in e["params"].items())
            f.write(f"E{e_counter} {e['from']} {e['to']} {kv}\n")
            e_counter += 1
        for e in p2["edges"]:
            kv = " ".join(f"{k}={v:g}" for k, v in e["params"].items())
            f.write(f"E{e_counter} {_bump_name(e['from'], offset)} "
                    f"{_bump_name(e['to'], offset)} {kv}\n")
            e_counter += 1
        f.write("\n")

        ext1 = p1["external"]
        ext2 = (_bump_name(p2["external"][0], offset),
                _bump_name(p2["external"][1], offset))
        f.write(f".external {ext1[0]} {ext1[1]}\n")
        f.write(f".external {ext2[0]} {ext2[1]}\n\n")

        f.write(f".freq {p1['freq']}\n\n.end\n")

    return {
        "external_coil1": ext1,
        "external_coil2": ext2,
        "offset": offset,
    }