#!/usr/bin/env python3
"""
Savestate: read/write all coil-related GUI settings to JSON.

File: <project_root>/savestate/savestate.json

Structure:
    {
      "param": {
        "TX": { ... per-tab dict ... },
        "RX": { ... }
      },
      "dxf": {
        "TX": { ... },
        "RX": { ... }
      },
      "sim": { ... }
    }

Corrupt/missing file → empty dict; callers fall back to defaults.
"""

import os, json

SAVESTATE_DIR_NAME = "savestate"
SAVESTATE_FILE = "savestate.json"


def _path(project_root):
    d = os.path.join(project_root, SAVESTATE_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, SAVESTATE_FILE)


def load(project_root):
    p = _path(project_root)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save(project_root, state):
    """Atomic-ish write via temp + rename."""
    p = _path(project_root)
    tmp = p + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, p)
    except Exception:
        # Best-effort; don't crash the GUI on disk errors.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def get_section(state, section, key):
    """Safe nested get: state[section][key] or None."""
    return (state or {}).get(section, {}).get(key)


def set_section(state, section, key, value):
    """Mutate state: state[section][key] = value, creating as needed."""
    state.setdefault(section, {})[key] = value