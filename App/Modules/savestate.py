#!/usr/bin/env python3
"""
Session persistence: serialize the GUI's user-visible settings to JSON
on close, deserialize on open. Fails safe — any read/parse error just
returns None so the GUI boots with defaults.
"""

import json
import os


STATE_VERSION = 1


def save_state(filepath, state_dict):
    """
    Write state_dict to filepath atomically-ish (write temp + rename).
    Returns True on success, False on any failure.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp = filepath + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": STATE_VERSION, "state": state_dict},
                      f, indent=2)
        os.replace(tmp, filepath)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def load_state(filepath):
    """
    Return state dict from filepath, or None if missing/unreadable/
    wrong-version. Never raises.
    """
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("version") != STATE_VERSION:
            return None
        return payload.get("state")
    except Exception:
        return None