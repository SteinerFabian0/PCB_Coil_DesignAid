"""
Strip results.json / refined_results.json down to the minimal schema the
reduced trainer needs (10 input fields + 5 output fields).

The trainer ignores extra keys, so this is purely cosmetic — files shrink
~3-4× and become readable at a glance.

Usage:
    python strip_to_minimal_schema.py <path_to_json> [<path_to_json> ...]

Each file is rewritten in place. A `.bak` copy is created next to it on the
first run; re-running is a no-op except for stripping any new fields that
crept in.
"""

import json
import os
import shutil
import sys

INPUT_FIELDS = [
    "tx_turns", "tx_l2_turns", "tx_width", "tx_od_mm",
    "rx_turns", "rx_width", "rx_od_mm",
    "freq_hz",
    "rx_topology",
]

OUTPUT_FIELDS = ["L_tx_uH", "L_rx_uH", "M_uH", "R_tx_ac", "R_rx_ac"]

KEEP = INPUT_FIELDS + OUTPUT_FIELDS


def strip_record(rec: dict) -> dict:
    return {k: rec[k] for k in KEEP if k in rec}


def strip_file(path: str) -> None:
    if not os.path.exists(path):
        print(f"  skip (missing): {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    records = data.get("results", [])
    if not records:
        print(f"  skip (no records): {path}")
        return

    sample_keys = set(records[0].keys())
    if sample_keys == set(KEEP):
        print(f"  already minimal: {path}")
        return

    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  backup -> {bak}")

    stripped = [strip_record(r) for r in records]
    missing = [
        k for k in INPUT_FIELDS + OUTPUT_FIELDS
        if k not in records[0]
    ]
    if missing:
        print(f"  WARN: source record missing {missing} — emitted records will lack them")

    size_before = os.path.getsize(path)
    out = {"meta": data.get("meta", {}), "results": stripped}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    size_after = os.path.getsize(path)
    print(f"  {path}: {len(stripped)} records  "
          f"{size_before // 1024} kB -> {size_after // 1024} kB")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        here = os.path.dirname(os.path.abspath(__file__))
        nn_dir = os.path.dirname(here)
        paths = [
            os.path.join(nn_dir, "NN_V7", "results.json"),
            os.path.join(nn_dir, "NN_V7", "refined_results.json"),
        ]
        print(f"No paths given — defaulting to NN_V7:")

    for p in paths:
        strip_file(p)


if __name__ == "__main__":
    main()
