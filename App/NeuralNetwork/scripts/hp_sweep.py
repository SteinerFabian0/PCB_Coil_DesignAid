"""
Hyperparameter sweep for the coil surrogate NN.

Loops over (batch_size, lr, lr_drop_epoch) combinations, invokes
train_surrogate.py once per combo (each into its own temp subfolder),
parses the BEST_VAL_MSE=... line, and keeps the artifacts of the trial
with the absolute lowest validation MSE — promoting them to the model
folder as surrogate_model.pth / x_scaler.pkl / y_scaler.pkl / loss_curve.png.

Per-trial subfolders are deleted as we go (only the best is retained,
then promoted and deleted). A hp_sweep_results.json summary is written
into the model folder.

Configuration is via env vars (set by the GUI launcher):
  SURROGATE_DATA          : path to results.json
  SURROGATE_OUTPUT_DIR    : model folder (results land here)
  SWEEP_EPOCHS            : epochs per trial (default 800)
  SWEEP_VAL_SPLIT         : val split (default 0.2)
  SWEEP_BATCHES           : CSV of batch sizes  (default "1024,2048,4096")
  SWEEP_LRS               : CSV of LRs          (default "0.005,0.001,0.0005")
  SWEEP_LR_DROPS          : CSV of LR-drop epochs (0 = no drop)
                            default "0,700"

A per-trial loss-curve PNG is copied to <OUTPUT_DIR>/hp_sweep_plots/
named "<rank>_<tag>.png" so the UI can show the full grid.

Total runs = |batches| * |lrs| * |lr_drops|.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from itertools import product

_HERE       = os.path.dirname(os.path.abspath(__file__))
_NN_DIR     = os.path.dirname(_HERE)
TRAIN_SCRIPT = os.path.join(_HERE, "train_surrogate.py")

DATA_PATH  = os.environ["SURROGATE_DATA"]
OUTPUT_DIR = os.environ["SURROGATE_OUTPUT_DIR"]

EPOCHS    = int(os.environ.get("SWEEP_EPOCHS",   "800"))
VAL_SPLIT = float(os.environ.get("SWEEP_VAL_SPLIT", "0.2"))

def _csv(name, default):
    return [s.strip() for s in os.environ.get(name, default).split(",") if s.strip()]

BATCHES  = [int(x)   for x in _csv("SWEEP_BATCHES",  "1024,2048,4096")]
LRS      = [float(x) for x in _csv("SWEEP_LRS",      "0.005,0.001,0.0005")]
LR_DROPS = [int(x)   for x in _csv("SWEEP_LR_DROPS", "0,700")]

ARTIFACTS = ("surrogate_model.pth", "x_scaler.pkl", "y_scaler.pkl", "loss_curve.png")
TRIAL_PREFIX = "_hp_trial_"
PLOTS_SUBDIR = "hp_sweep_plots"


def _parse_best(stdout: str) -> tuple[float, int]:
    best_mse, best_epoch = float("inf"), -1
    for line in stdout.splitlines():
        if line.startswith("BEST_VAL_MSE="):
            parts = dict(p.split("=", 1) for p in line.split() if "=" in p)
            try:
                best_mse   = float(parts.get("BEST_VAL_MSE", "inf"))
                best_epoch = int(parts.get("BEST_EPOCH",  "-1"))
            except ValueError:
                pass
    return best_mse, best_epoch


def _run_trial(trial_idx: int, batch: int, lr: float, drop_epoch: int) -> dict:
    tag = f"b{batch}_lr{lr:g}" + (f"_drop{drop_epoch}" if drop_epoch else "_nodrop")
    trial_dir = os.path.join(OUTPUT_DIR, f"{TRIAL_PREFIX}{trial_idx:02d}_{tag}")
    os.makedirs(trial_dir, exist_ok=True)

    env = os.environ.copy()
    env["SURROGATE_DATA"]           = DATA_PATH
    env["SURROGATE_OUTPUT_DIR"]     = trial_dir
    env["SURROGATE_EPOCHS"]         = str(EPOCHS)
    env["SURROGATE_BATCH_SIZE"]     = str(batch)
    env["SURROGATE_LR"]             = str(lr)
    env["SURROGATE_VAL_SPLIT"]      = str(VAL_SPLIT)
    env["SURROGATE_LR_DROP_EPOCH"]  = str(drop_epoch)
    env["SURROGATE_LR_DROP_FACTOR"] = "0.1"

    header = (f"\n=== Trial {trial_idx}: batch={batch} lr={lr} "
              f"lr_drop={'epoch ' + str(drop_epoch) if drop_epoch else 'none'} ===")
    print(header, flush=True)

    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", TRAIN_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=_NN_DIR,
    )
    captured = []
    for line in proc.stdout:
        line = line.rstrip()
        captured.append(line)
        print(f"  {line}", flush=True)
    proc.wait()
    elapsed = time.time() - t0

    best_mse, best_epoch = _parse_best("\n".join(captured))
    print(f"  -> best_val_mse={best_mse:.6f} (epoch {best_epoch})  [{elapsed:.1f}s]",
          flush=True)

    return {
        "trial":      trial_idx,
        "tag":        tag,
        "batch":      batch,
        "lr":         lr,
        "lr_drop_epoch": drop_epoch,
        "epochs":     EPOCHS,
        "best_val_mse": best_mse,
        "best_epoch": best_epoch,
        "elapsed_s":  elapsed,
        "rc":         proc.returncode,
        "trial_dir":  trial_dir,
    }


def _promote(trial_dir: str):
    """Copy the trial's artifacts into OUTPUT_DIR (overwriting any existing)."""
    for name in ARTIFACTS:
        src = os.path.join(trial_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_DIR, name))


def _cleanup(trial_dir: str):
    shutil.rmtree(trial_dir, ignore_errors=True)


def main():
    combos = list(product(BATCHES, LRS, LR_DROPS))
    print(f"HP sweep: {len(combos)} trials")
    print(f"  batches  : {BATCHES}")
    print(f"  lrs      : {LRS}")
    print(f"  lr_drops : {LR_DROPS}")
    print(f"  epochs   : {EPOCHS}")
    print(f"  data     : {DATA_PATH}")
    print(f"  out      : {OUTPUT_DIR}")

    plots_dir = os.path.join(OUTPUT_DIR, PLOTS_SUBDIR)
    shutil.rmtree(plots_dir, ignore_errors=True)
    os.makedirs(plots_dir, exist_ok=True)

    results = []
    best_so_far = float("inf")
    best_trial_dir = None
    best_record = None
    t_start = time.time()

    for i, (batch, lr, drop) in enumerate(combos, 1):
        rec = _run_trial(i, batch, lr, drop)
        results.append({k: v for k, v in rec.items() if k != "trial_dir"})

        # Preserve this trial's loss-curve PNG so the UI can render the grid.
        src_png = os.path.join(rec["trial_dir"], "loss_curve.png")
        if os.path.exists(src_png):
            dst_png = os.path.join(plots_dir, f"{i:02d}_{rec['tag']}.png")
            try:
                shutil.copy2(src_png, dst_png)
                results[-1]["plot"] = os.path.relpath(dst_png, OUTPUT_DIR)
            except OSError:
                pass

        mse = rec["best_val_mse"]
        if mse < best_so_far and rec["rc"] == 0:
            # New leader — drop the previous winner's folder, keep this one.
            if best_trial_dir is not None:
                _cleanup(best_trial_dir)
            best_so_far    = mse
            best_trial_dir = rec["trial_dir"]
            best_record    = results[-1]
            print(f"  *** New best: {mse:.6f} (trial {i}) ***", flush=True)
        else:
            _cleanup(rec["trial_dir"])

        # Write running summary so progress survives a crash.
        summary = {
            "data":     DATA_PATH,
            "epochs":   EPOCHS,
            "batches":  BATCHES,
            "lrs":      LRS,
            "lr_drops": LR_DROPS,
            "results":  results,
            "best":     best_record,
            "elapsed_s": time.time() - t_start,
        }
        with open(os.path.join(OUTPUT_DIR, "hp_sweep_results.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if best_trial_dir is not None:
        _promote(best_trial_dir)
        _cleanup(best_trial_dir)
        print(f"\nPromoted best trial -> {OUTPUT_DIR}")
    else:
        print("\nNo successful trial — nothing promoted.")

    # Final ranked table.
    ranked = sorted(results, key=lambda r: r["best_val_mse"])
    print("\nRanking (lowest val MSE first):")
    print(f"{'rank':>4}  {'trial':>5}  {'batch':>6}  {'lr':>8}  {'drop':>5}  "
          f"{'val_mse':>12}  {'epoch':>5}")
    for rank, r in enumerate(ranked, 1):
        print(f"{rank:>4}  {r['trial']:>5}  {r['batch']:>6}  {r['lr']:>8g}  "
              f"{r['lr_drop_epoch']:>5}  {r['best_val_mse']:>12.6f}  "
              f"{r['best_epoch']:>5}")

    total = time.time() - t_start
    print(f"\nSweep done in {total:.1f}s. Best val MSE: {best_so_far:.6f}")


if __name__ == "__main__":
    main()
