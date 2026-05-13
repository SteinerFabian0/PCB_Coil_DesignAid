"""
Surrogate Neural Network for PCB Coil Parameter Prediction.

Trains a feed-forward MLP that maps physical coil parameters -> electrical
characteristics, replacing slow FastHenry simulations during optimisation.

Inputs are raw geometry fields read directly from each simulation record.
No analytical "derived" features (Wheeler, fill factor, ln N², ...). On
this branch (NN_V8), TX & RX topologies and stackup are entirely fixed by
the trainer's domain.json — the only RX free variables that vary across
samples are outer turns, inner turns, trace width, and outer diameter
(plus frequency). The NN therefore has no topology feature at all.

Outputs : L_tx_uH, L_rx_uH, M_uH, R_tx_ac, R_rx_ac
"""

import contextlib
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HERE      = os.path.dirname(os.path.abspath(__file__))
_NN_DIR    = os.path.dirname(_HERE)

DATA_PATH  = os.environ.get("SURROGATE_DATA", os.path.join(_NN_DIR, "NN_V8", "results.json"))
OUTPUT_DIR = os.environ.get("SURROGATE_OUTPUT_DIR", os.path.join(_NN_DIR, "NN_V8"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS      = int(os.environ.get("SURROGATE_EPOCHS",     "600"))
BATCH_SIZE  = int(os.environ.get("SURROGATE_BATCH_SIZE", "1024"))
LR          = float(os.environ.get("SURROGATE_LR",       "1e-3"))
VAL_SPLIT   = float(os.environ.get("SURROGATE_VAL_SPLIT", "0.20"))
# Optional step-LR decay: at LR_DROP_EPOCH, multiply LR by LR_DROP_FACTOR.
# Setting LR_DROP_EPOCH=0 disables it (default).
LR_DROP_EPOCH  = int(os.environ.get("SURROGATE_LR_DROP_EPOCH",  "0"))
LR_DROP_FACTOR = float(os.environ.get("SURROGATE_LR_DROP_FACTOR", "0.1"))
RANDOM_SEED = 42
PRINT_EVERY = max(1, EPOCHS // 20)

# Numeric input columns. Only fields that vary across the LHS samples in
# this branch. Topology, stackup, spacings, gaps, pcb_gap, ground disc, and
# port_inside are all fixed by the trainer's domain.json and would
# StandardScaler-collapse to zero, so they're dropped entirely.
NUMERIC_INPUT_COLS = [
    "tx_turns", "tx_l2_turns", "tx_width", "tx_od_mm",
    "rx_turns", "rx_inner_turns", "rx_width", "rx_od_mm",
    "freq_hz",
]

OUTPUT_COLS = ["L_tx_uH", "L_rx_uH", "M_uH", "R_tx_ac", "R_rx_ac"]


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        raw = json.load(f)

    records = raw["results"]
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} simulation records from {path}")
    return df


# ---------------------------------------------------------------------------
# 2. Pre-process
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame):
    for c in NUMERIC_INPUT_COLS:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0).astype(float)

    X = df[NUMERIC_INPUT_COLS].astype(float)
    y = df[OUTPUT_COLS].astype(float)

    print(f"Input features  : {list(X.columns)}  ({X.shape[1]} total)")
    print(f"Output targets  : {list(y.columns)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s   = y_scaler.transform(y_val)

    print(f"Train: {len(X_train_s)}  |  Val: {len(X_val_s)}")

    return (X_train_s, X_val_s, y_train_s, y_val_s,
            x_scaler, y_scaler, X.shape[1])


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------

class CoilSurrogateNN(nn.Module):
    def __init__(self, n_in: int, n_out: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def train(model, X_train, y_train, X_val, y_val, device):
    """
    Train with all data resident on GPU and AMP forward/backward.

    For this surrogate (~30 input dims, ~3-5k records) the previous
    DataLoader-on-CPU path was launch-overhead bound — moving the full
    train/val tensors to GPU once eliminates per-batch host→device copies
    and lets the 4070 Ti spend its time on actual compute.
    """
    Xt_train = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32)).to(device)
    yt_train = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32)).to(device)
    Xt_val   = torch.from_numpy(np.ascontiguousarray(X_val,   dtype=np.float32)).to(device)
    yt_val   = torch.from_numpy(np.ascontiguousarray(y_val,   dtype=np.float32)).to(device)

    n_train = Xt_train.shape[0]
    n_val   = Xt_val.shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=30
    )

    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    def _autocast():
        return (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp else contextlib.nullcontext())

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        running = 0.0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb  = Xt_train[idx]
            yb  = yt_train[idx]
            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                pred = model(xb)
                loss = criterion(pred, yb)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running += loss.item() * xb.shape[0]
        train_loss = running / n_train

        model.eval()
        with torch.no_grad():
            with _autocast():
                pred = model(Xt_val)
                val_loss = criterion(pred.float(), yt_val).item()

        scheduler.step(val_loss)
        if LR_DROP_EPOCH and epoch == LR_DROP_EPOCH:
            for pg in optimizer.param_groups:
                pg["lr"] *= LR_DROP_FACTOR
            print(f"  [LR drop] epoch {epoch}: lr -> {optimizer.param_groups[0]['lr']:.2e}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{EPOCHS}  |  Train MSE: {train_loss:.6f}  |  Val MSE: {val_loss:.6f}")

    print(f"\nBest val loss  : {best_val_loss:.6f}  (epoch {best_epoch})")
    print(f"BEST_VAL_MSE={best_val_loss:.8f} BEST_EPOCH={best_epoch}")
    model.load_state_dict(best_state)
    return train_losses, val_losses, best_val_loss, best_epoch


# ---------------------------------------------------------------------------
# 5. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(model, x_scaler, y_scaler, train_losses, val_losses,
                   best_val_loss=None, best_epoch=None):
    model_path    = os.path.join(OUTPUT_DIR, "surrogate_model.pth")
    x_scaler_path = os.path.join(OUTPUT_DIR, "x_scaler.pkl")
    y_scaler_path = os.path.join(OUTPUT_DIR, "y_scaler.pkl")
    plot_path     = os.path.join(OUTPUT_DIR, "loss_curve.png")

    torch.save(model.state_dict(), model_path)
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    drop_note = (f"  |  LR drop @ epoch {LR_DROP_EPOCH} ×{LR_DROP_FACTOR}"
                 if LR_DROP_EPOCH else "  |  no LR drop")
    best_note = ""
    if best_val_loss is not None and best_epoch is not None:
        best_note = f"\nBest val MSE: {best_val_loss:.6f}  (epoch {best_epoch})"
    title = (f"Surrogate NN — batch={BATCH_SIZE}  lr={LR:g}"
             f"{drop_note}{best_note}")

    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label=f"Train MSE  (lr={LR:g})", linewidth=1.5)
    plt.plot(val_losses,   label=f"Val MSE  (batch={BATCH_SIZE})", linewidth=1.5)
    if LR_DROP_EPOCH and LR_DROP_EPOCH < len(train_losses):
        plt.axvline(LR_DROP_EPOCH, color="gray", linestyle="--", linewidth=0.8,
                    label=f"LR ×{LR_DROP_FACTOR} @ {LR_DROP_EPOCH}")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nSaved model    -> {model_path}")
    print(f"Saved x_scaler -> {x_scaler_path}")
    print(f"Saved y_scaler -> {y_scaler_path}")
    print(f"Saved plot     -> {plot_path}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    df = load_data(DATA_PATH)

    (X_train, X_val, y_train, y_val,
     x_scaler, y_scaler, n_inputs) = preprocess(df)

    model = CoilSurrogateNN(n_in=n_inputs).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters  |  Input dim: {n_inputs}\n")

    drop_note = (f", lr_drop@{LR_DROP_EPOCH} x{LR_DROP_FACTOR}"
                 if LR_DROP_EPOCH else "")
    print(f"Training {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}{drop_note}\n")
    train_losses, val_losses, best_val_loss, best_epoch = train(
        model, X_train, y_train, X_val, y_val, device)

    final_train = train_losses[-1]
    final_val   = val_losses[-1]
    print(f"\nFinal  — Train: {final_train:.6f}  |  Val: {final_val:.6f}")

    save_artifacts(model, x_scaler, y_scaler, train_losses, val_losses,
                   best_val_loss=best_val_loss, best_epoch=best_epoch)


if __name__ == "__main__":
    main()
