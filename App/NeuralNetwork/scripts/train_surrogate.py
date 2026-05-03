"""
Surrogate Neural Network for PCB Coil Parameter Prediction.

Trains a feed-forward MLP that maps physical coil parameters -> electrical characteristics,
replacing slow FastHenry simulations during optimization.

Inputs include the raw geometry columns plus analytical features
(Wheeler inductance, wire length, ln N^2, mean radius, fill factor, inner
diameter) that are added during the append step.  Topology is one-hot
encoded for both TX and RX.
 
Outputs : L_tx_uH, L_rx_uH, M_uH, R_tx_ac, R_rx_ac
"""

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
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HERE      = os.path.dirname(os.path.abspath(__file__))
_NN_DIR    = os.path.dirname(_HERE)
_SIMDATA   = os.path.join(_NN_DIR, "..", "SimulationData")

# Prefer the new global_results.json; fall back to legacy sweep_results.json
DATA_PATH  = os.environ.get(
    "SURROGATE_DATA",
    os.path.join(_SIMDATA, "global_results.json"),
)
if not os.path.exists(DATA_PATH):
    _legacy = os.path.join(_SIMDATA, "sweep_results.json")
    if os.path.exists(_legacy):
        DATA_PATH = _legacy

# Output directory — set SURROGATE_OUTPUT_DIR from the GUI to the selected model folder.
OUTPUT_DIR = os.environ.get("SURROGATE_OUTPUT_DIR", os.path.join(_NN_DIR, "NN_V2"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters — overridable via environment variables (set by the GUI)
EPOCHS      = int(os.environ.get("SURROGATE_EPOCHS",     "800"))
BATCH_SIZE  = int(os.environ.get("SURROGATE_BATCH_SIZE", "128"))
LR          = float(os.environ.get("SURROGATE_LR",       "1e-3"))
VAL_SPLIT   = float(os.environ.get("SURROGATE_VAL_SPLIT","0.20"))
RANDOM_SEED = 42
PRINT_EVERY = max(1, EPOCHS // 16)
 
# Raw geometry columns pulled directly from each result row.
BASE_INPUT_COLS = [
    "tx_turns", "tx_width", "tx_od_mm",
    "rx_od_mm", "rx_turns", "rx_width",
    "freq_hz",
    "tx_spacing_mm", "rx_spacing_mm",
    "tx_outer_gap_mm", "tx_inner_gap_mm",
    "rx_outer_gap_mm", "rx_inner_gap_mm",
    "ground_circle_dia_mm",
]
 
# Analytical features added during append step (one set each for TX and RX).
_FEATURE_SUFFIX = [
    "wire_length_mm", "wheeler_uh", "ln_n_sq",
    "mean_radius_mm", "fill_factor", "inner_diameter_mm",
]
DERIVED_INPUT_COLS = [f"{side}_{k}" for side in ("tx", "rx") for k in _FEATURE_SUFFIX]

OUTPUT_COLS = ["L_tx_uH", "L_rx_uH", "M_uH", "R_tx_ac", "R_rx_ac"]

TX_TOPOLOGY_COL = "tx_topology"
RX_TOPOLOGY_COL = "rx_topology"
 
# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        raw = json.load(f)

    records = raw["results"]
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} successful simulation records from {path}")
    return df


# ---------------------------------------------------------------------------
# 2. Pre-process
# ---------------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame, cols, default=0.0) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = default
        else:
            df[c] = df[c].fillna(default)
 

def preprocess(df: pd.DataFrame):
    # Tolerate legacy datasets that may lack some columns
    _ensure_columns(df, BASE_INPUT_COLS + DERIVED_INPUT_COLS, default=0.0)
    if TX_TOPOLOGY_COL not in df.columns:
        df[TX_TOPOLOGY_COL] = "parallel"
    if RX_TOPOLOGY_COL not in df.columns:
        df[RX_TOPOLOGY_COL] = "parallel"
 
    tx_dummies = pd.get_dummies(df[TX_TOPOLOGY_COL], prefix="tx_topo")
    rx_dummies = pd.get_dummies(df[RX_TOPOLOGY_COL], prefix="rx_topo")
 
    X = pd.concat(
        [df[BASE_INPUT_COLS + DERIVED_INPUT_COLS], tx_dummies, rx_dummies],
        axis=1,
    ).astype(float)
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

    return (
        X_train_s, X_val_s,
        y_train_s, y_val_s,
        x_scaler, y_scaler,
        X.shape[1],
    )


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

def make_loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=40
    )

    train_losses, val_losses = [], []
    best_val_loss  = float("inf")
    best_state     = None
    best_epoch     = 0

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_loader.dataset)

        # --- validate ---
        model.eval()
        with torch.no_grad():
            running = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                running += criterion(pred, yb).item() * len(xb)
        val_loss = running / len(val_loader.dataset)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{EPOCHS}  |  Train MSE: {train_loss:.6f}  |  Val MSE: {val_loss:.6f}")

    print(f"\nBest val loss  : {best_val_loss:.6f}  (epoch {best_epoch})")
    model.load_state_dict(best_state)
    return train_losses, val_losses


# ---------------------------------------------------------------------------
# 5. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(model, x_scaler, y_scaler, train_losses, val_losses):
    model_path   = os.path.join(OUTPUT_DIR, "surrogate_model.pth")
    x_scaler_path = os.path.join(OUTPUT_DIR, "x_scaler.pkl")
    y_scaler_path = os.path.join(OUTPUT_DIR, "y_scaler.pkl")
    plot_path    = os.path.join(OUTPUT_DIR, "loss_curve.png")

    torch.save(model.state_dict(), model_path)
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    # Loss curve
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="Train MSE", linewidth=1.5)
    plt.plot(val_losses,   label="Val MSE",   linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Surrogate NN — Training vs Validation Loss")
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

    (X_train, X_val,
     y_train, y_val,
     x_scaler, y_scaler,
     n_inputs) = preprocess(df)

    model = CoilSurrogateNN(n_in=n_inputs).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters  |  Input dim: {n_inputs}\n")

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)

    print(f"Training for {EPOCHS} epochs, batch size {BATCH_SIZE}, lr={LR}\n")
    train_losses, val_losses = train(model, train_loader, val_loader, device)

    final_train = train_losses[-1]
    final_val   = val_losses[-1]
    print(f"\nFinal  — Train: {final_train:.6f}  |  Val: {final_val:.6f}")

    save_artifacts(model, x_scaler, y_scaler, train_losses, val_losses)


if __name__ == "__main__":
    main()
