"""
Train a pattern classifier on top of precomputed MERT frame embeddings.

Select the model architecture with --model (default: cnn1d).
Available models are defined in models/ at the repo root.
    cnn1d – 1D residual CNN over temporal frames (default, recommended)
    mlp   – simple MLP with mean+max pooling (fast baseline)

Outputs (written to models/classifier/):
    best_model.pt      – state dict of the best validation checkpoint
    config.json        – hyperparams and label metadata
    training_log.csv   – per-epoch metrics

Usage:
    python scripts/train_classifier.py [options]

Requires:
    extract_embeddings.py to have been run first
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score

# Make the repo root importable so `models/` can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import build_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR     = REPO_ROOT / "data"
MODEL_DIR    = REPO_ROOT / "models" / "classifier"
SUMMARY_PATH = DATA_DIR / "dataset_summary.json"

LABEL_NAMES  = ["ANT", "SPR", "PDX", "AGR", "ALR", "GRF", "HRM", "SZE", "PXY"]

DEFAULT_MODEL      = "cnn1d"
DEFAULT_EPOCHS     = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR         = 1e-3
DEFAULT_WD         = 1e-4
DEFAULT_DROPOUT    = 0.3
DEFAULT_PATIENCE   = 10
DEFAULT_THRESHOLD  = 0.5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float, label_names: list[str]):
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)
    y     = labels.numpy().astype(int)

    macro_f1    = f1_score(y, preds, average="macro",  zero_division=0)
    micro_f1    = f1_score(y, preds, average="micro",  zero_division=0)
    macro_prec  = precision_score(y, preds, average="macro",  zero_division=0)
    macro_rec   = recall_score(y, preds, average="macro",  zero_division=0)
    per_class   = f1_score(y, preds, average=None, zero_division=0)

    metrics = {
        "macro_f1":   macro_f1,
        "micro_f1":   micro_f1,
        "macro_prec": macro_prec,
        "macro_rec":  macro_rec,
    }
    for name, score in zip(label_names, per_class):
        metrics[f"f1_{name}"] = score
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args) -> None:
    device = torch.device(args.device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset summary for label info
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    label_cols  = summary["label_cols"]
    n_labels    = len(label_cols)

    # Load precomputed frame embeddings
    print("Loading embeddings...")
    train_data = torch.load(DATA_DIR / "embeddings_train.pt", weights_only=False)
    val_data   = torch.load(DATA_DIR / "embeddings_val.pt",   weights_only=False)

    X_train, y_train = train_data["frames"].float(), train_data["labels"].float()
    X_val,   y_val   = val_data["frames"].float(),   val_data["labels"].float()
    # X: (N, T, D) — in_dim is the MERT hidden dim D
    in_dim = X_train.shape[2]

    print(f"  Train: {X_train.shape[0]} clips, frames={X_train.shape[1]}, in_dim={in_dim}")
    print(f"  Val:   {X_val.shape[0]} clips")

    # Compute pos_weight per label to handle class imbalance
    pos_weight = (y_train.shape[0] - y_train.sum(0)) / (y_train.sum(0).clamp(min=1))
    pos_weight = pos_weight.to(device)

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=(device.type=="cuda"))

    # Model, loss, optimiser, scheduler
    model     = build_model(args.model, in_dim=in_dim, n_labels=n_labels, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Training on: {device}\n")

    # Log file
    log_path = MODEL_DIR / "training_log.csv"
    log_header = ["epoch", "train_loss", "val_loss", "macro_f1", "micro_f1"] + \
                 [f"f1_{c}" for c in label_cols]

    best_val_f1  = -1.0
    patience_ctr = 0

    with open(log_path, "w", newline="") as lf:
        writer = csv.writer(lf)
        writer.writerow(log_header)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            # --- Train ---
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimiser.zero_grad()
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_ds)

            # --- Validate ---
            model.eval()
            val_loss   = 0.0
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    val_loss += criterion(logits, y_batch).item() * X_batch.size(0)
                    all_logits.append(logits.cpu())
                    all_labels.append(y_batch.cpu())
            val_loss  /= len(val_ds)
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)

            metrics = compute_metrics(all_logits, all_labels, args.threshold, label_cols)
            scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"macro_f1={metrics['macro_f1']:.4f}  micro_f1={metrics['micro_f1']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # Log
            row = [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                   f"{metrics['macro_f1']:.6f}", f"{metrics['micro_f1']:.6f}"]
            row += [f"{metrics[f'f1_{c}']:.6f}" for c in label_cols]
            writer.writerow(row)
            lf.flush()

            # Best model checkpoint
            if metrics["macro_f1"] > best_val_f1:
                best_val_f1 = metrics["macro_f1"]
                patience_ctr = 0
                torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
                print(f"  ✓ New best macro_f1={best_val_f1:.4f} – saved checkpoint")
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
                    break

    # Save config
    config = {
        "mert_model":   "m-a-p/MERT-v1-95M",
        "model_name":   args.model,
        "in_dim":       in_dim,
        "n_labels":     n_labels,
        "label_cols":   label_cols,
        "threshold":    args.threshold,
        "best_val_macro_f1": best_val_f1,
        "hyperparams": {
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "lr":         args.lr,
            "wd":         args.wd,
            "dropout":    args.dropout,
        },
    }
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete. Best val macro_f1: {best_val_f1:.4f}")
    print(f"Checkpoint: {MODEL_DIR / 'best_model.pt'}")
    print(f"Config:     {MODEL_DIR / 'config.json'}")
    print(f"Log:        {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a pattern classifier on MERT embeddings")
    p.add_argument("--model",       type=str,   default=DEFAULT_MODEL,
                   help="Model architecture: cnn1d (default) or mlp")
    p.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=DEFAULT_LR)
    p.add_argument("--wd",          type=float, default=DEFAULT_WD)
    p.add_argument("--dropout",     type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--patience",    type=int,   default=DEFAULT_PATIENCE)
    p.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
