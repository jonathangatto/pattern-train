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
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Make the repo root and scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model
from split_utils import load_embeddings, holdout_split, kfold_cv
from config import MERT_MODEL, MODEL_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR     = REPO_ROOT / "data"
SUMMARY_PATH = DATA_DIR / "dataset_summary.json"

LABEL_NAMES  = ["ANT", "SPR", "PDX", "AGR", "ALR", "GRF", "HRM", "SZE", "PXY"]

# Set to True to use a single linear layer (mean+max pool → linear) instead of
# the full CNN/MLP — useful to verify the embeddings carry learnable signal.
SIMPLE_MODE = False

DEFAULT_MODEL      = "cnn1d"
DEFAULT_EPOCHS     = 100
DEFAULT_BATCH_SIZE = 2048
DEFAULT_LR         = 1e-5
DEFAULT_WD         = 1e-4
DEFAULT_DROPOUT    = 0.4
DEFAULT_PATIENCE   = 100
DEFAULT_THRESHOLD  = 0.5
DEFAULT_TINY_SIZE  = 256
DEFAULT_TINY_LR    = 3e-3
DEFAULT_TINY_INNER_STEPS = 20


# ---------------------------------------------------------------------------
# Simple baseline classifier (used when SIMPLE_MODE = True)
# ---------------------------------------------------------------------------

class _SimpleClassifier(nn.Module):
    """Mean+max pool over T frames → single linear layer. Sanity-check baseline."""
    def __init__(self, in_dim: int, n_labels: int, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_dim * 2, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(1), x.max(1).values], dim=1)  # (B, 2D)
        return self.fc(pooled)


class _TinyMemorizer(nn.Module):
    """Nonlinear memorizer on pooled features for tiny-overfit diagnostics."""
    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        pooled_dim = in_dim * 2  # mean + max
        self.net = nn.Sequential(
            nn.Linear(pooled_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(1), x.max(1).values], dim=1)  # (B, 2D)
        return self.net(pooled)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float, label_names: list[str]):
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)
    y     = labels.numpy().astype(int)

    is_binary = (logits.shape[1] == 1)

    if is_binary:
        probs_1d = probs[:, 0]
        preds_1d = preds[:, 0]
        y_1d     = y[:, 0]
        macro_f1   = f1_score(y_1d, preds_1d, zero_division=0)
        micro_f1   = macro_f1
        macro_prec = precision_score(y_1d, preds_1d, zero_division=0)
        macro_rec  = recall_score(y_1d, preds_1d, zero_division=0)
        try:
            macro_auroc = roc_auc_score(y_1d, probs_1d)
        except ValueError:
            macro_auroc = float("nan")
        per_class = [macro_f1]
    else:
        macro_f1    = f1_score(y, preds, average="macro",  zero_division=0)
        micro_f1    = f1_score(y, preds, average="micro",  zero_division=0)
        macro_prec  = precision_score(y, preds, average="macro",  zero_division=0)
        macro_rec   = recall_score(y, preds, average="macro",  zero_division=0)
        per_class   = f1_score(y, preds, average=None, zero_division=0)
        # AUROC is threshold-independent: computed on raw probabilities.
        # May be undefined if a class has only one label value in the batch.
        try:
            macro_auroc = roc_auc_score(y, probs, average="macro")
        except ValueError:
            macro_auroc = float("nan")

    metrics = {
        "macro_auroc": macro_auroc,
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

def _run_training(X_all, y_all, train_idx, val_idx, args, label_cols, n_labels, output_dir, device):
    """Run a single training pass. Returns best validation AUROC."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    val_idx = torch.as_tensor(val_idx, dtype=torch.long)

    # Tiny overfit mode: train and validate on the same tiny subset.
    # Useful to quickly verify the model/pipeline can fit anything at all.
    if args.tiny_overfit:
        tiny_n = min(args.tiny_size, train_idx.shape[0])
        gen = torch.Generator().manual_seed(42)
        tiny_select = torch.randperm(train_idx.shape[0], generator=gen)[:tiny_n]
        train_idx = train_idx[tiny_select]
        val_idx = train_idx.clone()
        print(f"[TINY OVERFIT MODE] Using same {tiny_n} samples for train and val.")
        print(f"[TINY OVERFIT MODE] Positive rate: {y_all[train_idx].mean().item():.4f}")

    # X: (N, T, D) — in_dim is the MERT hidden dim D
    in_dim = X_all.shape[2]

    print(f"  Train: {train_idx.shape[0]} clips, frames={X_all.shape[1]}, in_dim={in_dim}")
    print(f"  Val:   {val_idx.shape[0]} clips")

    # Compute pos_weight per label to handle class imbalance.
    # In binary mode, force pos_weight=1.0 to keep the objective simple/stable.
    if args.binary:
        pos_weight = torch.ones(1, device=device)
        print("[BINARY MODE] Using pos_weight=1.0")
    else:
        y_train = y_all[train_idx]
        pos_weight = (y_train.shape[0] - y_train.sum(0)) / (y_train.sum(0).clamp(min=1))
        pos_weight = pos_weight.clamp(max=10.0).to(device)

    # Slice tensors for this split (fast in-RAM copy)
    X_train, y_train_ds = X_all[train_idx], y_all[train_idx]
    X_val,   y_val_ds   = X_all[val_idx],   y_all[val_idx]
    train_ds = TensorDataset(X_train, y_train_ds)
    val_ds   = TensorDataset(X_val, y_val_ds)
    if args.tiny_overfit:
        train_batch_size = len(train_ds)
        val_batch_size   = len(val_ds)
        train_shuffle = False
    else:
        train_batch_size = args.batch_size
        val_batch_size   = args.batch_size
        train_shuffle = True
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=train_shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=val_batch_size, shuffle=False)

    # Model, loss, optimiser
    if args.tiny_overfit:
        model = _TinyMemorizer(in_dim=in_dim, n_labels=n_labels).to(device)
        print("[TINY OVERFIT MODE] Using nonlinear memorizer classifier.")
    elif SIMPLE_MODE:
        model = _SimpleClassifier(in_dim=in_dim, n_labels=n_labels).to(device)
        print("[SIMPLE_MODE] Using single linear layer classifier.")
    else:
        model = build_model(args.model, in_dim=in_dim, n_labels=n_labels, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.tiny_overfit:
        tiny_lr = max(args.lr, DEFAULT_TINY_LR)
        optimiser = torch.optim.Adam(model.parameters(), lr=tiny_lr, weight_decay=0.0)
        print(f"[TINY OVERFIT MODE] Optimiser: Adam(lr={tiny_lr}, wd=0.0)")
    else:
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Training on: {device}\n")

    # Log file
    log_path = output_dir / "training_log.csv"
    log_header = ["epoch", "train_loss", "val_loss", "macro_auroc", "macro_f1", "micro_f1"] + \
                 [f"f1_{c}" for c in label_cols]

    best_val_f1  = -1.0
    best_epoch   = 0
    patience_ctr = 0

    with open(log_path, "w", newline="") as lf:
        writer = csv.writer(lf)
        writer.writerow(log_header)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            # --- Train ---
            model.train()
            train_loss = 0.0
            inner_steps = args.tiny_inner_steps if args.tiny_overfit else 1
            for _ in range(inner_steps):
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    optimiser.zero_grad()
                    logits = model(X_batch)
                    loss   = criterion(logits, y_batch)
                    loss.backward()
                    if not args.tiny_overfit:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimiser.step()
                    train_loss += loss.item() * X_batch.size(0)
            train_loss /= (len(train_ds) * inner_steps)

            # --- Validate ---
            model.eval()
            val_loss   = 0.0
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    logits = model(X_batch)
                    val_loss += criterion(logits, y_batch).item() * X_batch.size(0)
                    all_logits.append(logits.cpu())
                    all_labels.append(y_batch.cpu())
            val_loss  /= len(val_ds)
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)

            metrics = compute_metrics(all_logits, all_labels, args.threshold, label_cols)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"auroc={metrics['macro_auroc']:.4f}  "
                f"macro_f1={metrics['macro_f1']:.4f}  micro_f1={metrics['micro_f1']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # Log
            row = [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                   f"{metrics['macro_auroc']:.6f}", f"{metrics['macro_f1']:.6f}", f"{metrics['micro_f1']:.6f}"]
            row += [f"{metrics[f'f1_{c}']:.6f}" for c in label_cols]
            writer.writerow(row)
            lf.flush()

            # Best model checkpoint — tracked on AUROC (threshold-independent)
            if metrics["macro_auroc"] > best_val_f1:
                best_val_f1 = metrics["macro_auroc"]
                best_epoch   = epoch
                patience_ctr = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print(f"  ✓ New best macro_auroc={best_val_f1:.4f} – saved checkpoint")
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
                    break

    # Save config
    config = {
        "mert_model":   MERT_MODEL,
        "model_name":   args.model,
        "in_dim":       in_dim,
        "n_labels":     n_labels,
        "label_cols":   label_cols,
        "binary_mode":  args.binary,
        "tiny_overfit_mode": args.tiny_overfit,
        "tiny_size": args.tiny_size,
        "tiny_lr_floor": DEFAULT_TINY_LR,
        "tiny_inner_steps": args.tiny_inner_steps,
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
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete. Best val macro_auroc: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"Checkpoint: {output_dir / 'best_model.pt'}")
    print(f"Config:     {output_dir / 'config.json'}")
    print(f"Log:        {log_path}")

    return best_val_f1, best_epoch


def train(args) -> None:
    device = torch.device(args.device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset summary for label info
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    label_cols  = summary["label_cols"]
    n_labels    = len(label_cols)

    # Load unified embeddings (splitting happens here, not at extraction time)
    print("Loading embeddings...")
    data = load_embeddings()
    X_all = data["frames"].float()
    y_all = data["labels"].float()
    urls  = data["urls"]
    tastes_ids = data.get("tastes_ids")

    # Binary mode: collapse all 9 labels into a single presence/absence flag.
    if args.binary:
        y_all = (y_all.sum(dim=1, keepdim=True) > 0).float()
        label_cols = ["any_pattern"]
        n_labels   = 1
        print("[BINARY MODE] Predicting any-pattern-present (1) vs. none (0).")

    if args.split_strategy == "holdout":
        train_idx, val_idx, _ = holdout_split(
            urls, y_all, val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, seed=args.seed,
            tastes_ids=tastes_ids)
        print(f"Holdout split → train: {len(train_idx)}  val: {len(val_idx)}")
        _run_training(X_all, y_all, train_idx, val_idx,
                      args, label_cols, n_labels, MODEL_DIR, device)

    elif args.split_strategy == "kfold":
        fold_aurocs = []
        fold_best_epochs = []
        best_overall_auroc = -1.0
        best_fold = -1

        for fold, (train_idx, val_idx) in enumerate(
                kfold_cv(urls, y_all, n_folds=args.n_folds, seed=args.seed,
                         tastes_ids=tastes_ids)):
            fold_name = f"fold{fold + 1}"
            fold_dir  = MODEL_DIR / fold_name
            print(f"\n{'=' * 60}")
            print(f"  {fold_name} — train: {len(train_idx)}  val: {len(val_idx)}")
            print(f"{'=' * 60}")

            auroc, best_ep = _run_training(
                X_all, y_all, train_idx, val_idx,
                args, label_cols, n_labels, fold_dir, device)
            fold_aurocs.append(auroc)
            fold_best_epochs.append(best_ep)
            if auroc > best_overall_auroc:
                best_overall_auroc = auroc
                best_fold = fold + 1

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Summary
        print(f"\n{'=' * 60}")
        print(f"Cross-validation summary ({args.n_folds} folds):")
        for i, (auroc, ep) in enumerate(zip(fold_aurocs, fold_best_epochs)):
            marker = " ← best" if (i + 1) == best_fold else ""
            print(f"  Fold {i + 1}: AUROC = {auroc:.4f}  (best epoch {ep}){marker}")
        print(f"  Mean ± Std: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
        print(f"{'=' * 60}")

        # --- Final retrain on (almost) all data ---
        # Now that CV has validated the hyperparameters, retrain on as much
        # data as possible.  A small 5% holdout is kept purely for early
        # stopping; the rest is used for training.
        print(f"\n{'=' * 60}")
        print(f"  Final retrain — using ~95% of all data")
        print(f"{'=' * 60}")

        # Get a 5% stratified val set; merge the rest into train.
        # Use a different seed so the tiny val set isn't correlated with
        # the CV folds.
        final_train_idx, final_val_idx, final_leftover = holdout_split(
            urls, y_all,
            val_ratio=0.05, test_ratio=0.05,
            seed=args.seed + 1000,
            tastes_ids=tastes_ids)
        # Merge the leftover (test) slice back into training
        final_train_idx = np.concatenate([final_train_idx, final_leftover])

        _run_training(
            X_all, y_all, final_train_idx, final_val_idx,
            args, label_cols, n_labels, MODEL_DIR, device)

        print(f"\nFinal model saved to {MODEL_DIR}")


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
    p.add_argument("--tiny-overfit", action="store_true",
                   help="Use a tiny train subset as both train and val to sanity-check learnability")
    p.add_argument("--tiny-size",   type=int,   default=DEFAULT_TINY_SIZE,
                   help="Number of samples for tiny-overfit mode")
    p.add_argument("--tiny-inner-steps", type=int, default=DEFAULT_TINY_INNER_STEPS,
                   help="Number of optimization passes over the tiny set per epoch")
    p.add_argument("--binary",      action="store_true",
                   help="Collapse all labels to a single any-pattern-present binary target")
    p.add_argument("--split-strategy", type=str, default="holdout",
                   choices=["holdout", "kfold"],
                   help="Split strategy: holdout (default) or kfold cross-validation")
    p.add_argument("--n-folds",     type=int,   default=5,
                   help="Number of folds for kfold strategy (default: 5)")
    p.add_argument("--val-ratio",   type=float, default=0.10,
                   help="Validation ratio for holdout strategy (default: 0.10)")
    p.add_argument("--test-ratio",  type=float, default=0.10,
                   help="Test ratio for holdout strategy (default: 0.10)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for splitting (default: 42)")
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
