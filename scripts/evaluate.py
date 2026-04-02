"""
Evaluate the trained classifier on the held-out test set.

Loads models/classifier/best_model.pt and data/embeddings_test.pt and
reports per-pattern and aggregate metrics to stdout and to
models/classifier/evaluation_results.json.

Usage:
    python scripts/evaluate.py [--threshold 0.5] [--device cpu]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model
from split_utils import load_embeddings, holdout_split
from config import MODEL_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate classifier on test set")
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold (default: value from config.json)")
    p.add_argument("--val-ratio",  type=float, default=0.10,
                   help="Validation ratio used during training (default: 0.10)")
    p.add_argument("--test-ratio", type=float, default=0.10,
                   help="Test ratio used during training (default: 0.10)")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed for splitting (default: 42)")
    p.add_argument("--device",    type=str,   default="cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    # Load config
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found – run train_classifier.py first")
    with open(config_path) as f:
        config = json.load(f)

    label_cols = config["label_cols"]
    n_labels   = config["n_labels"]
    in_dim     = config["in_dim"]
    threshold  = args.threshold if args.threshold is not None else config["threshold"]

    print(f"Model:     {config['model_name']}")
    print(f"Labels ({n_labels}): {label_cols}")
    print(f"Threshold: {threshold}")

    # Load model
    model = build_model(config["model_name"], in_dim=in_dim, n_labels=n_labels)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    # Load unified embeddings and extract test set via holdout split
    data = load_embeddings()
    _, _, test_idx = holdout_split(
        data["urls"], data["labels"],
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
        tastes_ids=data.get("tastes_ids"))
    X_test = data["frames"][test_idx].float().to(device)
    y_test = data["labels"][test_idx].float()

    print(f"\nTest clips: {X_test.shape[0]}")

    # Inference
    with torch.no_grad():
        logits = model(X_test).cpu()

    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)
    y     = y_test.numpy().astype(int)

    # Aggregate metrics
    macro_f1   = f1_score(y, preds, average="macro",  zero_division=0)
    micro_f1   = f1_score(y, preds, average="micro",  zero_division=0)
    macro_prec = precision_score(y, preds, average="macro",  zero_division=0)
    macro_rec  = recall_score(y, preds, average="macro",  zero_division=0)
    h_loss     = hamming_loss(y, preds)

    try:
        roc_auc = roc_auc_score(y, probs, average="macro")
    except ValueError:
        roc_auc = float("nan")

    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 36)
    print(f"{'Macro F1':<25} {macro_f1:>10.4f}")
    print(f"{'Micro F1':<25} {micro_f1:>10.4f}")
    print(f"{'Macro Precision':<25} {macro_prec:>10.4f}")
    print(f"{'Macro Recall':<25} {macro_rec:>10.4f}")
    print(f"{'Hamming Loss':<25} {h_loss:>10.4f}")
    print(f"{'Macro ROC-AUC':<25} {roc_auc:>10.4f}")
    print("=" * 60)

    # Per-class metrics
    per_f1   = f1_score(y, preds, average=None, zero_division=0)
    per_prec = precision_score(y, preds, average=None, zero_division=0)
    per_rec  = recall_score(y, preds, average=None, zero_division=0)
    per_pos  = y.sum(axis=0)

    print(f"\n{'Label':<8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Support':>9}")
    print("-" * 48)
    for name, f1, prec, rec, sup in zip(label_cols, per_f1, per_prec, per_rec, per_pos):
        print(f"{name:<8} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {sup:>9}")

    # Full sklearn classification report (text)
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=label_cols, zero_division=0))

    # Save results
    results = {
        "threshold": threshold,
        "aggregate": {
            "macro_f1":   macro_f1,
            "micro_f1":   micro_f1,
            "macro_prec": macro_prec,
            "macro_rec":  macro_rec,
            "hamming_loss": h_loss,
            "macro_roc_auc": roc_auc if not np.isnan(roc_auc) else None,
        },
        "per_label": {
            name: {"f1": float(f1), "precision": float(prec), "recall": float(rec), "support": int(sup)}
            for name, f1, prec, rec, sup in zip(label_cols, per_f1, per_prec, per_rec, per_pos)
        },
    }
    out_path = MODEL_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
