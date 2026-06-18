"""
Aggregate cross-validation results from per-fold training logs.

The training script (train_classifier.py) prints a CV summary to the console but
does not persist it. However, every fold writes a per-epoch training_log.csv:

    models/classifier/<submits>/pattern_<X>/fold{1..N}/training_log.csv   (patterns)
    models/classifier/<submits>/fold{1..N}/training_log.csv               (frisson)

The per-fold validation AUROC reported in the printed summary is the best (max)
macro_auroc across that fold's epochs (the checkpoint is tracked on best AUROC).
This script reconstructs that summary into a single shareable CSV with one row
per model: each fold's best AUROC plus the mean and std across folds.

Usage:
    python scripts/aggregate_cv_results.py [--submits-dir DIR] [--output CSV]
"""

import argparse
import csv
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMITS_DIR = REPO_ROOT / "models" / "classifier" / "min_submits_90"


def best_auroc(fold_dir: Path) -> float | None:
    """Return the max macro_auroc across epochs in a fold's training_log.csv."""
    log = fold_dir / "training_log.csv"
    if not log.exists():
        return None
    best = None
    with open(log, newline="") as f:
        for row in csv.DictReader(f):
            try:
                a = float(row["macro_auroc"])
            except (KeyError, ValueError):
                continue
            if a != a:  # NaN
                continue
            best = a if best is None else max(best, a)
    return best


def fold_aurocs(model_dir: Path) -> list[float | None]:
    """Best AUROC per fold (sorted fold1, fold2, ...). None for missing folds."""
    fold_dirs = sorted(
        (d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("fold")),
        key=lambda d: int(d.name.replace("fold", "") or 0),
    )
    return [best_auroc(d) for d in fold_dirs]


def discover_models(submits_dir: Path) -> list[tuple[str, Path]]:
    """Return [(model_name, model_dir)] for every pattern and the frisson model."""
    models: list[tuple[str, Path]] = []
    for d in sorted(submits_dir.glob("pattern_*")):
        if d.is_dir():
            models.append((d.name.replace("pattern_", ""), d))
    # The general frisson model lives directly under submits_dir (its own folds).
    if any(submits_dir.glob("fold*")):
        models.append(("frisson", submits_dir))
    return models


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate per-fold CV AUROC into one CSV")
    p.add_argument("--submits-dir", type=str, default=str(DEFAULT_SUBMITS_DIR),
                   help="Directory holding pattern_<X>/ and frisson folds")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV (default: <submits-dir>/cv_summary.csv)")
    args = p.parse_args()

    submits_dir = Path(args.submits_dir)
    if not submits_dir.exists():
        raise SystemExit(f"ERROR: not found: {submits_dir}")
    output = Path(args.output) if args.output else submits_dir / "cv_summary.csv"

    models = discover_models(submits_dir)
    if not models:
        raise SystemExit(f"ERROR: no models found under {submits_dir}")

    n_folds = max(len(fold_aurocs(d)) for _, d in models)
    header = ["model"] + [f"fold{i + 1}_auroc" for i in range(n_folds)] + \
             ["mean_auroc", "std_auroc", "n_folds"]

    rows = []
    print(f"{'model':<12} " + " ".join(f"f{i+1}" for i in range(n_folds)) +
          "   mean ± std")
    for name, model_dir in models:
        aurocs = fold_aurocs(model_dir)
        present = [a for a in aurocs if a is not None]
        mean = statistics.mean(present) if present else float("nan")
        std = statistics.pstdev(present) if len(present) > 1 else 0.0
        cells = ["" if a is None else f"{a:.6f}" for a in aurocs]
        cells += [""] * (n_folds - len(cells))
        rows.append([name] + cells + [f"{mean:.6f}", f"{std:.6f}", str(len(present))])

        printed = " ".join("  -  " if a is None else f"{a:.3f}" for a in aurocs)
        printed += "  " * (n_folds - len(aurocs))
        print(f"{name:<12} {printed}   {mean:.4f} ± {std:.4f}")

    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"\nWrote {len(rows)} models → {output}")


if __name__ == "__main__":
    main()
