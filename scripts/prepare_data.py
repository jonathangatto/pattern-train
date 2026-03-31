"""
Prepare and clean the annotation data for training.

Reads the main CSV, filters to rows that have a corresponding audio file in
data/audio/, binarises the 9 pattern columns (0.5 → 1, empty → 0), performs a
song-level stratified train/val/test split, and writes three output files:

    data/metadata_train.csv
    data/metadata_val.csv
    data/metadata_test.csv

Each output file contains: id, moment_secs, song_length_secs, and the 9 label
columns (ANT, SPR, PDX, AGR, ALR, GRF, HRM, SZE, PXY).

Usage:
    python scripts/prepare_data.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
CSV_PATH    = REPO_ROOT / "data" / "dbo-moments-2-live.1774458459.csv"
AUDIO_DIR   = REPO_ROOT / "data" / "audio"
OUT_DIR     = REPO_ROOT / "data"

LABEL_COLS  = ["ANT", "SPR", "PDX", "AGR", "ALR", "GRF", "HRM", "SZE", "PXY"]
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binarise(value) -> int:
    """Convert raw pattern value to binary label (missing/0 → 0, else → 1)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    try:
        return 1 if float(value) >= 0.25 else 0
    except (ValueError, TypeError):
        return 0


def audio_exists(row_id) -> bool:
    """Return True if the audio file for this id is present on disk."""
    mp3 = AUDIO_DIR / f"{row_id}.mp3"
    webm = AUDIO_DIR / f"{row_id}.webm"
    return mp3.exists() or webm.exists()


def audio_path(row_id) -> str:
    """Return the path to the audio file for this id (mp3 preferred)."""
    mp3 = AUDIO_DIR / f"{row_id}.mp3"
    webm = AUDIO_DIR / f"{row_id}.webm"
    if mp3.exists():
        return str(mp3)
    return str(webm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"  Total rows: {len(df)}")

    # Keep only rows with a downloaded audio file
    df = df[df["id"].apply(audio_exists)].copy()
    print(f"  Rows with audio: {len(df)}")

    # Binarise label columns (handle missing/NaN/0.5)
    for col in LABEL_COLS:
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found – filling with 0")
            df[col] = 0
        else:
            df[col] = df[col].apply(binarise)

    # Record audio path
    df["audio_path"] = df["id"].apply(audio_path)

    # Keep only useful columns
    keep_cols = ["id", "moment_secs", "song_length_secs", "audio_path"] + LABEL_COLS
    df = df[keep_cols].reset_index(drop=True)

    # Drop rows where moment_secs is missing or non-numeric
    df["moment_secs"] = pd.to_numeric(df["moment_secs"], errors="coerce")
    df["song_length_secs"] = pd.to_numeric(df["song_length_secs"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["moment_secs"]).reset_index(drop=True)
    print(f"  Rows after dropping missing moment_secs: {len(df)} (dropped {before - len(df)})")

    # Print label distribution
    print("\nLabel distribution (% positive):")
    for col in LABEL_COLS:
        pct = df[col].mean() * 100
        print(f"  {col}: {pct:.1f}%")

    # -----------------------------------------------------------------------
    # Song-level split to prevent leakage
    # We aggregate labels per song id (any positive across clips counts),
    # split at the song level, then reassign clips.
    # -----------------------------------------------------------------------
    song_ids = df["id"].unique()
    song_labels = (
        df.groupby("id")[LABEL_COLS].max().loc[song_ids].values
    )

    # First split off test set
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
    )
    train_val_idx, test_idx = next(msss.split(song_ids, song_labels))
    test_song_ids = song_ids[test_idx]

    # Then split val from remaining train
    trainval_song_ids = song_ids[train_val_idx]
    trainval_labels   = song_labels[train_val_idx]
    val_ratio_adj     = VAL_RATIO / (1.0 - TEST_RATIO)  # adjust for reduced pool

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio_adj,
        random_state=RANDOM_SEED,
    )
    train_idx2, val_idx2 = next(msss2.split(trainval_song_ids, trainval_labels))
    train_song_ids = trainval_song_ids[train_idx2]
    val_song_ids   = trainval_song_ids[val_idx2]

    # Map back to clip-level rows
    df_train = df[df["id"].isin(train_song_ids)].reset_index(drop=True)
    df_val   = df[df["id"].isin(val_song_ids)].reset_index(drop=True)
    df_test  = df[df["id"].isin(test_song_ids)].reset_index(drop=True)

    print(f"\nSplit (clips):  train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")
    print(f"Split (songs):  train={len(train_song_ids)}  val={len(val_song_ids)}  test={len(test_song_ids)}")

    # Save
    train_path = OUT_DIR / "metadata_train.csv"
    val_path   = OUT_DIR / "metadata_val.csv"
    test_path  = OUT_DIR / "metadata_test.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

    # Write a small JSON summary useful for downstream scripts
    summary = {
        "label_cols": LABEL_COLS,
        "n_labels": len(LABEL_COLS),
        "splits": {
            "train": len(df_train),
            "val":   len(df_val),
            "test":  len(df_test),
        },
    }
    summary_path = OUT_DIR / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
