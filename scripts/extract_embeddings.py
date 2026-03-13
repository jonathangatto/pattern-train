"""
Extract MERT embeddings for all clips and save them to disk.

Reads dbo-moments-2-live.1772538551.csv directly, filters rows to those
with a downloaded audio file, binarises the 9 pattern label columns,
performs a song-level stratified train/val/test split, then runs the
frozen MERT-v1-95M encoder on each clip and saves the raw frame
embeddings (T × 768) — no pooling — so downstream models can apply their
own temporal operations.

Each clip is 3 seconds: 1 second before the annotated moment, the
annotated second itself, and 1 second after. Boundaries are zero-padded.

Outputs:
    data/embeddings_train.pt   → {"frames": Tensor(N,T,768), "labels": Tensor(N,9), "ids": list}
    data/embeddings_val.pt
    data/embeddings_test.pt
    data/dataset_summary.json  → label metadata used by training scripts

Usage:
    python scripts/extract_embeddings.py [--batch-size 16] [--device cuda]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR     = REPO_ROOT / "data"
CSV_PATH     = DATA_DIR / "dbo-moments-2-live.1772538551.csv"
AUDIO_DIR    = DATA_DIR / "audio"
SUMMARY_PATH = DATA_DIR / "dataset_summary.json"

MERT_MODEL      = "m-a-p/MERT-v1-95M"
SAMPLE_RATE     = 24_000   # MERT's native sample rate
EMBED_DIM       = 768      # MERT hidden size per frame

# Clip window: [moment_secs - CONTEXT_BEFORE, moment_secs + ANNOT_SECONDS + CONTEXT_AFTER]
ANNOT_SECONDS   = 1.0      # the annotated second
CONTEXT_BEFORE  = 1.0      # seconds of context before the annotation
CONTEXT_AFTER   = 1.0      # seconds of context after the annotation
CLIP_SECONDS    = CONTEXT_BEFORE + ANNOT_SECONDS + CONTEXT_AFTER  # 3.0 s total

LABEL_COLS   = ["ANT", "SPR", "PDX", "AGR", "ALR", "GRF", "HRM", "SZE", "PXY"]
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# Data preparation (filtering, binarisation, splitting)
# ---------------------------------------------------------------------------

def _binarise(value) -> int:
    """Convert raw pattern value to binary label (missing/0 → 0, else → 1)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    try:
        return 1 if float(value) >= 0.25 else 0
    except (ValueError, TypeError):
        return 0


def _audio_path(row_id) -> str | None:
    """Return path to audio file for this id, or None if not present."""
    for ext in ("mp3", "webm"):
        p = AUDIO_DIR / f"{row_id}.{ext}"
        if p.exists():
            return str(p)
    return None


def build_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load CSV, filter, binarise labels, split by song id.

    Returns (df_train, df_val, df_test, label_cols).
    """
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False, on_bad_lines='skip')
    print(f"  Total rows: {len(df)}")

    # Filter to rows with a downloaded audio file
    df["audio_path"] = df["id"].apply(_audio_path)
    df = df[df["audio_path"].notna()].copy()
    print(f"  Rows with audio: {len(df)}")

    # Binarise label columns
    for col in LABEL_COLS:
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found in CSV – filling with 0")
            df[col] = 0
        else:
            df[col] = df[col].apply(_binarise)

    # Keep only useful columns
    df["moment_secs"]      = pd.to_numeric(df["moment_secs"],      errors="coerce")
    df["song_length_secs"] = pd.to_numeric(df["song_length_secs"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["moment_secs"]).reset_index(drop=True)
    if before != len(df):
        print(f"  Dropped {before - len(df)} rows with missing moment_secs")

    df = df[["id", "moment_secs", "song_length_secs", "audio_path"] + LABEL_COLS]

    print("\nLabel distribution (% positive):")
    for col in LABEL_COLS:
        print(f"  {col}: {df[col].mean() * 100:.1f}%")

    # Song-level stratified split to avoid leakage
    song_ids    = df["id"].unique()
    song_labels = df.groupby("id")[LABEL_COLS].max().loc[song_ids].values

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(msss.split(song_ids, song_labels))

    trainval_ids    = song_ids[train_val_idx]
    trainval_labels = song_labels[train_val_idx]
    val_ratio_adj   = VAL_RATIO / (1.0 - TEST_RATIO)

    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adj, random_state=RANDOM_SEED)
    train_idx2, val_idx2 = next(msss2.split(trainval_ids, trainval_labels))

    df_train = df[df["id"].isin(trainval_ids[train_idx2])].reset_index(drop=True)
    df_val   = df[df["id"].isin(trainval_ids[val_idx2])].reset_index(drop=True)
    df_test  = df[df["id"].isin(song_ids[test_idx])].reset_index(drop=True)

    print(f"\nSplit → train: {len(df_train)}  val: {len(df_val)}  test: {len(df_test)} clips")
    return df_train, df_val, df_test, LABEL_COLS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClipDataset(Dataset):
    """Loads 3-sec clips on-the-fly: 1s before + annotated second + 1s after."""

    def __init__(self, meta_df: pd.DataFrame, label_cols: list[str]):
        self.meta         = meta_df.reset_index(drop=True)
        self.label_cols   = label_cols
        self.pre_samples  = int(CONTEXT_BEFORE * SAMPLE_RATE)
        self.target_len   = int(CLIP_SECONDS   * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row    = self.meta.iloc[idx]
        row_id = int(row["id"])
        moment = float(row["moment_secs"])
        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)

        try:
            waveform, sr = torchaudio.load(row["audio_path"])
        except Exception:
            return torch.zeros(self.target_len), labels, row_id

        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (samples,)

        # Resample to MERT native rate
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        # Desired window: [moment - CONTEXT_BEFORE, moment + ANNOT_SECONDS + CONTEXT_AFTER]
        annot_sample = int(moment * SAMPLE_RATE)
        start        = annot_sample - self.pre_samples
        end          = start + self.target_len

        # If window runs past the end, shift it left (but keep full length)
        if end > len(waveform):
            end   = len(waveform)
            start = end - self.target_len

        # Build clip with zero-padding for any out-of-bounds region
        if start >= 0:
            clip = waveform[start:end]
        else:
            # start is negative: pad the front
            pad_front = torch.zeros(-start)
            clip      = torch.cat([pad_front, waveform[0:end]])

        # Zero-pad tail if clip is still short (very short files)
        if clip.shape[0] < self.target_len:
            clip = torch.cat([clip, torch.zeros(self.target_len - clip.shape[0])])

        return clip, labels, row_id


def collate_fn(batch):
    clips  = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    ids    = [b[2] for b in batch]
    return clips, labels, ids


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_split(
    meta_df      : pd.DataFrame,
    label_cols   : list[str],
    processor    ,
    model        ,
    out_path     : Path,
    batch_size   : int,
    device       : torch.device,
) -> None:
    dataset = ClipDataset(meta_df, label_cols)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    all_embeddings = []
    all_labels     = []
    all_ids        = []

    for clips, labels, ids in tqdm(loader, desc=f"  → {out_path.name}"):
        # processor expects list of numpy arrays or a batched numpy array
        inputs = processor(
            clips.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)

        outputs = model(input_values, output_hidden_states=False)
        # last_hidden_state: (B, T, 768) — keep raw frames, no pooling
        hidden = outputs.last_hidden_state.cpu()  # (B, T, D)

        all_embeddings.append(hidden)
        all_labels.append(labels)
        all_ids.extend(ids)

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    labels_tensor     = torch.cat(all_labels, dim=0)

    torch.save(
        {"frames": embeddings_tensor, "labels": labels_tensor, "ids": all_ids},
        out_path,
    )
    print(f"  Saved {embeddings_tensor.shape[0]} clips (shape {tuple(embeddings_tensor.shape)}) to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract MERT embeddings for all splits")
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Build splits directly from the original CSV
    df_train, df_val, df_test, label_cols = build_splits()

    # Write dataset_summary.json so training scripts can read label metadata
    summary = {
        "label_cols": label_cols,
        "n_labels":   len(label_cols),
        "splits":     {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved dataset summary: {SUMMARY_PATH}")

    # Load MERT
    print(f"\nLoading MERT model: {MERT_MODEL}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model     = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model.eval()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {total_params:.1f}M")

    # Process each split
    for meta_df, out_path in [
        (df_train, DATA_DIR / "embeddings_train.pt"),
        (df_val,   DATA_DIR / "embeddings_val.pt"),
        (df_test,  DATA_DIR / "embeddings_test.pt"),
    ]:
        print(f"\nProcessing {out_path.name} ({len(meta_df)} clips)")
        extract_split(meta_df, label_cols, processor, model, out_path, args.batch_size, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
