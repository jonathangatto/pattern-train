"""
Extract MERT embeddings for all clips and save them to a single file.

Reads the annotations CSV, filters rows to those with a downloaded audio
file, binarises the 9 pattern label columns, adds negative samples, then
runs the frozen MERT-v1-330M encoder on each clip and saves the raw frame
embeddings (T × 1024) — no pooling — so downstream models can apply their
own temporal operations.

Train/val/test splitting is deferred to training time (see split_utils.py)
so that different split strategies (holdout, cross-validation) can be
explored without re-extracting embeddings.

Each clip window: CONTEXT_BEFORE + ANNOT_SECONDS + CONTEXT_AFTER seconds.
Boundaries are zero-padded.

Outputs:
    data/embeddings_all.pt     → {"frames": Tensor(N,T,1024), "labels": Tensor(N,9),
                                   "urls": list[str], "ids": list}
    data/dataset_summary.json  → label metadata used by training scripts

Usage:
    python scripts/extract_embeddings.py [--batch-size 16] [--device cuda]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR     = REPO_ROOT / "data"
CSV_PATH     = DATA_DIR / "dbo-moments-2-live.1774458459.csv"
AUDIO_DIR    = DATA_DIR / "audio"
SUMMARY_PATH = DATA_DIR / "dataset_summary.json"

MERT_MODEL      = "m-a-p/MERT-v1-330M"
SAMPLE_RATE     = 24_000   # MERT's native sample rate
EMBED_DIM       = 1024      # MERT hidden size per frame

# Clip window: [moment_secs - CONTEXT_BEFORE, moment_secs + ANNOT_SECONDS + CONTEXT_AFTER]
ANNOT_SECONDS   = 2.0      # the annotated second
CONTEXT_BEFORE  = 2.0      # seconds of context before the annotation
CONTEXT_AFTER   = 2.0      # seconds of context after the annotation
CLIP_SECONDS    = CONTEXT_BEFORE + ANNOT_SECONDS + CONTEXT_AFTER  # 3.0 s total

LABEL_COLS   = ["ANT", "SPR", "PDX", "AGR", "ALR", "GRF", "HRM", "SZE", "PXY"]
RANDOM_SEED  = 42
NEGATIVES_PER_ANNOTATION = 2


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
    """Return path to audio file for this id, or None if not present.

    Tries exact names first: {row_id}.mp3 / {row_id}.webm.
    Falls back to prefixed variants: {row_id}_*.mp3 / {row_id}_*.webm.
    """
    for ext in ("mp3", "webm"):
        p = AUDIO_DIR / f"{row_id}.{ext}"
        if p.exists():
            return str(p)

        prefixed_matches = sorted(AUDIO_DIR.glob(f"{row_id}_*.{ext}"))
        if prefixed_matches:
            return str(prefixed_matches[0])
    return None


def _add_negatives(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """For each annotated moment, add random moments from the same song URL
    whose full clip window does not overlap any annotated window.

    The negative row has all label columns set to 0.
    Song membership (train/val/test) is preserved because candidates are drawn
    only from songs already present in `df`.
    """
    # Half-open annotated window for a moment m:
    #   [m - CONTEXT_BEFORE,  m + ANNOT_SECONDS + CONTEXT_AFTER)
    # Two windows overlap when |c - m| < CLIP_SECONDS, so the minimum
    # safe distance between any candidate c and any annotated moment m is
    # CLIP_SECONDS (== CONTEXT_BEFORE + ANNOT_SECONDS + CONTEXT_AFTER).
    MIN_DIST   = CLIP_SECONDS          # seconds; guarantees zero window overlap
    MAX_TRIES  = 200                   # attempts per annotation before giving up

    neg_rows = []

    for song_url, group in df.groupby("URL"):
        annotated = group["moment_secs"].tolist()
        audio_path = group["audio_path"].iloc[0]
        row_id = group["id"].iloc[0]

        # Determine usable song length
        song_len = group["song_length_secs"].dropna()
        if len(song_len) > 0:
            duration = float(song_len.iloc[0])
        else:
            # Fall back: last annotation + full clip width
            duration = max(annotated) + CLIP_SECONDS

        # Valid candidate range so the full clip fits inside the song
        c_min = CONTEXT_BEFORE
        c_max = duration - ANNOT_SECONDS - CONTEXT_AFTER
        if c_max <= c_min:
            continue  # song too short to fit even one clip

        for _ in annotated:
            for _ in range(NEGATIVES_PER_ANNOTATION):
                for attempt in range(MAX_TRIES):
                    c = rng.uniform(c_min, c_max)
                    # Reject if candidate window overlaps any annotated window
                    if all(abs(c - m) >= MIN_DIST for m in annotated):
                        neg_rows.append({
                            "id":              row_id,
                            "URL":             song_url,
                            "tastes_id":       group["tastes_id"].iloc[0],
                            "moment_secs":     c,
                            "song_length_secs": duration,
                            "audio_path":      audio_path,
                            **{col: 0 for col in LABEL_COLS},
                        })
                        break
                # If MAX_TRIES exhausted without a valid candidate, skip silently

    if not neg_rows:
        return df

    df_neg = pd.DataFrame(neg_rows, columns=df.columns)
    return pd.concat([df, df_neg], ignore_index=True)


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Load CSV, filter, binarise labels, add negatives.

    Splitting is deferred to training time (see split_utils.py).
    Returns (df_all, label_cols).
    """
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
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
    if "URL" not in df.columns:
        raise ValueError("CSV is missing required column 'URL' for song-level grouping")
    df["URL"] = df["URL"].fillna("").astype(str).str.strip()

    before = len(df)
    df = df.dropna(subset=["moment_secs"]).reset_index(drop=True)
    if before != len(df):
        print(f"  Dropped {before - len(df)} rows with missing moment_secs")

    before_url = len(df)
    df = df[df["URL"] != ""].reset_index(drop=True)
    if before_url != len(df):
        print(f"  Dropped {before_url - len(df)} rows with missing URL")

    # Preserve tastes_id (genre) for stratified splitting downstream
    if "tastes_id" in df.columns:
        df["tastes_id"] = pd.to_numeric(df["tastes_id"], errors="coerce").fillna(-1).astype(int)
    else:
        print("  WARNING: column 'tastes_id' not found in CSV – genre stratification unavailable")
        df["tastes_id"] = -1

    df = df[["id", "URL", "tastes_id", "moment_secs", "song_length_secs", "audio_path"] + LABEL_COLS]

    n_annotations = len(df)
    print(f"  Annotations: {n_annotations}")

    print("\nLabel distribution (% positive):")
    for col in LABEL_COLS:
        print(f"  {col}: {df[col].mean() * 100:.1f}%")

    # Add negatives for the full dataset (they inherit the source song's URL,
    # so URL-level splitting keeps them in the correct split downstream).
    rng = random.Random(RANDOM_SEED)
    df = _add_negatives(df, rng)

    print(f"\nTotal clips (with negatives): {len(df)}  "
          f"({n_annotations} annotations + {len(df) - n_annotations} negatives)")
    return df, LABEL_COLS


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
    layer        : str = "last",
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

    use_hidden_states = (layer != "last")

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

        outputs = model(input_values, output_hidden_states=use_hidden_states)

        if layer == "last":
            hidden = outputs.last_hidden_state.cpu()  # (B, T, D)
        elif layer == "all":
            # Average all hidden states (feature extractor output + transformer layers)
            stacked = torch.stack(outputs.hidden_states, dim=0)  # (L, B, T, D)
            hidden = stacked.mean(dim=0).cpu()  # (B, T, D)
        else:
            # Specific layer index
            layer_idx = int(layer)
            hidden = outputs.hidden_states[layer_idx].cpu()  # (B, T, D)

        all_embeddings.append(hidden)
        all_labels.append(labels)
        all_ids.extend(ids)

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    labels_tensor     = torch.cat(all_labels, dim=0)

    # Metadata in the same order as embeddings (DataLoader with shuffle=False preserves order)
    urls_list      = meta_df["URL"].tolist() if "URL" in meta_df.columns else []
    tastes_id_list = meta_df["tastes_id"].tolist() if "tastes_id" in meta_df.columns else []

    torch.save(
        {"frames": embeddings_tensor, "labels": labels_tensor,
         "urls": urls_list, "tastes_ids": tastes_id_list, "ids": all_ids},
        out_path,
    )
    print(f"  Saved {embeddings_tensor.shape[0]} clips (shape {tuple(embeddings_tensor.shape)}) to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract MERT embeddings for all splits")
    p.add_argument("--batch-size", type=int,   default=96)
    p.add_argument("--layer",      type=str,   default="last",
                   help="Which hidden layer to extract: 'last' (default), 'all' "
                        "(average all layers), or an integer index (0 = feature "
                        "extractor output, 1..24 = transformer layers)")
    p.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Build full dataset (no splitting — that happens at training time)
    df_all, label_cols = build_dataset()

    # Write dataset_summary.json so training scripts can read label metadata
    summary = {
        "label_cols": label_cols,
        "n_labels":   len(label_cols),
        "total_clips": len(df_all),
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
    print(f"  Layer selection: {args.layer}")

    # Extract embeddings for all clips into a single file
    out_path = DATA_DIR / "embeddings_all.pt"
    print(f"\nProcessing all {len(df_all)} clips")
    extract_split(df_all, label_cols, processor, model, out_path, args.batch_size, device, layer=args.layer)

    print("\nDone.")


if __name__ == "__main__":
    main()
