"""
Multi-model per-moment inference.

Reads an existing predictions CSV (default: the Apr 16 GRF-filtered batch) and,
for every (youtube_url, time) moment, scores that moment with all nine pattern
models plus the two general frisson models. Each model becomes one probability
column in the output CSV.

Models (one prob_<name> column each):
    9 patterns  : ALR AGR GRF HRM SZE PXY SPR ANT PDX  (min_submits_90/pattern_<X>)
    frisson_90  : general any-pattern binary model     (min_submits_90)
    frisson_93  : general any-pattern binary model     (min_submits_93)

For each moment the 6-second window is rebuilt exactly as in training/inference
(window for second s spans [s, s + CLIP_SECONDS)), MERT embeddings are extracted
once, and every model head is applied to those shared embeddings.

Note: prob_GRF here comes from the min_submits_90 GRF model, so it may differ
slightly from the prob_GRF in the input file (produced by the older model).

Usage:
    python scripts/multi_inference.py [options]

Options:
    --input       Input CSV with 'youtube_url' and 'time' columns
                  (default: data/csv_prediction_batches/apr_16_results_grf.csv)
    --output      Output CSV path (default: <input_stem>_all_patterns.csv)
    --audio-dir   Directory of cached <video_id>.mp3 files; missing ones are
                  downloaded here (default: data/csv_prediction_batches/apr_16_audio)
    --batch-size  MERT batch size (default: 32)
    --device      Torch device (default: cuda if available, else cpu)
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import build_model
from config import MERT_MODEL, SAMPLE_RATE, CLIP_SECONDS, CLASSIFIER_DIR
from inference import (
    download_audio,
    load_waveform,
    extract_embeddings,
    extract_video_id,
    run_classifier,
)

DEFAULT_INPUT = (
    REPO_ROOT / "data" / "csv_prediction_batches" / "apr_16_results_grf.csv"
)
DEFAULT_AUDIO_DIR = REPO_ROOT / "data" / "csv_prediction_batches" / "apr_16_audio"

PATTERNS = ["ALR", "AGR", "GRF", "HRM", "SZE", "PXY", "SPR", "ANT", "PDX"]


def model_specs() -> list[tuple[str, Path]]:
    """Return [(column_name, model_dir)] for all models, one prob column each."""
    specs = [(p, CLASSIFIER_DIR / "min_submits_90" / f"pattern_{p}") for p in PATTERNS]
    specs.append(("frisson_90", CLASSIFIER_DIR / "min_submits_90"))
    specs.append(("frisson_93", CLASSIFIER_DIR / "min_submits_93"))
    return specs


def parse_time_to_second(t: str) -> int | None:
    """Parse 'M:SS' (or 'H:MM:SS') into integer seconds. None if not a time."""
    t = (t or "").strip()
    parts = t.split(":")
    if not parts or not all(p.isdigit() for p in parts):
        return None
    sec = 0
    for p in parts:
        sec = sec * 60 + int(p)
    return sec


def make_window(waveform: torch.Tensor, second: int) -> torch.Tensor | None:
    """Build the CLIP_SECONDS window starting at `second`, zero-padding the tail.

    Returns None if the window starts past the end of the audio.
    """
    clip_samples = int(CLIP_SECONDS * SAMPLE_RATE)
    start = int(second * SAMPLE_RATE)
    total = waveform.shape[0]
    if start >= total:
        return None
    end = start + clip_samples
    if end <= total:
        return waveform[start:end]
    available = waveform[start:total]
    pad = torch.zeros(clip_samples - available.shape[0])
    return torch.cat([available, pad])


def load_classifier(model_dir: Path, device: torch.device):
    """Load a trained single-label classifier head from a model directory."""
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    if cfg["n_labels"] != 1:
        raise ValueError(
            f"{model_dir} has n_labels={cfg['n_labels']}; expected 1 (single prob column)."
        )
    clf = build_model(cfg["model_name"], in_dim=cfg["in_dim"], n_labels=cfg["n_labels"])
    clf.load_state_dict(
        torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    clf.to(device).eval()
    return clf


def resolve_audio(url: str, audio_dir: Path) -> Path | None:
    """Return a cached <video_id>.mp3, downloading it if missing. None on failure."""
    video_id = extract_video_id(url)
    cached = audio_dir / f"{video_id}.mp3"
    if cached.exists():
        return cached
    try:
        return download_audio(url, audio_dir, filename=video_id)
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="replace").strip()
        print(f"  WARNING: yt-dlp failed for {url}: {err}", file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: download error for {url}: {exc}", file=sys.stderr)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score moments with all pattern + frisson models")
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--audio-dir", type=str, default=str(DEFAULT_AUDIO_DIR))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_all_patterns.csv")
    )
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    specs = model_specs()
    cols = [name for name, _ in specs]

    # Load MERT once.
    print(f"Loading MERT: {MERT_MODEL}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL, trust_remote_code=True)
    mert_model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True)
    mert_model.eval().to(device)

    # Load every classifier head.
    print(f"Loading {len(specs)} classifiers...")
    classifiers: dict[str, torch.nn.Module] = {}
    for name, model_dir in specs:
        if not (model_dir / "best_model.pt").exists():
            print(f"ERROR: missing model: {model_dir / 'best_model.pt'}", file=sys.stderr)
            sys.exit(1)
        classifiers[name] = load_classifier(model_dir, device)
    print(f"Columns    : {[f'prob_{c}' for c in cols]}")

    # Read input rows (preserve order) and collect the seconds needed per URL.
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for required in ("youtube_url", "time"):
            if required not in fields:
                print(f"ERROR: input CSV must have a '{required}' column", file=sys.stderr)
                sys.exit(1)
        rows = [{"youtube_url": r["youtube_url"].strip(), "time": (r["time"] or "").strip()} for r in reader]

    row_second = [parse_time_to_second(r["time"]) for r in rows]
    url_order: "OrderedDict[str, None]" = OrderedDict()
    url_seconds: dict[str, set[int]] = defaultdict(set)
    for r, sec in zip(rows, row_second):
        if r["youtube_url"] and sec is not None:
            url_order.setdefault(r["youtube_url"], None)
            url_seconds[r["youtube_url"]].add(sec)

    print(f"\nInput      : {input_path}  ({len(rows)} rows, {len(url_order)} URLs to score)")
    print(f"Audio dir  : {audio_dir}")
    print(f"Output     : {output_path}\n")

    # Score each URL once; embeddings are shared across all models.
    results: dict[tuple[str, int], dict[str, float]] = {}
    for i, url in enumerate(url_order, 1):
        seconds_sorted = sorted(url_seconds[url])
        print(f"[{i}/{len(url_order)}] {url}  ({len(seconds_sorted)} moments)")

        audio_path = resolve_audio(url, audio_dir)
        if audio_path is None:
            continue
        try:
            waveform, _ = load_waveform(audio_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not load audio — skipping. {exc}", file=sys.stderr)
            continue

        windows: list[torch.Tensor] = []
        valid_seconds: list[int] = []
        for s in seconds_sorted:
            w = make_window(waveform, s)
            if w is not None:
                windows.append(w)
                valid_seconds.append(s)
        if not windows:
            continue

        embeddings = extract_embeddings(windows, processor, mert_model, args.batch_size, device)
        per_model = {
            name: run_classifier(embeddings, clf, device)[:, 0].tolist()
            for name, clf in classifiers.items()
        }
        for idx, s in enumerate(valid_seconds):
            results[(url, s)] = {name: per_model[name][idx] for name in cols}

    # Write output, preserving input row order. Unscored rows get empty cells.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["youtube_url", "time"] + [f"prob_{c}" for c in cols])
        scored = 0
        for r, sec in zip(rows, row_second):
            key = (r["youtube_url"], sec) if sec is not None else None
            if key is not None and key in results:
                probs = results[key]
                writer.writerow([r["youtube_url"], r["time"]] + [f"{probs[c]:.6f}" for c in cols])
                scored += 1
            else:
                writer.writerow([r["youtube_url"], r["time"]] + [""] * len(cols))

    print(f"\nDone. {scored}/{len(rows)} rows scored → {output_path}")


if __name__ == "__main__":
    main()
