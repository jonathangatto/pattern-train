"""
Per-second pattern inference on a YouTube video.

Downloads a YouTube video's audio, extracts overlapping MERT frame embeddings
using the same 6-second sliding window used during training (shifted 1 second
at a time), runs the trained classifier, and writes per-second probabilities
to a CSV file.

Window layout (matches extract_embeddings.py):
    CLIP_SECONDS = CONTEXT_BEFORE + ANNOT_SECONDS + CONTEXT_AFTER = 6.0 s

Sliding:
    second 0 → window [0s, 6s)
    second 1 → window [1s, 7s)
    second 2 → window [2s, 8s)
    ... (windows past the audio end are zero-padded)

Usage:
    python scripts/inference.py --url <youtube_url> [options]

Outputs:
    CSV with columns: second, prob_<label>, ...  (one row per second)

Options:
    --url         YouTube URL to analyse (required unless --audio-path given)
    --audio-path  Use an already-downloaded audio file (skips yt-dlp)
    --output      Output CSV path (default: output.csv)
    --batch-size  MERT batch size (default: 32)
    --keep-audio  Keep the downloaded audio file after inference
    --device      Torch device (default: cuda if available, else cpu)
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import build_model
from config import (
    MERT_MODEL, SAMPLE_RATE,
    ANNOT_SECONDS, CONTEXT_BEFORE, CONTEXT_AFTER, CLIP_SECONDS,
    MODEL_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_seconds(sec: int) -> str:
    """Convert an integer number of seconds to M:SS format (e.g. 125 -> '2:05')."""
    minutes, secs = divmod(sec, 60)
    return f"{minutes}:{secs:02d}"


def extract_video_id(url: str) -> str:
    """Return the YouTube video ID from a URL, or the full URL if parsing fails."""
    parsed = urlparse(url)
    # Standard watch URL: ?v=VIDEO_ID
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    # Short URL: youtu.be/VIDEO_ID
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/")
    # Fallback: use last path segment
    return parsed.path.strip("/").split("/")[-1] or url


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def download_audio(url: str, output_dir: Path, filename: str = "audio") -> Path:
    """Download audio-only mp3 from a YouTube URL via yt-dlp.

    Args:
        url:        YouTube URL.
        output_dir: Directory to save the file.
        filename:   Stem for the output file (without extension).
    """
    output_template = str(output_dir / f"{filename}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--no-playlist",
        "--output", output_template,
        url,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    mp3_path = output_dir / f"{filename}.mp3"
    if mp3_path.exists():
        return mp3_path

    # yt-dlp may retain a different extension before conversion
    candidates = sorted(output_dir.glob(f"{filename}.*"))
    if not candidates:
        raise FileNotFoundError(f"yt-dlp produced no audio file in {output_dir}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_waveform(audio_path: Path) -> tuple[torch.Tensor, float]:
    """Load audio, convert to mono 24 kHz. Returns (waveform_1d, duration_secs)."""
    waveform, sr = torchaudio.load(str(audio_path))

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # (samples,)

    # Resample to MERT native rate
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    duration_secs = waveform.shape[0] / SAMPLE_RATE
    return waveform, duration_secs


# ---------------------------------------------------------------------------
# Sliding window construction
# ---------------------------------------------------------------------------

def make_windows(
    waveform: torch.Tensor, duration_secs: float
) -> tuple[list[torch.Tensor], list[int]]:
    """Generate overlapping CLIP_SECONDS windows, sliding 1 second at a time.

    Window for second s spans [s, s + CLIP_SECONDS).  Windows whose tail
    extends past the audio end are zero-padded (same as training).  One window
    is produced for every integer second from 0 up to (but not including) the
    audio duration.

    Returns:
        clips:   list[Tensor]  — each tensor has length CLIP_SECONDS * SAMPLE_RATE
        seconds: list[int]     — start second for each clip
    """
    clip_samples  = int(CLIP_SECONDS * SAMPLE_RATE)
    total_samples = waveform.shape[0]

    clips: list[torch.Tensor] = []
    seconds: list[int] = []

    start_sec = 0
    while start_sec < int(duration_secs):
        start_sample = int(start_sec * SAMPLE_RATE)
        end_sample   = start_sample + clip_samples

        if end_sample <= total_samples:
            clip = waveform[start_sample:end_sample]
        else:
            # Window runs past end-of-audio — zero-pad the tail
            available = waveform[start_sample:total_samples]
            pad       = torch.zeros(clip_samples - available.shape[0])
            clip      = torch.cat([available, pad])

        clips.append(clip)
        seconds.append(start_sec)
        start_sec += 1

    return clips, seconds


# ---------------------------------------------------------------------------
# MERT embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    clips: list[torch.Tensor],
    processor,
    mert_model,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run MERT on all clips in batches.

    Returns:
        Tensor of shape (N, T, 1024) — last hidden state for each window
    """
    all_hidden: list[torch.Tensor] = []

    for i in tqdm(range(0, len(clips), batch_size), desc="Extracting embeddings"):
        batch = clips[i : i + batch_size]
        batch_np = [c.numpy() for c in batch]

        inputs = processor(
            batch_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)

        outputs = mert_model(input_values)
        hidden  = outputs.last_hidden_state.cpu()  # (B, T, 1024)
        all_hidden.append(hidden)

    return torch.cat(all_hidden, dim=0)  # (N, T, 1024)


# ---------------------------------------------------------------------------
# Classifier inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_classifier(
    embeddings: torch.Tensor,
    classifier: torch.nn.Module,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Apply classifier to frame embeddings and return probabilities.

    Args:
        embeddings: (N, T, D)

    Returns:
        probs: (N, n_labels) — values in [0, 1]
    """
    classifier.eval()
    probs_list: list[torch.Tensor] = []

    for i in range(0, embeddings.shape[0], batch_size):
        batch  = embeddings[i : i + batch_size].to(device)
        logits = classifier(batch)
        probs_list.append(torch.sigmoid(logits).cpu())

    return torch.cat(probs_list, dim=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-second pattern inference on a YouTube video"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--url",
        type=str,
        default=None,
        help="YouTube URL to analyse",
    )
    mode.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a CSV file with a 'youtube_url' column for batch inference",
    )
    p.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Path to an already-downloaded audio file (only with --url; skips yt-dlp)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="output.csv",
        help="Output CSV path for single-URL mode (default: output.csv)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="MERT inference batch size (default: 32)",
    )
    p.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the downloaded audio file after inference",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-audio processing (shared by single-URL and CSV modes)
# ---------------------------------------------------------------------------

def process_audio(
    audio_path: Path,
    processor,
    mert_model,
    classifier: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    """Run the full pipeline on one audio file.

    Returns:
        seconds: list of integer start-seconds for each window.
        probs:   (N, n_labels) tensor of probabilities.
    """
    waveform, duration_secs = load_waveform(audio_path)
    clips, seconds = make_windows(waveform, duration_secs)
    print(f"  Duration : {duration_secs:.1f} s  |  Windows : {len(clips)}")

    embeddings = extract_embeddings(clips, processor, mert_model, batch_size, device)
    probs = run_classifier(embeddings, classifier, device)
    return seconds, probs


# ---------------------------------------------------------------------------
# Single-URL mode
# ---------------------------------------------------------------------------

def process_single_url(
    args: argparse.Namespace,
    device: torch.device,
    label_cols: list[str],
    processor,
    mert_model,
    classifier: torch.nn.Module,
) -> None:
    """Download (if needed) and run inference on a single YouTube URL or local file."""
    tmp_dir    = None
    audio_path: Path

    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        print(f"\nUsing audio file: {audio_path}")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="pattern_infer_")
        print(f"\nDownloading audio from: {args.url}")
        try:
            audio_path = download_audio(args.url, Path(tmp_dir))
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode(errors="replace").strip()
            print(f"yt-dlp error: {err}", file=sys.stderr)
            sys.exit(1)
        print(f"Saved to   : {audio_path}")

    try:
        seconds, probs = process_audio(
            audio_path, processor, mert_model, classifier, args.batch_size, device
        )

        # Write CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prob_cols = [f"prob_{lbl}" for lbl in label_cols]

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + prob_cols)
            for sec, row_probs in zip(seconds, probs.tolist()):
                writer.writerow([format_seconds(sec)] + [f"{p:.6f}" for p in row_probs])

        print(f"\nResults    : {output_path}  ({len(seconds)} rows)")

    finally:
        if tmp_dir is not None:
            if args.keep_audio:
                print(f"Audio kept : {audio_path}")
            else:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CSV batch inference
# ---------------------------------------------------------------------------

def process_csv(
    args: argparse.Namespace,
    device: torch.device,
    label_cols: list[str],
    processor,
    mert_model,
    classifier: torch.nn.Module,
) -> None:
    """Run batch inference on all YouTube URLs listed in a CSV file."""
    csv_path  = Path(args.csv)
    csv_dir   = csv_path.parent
    csv_stem  = csv_path.stem
    audio_dir = csv_dir / f"{csv_stem}_audio"
    out_path  = csv_dir / f"{csv_stem}_results.csv"

    audio_dir.mkdir(parents=True, exist_ok=True)

    prob_cols = [f"prob_{lbl}" for lbl in label_cols]

    # Read and deduplicate URLs
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "youtube_url" not in (reader.fieldnames or []):
            print("ERROR: input CSV must have a 'youtube_url' column", file=sys.stderr)
            sys.exit(1)
        urls = list(dict.fromkeys(
            row["youtube_url"].strip() for row in reader if row["youtube_url"].strip()
        ))

    print(f"Found {len(urls)} unique URLs in {csv_path}")
    print(f"Audio dir : {audio_dir}")
    print(f"Output    : {out_path}")

    all_rows: list[list] = []

    for i, url in enumerate(urls, 1):
        video_id   = extract_video_id(url)
        audio_path = audio_dir / f"{video_id}.mp3"
        print(f"\n[{i}/{len(urls)}] {url}")

        # Download (skip if already present)
        if audio_path.exists():
            print(f"  Audio cached : {audio_path}")
        else:
            try:
                print("  Downloading audio...")
                audio_path = download_audio(url, audio_dir, filename=video_id)
                print(f"  Saved to     : {audio_path}")
            except subprocess.CalledProcessError as exc:
                err = exc.stderr.decode(errors="replace").strip()
                print(f"  WARNING: yt-dlp failed. Error: {err}", file=sys.stderr)
                all_rows.append([url, "DOWNLOAD_FAILED"] + [""] * len(prob_cols))
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: download error. {exc}", file=sys.stderr)
                all_rows.append([url, "DOWNLOAD_FAILED"] + [""] * len(prob_cols))
                continue

        # Process
        try:
            seconds, probs = process_audio(
                audio_path, processor, mert_model, classifier, args.batch_size, device
            )
            for sec, row_probs in zip(seconds, probs.tolist()):
                all_rows.append([url, format_seconds(sec)] + [f"{p:.6f}" for p in row_probs])

        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: processing error — skipping. {exc}", file=sys.stderr)
            continue

    # Write combined results
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["youtube_url", "time"] + prob_cols)
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} rows written to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    if args.url is None and args.audio_path is None and args.csv is None:
        print("ERROR: provide --url, --csv, or --audio-path", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load classifier config
    # ------------------------------------------------------------------
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found – run train_classifier.py first"
        )
    with open(config_path) as f:
        config = json.load(f)

    label_cols = config["label_cols"]
    n_labels   = config["n_labels"]
    in_dim     = config["in_dim"]
    model_name = config["model_name"]

    print(f"Classifier : {model_name}")
    print(f"Labels     : {label_cols}")
    print(f"Device     : {device}")

    # ------------------------------------------------------------------
    # Load MERT
    # ------------------------------------------------------------------
    print(f"\nLoading MERT: {MERT_MODEL}")
    processor  = Wav2Vec2FeatureExtractor.from_pretrained(
        MERT_MODEL, trust_remote_code=True
    )
    mert_model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True)
    mert_model.eval()
    mert_model.to(device)

    # ------------------------------------------------------------------
    # Load classifier
    # ------------------------------------------------------------------
    model_pt = MODEL_DIR / "best_model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(
            f"{model_pt} not found – run train_classifier.py first"
        )
    classifier = build_model(model_name, in_dim=in_dim, n_labels=n_labels)
    classifier.load_state_dict(
        torch.load(model_pt, map_location=device, weights_only=True)
    )
    classifier.to(device)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if args.csv:
        process_csv(args, device, label_cols, processor, mert_model, classifier)
    else:
        process_single_url(args, device, label_cols, processor, mert_model, classifier)


if __name__ == "__main__":
    main()
