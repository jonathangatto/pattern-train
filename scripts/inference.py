"""
Run inference on a full song using a sliding 1-second window.

Loads the MERT model + trained classifier, slides a 1-second window across
the audio with a configurable hop, and outputs a timeline of pattern
activation probabilities.

Output (--output flag or stdout):
    JSON with:
        {
          "file": "...",
          "hop_seconds": 0.5,
          "labels": ["ANT", "SPR", ...],
          "timeline": [
            {"time": 0.0, "probs": {"ANT": 0.12, "SPR": 0.87, ...}},
            ...
          ]
        }

Usage:
    python scripts/inference.py --audio path/to/song.mp3
    python scripts/inference.py --audio path/to/song.mp3 --hop 0.5 --output results.json
    python scripts/inference.py --audio path/to/song.mp3 --threshold 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import build_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
MODEL_DIR    = REPO_ROOT / "models" / "classifier"
MERT_MODEL   = "m-a-p/MERT-v1-95M"
SAMPLE_RATE  = 24_000
# Clip window matches training: 1s before + annotated second + 1s after = 3s
CLIP_SAMPLES = SAMPLE_RATE * 3


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load audio, mix to mono, resample to target_sr. Returns 1D tensor."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


@torch.no_grad()
def run_inference(
    audio_path  : str,
    hop_seconds : float,
    threshold   : float,
    device      : torch.device,
    mert_processor,
    mert_model,
    classifier  : nn.Module,
    label_cols  : list[str],
) -> dict:
    waveform = load_audio(audio_path, SAMPLE_RATE)
    total_samples = len(waveform)
    hop_samples   = int(hop_seconds * SAMPLE_RATE)

    timeline = []
    start = 0

    while start + CLIP_SAMPLES <= total_samples:
        clip = waveform[start : start + CLIP_SAMPLES]

        # MERT embedding
        inputs = mert_processor(
            clip.unsqueeze(0).numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs["input_values"].to(device)
        frames = mert_model(input_values).last_hidden_state  # (1, T, D)

        # Classifier receives raw frames — pooling handled inside the model
        logits = classifier(frames)  # (1, n_labels)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        time_sec = start / SAMPLE_RATE
        timeline.append({
            "time":   round(time_sec, 3),
            "probs":  {col: round(p, 4) for col, p in zip(label_cols, probs)},
            "active": [col for col, p in zip(label_cols, probs) if p >= threshold],
        })

        start += hop_samples

    return {
        "file":        audio_path,
        "duration":    round(total_samples / SAMPLE_RATE, 3),
        "hop_seconds": hop_seconds,
        "threshold":   threshold,
        "labels":      label_cols,
        "timeline":    timeline,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sliding-window pattern inference on a full song")
    p.add_argument("--audio",     required=True, help="Path to audio file (mp3/wav/etc.)")
    p.add_argument("--hop",       type=float, default=0.5, help="Hop size in seconds (default 0.5)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Activation threshold (default: from config.json)")
    p.add_argument("--output",    type=str, default=None,
                   help="Path to write JSON output (default: print to stdout)")
    p.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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

    print(f"Device    : {device}", file=sys.stderr)
    print(f"Labels    : {label_cols}", file=sys.stderr)
    print(f"Threshold : {threshold}", file=sys.stderr)
    print(f"Hop       : {args.hop}s", file=sys.stderr)

    # Load MERT
    print("Loading MERT...", file=sys.stderr)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL, trust_remote_code=True)
    mert      = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True)
    mert.eval()
    mert.to(device)

    # Load classifier
    classifier = build_model(config["model_name"], in_dim=in_dim, n_labels=n_labels)
    classifier.load_state_dict(
        torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=True)
    )
    classifier.eval()
    classifier.to(device)

    # Run
    print(f"Analysing: {args.audio}", file=sys.stderr)
    results = run_inference(
        audio_path=args.audio,
        hop_seconds=args.hop,
        threshold=threshold,
        device=device,
        mert_processor=processor,
        mert_model=mert,
        classifier=classifier,
        label_cols=label_cols,
    )

    print(f"Produced {len(results['timeline'])} windows over {results['duration']}s", file=sys.stderr)

    output_json = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
