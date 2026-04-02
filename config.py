"""
Shared pipeline constants for pattern-train.

Imported by:
    scripts/extract_embeddings.py
    scripts/train_classifier.py
    scripts/evaluate.py
    scripts/inference.py
"""

from pathlib import Path

# Repository layout
REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "models" / "classifier"

# MERT encoder
MERT_MODEL = "m-a-p/MERT-v1-330M"
SAMPLE_RATE = 24_000   # MERT's native sample rate (Hz)
EMBED_DIM   = 1024     # MERT hidden size per frame

# Clip window geometry
# Each clip spans [moment - CONTEXT_BEFORE, moment + ANNOT_SECONDS + CONTEXT_AFTER]
CONTEXT_BEFORE = 2.0   # seconds of context before the annotation
ANNOT_SECONDS  = 2.0   # the annotated second
CONTEXT_AFTER  = 2.0   # seconds of context after the annotation
CLIP_SECONDS   = CONTEXT_BEFORE + ANNOT_SECONDS + CONTEXT_AFTER  # 6.0 s total
