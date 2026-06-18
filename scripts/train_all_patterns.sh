#!/usr/bin/env bash
set -euo pipefail

# Train a binary classifier for every pattern and the general frisson (binary) model.
# Each pattern model is saved under models/classifier/min_submits_90/pattern_<PATTERN>/.
# The general model is saved under models/classifier/min_submits_90/.
#
# Usage: bash train_all_patterns.sh

PATTERNS=(ALR AGR GRF HRM SZE PXY SPR ANT PDX)
MIN_SUBMITS=90
BATCH_SIZE=1024

# Reduce GPU memory fragmentation (avoids OOM when the card is nearly full).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL_BASE="models/classifier/min_submits_${MIN_SUBMITS}"

for PATTERN in "${PATTERNS[@]}"; do
    # config.json is written only after training fully completes; best_model.pt
    # appears at epoch 1, so it is not a reliable completion marker.
    DONE_MARKER="${MODEL_BASE}/pattern_${PATTERN}/config.json"
    if [[ -f "$DONE_MARKER" ]]; then
        echo "Skipping $PATTERN — already trained ($DONE_MARKER exists)"
        continue
    fi
    echo "======================================"
    echo "Training pattern: $PATTERN"
    echo "======================================"
    python scripts/train_classifier.py \
        --pattern "$PATTERN" \
        --split-strategy kfold \
        --n-folds 5 \
        --min-submits "$MIN_SUBMITS" \
        --batch-size "$BATCH_SIZE"
done

BINARY_DONE_MARKER="${MODEL_BASE}/config.json"
if [[ -f "$BINARY_DONE_MARKER" ]]; then
    echo "Skipping general frisson model — already trained ($BINARY_DONE_MARKER exists)"
else
    echo "======================================"
    echo "Training general frisson model (binary)"
    echo "======================================"
    python scripts/train_classifier.py \
        --binary \
        --split-strategy kfold \
        --n-folds 5 \
        --min-submits "$MIN_SUBMITS" \
        --batch-size "$BATCH_SIZE"
fi

echo ""
echo "All models trained. Output under models/classifier/min_submits_${MIN_SUBMITS}/."
