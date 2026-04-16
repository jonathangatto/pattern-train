# How to run?
# Extract once (saves data/embeddings_all.pt)
python scripts/extract_embeddings.py --device cuda

# Train with holdout (default)
python scripts/train_classifier.py

# Train with 5-fold cross-validation
python scripts/train_classifier.py --split-strategy kfold --n-folds 5

# Evaluate on held-out test set
python scripts/evaluate.py

---

## Inference

### Single URL

Run per-second pattern prediction on a single YouTube video:

```bash
python scripts/inference.py --url <youtube_url> [--output results.csv] [--batch-size 32] [--keep-audio] [--device cuda]
```

### Batch inference from CSV

Run predictions on multiple YouTube videos listed in a CSV file:

```bash
python scripts/inference.py --csv path/to/urls.csv [--batch-size 32] [--device cuda]
```

**Input CSV format** — must have a `youtube_url` column with one YouTube URL per row:

```
youtube_url
https://www.youtube.com/watch?v=34qC5ltiijQ
https://www.youtube.com/watch?v=81uJZIF9TCs
```

See `data/csv_prediction_batches/apr_16.csv` for a real example.

**What happens:**
- Audio is downloaded to `{csv_name}_audio/` in the same directory as the input CSV (e.g. `apr_16_audio/`)
- Each audio file is named by its YouTube video ID (e.g. `34qC5ltiijQ.mp3`), so re-runs skip already-downloaded files automatically
- Results are saved to `{csv_name}_results.csv` next to the input CSV (e.g. `apr_16_results.csv`)
- Duplicate URLs in the input CSV are deduplicated before processing
- Failed downloads are skipped with a warning and do not abort the batch

**Output CSV format** — one row per second per video, time shown in `M:SS` format:

```
youtube_url,time,prob_any_pattern
https://www.youtube.com/watch?v=34qC5ltiijQ,0:00,0.001670
https://www.youtube.com/watch?v=34qC5ltiijQ,0:01,0.001609
...
https://www.youtube.com/watch?v=81uJZIF9TCs,0:00,0.003210
```