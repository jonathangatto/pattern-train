"""
Upload locally downloaded audio files from data/audio/ to a Google Cloud
Storage bucket.

Configuration is read from a .env file at the repository root:
    GCS_BUCKET_NAME          – name of the GCS bucket (required)
    GCS_DESTINATION_PREFIX   – folder prefix inside the bucket (default: "audio/")

Resume-safe: files whose blob already exists in the bucket are skipped.

To download audio files first, run:
    python scripts/download_audio.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "audio"

# ---------------------------------------------------------------------------
# Config from .env
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=REPO_ROOT / ".env")

GCS_BUCKET_NAME        = os.getenv("GCS_BUCKET_NAME", "").strip()
GCS_DESTINATION_PREFIX = os.getenv("GCS_DESTINATION_PREFIX", "audio/").strip()

if not GCS_BUCKET_NAME:
    sys.exit(
        "ERROR: GCS_BUCKET_NAME is not set. "
        "Please fill it in the .env file at the repository root."
    )

# Normalise prefix: strip leading slash, ensure trailing slash
GCS_DESTINATION_PREFIX = GCS_DESTINATION_PREFIX.lstrip("/")
if GCS_DESTINATION_PREFIX and not GCS_DESTINATION_PREFIX.endswith("/"):
    GCS_DESTINATION_PREFIX += "/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def blob_exists(bucket: storage.Bucket, blob_name: str) -> bool:
    """Return True if the blob already exists in the bucket."""
    return bucket.blob(blob_name).exists()


def upload_file(bucket: storage.Bucket, local_path: Path, blob_name: str) -> None:
    """Upload a local file to the given GCS bucket path."""
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not AUDIO_DIR.exists():
        sys.exit(
            f"ERROR: Audio directory not found: {AUDIO_DIR}\n"
            "Run scripts/download_audio.py first."
        )

    mp3_files = sorted(AUDIO_DIR.glob("*.mp3"))
    total = len(mp3_files)

    if total == 0:
        print(f"No .mp3 files found in {AUDIO_DIR}. Nothing to upload.")
        return

    print(f"Found {total} local mp3 file(s) in {AUDIO_DIR}.")
    print(f"Target bucket: gs://{GCS_BUCKET_NAME}/{GCS_DESTINATION_PREFIX}")
    print("-" * 60)

    gcs_client = storage.Client()
    bucket     = gcs_client.bucket(GCS_BUCKET_NAME)

    errors:  list[str] = []
    skipped: int = 0
    uploaded: int = 0

    for idx, local_file in enumerate(mp3_files, start=1):
        blob_name = f"{GCS_DESTINATION_PREFIX}{local_file.name}"

        try:
            if blob_exists(bucket, blob_name):
                print(f"[{idx}/{total}] SKIP  – {local_file.name} already in GCS")
                skipped += 1
                continue
        except Exception as exc:
            msg = f"[{idx}/{total}] ERROR – could not check blob for {local_file.name}: {exc}"
            print(msg)
            errors.append(msg)
            continue

        try:
            print(f"[{idx}/{total}] UP    – {local_file.name} → gs://{GCS_BUCKET_NAME}/{blob_name}")
            upload_file(bucket, local_file, blob_name)
            uploaded += 1
        except Exception as exc:
            msg = f"[{idx}/{total}] ERROR – upload failed for {local_file.name}: {exc}"
            print(msg)
            errors.append(msg)

    print("-" * 60)
    print(f"Done. {uploaded} uploaded, {skipped} skipped, {len(errors)} error(s).")
    if errors:
        print("\nFailed files:")
        for e in errors:
            print(" ", e)


if __name__ == "__main__":
    main()
