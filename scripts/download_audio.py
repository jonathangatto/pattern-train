"""
Download audio (mp3) from YouTube URLs listed in the CSV and save them
locally to data/audio/<id>.mp3.

Resume-safe: rows whose local file already exists are skipped.

To upload the downloaded files to Google Cloud Storage, run:
    python scripts/upload_to_gcs.py
"""

import csv
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parent.parent
CSV_PATH       = REPO_ROOT / "data" / "dbo-moments-2-live.1774458459.csv"
AUDIO_DIR      = REPO_ROOT / "data" / "audio"
ERRORS_CSV_PATH = REPO_ROOT / "data" / f"{CSV_PATH.stem}_errors{CSV_PATH.suffix}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_audio(url: str, row_id: str, output_dir: Path) -> Path:
    """
    Use yt-dlp to download audio-only as mp3.
    Returns the path to the downloaded file.
    Raises subprocess.CalledProcessError on failure.
    """
    output_template = str(output_dir / f"{row_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",        # best quality VBR
        "--no-playlist",
        "--output", output_template,
        url,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_dir / f"{row_id}.mp3"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    total = len(rows)
    print(f"Loaded {total} rows from CSV.")
    print(f"Audio output directory: {AUDIO_DIR}")
    print("-" * 60)

    failed_rows: list[dict] = []

    for idx, row in enumerate(rows, start=1):
        row_id = row.get("id", "").strip()
        url    = row.get("URL", "").strip()

        if not row_id or not url:
            print(f"[{idx}/{total}] SKIP  – missing id or URL")
            continue

        local_file = AUDIO_DIR / f"{row_id}.mp3"

        if local_file.exists():
            print(f"[{idx}/{total}] SKIP  – {row_id}.mp3 already exists locally")
            continue

        prefixed_files = list(AUDIO_DIR.glob(f"{row_id}_*.mp3"))
        if prefixed_files:
            print(f"[{idx}/{total}] SKIP  – {prefixed_files[0].name} already exists locally (prefix match)")
            continue

        try:
            print(f"[{idx}/{total}] DL    – {url}")
            download_audio(url, row_id, AUDIO_DIR)
        except subprocess.CalledProcessError as exc:
            error_desc = (
                f"CalledProcessError: "
                f"{exc.stderr.decode(errors='replace').strip() or str(exc)}"
            )
            print(f"[{idx}/{total}] ERROR – row_id={row_id}: {error_desc}")
            failed_rows.append({**row, "error_description": error_desc})
        except Exception as exc:
            error_desc = f"{type(exc).__name__}: {exc}"
            print(f"[{idx}/{total}] ERROR – row_id={row_id}: {error_desc}")
            failed_rows.append({**row, "error_description": error_desc})

    # --- Write errors CSV -----------------------------------------------
    if failed_rows:
        fieldnames = list(rows[0].keys()) + ["error_description"]
        with ERRORS_CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failed_rows)
        print(f"\nError details written to: {ERRORS_CSV_PATH}")

    print("-" * 60)
    print(f"Done. {len(failed_rows)} error(s).")
    if failed_rows:
        print("\nFailed row IDs:")
        for r in failed_rows:
            print(f"  id={r.get('id')}  url={r.get('URL')}  – {r['error_description'][:120]}")


if __name__ == "__main__":
    main()
