"""
Download Harmonix dataset audio from YouTube.

Prerequisites:
  1. Clone the Harmonix repo into this project folder:
       git clone https://github.com/urinieto/harmonixset.git
  2. Run this script:
       python download_harmonix.py
"""

import os
import csv
import subprocess

BASE        = os.path.dirname(os.path.abspath(__file__))
HARMONIX_REPO = os.path.join(BASE, "harmonixset")
OUTDIR      = os.path.join(BASE, "harmonix_mp3s")
os.makedirs(OUTDIR, exist_ok=True)

urls_path = os.path.join(HARMONIX_REPO, "dataset", "youtube_urls.csv")
if not os.path.exists(urls_path):
    raise FileNotFoundError(
        f"Harmonix repo not found at {HARMONIX_REPO}\n"
        "Run: git clone https://github.com/urinieto/harmonixset.git"
    )

with open(urls_path) as f:
    rows = list(csv.DictReader(f))

print(f"Found {len(rows)} songs to download\n")

for row in rows:
    file_id = row["File"]
    url     = row["URL"]
    out_path = os.path.join(OUTDIR, f"{file_id}.mp3")

    if os.path.exists(out_path):
        print(f"  Already exists: {file_id} — skipping")
        continue

    print(f"  Downloading {file_id}")
    result = subprocess.run([
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", out_path,
        url
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.strip()[:120]}")
    else:
        print(f"    Done")

print(f"\nFinished. MP3s saved to {OUTDIR}")
