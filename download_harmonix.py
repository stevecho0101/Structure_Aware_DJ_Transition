# AI Usage:
# Similar to how download_salami.py, we consulted the best way to extract the songs and Claude suggested through youtube. 
# We asked to add formatted print such that we can track which songs are successfully downloaded
# and which one failed.

import os
import csv
import subprocess

BASE        = os.path.dirname(__file__) or "."
HARMONIX_REPO = os.path.join(BASE, "harmonixset")
OUTDIR      = os.path.join(BASE, "harmonix_mp3s")
os.makedirs(OUTDIR, exist_ok=True)

urls_path = os.path.join(HARMONIX_REPO, "dataset", "youtube_urls.csv")

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
        print(f"FAILED: {result.stderr.strip()[:120]}")
    else:
        print(f"Done")

print(f"\nFinished. MP3s saved to {OUTDIR}")
