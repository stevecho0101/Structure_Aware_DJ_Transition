import os
import pandas as pd
import subprocess

BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "salami_mp3s")
os.makedirs(OUTDIR, exist_ok=True)

# Load metadata and YouTube pairings
meta     = pd.read_csv(os.path.join(BASE, "salami-data-public/metadata/SALAMI_iTunes_library.csv"))
pairings = pd.read_csv(os.path.join(BASE, "matching-salami/salami_youtube_pairings.csv"))

# Merge all songs with YouTube pairings (no genre filter), cap at 150
merged = meta.merge(pairings, on="salami_id", how="inner").head(150)
print(f"Found {len(merged)} songs to download\n")

for _, row in merged.iterrows():
    song_id    = row["salami_id"]
    youtube_id = row["youtube_id"]
    name       = str(row.get("Name", song_id)).replace("/", "-")
    out_path   = os.path.join(OUTDIR, f"salami_{song_id}_{name}.mp3")

    if os.path.exists(out_path):
        print(f"  Already exists: salami_{song_id} — skipping")
        continue

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    print(f"  Downloading salami_{song_id}: {name}")
    result = subprocess.run([
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", out_path,
        url
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.strip()[:100]}")
    else:
        print(f"    Done")

print(f"\nFinished. MP3s saved to {OUTDIR}")
