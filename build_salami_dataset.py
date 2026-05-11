# Build feature CSVs for the SALAMI dataset.
# Run after download_salami.py has populated harmonix_mp3s/.

import os
import glob
import numpy as np
from spectrogram import extract_energy_features, get_timesteps, FEATURE_NAMES

BASE        = os.path.dirname(__file__) or "."
SALAMI_MP3S = os.path.join(BASE, "salami_mp3s")
SALAMI_ANN  = os.path.join(BASE, "salami-data-public/annotations")

# How each annotation in the dataset will map to our features
LABEL_MAP = {
    "intro":      1,
    "verse":      2,
    "pre-chorus": 2,
    "bridge":     2,
    "transition": 2,
    "chorus":     3,
    "hook":       3,
    "refrain":    3,
    "outro":      4,
    "end":        4,
    "coda":       4,
}

# Goes through SALAMI annotation and create a per-second label array of length n_seconds.
def load_salami_annotation(song_id: int, n_seconds: int) -> np.ndarray | None:
    ann_path = os.path.join(SALAMI_ANN, str(song_id), "parsed", "textfile1_functions.txt")
    if not os.path.exists(ann_path):
        return None

    boundaries = []  # list of (start_sec, label)
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            
            try:
                t = float(parts[0])
            except ValueError:
                continue
            
            section = parts[1].strip().lower()
            label = None
            
            for keyword, num in LABEL_MAP.items():
                if keyword in section:
                    label = num
                    break
            if label:
                boundaries.append((int(t), label))

    if len(boundaries) < 2:
        return None

    # Build per-second label array
    labels = np.zeros(n_seconds, dtype=int)
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else n_seconds
        labels[start:min(end, n_seconds)] = label

    # Fill any unlabeled seconds (label=0) using nearest labeled second
    for s in range(n_seconds):
        if labels[s] == 0:
            labels[s] = labels[s - 1] if s > 0 else 1

    return labels

# From the given MP#, extract features and labels pair
def process_mp3(mp3_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    features = extract_energy_features(mp3_path)
    n_seconds = features.shape[0]
    times     = get_timesteps(n_seconds)
    return features, times


# Process SALAMI songs

# AI Usage: Claude helped us how to format it into CSV file
print("---Processing SALAMI songs---")

for mp3_path in sorted(glob.glob(os.path.join(SALAMI_MP3S, "salami_*.mp3"))):
    filename = os.path.basename(mp3_path)
    song_id  = int(filename.split("_")[1])

    features  = extract_energy_features(mp3_path)
    n_seconds = features.shape[0]
    times     = get_timesteps(n_seconds)
    labels    = load_salami_annotation(song_id, n_seconds)

    # There are cases where there are no useful annotations, thus automatically deletes
    if labels is None:
        os.remove(mp3_path)
        print(f"  SKIP salami_{song_id} — no usable annotation, deleted MP3")
        continue

    # Save CSV with label column
    csv_path = mp3_path.replace(".mp3", "_features.csv")
    header   = "time_s," + ",".join(FEATURE_NAMES) + ",label"
    np.savetxt(csv_path, np.column_stack([times, features, labels]),
               delimiter=",", header=header, comments="", fmt="%.6f")

    print(f"  OK   salami_{song_id} — {n_seconds}s → {os.path.basename(csv_path)}")
