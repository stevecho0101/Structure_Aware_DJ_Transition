"""
Build feature CSVs for the Harmonix dataset.
Run after download_harmonix.py has populated harmonix_mp3s/.
"""

import os
import glob
import numpy as np
from spectrogram import extract_energy_features, get_timesteps, FEATURE_NAMES

BASE          = os.path.dirname(os.path.abspath(__file__))
HARMONIX_REPO = os.path.join(BASE, "harmonixset")
SEGMENTS_DIR  = os.path.join(HARMONIX_REPO, "dataset", "segments")
HARMONIX_MP3S = os.path.join(BASE, "harmonix_mp3s")

LABEL_MAP = {
    "intro":      1,
    "verse":      2,
    "pre-chorus": 2,
    "prechorus":  2,
    "pre_chorus": 2,
    "bridge":     2,
    "transition": 2,
    "break":      2,
    "chorus":     3,
    "hook":       3,
    "refrain":    3,
    "outro":      4,
    "end":        4,
    "coda":       4,
    "fade-out":   4,
    "fadeout":    4,
    "fade":       4,
}


def load_harmonix_annotation(file_id: str, n_seconds: int) -> np.ndarray | None:
    seg_path = os.path.join(SEGMENTS_DIR, f"{file_id}.txt")
    if not os.path.exists(seg_path):
        return None

    boundaries = []
    with open(seg_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            label_str = " ".join(parts[1:]).lower()
            label = None
            for keyword, num in LABEL_MAP.items():
                if keyword in label_str:
                    label = num
                    break
            if label:
                boundaries.append((int(t), label))

    if len(boundaries) < 2:
        return None

    labels = np.zeros(n_seconds, dtype=int)
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else n_seconds
        labels[start:min(end, n_seconds)] = label

    for s in range(n_seconds):
        if labels[s] == 0:
            labels[s] = labels[s - 1] if s > 0 else 1

    return labels


print("=== Processing Harmonix songs ===")

header = "time_s," + ",".join(FEATURE_NAMES) + ",label"
ok, skipped = 0, 0

for mp3_path in sorted(glob.glob(os.path.join(HARMONIX_MP3S, "*.mp3"))):
    file_id = os.path.splitext(os.path.basename(mp3_path))[0]

    features  = extract_energy_features(mp3_path)
    n_seconds = features.shape[0]
    times     = get_timesteps(n_seconds)
    labels    = load_harmonix_annotation(file_id, n_seconds)

    if labels is None:
        print(f"  SKIP {file_id} — no usable annotation")
        skipped += 1
        continue

    csv_path = mp3_path.replace(".mp3", "_features.csv")
    np.savetxt(csv_path, np.column_stack([times, features, labels]),
               delimiter=",", header=header, comments="", fmt="%.6f")

    ok += 1
    print(f"  OK   {file_id} — {n_seconds}s")

print(f"\n=== Done: {ok} processed, {skipped} skipped ===")
print(f"Label distribution:  1=intro  2=verse  3=chorus  4=outro")
