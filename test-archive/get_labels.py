import os
import re
import json
import numpy as np
import librosa
import lyricsgenius

# ── API credentials ────────────────────────────────────────────────────────────
GENIUS_TOKEN = "Kr6am66x4j5GkgmCWtl4_bN3kqiUL-_xQQBNzZHdGXkmplaCwIfaSvwUDVG_klEX"

# Spotify Audio Analysis was deprecated in late 2024 — using librosa instead

# ── Label mapping ──────────────────────────────────────────────────────────────
# Maps Genius section header keywords → numeric label
LABEL_MAP = {
    "intro":      1,
    "verse":      2,
    "pre-chorus": 2,
    "bridge":     2,
    "chorus":     3,
    "hook":       3,
    "refrain":    3,
    "outro":      4,
    "end":        4,
}

def detect_vocal_presence(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Returns a per-second array of vocal presence scores (0=no vocals, 1=vocals).
    Uses harmonic-percussive separation — vocals are harmonic and sit in 300Hz-3kHz.
    """
    # Separate harmonic (tonal) component — vocals live here
    y_harmonic, _ = librosa.effects.hpss(y)

    # Mel spectrogram of harmonic component only
    mel = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, hop_length=hop_length, n_mels=128)

    # Vocal frequency range: ~300Hz–3kHz → find corresponding mel bins
    freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr // 2)
    vocal_bins = np.where((freqs >= 300) & (freqs <= 3000))[0]

    # Average energy in vocal range per frame
    vocal_energy = mel[vocal_bins, :].mean(axis=0)

    # Aggregate to 1-second windows
    frames_per_sec = int(round(sr / hop_length))
    n_seconds = len(vocal_energy) // frames_per_sec
    vocal_per_sec = np.array([
        vocal_energy[s * frames_per_sec:(s + 1) * frames_per_sec].mean()
        for s in range(n_seconds)
    ])

    # Normalize to [0, 1]
    v_min, v_max = vocal_per_sec.min(), vocal_per_sec.max()
    return (vocal_per_sec - v_min) / (v_max - v_min + 1e-9)


def get_audio_sections(mp3_path: str, n_sections: int) -> list[dict]:
    """
    Detect exactly n_sections boundaries from the MP3 using librosa.
    n_sections is passed in from Genius so counts always match for direct mapping.
    """
    y, sr = librosa.load(mp3_path, sr=22050, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    R = librosa.segment.recurrence_matrix(mfcc, mode="affinity", sym=True)
    # k = n_sections - 1 because agglomerative returns interior boundaries (excludes 0 and end)
    bounds = librosa.segment.agglomerative(R, k=max(1, n_sections - 1))
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=512)

    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    duration = librosa.get_duration(y=y, sr=sr)
    bound_times = np.concatenate([[0.0], bound_times, [duration]])

    sections = []
    for i in range(len(bound_times) - 1):
        start = bound_times[i]
        end   = bound_times[i + 1]
        start_frame = librosa.time_to_frames(start, sr=sr, hop_length=512)
        end_frame   = librosa.time_to_frames(end,   sr=sr, hop_length=512)
        loudness = float(np.mean(rms[start_frame:end_frame]))
        sections.append({"start": start, "duration": end - start, "loudness": loudness})

    return sections


def get_genius_sections(artist: str, title: str) -> list[str]:
    """Return ordered list of section label names from Genius lyrics headers."""
    genius = lyricsgenius.Genius(GENIUS_TOKEN, remove_section_headers=False)
    song = genius.search_song(title, artist)
    if not song:
        print(f"  [Genius] Could not find: {artist} - {title}")
        return []

    # Extract all [Section Name] headers in order
    headers = re.findall(r"\[([^\]]+)\]", song.lyrics)
    labels = []
    for h in headers:
        h_lower = h.lower()
        matched = None
        for keyword, label in LABEL_MAP.items():
            if keyword in h_lower:
                matched = label
                break
        if matched:
            labels.append(matched)
    return labels


def match_sections(spotify_sections: list[dict], genius_labels: list[int]) -> dict:
    """
    Map Genius labels onto Spotify timestamps.
    If counts match: direct 1-to-1 mapping.
    If not: use heuristics (position + loudness) to infer labels.
    """
    n_spotify = len(spotify_sections)
    n_genius  = len(genius_labels)
    labels_out = {}

    if n_spotify == 0:
        return labels_out

    if n_genius > 0 and n_spotify == n_genius:
        # Perfect match — direct mapping
        for i, section in enumerate(spotify_sections):
            labels_out[int(section["start"])] = genius_labels[i]

    else:
        # Fallback heuristics using position and loudness
        print(f"  [Match] Spotify={n_spotify} sections, Genius={n_genius} — using heuristics")
        loudnesses = [s["loudness"] for s in spotify_sections]
        max_loud   = max(loudnesses)
        min_loud   = min(loudnesses)

        for i, section in enumerate(spotify_sections):
            start = int(section["start"])
            loud  = section["loudness"]
            # Normalize loudness to 0-1
            norm  = (loud - min_loud) / (max_loud - min_loud + 1e-9)

            if i == 0:
                label = 1  # intro
            elif i == n_spotify - 1:
                label = 4  # outro
            elif norm > 0.75:
                label = 3  # chorus (loudest sections)
            else:
                label = 2  # verse

            labels_out[start] = label

    return labels_out


def annotate_song(mp3_path: str, artist: str, title: str) -> dict:
    """Full pipeline: detect boundaries + fetch Genius labels, return {start_sec: label}."""
    print(f"\nAnnotating: {artist} - {title}")
    genius_labels  = get_genius_sections(artist, title)
    n = len(genius_labels) if genius_labels else 8
    audio_sections = get_audio_sections(mp3_path, n_sections=n)
    labels = match_sections(audio_sections, genius_labels)

    # Use vocal presence to fix intro detection:
    # if the first section has low vocals, it's an intro regardless of Genius label
    y, sr = librosa.load(mp3_path, sr=22050, mono=True)
    vocal_scores = detect_vocal_presence(y, sr)
    first_sec = min(labels.keys())
    if vocal_scores[first_sec] < 0.15:
        labels[first_sec] = 1  # force intro

    return labels


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))

    # Add songs here as (mp3_filename, artist, title)
    songs = [
        ("FE!N ft. Playboi Carti - Travis Scott.mp3", "Travis Scott", "FE!N"),
        ("STAY - Justin Bieber & The Kid Laroi.mp3",  "The Kid Laroi", "STAY"),
    ]

    output = {}
    for mp3_file, artist, title in songs:
        mp3_path = os.path.join(folder, mp3_file)
        if not os.path.exists(mp3_path):
            print(f"\nSkipping {title} — MP3 not found: {mp3_path}")
            continue
        annotations = annotate_song(mp3_path, artist, title)
        key = f"{title} - {artist}"
        output[key] = annotations
        print(f"  → {len(annotations)} sections found")

    with open("labels.json", "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved labels.json with {len(output)} songs.")
