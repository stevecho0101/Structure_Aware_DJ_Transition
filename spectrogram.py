import numpy as np
import librosa

# Parameters from the project spec
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 13

# Mel bin boundaries for low / low-mid / mid / high-mid / high (5 bands)
BAND_BOUNDARIES = [0, 20, 40, 70, 100, 128]  # indices into mel bins
BAND_NAMES     = ["low", "low_mid", "mid", "high_mid", "high"]
MFCC_NAMES     = [f"mfcc_{i+1}" for i in range(N_MFCC)]
CHROMA_NAMES   = [f"chroma_{i+1}" for i in range(12)]
CONTRAST_NAMES = [f"contrast_{i+1}" for i in range(6)]
EXTRA_NAMES    = ["rolloff", "zcr", "flatness"]
# 5 + 13 + 12 + 6 + 3 = 39
FEATURE_NAMES  = BAND_NAMES + MFCC_NAMES + CHROMA_NAMES + CONTRAST_NAMES + EXTRA_NAMES


def _aggregate_to_seconds(frames: np.ndarray, frames_per_sec: int) -> np.ndarray:
    """Average frame-level features into 1-second windows."""
    n_seconds = frames.shape[0] // frames_per_sec
    out = np.zeros((n_seconds, frames.shape[1]), dtype=np.float32)
    for s in range(n_seconds):
        out[s] = frames[s * frames_per_sec:(s + 1) * frames_per_sec].mean(axis=0)
    return out


def extract_energy_features(mp3_path: str) -> np.ndarray:
    """
    Load an MP3 and return a (T, 41) array of per-second features:
      - 5  mel energy bands      (low → high)
      - 13 MFCCs                 (timbre / tonal texture)
      - 12 chroma bins           (harmonic content)
      - 6  spectral contrast     (peak-valley difference per sub-band)
      - 1  spectral rolloff      (brightness)
      - 1  zero-crossing rate    (percussiveness / noisiness)
    All columns are min-max normalized to [0, 1] per song.
    """
    y, sr = librosa.load(mp3_path, sr=SR, mono=True)
    frames_per_sec = int(round(SR / HOP_LENGTH))  # ~43

    # ── 5 mel energy bands ────────────────────────────────────────────────────
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )  # (N_MELS, T)
    band_frames = np.zeros((mel.shape[1], len(BAND_NAMES)), dtype=np.float32)
    for i in range(len(BAND_NAMES)):
        lo, hi = BAND_BOUNDARIES[i], BAND_BOUNDARIES[i + 1]
        band_frames[:, i] = mel[lo:hi, :].mean(axis=0)
    band_sec = _aggregate_to_seconds(band_frames, frames_per_sec)

    # ── 13 MFCCs ──────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).T  # (T, 13)
    mfcc_sec = _aggregate_to_seconds(mfcc.astype(np.float32), frames_per_sec)

    # ── 12 chroma bins ────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).T  # (T, 12)
    chroma_sec = _aggregate_to_seconds(chroma.astype(np.float32), frames_per_sec)

    # ── 6 spectral contrast bands ─────────────────────────────────────────────
    # Measures peak-vs-valley energy difference — high in dense chorus sections,
    # low in sparse intros/verses.
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=6
    ).T  # (T, 6) — librosa returns (n_bands+1, T), we use first 6
    contrast_sec = _aggregate_to_seconds(contrast[:, :6].astype(np.float32), frames_per_sec)

    # ── 1 spectral rolloff ────────────────────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).T  # (T, 1)
    rolloff_sec = _aggregate_to_seconds(rolloff.astype(np.float32), frames_per_sec)

    # ── 1 zero-crossing rate ──────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH).T  # (T, 1)
    zcr_sec = _aggregate_to_seconds(zcr.astype(np.float32), frames_per_sec)

    # ── 1 spectral flatness ───────────────────────────────────────────────────
    # Low = tonal (vocals/melody); high = noisy (drums/static).
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH).T  # (T, 1)
    flat_sec = _aggregate_to_seconds(flatness.astype(np.float32), frames_per_sec)

    # ── Combine and normalize ─────────────────────────────────────────────────
    n_seconds = min(
        band_sec.shape[0], mfcc_sec.shape[0], chroma_sec.shape[0],
        contrast_sec.shape[0], rolloff_sec.shape[0], zcr_sec.shape[0],
        flat_sec.shape[0],
    )
    features = np.hstack([
        band_sec[:n_seconds],
        mfcc_sec[:n_seconds],
        chroma_sec[:n_seconds],
        contrast_sec[:n_seconds],
        rolloff_sec[:n_seconds],
        zcr_sec[:n_seconds],
        flat_sec[:n_seconds],
    ])  # (n_seconds, 39)

    col_min = features.min(axis=0)
    col_max = features.max(axis=0)
    features = (features - col_min) / (col_max - col_min + 1e-9)

    return features  # (n_seconds, 39), values in [0, 1]


def get_timesteps(n_seconds: int) -> np.ndarray:
    """Return integer second timestamps [0, 1, 2, ..., n_seconds-1]."""
    return np.arange(n_seconds)


if __name__ == "__main__":
    import glob
    import os

    mp3_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.mp3"))
    if not mp3_files:
        print("No MP3 files found in the same folder.")
        exit(1)
    path = mp3_files[0]
    features = extract_energy_features(path)
    times = get_timesteps(features.shape[0])

    # Save to CSV: columns are time_s, low, low_mid, mid, high_mid, high
    out_path = os.path.splitext(path)[0] + "_features.csv"
    header = "time_s," + ",".join(BAND_NAMES)
    data = np.column_stack([times, features])
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"Saved features to {out_path}")

    print(f"\nSong: {path}")
    print(f"Frames: {features.shape[0]}, Bands: {features.shape[1]}")
    print(f"Duration: {times[-1]:.1f}s")
    print(f"\nFirst 5 timesteps (columns = {BAND_NAMES}):")
    for i in range(min(5, len(features))):
        print(f"  t={times[i]:.3f}s  {np.round(features[i], 4)}")
    print(f"\nLast 5 timesteps:")
    for i in range(max(0, len(features) - 5), len(features)):
        print(f"  t={times[i]:.3f}s  {np.round(features[i], 4)}")
