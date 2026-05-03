import numpy as np
import librosa

# Parameters from the project spec
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Mel bin boundaries for low / low-mid / mid / high-mid / high (5 bands)
# With 128 mel bins: roughly split by perceptual frequency ranges
BAND_BOUNDARIES = [0, 20, 40, 70, 100, 128]  # indices into mel bins
BAND_NAMES = ["low", "low_mid", "mid", "high_mid", "high"]


def extract_energy_features(mp3_path: str) -> np.ndarray:
    """
    Load an MP3 and return a (T, 5) array of per-timestep band energies.

    Each row is one hop (~23ms at SR=22050, hop=512) and the 5 columns are
    mean mel-spectrogram energy in: low, low-mid, mid, high-mid, high bands.
    Values are in dB (log scale, normalized to the track peak).
    """
    y, sr = librosa.load(mp3_path, sr=SR, mono=True)

    # Compute mel-spectrogram (power)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )  # shape: (N_MELS, T)

    # Average energy across each frequency band for every timestep
    features = np.zeros((mel.shape[1], len(BAND_NAMES)))  # (T, 5)
    for i in range(len(BAND_NAMES)):
        lo, hi = BAND_BOUNDARIES[i], BAND_BOUNDARIES[i + 1]
        features[:, i] = mel[lo:hi, :].mean(axis=0)

    # Aggregate frames into 1-second windows by averaging
    frames_per_sec = int(round(SR / HOP_LENGTH))  # ~43 frames per second
    n_seconds = features.shape[0] // frames_per_sec
    seconds_features = np.zeros((n_seconds, len(BAND_NAMES)))
    for s in range(n_seconds):
        start = s * frames_per_sec
        end = (s + 1) * frames_per_sec
        seconds_features[s] = features[start:end].mean(axis=0)
    features = seconds_features

    # Min-max normalize each band column independently to [0, 1]
    col_min = features.min(axis=0)
    col_max = features.max(axis=0)
    features = (features - col_min) / (col_max - col_min + 1e-9) # Present dividing it by 0

    return features  # shape: (n_seconds, 5), values in [0, 1]


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
