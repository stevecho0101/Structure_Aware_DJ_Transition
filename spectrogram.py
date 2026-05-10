import numpy as np
import librosa

# AI Usage: We consulted some potential ways to add more features, as we noticed just using 5 mel energy bands was not
# enough to raise the validation accuracy. Claude suggested adding MFCCs, chroma bins, spectral contrast, spectral rolloff,
# zero-crossing rate, and spectral flatness, which resulted in a much better accuracy.

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

# Takes frame-level features and average them into 1-second window
# Convert frames into seconds as that is what we want for our data
def _aggregate_to_seconds(frames: np.ndarray, frames_per_sec: int) -> np.ndarray:
    n_seconds = frames.shape[0] // frames_per_sec
    out = np.zeros((n_seconds, frames.shape[1]), dtype=np.float32)
    for s in range(n_seconds):
        out[s] = frames[s * frames_per_sec:(s + 1) * frames_per_sec].mean(axis=0)
    return out

# Load an MP3 and return a (T, 41) array of per-second features:
#       - 5  mel energy bands     Shows energy of each of the frequency ranges (low, low-mid, mid, high-mid, high).
#                                 Captures how bright and heavy the songs sound
#       - 13 MFCCs                Mel-Frequency Cepstral Coefficients: Describe the shape of the sound's spectrum
#                                 Shows what the tone color is (how different each sound is)
#       - 12 chroma bins          Captures what notes/chords are played as a chorus, for example, return to the same chords
#                                 Show how strong each of the 12 notes is
#       - 6  spectral contrast    Captures the difference between the loud and quiet parts of the spectrum within each frequency sub-band
#                                 High contrast = intro/verse as there are fewer instruments and some frequencies dominate
#       - 1  spectral rolloff     Identifies the single frequency point below which 85% of the total energy sits
#                                 Higher = brighter sounding section
#       - 1  zero-crossing rate   Shows how often the audio waveform crosses zero per second
#                                 High ZCR = Waveform is oscillating a lot, which means noisy, percussive sounds
#                                 Low ZCR = Waveform is less oscillating, most likely a bass note, a pad, or a vocal held note
#       - 1 spectral flatness     Captures how noise-like or tone-like the audio is
#                                 High flatness = energy is spread across all frequencies, so it is more likely a noise
#                                 Low flatness = energy is focused at specific frequencies, so more likely a clear note, melody, or vocal
#     All columns are min-max normalized to [0, 1] per song.
def extract_energy_features(mp3_path: str) -> np.ndarray:
    y, sr = librosa.load(mp3_path, sr=SR, mono=True)
    frames_per_sec = int(round(SR / HOP_LENGTH))  # ~43

    # 5 mel energy bands
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )  # Shape ->(N_MELS, T): (Number of mel frequnecy bins, Number of time frames)
    band_frames = np.zeros((mel.shape[1], len(BAND_NAMES)), dtype=np.float32)
    for i in range(len(BAND_NAMES)):
        lo, hi = BAND_BOUNDARIES[i], BAND_BOUNDARIES[i + 1]
        band_frames[:, i] = mel[lo:hi, :].mean(axis=0)
    band_sec = _aggregate_to_seconds(band_frames, frames_per_sec)

    # 13 MFCCs
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).T  # Shape -> (T, 13)
    mfcc_sec = _aggregate_to_seconds(mfcc.astype(np.float32), frames_per_sec)

    # 12 chroma bins
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).T  # Shape -> (T, 12)
    chroma_sec = _aggregate_to_seconds(chroma.astype(np.float32), frames_per_sec)

    # 6 spectral contrast bands
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=6
    ).T  # Shape -> (T, 6) — librosa returns (n_bands+1, T), but we use first 6
    contrast_sec = _aggregate_to_seconds(contrast[:, :6].astype(np.float32), frames_per_sec)

    # 1 spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).T  # (T, 1)
    rolloff_sec = _aggregate_to_seconds(rolloff.astype(np.float32), frames_per_sec)

    # 1 zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH).T  # (T, 1)
    zcr_sec = _aggregate_to_seconds(zcr.astype(np.float32), frames_per_sec)

    # 1 spectral flatness 
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH).T  # (T, 1)
    flat_sec = _aggregate_to_seconds(flatness.astype(np.float32), frames_per_sec)

    # Combine and normalize
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

# Returns integer second timestamps
def get_timesteps(n_seconds: int) -> np.ndarray:
    return np.arange(n_seconds)
