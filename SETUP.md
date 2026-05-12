# Structure-Aware DJ Transition — Setup & Usage Guide

CNN + Bidirectional LSTM pipeline that detects per-second song structure (intro / verse / chorus / outro), then uses a Claude-powered agent to pick the best next song and transition effect for a live DJ set.

---

## Dependencies

**Python 3.10+** is required.

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` covers:

| Package | Purpose |
|---|---|
| `torch`, `numpy`, `scipy`, `scikit-learn` | Model training and inference |
| `librosa` | Audio feature extraction |
| `pydub` | Transition audio rendering |
| `flask` | Web app server |
| `anthropic` | Claude API for the DJ agent |
| `pandas` | Dataset label parsing |

**ffmpeg** is also required by both `librosa` and `pydub` to decode MP3s. Install it separately:

- **Mac:** `brew install ffmpeg`
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` folder to your system PATH
- **Linux:** `sudo apt install ffmpeg`

> **Windows PATH note:** If pydub still can't find ffmpeg after adding to PATH, add this to the top of `add_effect.py`, `agent.py`, and `app.py`:
> ```python
> import os
> os.environ["PATH"] = r"C:\path\to\ffmpeg\bin" + ";" + os.environ.get("PATH", "")
> ```

---

## Steps to Reproduce Results

### 1. Clone the repo and datasets

```bash
git clone https://github.com/stevecho0101/Structure_Aware_DJ_Transition.git
cd Structure_Aware_DJ_Transition

git clone https://github.com/DDMAL/salami-data-public.git
git clone https://github.com/jblsmith/matching-salami.git
git clone https://github.com/urinieto/harmonixset.git
```

Your folder should look like this:

```
Structure_Aware_DJ_Transition/
├── train_cnn_lstm.py
├── spectrogram.py
├── build_salami_dataset.py
├── build_harmonix_dataset.py
├── download_salami.py
├── download_harmonix.py
├── agent.py
├── app.py
├── add_effect.py
├── requirements.txt
├── salami-data-public/
├── matching-salami/
└── harmonixset/
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Download the MP3s

**SALAMI** (~237 usable songs, mixed genres):
```bash
python download_salami.py
```

**Harmonix** (~841 pop songs):
```bash
python download_harmonix.py
```

Both scripts skip already-downloaded files and can be resumed. Expect some failures due to copyright takedowns. Combined download takes 1–3 hours depending on connection speed.

Downloaded files go into `salami_mp3s/` and `harmonix_mp3s/` respectively.

---

### 4. Build the feature datasets

```bash
python build_salami_dataset.py
python build_harmonix_dataset.py
```

For each MP3, this extracts 39 per-second audio features and writes a `*_features.csv` alongside each MP3:

| Feature group | Count | What it captures |
|---|---|---|
| Mel energy bands | 5 | Low→high frequency energy (loudness profile) |
| MFCCs | 13 | Timbre and tone color |
| Chroma bins | 12 | Harmonic content / chord identity |
| Spectral contrast | 6 | Peak vs. valley energy per sub-band |
| Spectral rolloff | 1 | Brightness of the section |
| Zero-crossing rate | 1 | Percussiveness vs. tonal content |
| Spectral flatness | 1 | Noise-like vs. pitched content |

Labels: `1 = intro`, `2 = verse/bridge`, `3 = chorus/hook`, `4 = outro`

---

### 5. Train the model

```bash
python train_cnn_lstm.py
```

This reads all `*_features.csv` files from both dataset folders, performs an 80/20 train/val split, applies global z-score normalization, and trains the CNN + BiLSTM for up to 100 epochs with early stopping (patience = 15).

Saved outputs:
- `cnn_lstm_best.pt` — best model weights (by validation accuracy)
- `cnn_lstm_stats.npz` — per-feature mean and std used for normalization at inference time

Both files must remain in the same directory as `train_cnn_lstm.py` for inference to work.

**Model architecture:**
```
Input (T × 39)
  → Conv1d(39→32, k=3) + BatchNorm + ReLU + Dropout(0.3)
  → Conv1d(32→64, k=5) + BatchNorm + ReLU + Dropout(0.3)
  → BiLSTM(hidden=128, layers=2, dropout=0.5)
  → Dropout(0.5)
  → Linear(256→4)
Output (T × 4) — per-second class scores
```

Expected validation accuracy: ~52% on the combined SALAMI + Harmonix dataset.

---

### 6. Run inference on a new song

From the command line:
```bash
python train_cnn_lstm.py your_song.mp3
```

Or from Python:
```python
from train_cnn_lstm import predict, print_structure

print_structure("your_song.mp3")   # prints timestamped segment table

labels = predict("your_song.mp3")  # returns int array of 1..4 per second
```

---

## Using the Agent and Feedback Loop

The agent integrates the trained model with the Claude API to power a live DJ set with automatic song recommendations and transition effects.

### Setup

**1. Set your Anthropic API key** in `agent.py`:
```python
client = anthropic.Anthropic(api_key="your_key_here")
```

**2. Add songs to the sample pool:**
```bash
mkdir song_samples
```
Drop any MP3 files into `song_samples/`. Every file in this folder is automatically picked up by both the agent and the UI.

**3. Cache CNN labels** (run once, or whenever new songs are added):
```bash
python agent.py
```
This runs the CNN on every song in `song_samples/` and writes predictions to `labels_cache.json`. Caching avoids re-running the model on every page load — subsequent runs skip songs already in the cache.

**4. Start the web app:**
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

### How the Agent Works

When you pick a song and hit **Play**, the agent receives:
- The per-second structural labels of the **current song** (from the cache)
- The structural labels of all **candidate songs** in the pool
- A list of **recently played songs** (last 3 are in cooldown and excluded)
- Any **feedback** you have submitted from prior transitions in this session

It sends all of this to Claude along with a prompt that instructs it to:
1. Select the single best next song based on energy continuity
2. Pick a transition-out timestamp from the current song (end of a chorus or verse)
3. Pick a transition-in timestamp for the next song (start of a verse or chorus)
4. Choose the best transition effect for the genre pairing

Claude returns a JSON response with the recommendation, timestamps, effect choice, and its reasoning.

---

### Feedback Loop

After each transition plays, a **feedback card** appears in the UI. You can:
- Rate the transition **1–5 stars**
- Leave an optional **text comment**

This feedback is stored in memory for the current session and passed back to Claude on every subsequent call. The agent prompt instructs Claude to change its approach when recent ratings are low — for example, trying a different effect type or prioritizing a different energy level in the next song.

The feedback loop resets when you close or refresh the browser.

---

### Transition Effects

The agent picks one of four effects automatically (or you can override manually in the UI):

| Effect | Best for | What it does |
|---|---|---|
| `crossfade` | Any genre, safe default | Simple volume fade out / fade in with a 2-second overlap |
| `eq_sweep` | Hip-hop, funk, R&B | Cuts bass on the outgoing song, restores it on the incoming song |
| `lpf_sweep` | EDM, electronic | Low-pass filter muffles song 1 out while song 2 opens up |
| `beatmatch` | Songs with steady BPM | Adjusts song 2's playback speed to match song 1's BPM before fading in |

You can also apply effects manually from the command line:
```bash
python add_effect.py song1.mp3 120 song2.mp3 8 output.mp3 crossfade
python add_effect.py song1.mp3 120 song2.mp3 8 output.mp3 eq_sweep
python add_effect.py song1.mp3 120 song2.mp3 8 output.mp3 lpf_sweep
python add_effect.py song1.mp3 120 song2.mp3 8 output.mp3 beatmatch --bpm1 128 --bpm2 124
```

Arguments: `song1_path  song1_exit_sec  song2_path  song2_entry_sec  output_path  mode`
