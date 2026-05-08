# Structure-Aware DJ Transition

Music structure detection pipeline using a CNN + Bidirectional LSTM trained on SALAMI and Harmonix dataset annotations. Predicts per-second structural labels (intro, verse, chorus, outro) for any song.

---

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/stevecho0101/Structure_Aware_DJ_Transition.git
cd Structure_Aware_DJ_Transition
```

---

### 2. Clone the required datasets (into the same folder)

```bash
git clone https://github.com/DDMAL/salami-data-public.git
git clone https://github.com/jblsmith/matching-salami.git
git clone https://github.com/urinieto/harmonixset.git
```

Your folder should look like this:
```
Structure_Aware_DJ_Transition/
├── download_salami.py
├── download_harmonix.py
├── build_dataset.py
├── build_harmonix_dataset.py
├── train_cnn_lstm.py
├── spectrogram.py
├── salami-data-public/
├── matching-salami/
└── harmonixset/
```

---

### 3. Install dependencies

Requires **Python 3.10+**

```bash
pip install torch librosa scipy scikit-learn numpy pandas yt-dlp
```

Also install **ffmpeg** (required by yt-dlp to convert audio):

- **Mac:** `brew install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html and add to PATH
- **Linux:** `sudo apt install ffmpeg`

---

### 4. Download the MP3s

**SALAMI** (~237 usable songs, mixed genres):
```bash
python download_salami.py
```

**Harmonix** (~500+ pop songs):
```bash
python download_harmonix.py
```

Both scripts skip already-downloaded files so they can be resumed. Expect some failures due to copyright takedowns. Combined download takes 1–3 hours depending on your connection.

---

### 5. Build the datasets

```bash
python build_dataset.py
python build_harmonix_dataset.py
```

Extracts 39 per-second audio features from each MP3 and matches them to structural annotations:
- 5 mel energy bands (low → high)
- 13 MFCCs (timbre)
- 12 chroma bins (harmony)
- 6 spectral contrast bands
- 1 spectral rolloff
- 1 zero-crossing rate
- 1 spectral flatness (vocal proxy)

Labels: `1=intro  2=verse/bridge  3=chorus/hook  4=outro`

---

### 6. Train the model

```bash
python train_cnn_lstm.py
```

Trains a CNN + Bidirectional LSTM on all available CSVs from both datasets combined. Saves:
- `cnn_lstm_best.pt` — best model weights
- `cnn_lstm_stats.npz` — global normalization statistics (required for inference)

---

### 7. Predict structure for a new song

```bash
python train_cnn_lstm.py your_song.mp3
```

Or in Python:
```python
from train_cnn_lstm import predict, print_structure

print_structure("your_song.mp3")

labels = predict("your_song.mp3")  # array of 1..4 per second
```

---

## Model Architecture

```
Input (T, 39)
  → Conv1d(39→32, k=3) + BatchNorm + ReLU + Dropout(0.3)
  → Conv1d(32→64, k=5) + BatchNorm + ReLU + Dropout(0.3)
  → BiLSTM(hidden=128, layers=2, dropout=0.5)
  → Dropout(0.5)
  → Linear(256→4)
Output (T, 4) — per-second class probabilities
```

---

## Notes

- `salami_mp3s/` and `harmonix_mp3s/` are not in the repo. Run the download scripts to generate them.
- `cnn_lstm_best.pt` and `cnn_lstm_stats.npz` must be in the same folder as `train_cnn_lstm.py` for inference to work.
- Val accuracy: ~52% on combined SALAMI + Harmonix dataset. Works best on songs with acoustically distinct sections (e.g. songs where chorus hits noticeably harder than verse).

---

## AI Agent
`agent.py` takes the CNN's predicted labels and uses the Claude API to select the optimal transition point and recommend the best next song based on energy continuity. Tracks played songs so it won't repeat until all songs have been played.

Install additional dependencies:
```
pip install anthropic flask
```

Set your Anthropic API key in `agent.py` and `app.py` (ask Anthony for API Key):
```python
client = anthropic.Anthropic(api_key="your_key_here") 
```


Create the song samples folder and add a few mp3s:
```
mkdir song_samples
```
Drop any mp3 files into `song_samples/` — they will automatically show up in the UI and be used by the agent.

> **Windows note:** If pydub can't find ffmpeg even after adding to PATH, add this to the top of `add_effect.py` and `app.py` and `agent.py` (I had to do this):
> ```python
> import os
> os.environ["PATH"] = r"C:\path\to\ffmpeg\bin" + ";" + os.environ.get("PATH", "")
> ```

Run the agent directly:
```
python agent.py
```

---

## Live UI
Browser-based live player — pick a song, hit Analyze, and it automatically crossfades into the recommended next song at the right timestamp. Added a feature to rate and give feedback to the agent for it to try and give another transition selection.

```
python app.py
```
Open `http://localhost:5000`.
