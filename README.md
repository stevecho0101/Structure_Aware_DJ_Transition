# Structure-Aware DJ Transition: Setup and Usage Guide

## Dependencies

**Python 3.10+** required.

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

Folder Structure:

```
Structure_Aware_DJ_Transition/
- train_cnn_lstm.py
- spectrogram.py
- build_salami_dataset.py
- build_harmonix_dataset.py
- download_salami.py
- download_harmonix.py
- agent.py
- app.py
- requirements.txt
- salami-data-public/
- matching-salami/
- harmonixset/
- song_playlist/
- static/
  - index.html
```

---

### 2. Install dependencies

Downloading dependencies:

```
pip install -r requirements.txt
```

The `requirements.txt` covers:

- `torch`, `numpy`, `scipy`, `scikit-learn`,`librosa`, `pydub`,`flask`,`anthropic`, `pandas`

**ffmpeg** is also required by `librosa` and `pydub` to decode MP3s. Installation Instruction:

- Mac: `brew install ffmpeg`
- Windows:** Download the essential build from https://www.gyan.dev/ffmpeg/builds/. Access the folder, copy the path of the `/bin` folder, and go to windows->edit system variables->advanced->environment variables->path->new and paste the path

> **Windows PATH note:** If pydub still can't find ffmpeg after adding to PATH, add this to the top of `add_effect.py`, `agent.py`, and `app.py`:
> ```python
> import os
> os.environ["PATH"] = r"C:\path\to\ffmpeg\bin" + ";" + os.environ.get("PATH", "")
> ```
---

### 3. Download the MP3s

**SALAMI**:
```
python download_salami.py
```

**Harmonix**:
```
python download_harmonix.py
```

Both scripts skip existing files and can be resumed. There may be failures due to copyright takedowns.

The script will auto create `salami_mp3s/` and `harmonix_mp3s/`

Downloaded files go into `salami_mp3s/` and `harmonix_mp3s/` respectively

---

### 4. Build the feature datasets

```
python build_salami_dataset.py
python build_harmonix_dataset.py
```

For each MP3, the build files extract 39 per-second audio features and writes a `*_features.csv` alongside each MP3

---

### 5. Train the model

```
python train_cnn_lstm.py
```

Saved outputs:
- `cnn_lstm_best.pt` — best model weights (by validation accuracy)
- `cnn_lstm_stats.npz` — per-feature mean and std used for normalization at inference time

Both files must be in the same directory as `train_cnn_lstm.py` for inference to work

---

### 6. Run inference on a new song

If you wish to run inference on one mp3 use the command line:
```
python train_cnn_lstm.py your_song.mp3
```
Note: Make sure the mp3 file is in the same directory as train_cnn_lstm.py for it to work

---

## Using the Agent and Feedback Loop

The agent integrates the trained model with the Claude API to power a live DJ set with automatic song recommendations and transition effects.

### Setup

**1. Set your Anthropic API key** in `agent.py`:
```python
client = anthropic.Anthropic(api_key="your_key_here")
```
For Grading: If API key is needed, please contact Steve Cho (smcho@usc.edu)

**2. Add songs to the sample pool:**:
Drop any MP3 files into `song_playlist/`

Every file in this folder is automatically picked up by both the agent and the UI

**3. Cache CNN labels**:

Once sufficient songs are inside `song_playlist/` use the command line:
```
python agent.py
```
This runs the CNN on every song in `song_samples/` and writes predictions to `labels_cache.json`, such that we don't have to rerun the model everytime the app.py is ran

**4. Start the web app:**
```
python app.py
```
Open `http://localhost:5000` in your browser.

---
