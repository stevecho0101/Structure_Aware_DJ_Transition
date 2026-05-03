# Structure-Aware DJ Transition

Music structure detection pipeline using SALAMI annotations and mel-spectrogram features.

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
```

Your folder should look like this:
```
Structure_Aware_DJ_Transition/
├── download_salami.py
├── build_dataset.py
├── spectrogram.py
├── salami-data-public/
└── matching-salami/
```

---

### 3. Install dependencies

Requires **Python 3.10+**

```bash
pip install yt-dlp librosa numpy pandas
```

Also install **ffmpeg** (required by yt-dlp to convert audio):

- **Mac:** `brew install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html and add to PATH
- **Linux:** `sudo apt install ffmpeg`

---

### 4. Download the MP3s

```bash
python download_salami.py
```

Reads `matching-salami/salami_youtube_pairings.csv` to find YouTube matches for each SALAMI track, then downloads and converts them to MP3. Saves up to 150 songs to `salami_mp3s/`. Takes 10–20 minutes depending on your connection.

---

### 5. Generate the dataset

```bash
python build_dataset.py
```

Extracts 5-band mel-spectrogram energy features from each MP3 and matches them to SALAMI structural annotations.

Output:
- `X_train.npy` — feature matrix `(N_samples, 5)`
- `y_train.npy` — label array `(N_samples,)` where `1=intro, 2=verse, 3=chorus, 4=outro`
- `salami_mp3s/salami_<id>_<name>_features.csv` — per-song CSV with timestep features and labels

---

## Notes

- `salami_mp3s/` is not in the repo (1 GB). Run `download_salami.py` to generate it.
- `X_train.npy` and `y_train.npy` are not in the repo. Run `build_dataset.py` to generate them.
