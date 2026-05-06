"""
CNN + Bidirectional LSTM for song structure segmentation.

Input features: 5 per-second energy bands (low, low_mid, mid, high_mid, high)
Output labels:  1=intro  2=verse  3=chorus  4=outro  (per second)

Architecture:
  1D CNN  — detects local spectral patterns (ramps, drops) in a short window
  BiLSTM  — captures long-range structure dependencies across the full song
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_GLOB = [
    os.path.join(BASE, "salami_mp3s",   "*_features.csv"),
    os.path.join(BASE, "harmonix_mp3s", "*_features.csv"),
]

N_FEATURES = 39  # 5 bands + 13 MFCCs + 12 chroma + 6 contrast + 1 rolloff + 1 zcr + 1 flatness
N_CLASSES = 4
SEQ_LEN = 128  # seconds per training window
STRIDE  = 64   # overlap windows by 50%
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
EARLY_STOP_PATIENCE = 15


# ── Dataset ───────────────────────────────────────────────────────────────────

class SongDataset(Dataset):
    """Chops each song CSV into fixed-length windows for batching."""

    def __init__(self, csv_paths: list[str], seq_len: int = SEQ_LEN, stride: int = STRIDE):
        self.windows: list[tuple[torch.Tensor, torch.Tensor]] = []
        for path in csv_paths:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            X = data[:, 1:-1].astype(np.float32)        # (T, N_FEATURES)
            y = data[:, -1].astype(np.int64) - 1        # (T,)  0..3

            T = len(X)
            for start in range(0, T, stride):
                end = start + seq_len
                if end <= T:
                    X_win = X[start:end]
                    y_win = y[start:end]
                else:
                    # Pad the last short window; -1 is ignored by the loss
                    pad = end - T
                    X_win = np.vstack([X[start:], np.zeros((pad, N_FEATURES), np.float32)])
                    y_win = np.concatenate([y[start:], np.full(pad, -1, np.int64)])

                self.windows.append((torch.from_numpy(X_win), torch.from_numpy(y_win)))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i]


# ── Model ─────────────────────────────────────────────────────────────────────

class CnnLstm(nn.Module):
    """
    1D CNN extracts local temporal patterns from the 5-band energy sequence.
    Bidirectional LSTM then models long-range structure (verse → chorus → outro).
    """

    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1 — short-range patterns (~3 s)
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Block 2 — medium-range patterns (~5 s)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,   # sees past AND future context
        )

        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(256, n_classes)   # 128 * 2 directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 5)
        x = x.permute(0, 2, 1)          # → (batch, 5, seq_len)
        x = self.cnn(x)                  # → (batch, 64, seq_len)
        x = x.permute(0, 2, 1)          # → (batch, seq_len, 64)
        x, _ = self.lstm(x)              # → (batch, seq_len, 256)
        x = self.dropout(x)
        return self.head(x)              # → (batch, seq_len, n_classes)


# ── Training ──────────────────────────────────────────────────────────────────

LABEL_NAMES = {0: "intro", 1: "verse", 2: "chorus", 3: "outro"}


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct = np.zeros(N_CLASSES, dtype=int)
    per_class_total   = np.zeros(N_CLASSES, dtype=int)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if train:
                X_batch = X_batch + torch.randn_like(X_batch) * 0.05

            logits = model(X_batch)                          # (B, T, C)
            loss = criterion(logits.view(-1, N_CLASSES), y_batch.view(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            flat_y     = y_batch.view(-1)
            mask       = flat_y != -1
            preds      = logits.view(-1, N_CLASSES).argmax(dim=-1)
            correct   += (preds[mask] == flat_y[mask]).sum().item()
            total     += mask.sum().item()

            for c in range(N_CLASSES):
                class_mask = mask & (flat_y == c)
                per_class_total[c]   += class_mask.sum().item()
                per_class_correct[c] += (preds[class_mask] == c).sum().item()

    per_class_acc = np.divide(
        per_class_correct, per_class_total,
        out=np.zeros(N_CLASSES), where=per_class_total > 0
    )
    return total_loss / len(loader), correct / total, per_class_acc


def train():
    csv_paths = sorted(path for pattern in CSV_GLOB for path in glob.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No feature CSVs found in salami_mp3s/ or harmonix_mp3s/")

    print(f"Loaded {len(csv_paths)} songs")

    train_paths, val_paths = train_test_split(csv_paths, test_size=0.2, random_state=42)
    train_ds = SongDataset(train_paths)
    val_ds   = SongDataset(val_paths)
    print(f"Train windows: {len(train_ds)}   Val windows: {len(val_ds)}")

    # Global z-score normalization using training set statistics
    all_X = torch.cat([X for X, _ in train_ds.windows], dim=0)  # (total_seconds, N_FEATURES)
    g_mean = all_X.mean(dim=0)
    g_std  = all_X.std(dim=0).clamp(min=1e-6)
    train_ds.windows = [((X - g_mean) / g_std, y) for X, y in train_ds.windows]
    val_ds.windows   = [((X - g_mean) / g_std, y) for X, y in val_ds.windows]
    stats_path = os.path.join(BASE, "cnn_lstm_stats.npz")
    np.savez(stats_path, mean=g_mean.numpy(), std=g_std.numpy())
    print(f"Global stats saved: {stats_path}")

    # Count seconds per class in training set to compute inverse-frequency weights
    class_counts = np.zeros(N_CLASSES, dtype=np.float64)
    for _, y in train_ds:
        for c in range(N_CLASSES):
            class_counts[c] += (y == c).sum().item()
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    print("Class counts (train):", {LABEL_NAMES[c]: int(class_counts[c]) for c in range(N_CLASSES)})
    print("Class weights       :", {LABEL_NAMES[c]: f"{class_weights[c]:.2f}" for c in range(N_CLASSES)})

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    model     = CnnLstm().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weights_tensor)

    best_val_acc = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(BASE, "cnn_lstm_best.pt")

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _          = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        vl_loss, vl_acc, vl_cls_acc = run_epoch(model, val_dl,   criterion, optimizer, device, train=False)
        scheduler.step(vl_loss)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            star = " *"
        else:
            epochs_no_improve += 1
            star = ""

        if epoch % 5 == 0 or epoch == 1:
            cls = "  ".join(f"{LABEL_NAMES[c]}={vl_cls_acc[c]:.2f}" for c in range(N_CLASSES))
            print(f"{epoch:5d}  {tr_loss:10.4f}  {tr_acc:9.3f}  {vl_loss:8.4f}  {vl_acc:7.3f}{star}  [{cls}]")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop at epoch {epoch} (no val improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\nBest val accuracy : {best_val_acc:.3f}")
    print(f"Model saved       : {save_path}")


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(mp3_path: str, model_path: str | None = None) -> np.ndarray:
    """
    Predict per-second structure labels for an MP3 not seen during training.
    Returns an integer array of shape (T,) with values 1..4.
    """
    from spectrogram import extract_energy_features

    if model_path is None:
        model_path = os.path.join(BASE, "cnn_lstm_best.pt")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = CnnLstm()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    stats = np.load(os.path.join(BASE, "cnn_lstm_stats.npz"))
    g_mean = torch.tensor(stats["mean"], dtype=torch.float32).to(device)
    g_std  = torch.tensor(stats["std"],  dtype=torch.float32).to(device)

    X = extract_energy_features(mp3_path)                        # (T, 38)
    X_t = torch.from_numpy(X).unsqueeze(0).to(device)           # (1, T, 38)
    X_t = (X_t - g_mean) / g_std                                # global z-score

    with torch.no_grad():
        logits = model(X_t)                                      # (1, T, 4)
        preds = logits.squeeze(0).argmax(dim=-1).cpu().numpy()   # (T,)

    preds = preds + 1   # shift 0..3 → 1..4

    # Median filter to remove single-second flickering (kernel=9s → removes blips <5s)
    from scipy.signal import medfilt
    preds = medfilt(preds.astype(np.float32), kernel_size=9).astype(int)

    return preds


def print_structure(mp3_path: str, model_path: str | None = None):
    """Pretty-print predicted structure with timestamps."""
    LABEL_NAMES = {1: "intro", 2: "verse", 3: "chorus", 4: "outro"}
    preds = predict(mp3_path, model_path)
    print(f"\nPredicted structure for: {os.path.basename(mp3_path)}")
    print("-" * 40)
    prev, start = preds[0], 0
    for t, label in enumerate(preds):
        if label != prev or t == len(preds) - 1:
            end = t if label != prev else t + 1
            m, s = divmod(start, 60)
            print(f"  {m:02d}:{s:02d}  {LABEL_NAMES.get(prev, '?'):8s}  ({prev})")
            prev, start = label, t
    print("-" * 40)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Usage: python train_cnn_lstm.py path/to/song.mp3
        print_structure(sys.argv[1])
    else:
        train()
