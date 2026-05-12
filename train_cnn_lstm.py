# CNN + Bidirectional LSTM for song structure segmentation.

# Input features: 39 features
# Output labels:  1=intro  2=verse  3=chorus  4=outro  (per second)

# --Code Pipeline--
# Data Loading: Read all the CSV files inside salami_mp3s/ and harmonix_mp3s/
# where each row of the CSV files consists of 39 features for that one second

# Windowing: As some songs are longer than others, SongDataset splits each song
# into 128 second 50% overlapping windows, ensuring fixed-size inputs

# Nomalization: Computes statistics like the mean and standard deviation of each feature
# across all training seconds. Those are used to normalize each feature such that one single 
# feature does not dominate. Stats saved as cnn_lstm_stats.npz

# Class Weighting: Counts how many seconds belong to which class. As intro and outro
# occur less than the chorus and verse, it ensures they get higher weights, allowing the model
# to learn all 4 classes rather than just guessing verse and chorus.

# Training Loop: For each epoch, a forward pass through the CNN and BiLSTM is done. One batch consists
# of 16 windows, where each window is 128 seconds of a random section of a song. Those windows output predictions
# represented as a vector of size 4 (4 classes) per second. We then compute the loss and go through backpropagation, and compute
# gradients. Update the weights, then validate on a separate set of allocated songs. Utilize Early Stop if the validation error
# stays the same or worse for 15 epochs.

# Inference: Given a new MP3 file from the user, 39 features are extracted using spectrogram.py. Those data are normalized
# using cnn_lstm_stats.npz. Run through the model, giving prediction class scores per second. Median filters remove 
# any flickering and make sure the last segment is the outro.

# Architecture:
#   1D CNN  — detects local spectral patterns (ramps, drops) in a short window
#   BiLSTM  — captures long-range structure dependencies across the full song

# AI Usage:
# We prompted to Claude to provide a starting block to test different implementations. We also asked to help add LSTM because it was one of the alternative strategies we mentioned in the proposal.
# Claude also helped with syntax and with writing the script efficiently, as incorrect logic could lead to longer runtime/training.

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import medfilt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from spectrogram import extract_energy_features

BASE = os.path.dirname(__file__) or "."  # path to this script's directory (relative)
CSV_GLOB = [
    os.path.join(BASE, "salami_mp3s",   "*_features.csv"),   # SALAMI feature files
    os.path.join(BASE, "harmonix_mp3s", "*_features.csv"),   # Harmonix feature files
]

N_FEATURES = 39 # 5 bands + 13 MFCCs + 12 chroma + 6 contrast + 1 rolloff + 1 zcr + 1 flatness
N_CLASSES = 4 # intro, verse, chorus, outro
SEQ_LEN = 128 # number of seconds per training window
STRIDE  = 64 # step size between windows
BATCH_SIZE = 16 # number of windows per gradient update
EPOCHS = 100 # maximum training epochs before stopping
LR = 1e-3 # initial learning rate for Adam optimizer
EARLY_STOP_PATIENCE = 15 # stop if val accuracy doesn't improve for this many epochs


# Data Loading
# Splits each song CSB into fixed-legnth windows for batching
# As songs are different lengths, there must be some overlap between to ensure fixed-size inputs
class SongDataset(Dataset):

    def __init__(self, csv_paths: list[str], seq_len: int = SEQ_LEN, stride: int = STRIDE):
        self.windows: list[tuple[torch.Tensor, torch.Tensor]] = [] # stores all (X, y) window pairs
        for path in csv_paths:
            data = np.loadtxt(path, delimiter=",", skiprows=1) # load CSV, skip header row
            X = data[:, 1:-1].astype(np.float32) # skip time_s and label
            y = data[:, -1].astype(np.int64) - 1 # last column is label; shift 1..4 --> 0..3 for PyTorch

            T = len(X) # total seconds in this song
            for start in range(0, T, stride): # slide window across song with given stride
                end = start + seq_len
                if end <= T: # full window fits, use it directly
                    X_win = X[start:end]
                    y_win = y[start:end]
                else:
                    # last window is short, so pad with zeros so it's the same size as others
                    pad = end - T
                    X_win = np.vstack([X[start:], np.zeros((pad, N_FEATURES), np.float32)])
                    y_win = np.concatenate([y[start:], np.full(pad, -1, np.int64)])  # -1 = ignored by loss

                self.windows.append((torch.from_numpy(X_win), torch.from_numpy(y_win)))  # store as tensors

    def __len__(self) -> int:
        return len(self.windows) # total number of windows across all songs

    def __getitem__(self, i):
        return self.windows[i] # return (X_window, y_window) for index i


# Model 
# 1D CNN extracts local temporal patterns from the 39-feature sequence.
# Bidirectional LSTM then models long-range structure (verse -> chorus -> outro).
class CnnLstm(nn.Module):

    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1: detects short-range patterns (~3 seconds)
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1), # 1D convolution over time
            nn.BatchNorm1d(32), # normalizes activations to stabilize training
            nn.ReLU(), # non-linearity: keep only positive activations
            nn.Dropout(0.3), # randomly zero 30% of neurons to reduce overfitting
            # Block 2: detects medium-range patterns (~5 seconds)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # wider kernel sees more context
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.lstm = nn.LSTM(
            input_size=64, # matches CNN output channels
            hidden_size=128, # internal memory size per direction
            num_layers=2, # two stacked LSTM layers for deeper reasoning
            batch_first=True, # input shape is (batch, seq, features) not (seq, batch, features)
            dropout=0.5, # dropout between LSTM layers to reduce overfitting
            bidirectional=True, # runs forward AND backward so each second sees full song context
        )

        self.dropout = nn.Dropout(0.5) # additional dropout before classification head
        self.head = nn.Linear(256, n_classes) # 128 forward + 128 backward = 256 → 4 class scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # CNN expects (batch, channels, time) so swap features and time
        x = self.cnn(x) # extract local patterns
        x = x.permute(0, 2, 1) # LSTM expects (batch, time, features) so swap back
        x, _ = self.lstm(x) # model long-range structure
        x = self.dropout(x) # regularize before final prediction
        return self.head(x) # output one score per class per second 


# Training 
LABEL_NAMES = {0: "intro", 1: "verse", 2: "chorus", 3: "outro"}  # maps class index to readable name


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train: # train mode enables dropout/batchnorm updates; eval freezes them
        model.train()
        ctx = torch.enable_grad() # enable gradients only during training
    else:
        model.eval()  
        ctx = torch.no_grad()

    total_loss = 0.0
    correct = 0
    total = 0  

    per_class_correct = np.zeros(N_CLASSES, dtype=int) # correct predictions per class
    per_class_total   = np.zeros(N_CLASSES, dtype=int) # total samples per class

    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device) # move features to GPU/CPU
            y_batch = y_batch.to(device) # move labels to GPU/CPU

            if train:
                X_batch = X_batch + torch.randn_like(X_batch) * 0.05  # add small noise to regularize

            logits = model(X_batch) # forward pass to raw class scores (B, T, C)
            loss = criterion(logits.view(-1, N_CLASSES), y_batch.view(-1))  # flatten time dim for loss, -1 = figure out the dimension automatically

            if train:
                optimizer.zero_grad() # clear gradients from previous step
                loss.backward() # compute gradients via backprop
                nn.utils.clip_grad_norm_(model.parameters(), 1.0) # cap gradient size to prevent explosion
                optimizer.step() # update model weights

            total_loss += loss.item() # accumulate batch loss
            flat_y = y_batch.view(-1) # flatten labels to 1D
            mask = flat_y != -1 # ignore padded positions (label = -1)
            preds = logits.view(-1, N_CLASSES).argmax(dim=-1) # pick highest-score class per second
            correct += (preds[mask] == flat_y[mask]).sum().item() # count correct non-padded predictions
            total += mask.sum().item() # count total non-padded seconds

            for c in range(N_CLASSES):
                class_mask = mask & (flat_y == c) # find all real seconds belonging to class c
                per_class_total[c]   += class_mask.sum().item() # count them
                per_class_correct[c] += (preds[class_mask] == c).sum().item() # count correct ones

    per_class_acc = np.divide(
        per_class_correct, per_class_total,
        out=np.zeros(N_CLASSES), where=per_class_total > 0  # avoid division by zero for unseen classes
    )
    return total_loss / len(loader), correct / total, per_class_acc  # return avg loss, overall acc, per-class acc


def train():
    csv_paths = sorted(path for pattern in CSV_GLOB for path in glob.glob(pattern)) # collect all feature CSVs
    if not csv_paths:
        raise FileNotFoundError(f"No feature CSVs found in salami_mp3s/ or harmonix_mp3s/")

    print(f"Loaded {len(csv_paths)} songs")

    train_paths, val_paths = train_test_split(csv_paths, test_size=0.2, random_state=1) # 80/20 split
    train_ds = SongDataset(train_paths) # build windowed dataset for training songs
    val_ds   = SongDataset(val_paths) # build windowed dataset for validation songs
    print(f"Train windows: {len(train_ds)}   Val windows: {len(val_ds)}")

    # Global z-score normalization using training set statistics
    # Make sure the new songs are normalized the same way as the training data to make sure the 
    # the prediction is valid (We do not want different scale for different data)
    all_X = torch.cat([X for X, _ in train_ds.windows], dim=0)  # stack all training seconds
    g_mean = all_X.mean(dim=0) # per-feature mean across all training seconds
    g_std  = all_X.std(dim=0).clamp(min=1e-6) # per-feature std; clamped to avoid divide-by-zero
    train_ds.windows = [((X - g_mean) / g_std, y) for X, y in train_ds.windows] # normalize train
    val_ds.windows   = [((X - g_mean) / g_std, y) for X, y in val_ds.windows] # normalize val with same stats
    stats_path = os.path.join(BASE, "cnn_lstm_stats.npz")
    np.savez(stats_path, mean=g_mean.numpy(), std=g_std.numpy()) # save stats for inference time
    print(f"Global stats saved: {stats_path}")

    # Count seconds per class in training set to compute inverse-frequency weights
    # Make sure the model trains all four classes as intro and outro are rare (happens only once) and
    # might only focus on verse and chorus
    class_counts = np.zeros(N_CLASSES, dtype=np.float64)
    for _, y in train_ds:
        for c in range(N_CLASSES):
            class_counts[c] += (y == c).sum().item()   # count how many seconds belong to each class
    class_weights = 1.0 / (class_counts + 1e-6)        # rarer classes get higher weight
    class_weights = class_weights / class_weights.sum() * N_CLASSES  # normalize so weights sum to N_CLASSES
    print("Class counts (train):", {LABEL_NAMES[c]: int(class_counts[c]) for c in range(N_CLASSES)})
    print("Class weights       :", {LABEL_NAMES[c]: f"{class_weights[c]:.2f}" for c in range(N_CLASSES)})

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)  # shuffle for training
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # no shuffle for val

    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    model     = CnnLstm().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) # Adam with L2 regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5) # halve LR if val loss plateaus
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) # move class weights to device
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weights_tensor) # weighted loss, ignore padded positions

    best_val_acc = 0.0
    best_train_acc = 0.0
    epochs_no_improve = 0 # counter for early stopping
    save_path = os.path.join(BASE, "cnn_lstm_best.pt") # where to save the best model weights

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _ = run_epoch(model, train_dl, criterion, optimizer, device, train=True)   # training pass
        val_loss, val_acc, val_cls_acc = run_epoch(model, val_dl,   criterion, optimizer, device, train=False)  # validation pass
        scheduler.step(val_loss) # reduce LR if validation loss hasn't improved

        if val_acc > best_val_acc: # new best? save weights
            best_val_acc = val_acc
            best_train_acc = tr_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path) # save only the weights, not the whole model
            star = " *" # mark this epoch as a new best in the printout
        else:
            epochs_no_improve += 1 # no improvement this epoch
            star = ""

        if epoch % 5 == 0 or epoch == 1: # print every 5 epochs and the first epoch
            cls = "  ".join(f"{LABEL_NAMES[c]}={val_cls_acc[c]:.2f}" for c in range(N_CLASSES))
            print(f"{epoch:5d}  {tr_loss:10.4f}  {tr_acc:9.3f}  {val_loss:8.4f}  {val_acc:7.3f}{star}  [{cls}]")

        if epochs_no_improve >= EARLY_STOP_PATIENCE: # no improvement for too long? stop early
            print(f"\nEarly stop at epoch {epoch} (no val improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\nBest val accuracy : {best_val_acc:.3f}")
    print(f"Train accuracy at best val: {best_train_acc:.3f}")
    print(f"Model saved : {save_path}")


# Inference 
# Predict per-second structure labels for an MP3 not seen during training.
# Returns an integer array of shape (T,) with values 1 through 4
def predict(mp3_path: str) -> np.ndarray:
    model_path = os.path.join(BASE, "cnn_lstm_best.pt") # use the saved model

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = CnnLstm()
    model.load_state_dict(torch.load(model_path, map_location=device))  # load saved weights
    model.eval().to(device)  # set to eval mode (disables dropout) and move to device

    stats = np.load(os.path.join(BASE, "cnn_lstm_stats.npz")) # load training normalization stats
    g_mean = torch.tensor(stats["mean"], dtype=torch.float32).to(device) # per-feature mean
    g_std  = torch.tensor(stats["std"],  dtype=torch.float32).to(device) # per-feature std

    X = extract_energy_features(mp3_path) # extract 39 features per second
    X_t = torch.from_numpy(X).unsqueeze(0).to(device) # add batch dimension
    X_t = (X_t - g_mean) / g_std # apply same normalization used during training

    with torch.no_grad():  # no gradient needed for inference
        logits = model(X_t) # forward pass (1, T, 4)
        preds = logits.squeeze(0).argmax(dim=-1).cpu().numpy() # pick best class per second

    preds = preds + 1   # shift 0..3 to 1..4 to match original label convention

    # Median filter to remove single-second flickering
    preds = medfilt(preds.astype(np.float32), kernel_size=9).astype(int) # smooth noisy predictions

    # Force the last contiguous segment to outro
    last_label = preds[-1]
    i = len(preds) - 1
    while i >= 0 and preds[i] == last_label:
        i -= 1
    preds[i + 1:] = 4 # relabel the final segment as outro (4)

    return preds

# Print the results in a human-readable segments by writing timestamp
def print_structure(mp3_path: str):
    LABEL_NAMES = {1: "intro", 2: "verse", 3: "chorus", 4: "outro"}
    preds = predict(mp3_path) # get per-second label array
    print(f"\nPredicted structure for: {os.path.basename(mp3_path)}")
    print("-" * 40)
    prev, start = preds[0], 0 # track current segment label and its start time
    for t, label in enumerate(preds):
        if label != prev or t == len(preds) - 1: # segment ended? print it
            end = t if label != prev else t + 1
            m, s = divmod(start, 60) # convert seconds to mm:ss
            print(f"  {m:02d}:{s:02d}  {LABEL_NAMES.get(prev, '?'):8s}  ({prev})")
            prev, start = label, t # start tracking the new segment
    print("-" * 40)


# Main Function
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_structure(sys.argv[1]) # if an MP3 path is given, run inference
    else:
        train() # otherwise run the full training pipeline
