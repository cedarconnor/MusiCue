"""
Training script for the MusiCue drum classifier CNN.

Datasets required (download separately):
  - ENST-Drums: https://perso.telecom-paristech.fr/grichard/ENST-drums/
  - MDB Drums: https://github.com/MDanalysis/MDB-Drums
  - Slakh2100: https://zenodo.org/record/4599666

Usage:
  python scripts/train_drum_classifier.py --data-dir D:/drum_data --out models/drum_cnn.pt

The script expects pre-processed onset windows in HDF5 format:
  data-dir/
    onsets.h5  -- dataset with keys: 'audio' (N, 2205), 'labels' (N,) int in [0,5]
                  Label map: 0=kick, 1=snare, 2=hat, 3=tom, 4=cymbal, 5=other
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from musicue.analysis.drum_classifier import DRUM_CLASSES, DrumClassifierCNN, _onset_to_mel

SR = 44100
N_CLASSES = len(DRUM_CLASSES)


class DrumOnsetDataset(Dataset):
    def __init__(self, h5_path: Path) -> None:
        import h5py

        with h5py.File(str(h5_path), "r") as f:
            self.audio: np.ndarray = np.asarray(f["audio"])  # (N, 2205)
            self.labels: np.ndarray = np.asarray(f["labels"]).astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        mel = _onset_to_mel(self.audio[idx], sr=SR)
        target_frames = 44
        if mel.shape[1] < target_frames:
            mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
        else:
            mel = mel[:, :target_frames]
        return torch.from_numpy(mel[np.newaxis]), int(self.labels[idx])  # pyright: ignore[reportPrivateImportUsage]


def train(
    data_dir: Path,
    out_path: Path,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = DrumOnsetDataset(data_dir / "onsets.h5")
    n_val = int(len(dataset) * 0.15)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = DrumClassifierCNN(n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        val_acc = correct / max(total, 1)
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch:3d} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(out_path))
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/drum_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data_dir, args.out, args.epochs, args.batch_size, args.lr)
