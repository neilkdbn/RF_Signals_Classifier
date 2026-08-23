# train_2d.py
# =========================================================================
# CHUNK 4: 2D Training Loop
# Comprehensive training orchestrator for ResNet-18 and STFT-RADN.
# AdamW optimizer, CrossEntropy loss, ReduceLROnPlateau scheduler,
# early stopping on validation loss, and per-SNR accuracy logging
# to match Team A's benchmarking format.
# =========================================================================

import os
import sys
import time
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.signal import stft as scipy_stft

from dataset_loader import get_rf_dataloader
from models_2d import ResNet18_2D, STFTRADN

# Suppress scipy's harmless "complex input" warning during STFT
warnings.filterwarnings("ignore", message=".*complex.*return_onesided.*")


# ──────────────────────────────────────────────────────────────
# Transform: 1D IQ -> 2D Spectrogram (on-the-fly)
# ──────────────────────────────────────────────────────────────

class IQToSpectrogram:
    """
    On-the-fly transform bridging Person 1's DataLoader and Person 4's models.
    Converts a (2, 128) IQ tensor to a 2D spectrogram tensor using
    Person 3's exact Kaiser-windowed STFT parameters.

    Args:
        mode: "grayscale" -> (1, 64, 5) single power spectrogram
              "hybrid"    -> (3, 64, 5) [power_dB, phase, magnitude]
    """
    STFT_PARAMS = dict(window=("kaiser", 0.85), nperseg=64, noverlap=32)

    def __init__(self, mode="grayscale"):
        self.mode = mode

    def __call__(self, signal_tensor):
        signal_np = signal_tensor.numpy()
        I, Q = signal_np[0], signal_np[1]
        complex_signal = I + 1j * Q

        _, _, Zxx = scipy_stft(complex_signal, **self.STFT_PARAMS)

        if self.mode == "grayscale":
            power_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)
            spec = np.fft.fftshift(power_db, axes=0)
            return torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        else:  # hybrid
            Zxx_s = np.fft.fftshift(Zxx, axes=0)
            power_db = 10 * np.log10(np.abs(Zxx_s) ** 2 + 1e-10)
            phase = np.angle(Zxx_s)
            magnitude = np.abs(Zxx_s)
            return torch.tensor(
                np.stack([power_db, phase, magnitude], axis=0),
                dtype=torch.float32,
            )  # (3, H, W)


# ──────────────────────────────────────────────────────────────
# Training & Validation
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for signals, labels, _snrs in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * signals.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation pass. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for signals, labels, _snrs in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        outputs = model(signals)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * signals.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_per_snr(model, loader, device):
    """
    Per-SNR accuracy evaluation — matches Team A's benchmarking output format.
    Returns: dict {snr_dB: accuracy_%}
    """
    model.eval()
    snr_correct = {}
    snr_total = {}

    for signals, labels, snrs in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        outputs = model(signals)
        _, predicted = outputs.max(1)

        for i in range(labels.size(0)):
            snr = int(snrs[i].item())
            snr_correct.setdefault(snr, 0)
            snr_total.setdefault(snr, 0)
            snr_total[snr] += 1
            if predicted[i] == labels[i]:
                snr_correct[snr] += 1

    return {
        snr: 100.0 * snr_correct[snr] / snr_total[snr]
        for snr in sorted(snr_correct)
    }


# ──────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss has not improved for
    `patience` consecutive epochs.
    """

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ──────────────────────────────────────────────────────────────
# Main Training Orchestrator
# ──────────────────────────────────────────────────────────────

def main(args):
    # ── Device ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("   2D MODEL TRAINING LOOP")
    print("=" * 70)
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model}")
    print(f"  Input mode : {args.input_mode}")
    print(f"  LR         : {args.lr}  |  Weight decay: {args.weight_decay}")
    print(f"  Batch size : {args.batch_size}  |  Max epochs: {args.epochs}")
    print(f"  Patience   : {args.patience}")
    print("=" * 70)

    # ── Transform ──────────────────────────────────────────────
    transform = IQToSpectrogram(mode=args.input_mode)
    in_channels = 1 if args.input_mode == "grayscale" else 3

    # ── DataLoaders ────────────────────────────────────────────
    print("\n--- Loading dataset splits ---")
    train_loader = get_rf_dataloader(
        data_dir=args.data_dir, split="train",
        batch_size=args.batch_size, transform=transform,
    )
    val_loader = get_rf_dataloader(
        data_dir=args.data_dir, split="val",
        batch_size=args.batch_size, transform=transform,
    )
    test_loader = get_rf_dataloader(
        data_dir=args.data_dir, split="test",
        batch_size=args.batch_size, transform=transform,
    )
    print(f"  Train : {len(train_loader.dataset):>6} samples  ({len(train_loader)} batches)")
    print(f"  Val   : {len(val_loader.dataset):>6} samples  ({len(val_loader)} batches)")
    print(f"  Test  : {len(test_loader.dataset):>6} samples  ({len(test_loader)} batches)")

    # ── Model ──────────────────────────────────────────────────
    num_classes = len(np.unique(
        np.load(os.path.join(args.data_dir, "y_all.npy"))
    ))
    print(f"  Classes: {num_classes}")

    if args.model == "resnet18":
        model = ResNet18_2D(num_classes=num_classes, in_channels=in_channels)
    elif args.model == "stft-radn":
        model = STFTRADN(num_classes=num_classes, in_channels=in_channels)
    else:
        sys.exit(f"[ERROR] Unknown model: {args.model}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Optimizer / Loss / Scheduler ───────────────────────────
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stop = EarlyStopping(patience=args.patience)

    # ── Checkpoint dir ─────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, f"best_{args.model}.pt")

    # ── Training Loop ──────────────────────────────────────────
    print("\n" + "=" * 74)
    print(f"{'Ep':>4} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>9} | {'Val Acc':>8} | {'LR':>10} | {'Time':>6}")
    print("-" * 74)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        marker = " *" if val_acc > best_val_acc else ""
        print(
            f"{epoch:4d} | {train_loss:10.4f} | {train_acc:8.2f}% | "
            f"{val_loss:9.4f} | {val_acc:7.2f}% | {current_lr:10.6f} | "
            f"{elapsed:5.1f}s{marker}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "model_name": args.model,
                "num_classes": num_classes,
                "in_channels": in_channels,
            }, best_model_path)

        # Early stopping check
        early_stop.step(val_loss)
        if early_stop.should_stop:
            print(f"\n[EARLY STOP] No val loss improvement for {args.patience} epochs.")
            break

    print("=" * 74)

    # ── Final Test Evaluation ──────────────────────────────────
    print("\n--- Loading best checkpoint for final evaluation ---")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n  [TEST RESULT]  Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.2f}%")

    # ── Per-SNR Accuracy (Team A compatible format) ────────────
    print("\n--- Per-SNR Accuracy (Team A Benchmarking Format) ---")
    snr_acc = evaluate_per_snr(model, test_loader, device)

    print(f"{'SNR (dB)':>10} | {'Accuracy':>10} | {'Bar'}")
    print("-" * 50)
    for snr, acc in snr_acc.items():
        bar = "#" * int(acc / 5)
        print(f"{snr:>8d} dB | {acc:9.2f}% | {bar}")

    avg_snr_acc = np.mean(list(snr_acc.values()))
    print("-" * 50)
    print(f"{'Average':>10} | {avg_snr_acc:9.2f}%")

    print(f"\n[SAVED] Best checkpoint : {best_model_path}")
    print(f"[SAVED] Best val acc   : {best_val_acc:.2f}%")
    print(f"[DONE]  Test accuracy  : {test_acc:.2f}%")


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2D Model Training for RF Signal Classification"
    )
    parser.add_argument(
        "--model", type=str, default="resnet18",
        choices=["resnet18", "stft-radn"],
        help="Model architecture (default: resnet18)",
    )
    parser.add_argument(
        "--input-mode", type=str, default="grayscale",
        choices=["grayscale", "hybrid"],
        help="Spectrogram mode: grayscale (1ch) or hybrid (3ch)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Path to preprocessed dataset directory",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="AdamW learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience in epochs (default: 10)",
    )

    args = parser.parse_args()
    main(args)
