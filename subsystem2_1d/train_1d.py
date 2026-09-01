# subsystem2_1d/train_1d.py
# =========================================================
# Subsystem 2 -- 1D Time-Wave Model Training Pipeline
# Person 2 | feature/1d-model-p2
# =========================================================

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

_SUBSYSTEM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SUBSYSTEM_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from subsystem2_1d.classifiers_1d import build_model
from subsystem2_1d.dataloader_1d import get_1d_dataloaders


# =========================================================
# Early Stopping Class
# =========================================================

class EarlyStopping:
    """
    Monitors validation loss and tracks patience for early stopping.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.triggered = False

    def step(self, val_loss: float, epoch: int = 0) -> bool:
        """
        Record a validation loss step.

        Returns:
            bool: True if validation loss improved by at least min_delta, False otherwise.
        """
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = float(val_loss)
            self.best_epoch = epoch
            self.counter = 0
            self.triggered = False
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
            return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.triggered = False


# =========================================================
# Training Configuration
# =========================================================

@dataclass
class TrainingConfig:
    """Configuration parameters for 1D model training."""

    model_name: str = "cnn1d"
    num_classes: int = 11
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-4
    checkpoint_dir: str = os.path.join(_SUBSYSTEM_DIR, "checkpoints")
    device: Optional[str] = None
    batch_size: int = 1024
    data_dir: str = "data"


# =========================================================
# Main Training Function
# =========================================================

def train_model(
    model: Optional[nn.Module] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """
    Train a 1D modulation classifier with AdamW and Early Stopping.

    Args:
        model: PyTorch model instance (if None, instantiated via config.model_name).
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: TrainingConfig object with hyperparameters.

    Returns:
        Dict[str, Any]: History dictionary with training and validation metrics.
    """
    if config is None:
        config = TrainingConfig()

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        config.checkpoint_dir, f"best_{config.model_name}.pt"
    )

    device_str = (
        config.device
        if config.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    if model is None:
        model = build_model(config.model_name, num_classes=config.num_classes)
    model = model.to(device)

    if train_loader is None or val_loader is None:
        train_loader, val_loader, _ = get_1d_dataloaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=0 if sys.platform == "win32" else 2,
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    es = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rates": [],
        "checkpoint_path": checkpoint_path,
        "early_stopping_counter": 0,
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }

    print(f"Training {config.model_name} on device: {device}")

    for epoch in range(1, config.epochs + 1):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = float(train_loss / max(train_total, 1))
        train_acc = float(train_correct / max(train_total, 1))

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * signals.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = float(val_loss / max(val_total, 1))
        val_acc = float(val_correct / max(val_total, 1))

        # Current LR
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] | "
            f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%"
        )

        improved = es.step(avg_val_loss, epoch=epoch)
        if improved:
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
            }
            torch.save(checkpoint_payload, checkpoint_path)
            print(f"  -> Best model saved to {checkpoint_path}")

        if es.triggered:
            print(
                f"\n[Early Stopping] Validation loss did not improve for "
                f"{config.patience} epochs. Stopped at epoch {epoch}."
            )
            break

    history["early_stopping_counter"] = es.counter
    history["best_epoch"] = es.best_epoch
    history["best_val_loss"] = es.best_loss

    return history


# =========================================================
# CLI Entry Point
# =========================================================

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train Subsystem 2 (1D) models")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn1d",
        choices=["cnn1d", "cnn_transformer"],
        help="Architecture to train",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(_SUBSYSTEM_DIR, "checkpoints"),
    )
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        num_classes=11,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    train_model(config=config)


if __name__ == "__main__":
    _cli()
