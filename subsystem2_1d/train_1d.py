# subsystem2_1d/train_1d.py
# =========================================================
# Subsystem 2 -- 1D RF Classifier Training Pipeline
# Person 2 | feature/1d-model-p2
#
# Usage (CLI):
#   python subsystem2_1d/train_1d.py --model cnn1d --epochs 50
#   python subsystem2_1d/train_1d.py --model cnn_transformer --batch_size 512
#
# Importable API:
#   from subsystem2_1d.train_1d import train_model, EarlyStopping, TrainingConfig
# =========================================================

import os
import sys
import time
import argparse
from dataclasses import dataclass

_SUBSYSTEM_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT      = os.path.dirname(_SUBSYSTEM_DIR)
_CHECKPOINT_DIR = os.path.join(_SUBSYSTEM_DIR, "checkpoints")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple


# ---------------------------------------------------------
# Training Configuration Dataclass
# ---------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Immutable configuration bundle passed to train_model().

    All CLI arguments map 1-to-1 to these fields so the
    training loop itself is fully agnostic of argparse.

    Args:
        model_name     : Registry key — 'cnn1d' or 'cnn_transformer'.
        num_classes    : Number of output classes (11 for RadioML 2016.10a).
        epochs         : Maximum number of training epochs.
        learning_rate  : Initial AdamW learning rate.
        weight_decay   : AdamW weight decay (L2 regularisation).
        patience       : EarlyStopping patience in epochs.
        checkpoint_dir : Directory for saving best .pt checkpoints.
                         Defaults to subsystem2_1d/checkpoints/ if empty.
    """
    model_name:     str   = "cnn1d"
    num_classes:    int   = 11
    epochs:         int   = 50
    learning_rate:  float = 1e-3
    weight_decay:   float = 1e-4
    patience:       int   = 5
    checkpoint_dir: str   = ""    # resolved at runtime if empty


# ---------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------

class EarlyStopping:
    """
    Monitors validation loss and signals when training should halt.

    Increments an internal counter each epoch the validation loss
    fails to improve by more than `min_delta`. Triggers when the
    counter reaches `patience`.

    Args:
        patience  : Number of non-improving epochs before triggering.
        min_delta : Minimum absolute improvement to qualify as improvement.

    Attributes:
        counter   (int):   Consecutive non-improving epochs.
        triggered (bool):  True once patience is exhausted.
        best_loss (float): Best validation loss seen so far.
        best_epoch(int):   Epoch index at which best_loss was achieved.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.triggered = False
        self.best_epoch = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """
        Process one epoch of validation loss.

        Returns:
            bool: True if val_loss improved (checkpoint should be saved).
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            return True        # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
            return False       # no improvement

    def reset(self) -> None:
        """Reset state to allow reuse across multiple training runs."""
        self.best_loss  = float("inf")
        self.counter    = 0
        self.triggered  = False
        self.best_epoch = 0


# ---------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------

def _train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Single training epoch.

    Returns:
        (avg_loss, avg_accuracy)  averaged over all samples in the loader.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs          = signals.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += bs

    return total_loss / n, correct / n


@torch.no_grad()
def _validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Validation pass (no gradient computation).

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        logits  = model(signals)
        loss    = criterion(logits, labels)
        bs          = signals.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += bs

    return total_loss / n, correct / n


# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------

def train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    config:       TrainingConfig,
) -> Dict:
    """
    Full training loop: AdamW + ReduceLROnPlateau + EarlyStopping
    + best-checkpoint saving.

    Saves the best model weights to:
        <config.checkpoint_dir>/best_<config.model_name>.pt

    Args:
        model        : Untrained (or pre-trained) PyTorch model.
        train_loader : DataLoader for the training split.
        val_loader   : DataLoader for the validation split.
        config       : TrainingConfig dataclass instance.

    Returns:
        history (dict) with keys:
            train_loss, train_acc, val_loss, val_acc, learning_rates  -- lists
            best_val_loss, best_epoch, stopped_early, checkpoint_path -- scalars
            early_stopping_counter, early_stopping_triggered          -- final ES state
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # ReduceLROnPlateau: halve LR if val_loss stagnates for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # Resolve checkpoint directory
    ckpt_dir = config.checkpoint_dir if config.checkpoint_dir else _CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, f"best_{config.model_name}.pt")

    early_stopping = EarlyStopping(patience=config.patience)

    history: Dict = {
        "model_name":               config.model_name,
        "device":                   str(device),
        "train_loss":               [],
        "train_acc":                [],
        "val_loss":                 [],
        "val_acc":                  [],
        "learning_rates":           [],
        "best_val_loss":            float("inf"),
        "best_epoch":               0,
        "stopped_early":            False,
        "checkpoint_path":          checkpoint_path,
        "early_stopping_counter":   0,
        "early_stopping_triggered": False,
    }

    # ----------------------------------------------------------
    # Print training header
    # ----------------------------------------------------------
    _sep = "=" * 80
    print(f"\n{_sep}")
    print(f"  [Subsystem2] Training  : {config.model_name}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {config.epochs}  |  LR: {config.learning_rate}"
          f"  |  WD: {config.weight_decay}")
    print(f"  Patience     : {config.patience} epochs")
    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"{_sep}")

    _hdr = (
        f"  {'Epoch':>5} | {'TrainLoss':>9} | {'TrainAcc':>8} |"
        f" {'ValLoss':>8} | {'ValAcc':>7} | {'LR':>9} | Status"
    )
    print(f"\n{_hdr}")
    print(f"  {'-' * 76}")

    # ----------------------------------------------------------
    # Epoch loop
    # ----------------------------------------------------------
    for epoch in range(1, config.epochs + 1):
        t_epoch = time.time()

        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)      # update LR based on val_loss plateau

        # Record metrics
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["learning_rates"].append(current_lr)

        # Early stopping check + checkpoint save
        improved = early_stopping.step(val_loss, epoch)

        if improved:
            torch.save(
                {
                    "epoch":              epoch,
                    "model_name":         config.model_name,
                    "num_classes":        config.num_classes,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":           float(val_loss),
                    "val_acc":            float(val_acc),
                    "config":             vars(config),
                },
                checkpoint_path,
            )
            history["best_val_loss"] = float(val_loss)
            history["best_epoch"]    = epoch
            status = "[BEST]"
        else:
            status = f"[{early_stopping.counter}/{config.patience}]"

        elapsed = time.time() - t_epoch
        print(
            f"  {epoch:>5} | {train_loss:>9.4f} | {train_acc * 100:>7.2f}% |"
            f" {val_loss:>8.4f} | {val_acc * 100:>6.2f}% | {current_lr:>9.2e} |"
            f" {status:<14}  ({elapsed:.1f}s)"
        )

        if early_stopping.triggered:
            print(
                f"\n  [EarlyStopping] Patience={config.patience} exhausted at "
                f"epoch {epoch}. Best epoch: {early_stopping.best_epoch}."
            )
            history["stopped_early"] = True
            break

    # ----------------------------------------------------------
    # Finalise history
    # ----------------------------------------------------------
    history["early_stopping_counter"]   = early_stopping.counter
    history["early_stopping_triggered"] = early_stopping.triggered

    print(f"\n  Training complete.")
    print(f"  Best val loss : {history['best_val_loss']:.6f}  "
          f"(epoch {history['best_epoch']})")
    print(f"  Checkpoint    : {checkpoint_path}")
    print(_sep)

    return history


# ---------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a 1D RF Signal Classifier -- Subsystem 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="cnn1d",
        choices=["cnn1d", "cnn_transformer"],
        help="Model architecture to train.",
    )
    parser.add_argument("--data_dir",      type=str,   default="./data")
    parser.add_argument("--batch_size",    type=int,   default=1024)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--patience",      type=int,   default=5)
    parser.add_argument("--num_classes",   type=int,   default=11)
    args = parser.parse_args()

    from subsystem2_1d.classifiers_1d import build_model
    from subsystem2_1d.dataloader_1d  import get_1d_dataloaders

    config = TrainingConfig(
        model_name=args.model,
        num_classes=args.num_classes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    model = build_model(args.model, num_classes=args.num_classes)
    model.get_param_count()

    train_loader, val_loader, _ = get_1d_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )

    train_model(model, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
