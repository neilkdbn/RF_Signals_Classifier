# subsystem2_1d/eval_1d.py
# =========================================================
# Subsystem 2 -- 1D RF Classifier Evaluation Pipeline
# Person 2 | feature/1d-model-p2
#
# Usage (CLI):
#   python subsystem2_1d/eval_1d.py \
#       --model cnn1d \
#       --checkpoint subsystem2_1d/checkpoints/best_cnn1d.pt \
#       --data_dir ./data
#
# Importable API:
#   from subsystem2_1d.eval_1d import evaluate_model, print_edge_suitability_report
# =========================================================

import os
import sys
import json
import time
import argparse

# Must set backend BEFORE importing pyplot (headless / CI safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional
import torch
import torch.nn as nn

_SUBSYSTEM_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT      = os.path.dirname(_SUBSYSTEM_DIR)
_CHECKPOINT_DIR = os.path.join(_SUBSYSTEM_DIR, "checkpoints")
_RESULTS_DIR    = os.path.join(_SUBSYSTEM_DIR, "results")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------
# RadioML 2016.10a Dataset Constants
# ---------------------------------------------------------

# 20 SNR levels in steps of 2 dB (matches Person 1's dataset)
SNR_LEVELS: List[int] = list(range(-20, 20, 2))   # [-20, -18, ..., 18]

# Target SNRs for confusion matrix generation
CONFUSION_SNR_TARGETS: List[int] = [-10, 0, 10]

NUM_CLASSES = 11
MOD_NAMES = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK",  "PAM4",  "QAM16", "QAM64", "QPSK", "WBFM",
]


# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name:      str,
    num_classes:     int = NUM_CLASSES,
) -> nn.Module:
    """
    Re-instantiate a model architecture and load saved weights.

    Args:
        checkpoint_path : Absolute/relative path to the .pt file.
        model_name      : Registry key ('cnn1d' or 'cnn_transformer').
        num_classes     : Number of output classes.

    Returns:
        nn.Module in eval() mode on CPU.
    """
    from subsystem2_1d.classifiers_1d import build_model

    model = build_model(model_name, num_classes=num_classes)
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------
# Batch Inference
# ---------------------------------------------------------

@torch.no_grad()
def _batch_inference(
    model:      nn.Module,
    X:          np.ndarray,
    batch_size: int = 512,
    device:     str = "cpu",
) -> np.ndarray:
    """
    Run batched forward pass over numpy array X.

    Returns:
        np.ndarray of integer predicted class indices, shape (N,).
    """
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        batch  = torch.tensor(X[i : i + batch_size]).to(device)
        logits = model(batch)
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------
# Per-SNR Accuracy
# ---------------------------------------------------------

def compute_per_snr_accuracy(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    snr_labels: np.ndarray,
) -> Dict[int, float]:
    """
    Compute classification accuracy at each distinct SNR level.

    Skips SNR levels with zero samples — useful for subsets of the dataset.

    Args:
        y_true     : Ground-truth integer labels, shape (N,).
        y_pred     : Predicted integer labels, shape (N,).
        snr_labels : Integer SNR in dB per sample, shape (N,).

    Returns:
        dict mapping SNR (int, dB) -> accuracy (float in [0,1]).
    """
    per_snr: Dict[int, float] = {}
    for snr in SNR_LEVELS:
        mask = snr_labels == snr
        if mask.sum() == 0:
            continue
        per_snr[snr] = float((y_true[mask] == y_pred[mask]).mean())
    return per_snr


# ---------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------

def _plot_confusion_matrix(
    cm:         np.ndarray,
    snr:        int,
    model_name: str,
    output_dir: str,
    num_classes: int = NUM_CLASSES,
) -> str:
    """
    Render a row-normalised confusion matrix heatmap and save as PNG.

    Args:
        cm          : Raw count confusion matrix, shape (num_classes, num_classes).
        snr         : SNR level this matrix was computed at (dB).
        model_name  : Used in the file name and plot title.
        output_dir  : Directory to write the PNG.
        num_classes : Number of modulation classes.

    Returns:
        Absolute path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Row-normalise: each row sums to 1 (recall per true class).
    # Use a safe denominator to avoid numpy's "divide by zero" RuntimeWarning:
    # np.where evaluates BOTH branches eagerly, so cm/row_sums still executes
    # on zero rows even though those cells would be masked.
    row_sums   = cm.sum(axis=1, keepdims=True).astype(float)
    safe_denom = np.where(row_sums > 0, row_sums, 1.0)   # 1.0 where empty → ratio=0
    cm_norm    = np.where(row_sums > 0, cm / safe_denom, 0.0)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                   vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)", fontsize=10)

    labels = (MOD_NAMES[:num_classes]
              if num_classes <= len(MOD_NAMES)
              else [str(i) for i in range(num_classes)])

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate each cell with raw sample count
    thresh = cm_norm.max() / 2.0 if cm_norm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(int(cm[i, j])),
                    ha="center", va="center", color=color, fontsize=7)

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    snr_str = f"+{snr} dB" if snr >= 0 else f"{snr} dB"
    ax.set_title(
        f"Confusion Matrix @ {snr_str} SNR  |  {model_name}",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    png_name = f"confusion_matrix_{snr}db_{model_name}.png"
    png_path = os.path.join(output_dir, png_name)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ---------------------------------------------------------
# Latency Profiling
# ---------------------------------------------------------

def measure_inference_latency(
    model:   nn.Module,
    warmup:  int = 100,
    n_timed: int = 200,
) -> float:
    """
    Measure mean per-sample inference latency on CPU (ms).

    Runs `warmup` forward passes to bypass JIT / cache cold-start,
    then times `n_timed` passes and computes the mean.

    Args:
        model   : Model to profile (moved to CPU; set to eval).
        warmup  : Number of un-timed warm-up forward passes.
        n_timed : Number of timed forward passes.

    Returns:
        Mean latency per single sample in milliseconds.
    """
    model = model.cpu()
    model.eval()
    dummy = torch.randn(1, 2, 128)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_timed):
            model(dummy)
    t_end = time.perf_counter()

    return ((t_end - t_start) / n_timed) * 1000.0   # ms


# ---------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------

def evaluate_model(
    model:       nn.Module,
    X:           np.ndarray,
    y:           np.ndarray,
    snr_labels:  np.ndarray,
    model_name:  str,
    output_dir:  str  = _RESULTS_DIR,
    batch_size:  int  = 512,
    num_classes: int  = NUM_CLASSES,
) -> Dict:
    """
    Comprehensive evaluation of a trained 1D RF classifier.

    Produces:
      * Per-SNR accuracy table (printed) + JSON file
      * Confusion matrices at -10, 0, +10 dB (PNG + .npy per target)
      * CPU inference latency (ms/sample)

    All output files are written to `output_dir`. Evaluation always runs
    on CPU for reproducible latency benchmarking.

    Args:
        model       : Trained model (any device; internally moved to CPU).
        X           : Test signals, shape (N, 2, 128) float32.
        y           : Integer true labels, shape (N,) int64.
        snr_labels  : Integer SNR (dB) per sample, shape (N,).
        model_name  : String key used to name all output files.
        output_dir  : Directory for all saved results.
        batch_size  : Batch size for batched inference.
        num_classes : Number of modulation classes.

    Returns:
        dict with keys:
            overall_accuracy, per_snr_accuracy, peak_accuracy, peak_snr,
            inference_latency_ms, param_count, n_snr_levels, saved_files.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model.cpu()
    model.eval()

    # ----------------------------------------------------------
    # 1. Batch Inference
    # ----------------------------------------------------------
    print(f"\n[Eval:{model_name}] Inference on {len(X):,} samples...")
    y_pred      = _batch_inference(model, X, batch_size=batch_size, device="cpu")
    overall_acc = float((y == y_pred).mean())
    print(f"[Eval:{model_name}] Overall accuracy: {overall_acc * 100:.2f}%")

    # ----------------------------------------------------------
    # 2. Per-SNR Accuracy Table + JSON
    # ----------------------------------------------------------
    per_snr_acc = compute_per_snr_accuracy(y, y_pred, snr_labels)

    print(f"\n[Eval:{model_name}] Per-SNR Accuracy (RadioML 2016.10a range):")
    print(f"  {'SNR (dB)':>9} | {'Accuracy':>9} | {'Samples':>7}")
    print(f"  {'-'*32}")
    for snr in sorted(per_snr_acc.keys()):
        n_at_snr = int((snr_labels == snr).sum())
        print(f"  {snr:>+9d} | {per_snr_acc[snr] * 100:>8.2f}% | {n_at_snr:>7}")

    # Save JSON (string-keyed for JSON compliance; int-keyed also in the dict)
    json_name = f"accuracy_vs_snr_{model_name}.json"
    json_path = os.path.join(output_dir, json_name)
    json_payload = {
        "model_name":         model_name,
        "overall_accuracy":   overall_acc,
        "per_snr_accuracy":   {str(k): v for k, v in per_snr_acc.items()},
        "snr_levels_present": sorted(per_snr_acc.keys()),
        "num_snr_levels":     len(per_snr_acc),
    }
    with open(json_path, "w") as fh:
        json.dump(json_payload, fh, indent=2)
    print(f"\n[Eval:{model_name}] Accuracy JSON saved: {json_path}")

    # ----------------------------------------------------------
    # 3. Confusion Matrices at Target SNRs
    # ----------------------------------------------------------
    saved_cm_files: Dict[int, Dict[str, str]] = {}
    for snr_target in CONFUSION_SNR_TARGETS:
        mask     = snr_labels == snr_target
        n_at_snr = int(mask.sum())
        if n_at_snr == 0:
            print(f"[Eval:{model_name}] No samples at {snr_target:+d} dB -- skipping CM.")
            continue

        cm = confusion_matrix(
            y[mask], y_pred[mask],
            labels=list(range(num_classes)),
        )

        # Save raw numpy array
        npy_name = f"confusion_matrix_{snr_target}db_{model_name}.npy"
        npy_path = os.path.join(output_dir, npy_name)
        np.save(npy_path, cm)

        # Save PNG
        png_path = _plot_confusion_matrix(
            cm, snr_target, model_name, output_dir, num_classes
        )
        saved_cm_files[snr_target] = {"npy": npy_path, "png": png_path}
        print(
            f"[Eval:{model_name}] CM @ {snr_target:+d} dB  "
            f"({n_at_snr} samples) -> {os.path.basename(npy_path)}"
        )

    # ----------------------------------------------------------
    # 4. Inference Latency
    # ----------------------------------------------------------
    print(f"\n[Eval:{model_name}] Profiling CPU latency (warm-up=100, timed=200)...")
    latency_ms = measure_inference_latency(model, warmup=100, n_timed=200)
    print(f"[Eval:{model_name}] Avg latency: {latency_ms:.3f} ms/sample")

    # ----------------------------------------------------------
    # 5. Assemble Results Dict
    # ----------------------------------------------------------
    peak_snr = max(per_snr_acc, key=per_snr_acc.get) if per_snr_acc else None
    peak_acc = per_snr_acc.get(peak_snr, 0.0) if peak_snr is not None else 0.0
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_name":           model_name,
        "overall_accuracy":     overall_acc,
        "per_snr_accuracy":     per_snr_acc,
        "peak_accuracy":        peak_acc,
        "peak_snr":             peak_snr,
        "inference_latency_ms": latency_ms,
        "param_count":          param_count,
        "n_snr_levels":         len(per_snr_acc),
        "saved_files": {
            "json":               json_path,
            "confusion_matrices": saved_cm_files,
        },
    }


# ---------------------------------------------------------
# Edge Suitability Comparison Report
# ---------------------------------------------------------

def print_edge_suitability_report(results_list: List[Dict]) -> None:
    """
    Print a formatted side-by-side comparison of multiple model evaluation results.

    Contrasts: overall accuracy, peak accuracy, peak SNR, parameter count,
    and average CPU inference latency.

    Args:
        results_list : List of result dicts returned by evaluate_model().
    """
    _sep = "=" * 74
    print(f"\n{_sep}")
    print("  EDGE SUITABILITY REPORT -- Subsystem 2 (1D Time-Wave Approach)")
    print(_sep)

    col = 32
    metric_rows = [
        ("Overall Test Accuracy",         "overall_accuracy",     "{:.2%}"),
        ("Peak Accuracy (best SNR level)", "peak_accuracy",        "{:.2%}"),
        ("Peak SNR (dB)",                  "peak_snr",             "{:+d} dB"),
        ("Trainable Parameters",           "param_count",          "{:,}"),
        ("Avg CPU Latency (ms/sample)",    "inference_latency_ms", "{:.3f} ms"),
        ("SNR Levels Evaluated",           "n_snr_levels",         "{:d} / 20"),
    ]

    header = f"  {'Metric':<{col}}"
    for r in results_list:
        header += f"  {r['model_name']:>22}"
    print(f"\n{header}")
    print(f"  {'-' * (col + 26 * len(results_list))}")

    for label, key, fmt in metric_rows:
        row = f"  {label:<{col}}"
        for r in results_list:
            val = r.get(key)
            if val is None:
                formatted = "N/A"
            else:
                try:
                    formatted = fmt.format(val)
                except (TypeError, ValueError):
                    formatted = str(val)
            row += f"  {formatted:>22}"
        print(row)

    print(f"\n{_sep}")

    # Compact verdict
    if len(results_list) == 2:
        r_a, r_b = results_list[0], results_list[1]
        lat_a = r_a.get("inference_latency_ms", 0)
        lat_b = r_b.get("inference_latency_ms", 0)
        faster = r_a["model_name"] if lat_a < lat_b else r_b["model_name"]
        acc_a  = r_a.get("overall_accuracy", 0)
        acc_b  = r_b.get("overall_accuracy", 0)
        more_accurate = r_a["model_name"] if acc_a >= acc_b else r_b["model_name"]
        print(f"\n  Verdict: '{more_accurate}' has higher overall accuracy.")
        print(f"           '{faster}' is faster for edge deployment.")
    print(f"{_sep}\n")


# ---------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained 1D RF Signal Classifier -- Subsystem 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["cnn1d", "cnn_transformer"],
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint file saved by train_1d.py",
    )
    parser.add_argument("--data_dir",    type=str, default="./data")
    parser.add_argument("--batch_size",  type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    args = parser.parse_args()

    # Import private constants from dataloader to load Person 1's test split
    from subsystem2_1d.dataloader_1d import _load_npy, _DATA_FILES, _SPLIT_FILES

    X_all    = _load_npy(args.data_dir, _DATA_FILES["X"])
    y_all    = _load_npy(args.data_dir, _DATA_FILES["y"])
    snrs_all = _load_npy(args.data_dir, _DATA_FILES["snrs"])
    test_idx = _load_npy(args.data_dir, _SPLIT_FILES["test"])

    model = load_model_from_checkpoint(
        args.checkpoint, args.model, args.num_classes
    )
    model.get_param_count()

    results = evaluate_model(
        model,
        X          = X_all[test_idx],
        y          = y_all[test_idx],
        snr_labels = snrs_all[test_idx],
        model_name = args.model,
        output_dir = _RESULTS_DIR,
        batch_size = args.batch_size,
        num_classes= args.num_classes,
    )

    print_edge_suitability_report([results])


if __name__ == "__main__":
    main()
