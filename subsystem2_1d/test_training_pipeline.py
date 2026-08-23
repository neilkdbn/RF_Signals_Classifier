# subsystem2_1d/test_training_pipeline.py
# =========================================================
# Subsystem 2 -- Training & Evaluation Pipeline Tests
# Person 2 | feature/1d-model-p2
#
# Run from the repo root:
#   python subsystem2_1d/test_training_pipeline.py
#
# All I/O goes to a temporary directory -- nothing is written
# to subsystem2_1d/checkpoints/ or subsystem2_1d/results/.
#
# Test Groups:
#   T-1  EarlyStopping class -- unit tests (no model needed)
#   T-2  Training smoke test -- CNN1DClassifier (2 epochs)
#   T-3  Training smoke test -- CNNTransformerHybrid (2 epochs)
#   T-4  Evaluation dry-run  -- CNN1DClassifier
#   T-5  Evaluation dry-run  -- CNNTransformerHybrid
# =========================================================

import sys
import os

# Force UTF-8 on Windows cp1252 terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import copy
import json
import shutil
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader

from subsystem2_1d.classifiers_1d import build_model
from subsystem2_1d.dataloader_1d   import RFSignalDataset1D
from subsystem2_1d.train_1d        import EarlyStopping, TrainingConfig, train_model
from subsystem2_1d.eval_1d         import (
    evaluate_model,
    load_model_from_checkpoint,
    SNR_LEVELS,
    CONFUSION_SNR_TARGETS,
    NUM_CLASSES,
)


# =========================================================
# Terminal colour helpers (ASCII-safe)
# =========================================================
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"

_results = {"passed": 0, "failed": 0}


def _report(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        _results["passed"] += 1
        print(f"  {PASS}  {label}")
    else:
        _results["failed"] += 1
        print(f"  {FAIL}  {label}")
    if detail:
        print(f"         {detail}")


# =========================================================
# Synthetic Data Factories
# =========================================================

NUM_CLASSES_MOCK = NUM_CLASSES   # 11
N_TRAIN          = 100
N_VAL            = 40
N_PER_SNR_TEST   = 10    # samples per SNR level in the dedicated test set
                          # => 20 * 10 = 200 test samples total

_RNG = np.random.default_rng(seed=42)


def _make_train_val_loaders(batch_size: int = 16):
    """Return (train_loader, val_loader) over synthetic data."""
    n = N_TRAIN + N_VAL
    X = _RNG.standard_normal((n, 2, 128)).astype(np.float32)
    y = _RNG.integers(0, NUM_CLASSES_MOCK, n).astype(np.int64)

    train_ds = RFSignalDataset1D(X[:N_TRAIN], y[:N_TRAIN])
    val_ds   = RFSignalDataset1D(X[N_TRAIN:], y[N_TRAIN:])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
    )


def _make_test_data():
    """
    Build a test set with GUARANTEED coverage of all 20 SNR levels.

    Constructs N_PER_SNR_TEST samples for each SNR level, so every SNR in
    SNR_LEVELS appears exactly N_PER_SNR_TEST times in the returned array.
    Total samples: 20 * 10 = 200.
    """
    n_total = len(SNR_LEVELS) * N_PER_SNR_TEST
    X    = _RNG.standard_normal((n_total, 2, 128)).astype(np.float32)
    y    = _RNG.integers(0, NUM_CLASSES_MOCK, n_total).astype(np.int64)
    snrs = np.array(
        [snr for snr in SNR_LEVELS for _ in range(N_PER_SNR_TEST)],
        dtype=np.int32,
    )
    return X, y, snrs


# =========================================================
# T-1: EarlyStopping Unit Tests
# =========================================================

def run_t1_early_stopping():
    print(f"\n{CYAN}[T-1]  EarlyStopping -- unit tests{RESET}")

    es = EarlyStopping(patience=3, min_delta=1e-4)

    # Improvement step
    improved = es.step(1.0, epoch=1)
    _report("First step with improvement returns True",   improved is True)
    _report("Counter remains 0 after improvement",        es.counter == 0,
            f"got {es.counter}")
    _report("best_loss updated to 1.0",                  abs(es.best_loss - 1.0) < 1e-9,
            f"got {es.best_loss}")
    _report("best_epoch updated to 1",                   es.best_epoch == 1)

    # Non-improving step 1
    es.step(1.5, epoch=2)
    _report("Counter increments to 1 (no improvement)",  es.counter == 1,
            f"got {es.counter}")
    _report("triggered is False at counter=1",           es.triggered is False)

    # Non-improving step 2
    es.step(1.5, epoch=3)
    _report("Counter increments to 2",                   es.counter == 2,
            f"got {es.counter}")
    _report("triggered is False at counter=2",           es.triggered is False)

    # Non-improving step 3 -- triggers patience
    es.step(1.5, epoch=4)
    _report("Counter reaches patience=3",                es.counter == 3,
            f"got {es.counter}")
    _report("triggered fires when counter == patience",  es.triggered is True)

    # Late improvement resets counter (even after trigger)
    es.step(0.5, epoch=5)
    _report("Counter resets to 0 on late improvement",   es.counter == 0,
            f"got {es.counter}")
    _report("best_loss updated to 0.5",                  abs(es.best_loss - 0.5) < 1e-9)
    _report("best_epoch updated to 5",                   es.best_epoch == 5)

    # reset() restores pristine state
    es.reset()
    _report("reset() sets counter=0",                    es.counter == 0)
    _report("reset() sets triggered=False",              es.triggered is False)
    _report("reset() sets best_loss=inf",                es.best_loss == float("inf"))
    _report("reset() sets best_epoch=0",                 es.best_epoch == 0)


# =========================================================
# T-2 / T-3: Training Smoke Tests (2 epochs)
# =========================================================

def run_training_smoke_test(model_name: str, tmp_ckpt_dir: str) -> str:
    """
    Run a 2-epoch training loop on synthetic data.

    Verifies:
      - No exceptions during training
      - History dict structure
      - Model parameters change (gradients flowed)
      - Checkpoint file is saved and loadable

    Returns:
        Absolute path to the saved checkpoint (.pt file).
    """
    print(f"\n{CYAN}[Training smoke] -- {model_name}  (2 epochs, {N_TRAIN} samples){RESET}")

    train_loader, val_loader = _make_train_val_loaders(batch_size=16)
    model = build_model(model_name, num_classes=NUM_CLASSES_MOCK)

    # Snapshot initial parameter values to detect gradient flow later
    initial_params = {
        name: p.clone().detach()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    config = TrainingConfig(
        model_name    = model_name,
        num_classes   = NUM_CLASSES_MOCK,
        epochs        = 2,
        learning_rate = 1e-3,
        weight_decay  = 1e-4,
        patience      = 5,
        checkpoint_dir= tmp_ckpt_dir,
    )

    history = train_model(model, train_loader, val_loader, config)

    # --- Assertions ---

    _report(f"[{model_name}] train_model returns a dict",
            isinstance(history, dict))

    required_keys = {"train_loss", "val_loss", "train_acc",
                     "val_acc", "learning_rates", "checkpoint_path",
                     "early_stopping_counter"}
    _report(f"[{model_name}] History has all required keys",
            required_keys.issubset(history.keys()),
            f"Missing: {required_keys - history.keys()}")

    _report(f"[{model_name}] History contains 2 epoch entries",
            len(history["train_loss"]) == 2,
            f"got {len(history['train_loss'])}")

    _report(f"[{model_name}] All train/val losses are finite",
            all(
                np.isfinite(v)
                for v in history["train_loss"] + history["val_loss"]
            ))

    _report(f"[{model_name}] All train/val accuracies in [0, 1]",
            all(
                0.0 <= v <= 1.0
                for v in history["train_acc"] + history["val_acc"]
            ))

    # Gradient flow: at least one parameter must have changed
    params_changed = any(
        not torch.allclose(p.detach(), initial_params[name], atol=1e-10)
        for name, p in model.named_parameters()
        if p.requires_grad and name in initial_params
    )
    _report(f"[{model_name}] Parameters updated by optimizer (gradient flow confirmed)",
            params_changed)

    # Checkpoint existence
    ckpt_path = os.path.join(tmp_ckpt_dir, f"best_{model_name}.pt")
    _report(f"[{model_name}] Checkpoint file saved: best_{model_name}.pt",
            os.path.isfile(ckpt_path))

    # Checkpoint validity
    if os.path.isfile(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _report(f"[{model_name}] Checkpoint contains model_state_dict",
                    "model_state_dict" in ckpt)
            _report(f"[{model_name}] Checkpoint contains epoch number",
                    "epoch" in ckpt and isinstance(ckpt["epoch"], int))
            _report(f"[{model_name}] Checkpoint contains val_loss",
                    "val_loss" in ckpt and np.isfinite(ckpt["val_loss"]))
        except Exception as exc:
            _report(f"[{model_name}] Checkpoint is torch-loadable", False, str(exc))

    # Early stopping counter is a non-negative integer
    es_counter = history.get("early_stopping_counter", -1)
    _report(f"[{model_name}] early_stopping_counter >= 0  (got {es_counter})",
            isinstance(es_counter, int) and es_counter >= 0)

    return ckpt_path


# =========================================================
# T-4 / T-5: Evaluation Dry-Run
# =========================================================

def run_eval_dry_run(model_name: str, ckpt_path: str, tmp_results_dir: str) -> None:
    """
    Load a checkpoint and run evaluate_model() on synthetic test data.

    Verifies:
      - accuracy_vs_snr JSON generated with all 20 SNR data points
      - confusion matrix .npy files exist at -10, 0, +10 dB
      - confusion matrix .png files exist at -10, 0, +10 dB
      - confusion matrix arrays have correct shape (11, 11)
      - latency is a positive float
      - overall accuracy is within [0, 1]
    """
    print(f"\n{CYAN}[Eval dry-run] -- {model_name}{RESET}")

    X_test, y_test, snr_test = _make_test_data()

    # Load model from the checkpoint we just saved
    try:
        model = load_model_from_checkpoint(ckpt_path, model_name, NUM_CLASSES_MOCK)
        _report(f"[{model_name}] Model loaded from checkpoint successfully", True)
    except Exception as exc:
        _report(f"[{model_name}] Model loaded from checkpoint", False, str(exc))
        return

    results = evaluate_model(
        model        = model,
        X            = X_test,
        y            = y_test,
        snr_labels   = snr_test,
        model_name   = model_name,
        output_dir   = tmp_results_dir,
        batch_size   = 64,
        num_classes  = NUM_CLASSES_MOCK,
    )

    _report(f"[{model_name}] evaluate_model returns a dict",
            isinstance(results, dict))

    # ----------------------------------------------------------
    # JSON with 20 SNR data points
    # ----------------------------------------------------------
    json_path = results.get("saved_files", {}).get("json", "")
    _report(f"[{model_name}] accuracy_vs_snr JSON file exists",
            os.path.isfile(json_path),
            f"path: {json_path}")

    if os.path.isfile(json_path):
        with open(json_path) as fh:
            jdata = json.load(fh)
        n_snr_keys = len(jdata.get("per_snr_accuracy", {}))
        _report(
            f"[{model_name}] JSON contains all 20 SNR data points (got {n_snr_keys})",
            n_snr_keys == 20,
            f"Expected 20, got {n_snr_keys}. "
            f"Present keys: {sorted(int(k) for k in jdata.get('per_snr_accuracy', {}))}",
        )
        _report(
            f"[{model_name}] JSON 'overall_accuracy' key present and in [0,1]",
            0.0 <= jdata.get("overall_accuracy", -1) <= 1.0,
        )

    # ----------------------------------------------------------
    # Confusion matrices at -10, 0, +10 dB
    # ----------------------------------------------------------
    cm_files = results.get("saved_files", {}).get("confusion_matrices", {})
    for snr_target in CONFUSION_SNR_TARGETS:
        snr_entry = cm_files.get(snr_target, {})

        npy_path = snr_entry.get("npy", "")
        npy_ok   = os.path.isfile(npy_path)
        _report(
            f"[{model_name}] CM .npy exists @ {snr_target:+d} dB",
            npy_ok,
            f"expected path: {npy_path}",
        )
        if npy_ok:
            cm = np.load(npy_path)
            _report(
                f"[{model_name}] CM shape == (11, 11) @ {snr_target:+d} dB  "
                f"(got {cm.shape})",
                cm.shape == (NUM_CLASSES_MOCK, NUM_CLASSES_MOCK),
            )
            _report(
                f"[{model_name}] CM dtype is integer @ {snr_target:+d} dB",
                np.issubdtype(cm.dtype, np.integer),
                f"got dtype={cm.dtype}",
            )
            _report(
                f"[{model_name}] CM sum == N_PER_SNR_TEST={N_PER_SNR_TEST} @ {snr_target:+d} dB",
                cm.sum() == N_PER_SNR_TEST,
                f"got sum={cm.sum()}",
            )

        png_path = snr_entry.get("png", "")
        _report(
            f"[{model_name}] CM .png exists @ {snr_target:+d} dB",
            os.path.isfile(png_path),
        )

    # ----------------------------------------------------------
    # Latency and overall accuracy sanity checks
    # ----------------------------------------------------------
    lat = results.get("inference_latency_ms", -1.0)
    _report(f"[{model_name}] Latency > 0 ms  (got {lat:.3f} ms)",
            lat > 0.0)

    oa = results.get("overall_accuracy", -1.0)
    _report(f"[{model_name}] Overall accuracy in [0, 1]  (got {oa:.4f})",
            0.0 <= oa <= 1.0)

    n_snr = results.get("n_snr_levels", 0)
    _report(f"[{model_name}] n_snr_levels == 20  (got {n_snr})",
            n_snr == 20)


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    # Use a temporary directory so nothing pollutes the real
    # subsystem2_1d/checkpoints/ or subsystem2_1d/results/ directories.
    tmp_root     = tempfile.mkdtemp(prefix="subsystem2_pipeline_test_")
    tmp_ckpt_dir = os.path.join(tmp_root, "checkpoints")
    tmp_res_dir  = os.path.join(tmp_root, "results")
    os.makedirs(tmp_ckpt_dir, exist_ok=True)
    os.makedirs(tmp_res_dir,  exist_ok=True)

    print(f"\n{BOLD}{'='*68}{RESET}")
    print(f"{BOLD}  Subsystem 2 -- Training & Evaluation Pipeline Tests{RESET}")
    print(f"{BOLD}  feature/1d-model-p2 | Person 2{RESET}")
    print(f"{BOLD}{'='*68}{RESET}")
    print(f"  {INFO} Temp dir: {tmp_root}")
    print(f"  {INFO} Train set: {N_TRAIN} samples  |  Val: {N_VAL}  |"
          f"  Test: {len(SNR_LEVELS) * N_PER_SNR_TEST} (all 20 SNRs covered)")

    try:
        # T-1: EarlyStopping unit tests (no model required)
        run_t1_early_stopping()

        # T-2 / T-3: Training smoke tests
        print(f"\n{BOLD}{'='*68}{RESET}")
        print(f"{BOLD}  Training Smoke Tests (2 epochs each){RESET}")
        print(f"{BOLD}{'='*68}{RESET}")

        ckpt_cnn1d  = run_training_smoke_test("cnn1d",           tmp_ckpt_dir)
        ckpt_hybrid = run_training_smoke_test("cnn_transformer",  tmp_ckpt_dir)

        # T-4 / T-5: Evaluation dry-runs
        print(f"\n{BOLD}{'='*68}{RESET}")
        print(f"{BOLD}  Evaluation Dry-Runs{RESET}")
        print(f"{BOLD}{'='*68}{RESET}")

        run_eval_dry_run("cnn1d",          ckpt_cnn1d,  tmp_res_dir)
        run_eval_dry_run("cnn_transformer", ckpt_hybrid, tmp_res_dir)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\n  {INFO} Temp directory cleaned up.")

    # ----------------------------------------------------------
    # Final Summary
    # ----------------------------------------------------------
    p = _results["passed"]
    f = _results["failed"]

    print(f"\n{BOLD}{'='*68}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"  {GREEN}Passed : {p}{RESET}")
    print(f"  {RED}Failed : {f}{RESET}")
    print(f"{BOLD}{'='*68}{RESET}")

    if f > 0:
        print(f"\n{RED}{BOLD}  [!!] {f} test(s) FAILED.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}  [OK] All {p} pipeline tests PASSED!{RESET}")
        sys.exit(0)
