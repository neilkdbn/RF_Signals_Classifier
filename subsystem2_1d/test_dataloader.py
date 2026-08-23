# subsystem2_1d/test_dataloader.py
# =========================================================
# Subsystem 2 -- End-to-End Verification Test Suite
# Person 2 | feature/1d-model-p2
#
# Run from the repo root:
#   python subsystem2_1d/test_dataloader.py
#
# Tests are split into two groups:
#   GROUP A — Always run (no data files required)
#   GROUP B — Run only if data/X_all.npy is present
# =========================================================

import sys
import os

# Force UTF-8 output on Windows so Unicode symbols don't crash cp1252 terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch

# Ensure repo root is on path (so `from subsystem2_1d import ...` works)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from subsystem2_1d.normalization_1d import normalize_1d_iq, IQAugmentation
from subsystem2_1d.dataloader_1d    import get_1d_dataloaders, RFSignalDataset1D

# =========================================================
# Colour helpers for pretty terminal output
# =========================================================
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
SKIP = f"{YELLOW}[SKIP]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"
OK   = "[OK] "
BAD  = "[!!]"

_results = {"passed": 0, "failed": 0, "skipped": 0}

def _report(label, passed, detail=""):
    if passed:
        _results["passed"] += 1
        print(f"  {PASS}  {label}")
    else:
        _results["failed"] += 1
        print(f"  {FAIL}  {label}")
    if detail:
        print(f"         {detail}")

def _skip(label, reason=""):
    _results["skipped"] += 1
    print(f"  {SKIP}  {label}  ({reason})")


# =========================================================
# GROUP A — No dataset files required
# =========================================================

def run_group_a():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD} GROUP A — Core Utility Tests (no data files needed){RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # ----------------------------------------------------------
    # A-1: normalize_1d_iq — single sample (2, 128)
    # ----------------------------------------------------------
    print(f"\n{CYAN}A-1  normalize_1d_iq — single sample (2, 128){RESET}")
    raw_single = np.random.randn(2, 128).astype(np.float32) * 5.0 + 3.0
    out_single = normalize_1d_iq(raw_single)

    _report("Output shape is (2, 128)", out_single.shape == (2, 128),
            f"got {out_single.shape}")
    _report("Output dtype is float32",  out_single.dtype == np.float32,
            f"got {out_single.dtype}")

    for ch, name in enumerate(["I", "Q"]):
        ch_mean = float(out_single[ch].mean())
        ch_std  = float(out_single[ch].std())
        _report(
            f"Channel {name}: mean ~= 0.0  (got {ch_mean:+.6f})",
            abs(ch_mean) < 1e-4,
        )
        _report(
            f"Channel {name}: std  ~= 1.0  (got {ch_std:.6f})",
            abs(ch_std - 1.0) < 1e-3,
        )

    # ----------------------------------------------------------
    # A-2: normalize_1d_iq — batched (N, 2, 128)
    # ----------------------------------------------------------
    print(f"\n{CYAN}A-2  normalize_1d_iq — batched (N=32, 2, 128){RESET}")
    N = 32
    raw_batch = np.random.randn(N, 2, 128).astype(np.float32) * 10.0 + 7.0
    out_batch = normalize_1d_iq(raw_batch)

    _report("Output shape is (32, 2, 128)", out_batch.shape == (N, 2, 128),
            f"got {out_batch.shape}")

    # Check per-sample statistics — every sample should have mean≈0, std≈1
    means = out_batch.mean(axis=2)  # (N, 2)
    stds  = out_batch.std(axis=2)   # (N, 2)
    _report(
        f"All per-sample channel means ~= 0.0  (max |mean| = {np.abs(means).max():.6f})",
        np.all(np.abs(means) < 1e-3),
    )
    _report(
        f"All per-sample channel stds  ~= 1.0  (max |std-1| = {np.abs(stds - 1.0).max():.6f})",
        np.all(np.abs(stds - 1.0) < 1e-2),
    )

    # ----------------------------------------------------------
    # A-3: normalize_1d_iq — unsupported shape raises ValueError
    # ----------------------------------------------------------
    print(f"\n{CYAN}A-3  normalize_1d_iq — bad shape raises ValueError{RESET}")
    try:
        normalize_1d_iq(np.zeros((128,)))
        _report("ValueError raised on 1-D input", False,
                "Expected ValueError but none was raised!")
    except ValueError:
        _report("ValueError raised on 1-D input", True)

    # ----------------------------------------------------------
    # A-4: IQAugmentation — AWGN noise injection
    # ----------------------------------------------------------
    print(f"\n{CYAN}A-4  IQAugmentation.add_awgn_noise{RESET}")
    aug = IQAugmentation(enabled=True)
    iq  = torch.randn(2, 128)

    noisy = aug.add_awgn_noise(iq, snr_db=5.0)

    _report("Output shape preserved (2, 128)",   noisy.shape == (2, 128),
            f"got {noisy.shape}")
    _report("Output is a torch.Tensor",          isinstance(noisy, torch.Tensor))
    _report("Noise was actually added (tensors differ)", not torch.allclose(iq, noisy))

    # ----------------------------------------------------------
    # A-5: IQAugmentation — random phase rotation
    # ----------------------------------------------------------
    print(f"\n{CYAN}A-5  IQAugmentation.random_phase_rotation{RESET}")
    rotated = aug.random_phase_rotation(iq)

    _report("Output shape preserved (2, 128)",   rotated.shape == (2, 128),
            f"got {rotated.shape}")

    # Rotation preserves L2-norm of the complex signal
    orig_power    = (iq[0]**2 + iq[1]**2).sum().item()
    rotated_power = (rotated[0]**2 + rotated[1]**2).sum().item()
    _report(
        f"L2-norm preserved after rotation  "
        f"(orig={orig_power:.4f}, rot={rotated_power:.4f})",
        abs(orig_power - rotated_power) < 1e-3,
    )

    # ----------------------------------------------------------
    # A-6: IQAugmentation — disabled mode returns original tensor
    # ----------------------------------------------------------
    print(f"\n{CYAN}A-6  IQAugmentation disabled mode{RESET}")
    aug_off = IQAugmentation(enabled=False)
    result  = aug_off(iq)
    _report("Disabled augmenter returns input unchanged",
            torch.allclose(iq, result))

    # ----------------------------------------------------------
    # A-7: RFSignalDataset1D — with synthetic data
    # ----------------------------------------------------------
    print(f"\n{CYAN}A-7  RFSignalDataset1D — synthetic data smoke test{RESET}")
    X_syn = np.random.randn(200, 2, 128).astype(np.float32)
    y_syn = np.random.randint(0, 11, size=(200,), dtype=np.int64)
    ds    = RFSignalDataset1D(X_syn, y_syn)

    _report("Dataset __len__ == 200",       len(ds) == 200)
    _report("num_classes property correct", ds.num_classes == 11,
            f"got {ds.num_classes}")

    sig, lbl = ds[0]
    _report("__getitem__ signal shape (2, 128)", sig.shape == (2, 128),
            f"got {sig.shape}")
    _report("__getitem__ signal dtype float32",  sig.dtype == torch.float32,
            f"got {sig.dtype}")
    _report("__getitem__ label dtype int64",     lbl.dtype == torch.int64,
            f"got {lbl.dtype}")

    # ----------------------------------------------------------
    # A-8: RFSignalDataset1D — wrong shape raises ValueError
    # ----------------------------------------------------------
    print(f"\n{CYAN}A-8  RFSignalDataset1D — shape guard{RESET}")
    try:
        _ = RFSignalDataset1D(np.zeros((200, 1, 128), dtype=np.float32), y_syn)
        _report("ValueError raised on wrong signal shape", False,
                "Expected ValueError but none was raised!")
    except ValueError:
        _report("ValueError raised on wrong signal shape", True)


# =========================================================
# GROUP B — Requires data/ directory with .npy files
# =========================================================

def run_group_b(data_dir: str = "./data"):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD} GROUP B — Full DataLoader Tests (requires data/ files){RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Resolve the data directory
    candidates = [
        data_dir,
        os.path.join(_REPO_ROOT, "data"),
    ]
    resolved = None
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "X_all.npy")):
            resolved = os.path.abspath(c)
            break

    if resolved is None:
        skip_msg = (
            "data/X_all.npy not found — run 'python dataset_agent.py' first"
        )
        for label in [
            "train/val/test DataLoaders initialize successfully",
            "train batch shape == [B, 2, 128]",
            "label batch shape == [B]",
            "No index overlap across train / val / test",
        ]:
            _skip(label, skip_msg)
        return

    print(f"  {INFO}  data directory resolved: {resolved}")

    # ----------------------------------------------------------
    # B-1: DataLoaders initialize
    # ----------------------------------------------------------
    print(f"\n{CYAN}B-1  get_1d_dataloaders — initialization{RESET}")
    BATCH = 64
    try:
        train_loader, val_loader, test_loader = get_1d_dataloaders(
            data_dir=resolved,
            batch_size=BATCH,
            num_workers=0,   # safe default for CI / Windows
        )
        _report("train_loader created", True)
        _report("val_loader   created", True)
        _report("test_loader  created", True)
    except Exception as exc:
        _report("DataLoaders created without exception", False, str(exc))
        return

    # ----------------------------------------------------------
    # B-2: Batch shapes
    # ----------------------------------------------------------
    print(f"\n{CYAN}B-2  First training batch — shape verification{RESET}")
    signals, labels = next(iter(train_loader))

    _report(
        f"Signal batch shape == [{BATCH}, 2, 128]  (got {list(signals.shape)})",
        signals.shape == torch.Size([BATCH, 2, 128]),
    )
    _report(
        f"Label  batch shape == [{BATCH}]  (got {list(labels.shape)})",
        labels.shape == torch.Size([BATCH]),
    )
    _report("Signal dtype is torch.float32", signals.dtype == torch.float32,
            f"got {signals.dtype}")
    _report("Label  dtype is torch.int64",   labels.dtype  == torch.int64,
            f"got {labels.dtype}")

    # ----------------------------------------------------------
    # B-3: Split sizes add up to the total dataset
    # ----------------------------------------------------------
    print(f"\n{CYAN}B-3  Split size sanity check{RESET}")
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    total   = n_train + n_val + n_test

    print(f"  {INFO}  train={n_train:,}  val={n_val:,}  test={n_test:,}  total={total:,}")

    _report("Total samples = 220 000", total == 220_000, f"got {total}")
    _report("Train ~= 70%",
            0.68 < n_train / total < 0.72,
            f"ratio={n_train/total:.3f}")
    _report("Val   ~= 20%",
            0.18 < n_val   / total < 0.22,
            f"ratio={n_val/total:.3f}")
    _report("Test  ~= 10%",
            0.08 < n_test  / total < 0.12,
            f"ratio={n_test/total:.3f}")

    # ----------------------------------------------------------
    # B-4: No index overlap between splits
    # ----------------------------------------------------------
    print(f"\n{CYAN}B-4  Index overlap verification{RESET}")
    train_idx = np.load(os.path.join(resolved, "train_idx.npy"))
    val_idx   = np.load(os.path.join(resolved, "val_idx.npy"))
    test_idx  = np.load(os.path.join(resolved, "test_idx.npy"))

    train_set = set(train_idx.tolist())
    val_set   = set(val_idx.tolist())
    test_set  = set(test_idx.tolist())

    tv_overlap  = len(train_set & val_set)
    tt_overlap  = len(train_set & test_set)
    vt_overlap  = len(val_set   & test_set)
    all_overlap = len(train_set & val_set & test_set)

    _report(f"Train & Val  overlap = 0  (found {tv_overlap})",   tv_overlap  == 0)
    _report(f"Train & Test overlap = 0  (found {tt_overlap})",   tt_overlap  == 0)
    _report(f"Val   & Test overlap = 0  (found {vt_overlap})",   vt_overlap  == 0)
    _report(f"Train & Val & Test   = 0  (found {all_overlap})",  all_overlap == 0)

    union_size = len(train_set | val_set | test_set)
    _report(
        f"Union of all indices == {total}  (got {union_size})",
        union_size == total,
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Subsystem 2 -- 1D DataLoader Verification Suite{RESET}")
    print(f"{BOLD}  feature/1d-model-p2 | Person 2{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    run_group_a()
    run_group_b()

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    p = _results["passed"]
    f = _results["failed"]
    s = _results["skipped"]

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"  {GREEN}Passed  : {p}{RESET}")
    print(f"  {RED}Failed  : {f}{RESET}")
    print(f"  {YELLOW}Skipped : {s}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    if f > 0:
        print(f"\n{RED}{BOLD}  [!!] {f} test(s) FAILED.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}  [OK] All runnable tests PASSED!{RESET}")
        sys.exit(0)
