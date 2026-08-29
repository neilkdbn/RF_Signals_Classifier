# -*- coding: utf-8 -*-
# precompute_stft.py
# =========================================================================
# One-time preprocessing step: converts all 220,000 IQ signals in X_all.npy
# into pre-computed 3-channel STFT spectrograms and saves to:
#   ./data/spectrograms_all.npy  →  shape (220000, 3, 64, 5), float32
#
# Channels stored:
#   ch0: power_db   (used alone in "grayscale" mode)
#   ch1: phase
#   ch2: magnitude  (ch0+ch1+ch2 used in "hybrid" mode)
#
# Run ONCE before training:
#   python precompute_stft.py
#
# Output file is memory-mapped during training — no full load into RAM.
# =========================================================================

import os
import time
import warnings
import numpy as np
from scipy.signal import stft as scipy_stft

warnings.filterwarnings("ignore", message=".*complex.*return_onesided.*")

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR    = "./data"
INPUT_FILE  = os.path.join(DATA_DIR, "X_all.npy")
OUTPUT_FILE = os.path.join(DATA_DIR, "spectrograms_all.npy")

STFT_PARAMS = dict(window=("kaiser", 0.85), nperseg=32, noverlap=28)
EXPECTED_H, EXPECTED_W = 32, 25
BATCH_REPORT = 10_000          # print progress every N samples


# ── Core converter ────────────────────────────────────────────────────────

def compute_hybrid_spectrogram(signal_np: np.ndarray) -> np.ndarray:
    """
    Convert one (2, 128) IQ signal → (3, 64, 5) float32 spectrogram.
    Uses identical Kaiser-windowed STFT parameters as stft_converter.py.

    Returns:
        np.ndarray shape (3, H, W):
            [0] power_db   — 10·log10(|STFT|² + ε)
            [1] phase      — angle(STFT)
            [2] magnitude  — |STFT|
    """
    I, Q = signal_np[0], signal_np[1]
    complex_signal = I + 1j * Q

    _, _, Zxx = scipy_stft(complex_signal, **STFT_PARAMS)
    Zxx = np.fft.fftshift(Zxx, axes=0)               # centre DC

    power_db  = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-10).astype(np.float32)
    phase     = np.angle(Zxx).astype(np.float32)
    magnitude = np.abs(Zxx).astype(np.float32)

    return np.stack([power_db, phase, magnitude], axis=0)  # (3, H, W)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("   STFT PRE-COMPUTATION  --  stft_converter -> spectrograms_all.npy")
    print("=" * 65)

    # ── Verify input ──────────────────────────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    if os.path.exists(OUTPUT_FILE):
        print(f"\n[SKIP] Output already exists: {OUTPUT_FILE}")
        arr = np.load(OUTPUT_FILE, mmap_mode="r")
        print(f"       Shape: {arr.shape}  dtype: {arr.dtype}")
        print("\nDelete the file and re-run to force recompute.")
        raise SystemExit(0)

    # ── Load raw IQ ───────────────────────────────────────────────────────
    print(f"\nLoading {INPUT_FILE} …")
    X = np.load(INPUT_FILE)                            # (N, 2, 128)
    N = len(X)
    print(f"  Loaded {N:,} signals  shape={X.shape}  dtype={X.dtype}")

    # Validate STFT output shape with one sample
    sample_spec = compute_hybrid_spectrogram(X[0])
    H, W = sample_spec.shape[1], sample_spec.shape[2]
    assert (H, W) == (EXPECTED_H, EXPECTED_W), \
        f"Unexpected spectrogram shape ({H}, {W}); expected ({EXPECTED_H}, {EXPECTED_W})"
    print(f"  STFT output validated: (3, {H}, {W})  OK")

    # ── Pre-allocate output array ─────────────────────────────────────────
    out_shape = (N, 3, H, W)
    ram_mb = N * 3 * H * W * 4 / 1e6
    print(f"\nAllocating output array {out_shape}  ({ram_mb:.0f} MB RAM) …")
    spectrograms = np.empty(out_shape, dtype=np.float32)

    # ── Convert ──────────────────────────────────────────────────────────
    print(f"\nConverting {N:,} signals — reporting every {BATCH_REPORT:,} …\n")
    t_start = time.perf_counter()

    for i in range(N):
        spectrograms[i] = compute_hybrid_spectrogram(X[i])

        if (i + 1) % BATCH_REPORT == 0 or i == 0:
            elapsed  = time.perf_counter() - t_start
            rate     = (i + 1) / elapsed
            eta_s    = (N - i - 1) / rate
            eta_m    = eta_s / 60
            pct      = 100.0 * (i + 1) / N
            print(f"  [{pct:5.1f}%]  {i+1:>7,}/{N:,}"
                  f"  |  {rate:,.0f} sig/s"
                  f"  |  ETA: {eta_m:.1f} min")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n[DONE] Converted {N:,} signals in {elapsed_total/60:.1f} min  "
          f"({elapsed_total:.1f}s)")

    # ── Save as raw .npy (supports mmap_mode during training) ────────────
    print(f"\nSaving to {OUTPUT_FILE} …")
    np.save(OUTPUT_FILE, spectrograms)
    file_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"[SAVED]  {OUTPUT_FILE}  ({file_mb:.0f} MB on disk)")
    print(f"  Shape : {spectrograms.shape}")
    print(f"  dtype : {spectrograms.dtype}")
    print(f"\n  ✓ Ready for fast training. Run: python train_2d.py")
