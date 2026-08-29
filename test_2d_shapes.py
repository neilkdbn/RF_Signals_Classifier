# test_2d_shapes.py
# =========================================================================
# CHUNK 1: DevOps Scaffolding & 2D Data Ingestion
# Validates that Person 3's Kaiser-windowed STFT converter produces
# correctly shaped 2D tensors for downstream CNN models (ResNet-18, STFT-RADN).
# Uses synthetic I/Q data so this runs in CI without real dataset files.
# =========================================================================

import sys
import numpy as np
import torch
from scipy.signal import stft as scipy_stft


# ──────────────────────────────────────────────────────────────
# Constants: Must match Person 3's stft_converter.py parameters
# ──────────────────────────────────────────────────────────────

STFT_WINDOW = ("kaiser", 0.85)
STFT_NPERSEG = 64
STFT_NOVERLAP = 32
SEQ_LEN = 128          # RadioML 2016.10a I/Q sequence length
BATCH_SIZE = 8

# Expected spectrogram dimensions from complex-input two-sided STFT:
#   H = nperseg = 32  (frequency bins, two-sided for complex signal)
#   W = 33             (time bins)
EXPECTED_H = 32
EXPECTED_W = 33


# ──────────────────────────────────────────────────────────────
# Converter Utilities (mirror Person 3's logic for CI isolation)
# ──────────────────────────────────────────────────────────────

def compute_spectrogram(signal_np):
    """
    Convert a single (2, 128) I/Q numpy array to a 2D power spectrogram.
    Mirrors the exact logic in stft_converter.py → compute_stft().

    Returns:
        power_db: np.ndarray of shape (H, W)
    """
    I = signal_np[0]
    Q = signal_np[1]
    complex_signal = I + 1j * Q

    _, _, Zxx = scipy_stft(
        complex_signal,
        window=STFT_WINDOW,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
    )

    power = np.abs(Zxx) ** 2
    power_db = 10 * np.log10(power + 1e-10)

    # Center zero-frequency (matches Person 3's fftshift)
    power_db = np.fft.fftshift(power_db, axes=0)

    return power_db


def batch_to_grayscale_tensor(signals_np):
    """
    Convert a batch of (B, 2, 128) I/Q signals to grayscale spectrogram tensors.

    Returns:
        torch.Tensor of shape [B, 1, H, W]
    """
    spectrograms = []
    for i in range(signals_np.shape[0]):
        spec = compute_spectrogram(signals_np[i])
        spectrograms.append(spec)

    specs = np.array(spectrograms, dtype=np.float32)  # (B, H, W)
    tensor = torch.from_numpy(specs).unsqueeze(1)      # (B, 1, H, W)
    return tensor


def batch_to_hybrid_tensor(signals_np):
    """
    Convert a batch of (B, 2, 128) I/Q signals to 3-channel hybrid tensors.
    Channels: [Power (dB), Phase, Magnitude]
    Compatible with pretrained ResNet-18 that expects 3-channel input.

    Returns:
        torch.Tensor of shape [B, 3, H, W]
    """
    batch = []
    for i in range(signals_np.shape[0]):
        I = signals_np[i, 0]
        Q = signals_np[i, 1]
        complex_signal = I + 1j * Q

        _, _, Zxx = scipy_stft(
            complex_signal,
            window=STFT_WINDOW,
            nperseg=STFT_NPERSEG,
            noverlap=STFT_NOVERLAP,
        )
        Zxx = np.fft.fftshift(Zxx, axes=0)

        power_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)
        phase = np.angle(Zxx)
        magnitude = np.abs(Zxx)

        hybrid = np.stack([power_db, phase, magnitude], axis=0)  # (3, H, W)
        batch.append(hybrid)

    batch_np = np.array(batch, dtype=np.float32)  # (B, 3, H, W)
    return torch.from_numpy(batch_np)


# ──────────────────────────────────────────────────────────────
# Test Suite
# ──────────────────────────────────────────────────────────────

def test_single_spectrogram_shape():
    """Verify a single spectrogram has shape (H, W) = (64, 5)."""
    signal = np.random.randn(2, SEQ_LEN).astype(np.float32)
    spec = compute_spectrogram(signal)

    assert spec.shape == (EXPECTED_H, EXPECTED_W), \
        f"Expected ({EXPECTED_H}, {EXPECTED_W}), got {spec.shape}"
    print(f"[PASS] Single spectrogram shape: {spec.shape}")


def test_grayscale_batch_shape():
    """Verify grayscale batch has shape [B, 1, H, W]."""
    signals = np.random.randn(BATCH_SIZE, 2, SEQ_LEN).astype(np.float32)
    tensor = batch_to_grayscale_tensor(signals)

    expected = (BATCH_SIZE, 1, EXPECTED_H, EXPECTED_W)
    assert tensor.shape == torch.Size(expected), \
        f"Expected {expected}, got {tuple(tensor.shape)}"
    assert tensor.dtype == torch.float32, \
        f"Expected float32, got {tensor.dtype}"
    print(f"[PASS] Grayscale batch shape: {tuple(tensor.shape)}")


def test_hybrid_batch_shape():
    """Verify 3-channel hybrid batch has shape [B, 3, H, W]."""
    signals = np.random.randn(BATCH_SIZE, 2, SEQ_LEN).astype(np.float32)
    tensor = batch_to_hybrid_tensor(signals)

    expected = (BATCH_SIZE, 3, EXPECTED_H, EXPECTED_W)
    assert tensor.shape == torch.Size(expected), \
        f"Expected {expected}, got {tuple(tensor.shape)}"
    assert tensor.dtype == torch.float32, \
        f"Expected float32, got {tensor.dtype}"
    print(f"[PASS] Hybrid batch shape: {tuple(tensor.shape)}")


def test_no_nan_or_inf():
    """Verify no NaN or Inf values in converted tensors."""
    signals = np.random.randn(BATCH_SIZE, 2, SEQ_LEN).astype(np.float32)

    gray = batch_to_grayscale_tensor(signals)
    assert not torch.isnan(gray).any(), "NaN detected in grayscale tensor"
    assert not torch.isinf(gray).any(), "Inf detected in grayscale tensor"

    hybrid = batch_to_hybrid_tensor(signals)
    assert not torch.isnan(hybrid).any(), "NaN detected in hybrid tensor"
    assert not torch.isinf(hybrid).any(), "Inf detected in hybrid tensor"

    print("[PASS] No NaN or Inf values detected in any tensor")


def test_stft_converter_consistency():
    """Verify our logic matches Person 3's stft_converter.compute_stft()."""
    from stft_converter import compute_stft

    signal = np.random.randn(2, SEQ_LEN).astype(np.float32)

    _, _, p3_spec = compute_stft(signal)
    our_spec = compute_spectrogram(signal)

    assert p3_spec.shape == our_spec.shape, \
        f"Shape mismatch: Person3={p3_spec.shape} vs ours={our_spec.shape}"
    assert np.allclose(p3_spec, our_spec, atol=1e-5), \
        "Numerical mismatch with Person 3's stft_converter.compute_stft()"
    print(f"[PASS] Consistency with stft_converter.py verified (shape={our_spec.shape})")


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("   CHUNK 1: 2D TENSOR SHAPE VALIDATION SUITE")
    print("=" * 58)

    tests = [
        test_single_spectrogram_shape,
        test_grayscale_batch_shape,
        test_hybrid_batch_shape,
        test_no_nan_or_inf,
        test_stft_converter_consistency,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 58)
    print(f"   RESULTS: {passed} passed, {failed} failed")
    print("=" * 58)

    if failed > 0:
        sys.exit(1)

    print("\n[SUCCESS] All 2D tensor shape validations passed!")
    sys.exit(0)
