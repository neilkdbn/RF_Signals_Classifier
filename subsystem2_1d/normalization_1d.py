# subsystem2_1d/normalization_1d.py
# =========================================================
# Subsystem 2 — 1D IQ Normalization & Augmentation
# Person 2 | feature/1d-model-p2
#
# NOTE: Person 1 (dataset_agent.py) already applies offline
#       per-sample, per-channel zero-mean / unit-variance
#       normalization when generating X_all.npy.
#       The normalize_1d_iq() function here is therefore
#       intended for:
#         (a) online/on-the-fly use when raw, un-normalized
#             IQ arrays enter the pipeline at inference time.
#         (b) unit-testing and verification purposes.
#       Do NOT double-normalize the pre-processed X_all.npy
#       arrays — they are already normalized.
# =========================================================

import math
import numpy as np
import torch


# ---------------------------------------------------------
# 1. Normalization Function
# ---------------------------------------------------------

def normalize_1d_iq(iq_samples: np.ndarray) -> np.ndarray:
    """
    Per-instance, per-channel zero-mean / unit-variance normalization
    for raw IQ signal arrays.

    Formula:
        normalized = (x - μ) / (σ + 1e-6)

    where μ and σ are computed independently for each channel (I and Q)
    within each sample instance.

    Args:
        iq_samples (np.ndarray):
            Either a single sample of shape  (2, 128)
            or a batch of samples of shape   (N, 2, 128).

    Returns:
        np.ndarray: Normalized array of the same shape as input, dtype float32.

    Raises:
        ValueError: If input array has an unsupported number of dimensions.

    Example:
        >>> raw = np.random.randn(2, 128).astype(np.float32)
        >>> out = normalize_1d_iq(raw)
        >>> assert out.shape == (2, 128)
        >>> # Per-channel mean ≈ 0.0, std ≈ 1.0
    """
    iq_samples = np.asarray(iq_samples, dtype=np.float32)

    if iq_samples.ndim == 2:
        # Single sample: (2, 128)
        # Compute stats along axis=1 (time axis), keep dims for broadcasting
        mean = iq_samples.mean(axis=1, keepdims=True)  # (2, 1)
        std  = iq_samples.std(axis=1,  keepdims=True)  # (2, 1)
        return (iq_samples - mean) / (std + 1e-6)

    elif iq_samples.ndim == 3:
        # Batched: (N, 2, 128)
        # Compute stats along axis=2 (time axis), keep dims for broadcasting
        mean = iq_samples.mean(axis=2, keepdims=True)  # (N, 2, 1)
        std  = iq_samples.std(axis=2,  keepdims=True)  # (N, 2, 1)
        return (iq_samples - mean) / (std + 1e-6)

    else:
        raise ValueError(
            f"normalize_1d_iq expects input of shape (2, 128) or (N, 2, 128), "
            f"got shape {iq_samples.shape}."
        )


# ---------------------------------------------------------
# 2. IQ Augmentation Class
# ---------------------------------------------------------

class IQAugmentation:
    """
    Physical-layer data augmentation for 1D complex IQ signals.

    Each augmentation method operates on a PyTorch tensor of shape (2, 128),
    where channel 0 = I (In-Phase) and channel 1 = Q (Quadrature).

    All operations model realistic RF channel distortions to improve
    the model's robustness to unseen propagation conditions.

    Args:
        enabled (bool): Master switch. If False, __call__ returns the
                        input tensor unchanged. Useful for toggling
                        augmentation between train and eval modes.

    Example:
        >>> aug = IQAugmentation(enabled=True)
        >>> iq_tensor = torch.randn(2, 128)
        >>> noisy = aug.add_awgn_noise(iq_tensor, snr_db=10.0)
        >>> rotated = aug.random_phase_rotation(iq_tensor)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    # ----------------------------------------------------------
    # 2a. AWGN Noise Injection
    # ----------------------------------------------------------

    def add_awgn_noise(
        self, iq: torch.Tensor, snr_db: float = 10.0
    ) -> torch.Tensor:
        """
        Adds Additive White Gaussian Noise (AWGN) to simulate a noisy channel.

        The noise power is derived from the target Signal-to-Noise Ratio (SNR)
        in decibels:
            N_power = S_power / 10^(SNR_dB / 10)

        where S_power is the mean power of the input IQ signal.

        Args:
            iq     (torch.Tensor): Input IQ tensor of shape (2, 128).
            snr_db (float):        Target SNR in dB. Lower values = more noise.
                                   Typical range: [-20, 30] dB.

        Returns:
            torch.Tensor: Noise-corrupted IQ tensor of shape (2, 128).
        """
        if not self.enabled:
            return iq

        # Compute signal power per channel independently
        signal_power = iq.pow(2).mean(dim=1, keepdim=True)  # (2, 1)

        # Derive noise power from SNR
        snr_linear   = 10.0 ** (snr_db / 10.0)
        noise_power  = signal_power / snr_linear              # (2, 1)
        noise_std    = torch.sqrt(noise_power)                # (2, 1)

        # Sample Gaussian noise scaled to the computed std
        noise = torch.randn_like(iq) * noise_std             # (2, 128)

        return iq + noise

    # ----------------------------------------------------------
    # 2b. Random Phase Rotation
    # ----------------------------------------------------------

    def random_phase_rotation(self, iq: torch.Tensor) -> torch.Tensor:
        """
        Applies a random constant phase rotation to the complex IQ signal.

        Models carrier phase offset, a common impairment in real RF systems.
        The rotation is:
            z_rotated = z * e^(j * θ)

        where z = I + jQ and θ is uniformly sampled from [0, 2π).

        In Cartesian coordinates (I, Q), this is:
            I' = I * cos(θ) - Q * sin(θ)
            Q' = I * sin(θ) + Q * cos(θ)

        Args:
            iq (torch.Tensor): Input IQ tensor of shape (2, 128).

        Returns:
            torch.Tensor: Phase-rotated IQ tensor of shape (2, 128).
        """
        if not self.enabled:
            return iq

        # Sample a random phase uniformly from [0, 2π)
        theta = torch.empty(1).uniform_(0.0, 2.0 * math.pi).item()

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        I = iq[0]  # (128,)
        Q = iq[1]  # (128,)

        I_rotated = cos_t * I - sin_t * Q
        Q_rotated = sin_t * I + cos_t * Q

        return torch.stack([I_rotated, Q_rotated], dim=0)  # (2, 128)

    # ----------------------------------------------------------
    # 2c. Callable Interface — compose both augmentations
    # ----------------------------------------------------------

    def __call__(
        self,
        iq: torch.Tensor,
        apply_noise: bool = True,
        snr_db: float = 10.0,
        apply_rotation: bool = True,
    ) -> torch.Tensor:
        """
        Apply the full augmentation pipeline to a single IQ tensor.

        Args:
            iq             (torch.Tensor): Shape (2, 128).
            apply_noise    (bool):         Whether to inject AWGN noise.
            snr_db         (float):        SNR for AWGN injection (dB).
            apply_rotation (bool):         Whether to apply phase rotation.

        Returns:
            torch.Tensor: Augmented IQ tensor of shape (2, 128).
        """
        if not self.enabled:
            return iq

        if apply_noise:
            iq = self.add_awgn_noise(iq, snr_db=snr_db)

        if apply_rotation:
            iq = self.random_phase_rotation(iq)

        return iq

    def __repr__(self) -> str:
        return f"IQAugmentation(enabled={self.enabled})"
