import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from load_mat_dataset import load_matlab_signals


def compute_stft(signal):
    """
    Convert one I/Q signal into a 2D power spectrogram.

    Input:
        signal: shape (2, 128)
                signal[0] = I
                signal[1] = Q

    Returns:
        frequencies: frequency bins
        times: time bins
        power_db: 2D power spectrogram
    """

    # Separate I and Q
    I = signal[0]
    Q = signal[1]

    # Convert I/Q into a complex RF signal
    complex_signal = I + 1j * Q

    # Compute two-sided STFT
    frequencies, times, Zxx = stft(
        complex_signal,
        window=("kaiser", 0.85),
        nperseg=64,
        noverlap=32
    )

    # Convert complex STFT to power
    power = np.abs(Zxx) ** 2

    # Convert power to dB
    power_db = 10 * np.log10(power + 1e-10)

    # Shift zero frequency to the center
    frequencies = np.fft.fftshift(frequencies)
    power_db = np.fft.fftshift(power_db, axes=0)

    return frequencies, times, power_db


def convert_dataset(dataset):
    """
    Convert the complete dataset into 2D STFT spectrograms.

    Input:
        dataset: PyTorch TensorDataset containing
                 signals of shape (2, 128)

    Returns:
        spectrograms: NumPy array of shape (N, 64, 5)
        labels: NumPy array of shape (N,)
    """

    spectrograms = []
    labels = []

    print("\n==================================================")
    print("       CONVERTING DATASET TO 2D STFT")
    print("==================================================")

    total = len(dataset)

    for i in range(total):

        signal, label = dataset[i]

        # Convert PyTorch tensor to NumPy
        signal = signal.numpy()

        # Compute STFT spectrogram
        _, _, power_db = compute_stft(signal)

        spectrograms.append(power_db)
        labels.append(label.item())

        # Progress update
        if (i + 1) % 500 == 0 or i == 0:
            print(f"Processed {i + 1}/{total} signals")

    # Convert lists into NumPy arrays
    spectrograms = np.array(spectrograms, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print("\n[PASS] Dataset conversion complete!")
    print(f"-> Spectrogram shape: {spectrograms.shape}")
    print(f"-> Labels shape     : {labels.shape}")

    return spectrograms, labels


if __name__ == "__main__":

    # Load MATLAB dataset
    dataset, modulations = load_matlab_signals()

    print(f"\nTotal signals: {len(dataset)}")

    # Convert all 5000 signals
    spectrograms, labels = convert_dataset(dataset)

    # Check label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)

    print("\nLabel distribution:")

    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} samples")

    # Save the converted dataset
    np.savez(
        "stft_dataset.npz",
        spectrograms=spectrograms,
        labels=labels
    )

    print("\n[PASS] Saved STFT dataset:")
    print("-> stft_dataset.npz")

    # Plot the first spectrogram
    signal, label = dataset[0]

    frequencies, times, power_db = compute_stft(signal.numpy())

    plt.figure(figsize=(8, 5))

    plt.pcolormesh(
        times,
        frequencies,
        power_db,
        shading="auto"
    )

    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(
        f"Kaiser-Windowed STFT Spectrogram - Label {label.item()}"
    )

    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.show()