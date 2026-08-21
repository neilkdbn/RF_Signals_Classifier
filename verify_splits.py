import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def verify_and_plot_dataset(data_dir="./data", output_png="verification_results.png"):
    print("--- 1. Verifying Processed Dataset Files ---")
    files = ["X_all.npy", "y_all.npy", "snrs_all.npy", "train_idx.npy", "val_idx.npy", "test_idx.npy"]
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[FAIL] Missing {f} in {data_dir}")
        arr = np.load(path)
        print(f"[OK] {f:15s} shape: {str(arr.shape):20s} dtype: {arr.dtype}")

    X = np.load(os.path.join(data_dir, "X_all.npy"))
    y = np.load(os.path.join(data_dir, "y_all.npy"))
    snrs = np.load(os.path.join(data_dir, "snrs_all.npy"))
    train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, "test_idx.npy"))

    # Verify no index overlaps
    all_indices = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_indices) == len(set(all_indices)), "Found overlapping indices between splits!"
    assert len(all_indices) == len(X), "Total split indices do not match total samples!"

    print(f"\nNo index overlap detected across Train ({len(train_idx)}), Val ({len(val_idx)}), and Test ({len(test_idx)}) splits.")
    print("Zero-mean & Unit-variance sample check:")
    print(f"  Sample 0 mean: {np.mean(X[0]):.6f} (expected ~0.0)")
    print(f"  Sample 0 std : {np.std(X[0]):.6f} (expected ~1.0)")

    print("\n--- 2. Generating Comprehensive Visualizations (PNG) ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    mod_names = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

    # 1. Sample Time-Domain Waveform (I & Q channels)
    ax1 = fig.add_subplot(gs[0, 0])
    sample_idx = 0
    t = np.arange(X.shape[2])
    ax1.plot(t, X[sample_idx, 0, :], label='In-Phase (I)', color='#1f77b4', lw=1.8)
    ax1.plot(t, X[sample_idx, 1, :], label='Quadrature (Q)', color='#ff7f0e', lw=1.8, alpha=0.85)
    mod_idx = y[sample_idx]
    mod_label = mod_names[mod_idx] if mod_idx < len(mod_names) else f"Mod {mod_idx}"
    ax1.set_title(f"Time-Domain Waveform\n({mod_label} @ {snrs[sample_idx]} dB)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time Sample (0-127)")
    ax1.set_ylabel("Normalized Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. I/Q Constellation Diagram (High SNR sample)
    ax2 = fig.add_subplot(gs[0, 1])
    high_snr_mask = (snrs == 18) & (y == mod_idx)
    high_snr_indices = np.where(high_snr_mask)[0]
    if len(high_snr_indices) > 0:
        c_idx = high_snr_indices[0]
        i_vals = X[c_idx, 0, :]
        q_vals = X[c_idx, 1, :]
        ax2.scatter(i_vals, q_vals, c='#2ca02c', alpha=0.7, edgecolors='k', s=35)
        ax2.set_title(f"I/Q Constellation Diagram\n({mod_label} @ 18 dB SNR)", fontsize=12, fontweight='bold')
    else:
        ax2.scatter(X[sample_idx, 0, :], X[sample_idx, 1, :], c='#2ca02c', alpha=0.7, edgecolors='k', s=35)
        ax2.set_title(f"I/Q Constellation Diagram\n({mod_label})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("In-Phase (I)")
    ax2.set_ylabel("Quadrature (Q)")
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. Spectrogram (STFT with Kaiser Window)
    ax3 = fig.add_subplot(gs[0, 2])
    complex_sig = X[sample_idx, 0, :] + 1j * X[sample_idx, 1, :]
    f, t_spec, Zxx = signal.stft(complex_sig, nperseg=32, noverlap=24, window=('kaiser', 5.0), return_onesided=False)
    spec_mag = np.abs(np.fft.fftshift(Zxx, axes=0))
    im = ax3.pcolormesh(t_spec, np.fft.fftshift(f), 20 * np.log10(spec_mag + 1e-6), shading='gouraud', cmap='viridis')
    ax3.set_title(f"Kaiser-Windowed STFT Spectrogram\n({mod_label})", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time Segment")
    ax3.set_ylabel("Normalized Frequency")
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label("Magnitude (dB)")

    # 4. Class Distribution Across Splits
    ax4 = fig.add_subplot(gs[1, 0:2])
    num_classes = len(np.unique(y))
    class_counts_train = np.bincount(y[train_idx], minlength=num_classes)
    class_counts_val = np.bincount(y[val_idx], minlength=num_classes)
    class_counts_test = np.bincount(y[test_idx], minlength=num_classes)

    x_bars = np.arange(num_classes)
    width = 0.25
    ax4.bar(x_bars - width, class_counts_train, width=width, label=f'Train ({len(train_idx)})', color='#1f77b4')
    ax4.bar(x_bars, class_counts_val, width=width, label=f'Val ({len(val_idx)})', color='#ff7f0e')
    ax4.bar(x_bars + width, class_counts_test, width=width, label=f'Test ({len(test_idx)})', color='#2ca02c')
    ax4.set_title("Stratified Class Distribution Across Dataset Splits", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Modulation Scheme")
    ax4.set_ylabel("Sample Count")
    display_names = [mod_names[i] if i < len(mod_names) else str(i) for i in range(num_classes)]
    ax4.set_xticks(x_bars)
    ax4.set_xticklabels(display_names, rotation=30, ha='right')
    ax4.legend(loc='lower right')
    ax4.grid(True, linestyle='--', alpha=0.6)

    # 5. SNR Distribution Across Splits
    ax5 = fig.add_subplot(gs[1, 2])
    unique_snrs = np.unique(snrs)
    snr_counts_train = [np.sum(snrs[train_idx] == s) for s in unique_snrs]
    snr_counts_val = [np.sum(snrs[val_idx] == s) for s in unique_snrs]
    snr_counts_test = [np.sum(snrs[test_idx] == s) for s in unique_snrs]

    ax5.plot(unique_snrs, snr_counts_train, marker='o', label='Train', color='#1f77b4', lw=2)
    ax5.plot(unique_snrs, snr_counts_val, marker='s', label='Val', color='#ff7f0e', lw=2)
    ax5.plot(unique_snrs, snr_counts_test, marker='^', label='Test', color='#2ca02c', lw=2)
    ax5.set_title("SNR Distribution Across Splits", fontsize=12, fontweight='bold')
    ax5.set_xlabel("SNR (dB)")
    ax5.set_ylabel("Sample Count")
    ax5.legend(loc='center right')
    ax5.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("RF Signal Classifier — Dataset Verification & Split Analytics", fontsize=15, fontweight='bold')
    
    png_path = os.path.join(".", output_png)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SUCCESS] Verification plot successfully saved to: {os.path.abspath(png_path)}")
    print("--- Verification PASSED ---")
    return True

if __name__ == "__main__":
    verify_and_plot_dataset()
