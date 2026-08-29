import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn as nn

# Local imports
from models_2d import ResNet18_2D
from dataset_loader import get_spectrogram_dataloader

def build_dashboard():
    print("=" * 60)
    print("      BUILDING 2D VS 1D COMPARISON DASHBOARD")
    print("=" * 60)

    # 1. Setup paths and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running inference on: {device}")
    
    checkpoint_path = './checkpoints/best_resnet18.pt'
    data_dir = './data'
    out_dir = './dashboard_visuals'
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")

    # 2. Load Checkpoint and initialize model
    print("[*] Loading best 2D ResNet checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    in_channels = checkpoint.get('in_channels', 3) # default to hybrid if not found
    
    # Check classes list
    classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    num_classes = len(classes)

    model = ResNet18_2D(num_classes=num_classes, in_channels=in_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 3. Load Test Data
    print("[*] Spinning up the Memory-Mapped DataLoader...")
    # NOTE: Since precompute_stft saved 3-channel, in_channels=3 is assumed by default for grayscale config handling,
    # but the Colab trained model actually used in_channels=1 (grayscale) if they didn't change it. 
    # Let's ensure it matches the checkpoint shape:
    actual_in_ch = list(checkpoint['model_state_dict'].values())[0].shape[1]
    
    test_loader = get_spectrogram_dataloader(data_dir=data_dir, split='test', batch_size=256, in_channels=actual_in_ch, num_workers=0)

    # 4. Run Inference
    print("[*] Running inference on the test set... (this takes ~30 seconds)")
    y_true = []
    y_pred = []
    y_snr = []

    te_corr, te_total = 0, 0
    snr_correct = {}
    snr_total = {}

    with torch.no_grad():
        for signals, labels, snrs in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            out = model(signals)
            pred = out.argmax(1)
            
            # Store for confusion matrix
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_snr.extend(snrs.cpu().numpy())
            
            # Metrics
            te_corr += pred.eq(labels).sum().item()
            te_total += labels.size(0)
            
            for i in range(labels.size(0)):
                s = int(snrs[i].item())
                snr_correct.setdefault(s, 0)
                snr_total.setdefault(s, 0)
                snr_total[s] += 1
                if pred[i] == labels[i]:
                    snr_correct[s] += 1

    overall_2d_acc = 100. * te_corr / te_total
    print(f"\n[+] Inference Complete. Overall 2D Accuracy: {overall_2d_acc:.2f}%")

    # =========================================================================
    # PLOT 1: Confusion Matrix
    # =========================================================================
    print(f"[*] Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f"2D ResNet-18 Confusion Matrix (Overall Acc: {overall_2d_acc:.2f}%)")
    plt.ylabel('True Modulation')
    plt.xlabel('Predicted Modulation')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix_2d.png'), dpi=300)
    plt.close()

    # =========================================================================
    # PLOT 2: Accuracy vs SNR (with 1D Baseline)
    # =========================================================================
    print(f"[*] Generating Accuracy vs SNR curve...")
    sorted_snrs = sorted(list(snr_correct.keys()))
    accs_2d = [100. * snr_correct[s] / snr_total[s] for s in sorted_snrs]
    
    # Academic Baseline approx for 1D (assuming 46% avg translates to roughly ~70% high SNR)
    # We will just plot a flat baseline reference line for the 46% average comparison
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_snrs, accs_2d, marker='o', linewidth=2, color='#1f77b4', label='2D ResNet-18 (STFT)')
    plt.axhline(y=overall_2d_acc, color='#1f77b4', linestyle='--', alpha=0.5, label=f'2D Avg ({overall_2d_acc:.2f}%)')
    plt.axhline(y=46.00, color='#d62728', linestyle='--', linewidth=2, label='1D CNN Avg Baseline (46.00%)')
    
    plt.title('Modulation Classification Accuracy vs SNR')
    plt.xlabel('Signal-to-Noise Ratio (dB)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_vs_snr.png'), dpi=300)
    plt.close()

    # =========================================================================
    # PLOT 3: 1D vs 2D Final Comparison Bar Chart
    # =========================================================================
    print(f"[*] Generating Final Comparison Bar Chart...")
    plt.figure(figsize=(8, 6))
    models = ['1D CNN (Raw IQ)', '2D ResNet-18 (STFT)']
    accuracies = [46.00, overall_2d_acc]
    
    colors = ['#d62728', '#2ca02c'] if overall_2d_acc > 46.00 else ['#2ca02c', '#d62728']
    
    bars = plt.bar(models, accuracies, color=colors, width=0.5)
    
    # Add percentage text on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('Final Model Performance Comparison', fontsize=14)
    plt.ylabel('Overall Accuracy (%)')
    plt.ylim([0, 100])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1d_vs_2d_comparison.png'), dpi=300)
    plt.close()

    print(f"\n[SUCCESS] Dashboard generated in {out_dir}/")
    for f in os.listdir(out_dir):
        print(f"  -> {f}")

if __name__ == '__main__':
    build_dashboard()
