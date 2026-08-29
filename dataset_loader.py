# dataset_loader.py
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RFSignalDataset(Dataset):
    """
    Standardised PyTorch Dataset to load the RadioML 2016.10a signal splits.
    Guarantees that both the 1D (raw IQ) and 2D (spectrogram) teams train and 
    evaluate on identical, non-overlapping splits.
    """
    def __init__(self, data_dir="./data", split="train", transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing processed .npy files.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample (e.g., STFT for 2D).
        """
        # Auto-detect data directory if default is used
        if not os.path.exists(data_dir):
            alt_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "RF_Signals_Classifier", "data"),
                os.path.join(".", "RF_Signals_Classifier", "data"),
                os.path.join(".", "data"),
                os.path.join("..", "data")
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    data_dir = alt
                    break

        self.data_dir = data_dir
        self.split = split.lower()
        self.transform = transform
        
        # Verify that preprocessed files exist
        required_files = ["X_all.npy", "y_all.npy", "snrs_all.npy", "train_idx.npy", "val_idx.npy", "test_idx.npy"]
        for f in required_files:
            file_path = os.path.join(self.data_dir, f)
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Could not find '{f}' in '{self.data_dir}'. "
                    "Please ensure you run 'python dataset_agent.py' first!"
                )
        
        # Load preprocessed arrays
        self.X = np.load(os.path.join(self.data_dir, "X_all.npy"))  # Shape: (220000, 2, 128)
        self.y = np.load(os.path.join(self.data_dir, "y_all.npy"))  # Shape: (220000,)
        self.snrs = np.load(os.path.join(self.data_dir, "snrs_all.npy"))  # Shape: (220000,)
        
        # Select the locked index split
        if self.split == "train":
            self.indices = np.load(os.path.join(self.data_dir, "train_idx.npy"))
        elif self.split == "val":
            self.indices = np.load(os.path.join(self.data_dir, "val_idx.npy"))
        elif self.split == "test":
            self.indices = np.load(os.path.join(self.data_dir, "test_idx.npy"))
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Resolve the actual index in the parent array
        real_idx = self.indices[idx]
        
        # Extract the signal (2, 128), label, and SNR
        signal = self.X[real_idx]
        label = self.y[real_idx]
        snr = self.snrs[real_idx]
        
        # Convert to PyTorch tensors
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Apply transform if provided (e.g., Team B converting 1D IQ to 2D Spectrograms on-the-fly)
        if self.transform:
            signal_tensor = self.transform(signal_tensor)
            
        return signal_tensor, label_tensor, snr


def get_rf_dataloader(data_dir="./data", split="train", batch_size=64, shuffle=None, num_workers=0, transform=None):
    """
    Utility function to construct a PyTorch DataLoader.
    
    Args:
        data_dir (str): Path to processed data.
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Size of batches.
        shuffle (bool): Whether to shuffle. Defaults to True for train, False for val/test.
        num_workers (int): Number of subprocesses for data loading.
        transform (callable, optional): Transform to apply to data samples.
    """
    if shuffle is None:
        shuffle = True if split == "train" else False
        
    dataset = RFSignalDataset(data_dir=data_dir, split=split, transform=transform)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader


# ==========================================
# Self-Verification Test
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("      RF SIGNAL CLASSIFICATION: LOADER TEST       ")
    print("==================================================")
    try:
        # Load training split
        print("\n1. Initializing training dataset...")
        train_loader = get_rf_dataloader(split="train", batch_size=64)
        print(f"[PASS] Dataset initialized. Total batches: {len(train_loader)}")
        
        # Load validation split
        print("\n2. Initializing validation dataset...")
        val_loader = get_rf_dataloader(split="val", batch_size=128)
        print(f"[PASS] Dataset initialized. Total batches: {len(val_loader)}")
        
        # Load test split
        print("\n3. Initializing test dataset...")
        test_loader = get_rf_dataloader(split="test", batch_size=128)
        print(f"[PASS] Dataset initialized. Total batches: {len(test_loader)}")
        
        # Pull a single batch to check shapes
        print("\n4. Verification test: Pulling a batch of data...")
        for signals, labels, snrs in train_loader:
            print("\n=== SUCCESS: PYTORCH LOADER VERIFIED ===")
            print(f"Loaded signal batch shape : {signals.shape} (Expected: [64, 2, 128])")
            print(f"Loaded label batch shape  : {labels.shape} (Expected: [64])")
            print(f"First 5 labels in batch   : {labels[:5].numpy()}")
            print(f"First 5 SNRs in batch     : {snrs[:5].numpy()} dB")
            print("=========================================\n")
            break
            
    except Exception as e:
        print(f"\n[FAIL] PyTorch loader verification failed: {e}")
        print("Please ensure you run 'python dataset_agent.py' first to prepare the arrays.")


# ==========================================
# 2D Spectrogram Dataset (pre-computed STFT)
# ==========================================

class SpectrogramDataset(Dataset):
    """
    Fast 2D Dataset that loads pre-computed STFT spectrograms from
    ./data/spectrograms_all.npy via memory-mapping.

    Eliminates the on-the-fly scipy.signal.stft() bottleneck entirely.
    The full (220000, 3, 64, 5) array is memory-mapped — only the pages
    actually requested by the DataLoader are paged into RAM.

    Args:
        data_dir   : directory containing spectrograms_all.npy + index files.
        split      : 'train', 'val', or 'test'.
        in_channels: 1 → grayscale (power_db only), 3 → hybrid (all channels).

    Run once before training:
        python precompute_stft.py
    """

    SPECTROGRAM_FILE = "spectrograms_all.npy"

    def __init__(self, data_dir: str = "./data", split: str = "train",
                 in_channels: int = 1):
        self.in_channels = in_channels
        split = split.lower()

        # ── Resolve data_dir ────────────────────────────────────────────
        if not os.path.exists(data_dir):
            for alt in [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                os.path.join(".", "data"),
                os.path.join("..", "data"),
            ]:
                if os.path.exists(alt):
                    data_dir = alt
                    break

        # ── Validate pre-computed file ───────────────────────────────────
        spec_path = os.path.join(data_dir, self.SPECTROGRAM_FILE)
        if not os.path.exists(spec_path):
            raise FileNotFoundError(
                f"Pre-computed spectrogram file not found:\n  {spec_path}\n"
                "Run first:  python precompute_stft.py"
            )

        # ── Memory-map the full spectrogram array (no RAM load) ──────────
        # mmap_mode='r' → OS pages only the accessed rows into RAM.
        self._spectrograms = np.load(spec_path, mmap_mode="r")  # (N, 3, H, W)

        # ── Load labels & SNRs (tiny, load fully) ────────────────────────
        self._labels = np.load(os.path.join(data_dir, "y_all.npy"))    # (N,)
        self._snrs   = np.load(os.path.join(data_dir, "snrs_all.npy")) # (N,)

        # ── Load split indices ───────────────────────────────────────────
        idx_map = {"train": "train_idx.npy", "val": "val_idx.npy", "test": "test_idx.npy"}
        if split not in idx_map:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        self._indices = np.load(os.path.join(data_dir, idx_map[split]))
        
        self.split_name = split
        self.transform = None

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        real_idx = self._indices[idx]

        # np.array() materialises one row from the mmap into a regular array
        spec = np.array(self._spectrograms[real_idx], dtype=np.float32)  # (3, H, W)

        if self.in_channels == 1:
            spec = spec[:1]   # keep only power_db channel → (1, H, W)

        signal_tensor = torch.from_numpy(spec)
        if self.transform is not None:
            signal_tensor = self.transform(signal_tensor)
            
        label_tensor  = torch.tensor(int(self._labels[real_idx]),  dtype=torch.long)
        snr           = self._snrs[real_idx]

        return signal_tensor, label_tensor, snr


def get_spectrogram_dataloader(
    data_dir: str    = "./data",
    split: str       = "train",
    batch_size: int  = 64,
    in_channels: int = 1,
    shuffle: bool    = None,
    num_workers: int = None,
) -> DataLoader:
    """
    Construct a fast DataLoader backed by SpectrogramDataset.

    num_workers defaults to min(4, cpu_count) for maximum throughput.
    persistent_workers=True avoids worker-respawn overhead between epochs.
    pin_memory=True (when CUDA available) accelerates host→GPU transfers.
    """
    if shuffle is None:
        shuffle = (split == "train")
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    dataset = SpectrogramDataset(
        data_dir=data_dir, split=split, in_channels=in_channels
    )

    use_pin_memory = torch.cuda.is_available()
    use_persistent = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
    )
    return loader
