# subsystem2_1d/dataloader_1d.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Callable

class RFSignalDataset1D(Dataset):
    """
    PyTorch Dataset for 1D IQ signal classification.
    Expects X of shape (N, 2, 128) and y of shape (N,).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, transform: Optional[Callable] = None):
        if X.ndim != 3 or X.shape[1] != 2 or X.shape[2] != 128:
            raise ValueError(f"Expected X of shape (N, 2, 128), got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: len(X)={len(X)} != len(y)={len(y)}")

        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.X)
    
    @property
    def num_classes(self) -> int:
        if len(self.y) == 0:
            return 11
        return int(np.max(self.y)) + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal_tensor = torch.from_numpy(self.X[idx])
        if self.transform is not None:
            signal_tensor = self.transform(signal_tensor)
        label_tensor = torch.tensor(self.y[idx], dtype=torch.int64)
        return signal_tensor, label_tensor

def get_1d_dataloaders(
    data_dir: str = "data",
    batch_size: int = 1024,
    num_workers: int = 2,
    train_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to load the master arrays and split indices,
    slice them accordingly, and return the three PyTorch DataLoaders.
    """
    X_all_path = os.path.join(data_dir, "X_all.npy")
    y_all_path = os.path.join(data_dir, "y_all.npy")
    
    if not os.path.exists(X_all_path) or not os.path.exists(y_all_path):
        raise FileNotFoundError(f"Missing master data files in {data_dir}")

    X_all = np.load(X_all_path)
    y_all = np.load(y_all_path)
    
    train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, "test_idx.npy"))
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    train_ds = RFSignalDataset1D(X_train, y_train, transform=train_transform)
    val_ds = RFSignalDataset1D(X_val, y_val)
    test_ds = RFSignalDataset1D(X_test, y_test)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
                                 
    return train_loader, val_loader, test_loader
