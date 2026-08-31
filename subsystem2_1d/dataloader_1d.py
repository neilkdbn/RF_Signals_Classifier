# subsystem2_1d/dataloader_1d.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RFSignalDataset1D(Dataset):
    """
    PyTorch Dataset for 1D IQ signal classification.
    Expects X of shape (N, 2, 128) and y of shape (N,).
    """
    def __init__(self, X, y):
        # Ensure correct shapes and types
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Convert the NumPy slices to PyTorch tensors on the fly
        signal_tensor = torch.from_numpy(self.X[idx])
        label_tensor = torch.tensor(self.y[idx], dtype=torch.int64)
        return signal_tensor, label_tensor

def get_1d_dataloaders(data_dir="../data", batch_size=1024, num_workers=2):
    """
    Factory function to load the master arrays and split indices,
    slice them accordingly, and return the three PyTorch DataLoaders.
    """
    # 1. Load the master arrays
    X_all_path = os.path.join(data_dir, "X_all.npy")
    y_all_path = os.path.join(data_dir, "y_all.npy")
    
    print(f"Loading master dataset arrays from {data_dir}...")
    X_all = np.load(X_all_path)
    y_all = np.load(y_all_path)
    
    # 2. Load the split contract indices
    train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, "test_idx.npy"))
    
    # 3. Slice the master arrays using the locked indices
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    # 4. Instantiate Datasets
    train_ds = RFSignalDataset1D(X_train, y_train)
    val_ds = RFSignalDataset1D(X_val, y_val)
    test_ds = RFSignalDataset1D(X_test, y_test)
    
    # 5. Build and return DataLoaders
    # Note: Only shuffle the training loader!
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
                             
    return train_loader, val_loader, test_loader
