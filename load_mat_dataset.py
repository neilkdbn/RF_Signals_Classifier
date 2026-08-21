# load_mat_dataset.py
import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_matlab_signals(mat_path="custom_rf_dataset.mat"):
    # Auto-detect mat_path across relative and absolute directory locations
    if not os.path.exists(mat_path):
        alt_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), mat_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", mat_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "RF_Signals_Classifier", mat_path),
            os.path.join(".", mat_path),
            os.path.join("..", mat_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", mat_path),
            os.path.join("D:\\College\\RF_Classifier", mat_path),
            os.path.join("D:\\College\\RF_Classifier\\RF_Signals_Classifier", mat_path)
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                mat_path = alt
                break

    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"Could not find '{mat_path}'.\n"
            "Make sure 'custom_rf_dataset.mat' exists in the project directory."
        )
        
    print("==================================================")
    print("      LOADING CUSTOM MATLAB DATASET INTO PYTORCH  ")
    print("==================================================")
    print(f"Reading from: {os.path.abspath(mat_path)}")
    
    # Load the .mat file using scipy
    mat_data = sio.loadmat(mat_path)
    
    # Extract raw numpy arrays
    X = mat_data['X_custom'].astype(np.float32)       # Shape: (5000, 2, 128)
    y = mat_data['y_custom'].squeeze().astype(np.int64) # Shape: (5000,)
    
    # Convert cell array of strings from MATLAB to a clean Python list
    raw_mods = mat_data['modulations']
    modulations = [str(m[0][0]) if isinstance(m[0], (list, np.ndarray)) else str(m[0]) for m in raw_mods]
    
    print("\n[PASS] MATLAB .mat file loaded perfectly!")
    print(f"-> Signals array shape: {X.shape} (Expected: 5000, 2, 128)")
    print(f"-> Labels array shape : {y.shape} (Expected: 5000)")
    print(f"-> Signal Classes     : {modulations}")
    print("==================================================\n")
    
    # Convert arrays to PyTorch Tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    
    return TensorDataset(X_tensor, y_tensor), modulations

if __name__ == "__main__":
    try:
        # Load data
        dataset, modulations = load_matlab_signals()
        
        # Initialize a PyTorch DataLoader with batch size 32
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Test pull a batch
        for signals, labels in loader:
            print("[PASS] PyTorch DataLoader initialized successfully!")
            print(f"-> Batch Signals Shape: {signals.shape} (Expected: 32, 2, 128)")
            print(f"-> Batch Labels Shape : {labels.shape} (Expected: 32)")
            print("\n=== SUCCESS: END-TO-END DATA PIPELINE IS 100% OPERATIONAL ===")
            break
            
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
