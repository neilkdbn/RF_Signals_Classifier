# dataset_agent.py
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetAgent:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.pkl_path = os.path.join(self.data_dir, "RML2016.10a_dict.pkl")
        
    def process_and_split(self, random_seed=42):
        """Loads, normalizes, splits, and caches the dataset."""
        # 1. Safety check to make sure the copy-paste worked
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(
                f"\n[ERROR] Could not find 'RML2016.10a_dict.pkl' in {self.data_dir}!\n"
                "Please make sure you copy-pasted your unpacked dataset file there."
            )
            
        print("1. Found unzipped dataset! Loading pickle file into memory...")
        with open(self.pkl_path, 'rb') as f:
            # Safely load the legacy Python 2 pickle file in Python 3
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            raw_data = u.load()
            
        X = []
        labels = []
        snrs = []
        
        # Unpack dictionary structure: { (modulation, SNR): [samples, 2, 128] }
        for key in raw_data.keys():
            mod_type, snr = key
            for sample in raw_data[key]:
                X.append(sample)
                labels.append(mod_type)
                snrs.append(snr)
                
        X = np.array(X, dtype=np.float32)       # Shape: (220000, 2, 128)
        labels = np.array(labels)
        snrs = np.array(snrs, dtype=np.int32)
        
        # Convert text labels to unique integers
        unique_mods = sorted(list(set(labels)))
        mod_to_idx = {mod: idx for idx, mod in enumerate(unique_mods)}
        y = np.array([mod_to_idx[l] for l in labels], dtype=np.int64)
        
        print(f"Modulation schemes found: {unique_mods}")
        print("2. Normalizing signals (Zero-mean & Unit-variance)...")
        
        # Fast normalization to prevent deep learning gradient explosion
        for i in range(len(X)):
            mean = np.mean(X[i], axis=1, keepdims=True)
            std = np.std(X[i], axis=1, keepdims=True) + 1e-6
            X[i] = (X[i] - mean) / std
            
        print("3. Creating locked 70:20:10 stratified train/val/test splits...")
        indices = np.arange(len(X))
        
        # Split off 10% for the final test set
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.10, random_state=random_seed, stratify=y
        )
        # Split remaining 90% into 70% Train and 20% Validation
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.222, random_state=random_seed, stratify=y[train_val_idx]
        )
        
        print("4. Saving split indices and normalized data arrays to disk...")
        np.save(os.path.join(self.data_dir, "train_idx.npy"), train_idx)
        np.save(os.path.join(self.data_dir, "val_idx.npy"), val_idx)
        np.save(os.path.join(self.data_dir, "test_idx.npy"), test_idx)
        np.save(os.path.join(self.data_dir, "X_all.npy"), X)
        np.save(os.path.join(self.data_dir, "y_all.npy"), y)
        np.save(os.path.join(self.data_dir, "snrs_all.npy"), snrs)
        
        print("\n=== SUCCESS: DATA PREPARATION COMPLETE ===")
        print(f"Total processed samples: {len(X)}")
        print(f"-> Training set size   : {len(train_idx)} samples")
        print(f"-> Validation set size : {len(val_idx)} samples")
        print(f"-> Testing set size    : {len(test_idx)} samples")
        print("==========================================\n")

if __name__ == "__main__":
    agent = DatasetAgent()
    agent.process_and_split()
