import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PhosphoCSVDataset(Dataset):
    def __init__(self, csv_file_path):
        """
        Dataset class for loading features directly from a CSV file.
        
        Args:
            csv_file_path (str): Path to the CSV file containing features and labels.
                                 Assumes columns: 'uniprot_id', 'location', 'label', 
                                 and feature columns.
        """
        # Load the CSV into memory
        self.data_frame = pd.read_csv(csv_file_path)
        
        # 1. Identify Feature Columns
        # We assume "uniprot_id", "location", and "label" are NOT features.
        # All other columns are treated as input features for the MLP.
        self.ignore_cols = ['uniprot_id', 'location', 'label']
        self.feature_cols = [c for c in self.data_frame.columns if c not in self.ignore_cols]
        
        print(f"Loaded {len(self.data_frame)} samples.")
        print(f"Detected {len(self.feature_cols)} feature columns.")

        # 2. Pre-convert data to Tensors to speed up training
        # Convert feature columns to float32 tensor
        self.features = torch.tensor(
            self.data_frame[self.feature_cols].values, 
            dtype=torch.float32
        )
        
        # 3. Process Labels
        # If label is 'P'/'U', convert to 1/0. If already 1/0, keep as is.
        if self.data_frame['label'].dtype == object:
            # Map 'P' to 1.0, everything else (e.g., 'U') to 0.0
            labels_numeric = self.data_frame['label'].apply(lambda x: 1.0 if x == 'P' else 0.0).values
        else:
            labels_numeric = self.data_frame['label'].values.astype(float)
            
        self.labels = torch.tensor(labels_numeric, dtype=torch.float32)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Return the pre-loaded tensor directly (Very fast)
        return self.features[idx], self.labels[idx]

    def get_input_dim(self):
        """Returns the number of input features."""
        return len(self.feature_cols)