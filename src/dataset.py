import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PhosphoCSVDataset(Dataset):
    def __init__(self, csv_file_path):
        self.data_frame = pd.read_csv(csv_file_path)
        
        # 1. Identify "Original" Feature Columns (feat_0 ... feat_44)
        self.feature_cols = [c for c in self.data_frame.columns if c.startswith('feat_')]
        
        # 2. Identify "Sequence" Columns (seq_0 ... seq_14) [NEW]
        self.seq_cols = [c for c in self.data_frame.columns if c.startswith('seq_')]
        
        # Pre-load tensors
        self.features = torch.tensor(self.data_frame[self.feature_cols].values, dtype=torch.float32)
        self.sequences = torch.tensor(self.data_frame[self.seq_cols].values, dtype=torch.long) # [NEW] Integers
        
        # Labels
        if self.data_frame['label'].dtype == object:
            labels_numeric = self.data_frame['label'].apply(lambda x: 1.0 if x == 'P' else 0.0).values
        else:
            labels_numeric = self.data_frame['label'].values.astype(float)
        self.labels = torch.tensor(labels_numeric, dtype=torch.float32)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Return 3 items now: Features, Sequence, Label
        return self.features[idx], self.sequences[idx], self.labels[idx]

    def get_input_dim(self):
        return len(self.feature_cols)