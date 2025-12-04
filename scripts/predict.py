import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CNN_attention import PhosphoCNNAttention

# --- Inference Dataset Class ---
class InferenceDataset(Dataset):
    def __init__(self, csv_file_path):
        """
        Dataset for inference. 
        Differs from PhosphoCSVDataset as it doesn't require a 'label' column.
        """
        self.data_frame = pd.read_csv(csv_file_path)
        
        # 1. Identify Feature Columns (feat_0 ... feat_92)
        self.feature_cols = [c for c in self.data_frame.columns if c.startswith('feat_')]
        
        # 2. Identify Sequence Columns (seq_0 ... seq_30)
        self.seq_cols = [c for c in self.data_frame.columns if c.startswith('seq_')]
        
        # Pre-load tensors
        self.features = torch.tensor(self.data_frame[self.feature_cols].values, dtype=torch.float32)
        self.sequences = torch.tensor(self.data_frame[self.seq_cols].values, dtype=torch.long)
        
        # Store metadata for saving results
        self.meta_cols = ['uniprot_id', 'location']
        # If label exists, keep it for reference, otherwise ignore
        if 'label' in self.data_frame.columns:
            self.meta_cols.append('label')
            
        self.metadata = self.data_frame[self.meta_cols].copy()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Returns only features and sequence
        return self.features[idx], self.sequences[idx]

    def get_input_dim(self):
        return len(self.feature_cols)

def get_args():
    parser = argparse.ArgumentParser(description='Run Inference using Trained Model')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV (Window 31 format)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--output_csv', type=str, default='predictions.csv', help='Path to save results')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold (Use the best one from training)')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on {device}...")

    # 1. Load Data
    print(f"Loading data from {args.input_csv}...")
    dataset = InferenceDataset(args.input_csv)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    input_dim = dataset.get_input_dim()
    print(f"Input Feature Dimension: {input_dim}")
    print(f"Total samples: {len(dataset)}")

    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    model = PhosphoCNNAttention(input_dim=input_dim, output_dim=1).to(device)
    
    # Load weights
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Inference Loop
    all_probs = []
    
    print("Starting prediction...")
    with torch.no_grad():
        for inputs_feat, inputs_seq in tqdm(dataloader):
            inputs_feat = inputs_feat.to(device)
            inputs_seq = inputs_seq.to(device)
            
            # Forward pass
            outputs = model(inputs_feat, inputs_seq)
            
            # Convert logits to probability (Sigmoid)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy().flatten())

    # 4. Save Results
    print("Saving results...")
    results_df = dataset.metadata.copy()
    results_df['probability'] = all_probs
    results_df['prediction'] = (np.array(all_probs) > args.threshold).astype(int) # 0 or 1
    
    # Save to CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Done! Predictions saved to {args.output_csv}")
    
    # Optional: Print some stats
    num_pos = results_df['prediction'].sum()
    print(f"Predicted Positive Sites: {num_pos} / {len(results_df)} ({num_pos/len(results_df)*100:.2f}%)")

if __name__ == "__main__":
    main()