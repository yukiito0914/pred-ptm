import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CNN_attention import PhosphoCNNAttention
from src.dataset import PhosphoCSVDataset

def get_args():
    parser = argparse.ArgumentParser(description='Filter Reliable Negatives for PU Learning')
    # original training data
    parser.add_argument('--input_csv', type=str, default='data/train_data_w31.csv')
    # pre-trained model
    parser.add_argument('--model_path', type=str, default='experiments_final/best_model.pt')
    # output
    parser.add_argument('--output_csv', type=str, default='data/train_data_clean.csv')
    # threshold for true negatives
    parser.add_argument('--neg_threshold', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=2048)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Filtering Negatives on {device}...")

    # 1. Load Original Data
    print(f"Loading original data: {args.input_csv}")
    dataset = PhosphoCSVDataset(args.input_csv)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    input_dim = dataset.get_input_dim()
    
    # 2. Load Trained Model
    print(f"Loading model: {args.model_path}")
    model = PhosphoCNNAttention(input_dim=input_dim, output_dim=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 3. Predict Probabilities
    all_probs = []
    print("Predicting probabilities on training data...")
    
    with torch.no_grad():
        for inputs_feat, inputs_seq, _ in tqdm(dataloader):
            inputs_feat = inputs_feat.to(device)
            inputs_seq = inputs_seq.to(device)
            
            outputs = model(inputs_feat, inputs_seq)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy().flatten())
            
    # 4. Filter Data
    df = dataset.data_frame
    df['prob'] = all_probs
    
    print(f"Total samples before filtering: {len(df)}")
    
    # --- Selection Logic ---
    # Case A: Label is Positive (P) -> Always Keep
    # Case B: Label is Negative/Unlabeled (U) AND Probability <= Threshold -> Keep as Reliable Negative
    # Case C: Label is U AND Probability > Threshold -> Discard (Likely Hidden Positive)
    
    # Note: dataset.labels is already converted to 1.0 (P) and 0.0 (U) in the class,
    # but here we use the raw dataframe 'label' column which might be 'P'/'U' or 1/0.
    # Let's assume standardized 1/0 or P/U check.
    
    keep_indices = []
    discard_count = 0
    
    for idx, row in df.iterrows():
        # Check label (handle both string 'P' and numeric 1)
        is_positive = (row['label'] == 'P') or (row['label'] == 1.0)
        
        if is_positive:
            keep_indices.append(idx)
        else:
            # It's Unlabeled/Negative
            if row['prob'] <= args.neg_threshold:
                keep_indices.append(idx)
            else:
                discard_count += 1

    # Create new DataFrame
    df_clean = df.loc[keep_indices].drop(columns=['prob'])
    
    print(f"Discarded (Risky Negatives): {discard_count}")
    print(f"Remaining samples: {len(df_clean)}")
    
    # 5. Save
    df_clean.to_csv(args.output_csv, index=False)
    print(f"Cleaned dataset saved to: {args.output_csv}")
    print("Ready for Retraining!")

if __name__ == "__main__":
    main()