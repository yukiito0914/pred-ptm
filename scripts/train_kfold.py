import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.MLP import PhosphoMLP
# from src.CNN import PhosphoCNN
from src.CNN_attention import PhosphoCNNAttention
from src.dataset import PhosphoCSVDataset
from src.utils import EarlyStopping

def get_args():
    parser = argparse.ArgumentParser(description='Train MLP with K-Fold CV (Embedding)')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--output_dir', type=str, default='results', help='Output dir')
    return parser.parse_args()

def calculate_pos_weight(dataset_subset):
    indices = dataset_subset.indices
    all_labels = dataset_subset.dataset.labels
    subset_labels = all_labels[indices]
    num_pos = subset_labels.sum().item()
    num_neg = len(subset_labels) - num_pos
    if num_pos == 0: return torch.tensor(1.0)
    return torch.tensor(num_neg / num_pos)

def find_optimal_threshold(targets, probs):
    best_th = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.1, 0.96, 0.01)
    
    for th in thresholds:
        preds = (probs > th).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1

def validate(model, loader, device, criterion):
    """
    [UPDATED] Handles 3-item tuple from dataset (features, sequences, targets)
    """
    model.eval()
    val_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        # [CHANGE 1] Unpack 3 items
        for inputs_feat, inputs_seq, targets in loader:
            # [CHANGE 2] Move all to device
            inputs_feat = inputs_feat.to(device)
            inputs_seq = inputs_seq.to(device)
            targets = targets.to(device).view(-1, 1)
            
            # [CHANGE 3] Pass 2 inputs to model
            outputs = model(inputs_feat, inputs_seq)
            
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = val_loss / len(loader)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    preds_def = (all_probs > 0.5).astype(int)
    rec_def = recall_score(all_targets, preds_def, zero_division=0)
    
    return avg_loss, rec_def, all_targets, all_probs

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running K-Fold CV on {device}...")

    # Load dataset (make sure it's the one with 'seq_X' columns!)
    full_dataset = PhosphoCSVDataset(csv_file_path=args.csv_path)
    input_dim = full_dataset.get_input_dim() # Should be 45
    all_labels = full_dataset.labels.numpy()

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n{'='*20} Fold {fold+1}/{args.k_folds} {'='*20}")
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        
        # model = PhosphoMLP(input_dim=input_dim, output_dim=1).to(device)
        # model = PhosphoCNN(input_dim=input_dim, output_dim=1).to(device)
        model = PhosphoCNNAttention(input_dim=input_dim, output_dim=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        pos_weight = calculate_pos_weight(train_subset).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        fold_save_path = os.path.join(args.output_dir, f'model_fold{fold+1}.pt')
        early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=fold_save_path)

        best_fold_metrics = {}

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            
            # [CHANGE 1] Unpack 3 items
            for inputs_feat, inputs_seq, targets in train_loader:
                # [CHANGE 2] Move all to device
                inputs_feat = inputs_feat.to(device)
                inputs_seq = inputs_seq.to(device)
                targets = targets.to(device).view(-1, 1)
                
                optimizer.zero_grad()
                
                # [CHANGE 3] Pass 2 inputs to model
                outputs = model(inputs_feat, inputs_seq)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validate
            val_loss, val_rec_def, val_targets, val_probs = validate(model, val_loader, device, criterion)
            
            print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Recall(0.5): {val_rec_def:.4f}")

            early_stopping(val_loss, model)
            
            if val_loss == early_stopping.val_loss_min:
                best_th, best_f1 = find_optimal_threshold(val_targets, val_probs)
                preds_opt = (val_probs > best_th).astype(int)
                
                acc = accuracy_score(val_targets, preds_opt)
                prec = precision_score(val_targets, preds_opt, zero_division=0)
                rec = recall_score(val_targets, preds_opt, zero_division=0)
                mcc = matthews_corrcoef(val_targets, preds_opt)
                
                best_fold_metrics = {
                    'fold': fold + 1,
                    'threshold': best_th,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': best_f1,
                    'mcc': mcc
                }

            if early_stopping.early_stop:
                print("Early stopping.")
                break
        
        print(f"Fold {fold+1} Best Result (Threshold {best_fold_metrics['threshold']:.2f}):")
        print(f"  ACC : {best_fold_metrics['accuracy']:.4f}")
        print(f"  PREC: {best_fold_metrics['precision']:.4f}")
        print(f"  REC : {best_fold_metrics['recall']:.4f}")
        print(f"  F1  : {best_fold_metrics['f1']:.4f}")
        print(f"  MCC : {best_fold_metrics['mcc']:.4f}")
        
        fold_results.append(best_fold_metrics)

    print(f"\n{'='*20} K-Fold Summary (Average) {'='*20}")
    print(f"Avg Threshold : {np.mean([r['threshold'] for r in fold_results]):.2f}")
    print(f"Avg Accuracy  : {np.mean([r['accuracy'] for r in fold_results]):.4f}")
    print(f"Avg Precision : {np.mean([r['precision'] for r in fold_results]):.4f}")
    print(f"Avg Recall    : {np.mean([r['recall'] for r in fold_results]):.4f}")
    print(f"Avg F1 Score  : {np.mean([r['f1'] for r in fold_results]):.4f}")
    print(f"Avg MCC       : {np.mean([r['mcc'] for r in fold_results]):.4f}")

if __name__ == "__main__":
    main()