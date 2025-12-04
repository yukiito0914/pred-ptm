import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CNN_attention import PhosphoCNNAttention
from src.dataset import PhosphoCSVDataset
from src.utils import EarlyStopping, plot_loss_curves

def get_args():
    parser = argparse.ArgumentParser(description='Train Final Model (CNN + Attention)')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='experiments_final', help='Output directory')
    return parser.parse_args()

def calculate_pos_weight(dataset):
    labels = dataset.labels
    num_pos = labels.sum().item()
    num_neg = len(labels) - num_pos
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
    model.eval()
    val_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs_feat, inputs_seq, targets in loader:
            inputs_feat = inputs_feat.to(device)
            inputs_seq = inputs_seq.to(device)
            targets = targets.to(device).view(-1, 1)
            
            outputs = model(inputs_feat, inputs_seq)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = val_loss / len(loader)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Calculate default metrics (Threshold 0.5) for monitoring
    preds_def = (all_probs > 0.5).astype(int)
    rec_def = recall_score(all_targets, preds_def, zero_division=0)
    
    return avg_loss, rec_def, all_targets, all_probs

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Final Training on {device}...")

    # 1. Load Dataset
    full_dataset = PhosphoCSVDataset(csv_file_path=args.csv_path)
    input_dim = full_dataset.get_input_dim() # Should be 93 for Window 31
    print(f"Input Feature Dimension: {input_dim}")

    # 2. Split Train/Val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Setup Model
    model = PhosphoCNNAttention(input_dim=input_dim, output_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # We calculate pos_weight based on the FULL dataset for simplicity, 
    # or you can calculate it based on train_dataset only if you iterate through it.
    pos_weight = calculate_pos_weight(full_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    model_save_path = os.path.join(args.output_dir, 'best_model.pt')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=model_save_path)

    # 4. Training Loop
    train_losses = []
    val_losses = []
    
    best_metrics = {}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for inputs_feat, inputs_seq, targets in train_loader:
            inputs_feat = inputs_feat.to(device)
            inputs_seq = inputs_seq.to(device)
            targets = targets.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs_feat, inputs_seq)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss, val_rec_def, val_targets, val_probs = validate(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Recall(0.5): {val_rec_def:.4f}")

        # Check Early Stopping
        early_stopping(val_loss, model)
        
        # If best model, calculate detailed metrics
        if val_loss == early_stopping.val_loss_min:
            best_th, best_f1 = find_optimal_threshold(val_targets, val_probs)
            preds_opt = (val_probs > best_th).astype(int)
            
            acc = accuracy_score(val_targets, preds_opt)
            prec = precision_score(val_targets, preds_opt, zero_division=0)
            rec = recall_score(val_targets, preds_opt, zero_division=0)
            mcc = matthews_corrcoef(val_targets, preds_opt)
            cm = confusion_matrix(val_targets, preds_opt)
            
            best_metrics = {
                'threshold': best_th,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': best_f1,
                'mcc': mcc,
                'confusion_matrix': cm
            }

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 5. Final Report
    print("\n" + "="*30)
    print("FINAL RESULTS (Best Validation Model)")
    print("="*30)
    print(f"Optimal Threshold : {best_metrics['threshold']:.2f}")
    print(f"Accuracy          : {best_metrics['accuracy']:.4f}")
    print(f"Precision         : {best_metrics['precision']:.4f}")
    print(f"Recall            : {best_metrics['recall']:.4f}")
    print(f"F1 Score          : {best_metrics['f1']:.4f}")
    print(f"MCC               : {best_metrics['mcc']:.4f}")
    print("\nConfusion Matrix:")
    print(best_metrics['confusion_matrix'])
    print("="*30)
    
    # Save Loss Curve
    plot_loss_curves(train_losses, val_losses, os.path.join(args.output_dir, 'loss_curve.png'))
    print(f"Model saved to {model_save_path}")
    print(f"Loss curve saved to {args.output_dir}/loss_curve.png")

if __name__ == "__main__":
    main()