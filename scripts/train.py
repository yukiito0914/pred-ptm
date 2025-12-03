import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.MLP import PhosphoMLP
from src.dataset import PhosphoCSVDataset  # [UPDATED] Import the new class
from src.utils import EarlyStopping, plot_loss_curves

def get_args():
    parser = argparse.ArgumentParser(description='Train MLP with CSV Input')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Paths
    # [UPDATED] We only need the CSV path now, not the label+feature_dir separately
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Output directory')
    
    return parser.parse_args()

def calculate_pos_weight(dataset):
    """Calculates weight for positive class based on label imbalance."""
    # dataset.labels is already a Tensor of 0s and 1s
    num_pos = dataset.labels.sum().item()
    num_neg = len(dataset) - num_pos
    
    if num_pos == 0: return torch.tensor(1.0)
    return torch.tensor(num_neg / num_pos)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    # 1. Load Dataset
    print(f"Loading data from {args.csv_path}...")
    full_dataset = PhosphoCSVDataset(csv_file_path=args.csv_path)
    
    # input_dim is determined dynamically from the CSV columns
    input_dim = full_dataset.get_input_dim()
    print(f"Input Feature Dimension: {input_dim}")
    
    # Split Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Setup Model
    model = PhosphoMLP(input_dim=input_dim, output_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    pos_weight = calculate_pos_weight(full_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    early_stopping = EarlyStopping(
        patience=args.patience, 
        path=os.path.join(args.output_dir, 'best_model.pt')
    )

    # 3. Training Loop
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Early Stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    plot_loss_curves(train_losses, val_losses, os.path.join(args.output_dir, 'loss_curve.png'))
    print("Training finished.")

if __name__ == "__main__":
    main()