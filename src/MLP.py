import torch
import torch.nn as nn

class PhosphoMLP(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=64, output_dim=1):
        """
        Simple MLP for Phosphorylation Prediction.
        
        Args:
            input_dim (int): Dimension of input features (Window Size * Features per residue).
            hidden_dim (int): Number of units in hidden layers.
            output_dim (int): Output dimension (1 for binary classification with logits).
        """
        super(PhosphoMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Stabilize training
            nn.ReLU(),
            nn.Dropout(0.1),             # Prevent overfitting
            
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output Layer
            # Note: No Sigmoid here because we use BCEWithLogitsLoss later.
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)