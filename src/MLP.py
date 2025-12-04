import torch
import torch.nn as nn

class PhosphoMLP(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=64, output_dim=1):
        super(PhosphoMLP, self).__init__()
        
        # --- [NEW] Embedding Layer ---
        # 21 kinds of AA (including padding=0)
        # Embedding Dimension = 10 (each AA becomes a vector of size 10)
        self.emb_dim = 10
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=self.emb_dim, padding_idx=0)
        
        # Window size calculation
        # input_dim was 45 (15 * 3 features)
        # We assume Window Size is input_dim / 3 = 15
        self.window_size = input_dim // 3
        
        # --- Calculate Total Input Dimension ---
        # Original Features: 45
        # Embedded Features: 15 (window) * 10 (emb_dim) = 150
        # Total: 45 + 150 = 195
        total_input_dim = input_dim + (self.window_size * self.emb_dim)
        
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim), # 195 -> 64
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x_features, x_seq):
        """
        x_features: (Batch, 45) - pLDDT, SASA, Hydro
        x_seq:      (Batch, 15) - Integer indices of amino acids
        """
        # 1. Get Embeddings
        # x_seq (Batch, 15) -> emb (Batch, 15, 10)
        emb = self.embedding(x_seq.long())
        
        # Flatten embeddings: (Batch, 15, 10) -> (Batch, 150)
        emb_flat = emb.view(emb.size(0), -1)
        
        # 2. Concatenate with original features
        # [Batch, 45] + [Batch, 150] -> [Batch, 195]
        x_combined = torch.cat([x_features, emb_flat], dim=1)
        
        # 3. Pass through MLP
        return self.network(x_combined)