import torch
import torch.nn as nn

class PhosphoCNNAttention(nn.Module):
    def __init__(self, input_dim=93, hidden_dim=128, output_dim=1):
        """
        Hybrid Model: Deep 1D-CNN (3 Layers) + Self-Attention Mechanism.
        
        Architecture:
          1. Embedding: Maps amino acid indices to dense vectors.
          2. Deep CNN: Extracts local motifs (e.g., R-R-S) and wider context.
          3. Self-Attention: Captures global dependencies between extracted motifs.
          4. Residual Connection: Stabilizes training by adding CNN output to Attention output.
          5. Classifier: Combines sequence features with structural features (SASA, etc.) for final prediction.
        """
        super(PhosphoCNNAttention, self).__init__()
        
        # --- 1. Embedding Layer ---
        # Maps 21 amino acid types to 10-dimensional vectors
        self.emb_dim = 10
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=self.emb_dim, padding_idx=0)
        
        # --- 2. Deep CNN Layers (Feature Extractor) ---
        # Progressively increase channels: 10 -> 32 -> 64 -> 128
        self.cnn_channels = 128
        
        self.cnn = nn.Sequential(
            # Layer 1: Local properties (Charge, Hydrophobicity, etc.)
            nn.Conv1d(in_channels=self.emb_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 2: Motif patterns (e.g., Kinase recognition motifs)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3: Wider context
            nn.Conv1d(in_channels=64, out_channels=self.cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # --- 3. Attention Mechanism ---
        # Multihead Attention to find relationships between different parts of the sequence.
        # embed_dim must match the output channels of the CNN (128).
        self.attention = nn.MultiheadAttention(embed_dim=self.cnn_channels, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.cnn_channels) # Normalization for stability
        
        # --- Dimension Calculation ---
        # Window size is maintained at 31 (due to padding=1 in CNN)
        # input_dim comes as 93 (31 * 3 features). So window_len is 31.
        self.window_len = input_dim // 3
        
        # Flatten size = Length(31) * Channels(128) = 3968
        self.cnn_flat_dim = self.window_len * self.cnn_channels
        
        # --- 4. Classifier (MLP) ---
        # Input: CNN/Attn Output (3968) + Structural Features (93)
        total_dense_input = self.cnn_flat_dim + input_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_dense_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x_features, x_seq):
        """
        x_features: (Batch, 93) - Structural features (pLDDT, SASA, Hydro)
        x_seq:      (Batch, 31) - Sequence indices
        """
        # --- A. Sequence Processing ---
        
        # 1. Embedding
        # (Batch, 31) -> (Batch, 31, 10)
        emb = self.embedding(x_seq.long())
        
        # 2. CNN (Local Patterns)
        # Permute for CNN: (Batch, 10, 31)
        emb_cnn = emb.permute(0, 2, 1)
        cnn_out = self.cnn(emb_cnn) # Output: (Batch, 128, 31)
        
        # 3. Self-Attention (Global Context)
        # Permute for Attention: (Batch, 31, 128)
        # Attention expects (Batch, Seq_Len, Features)
        cnn_out_permuted = cnn_out.permute(0, 2, 1)
        
        # Apply Attention
        # query, key, value are all the same (Self-Attention)
        attn_out, _ = self.attention(cnn_out_permuted, cnn_out_permuted, cnn_out_permuted)
        
        # 4. Residual Connection & Normalization (Crucial for deep models)
        # Add original CNN output to Attention output (ResNet-style)
        # This allows the model to use raw CNN features if Attention isn't helpful yet.
        x_seq_final = self.attn_norm(cnn_out_permuted + attn_out)
        
        # Flatten: (Batch, 31, 128) -> (Batch, 3968)
        feat_flat = x_seq_final.reshape(x_seq_final.size(0), -1)
        
        # --- B. Combine with Structure ---
        # Concatenate: [Structure(93) + Sequence(3968)]
        x_combined = torch.cat([x_features, feat_flat], dim=1)
        
        # --- C. Final Classification ---
        return self.classifier(x_combined)