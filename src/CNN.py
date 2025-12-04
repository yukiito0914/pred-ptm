import torch
import torch.nn as nn

class PhosphoCNN(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=64, output_dim=1):
        """
        1D-CNN model for Phosphorylation Prediction.
        
        Args:
            input_dim (int): Dimension of structural features (Window * 3). Default 45.
        """
        super(PhosphoCNN, self).__init__()
        
        # --- 1. Embedding Layer ---
        self.emb_dim = 10
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=self.emb_dim, padding_idx=0)
        
        # --- 2. CNN Layers (Feature Extractor for Sequence) ---
        self.cnn_channels = 32
        self.kernel_size = 3
        
        self.cnn = nn.Sequential(
            # Input: [Batch, Emb_Dim(10), Length(15)]
            nn.Conv1d(in_channels=self.emb_dim, out_channels=self.cnn_channels, 
                      kernel_size=self.kernel_size, padding=1), 
            nn.BatchNorm1d(self.cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # You can add a second CNN layer if you want deeper extraction
            # nn.Conv1d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
        )
        
        # --- Calculate Dimensions ---
        # Window size = 45 // 3 = 15
        self.window_size = input_dim // 3
        
        # CNN Output Size calculation:
        # Since we use padding=1 and kernel=3, the length remains 15.
        # Flattened size = Length(15) * Out_Channels(32) = 480
        self.cnn_flat_dim = self.window_size * self.cnn_channels
        
        # Total Input for the final Classifier
        # CNN_Output(480) + Structural_Features(45) = 525
        total_dense_input = self.cnn_flat_dim + input_dim
        
        # --- 3. Classifier (Dense Layers) ---
        self.classifier = nn.Sequential(
            nn.Linear(total_dense_input, hidden_dim),
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
        x_features: (Batch, 45) - Structural features
        x_seq:      (Batch, 15) - Amino acid indices
        """
        # A. Process Sequence with CNN
        # 1. Embedding: (Batch, 15) -> (Batch, 15, 10)
        emb = self.embedding(x_seq.long())
        
        # 2. Permute for CNN: PyTorch CNN expects (Batch, Channels, Length)
        # (Batch, 15, 10) -> (Batch, 10, 15)
        emb = emb.permute(0, 2, 1)
        
        # 3. Apply CNN
        # (Batch, 10, 15) -> (Batch, 32, 15)
        cnn_out = self.cnn(emb)
        
        # 4. Flatten
        # (Batch, 32, 15) -> (Batch, 480)
        cnn_flat = cnn_out.view(cnn_out.size(0), -1)
        
        # B. Combine with Structural Features
        # Concatenate: [Batch, 45] + [Batch, 480] -> [Batch, 525]
        x_combined = torch.cat([x_features, cnn_flat], dim=1)
        
        # C. Final Classification
        return self.classifier(x_combined)