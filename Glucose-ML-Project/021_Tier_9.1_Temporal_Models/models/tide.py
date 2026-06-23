import torch
import torch.nn as nn
from .revin import RevIN


class TiDE(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 proj_dim: int = 4, hidden_dim: int = 128, decoder_dim: int = 8,
                 dropout: float = 0.1, use_revin: bool = True):
        """
        TiDE Model (Time-series Information Dissemination Engine)
        https://arxiv.org/abs/2304.08965
        
        Args:
            seq_len (int): Length of lookback sequence.
            pred_len (int): Length of prediction horizon.
            num_features (int): Number of input channels/features.
            proj_dim (int): Projection dimension for input features.
            hidden_dim (int): Hidden dimension size for Encoder/Decoder MLPs.
            decoder_dim (int): Representation dimension for each output step in Decoder.
            dropout (float): Dropout probability.
            use_revin (bool): Whether to use Reversible Instance Normalization.
        """
        super(TiDE, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.use_revin = use_revin
        
        if self.use_revin:
            self.revin = RevIN(num_features)
            
        # 1. Feature Projection (maps input channels to proj_dim)
        self.feature_proj = nn.Linear(num_features, proj_dim)
        
        # 2. Encoder MLP
        encoded_size = seq_len * proj_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoded_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Decoder MLP
        decoded_size = pred_len * decoder_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, decoded_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. Output Projection (maps step representations to 1 target channel)
        self.output_proj = nn.Linear(decoder_dim, 1)
        
        # 5. Direct Linear Residual Connection (from past glucose to future glucose)
        # Note: Glucose is assumed to be at index 0 of the input features
        self.linear_residual = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, seq_len, num_features]
        
        # Save raw glucose sequence (index 0) for residual connection
        glucose_past = x[:, :, 0]  # [B, seq_len]
        
        if self.use_revin:
            x = self.revin(x, 'norm')
            
        B, L, C = x.size()
        
        # 1. Project features step-wise
        x_proj = self.feature_proj(x)  # [B, seq_len, proj_dim]
        
        # 2. Flatten and encode
        x_flat = x_proj.reshape(B, -1)  # [B, seq_len * proj_dim]
        e = self.encoder(x_flat)       # [B, hidden_dim]
        
        # 3. Decode
        d = self.decoder(e)            # [B, pred_len * decoder_dim]
        d_reshaped = d.reshape(B, self.pred_len, -1)  # [B, pred_len, decoder_dim]
        
        # 4. Output projection
        out = self.output_proj(d_reshaped)  # [B, pred_len, 1]
        
        if self.use_revin:
            # Denormalize. RevIN expects target size matching num_features, so we pad out to num_features first, 
            # denormalize, and then slice the target channel (index 0).
            # This is mathematically consistent because the scale and mean of the glucose channel are preserved.
            if self.num_features > 1:
                # Pad out with zeros for denorm: [B, pred_len, num_features]
                out_padded = torch.zeros(B, self.pred_len, self.num_features, dtype=out.dtype, device=out.device)
                out_padded[:, :, 0:1] = out
                out_padded = self.revin(out_padded, 'denorm')
                out = out_padded[:, :, 0:1]
            else:
                out = self.revin(out, 'denorm')
                
        # 5. Direct linear residual connection
        res = self.linear_residual(glucose_past).unsqueeze(-1)  # [B, pred_len, 1]
        
        final_output = out + res  # [B, pred_len, 1]
        
        return final_output
