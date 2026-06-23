import torch
import torch.nn as nn
from .revin import RevIN


class SOFTS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 d_model: int = 64, dropout: float = 0.1, use_revin: bool = True):
        """
        SOFTS Model (Scalable One-pass Fourier Time Series)
        https://arxiv.org/abs/2404.14197
        
        Args:
            seq_len (int): Length of lookback sequence.
            pred_len (int): Length of prediction horizon.
            num_features (int): Number of input channels/features.
            d_model (int): Hidden dimension size for global representations.
            dropout (float): Dropout probability.
            use_revin (bool): Whether to use Reversible Instance Normalization.
        """
        super(SOFTS, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.use_revin = use_revin
        
        if self.use_revin:
            self.revin = RevIN(num_features)
            
        # Channel-independent embedding layer (maps seq_len to d_model)
        self.emb_linear = nn.Linear(seq_len, d_model)
        
        # Centralized Global Interaction MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Channel-wise fusion layer
        self.fusion = nn.Linear(2 * d_model, d_model)
        
        # Forecasting projection layer
        self.predict_linear = nn.Linear(d_model, pred_len)
        
        # If input has multiple channels, project back to 1 target channel (glucose) at the end
        if num_features > 1:
            self.projection = nn.Linear(num_features, 1)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, seq_len, num_features]
        
        if self.use_revin:
            x = self.revin(x, 'norm')
            
        B, L, C = x.size()
        
        # 1. Transpose to [B, C, L] to embed time steps for each channel independently
        x_trans = x.transpose(1, 2)
        x_emb = self.emb_linear(x_trans)  # [B, C, d_model]
        
        # 2. Centralized Global Pooling (average over channels)
        x_global = x_emb.mean(dim=1)  # [B, d_model]
        
        # 3. Apply global MLP to aggregate feature interaction
        x_global = self.global_mlp(x_global)  # [B, d_model]
        
        # 4. Broadcast global context and concatenate with channel embeddings
        x_global_rep = x_global.unsqueeze(1).repeat(1, C, 1)  # [B, C, d_model]
        x_combined = torch.cat([x_emb, x_global_rep], dim=-1)   # [B, C, 2 * d_model]
        
        # 5. Fuse channel information
        x_fused = self.fusion(x_combined)  # [B, C, d_model]
        
        # 6. Predict future values
        out = self.predict_linear(x_fused)  # [B, C, pred_len]
        out = out.transpose(1, 2)          # [B, pred_len, C]
        
        if self.use_revin:
            out = self.revin(out, 'denorm')
            
        if self.projection is not None:
            out = self.projection(out)  # [B, pred_len, 1]
            
        return out
