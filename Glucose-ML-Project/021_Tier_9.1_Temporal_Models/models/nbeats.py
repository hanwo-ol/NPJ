import torch
import torch.nn as nn
from .revin import RevIN


class NBeatsBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, hidden_dim: int = 256, theta_dim: int = 64):
        """
        Generic N-BEATS Block.
        """
        super(NBeatsBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.backcast_proj = nn.Linear(hidden_dim, theta_dim)
        self.forecast_proj = nn.Linear(hidden_dim, theta_dim)
        
        # Generic basis projections
        self.backcast_basis = nn.Linear(theta_dim, seq_len)
        self.forecast_basis = nn.Linear(theta_dim, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B * C, seq_len]
        h = self.fc(x)
        theta_b = self.backcast_proj(h)
        theta_f = self.forecast_proj(h)
        
        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 hidden_dim: int = 256, theta_dim: int = 64, n_blocks: int = 4,
                 use_revin: bool = True):
        """
        N-BEATS Model
        https://arxiv.org/abs/1905.10437
        
        Args:
            seq_len (int): Length of lookback sequence.
            pred_len (int): Length of prediction horizon.
            num_features (int): Number of input channels/features.
            hidden_dim (int): Hidden dimension size for MLP.
            theta_dim (int): Dimension of expansion coefficients.
            n_blocks (int): Number of blocks in the stack.
            use_revin (bool): Whether to use Reversible Instance Normalization.
        """
        super(NBeats, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.use_revin = use_revin

        if self.use_revin:
            self.revin = RevIN(num_features)
            
        self.blocks = nn.ModuleList([
            NBeatsBlock(seq_len, pred_len, hidden_dim, theta_dim)
            for _ in range(n_blocks)
        ])
        
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
        
        # Flatten channels into batch dimension for channel independence: [B * C, L]
        x_flat = x.transpose(1, 2).reshape(B * C, L)
        
        forecast_total = torch.zeros(B * C, self.pred_len, dtype=x.dtype, device=x.device)
        
        for block in self.blocks:
            backcast, forecast = block(x_flat)
            x_flat = x_flat - backcast
            forecast_total = forecast_total + forecast
            
        # Reshape back to [B, pred_len, C]
        out = forecast_total.reshape(B, C, self.pred_len).transpose(1, 2)
        
        if self.use_revin:
            out = self.revin(out, 'denorm')
            
        if self.projection is not None:
            out = self.projection(out)  # [B, pred_len, 1]
            
        return out
