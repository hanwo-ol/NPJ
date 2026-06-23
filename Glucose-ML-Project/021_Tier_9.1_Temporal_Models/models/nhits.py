import torch
import torch.nn as nn
import torch.nn.functional as F
from .revin import RevIN


class NHitsBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, downsample_rate: int,
                 hidden_dim: int = 256, theta_dim: int = 64):
        """
        N-HiTS Block with Multi-rate Pooling and Hierarchical Interpolation.
        """
        super(NHitsBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.downsample_rate = downsample_rate
        
        # Calculate downsampled sequence lengths
        # Ensure we always have at least 1 step
        self.seq_len_down = max(1, seq_len // downsample_rate)
        self.pred_len_down = max(1, pred_len // downsample_rate)
        
        # Pooling for downsampling
        if downsample_rate > 1:
            self.pooling = nn.AdaptiveAvgPool1d(self.seq_len_down)
        else:
            self.pooling = None
            
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len_down, hidden_dim),
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
        
        self.backcast_basis = nn.Linear(theta_dim, self.seq_len_down)
        self.forecast_basis = nn.Linear(theta_dim, self.pred_len_down)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B * C, seq_len]
        
        # 1. Downsample lookback
        if self.pooling is not None:
            # x shape: [B * C, seq_len] -> [B * C, 1, seq_len]
            x_down = self.pooling(x.unsqueeze(1)).squeeze(1)
        else:
            x_down = x
            
        # 2. MLP mapping
        h = self.fc(x_down)
        theta_b = self.backcast_proj(h)
        theta_f = self.forecast_proj(h)
        
        backcast_down = self.backcast_basis(theta_b)
        forecast_down = self.forecast_basis(theta_f)
        
        # 3. Hierarchical Interpolation (Upsampling back to original seq_len / pred_len)
        if self.downsample_rate > 1:
            # Interpolate expects 3D tensor [N, C, L] -> [B * C, 1, L_down]
            backcast = F.interpolate(backcast_down.unsqueeze(1), size=self.seq_len, 
                                     mode='linear', align_corners=True).squeeze(1)
            forecast = F.interpolate(forecast_down.unsqueeze(1), size=self.pred_len, 
                                     mode='linear', align_corners=True).squeeze(1)
        else:
            backcast = backcast_down
            forecast = forecast_down
            
        return backcast, forecast


class NHits(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 hidden_dim: int = 256, theta_dim: int = 64, 
                 downsample_rates: list[int] = [8, 4, 2, 1], use_revin: bool = True):
        """
        N-HiTS Model
        https://arxiv.org/abs/2201.12886
        
        Args:
            seq_len (int): Length of lookback sequence.
            pred_len (int): Length of prediction horizon.
            num_features (int): Number of input channels/features.
            hidden_dim (int): Hidden dimension size for MLP.
            theta_dim (int): Dimension of expansion coefficients.
            downsample_rates (list[int]): List of downsampling rates (one per block).
            use_revin (bool): Whether to use Reversible Instance Normalization.
        """
        super(NHits, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.use_revin = use_revin
        
        if self.use_revin:
            self.revin = RevIN(num_features)
            
        self.blocks = nn.ModuleList([
            NHitsBlock(seq_len, pred_len, rate, hidden_dim, theta_dim)
            for rate in downsample_rates
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
        
        # Flatten channels for channel-independent projection: [B * C, L]
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
