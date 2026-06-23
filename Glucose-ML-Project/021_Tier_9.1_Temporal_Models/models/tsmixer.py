import torch
import torch.nn as nn
from .revin import RevIN


class ResBlock(nn.Module):
    def __init__(self, seq_len: int, num_features: int, hidden_dim: int, dropout: float = 0.1):
        super(ResBlock, self).__init__()
        
        # Temporal Mixing
        self.temporal_norm = nn.LayerNorm(num_features)
        self.temporal_linear = nn.Linear(seq_len, seq_len)
        self.temporal_act = nn.ReLU()
        self.temporal_dropout = nn.Dropout(dropout)
        
        # Feature Mixing (MLP)
        self.feature_norm = nn.LayerNorm(num_features)
        self.feature_linear1 = nn.Linear(num_features, hidden_dim)
        self.feature_act = nn.ReLU()
        self.feature_dropout1 = nn.Dropout(dropout)
        self.feature_linear2 = nn.Linear(hidden_dim, num_features)
        self.feature_dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, L, C]
        
        # 1. Temporal Mixing (mix along time axis L)
        residual = x
        x_norm = self.temporal_norm(x)
        x_temp = x_norm.transpose(1, 2)  # [B, C, L]
        x_temp = self.temporal_linear(x_temp)
        x_temp = self.temporal_act(x_temp)
        x_temp = self.temporal_dropout(x_temp)
        x = residual + x_temp.transpose(1, 2)  # [B, L, C]
        
        # 2. Feature Mixing (mix along channel axis C)
        residual = x
        x_norm = self.feature_norm(x)
        x_temp = self.feature_linear1(x_norm)
        x_temp = self.feature_act(x_temp)
        x_temp = self.feature_dropout1(x_temp)
        x_temp = self.feature_linear2(x_temp)
        x_temp = self.feature_dropout2(x_temp)
        x = residual + x_temp  # [B, L, C]
        
        return x


class TSMixer(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 hidden_dim: int = 64, n_blocks: int = 2, dropout: float = 0.1,
                 use_revin: bool = True):
        """
        TSMixer Model
        https://arxiv.org/abs/2303.06053
        
        Args:
            seq_len (int): Length of lookback sequence.
            pred_len (int): Length of prediction horizon.
            num_features (int): Number of input channels/features.
            hidden_dim (int): Hidden dimension size for feature mixing MLP.
            n_blocks (int): Number of Mixer blocks.
            dropout (float): Dropout probability.
            use_revin (bool): Whether to use Reversible Instance Normalization.
        """
        super(TSMixer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.use_revin = use_revin
        
        if self.use_revin:
            self.revin = RevIN(num_features)
            
        self.blocks = nn.ModuleList([
            ResBlock(seq_len, num_features, hidden_dim, dropout)
            for _ in range(n_blocks)
        ])
        
        # Project from seq_len to pred_len along time axis
        self.time_project = nn.Linear(seq_len, pred_len)
        
        # If input has multiple channels, project back to 1 target channel (glucose) at the end
        if num_features > 1:
            self.projection = nn.Linear(num_features, 1)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, seq_len, num_features]
        
        if self.use_revin:
            x = self.revin(x, 'norm')
            
        for block in self.blocks:
            x = block(x)
            
        # Map time dimension: [B, L, C] -> [B, C, L] -> Linear(L, pred_len) -> [B, C, pred_len] -> [B, pred_len, C]
        x = x.transpose(1, 2)
        x = self.time_project(x)
        x = x.transpose(1, 2)  # [B, pred_len, C]
        
        if self.use_revin:
            x = self.revin(x, 'denorm')
            
        if self.projection is not None:
            x = self.projection(x)  # [B, pred_len, 1]
            
        return x
