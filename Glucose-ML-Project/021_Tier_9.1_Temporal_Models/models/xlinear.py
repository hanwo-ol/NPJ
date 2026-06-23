import torch
import torch.nn as nn
from .revin import RevIN
from .dlinear import SeriesDecomp


class XLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 kernel_size: int = 25, use_revin: bool = True):
        """
        XLinear (Extended Linear) Model
        Generalizes DLinear by incorporating cross-channel linear interactions.
        
        Args:
            seq_len (int): Length of lookback sequence.
            pred_len (int): Length of prediction horizon.
            num_features (int): Number of input channels/features.
            kernel_size (int): Size of the moving average filter.
            use_revin (bool): Whether to use Reversible Instance Normalization.
        """
        super(XLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.use_revin = use_revin

        self.decomp = SeriesDecomp(kernel_size)
        
        # 1. Temporal projection (same as DLinear)
        self.Linear_Seasonal = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_features)
        ])
        self.Linear_Trend = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_features)
        ])
        
        # 2. Cross-channel linear interaction (new in XLinear)
        self.Channel_Seasonal = nn.Linear(num_features, num_features)
        self.Channel_Trend = nn.Linear(num_features, num_features)
        
        if self.use_revin:
            self.revin = RevIN(num_features)
            
        # If input has multiple channels, project back to 1 target channel (glucose) at the end
        if num_features > 1:
            self.projection = nn.Linear(num_features, 1)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, seq_len, num_features]
        
        if self.use_revin:
            x = self.revin(x, 'norm')
            
        # Series Decomposition
        seasonal_init, trend_init = self.decomp(x)
        
        # Temporal projection per channel
        seasonal_temp = torch.zeros([x.size(0), self.pred_len, self.num_features], dtype=x.dtype, device=x.device)
        trend_temp = torch.zeros([x.size(0), self.pred_len, self.num_features], dtype=x.dtype, device=x.device)
        
        for i in range(self.num_features):
            seasonal_temp[:, :, i] = self.Linear_Seasonal[i](seasonal_init[:, :, i])
            trend_temp[:, :, i] = self.Linear_Trend[i](trend_init[:, :, i])
            
        # Cross-channel interaction mapping
        seasonal_output = self.Channel_Seasonal(seasonal_temp)  # [B, pred_len, num_features]
        trend_output = self.Channel_Trend(trend_temp)           # [B, pred_len, num_features]
        
        x = seasonal_output + trend_output
        
        if self.use_revin:
            x = self.revin(x, 'denorm')
            
        if self.projection is not None:
            x = self.projection(x)  # [B, pred_len, 1]
            
        return x
