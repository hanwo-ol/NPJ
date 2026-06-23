import torch
import torch.nn as nn
from .revin import RevIN


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        """
        Moving average block to extract the trend of a time series.
        """
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Steps, Channels]
        # AvgPool1d expects: [Batch, Channels, Steps]
        # Pad at the front and end by repeating the edge values to prevent edge effects
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.avg(x)
        x = x.transpose(1, 2)  # [B, L, C]
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        """
        Series decomposition block to split series into Trend and Seasonal (residual).
        """
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int, 
                 kernel_size: int = 25, use_revin: bool = True):
        """
        DLinear Model
        https://arxiv.org/abs/2205.13504
        
        Args:
            seq_len (int): Length of lookback sequence.
            pred_len (int): Length of prediction horizon.
            num_features (int): Number of input channels/features.
            kernel_size (int): Size of the moving average filter.
            use_revin (bool): Whether to use Reversible Instance Normalization.
        """
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.use_revin = use_revin

        self.decomp = SeriesDecomp(kernel_size)
        
        # Channel-independent linear layers for seasonal and trend components
        self.Linear_Seasonal = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_features)
        ])
        self.Linear_Trend = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_features)
        ])
        
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
        
        # Channel-independent linear forecasting
        seasonal_output = torch.zeros([x.size(0), self.pred_len, self.num_features], dtype=x.dtype, device=x.device)
        trend_output = torch.zeros([x.size(0), self.pred_len, self.num_features], dtype=x.dtype, device=x.device)
        
        for i in range(self.num_features):
            seasonal_output[:, :, i] = self.Linear_Seasonal[i](seasonal_init[:, :, i])
            trend_output[:, :, i] = self.Linear_Trend[i](trend_init[:, :, i])
            
        x = seasonal_output + trend_output  # [B, pred_len, num_features]
        
        if self.use_revin:
            x = self.revin(x, 'denorm')
            
        # Project to target 1 channel if multivariate input
        if self.projection is not None:
            x = self.projection(x)  # [B, pred_len, 1]
            
        return x
