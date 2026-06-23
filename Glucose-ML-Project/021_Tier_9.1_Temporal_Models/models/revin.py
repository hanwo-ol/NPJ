import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        Reversible Instance Normalization (RevIN)
        https://openreview.net/forum?id=qn7t3GmqEs0
        
        Args:
            num_features (int): Number of input channels (features).
            eps (float): Small value for numerical stability.
            affine (bool): If True, apply learnable scale (affine weight) and shift (affine bias).
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self._init_params()

    def _init_params(self):
        # Learnable scale and shift parameters
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Steps, Channels]
            mode (str): 'norm' for normalization, 'denorm' for denormalization
        Returns:
            torch.Tensor: Normalized or denormalized tensor of the same shape.
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unsupported RevIN mode: {mode}")
        return x

    def _get_statistics(self, x: torch.Tensor):
        # Calculate mean and standard deviation along the time axis (dim=1)
        # x shape: [B, L, C] -> mean/std shape: [B, 1, C]
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x
