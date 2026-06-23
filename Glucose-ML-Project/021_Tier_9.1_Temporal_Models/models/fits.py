import torch
import torch.nn as nn
import torch.nn.functional as F
from .revin import RevIN


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Linear layer for complex-valued tensors: Y = X * W + b.
        Compatible with older PyTorch versions.
        """
        super(ComplexLinear, self).__init__()
        # Initialize weights with standard scaling
        self.real_weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / (in_features ** 0.5)))
        self.imag_weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / (in_features ** 0.5)))
        self.real_bias = nn.Parameter(torch.zeros(out_features))
        self.imag_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: [Batch, Channels, in_features]
        x_real = x.real
        x_imag = x.imag
        
        # Complex multiplication: (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
        out_real = F.linear(x_real, self.real_weight, self.real_bias) - F.linear(x_imag, self.imag_weight, self.imag_bias)
        out_imag = F.linear(x_real, self.imag_weight, self.imag_bias) + F.linear(x_imag, self.real_weight, self.real_bias)
        
        return torch.complex(out_real, out_imag)


class FITS(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 cut_freq: int = None, use_revin: bool = True):
        """
        FITS Model (Frequency Interpolation Time Series)
        https://arxiv.org/abs/2307.03750
        
        Args:
            seq_len (int): Length of lookback sequence.
            pred_len (int): Length of prediction horizon.
            num_features (int): Number of input channels/features.
            cut_freq (int): Number of low-frequency components to keep. Defaults to all.
            use_revin (bool): Whether to use Reversible Instance Normalization.
        """
        super(FITS, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.use_revin = use_revin
        
        # Calculate full frequency dimension for lookback and prediction
        self.seq_len_freq = seq_len // 2 + 1
        self.pred_len_freq = pred_len // 2 + 1
        
        # If cut_freq is specified, we perform low-pass filtering by slicing frequencies
        if cut_freq is not None:
            self.cut_freq = min(cut_freq, self.seq_len_freq)
        else:
            self.cut_freq = self.seq_len_freq

        # Complex linear mapping in frequency domain
        self.frequency_mapping = ComplexLinear(self.cut_freq, self.pred_len_freq)
        
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
            
        B, L, C = x.size()
        
        # 1. Real Fast Fourier Transform along time dimension (dim=1)
        # x_freq shape: [B, seq_len_freq, C]
        x_freq = torch.fft.rfft(x, dim=1)
        
        # 2. Low-pass filtering: slice first cut_freq frequencies
        x_freq_cut = x_freq[:, :self.cut_freq, :]
        
        # 3. Transpose to [B, C, cut_freq] for linear mapping along frequency axis
        x_freq_cut = x_freq_cut.transpose(1, 2)
        
        # 4. Apply Complex Linear mapping (Frequency Interpolation)
        # y_freq_cut shape: [B, C, pred_len_freq]
        y_freq_cut = self.frequency_mapping(x_freq_cut)
        
        # 5. Transpose back: [B, pred_len_freq, C]
        y_freq = y_freq_cut.transpose(1, 2)
        
        # 6. Inverse Real FFT to return to time domain
        # out shape: [B, pred_len, C]
        out = torch.fft.irfft(y_freq, n=self.pred_len, dim=1)
        
        if self.use_revin:
            out = self.revin(out, 'denorm')
            
        if self.projection is not None:
            out = self.projection(out)  # [B, pred_len, 1]
            
        return out
