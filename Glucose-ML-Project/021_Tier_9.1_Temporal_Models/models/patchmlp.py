import torch
import torch.nn as nn
import torch.nn.functional as F
from .revin import RevIN


class PatchMLP(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, num_features: int,
                 patch_len: int = 2, stride: int = 1, d_model: int = 32,
                 dropout: float = 0.1, use_revin: bool = True):
        """
        PatchMLP Model
        https://arxiv.org/abs/2306.06054
        
        Args:
            seq_len (int): Length of lookback sequence.
            pred_len (int): Length of prediction horizon.
            num_features (int): Number of input channels/features.
            patch_len (int): Length of each patch.
            stride (int): Stride for patch segmentation.
            d_model (int): Hidden dimension size for patch embeddings.
            dropout (float): Dropout probability.
            use_revin (bool): Whether to use Reversible Instance Normalization.
        """
        super(PatchMLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.use_revin = use_revin
        
        # Ensure patch_len is not greater than seq_len
        self.patch_len = min(patch_len, seq_len)
        self.stride = stride
        
        # Calculate padding and number of patches
        pad_size = (seq_len - self.patch_len) % stride
        if pad_size != 0:
            self.pad_len = stride - pad_size
        else:
            self.pad_len = 0
            
        self.padded_seq_len = seq_len + self.pad_len
        self.num_patches = (self.padded_seq_len - self.patch_len) // stride + 1

        if self.use_revin:
            self.revin = RevIN(num_features)
            
        # 1. Patch Embedding Layer
        self.patch_embed = nn.Linear(self.patch_len, d_model)
        
        # 2. Patch-Mixing MLP (mixes across patches)
        self.patch_mixing = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(self.num_patches, self.num_patches),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Hidden-Mixing MLP (mixes across hidden dimensions)
        self.hidden_mixing = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. Final linear forecasting layer
        self.predict_linear = nn.Linear(self.num_patches * d_model, pred_len)
        
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
        
        # Flatten channels into batch dimension: [B * C, L]
        x_flat = x.transpose(1, 2).reshape(B * C, L)
        
        # Pad at the beginning if necessary to divide into patches nicely
        if self.pad_len > 0:
            # Repeat the first value to pad
            pad_val = x_flat[:, 0:1].repeat(1, self.pad_len)
            x_flat = torch.cat([pad_val, x_flat], dim=1)
            
        # 1. Unfold into patches: [B * C, num_patches, patch_len]
        # unfold expects [N, C, L] -> [B * C, 1, padded_seq_len]
        x_patches = x_flat.unsqueeze(1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_patches = x_patches.squeeze(1)  # [B * C, num_patches, patch_len]
        
        # 2. Embed patches: [B * C, num_patches, d_model]
        x_emb = self.patch_embed(x_patches)
        
        # 3. Patch Mixing
        # norm operates on last dim -> [B*C, num_patches, d_model]. We transpose for time mixing
        residual = x_emb
        x_norm = self.patch_mixing[0](x_emb)  # LayerNorm
        x_norm = x_norm.transpose(1, 2)       # [B * C, d_model, num_patches]
        x_norm = self.patch_mixing[1:](x_norm)  # Linear -> ReLU -> Dropout
        x_emb = residual + x_norm.transpose(1, 2)  # [B * C, num_patches, d_model]
        
        # 4. Hidden Mixing
        residual = x_emb
        x_norm = self.hidden_mixing(x_emb)     # LayerNorm -> Linear -> ReLU -> Dropout
        x_emb = residual + x_norm
        
        # 5. Flatten patches and predict
        x_flat_pred = x_emb.reshape(B * C, -1)  # [B * C, num_patches * d_model]
        out_flat = self.predict_linear(x_flat_pred)  # [B * C, pred_len]
        
        # Reshape back to [B, pred_len, C]
        out = out_flat.reshape(B, C, self.pred_len).transpose(1, 2)
        
        if self.use_revin:
            out = self.revin(out, 'denorm')
            
        if self.projection is not None:
            out = self.projection(out)  # [B, pred_len, 1]
            
        return out
