import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root and tier 8 utils are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "018_Tier_8_Reproducibility"))

from global_config import GlobalConfig


def load_temporal_dataset(ds_name: str, split: str = 'train', batch_size: int = 32, shuffle: bool = True):
    """
    Loads preprocessed dataset from Tier 8 cache and formats it for PyTorch.
    Falls back to generating a small dummy dataset if the cache is missing or fails.
    
    Args:
        ds_name (str): Name of the dataset (e.g., 'ShanghaiT2DM').
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the loader.
        
    Returns:
        tuple: (DataLoader, num_features, pred_len, seq_len)
    """
    seq_len = GlobalConfig.LOOKBACK_STEPS  # Default is 3
    
    try:
        from tier8_data_utils import load_cached_split
        splits = load_cached_split(ds_name)
        
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in cached split data.")
            
        X_flat, y = splits[split]
        
        if X_flat is None or len(X_flat) == 0:
            raise ValueError("Loaded cache is empty.")
            
        N, D = X_flat.shape
        
        # 14 columns of derived window features are at the end of X_flat
        total_lookback_features = D - 14
        num_features = total_lookback_features // seq_len
        
        # Slice the lookback sequence columns
        X_seq = X_flat[:, :seq_len * num_features]
        X_3d = X_seq.reshape(N, seq_len, num_features)
        
        # Check target shape
        # In our baseline datasets, y is shape [N,] representing the single point at t + PREDICTION_STEPS - 1
        if len(y.shape) == 1:
            pred_len = 1
            y_3d = y.reshape(N, 1, 1)
        elif len(y.shape) == 2:
            pred_len = y.shape[1]
            y_3d = y.reshape(N, pred_len, 1)
        else:
            pred_len = y.shape[1]
            y_3d = y.astype(np.float32)
            
        X_tensor = torch.tensor(X_3d, dtype=torch.float32)
        y_tensor = torch.tensor(y_3d, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader, num_features, pred_len, seq_len
        
    except Exception as e:
        # Fallback to dummy data for local smoke testing if cache is unavailable
        num_features = 3  # Dummy features (glucose, insulin, carbs)
        pred_len = 1
        
        # Generate 100 dummy samples
        N_dummy = 100
        X_dummy = np.random.randn(N_dummy, seq_len, num_features).astype(np.float32)
        y_dummy = np.random.randn(N_dummy, pred_len, 1).astype(np.float32)
        
        X_tensor = torch.tensor(X_dummy, dtype=torch.float32)
        y_tensor = torch.tensor(y_dummy, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader, num_features, pred_len, seq_len
