import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Add current folder to path
sys.path.append(str(Path(__file__).resolve().parent))

from models import DLinear, TSMixer, XLinear, TiDE, NHits, NBeats, FITS, SOFTS, PatchMLP
from data_loader import load_temporal_dataset


def run_model_smoke_test(model_class, model_name, seq_len, pred_len, num_features, kwargs={}):
    """
    Runs a smoke test for a single model:
    - Forward pass shape check
    - Backward pass gradient flow check
    """
    print(f"Testing model: {model_name}...")
    
    # 1. Instantiate model
    model = model_class(seq_len=seq_len, pred_len=pred_len, num_features=num_features, **kwargs)
    model.train()
    
    # 2. Forward pass with dummy input
    batch_size = 8
    x_dummy = torch.randn(batch_size, seq_len, num_features)
    y_dummy = torch.randn(batch_size, pred_len, 1)
    
    try:
        out = model(x_dummy)
    except Exception as e:
        print(f"  [ERROR] {model_name} forward pass failed: {e}")
        return False
        
    expected_shape = (batch_size, pred_len, 1)
    if out.shape != expected_shape:
        print(f"  [ERROR] {model_name} output shape mismatch. Expected {expected_shape}, got {out.shape}")
        return False
        
    # 3. Backward pass (Gradient flow check)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    
    loss = nn.L1Loss()(out, y_dummy)
    
    try:
        loss.backward()
    except Exception as e:
        print(f"  [ERROR] {model_name} backward pass failed: {e}")
        return False
        
    # Check if gradients were computed for at least one parameter
    grad_computed = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check if gradient has non-zero values
            if torch.sum(torch.abs(param.grad)) > 0:
                grad_computed = True
                break
                
    if not grad_computed:
        print(f"  [ERROR] {model_name} no gradients computed.")
        return False
        
    # Take a single optimizer step
    optimizer.step()
    
    print(f"  [PASSED] {model_name} successfully passed forward/backward tests.")
    return True


def run_dataloader_integration(loader, model_class, model_name, seq_len, pred_len, num_features, kwargs={}):
    """
    Runs a single step training test using the actual DataLoader.
    """
    print(f"Testing DataLoader integration with {model_name}...")
    model = model_class(seq_len=seq_len, pred_len=pred_len, num_features=num_features, **kwargs)
    model.train()
    
    # Extract one batch from loader
    x_batch, y_batch = next(iter(loader))
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    
    try:
        out = model(x_batch)
        loss = nn.L1Loss()(out, y_batch)
        loss.backward()
        optimizer.step()
        print(f"  [PASSED] {model_name} successfully processed batch of shape {list(x_batch.shape)}")
        return True
    except Exception as e:
        print(f"  [ERROR] {model_name} failed on real batch processing: {e}")
        return False


def main():
    print("=" * 60)
    print("  Temporal Models Smoke Test Suite")
    print("=" * 60)
    
    # Check if CPU mode is active
    device = torch.device('cpu')
    print(f"Running on Device: {device} (Forced CPU-only)")
    
    # Setup test dimensions
    seq_len = 3
    pred_len = 1
    num_features = 3
    
    # Model registry to test
    # Key: Model Name -> (Class, kwargs)
    models_to_test = {
        # --- Rank 1: Linear & Mixer Models ---
        'DLinear':  (DLinear,  {'kernel_size': 3}),
        'TSMixer':  (TSMixer,  {'hidden_dim': 16, 'n_blocks': 1}),
        'XLinear':  (XLinear,  {'kernel_size': 3}),
        
        # --- Rank 2: Encoder-Decoder & Basis Expansion Models ---
        'TiDE':     (TiDE,     {'proj_dim': 2, 'hidden_dim': 32, 'decoder_dim': 4}),
        'NHits':    (NHits,    {'hidden_dim': 32, 'theta_dim': 16, 'downsample_rates': [2, 1]}),
        'NBeats':   (NBeats,   {'hidden_dim': 32, 'theta_dim': 16, 'n_blocks': 2}),
        
        # --- Rank 3: Frequency & Segment Mixing Models ---
        'FITS':     (FITS,     {'cut_freq': 2}),
        'SOFTS':    (SOFTS,    {'d_model': 16}),
        'PatchMLP': (PatchMLP, {'patch_len': 2, 'stride': 1, 'd_model': 16})
    }
    
    results = {}
    
    # Step 1: Run Dummy Smoke Tests for Rank 1, Rank 2, Rank 3 sequentially
    print("\n--- Phase 1: Sequential Model Validation ---")
    for name, (model_class, kwargs) in models_to_test.items():
        t0 = time.perf_counter()
        passed = run_model_smoke_test(model_class, name, seq_len, pred_len, num_features, kwargs)
        elapsed = time.perf_counter() - t0
        results[name] = {
            'smoke_passed': passed,
            'smoke_time_sec': elapsed,
            'loader_passed': False
        }
        print("-" * 40)
        
    # Step 2: Load DataLoader (uses cache if exists, else falls back to dummy dataset)
    print("\n--- Phase 2: DataLoader Integration Validation ---")
    loader, ds_features, ds_pred_len, ds_seq_len = load_temporal_dataset('ShanghaiT2DM', split='train', batch_size=4)
    print(f"Data Loader Loaded successfully. Features: {ds_features}, Target steps: {ds_pred_len}, Lookback steps: {ds_seq_len}")
    
    for name, (model_class, kwargs) in models_to_test.items():
        t0 = time.perf_counter()
        passed = run_dataloader_integration(loader, model_class, name, ds_seq_len, ds_pred_len, ds_features, kwargs)
        results[name]['loader_passed'] = passed
        results[name]['loader_time_sec'] = time.perf_counter() - t0
        print("-" * 40)
        
    # Step 3: Print Summary Table
    print("\n" + "=" * 60)
    print("=== SMOKE TEST SUMMARY REPORT ===")
    print("=" * 60)
    print(f"{'Model':<12} | {'Dummy Test':<10} | {'DataLoader Test':<15} | {'Total Time (s)':<15}")
    print("-" * 60)
    
    all_success = True
    for name, res in results.items():
        smoke_str = "PASSED" if res['smoke_passed'] else "FAILED"
        loader_str = "PASSED" if res['loader_passed'] else "FAILED"
        total_time = res['smoke_time_sec'] + res.get('loader_time_sec', 0.0)
        
        print(f"{name:<12} | {smoke_str:<10} | {loader_str:<15} | {total_time:.4f}s")
        if not (res['smoke_passed'] and res['loader_passed']):
            all_success = False
            
    print("=" * 60)
    if all_success:
        print("  ALL 9 MODELS SUCCESSFULLY PASSED INTEGRATION TESTS!")
    else:
        print("  SOME TESTS FAILED. PLEASE CHECK THE ERRORS ABOVE.")
    print("=" * 60)


if __name__ == '__main__':
    main()
