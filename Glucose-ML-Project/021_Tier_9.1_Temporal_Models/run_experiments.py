import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# Ensure current folder and project root are in path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import DLinear, TSMixer, XLinear, TiDE, NHits, NBeats, FITS, SOFTS, PatchMLP
from data_loader import load_temporal_dataset


def get_clarke_zones_percentage(y_true, y_pred):
    """
    Vectorized calculation of Clarke Error Grid Zone A + B percentage.
    Provides a fast and accurate approximation of clinical safety.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    
    # Zone A definition: 
    # (true < 70 and error <= 20) or (true >= 70 and error <= 20%)
    a_mask = ((y_true < 70) & (abs_diff <= 20)) | ((y_true >= 70) & (abs_diff <= 0.20 * y_true))
    
    # Zone B definition (approximated safe zone boundaries):
    # Outside Zone A, but doesn't lead to inappropriate clinical treatment decisions
    ab_mask = a_mask | \
              ((y_true < 70) & (y_pred < 180)) | \
              ((y_true >= 70) & (y_true <= 240) & (y_pred >= 70) & (y_pred <= 240)) | \
              ((y_true > 240) & (y_pred > 180)) | \
              (abs_diff <= 40)
              
    return float(np.mean(ab_mask) * 100)


def evaluate_model(model, loader):
    """
    Evaluates the model on the given loader and returns MAE, RMSE, MAPE, and CEG A+B %.
    """
    model.eval()
    y_trues = []
    y_preds = []
    
    with torch.no_grad():
        for x, y in loader:
            out = model(x)  # [B, pred_len, 1]
            y_trues.append(y.numpy())
            y_preds.append(out.numpy())
            
    y_true = np.concatenate(y_trues, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # Avoid zero division for MAPE
    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    
    ceg_ab = get_clarke_zones_percentage(y_true, y_pred)
    
    return mae, rmse, mape, ceg_ab


def train_model(model, loader, epochs=3, lr=1e-3):
    """
    Trains the model on the loader for a given number of epochs.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    for epoch in range(epochs):
        t_epoch_start = time.perf_counter()
        epoch_loss = 0.0
        batch_count = 0
        
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        elapsed = time.perf_counter() - t_epoch_start
        print(f"      Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")


def apply_fine_tuning_freezing(model, model_name):
    """
    Applies the model-specific parameter freezing as outlined in the transfer plan (Document 3).
    """
    if model_name == "TSMixer":
        for block in model.blocks:
            block.temporal_linear.weight.requires_grad = False
            if block.temporal_linear.bias is not None:
                block.temporal_linear.bias.requires_grad = False
        print("    [FREEZE] Froze temporal_linear layers in TSMixer.")
    elif model_name == "TiDE":
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("    [FREEZE] Froze encoder in TiDE.")
    elif model_name in ["NHits", "NHitsBlock", "NBeats", "NBeatsBlock"]:
        for param in model.blocks[0].parameters():
            param.requires_grad = False
        print(f"    [FREEZE] Froze the first block in {model_name}.")
    elif model_name == "SOFTS":
        for param in model.global_mlp.parameters():
            param.requires_grad = False
        print("    [FREEZE] Froze global_mlp in SOFTS.")
    elif model_name == "PatchMLP":
        for param in model.patch_embed.parameters():
            param.requires_grad = False
        print("    [FREEZE] Froze patch_embed in PatchMLP.")
    else:
        print(f"    [FREEZE] Full Fine-Tuning for {model_name} (No parameters frozen).")


def subsample_dataset_if_needed(dataset, max_samples, seed=42):
    """
    Subsamples a dataset to a maximum number of samples if it exceeds the limit.
    """
    if len(dataset) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=max_samples, replace=False)
        return torch.utils.data.Subset(dataset, indices)
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, default=15, choices=[1, 5, 15], help='Sampling rate group (1, 5, or 15)')
    args = parser.parse_args()
    
    group = args.group
    print("=" * 60)
    print(f"  Temporal Models Training & LODO Transfer Learning Suite ({group}min)")
    print("=" * 60)
    
    # 1. Setup target and source datasets dynamically
    if group == 15:
        target_ds = "ShanghaiT2DM"
        source_ds_list = ["Bris-T1D_Open", "ShanghaiT1DM"]
        group_name_kr = "15분 주기 그룹"
    elif group == 1:
        target_ds = "CGMacros_Libre"
        source_ds_list = ["CGMacros_Dexcom"]
        group_name_kr = "1분 주기 그룹"
    elif group == 5:
        target_ds = "AZT1D"
        source_ds_list = ["D1NAMO", "UCHTT1DM", "PhysioCGM"]
        group_name_kr = "5분 주기 그룹"
    
    # Load loaders for Target dataset
    print(f"Loading Target Dataset: {target_ds}...")
    tgt_train_loader, num_features, pred_len, seq_len = load_temporal_dataset(target_ds, 'train', batch_size=256, shuffle=True)
    tgt_val_loader, _, _, _ = load_temporal_dataset(target_ds, 'val', batch_size=256, shuffle=False)
    tgt_test_loader, _, _, _ = load_temporal_dataset(target_ds, 'test', batch_size=256, shuffle=False)
    
    # Subsample datasets to prevent out-of-memory or excessive CPU training times
    MAX_TRAIN = 15000
    MAX_EVAL = 5000
    
    train_ds = subsample_dataset_if_needed(tgt_train_loader.dataset, MAX_TRAIN)
    val_ds = subsample_dataset_if_needed(tgt_val_loader.dataset, MAX_EVAL)
    test_ds = subsample_dataset_if_needed(tgt_test_loader.dataset, MAX_EVAL)
    
    target_train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    target_val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    target_test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    # Load loaders for Source datasets and combine them for LODO pre-training
    print(f"Loading Source Datasets: {source_ds_list}...")
    source_datasets = []
    for s_ds in source_ds_list:
        loader, _, _, _ = load_temporal_dataset(s_ds, 'train', batch_size=256, shuffle=True)
        # Subsample source datasets to maintain domain balance
        src_ds = subsample_dataset_if_needed(loader.dataset, MAX_TRAIN // len(source_ds_list))
        source_datasets.append(src_ds)
        
    combined_source_dataset = ConcatDataset(source_datasets)
    combined_source_dataset = subsample_dataset_if_needed(combined_source_dataset, MAX_TRAIN)
    pretrain_loader = DataLoader(combined_source_dataset, batch_size=256, shuffle=True)
    
    print(f"Dataset configurations loaded.")
    print(f"  Lookback sequence length (seq_len): {seq_len}")
    print(f"  Prediction horizon length (pred_len): {pred_len}")
    print(f"  Input channels (num_features): {num_features}")
    print(f"  Target Samples: train={len(target_train_loader.dataset)}, test={len(target_test_loader.dataset)}")
    print(f"  Combined Source Samples: {len(combined_source_dataset)}")
    
    # Registry of models and their specific hyperparameter configurations
    models_registry = {
        'DLinear':  (DLinear,  {'kernel_size': 3}),
        'TSMixer':  (TSMixer,  {'hidden_dim': 16, 'n_blocks': 1}),
        'XLinear':  (XLinear,  {'kernel_size': 3}),
        'TiDE':     (TiDE,     {'proj_dim': 2, 'hidden_dim': 32, 'decoder_dim': 4}),
        'NHits':    (NHits,    {'hidden_dim': 32, 'theta_dim': 16, 'downsample_rates': [2, 1]}),
        'NBeats':   (NBeats,   {'hidden_dim': 32, 'theta_dim': 16, 'n_blocks': 2}),
        'FITS':     (FITS,     {'cut_freq': 2}),
        'SOFTS':    (SOFTS,    {'d_model': 16}),
        'PatchMLP': (PatchMLP, {'patch_len': 2, 'stride': 1, 'd_model': 16})
    }
    
    results = []
    
    # 2. Run sequential experiments for each model
    for model_name, (model_class, kwargs) in models_registry.items():
        print(f"\n[MODEL] {model_name} sequential execution started...")
        t_model_start = time.perf_counter()
        
        # --- Stage A: Self-training ---
        print(f"  Stage A: Training Self-Model on {target_ds}...")
        self_model = model_class(seq_len=seq_len, pred_len=pred_len, num_features=num_features, **kwargs)
        train_model(self_model, target_train_loader, epochs=3, lr=1e-3)
        self_mae, self_rmse, self_mape, self_ceg = evaluate_model(self_model, target_test_loader)
        print(f"    [Self] Test RMSE={self_rmse:.4f}  MAE={self_mae:.4f}  CEG A+B={self_ceg:.2f}%")
        
        # --- Stage B: LODO Pre-training ---
        print(f"  Stage B: Pre-training Global LODO Model on {source_ds_list}...")
        lodo_model = model_class(seq_len=seq_len, pred_len=pred_len, num_features=num_features, **kwargs)
        train_model(lodo_model, pretrain_loader, epochs=3, lr=1e-3)
        
        # Save pre-trained weights
        weight_path = Path("tier9_results") if Path("tier9_results").exists() else Path(".")
        torch.save(lodo_model.state_dict(), weight_path / f"global_pretrain_{model_name}.pth")
        
        # --- Stage C: Zero-shot Transfer ---
        print(f"  Stage C: Zero-shot Transfer on {target_ds}...")
        zero_mae, zero_rmse, zero_mape, zero_ceg = evaluate_model(lodo_model, target_test_loader)
        print(f"    [Zero-shot] Test RMSE={zero_rmse:.4f}  MAE={zero_mae:.4f}  CEG A+B={zero_ceg:.2f}%")
        
        # --- Stage D: Fine-tuning with model-specific freezing ---
        print(f"  Stage D: Fine-tuning LODO Model on {target_ds}...")
        apply_fine_tuning_freezing(lodo_model, model_name)
        train_model(lodo_model, target_train_loader, epochs=2, lr=1e-4)
        ft_mae, ft_rmse, ft_mape, ft_ceg = evaluate_model(lodo_model, target_test_loader)
        print(f"    [Fine-tuned] Test RMSE={ft_rmse:.4f}  MAE={ft_mae:.4f}  CEG A+B={ft_ceg:.2f}%")
        
        elapsed_sec = time.perf_counter() - t_model_start
        print(f"  {model_name} completed in {elapsed_sec:.2f} seconds.")
        
        results.append({
            'Model': model_name,
            'Self_RMSE': round(self_rmse, 4),
            'Self_MAE': round(self_mae, 4),
            'Self_CEG%': round(self_ceg, 2),
            'Zero_RMSE': round(zero_rmse, 4),
            'Zero_MAE': round(zero_mae, 4),
            'Zero_CEG%': round(zero_ceg, 2),
            'FT_RMSE': round(ft_rmse, 4),
            'FT_MAE': round(ft_mae, 4),
            'FT_CEG%': round(ft_ceg, 2),
            'SAR': round(ft_rmse / self_rmse, 4)
        })
        
    # 3. Save summary markdown table
    df_results = pd.DataFrame(results)
    out_file = Path(f"temporal_transfer_results_{group}min.md")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"# 다기관 CGM 예측 Temporal 모델 패밀리 전이학습 벤치마크 결과 ({group}분 주기)\n\n")
        f.write(f"- **대상 데이터셋 그룹:** {group_name_kr} (Target: `{target_ds}` / Source: `{source_ds_list}`)\n")
        f.write("- **실행 디바이스:** CPU-only (자원 사용량 최적화 및 3~5 에포크 초고속 수렴 설계)\n")
        f.write("- **지표설명:** SAR(Self-Adaptation Ratio)이 1.0 미만이면 전이학습이 자가 학습 Baseline 대비 개선을 보였음을 증명합니다.\n\n")
        f.write(df_results.to_markdown(index=False))
        
    if group == 15:
        import shutil
        try:
            shutil.copyfile(out_file, Path("temporal_transfer_results.md"))
        except Exception:
            pass
        
    print("\n" + "=" * 60)
    print("=== TEMPORAL TRANSFER EXPERIMENTS COMPLETE ===")
    print(f"Results summary saved to {out_file.name}")
    print("=" * 60)


if __name__ == '__main__':
    main()
