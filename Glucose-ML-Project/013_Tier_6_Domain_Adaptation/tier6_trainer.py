"""
Tier 6: Ablation Experiment Trainer
===================================
Contains training functions, evaluation logic, and ablation iteration logic.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from tier6_data_utils import load_dataset_tier6
from tier6_transfer_utils import apply_coral, apply_tca
from tier6_config import Tier6Config

sys.path.append(str(Path(__file__).parent.parent / "012_Tier_3_Advanced_ML"))
try:
    from tier3_data_utils import discover_datasets, log, mape
    from tier5_experiment_utils import calculate_kappa
except ImportError:
    print("Warning: Missing Tier 3 utility references.")
    
def train_lgb(X_train, y_train, X_val, y_val, init_model=None):
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    
    gbm = lgb.train(
        Tier6Config.LGBM_PARAMS,
        dtrain,
        num_boost_round=Tier6Config.LGBM_ROUNDS,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(stopping_rounds=Tier6Config.LGBM_EARLY_STOPPING, verbose=False)],
        init_model=init_model
    )
    return gbm

def evaluate(model, X_test, y_test, method_name, target_name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    kappa_trend = calculate_kappa(y_test, preds) if 'calculate_kappa' in globals() else 0.0
    log(f"[{method_name} -> {target_name}] RMSE: {rmse:.3f} | Kappa Trend: {kappa_trend:.3f}")
    return {"rmse": rmse, "kappa": kappa_trend, "method": method_name, "target": target_name}

def execute_ablation(config):
    """
    Executes ablation study based on the config.
    config keys: 'ablation_mode' (A, B, C, D)
    """
    datasets = discover_datasets(Tier6Config.DATA_DIR)
    results = []
    
    log(f"Starting Ablation Mode: {config['ablation_mode']}")

    for dset_path in datasets:
        target_name = dset_path.name
        log(f"\nTarget: {target_name}")

        if config['ablation_mode'] == 'A':
            # Path A: eGMI Imputation vs Tier 5 Median
            data = load_dataset_tier6(dset_path, use_egmi=True, use_umd_local=False)
            if data is not None:
                model = train_lgb(data['X_global_train'], data['y_train'], data['X_global_val'], data['y_val'])
                res = evaluate(model, data['X_global_test'], data['y_test'], "T6_A_eGMI_self", target_name)
                results.append(res)
                
        elif config['ablation_mode'] == 'B':
            # Path B: UMD Virtual Features (No Local)
            data = load_dataset_tier6(dset_path, use_egmi=False, use_umd_local=True)
            if data is not None:
                # Global Model (Without Local)
                model_glb = train_lgb(data['X_global_train'], data['y_train'], data['X_global_val'], data['y_val'])
                # Specialized tuned with UMD Local
                train_init = model_glb.predict(data['X_global_train'])
                val_init = model_glb.predict(data['X_global_val'])
                test_init = model_glb.predict(data['X_global_test'])
                
                dtrain_sp = lgb.Dataset(data['X_local_train'], label=data['y_train'], init_score=train_init)
                dval_sp = lgb.Dataset(data['X_local_val'], label=data['y_val'], init_score=val_init)
                model_sp = lgb.train(Tier6Config.LGBM_SP_PARAMS, 
                                    dtrain_sp, valid_sets=[dval_sp], num_boost_round=Tier6Config.LGBM_SP_ROUNDS,
                                    callbacks=[lgb.early_stopping(Tier6Config.LGBM_SP_EARLY_STOPPING, verbose=False)])
                
                preds = test_init + model_sp.predict(data['X_local_test'])
                rmse = np.sqrt(mean_squared_error(data['y_test'], preds))
                kappa = calculate_kappa(data['y_test'], preds) if 'calculate_kappa' in globals() else 0.0
                log(f"[T6_B_UMD_virtual -> {target_name}] RMSE: {rmse:.3f} | Kappa: {kappa:.3f}")
                results.append({"rmse": rmse, "kappa": kappa, "method": "T6_B_UMD_virtual", "target": target_name})
                
        elif config['ablation_mode'] == 'C':
            # Path C: CORAL Domain Alignment
            data_t = load_dataset_tier6(dset_path, use_egmi=False, use_umd_local=False)
            if data_t is not None:
                source_X_list = []
                for s_dset in datasets:
                    if s_dset != dset_path:
                        s_data = load_dataset_tier6(s_dset, use_egmi=False, use_umd_local=False)
                        if s_data is not None:
                            source_X_list.append(s_data['X_global_train'])
                
                if len(source_X_list) > 0:
                    source_x = np.vstack(source_X_list)
                    # Subsample Source to 100k to prevent massive memory usage during covariance calculation
                    if len(source_x) > 100000:
                        idx = np.random.choice(len(source_x), 100000, replace=False)
                        source_x = source_x[idx]
                else:
                    source_x = data_t['X_global_train'] # fallback
                    
                target_x = data_t['X_global_test']
                aligned_source = apply_coral(source_x, target_x)
                model = train_lgb(aligned_source, data_t['y_train'], data_t['X_global_val'], data_t['y_val'])
                res = evaluate(model, data_t['X_global_test'], data_t['y_test'], "T6_C_CORAL_aligned", target_name)
                results.append(res)
                
        elif config['ablation_mode'] == 'D':
            # Path D: TCA Feature Extraction Projection
            data_t = load_dataset_tier6(dset_path, use_egmi=False, use_umd_local=False)
            if data_t is not None:
                source_X_list = []
                for s_dset in datasets:
                    if s_dset != dset_path:
                        s_data = load_dataset_tier6(s_dset, use_egmi=False, use_umd_local=False)
                        if s_data is not None:
                            source_X_list.append(s_data['X_global_train'])
                
                if len(source_X_list) > 0:
                    source_x = np.vstack(source_X_list)
                    if len(source_x) > 100000:
                        idx = np.random.choice(len(source_x), 100000, replace=False)
                        source_x = source_x[idx]
                else:
                    source_x = data_t['X_global_train'] # fallback
                    
                target_x = data_t['X_global_test']
                aligned_source = apply_tca(source_x, target_x, n_components=Tier6Config.TCA_N_COMPONENTS)
                model = train_lgb(aligned_source, data_t['y_train'], data_t['X_global_val'], data_t['y_val'])
                res = evaluate(model, data_t['X_global_test'], data_t['y_test'], "T6_D_TCA_aligned", target_name)
                results.append(res)
        
    df_res = pd.DataFrame(results)
    Tier6Config.OUT_DIR.mkdir(exist_ok=True)
    out_path = Tier6Config.OUT_DIR / f"tier6_ablation_{config['ablation_mode']}_results.csv"
    if not df_res.empty:
        df_res.to_csv(out_path, index=False)
        log(f"Ablation Complete. Saved to {out_path}")
