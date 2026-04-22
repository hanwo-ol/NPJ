"""
Tier 6: Centralized Configuration
=================================
Separates fixed hyperparameters, paths, and default settings from execution logic.
"""

from pathlib import Path

class Tier6Config:
    # ---------------------------------------------
    # 1. Directory Structure
    # ---------------------------------------------
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "003_Glucose-ML-collection"
    OUT_DIR = Path(__file__).parent / "tier6_results"
    OUT_DIR.mkdir(exist_ok=True)
    LOG_FILE = OUT_DIR / "tier6_experiment_C_D.log"
    
    # ---------------------------------------------
    # 2. General Settings
    # ---------------------------------------------
    RANDOM_SEED = 42
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    
    # ---------------------------------------------
    # 3. Model Hyperparameters (LightGBM)
    # ---------------------------------------------
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }
    LGBM_ROUNDS = 1000
    LGBM_EARLY_STOPPING = 50
    
    LGBM_SP_PARAMS = {
        'objective': 'regression',
        'learning_rate': 0.01,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }
    LGBM_SP_ROUNDS = 500
    LGBM_SP_EARLY_STOPPING = 20
    
    # ---------------------------------------------
    # 4. Feature Configs
    # ---------------------------------------------
    N_GLOBAL_TS_FEATURES = 20
    TCA_N_COMPONENTS = 10
