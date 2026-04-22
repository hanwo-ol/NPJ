"""
Tier 3 — Step 01: Baseline Boosting Models
=============================================
XGBoost, LightGBM, CatBoost를 동일 Feature space에서 디폴트 하이퍼파라미터로
학습하여 각 모델의 기저 성능(Baseline)을 측정한다.

※ 트리 기반 부스팅 모델은 Feature Scaling이 불필요하므로 StandardScaler를 적용하지 않는다.
※ float32 다운캐스팅으로 메모리 50% 절감.
※ 각 모델 학습 후 gc.collect()로 메모리 즉시 회수.

Usage:
    cd 012_Tier_3_Advanced_ML
    python 01_baseline_boosting.py
"""

import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from tier3_data_utils import (
    discover_datasets, load_dataset, extract_top_features,
    mape, log, downcast_to_float32, GlobalConfig
)


# ═══════════════════════════════════════════════════
# Baseline Hyperparameters (Optuna 튜닝 전 기본값)
# ═══════════════════════════════════════════════════

N_JOBS = GlobalConfig.N_JOBS

XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'n_jobs': N_JOBS,
    'random_state': GlobalConfig.SEED,
    'verbosity': 0,
}

LGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.1,
    'num_leaves': 127,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': N_JOBS,
    'random_state': GlobalConfig.SEED,
    'verbose': -1,
}

CATBOOST_PARAMS = {
    'iterations': 300,
    'depth': 8,
    'learning_rate': 0.1,
    'thread_count': N_JOBS,
    'random_seed': GlobalConfig.SEED,
    'verbose': 0,
}


# ═══════════════════════════════════════════════════
# Training Functions
# ═══════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """XGBoost sklearn API 래퍼로 학습 + 평가."""
    t0 = time.time()
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    pred = model.predict(X_test)
    elapsed = time.time() - t0

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    mape_val = mape(y_test, pred)
    top5 = extract_top_features(model.feature_importances_, feature_names)

    del model
    gc.collect()
    return rmse, mae, mape_val, top5, elapsed


def train_lightgbm(X_train, y_train, X_test, y_test, feature_names):
    """LightGBM sklearn API 래퍼로 학습 + 평가."""
    t0 = time.time()
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.log_evaluation(0)])
    pred = model.predict(X_test)
    elapsed = time.time() - t0

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    mape_val = mape(y_test, pred)
    top5 = extract_top_features(model.feature_importances_ / model.feature_importances_.sum(),
                                feature_names)

    del model
    gc.collect()
    return rmse, mae, mape_val, top5, elapsed


def train_catboost(X_train, y_train, X_test, y_test, feature_names):
    """CatBoost Pool + sklearn 호환 모델로 학습 + 평가."""
    t0 = time.time()
    model = cb.CatBoostRegressor(**CATBOOST_PARAMS)

    train_pool = cb.Pool(X_train, y_train)
    test_pool = cb.Pool(X_test, y_test)

    model.fit(train_pool, eval_set=test_pool, verbose=0)
    pred = model.predict(X_test)
    elapsed = time.time() - t0

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    mape_val = mape(y_test, pred)
    imp = model.get_feature_importance() / model.get_feature_importance().sum()
    top5 = extract_top_features(imp, feature_names)

    del model, train_pool, test_pool
    gc.collect()
    return rmse, mae, mape_val, top5, elapsed


# ═══════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════

def main():
    out_md = Path("02_Baseline_Boosting_Results.md")
    datasets = discover_datasets()

    log(f"Discovered {len(datasets)} datasets.")
    log(f"Models: XGBoost {xgb.__version__}, "
        f"LightGBM {lgb.__version__}, "
        f"CatBoost {cb.__version__}")

    results = []

    for dset in datasets:
        data = load_dataset(dset)
        if data is None:
            continue

        ds = data['name']
        X_tr = data['X_train']
        y_tr = data['y_train']
        X_te = data['X_test']
        y_te = data['y_test']
        fnames = data['feature_names']

        # ── XGBoost ──
        log(f"  [XGBoost] Training (n_estimators={XGB_PARAMS['n_estimators']}, "
            f"max_depth={XGB_PARAMS['max_depth']})...")
        xgb_rmse, xgb_mae, xgb_mape, xgb_top5, xgb_time = \
            train_xgboost(X_tr, y_tr, X_te, y_te, fnames)
        log(f"    RMSE={xgb_rmse:.2f}  MAE={xgb_mae:.2f}  "
            f"MAPE={xgb_mape:.1f}%  ({xgb_time:.0f}s)")

        # ── LightGBM ──
        log(f"  [LightGBM] Training (n_estimators={LGBM_PARAMS['n_estimators']}, "
            f"num_leaves={LGBM_PARAMS['num_leaves']})...")
        lgbm_rmse, lgbm_mae, lgbm_mape, lgbm_top5, lgbm_time = \
            train_lightgbm(X_tr, y_tr, X_te, y_te, fnames)
        log(f"    RMSE={lgbm_rmse:.2f}  MAE={lgbm_mae:.2f}  "
            f"MAPE={lgbm_mape:.1f}%  ({lgbm_time:.0f}s)")

        # ── CatBoost ──
        log(f"  [CatBoost] Training (iterations={CATBOOST_PARAMS['iterations']}, "
            f"depth={CATBOOST_PARAMS['depth']})...")
        cat_rmse, cat_mae, cat_mape, cat_top5, cat_time = \
            train_catboost(X_tr, y_tr, X_te, y_te, fnames)
        log(f"    RMSE={cat_rmse:.2f}  MAE={cat_mae:.2f}  "
            f"MAPE={cat_mape:.1f}%  ({cat_time:.0f}s)")

        # ── Best Model Selection ──
        models_rmse = {
            'XGBoost': (xgb_rmse, xgb_mae, xgb_mape, xgb_top5),
            'LightGBM': (lgbm_rmse, lgbm_mae, lgbm_mape, lgbm_top5),
            'CatBoost': (cat_rmse, cat_mae, cat_mape, cat_top5),
        }
        best_name = min(models_rmse, key=lambda k: models_rmse[k][0])
        best_rmse, best_mae, best_mape, best_top5 = models_rmse[best_name]

        log(f"  ★ Best: {best_name} (RMSE={best_rmse:.2f})")

        results.append({
            'Dataset': ds,
            'Windows': data['n_windows'],
            'Dim': data['feature_dim'],
            'XGB_RMSE': round(xgb_rmse, 2),
            'LGBM_RMSE': round(lgbm_rmse, 2),
            'Cat_RMSE': round(cat_rmse, 2),
            'Best': best_name,
            'Best_RMSE': round(best_rmse, 2),
            'Best_MAE': round(best_mae, 2),
            'Best_MAPE%': round(best_mape, 1),
            'Best_Top5': best_top5,
            'XGB_Time': f"{xgb_time:.0f}s",
            'LGBM_Time': f"{lgbm_time:.0f}s",
            'Cat_Time': f"{cat_time:.0f}s",
        })

        # Memory cleanup between datasets
        del data, X_tr, y_tr, X_te, y_te
        gc.collect()

    # ═══════════════════════════════════════════════════
    # Output Results
    # ═══════════════════════════════════════════════════
    rdf = pd.DataFrame(results)

    # Summary table (compact for readability)
    summary_cols = ['Dataset', 'Windows', 'Dim',
                    'XGB_RMSE', 'LGBM_RMSE', 'Cat_RMSE',
                    'Best', 'Best_RMSE', 'Best_MAE', 'Best_MAPE%']
    detail_cols = ['Dataset', 'Best_Top5', 'XGB_Time', 'LGBM_Time', 'Cat_Time']

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# Tier 3 Baseline Boosting Results\n\n")
        f.write("XGBoost / LightGBM / CatBoost 디폴트 하이퍼파라미터 기저 성능 비교\n\n")
        f.write("## 1. 성능 비교표 (RMSE mg/dL)\n\n")
        f.write(rdf[summary_cols].to_markdown(index=False))
        f.write("\n\n## 2. Best Model Feature Importance & Training Time\n\n")
        f.write(rdf[detail_cols].to_markdown(index=False))
        f.write("\n")

    log(f"\n{'=' * 60}")
    log(f"Done. {len(results)} datasets processed -> {out_md}")


if __name__ == "__main__":
    main()
