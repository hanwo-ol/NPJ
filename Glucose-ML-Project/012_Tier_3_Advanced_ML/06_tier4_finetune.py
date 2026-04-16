"""
Tier 4 — Phase 2b: Fine-Tuning Transfer Experiment
====================================================
as-is (Phase 2) vs fine-tuned (Phase 2b) 비교.
LightGBM의 init_model을 사용하여 source 모델을 target 데이터로 미세 조정.

Yang et al. (2022) 3단계 구조:
  (1) As-is: source 모델을 target에 그대로 적용 (Phase 2에서 완료)
  (2) Fine-tuned: source 모델 + target train 데이터로 추가 학습

Usage:
    cd 012_Tier_3_Advanced_ML
    python 06_tier4_finetune.py
"""

import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, f1_score
import lightgbm as lgb

from tier3_data_utils import log, mape, downcast_to_float32
# Reuse the global-feature loader from Phase 2
from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import functions from Phase 2 script
from importlib.util import spec_from_file_location, module_from_spec
phase2_path = Path(__file__).parent / "05_tier4_transfer.py"
spec = spec_from_file_location("phase2", phase2_path)
phase2 = module_from_spec(spec)
spec.loader.exec_module(phase2)

load_dataset_global_features = phase2.load_dataset_global_features
compute_metrics = phase2.compute_metrics
compute_trend_metrics = phase2.compute_trend_metrics
compute_dataset_stats = phase2.compute_dataset_stats
compute_similarity_matrix = phase2.compute_similarity_matrix


N_JOBS = 16
LGBM_DEFAULT = {
    'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
    'num_leaves': 127, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'n_jobs': N_JOBS, 'verbose': -1,
}
FINETUNE_PARAMS = {
    'n_estimators': 100,  # 추가 100 라운드만
    'max_depth': 8, 'learning_rate': 0.05,  # 낮은 학습률
    'num_leaves': 127, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'n_jobs': N_JOBS, 'verbose': -1,
}

OUT_DIR = Path("tier4_results")
OUT_DIR.mkdir(exist_ok=True)


def main():
    from tier3_data_utils import discover_datasets

    t_start = time.time()
    datasets_paths = discover_datasets()
    log(f"Discovered {len(datasets_paths)} datasets")

    # Load all datasets
    loaded = {}
    valid_names = []
    stats_list = []

    for dpath in datasets_paths:
        data = load_dataset_global_features(dpath, seed=42)
        if data is None:
            log(f"  [SKIP] {dpath.name}")
            continue
        loaded[data['name']] = data
        stats_list.append(compute_dataset_stats(data))
        valid_names.append(data['name'])

    log(f"\n  Loaded {len(loaded)} datasets")

    sim_matrix = compute_similarity_matrix(stats_list)

    # Load Phase 2 as-is results for comparison
    asis_df = pd.read_csv(OUT_DIR / "phase2_pairwise_transfer.csv")

    results = []
    n = len(valid_names)
    total_pairs = n * n
    pair_count = 0

    for src_name in valid_names:
        src = loaded[src_name]

        # Step 1: Train source model
        src_model = lgb.LGBMRegressor(**{**LGBM_DEFAULT, 'random_state': 42})
        src_model.fit(src['X_train'], src['y_train'],
                      eval_set=[(src['X_val'], src['y_val'])],
                      callbacks=[lgb.log_evaluation(0)])

        # Save source model to temp file for init_model
        src_model_path = OUT_DIR / f"_tmp_src_{src_name}.txt"
        src_model.booster_.save_model(str(src_model_path))

        for tgt_name in valid_names:
            pair_count += 1
            tgt = loaded[tgt_name]

            if src_name == tgt_name:
                # Self-evaluation: just use source model directly (same as as-is)
                pred_ft = src_model.predict(tgt['X_test'])
                ft_time = 0
            else:
                # Fine-tune: use target's train data to continue training
                t0 = time.time()
                ft_model = lgb.LGBMRegressor(**{**FINETUNE_PARAMS, 'random_state': 42})
                ft_model.fit(tgt['X_train'], tgt['y_train'],
                             eval_set=[(tgt['X_val'], tgt['y_val'])],
                             callbacks=[lgb.log_evaluation(0)],
                             init_model=str(src_model_path))
                pred_ft = ft_model.predict(tgt['X_test'])
                ft_time = time.time() - t0
                del ft_model

            # Metrics
            m = compute_metrics(tgt['y_test'], pred_ft)
            current_g = tgt['X_test'][:, 5]
            t_m = compute_trend_metrics(tgt['y_test'], pred_ft, current_g)
            m.update(t_m)

            # Get as-is result for comparison
            asis_row = asis_df[(asis_df['source'] == src_name) & (asis_df['target'] == tgt_name)]
            asis_rmse = float(asis_row['RMSE'].values[0]) if len(asis_row) > 0 else np.nan

            m['source'] = src_name
            m['target'] = tgt_name
            m['is_self'] = src_name == tgt_name
            m['similarity'] = sim_matrix[valid_names.index(src_name),
                                         valid_names.index(tgt_name)]
            m['RMSE_asis'] = round(asis_rmse, 3)
            m['Delta_RMSE'] = round(m['RMSE'] - asis_rmse, 3)
            m['finetune_time'] = round(ft_time, 1)
            results.append(m)

            improved = "↓" if m['Delta_RMSE'] < -0.5 else ("↑" if m['Delta_RMSE'] > 0.5 else "→")
            marker = "★" if src_name == tgt_name else " "
            log(f"  [{pair_count}/{total_pairs}] {marker} {src_name} → {tgt_name}  "
                f"as-is={asis_rmse:.2f}  ft={m['RMSE']:.2f}  Δ={m['Delta_RMSE']:+.2f} {improved}")

        # Cleanup
        if src_model_path.exists():
            src_model_path.unlink()
        del src_model
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "phase2b_finetune_transfer.csv", index=False)
    log(f"\n  Phase 2b saved -> {OUT_DIR / 'phase2b_finetune_transfer.csv'}")

    # Summary
    cross = df[~df['is_self']]
    improved_count = (cross['Delta_RMSE'] < -0.5).sum()
    worsened_count = (cross['Delta_RMSE'] > 0.5).sum()
    unchanged_count = len(cross) - improved_count - worsened_count

    log(f"\n  === Fine-tuning Summary ===")
    log(f"  Cross-dataset pairs: {len(cross)}")
    log(f"  Improved (Δ < -0.5): {improved_count} ({improved_count/len(cross)*100:.0f}%)")
    log(f"  Worsened (Δ > +0.5): {worsened_count} ({worsened_count/len(cross)*100:.0f}%)")
    log(f"  Unchanged: {unchanged_count} ({unchanged_count/len(cross)*100:.0f}%)")
    log(f"  Mean ΔRMSE: {cross['Delta_RMSE'].mean():+.3f} mg/dL")
    log(f"  Mean as-is RMSE: {cross['RMSE_asis'].mean():.3f}")
    log(f"  Mean fine-tuned RMSE: {cross['RMSE'].mean():.3f}")

    elapsed = (time.time() - t_start) / 60
    log(f"\nTotal elapsed: {elapsed:.1f} min")
    log("Done.")


if __name__ == "__main__":
    main()
