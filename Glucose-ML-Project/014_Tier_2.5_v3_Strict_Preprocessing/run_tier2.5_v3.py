import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

from tier2_5_v3_config import Tier2_5_v3_Config
from tier2_5_v3_data_utils import (
    resolve_glucose_files, load_and_resolve_schema,
    apply_strict_filters, detect_sampling_rate_min,
    build_windows_no_interpolation
)

# ─── Sampling Rate Groups ───
# Defined here (not in config) since these are run-level reporting categories.
# Window sizes are fixed (LOOKBACK_STEPS=3, PREDICTION_STEPS=3) for all groups.
# Groups exist purely to separate performance reporting by sensor type.
RATE_GROUPS = {
    '1min':  {'rate_min': 0.5,  'rate_max': 2.0,  'pred_time_min': 3},
    '5min':  {'rate_min': 2.0,  'rate_max': 8.0,  'pred_time_min': 15},
    '15min': {'rate_min': 8.0,  'rate_max': 20.0, 'pred_time_min': 45},
}
MIN_SUBJECTS_PER_GROUP = 5


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    valid = y_true != 0
    return np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100 if valid.sum() > 0 else np.nan


def assign_rate_group(rate_min: float) -> str | None:
    for gname, g in RATE_GROUPS.items():
        if g['rate_min'] <= rate_min < g['rate_max']:
            return gname
    return None


def subject_split_3way(n: int, seed: int):
    """
    Subject 단위 3-way 분리 (Temporal Integrity Rule).
    배정은 무작위로 결정하며, Subject 내부 시계열 순서는 보존한다.
    비율: GlobalConfig.TRAIN_RATIO / VAL_RATIO / TEST_RATIO
    """
    from tier2_5_v3_config import Tier2_5_v3_Config
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * Tier2_5_v3_Config.TRAIN_RATIO)
    n_val   = int(n * Tier2_5_v3_Config.VAL_RATIO)
    train_idx = idx[:n_train].tolist()
    val_idx   = idx[n_train:n_train + n_val].tolist()
    test_idx  = idx[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                       feature_names, group_name, pred_time_min, lg):
    lg(f"\n  Training LightGBM [{group_name}] (predicts {pred_time_min} min ahead) ...")
    model = LGBMRegressor(
        n_estimators=500,
        random_state=Tier2_5_v3_Config.SEED,
        n_jobs=Tier2_5_v3_Config.N_JOBS
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[__import__('lightgbm').early_stopping(50, verbose=False),
                   __import__('lightgbm').log_evaluation(0)]
    )
    preds = model.predict(X_test)
    rmse     = np.sqrt(mean_squared_error(y_test, preds))
    mae      = mean_absolute_error(y_test, preds)
    mape_val = mape(y_test, preds)
    lg(f"  RMSE: {rmse:.2f} mg/dL  |  MAE: {mae:.2f}  |  MAPE: {mape_val:.2f}%")
    top5 = np.argsort(model.feature_importances_)[::-1][:5]
    top5_str = ", ".join(f"{feature_names[i]}({model.feature_importances_[i]})" for i in top5)
    lg(f"  Top-5 Features: {top5_str}")
    return {'group': group_name, 'pred_min': pred_time_min, 'rmse': rmse, 'mae': mae, 'mape': mape_val}


def main():
    log_path = Tier2_5_v3_Config.OUTPUT_DIR / "tier2.5_v3_execution.log"
    log_file = open(log_path, "w", encoding='utf-8')

    def lg(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    lg("=== Tier 2.5 v3 (Final): 3-Step Prediction, Per-Rate-Group ===")
    lg(f"Lookback Steps   : {Tier2_5_v3_Config.LOOKBACK_STEPS}")
    lg(f"Prediction Steps : {Tier2_5_v3_Config.PREDICTION_STEPS}")
    lg(f"Excluded Datasets: {Tier2_5_v3_Config.EXCLUDED_DATASETS}")

    datasets_dir = Tier2_5_v3_Config.DATA_ROOT
    all_datasets = sorted([d for d in os.listdir(datasets_dir)
                           if os.path.isdir(os.path.join(datasets_dir, d))])

    group_data: dict[str, list] = {g: [] for g in RATE_GROUPS}
    group_feature_names: dict[str, list] = {}

    g_excluded = 0
    g_files_processed = 0
    g_files_unclassified = 0
    g_dropped_range = 0
    g_dropped_roc   = 0
    g_skipped_windows = 0

    for ds_name in tqdm(all_datasets, desc="Processing Datasets"):
        # Excluded datasets
        if ds_name in Tier2_5_v3_Config.EXCLUDED_DATASETS:
            g_excluded += 1
            continue

        ds_path = Path(datasets_dir) / ds_name
        csv_files = resolve_glucose_files(ds_path, ds_name)
        if not csv_files:
            continue

        for csv_path in csv_files:
            df, target_col = load_and_resolve_schema(csv_path)
            if df is None or target_col is None:
                continue

            df, stats = apply_strict_filters(df, target_col)
            g_dropped_range += stats['dropped_range']
            g_dropped_roc   += stats['dropped_roc']

            if 'timestamp' not in df.columns:
                continue
            ts = pd.to_datetime(df['timestamp'], errors='coerce').values
            detected_rate = detect_sampling_rate_min(ts)
            if detected_rate <= 0:
                continue

            group_name = assign_rate_group(detected_rate)
            if group_name is None:
                g_files_unclassified += 1
                continue

            # Windows use fixed LOOKBACK_STEPS/PREDICTION_STEPS;
            # detected_rate is passed only for gap threshold computation.
            X, Y, f_names, w_stats = build_windows_no_interpolation(
                df, target_col, detected_rate
            )
            g_skipped_windows += w_stats['skipped_windows']

            if X.shape[0] > 0:
                group_data[group_name].append((X, Y))
                if group_name not in group_feature_names:
                    group_feature_names[group_name] = f_names
                g_files_processed += 1

    lg("\n=== Data Extraction Complete ===")
    lg(f"Datasets Scanned       : {len(all_datasets)}")
    lg(f"Datasets Excluded      : {g_excluded} {Tier2_5_v3_Config.EXCLUDED_DATASETS}")
    lg(f"Files Yielding Data    : {g_files_processed}")
    lg(f"Files Unclassified     : {g_files_unclassified}")
    lg(f"Range Exclusions       : {g_dropped_range:,}")
    lg(f"RoC Exclusions         : {g_dropped_roc:,}")
    lg(f"Dropped Windows        : {g_skipped_windows:,}")
    for gname, subjects in group_data.items():
        n_w = sum(len(y) for _, y in subjects)
        pred_min = RATE_GROUPS[gname]['pred_time_min']
        lg(f"  [{gname}] subjects={len(subjects)}, windows={n_w:,}, predicts {pred_min} min ahead")

    # ─── Per-Group Training ───
    lg("\n=== Per-Rate-Group Training (Fixed 3-step prediction) ===")
    all_results = []

    for group_name, subjects in group_data.items():
        n_subj = len(subjects)
        pred_min = RATE_GROUPS[group_name]['pred_time_min']

        if n_subj < MIN_SUBJECTS_PER_GROUP:
            lg(f"\n  [{group_name}] Skipped: only {n_subj} subjects (min={MIN_SUBJECTS_PER_GROUP})")
            continue

        train_idx, val_idx, test_idx = subject_split_3way(
            n_subj, Tier2_5_v3_Config.SEED
        )

        X_train = np.vstack([subjects[i][0] for i in train_idx])
        y_train = np.concatenate([subjects[i][1] for i in train_idx])
        X_val   = np.vstack([subjects[i][0] for i in val_idx])
        y_val   = np.concatenate([subjects[i][1] for i in val_idx])
        X_test  = np.vstack([subjects[i][0] for i in test_idx])
        y_test  = np.concatenate([subjects[i][1] for i in test_idx])

        f_names = group_feature_names.get(group_name, [])
        lg(f"\n  [{group_name}] train_subj={len(train_idx)} ({len(y_train):,} win) | "
           f"val_subj={len(val_idx)} ({len(y_val):,} win) | "
           f"test_subj={len(test_idx)} ({len(y_test):,} win)")

        result = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                                    f_names, group_name, pred_min, lg)
        all_results.append(result)

    # ─── Summary ───
    lg("\n=== Performance Summary ===")
    lg(f"{'Group':<10} {'Pred (min)':>12} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    lg("-" * 56)
    for r in all_results:
        lg(f"{r['group']:<10} {r['pred_min']:>12} {r['rmse']:>10.2f} {r['mae']:>10.2f} {r['mape']:>10.2f}%")

    log_file.close()


if __name__ == "__main__":
    main()
