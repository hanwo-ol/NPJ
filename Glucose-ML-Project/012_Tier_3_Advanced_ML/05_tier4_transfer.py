"""
Tier 4 — Cross-Dataset Transfer Learning Experiments
=====================================================
Phase 1: Within Variation  (random seed ×5)
Phase 2: Between Variation  (LODO as-is, pairwise transfer matrix)
Phase 3: Post-hoc Classification Metrics (Range, Hypo, Trend)
Phase 4: Similarity-based analysis

※ Feature harmonization: 모든 데이터셋의 feature dimension이 다르므로
   global feature set (CGM lookback 6 + derived 14 = 공통 20차원)만 사용.
   이는 "global feature"로서 전이학습에 필요한 최소 공통 집합이다.

Usage:
    cd 012_Tier_3_Advanced_ML
    python 05_tier4_transfer.py
"""

import gc
import json
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    f1_score, cohen_kappa_score, confusion_matrix
)
import lightgbm as lgb
from tier3_data_utils import (
    discover_datasets, log, mape, downcast_to_float32,
    get_numeric_cols, build_windows_with_features
)
from tqdm import tqdm

# ══════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════

N_SEEDS = 5
N_JOBS = 16
LGBM_DEFAULT = {
    'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
    'num_leaves': 127, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'n_jobs': N_JOBS, 'verbose': -1,
}

OUT_DIR = Path("tier4_results")
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════
# 1. Global-Feature-Only Dataset Loader
#    (모든 데이터셋 공통: glucose_lookback×6 + derived 14 = 20 dim)
# ══════════════════════════════════════════════════════

def load_dataset_global_features(dset_path, train_ratio=0.7, val_ratio=0.15,
                                  seed=42):
    """
    Load dataset with ONLY global features (glucose lookback + derived).
    Covariates (insulin, carbs etc.) are excluded for cross-dataset compatibility.
    Uses patient-level temporal split (70/15/15).
    """
    ds = dset_path.name
    aug = dset_path / f"{ds}-time-augmented"
    pfiles = sorted(aug.glob("*.csv"))
    if not pfiles:
        return None

    # We only use glucose_value_mg_dl as the base feature
    feat_cols = ['glucose_value_mg_dl']

    all_X, all_Y = [], []
    patient_ids = []

    for pid, pf in enumerate(pfiles):
        df = pd.read_csv(pf, low_memory=False)
        if 'timestamp' not in df.columns or 'glucose_value_mg_dl' not in df.columns:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'glucose_value_mg_dl'])

        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        Xm, Y, fnames = build_windows_with_features(df, feat_cols)
        if len(Y) < 10:
            continue

        all_X.append(Xm)
        all_Y.append(Y)
        patient_ids.extend([pid] * len(Y))

    if not all_X:
        return None

    X = np.vstack(all_X)
    Y = np.concatenate(all_Y)
    patient_ids = np.array(patient_ids)

    # Patient-level temporal split
    unique_pids = np.unique(patient_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_pids)

    n_train = int(len(unique_pids) * train_ratio)
    n_val = int(len(unique_pids) * val_ratio)

    train_pids = set(unique_pids[:n_train])
    val_pids = set(unique_pids[n_train:n_train + n_val])
    test_pids = set(unique_pids[n_train + n_val:])

    train_mask = np.isin(patient_ids, list(train_pids))
    val_mask = np.isin(patient_ids, list(val_pids))
    test_mask = np.isin(patient_ids, list(test_pids))

    # Keep only global features (first 6 glucose lookback cols + 14 derived)
    # build_windows_with_features with only glucose_value_mg_dl yields:
    #   cols 0..5 = glucose_value_mg_dl_t-5 .. t-0
    #   cols 6..19 = derived (Velocity, Accel, ..., tod_sin, tod_cos)
    n_global = 20  # 6 lookback + 14 derived
    if X.shape[1] > n_global:
        X = X[:, :n_global]

    # Feature names (global set)
    global_fnames = [f'glucose_t-{5-i}' for i in range(6)] + [
        'Velocity', 'Acceleration', 'Window_Mean', 'Window_Std',
        'TIR', 'TAR', 'TBR', 'LBGI', 'HBGI', 'Window_AUC',
        'Jerk', 'SD1', 'tod_sin', 'tod_cos'
    ]

    return {
        'name': ds,
        'X_train': downcast_to_float32(X[train_mask]),
        'y_train': Y[train_mask],
        'X_val': downcast_to_float32(X[val_mask]),
        'y_val': Y[val_mask],
        'X_test': downcast_to_float32(X[test_mask]),
        'y_test': Y[test_mask],
        'feature_names': global_fnames[:min(n_global, X.shape[1])],
        'n_subjects': len(unique_pids),
        'n_windows': len(Y),
    }


# ══════════════════════════════════════════════════════
# 2. Metrics
# ══════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    """Regression + post-hoc classification metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    # Post-hoc: Range classification (TBR/TIR/TAR)
    def to_range(y):
        return np.where(y < 70, 0, np.where(y > 180, 2, 1))

    r_true = to_range(y_true)
    r_pred = to_range(y_pred)
    kappa_range = cohen_kappa_score(r_true, r_pred)
    f1_range = f1_score(r_true, r_pred, average='macro', zero_division=0)

    # Post-hoc: Hypo prediction (binary)
    h_true = (y_true < 70).astype(int)
    h_pred = (y_pred < 70).astype(int)
    hypo_prev = h_true.mean()
    if h_true.sum() > 0:
        kappa_hypo = cohen_kappa_score(h_true, h_pred)
        # Sensitivity = TP / (TP + FN)
        tp = ((h_true == 1) & (h_pred == 1)).sum()
        fn = ((h_true == 1) & (h_pred == 0)).sum()
        hypo_sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    else:
        kappa_hypo = np.nan
        hypo_sens = np.nan

    # Post-hoc: Trend classification (5-class based on delta)
    # delta = y_pred - current glucose (glucose_t-0 is at index 5 of X)
    # We compute from y_true/y_pred difference to baseline
    delta = y_pred - y_true  # This is the prediction error, not trend
    # For trend, we'd need X[:, 5] but we don't have it here
    # So we skip trend in this function and compute it separately

    return {
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'MAPE': round(mape_val, 2),
        'Kappa_Range': round(kappa_range, 4),
        'F1_Range': round(f1_range, 4),
        'Kappa_Hypo': round(kappa_hypo, 4) if not np.isnan(kappa_hypo) else None,
        'Hypo_Sens': round(hypo_sens, 4) if not np.isnan(hypo_sens) else None,
        'Hypo_Prev': round(hypo_prev, 4),
    }


def compute_trend_metrics(y_true, y_pred, current_glucose):
    """Trend classification: 5-class based on Δ = predicted - current."""
    delta_true = y_true - current_glucose
    delta_pred = y_pred - current_glucose

    def to_trend(d):
        return np.where(d < -30, 0,
               np.where(d < -15, 1,
               np.where(d <= 15, 2,
               np.where(d <= 30, 3, 4))))

    t_true = to_trend(delta_true)
    t_pred = to_trend(delta_pred)

    kappa_trend = cohen_kappa_score(t_true, t_pred, weights='quadratic')
    f1_trend = f1_score(t_true, t_pred, average='macro', zero_division=0)

    return {
        'Kappa_Trend': round(kappa_trend, 4),
        'F1_Trend': round(f1_trend, 4),
    }


# ══════════════════════════════════════════════════════
# 3. Dataset Similarity Index
# ══════════════════════════════════════════════════════

def compute_dataset_stats(data):
    """Extract summary statistics for similarity computation."""
    y = data['y_train']
    return {
        'mean_glucose': np.mean(y),
        'std_glucose': np.std(y),
        'cv_glucose': np.std(y) / np.mean(y) if np.mean(y) > 0 else 0,
        'tir': np.mean((y >= 70) & (y <= 180)),
        'tar': np.mean(y > 180),
        'tbr': np.mean(y < 70),
        'median_glucose': np.median(y),
        'iqr_glucose': np.percentile(y, 75) - np.percentile(y, 25),
        'n_subjects': data['n_subjects'],
        'n_windows': data['n_windows'],
    }


def compute_similarity_matrix(stats_list):
    """Compute pairwise dataset similarity from summary stats."""
    n = len(stats_list)
    keys = ['mean_glucose', 'std_glucose', 'cv_glucose', 'tir', 'tar', 'tbr',
            'median_glucose', 'iqr_glucose']

    # Normalize features
    vals = np.array([[s[k] for k in keys] for s in stats_list])
    mins = vals.min(axis=0)
    maxs = vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    norm = (vals - mins) / ranges

    # Manhattan distance → similarity
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.sum(np.abs(norm[i] - norm[j]))
            sim[i, j] = np.exp(-dist / len(keys))

    return sim


# ══════════════════════════════════════════════════════
# PHASE 1: Within Variation (Random Seed ×5)
# ══════════════════════════════════════════════════════

def run_within_variation(datasets_paths):
    """Run same model with 5 different random seeds on each dataset."""
    log("\n" + "=" * 70)
    log("PHASE 1: Within Variation (5 seeds × LightGBM)")
    log("=" * 70)

    results = []

    for dpath in datasets_paths:
        ds = dpath.name
        seed_metrics = []

        for seed in range(N_SEEDS):
            data = load_dataset_global_features(dpath, seed=seed * 17 + 42)
            if data is None:
                break

            model = lgb.LGBMRegressor(**{**LGBM_DEFAULT, 'random_state': seed * 17 + 42})
            model.fit(data['X_train'], data['y_train'],
                      eval_set=[(data['X_val'], data['y_val'])],
                      callbacks=[lgb.log_evaluation(0)])
            pred = model.predict(data['X_test'])

            m = compute_metrics(data['y_test'], pred)
            # Trend metrics
            current_g = data['X_test'][:, 5]  # glucose_t-0
            t_m = compute_trend_metrics(data['y_test'], pred, current_g)
            m.update(t_m)
            m['seed'] = seed
            m['dataset'] = ds
            seed_metrics.append(m)

            log(f"  [{ds}] seed={seed}  RMSE={m['RMSE']:.3f}  "
                f"Kappa_Range={m['Kappa_Range']:.4f}  Kappa_Trend={m['Kappa_Trend']:.4f}")

            del model, data, pred
            gc.collect()

        if seed_metrics:
            rmses = [m['RMSE'] for m in seed_metrics]
            log(f"  [{ds}] Within SD(RMSE) = {np.std(rmses):.4f} mg/dL")
        results.extend(seed_metrics)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "phase1_within_variation.csv", index=False)
    log(f"\n  Phase 1 saved -> {OUT_DIR / 'phase1_within_variation.csv'}")
    return df


# ══════════════════════════════════════════════════════
# PHASE 2: Between Variation (LODO → pairwise as-is)
# ══════════════════════════════════════════════════════

def run_between_variation(datasets_paths):
    """
    For each dataset pair (i → j):
      Train on dataset i → Evaluate on dataset j's test set (as-is, no adaptation).
    Also includes self-evaluation (i → i) as diagonal baseline.
    """
    log("\n" + "=" * 70)
    log("PHASE 2: Between Variation (Pairwise As-Is Transfer)")
    log("=" * 70)

    # Pre-load all datasets (global features only, seed=42)
    loaded = {}
    stats_list = []
    valid_names = []

    for dpath in datasets_paths:
        data = load_dataset_global_features(dpath, seed=42)
        if data is None:
            log(f"  [SKIP] {dpath.name}")
            continue
        loaded[data['name']] = data
        stats_list.append(compute_dataset_stats(data))
        valid_names.append(data['name'])

    log(f"\n  Loaded {len(loaded)} datasets with global features (20 dim)")

    # Similarity matrix
    sim_matrix = compute_similarity_matrix(stats_list)
    sim_df = pd.DataFrame(sim_matrix, index=valid_names, columns=valid_names)
    sim_df.to_csv(OUT_DIR / "similarity_matrix.csv")
    log(f"  Similarity matrix saved -> {OUT_DIR / 'similarity_matrix.csv'}")

    # Dataset stats
    stats_df = pd.DataFrame(stats_list, index=valid_names)
    stats_df.to_csv(OUT_DIR / "dataset_stats.csv")

    # Pairwise transfer
    results = []
    n = len(valid_names)
    total_pairs = n * n
    pair_count = 0

    for src_name in valid_names:
        src = loaded[src_name]

        # Train on source
        t0 = time.time()
        model = lgb.LGBMRegressor(**{**LGBM_DEFAULT, 'random_state': 42})
        model.fit(src['X_train'], src['y_train'],
                  eval_set=[(src['X_val'], src['y_val'])],
                  callbacks=[lgb.log_evaluation(0)])
        train_time = time.time() - t0

        for tgt_name in valid_names:
            pair_count += 1
            tgt = loaded[tgt_name]

            pred = model.predict(tgt['X_test'])
            m = compute_metrics(tgt['y_test'], pred)

            # Trend
            current_g = tgt['X_test'][:, 5]
            t_m = compute_trend_metrics(tgt['y_test'], pred, current_g)
            m.update(t_m)

            m['source'] = src_name
            m['target'] = tgt_name
            m['is_self'] = src_name == tgt_name
            m['similarity'] = sim_matrix[valid_names.index(src_name),
                                         valid_names.index(tgt_name)]
            m['train_time'] = round(train_time, 1)
            results.append(m)

            marker = "★" if src_name == tgt_name else " "
            log(f"  [{pair_count}/{total_pairs}] {marker} {src_name} → {tgt_name}  "
                f"RMSE={m['RMSE']:.2f}  Kappa={m['Kappa_Range']:.3f}  "
                f"sim={m['similarity']:.3f}")

        del model
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "phase2_pairwise_transfer.csv", index=False)
    log(f"\n  Phase 2 saved -> {OUT_DIR / 'phase2_pairwise_transfer.csv'}")

    # Cleanup loaded data
    del loaded
    gc.collect()

    return df, sim_df


# ══════════════════════════════════════════════════════
# PHASE 3: Report Generation
# ══════════════════════════════════════════════════════

def generate_report(within_df, between_df, sim_df):
    """Generate markdown report combining all phases."""
    log("\n" + "=" * 70)
    log("PHASE 3: Report Generation")
    log("=" * 70)

    report = []
    report.append("# Tier 4 — Cross-Dataset Transfer Learning Results\n")
    report.append(f"> Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    # ── Phase 1 Summary ──
    report.append("## 1. Within Variation (Random Seed ×5)\n\n")
    report.append("> 같은 데이터셋에서 random seed만 바꿨을 때의 성능 변동\n\n")

    within_summary = within_df.groupby('dataset').agg({
        'RMSE': ['mean', 'std', 'min', 'max'],
        'Kappa_Range': ['mean', 'std'],
        'Kappa_Trend': ['mean', 'std'],
    }).round(4)
    within_summary.columns = ['RMSE_mean', 'RMSE_std', 'RMSE_min', 'RMSE_max',
                               'Kappa_Range_mean', 'Kappa_Range_std',
                               'Kappa_Trend_mean', 'Kappa_Trend_std']
    report.append(within_summary.to_markdown())
    report.append("\n\n")

    avg_within_sd = within_summary['RMSE_std'].mean()
    report.append(f"> **평균 Within SD(RMSE) = {avg_within_sd:.4f} mg/dL**\n\n")

    # ── Phase 2 Summary ──
    report.append("---\n\n## 2. Between Variation (Pairwise As-Is Transfer)\n\n")
    report.append("> Source에서 학습한 LightGBM 모델을 Target에 그대로 적용한 결과\n\n")

    # Transfer matrix (RMSE)
    report.append("### 2.1 Transfer Matrix (RMSE, mg/dL)\n\n")
    pivot_rmse = between_df.pivot(index='source', columns='target', values='RMSE')
    report.append(pivot_rmse.round(2).to_markdown())
    report.append("\n\n")

    # Transfer matrix (Kappa_Range)
    report.append("### 2.2 Transfer Matrix (Cohen's Kappa — Range Classification)\n\n")
    pivot_kappa = between_df.pivot(index='source', columns='target', values='Kappa_Range')
    report.append(pivot_kappa.round(3).to_markdown())
    report.append("\n\n")

    # Transfer matrix (Kappa_Trend)
    report.append("### 2.3 Transfer Matrix (Weighted Kappa — Trend Classification)\n\n")
    pivot_trend = between_df.pivot(index='source', columns='target', values='Kappa_Trend')
    report.append(pivot_trend.round(3).to_markdown())
    report.append("\n\n")

    # Diagonal vs Off-diagonal
    self_df = between_df[between_df['is_self']]
    cross_df = between_df[~between_df['is_self']]

    report.append("### 2.4 Self vs Cross-Dataset Performance\n\n")
    report.append("| Metric | Self (diagonal) | Cross (off-diagonal) | Δ |\n")
    report.append("|:---|:---:|:---:|:---:|\n")
    for metric in ['RMSE', 'MAE', 'Kappa_Range', 'Kappa_Trend']:
        s = self_df[metric].mean()
        c = cross_df[metric].mean()
        d = c - s
        report.append(f"| {metric} | {s:.3f} | {c:.3f} | {d:+.3f} |\n")
    report.append("\n")

    avg_between_sd = cross_df.groupby('target')['RMSE'].std().mean()
    report.append(f"> **평균 Between SD(RMSE) = {avg_between_sd:.3f} mg/dL**\n\n")
    report.append(f"> Within SD = {avg_within_sd:.4f} vs Between SD = {avg_between_sd:.3f} → "
                  f"**Between/Within ratio = {avg_between_sd / avg_within_sd:.1f}x**\n\n")

    # Similarity vs RMSE correlation
    report.append("### 2.5 Similarity vs Transfer Performance\n\n")
    cross_df_copy = cross_df.copy()
    if len(cross_df_copy) > 5:
        corr = cross_df_copy[['similarity', 'RMSE']].corr().iloc[0, 1]
        report.append(f"> Pearson correlation (Similarity vs RMSE): **r = {corr:.3f}**\n")
        report.append(f"> {'유사한 데이터셋끼리 전이 성능이 좋다' if corr < -0.3 else '유사도와 전이 성능의 관계가 불분명하다'}\n\n")

    # Similarity matrix
    report.append("### 2.6 Dataset Similarity Matrix\n\n")
    report.append(sim_df.round(3).to_markdown())
    report.append("\n\n")

    # ── Within vs Between Comparison ──
    report.append("---\n\n## 3. Within vs Between Variation 비교 (Key Result)\n\n")
    report.append("| 변동 원인 | 평균 SD(RMSE) mg/dL | 해석 |\n")
    report.append("|:---|:---:|:---|\n")
    report.append(f"| HPO (Tier 3) | 0.004 | 무의미 |\n")
    report.append(f"| Random Seed (Phase 1) | {avg_within_sd:.4f} | {'무의미' if avg_within_sd < 0.5 else '유의미'} |\n")
    report.append(f"| Cross-Dataset (Phase 2) | {avg_between_sd:.3f} | {'실질적 변동' if avg_between_sd > 1.0 else '소규모 변동'} |\n")
    report.append(f"| **Between/Within Ratio** | **{avg_between_sd / max(avg_within_sd, 0.001):.1f}x** | - |\n")
    report.append("\n")

    out_path = OUT_DIR / "05_Tier4_Transfer_Results.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    log(f"  Report saved -> {out_path}")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    t_start = time.time()
    datasets = discover_datasets()
    log(f"Discovered {len(datasets)} datasets")
    log(f"LightGBM {lgb.__version__}")
    log(f"Output: {OUT_DIR.resolve()}\n")

    # Phase 1: Within Variation
    within_df = run_within_variation(datasets)

    # Phase 2: Between Variation
    between_df, sim_df = run_between_variation(datasets)

    # Phase 3: Report
    generate_report(within_df, between_df, sim_df)

    elapsed = (time.time() - t_start) / 3600
    log(f"\nTotal elapsed: {elapsed:.2f}h")
    log("Done.")


if __name__ == "__main__":
    main()
