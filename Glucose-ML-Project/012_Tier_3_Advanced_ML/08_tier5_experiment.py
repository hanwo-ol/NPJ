"""
Tier 5 — Full Nalmpatian Framework with Q/M + Local Features + Patient ID
==========================================================================
Tier 4의 3가지 한계를 해결하고, 6가지 방법을 비교.

Methods:
  T4_self_20      : Tier 4 baseline (global 20 dim only)
  T5_self_26      : + Q/M static (age, sex, HbA1c, cohort) → 26 dim
  T5_self_full    : + Local dynamic features (insulin, HR, nutrition)
  T5_pool_ft_26   : 11개 풀링 Global Model(26 dim) → FT
  T5_nalm_patient : Patient-Level Resampling(정확한 환자 ID) → FT
  T5_specialized  : Global → Specialized(full features) → FT on target

Usage:
    cd 012_Tier_3_Advanced_ML
    python 08_tier5_experiment.py
"""

import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, f1_score
import lightgbm as lgb
from tier3_data_utils import discover_datasets, log, mape, downcast_to_float32

# Import Tier 4 metrics
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("phase2", Path(__file__).parent / "05_tier4_transfer.py")
phase2 = module_from_spec(spec)
spec.loader.exec_module(phase2)
compute_metrics = phase2.compute_metrics
compute_trend_metrics = phase2.compute_trend_metrics
compute_dataset_stats = phase2.compute_dataset_stats
compute_similarity_matrix = phase2.compute_similarity_matrix

# Import Tier 5 loader
from tier5_data_utils import load_dataset_tier5, DATASET_COHORT

# Also import Tier 4 loader for baseline comparison
load_dataset_global_features = phase2.load_dataset_global_features

N_JOBS = 16
LGBM_DEFAULT = {
    'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
    'num_leaves': 127, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'n_jobs': N_JOBS, 'verbose': -1, 'random_state': 42,
}
FINETUNE_PARAMS = {
    'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05,
    'num_leaves': 127, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'n_jobs': N_JOBS, 'verbose': -1, 'random_state': 42,
}

OUT_DIR = Path("tier5_results")
OUT_DIR.mkdir(exist_ok=True)


def eval_predictions(y_true, y_pred, X_test):
    """Compute all metrics. X_test used for glucose_t-0 (col 5) for trend."""
    m = compute_metrics(y_true, y_pred)
    current_g = X_test[:, 5]  # glucose_t-0
    t_m = compute_trend_metrics(y_true, y_pred, current_g)
    m.update(t_m)
    return m


def main():
    t_start = time.time()
    datasets_paths = discover_datasets()
    log(f"Discovered {len(datasets_paths)} datasets")

    # ── Load all datasets (both Tier 4 and Tier 5) ──
    t4_loaded = {}   # Tier 4: global 20 dim
    t5_loaded = {}   # Tier 5: global 26 + local
    stats_list = []
    valid_names = []

    for dpath in datasets_paths:
        # Tier 4 baseline
        data4 = load_dataset_global_features(dpath, seed=42)
        if data4 is None:
            continue

        # Tier 5 extended
        data5 = load_dataset_tier5(dpath, seed=42, include_local=True)
        if data5 is None:
            continue

        t4_loaded[data4['name']] = data4
        t5_loaded[data5['name']] = data5
        stats_list.append(compute_dataset_stats(data4))
        valid_names.append(data4['name'])

    n = len(valid_names)
    log(f"Loaded {n} datasets (Tier 4: 20dim, Tier 5: 26+local)")

    for name in valid_names:
        d5 = t5_loaded[name]
        log(f"  {name:20s}  global={d5['n_global_dim']}  local={d5['n_local_dim']}  "
            f"subjects={d5['n_subjects']}  cohort={d5['cohort']}")

    # Similarity matrix
    sim_matrix = compute_similarity_matrix(stats_list)

    # ═══════════════════════════════════════════════
    # LODO Loop: 6 methods × 12 targets
    # ═══════════════════════════════════════════════
    all_results = []

    for mi, tgt_name in enumerate(valid_names):
        log(f"\n{'='*60}")
        log(f"[{mi+1}/{n}] TARGET = {tgt_name} (cohort={t5_loaded[tgt_name]['cohort']})")
        log(f"{'='*60}")

        tgt4 = t4_loaded[tgt_name]
        tgt5 = t5_loaded[tgt_name]
        source_names = [s for s in valid_names if s != tgt_name]
        source_indices = [valid_names.index(s) for s in source_names]
        tgt_idx = valid_names.index(tgt_name)

        raw_sims = np.array([sim_matrix[si, tgt_idx] for si in source_indices])
        norm_sims = raw_sims / raw_sims.sum()

        # ── Method 1: T4_self_20 (Tier 4 baseline) ──
        t0 = time.time()
        m1 = lgb.LGBMRegressor(**LGBM_DEFAULT)
        m1.fit(tgt4['X_train'], tgt4['y_train'],
               eval_set=[(tgt4['X_val'], tgt4['y_val'])],
               callbacks=[lgb.log_evaluation(0)])
        pred1 = m1.predict(tgt4['X_test'])
        r1 = eval_predictions(tgt4['y_test'], pred1, tgt4['X_test'])
        r1['method'] = 'T4_self_20'
        r1['target'] = tgt_name
        r1['n_features'] = 20
        r1['time'] = round(time.time() - t0, 1)
        all_results.append(r1)
        log(f"  [T4_self_20]     RMSE={r1['RMSE']:.2f}  (20 dim)")
        del m1; gc.collect()

        # ── Method 2: T5_self_26 (Global 26 dim = 20+Q/M+cohort) ──
        t0 = time.time()
        m2 = lgb.LGBMRegressor(**LGBM_DEFAULT)
        m2.fit(tgt5['X_global_train'], tgt5['y_train'],
               eval_set=[(tgt5['X_global_val'], tgt5['y_val'])],
               callbacks=[lgb.log_evaluation(0)])
        pred2 = m2.predict(tgt5['X_global_test'])
        r2 = eval_predictions(tgt5['y_test'], pred2, tgt5['X_global_test'])
        r2['method'] = 'T5_self_26'
        r2['target'] = tgt_name
        r2['n_features'] = tgt5['n_global_dim']
        r2['time'] = round(time.time() - t0, 1)
        all_results.append(r2)
        delta_qm = r2['RMSE'] - r1['RMSE']
        log(f"  [T5_self_26]     RMSE={r2['RMSE']:.2f}  (26 dim, Q/M effect: {delta_qm:+.2f})")
        del m2; gc.collect()

        # ── Method 3: T5_self_full (Global 26 + Local dynamic) ──
        if tgt5['n_local_dim'] > 0:
            t0 = time.time()
            m3 = lgb.LGBMRegressor(**LGBM_DEFAULT)
            m3.fit(tgt5['X_full_train'], tgt5['y_train'],
                   eval_set=[(tgt5['X_full_val'], tgt5['y_val'])],
                   callbacks=[lgb.log_evaluation(0)])
            pred3 = m3.predict(tgt5['X_full_test'])
            r3 = eval_predictions(tgt5['y_test'], pred3, tgt5['X_full_test'])
            r3['method'] = 'T5_self_full'
            r3['target'] = tgt_name
            r3['n_features'] = tgt5['n_global_dim'] + tgt5['n_local_dim']
            r3['time'] = round(time.time() - t0, 1)
            all_results.append(r3)
            delta_local = r3['RMSE'] - r2['RMSE']
            log(f"  [T5_self_full]   RMSE={r3['RMSE']:.2f}  "
                f"({r3['n_features']} dim, local effect: {delta_local:+.2f})")
            del m3; gc.collect()
        else:
            # No local features, copy self_26
            r3 = r2.copy()
            r3['method'] = 'T5_self_full'
            all_results.append(r3)
            log(f"  [T5_self_full]   RMSE={r3['RMSE']:.2f}  (no local features, = self_26)")

        # ── Method 4: T5_pool_ft_26 (Pool 11 sources global 26 → FT) ──
        t0 = time.time()
        pool_X = np.vstack([t5_loaded[s]['X_global_train'] for s in source_names])
        pool_y = np.concatenate([t5_loaded[s]['y_train'] for s in source_names])
        pool_vX = np.vstack([t5_loaded[s]['X_global_val'] for s in source_names])
        pool_vy = np.concatenate([t5_loaded[s]['y_val'] for s in source_names])

        global_model = lgb.LGBMRegressor(**LGBM_DEFAULT)
        global_model.fit(pool_X, pool_y,
                         eval_set=[(pool_vX, pool_vy)],
                         callbacks=[lgb.log_evaluation(0)])
        t_global = time.time() - t0

        # Save and fine-tune on target
        gm_path = OUT_DIR / f"_tmp_global_{tgt_name}.txt"
        global_model.booster_.save_model(str(gm_path))

        ft_model = lgb.LGBMRegressor(**FINETUNE_PARAMS)
        ft_model.fit(tgt5['X_global_train'], tgt5['y_train'],
                     eval_set=[(tgt5['X_global_val'], tgt5['y_val'])],
                     callbacks=[lgb.log_evaluation(0)],
                     init_model=str(gm_path))
        pred4 = ft_model.predict(tgt5['X_global_test'])
        r4 = eval_predictions(tgt5['y_test'], pred4, tgt5['X_global_test'])
        r4['method'] = 'T5_pool_ft_26'
        r4['target'] = tgt_name
        r4['n_features'] = tgt5['n_global_dim']
        r4['time'] = round(t_global + (time.time() - t0 - t_global), 1)
        all_results.append(r4)
        log(f"  [T5_pool_ft_26]  RMSE={r4['RMSE']:.2f}  (pooled 26 dim → FT)")

        del pool_X, pool_y, pool_vX, pool_vy, ft_model
        gc.collect()

        # ── Method 5: T5_nalm_patient (Patient-Level Resampling with real IDs) ──
        t0 = time.time()
        patient_X_list, patient_y_list = [], []

        for si, sname in enumerate(source_names):
            src = t5_loaded[sname]
            # True patient-level selection using preserved patient_ids
            src_pids = np.unique(src['patient_ids_train'])
            n_select = max(1, int(len(src_pids) * norm_sims[si] * 5))
            n_select = min(n_select, len(src_pids))

            rng = np.random.RandomState(42 + si)
            selected_pids = rng.choice(src_pids, size=n_select, replace=False)

            # Select ALL windows from chosen patients (preserving time structure)
            pid_mask = np.isin(src['patient_ids_train'], selected_pids)
            patient_X_list.append(src['X_global_train'][pid_mask])
            patient_y_list.append(src['y_train'][pid_mask])

        patient_X = np.vstack(patient_X_list)
        patient_y = np.concatenate(patient_y_list)

        # Global model → patient-resampled data → target FT
        patient_model = lgb.LGBMRegressor(**FINETUNE_PARAMS)
        patient_model.fit(patient_X, patient_y,
                          eval_set=[(tgt5['X_global_val'], tgt5['y_val'])],
                          callbacks=[lgb.log_evaluation(0)],
                          init_model=str(gm_path))

        pm_path = OUT_DIR / f"_tmp_patient_{tgt_name}.txt"
        patient_model.booster_.save_model(str(pm_path))

        ft_patient = lgb.LGBMRegressor(**{**FINETUNE_PARAMS, 'n_estimators': 50})
        ft_patient.fit(tgt5['X_global_train'], tgt5['y_train'],
                       eval_set=[(tgt5['X_global_val'], tgt5['y_val'])],
                       callbacks=[lgb.log_evaluation(0)],
                       init_model=str(pm_path))
        pred5 = ft_patient.predict(tgt5['X_global_test'])
        r5 = eval_predictions(tgt5['y_test'], pred5, tgt5['X_global_test'])
        r5['method'] = 'T5_nalm_patient'
        r5['target'] = tgt_name
        r5['n_features'] = tgt5['n_global_dim']
        r5['time'] = round(t_global + (time.time() - t0), 1)
        all_results.append(r5)
        log(f"  [T5_nalm_pat]    RMSE={r5['RMSE']:.2f}  "
            f"(patient-level resampling, {len(patient_y):,} windows)")

        del patient_X, patient_y, patient_model, ft_patient
        gc.collect()

        # ── Method 6: T5_specialized (Global→Specialized with local features) ──
        # Nalmpatian 2-stage: Global Model predicts → Specialized Model learns residuals
        # init_model won't work (dim mismatch), so we use init_score (residual learning)
        if tgt5['n_local_dim'] > 0:
            t0 = time.time()
            # Step 1: Get Global Model's predictions as initial scores
            global_pred_train = global_model.predict(tgt5['X_global_train'])
            global_pred_val   = global_model.predict(tgt5['X_global_val'])
            global_pred_test  = global_model.predict(tgt5['X_global_test'])

            # Step 2: Train Specialized Model on FULL features (global+local)
            #         with init_score = Global Model's predictions
            #         → Specialized Model only learns the RESIDUAL (what Global missed)
            train_ds = lgb.Dataset(tgt5['X_full_train'], tgt5['y_train'],
                                   init_score=global_pred_train, free_raw_data=False)
            val_ds   = lgb.Dataset(tgt5['X_full_val'], tgt5['y_val'],
                                   init_score=global_pred_val, free_raw_data=False)

            spec_params = {
                'objective': 'regression', 'metric': 'rmse',
                'max_depth': 8, 'learning_rate': 0.05,
                'num_leaves': 127, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'n_jobs': N_JOBS, 'verbose': -1, 'seed': 42,
            }
            spec_booster = lgb.train(
                spec_params, train_ds, num_boost_round=100,
                valid_sets=[val_ds], callbacks=[lgb.log_evaluation(0)]
            )
            # Final prediction = Global prediction + Specialized residual
            spec_residual = spec_booster.predict(tgt5['X_full_test'])
            pred6 = global_pred_test + spec_residual

            r6 = eval_predictions(tgt5['y_test'], pred6, tgt5['X_full_test'])
            r6['method'] = 'T5_specialized'
            r6['target'] = tgt_name
            r6['n_features'] = tgt5['n_global_dim'] + tgt5['n_local_dim']
            r6['time'] = round(t_global + (time.time() - t0), 1)
            all_results.append(r6)
            log(f"  [T5_specialized] RMSE={r6['RMSE']:.2f}  "
                f"(global+residual, {r6['n_features']} dim)")
            del spec_booster, train_ds, val_ds; gc.collect()
        else:
            r6 = r4.copy()
            r6['method'] = 'T5_specialized'
            all_results.append(r6)
            log(f"  [T5_specialized] RMSE={r6['RMSE']:.2f}  (no local feats, = pool_ft)")

        # Cleanup temp files
        for p in [gm_path, pm_path]:
            if p.exists():
                p.unlink()
        del global_model; gc.collect()

        # Summary
        log(f"\n  --- {tgt_name} Summary ---")
        tgt_results = [r for r in all_results if r['target'] == tgt_name]
        for r in tgt_results:
            log(f"    {r['method']:20s}  RMSE={r['RMSE']:.2f}  "
                f"Kappa_R={r['Kappa_Range']:.3f}  feat={r['n_features']}")

    # ═══════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_DIR / "tier5_results.csv", index=False)

    log(f"\n{'='*60}")
    log(f"FINAL SUMMARY — Tier 5 Method Comparison")
    log(f"{'='*60}")

    methods = ['T4_self_20', 'T5_self_26', 'T5_self_full',
               'T5_pool_ft_26', 'T5_nalm_patient', 'T5_specialized']

    summary = df.groupby('method').agg({
        'RMSE': 'mean', 'MAE': 'mean',
        'Kappa_Range': 'mean', 'Kappa_Trend': 'mean',
    }).round(3)
    summary = summary.reindex([m for m in methods if m in summary.index])
    log(f"\n{summary.to_string()}")

    # Q/M effect
    if 'T4_self_20' in summary.index and 'T5_self_26' in summary.index:
        qm_effect = summary.loc['T5_self_26', 'RMSE'] - summary.loc['T4_self_20', 'RMSE']
        log(f"\n  Q/M Effect (self_26 - self_20): {qm_effect:+.3f} mg/dL")

    log(f"\n  Results saved -> {OUT_DIR / 'tier5_results.csv'}")
    elapsed = (time.time() - t_start) / 60
    log(f"\nTotal elapsed: {elapsed:.1f} min")
    log("Done.")


if __name__ == "__main__":
    main()
