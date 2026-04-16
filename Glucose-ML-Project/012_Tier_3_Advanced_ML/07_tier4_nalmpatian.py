"""
Tier 4 — Phase 3: Nalmpatian-Style Transfer Learning
======================================================
Nalmpatian et al. (2025) 프레임워크를 CGM 시계열에 적용.
Step 4(합성 데이터)를 Patient-Level Resampling으로 대체.

Methods compared:
  1. self       — M 자체만으로 학습 (baseline)
  2. asis_best  — 가장 유사한 1개 source 모델을 그대로 적용
  3. ft_single  — 가장 유사한 1개 source → M으로 fine-tune (Phase 2b)
  4. nalm_simple — 11개 source 모델의 유사도 가중 앙상블
  5. nalm_pooled — 11개 풀링 → Global Model → M에 그대로 적용
  6. nalm_pooled_ft — 11개 풀링 → Global Model → M train으로 fine-tune
  7. nalm_patient — Patient-Level Resampling + Global Model + fine-tune

Usage:
    cd 012_Tier_3_Advanced_ML
    python 07_tier4_nalmpatian.py
"""

import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
import lightgbm as lgb
from tier3_data_utils import discover_datasets, log, mape, downcast_to_float32

# Reuse loader from Phase 2
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("phase2", Path(__file__).parent / "05_tier4_transfer.py")
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
    'n_jobs': N_JOBS, 'verbose': -1, 'random_state': 42,
}
FINETUNE_PARAMS = {
    'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05,
    'num_leaves': 127, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'n_jobs': N_JOBS, 'verbose': -1, 'random_state': 42,
}

OUT_DIR = Path("tier4_results")
OUT_DIR.mkdir(exist_ok=True)


def eval_predictions(y_true, y_pred, X_test):
    """Compute all metrics."""
    m = compute_metrics(y_true, y_pred)
    current_g = X_test[:, 5]  # glucose_t-0
    t_m = compute_trend_metrics(y_true, y_pred, current_g)
    m.update(t_m)
    return m


def main():
    t_start = time.time()
    datasets_paths = discover_datasets()
    log(f"Discovered {len(datasets_paths)} datasets")

    # ── Load all datasets ──
    loaded = {}
    stats_list = []
    valid_names = []

    for dpath in datasets_paths:
        data = load_dataset_global_features(dpath, seed=42)
        if data is None:
            continue
        loaded[data['name']] = data
        stats_list.append(compute_dataset_stats(data))
        valid_names.append(data['name'])

    n = len(valid_names)
    log(f"Loaded {n} datasets")

    # Similarity matrix
    sim_matrix = compute_similarity_matrix(stats_list)

    # ── LODO Loop ──
    all_results = []

    for mi, tgt_name in enumerate(valid_names):
        log(f"\n{'='*60}")
        log(f"[{mi+1}/{n}] TARGET = {tgt_name}")
        log(f"{'='*60}")

        tgt = loaded[tgt_name]
        source_names = [s for s in valid_names if s != tgt_name]
        source_indices = [valid_names.index(s) for s in source_names]
        tgt_idx = valid_names.index(tgt_name)

        # Similarity scores (normalized)
        raw_sims = np.array([sim_matrix[si, tgt_idx] for si in source_indices])
        norm_sims = raw_sims / raw_sims.sum()

        # ────────────────────────────────────────────
        # Method 1: self (baseline)
        # ────────────────────────────────────────────
        t0 = time.time()
        model_self = lgb.LGBMRegressor(**LGBM_DEFAULT)
        model_self.fit(tgt['X_train'], tgt['y_train'],
                       eval_set=[(tgt['X_val'], tgt['y_val'])],
                       callbacks=[lgb.log_evaluation(0)])
        pred_self = model_self.predict(tgt['X_test'])
        m_self = eval_predictions(tgt['y_test'], pred_self, tgt['X_test'])
        m_self['method'] = 'self'
        m_self['target'] = tgt_name
        m_self['time'] = round(time.time() - t0, 1)
        all_results.append(m_self)
        log(f"  [self]         RMSE={m_self['RMSE']:.2f}  Kappa={m_self['Kappa_Range']:.3f}")
        del model_self
        gc.collect()

        # ────────────────────────────────────────────
        # Train individual source models (needed for methods 2-4)
        # ────────────────────────────────────────────
        source_models = {}
        for si, sname in enumerate(source_names):
            src = loaded[sname]
            model = lgb.LGBMRegressor(**LGBM_DEFAULT)
            model.fit(src['X_train'], src['y_train'],
                      eval_set=[(src['X_val'], src['y_val'])],
                      callbacks=[lgb.log_evaluation(0)])
            source_models[sname] = model

        # ────────────────────────────────────────────
        # Method 2: asis_best (most similar source, no adaptation)
        # ────────────────────────────────────────────
        best_src_idx = np.argmax(raw_sims)
        best_src_name = source_names[best_src_idx]
        pred_asis = source_models[best_src_name].predict(tgt['X_test'])
        m_asis = eval_predictions(tgt['y_test'], pred_asis, tgt['X_test'])
        m_asis['method'] = 'asis_best'
        m_asis['target'] = tgt_name
        m_asis['best_source'] = best_src_name
        m_asis['time'] = 0
        all_results.append(m_asis)
        log(f"  [asis_best]    RMSE={m_asis['RMSE']:.2f}  (src={best_src_name}, sim={raw_sims[best_src_idx]:.3f})")

        # ────────────────────────────────────────────
        # Method 3: ft_single (fine-tune best source → target)
        # ────────────────────────────────────────────
        t0 = time.time()
        best_model_path = OUT_DIR / f"_tmp_best_{best_src_name}.txt"
        source_models[best_src_name].booster_.save_model(str(best_model_path))
        ft_model = lgb.LGBMRegressor(**{**FINETUNE_PARAMS})
        ft_model.fit(tgt['X_train'], tgt['y_train'],
                     eval_set=[(tgt['X_val'], tgt['y_val'])],
                     callbacks=[lgb.log_evaluation(0)],
                     init_model=str(best_model_path))
        pred_ft = ft_model.predict(tgt['X_test'])
        m_ft = eval_predictions(tgt['y_test'], pred_ft, tgt['X_test'])
        m_ft['method'] = 'ft_single'
        m_ft['target'] = tgt_name
        m_ft['best_source'] = best_src_name
        m_ft['time'] = round(time.time() - t0, 1)
        all_results.append(m_ft)
        log(f"  [ft_single]    RMSE={m_ft['RMSE']:.2f}  (ft from {best_src_name})")
        del ft_model
        if best_model_path.exists():
            best_model_path.unlink()

        # ────────────────────────────────────────────
        # Method 4: nalm_simple (similarity-weighted ensemble)
        # ────────────────────────────────────────────
        t0 = time.time()
        preds_all = np.zeros((len(tgt['X_test']), len(source_names)))
        for si, sname in enumerate(source_names):
            preds_all[:, si] = source_models[sname].predict(tgt['X_test'])
        pred_nalm_simple = preds_all @ norm_sims  # weighted average
        m_nalm_s = eval_predictions(tgt['y_test'], pred_nalm_simple, tgt['X_test'])
        m_nalm_s['method'] = 'nalm_simple'
        m_nalm_s['target'] = tgt_name
        m_nalm_s['time'] = round(time.time() - t0, 1)
        all_results.append(m_nalm_s)
        log(f"  [nalm_simple]  RMSE={m_nalm_s['RMSE']:.2f}  (11-model ensemble)")

        # Also try uniform ensemble for comparison
        pred_uniform = preds_all.mean(axis=1)
        m_uniform = eval_predictions(tgt['y_test'], pred_uniform, tgt['X_test'])
        m_uniform['method'] = 'nalm_uniform'
        m_uniform['target'] = tgt_name
        m_uniform['time'] = 0
        all_results.append(m_uniform)
        log(f"  [nalm_uniform] RMSE={m_uniform['RMSE']:.2f}  (11-model uniform)")

        # Cleanup source models
        for sm in source_models.values():
            del sm
        del source_models, preds_all
        gc.collect()

        # ────────────────────────────────────────────
        # Method 5: nalm_pooled (pool 11 sources → Global Model)
        # ────────────────────────────────────────────
        t0 = time.time()
        pool_X = np.vstack([loaded[s]['X_train'] for s in source_names])
        pool_y = np.concatenate([loaded[s]['y_train'] for s in source_names])
        pool_val_X = np.vstack([loaded[s]['X_val'] for s in source_names])
        pool_val_y = np.concatenate([loaded[s]['y_val'] for s in source_names])

        global_model = lgb.LGBMRegressor(**LGBM_DEFAULT)
        global_model.fit(pool_X, pool_y,
                         eval_set=[(pool_val_X, pool_val_y)],
                         callbacks=[lgb.log_evaluation(0)])
        train_time_global = time.time() - t0

        pred_pooled = global_model.predict(tgt['X_test'])
        m_pooled = eval_predictions(tgt['y_test'], pred_pooled, tgt['X_test'])
        m_pooled['method'] = 'nalm_pooled'
        m_pooled['target'] = tgt_name
        m_pooled['time'] = round(train_time_global, 1)
        all_results.append(m_pooled)
        log(f"  [nalm_pooled]  RMSE={m_pooled['RMSE']:.2f}  ({len(pool_y):,} windows pooled, {train_time_global:.0f}s)")

        del pool_X, pool_y, pool_val_X, pool_val_y
        gc.collect()

        # ────────────────────────────────────────────
        # Method 6: nalm_pooled_ft (Global Model → fine-tune on target)
        # ────────────────────────────────────────────
        t0 = time.time()
        global_model_path = OUT_DIR / f"_tmp_global_{tgt_name}.txt"
        global_model.booster_.save_model(str(global_model_path))

        ft_global = lgb.LGBMRegressor(**FINETUNE_PARAMS)
        ft_global.fit(tgt['X_train'], tgt['y_train'],
                      eval_set=[(tgt['X_val'], tgt['y_val'])],
                      callbacks=[lgb.log_evaluation(0)],
                      init_model=str(global_model_path))
        pred_pooled_ft = ft_global.predict(tgt['X_test'])
        m_pooled_ft = eval_predictions(tgt['y_test'], pred_pooled_ft, tgt['X_test'])
        m_pooled_ft['method'] = 'nalm_pooled_ft'
        m_pooled_ft['target'] = tgt_name
        m_pooled_ft['time'] = round(train_time_global + (time.time() - t0), 1)
        all_results.append(m_pooled_ft)
        log(f"  [nalm_pool_ft] RMSE={m_pooled_ft['RMSE']:.2f}  (Global→FT on target)")

        del ft_global
        gc.collect()

        # ────────────────────────────────────────────
        # Method 7: nalm_patient (Patient-Level Resampling + FT)
        #   Step 4 대안: source에서 유사도 비례로 환자 선택 → 풀링 → 모델 학습
        # ────────────────────────────────────────────
        t0 = time.time()
        TARGET_PATIENTS = 200  # 총 선택할 가상 환자 수
        patient_X_list, patient_y_list = [], []

        for si, sname in enumerate(source_names):
            src = loaded[sname]
            n_patients_from_j = max(1, int(TARGET_PATIENTS * norm_sims[si]))

            # Patient-level: 각 source의 train data를 환자 수 비례로 선택
            # (현재 구조에서 환자 구분이 X에 없으므로, 윈도우 수 비례로 근사)
            n_windows_from_j = int(len(src['y_train']) * norm_sims[si] * 5)
            n_windows_from_j = min(n_windows_from_j, len(src['y_train']))
            n_windows_from_j = max(100, n_windows_from_j)

            rng = np.random.RandomState(42 + si)
            idx = rng.choice(len(src['y_train']), size=n_windows_from_j, replace=False)
            patient_X_list.append(src['X_train'][idx])
            patient_y_list.append(src['y_train'][idx])

        patient_X = np.vstack(patient_X_list)
        patient_y = np.concatenate(patient_y_list)

        # Train on resampled data (using global model as init)
        patient_model = lgb.LGBMRegressor(**FINETUNE_PARAMS)
        patient_model.fit(patient_X, patient_y,
                          eval_set=[(tgt['X_val'], tgt['y_val'])],
                          callbacks=[lgb.log_evaluation(0)],
                          init_model=str(global_model_path))

        # Then fine-tune on target
        patient_model_path = OUT_DIR / f"_tmp_patient_{tgt_name}.txt"
        patient_model.booster_.save_model(str(patient_model_path))

        ft_patient = lgb.LGBMRegressor(**{**FINETUNE_PARAMS, 'n_estimators': 50})
        ft_patient.fit(tgt['X_train'], tgt['y_train'],
                       eval_set=[(tgt['X_val'], tgt['y_val'])],
                       callbacks=[lgb.log_evaluation(0)],
                       init_model=str(patient_model_path))
        pred_patient = ft_patient.predict(tgt['X_test'])

        m_patient = eval_predictions(tgt['y_test'], pred_patient, tgt['X_test'])
        m_patient['method'] = 'nalm_patient'
        m_patient['target'] = tgt_name
        m_patient['time'] = round(train_time_global + (time.time() - t0), 1)
        all_results.append(m_patient)
        log(f"  [nalm_patient] RMSE={m_patient['RMSE']:.2f}  "
            f"(Patient-Level Resampling, {len(patient_y):,} windows → FT)")

        # Cleanup
        for p in [global_model_path, patient_model_path]:
            if p.exists():
                p.unlink()
        del global_model, patient_model, ft_patient, patient_X, patient_y
        gc.collect()

        # ── Summary for this target ──
        log(f"\n  --- {tgt_name} Summary ---")
        target_results = [r for r in all_results if r['target'] == tgt_name]
        for r in target_results:
            log(f"    {r['method']:20s}  RMSE={r['RMSE']:.2f}  "
                f"Kappa_R={r['Kappa_Range']:.3f}  Kappa_T={r['Kappa_Trend']:.3f}")

    # ══════════════════════════════════════════
    # Final Report
    # ══════════════════════════════════════════
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_DIR / "phase3_nalmpatian.csv", index=False)

    log(f"\n{'='*60}")
    log(f"FINAL SUMMARY — Method Comparison")
    log(f"{'='*60}")

    methods = ['self', 'asis_best', 'ft_single', 'nalm_uniform', 'nalm_simple',
               'nalm_pooled', 'nalm_pooled_ft', 'nalm_patient']

    summary = df.groupby('method').agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'Kappa_Range': 'mean',
        'Kappa_Trend': 'mean',
    }).round(3)

    # Reorder
    summary = summary.reindex([m for m in methods if m in summary.index])

    log(f"\n{summary.to_string()}")
    log(f"\n  Results saved -> {OUT_DIR / 'phase3_nalmpatian.csv'}")

    elapsed = (time.time() - t_start) / 60
    log(f"\nTotal elapsed: {elapsed:.1f} min")
    log("Done.")


if __name__ == "__main__":
    main()
