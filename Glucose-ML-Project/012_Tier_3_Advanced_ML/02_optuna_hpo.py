"""
Tier 3 — Step 02: Optuna Hyperparameter Optimization
======================================================
LightGBM / CatBoost를 대상으로 Bayesian HPO를 수행한다.

설계 근거: 03_optuna_hpo_plan.md v4
  - Patient-wise 70/15/15 시계열 분할 (전체 병합 후 단순 분할 금지)
  - Single-objective: minimize RMSE (MAE/MAPE는 user_attr로 기록)
  - 데이터셋 크기별 n_trials / timeout 동적 할당
  - LightGBM: LightGBMPruningCallback (round-level pruning)
  - CatBoost: 네이티브 커스텀 Pruning Callback (C컴파일러 미필요)
  - CatBoost depth 상한: 데이터셋 규모별 동적 제한 (OOM 방지)
  - vstack 직후 원본 리스트 del + gc.collect() (메모리 스파이크 방지)
  - 최종 평가: Best Trial 모델 객체를 X_test에 직접 적용 (Tier 3)

Usage:
    cd 012_Tier_3_Advanced_ML
    python 02_optuna_hpo.py
"""

import gc
import json
import math
import time
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

import optuna
import lightgbm as lgb
import catboost as cb
from optuna.integration import LightGBMPruningCallback

from tier3_data_utils import (
    get_numeric_cols, build_windows_with_features,
    mape, log, downcast_to_float32, discover_datasets
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════
# 1. Global Config
# ═══════════════════════════════════════════════════

N_JOBS = 16          # 모델 학습 병렬 코어 (20스레드 중 16개)
TOTAL_BUDGET_SEC = 36_000  # 전체 오버나이트 예산: 10시간

# 데이터셋 규모 분류 (Window 수 기준)
SCALE_THRESHOLDS = {
    'small':  500_000,     # < 50만
    'medium': 5_000_000,   # 50만 ~ 500만
    # >= 500만 → 'large'
}

# 규모별 n_trials (모델당)
N_TRIALS_MAP = {'small': 60, 'medium': 30, 'large': 6}

# 규모별 per-dataset timeout (초, 모델 2개 합산)
TIMEOUT_MAP = {'small': 600, 'medium': 1_800, 'large': 4_800}

# CatBoost depth 상한 (Oblivious Tree O(2^d) 폭발 방지)
CATBOOST_MAX_DEPTH = {'small': 11, 'medium': 10, 'large': 8}

RESULTS_JSON = Path("hpo_best_params.json")
RESULTS_MD   = Path("04_HPO_Results.md")


# ═══════════════════════════════════════════════════
# 2. Dataset Scale Helper
# ═══════════════════════════════════════════════════

def get_scale(n_windows: int) -> str:
    if n_windows < SCALE_THRESHOLDS['small']:
        return 'small'
    elif n_windows < SCALE_THRESHOLDS['medium']:
        return 'medium'
    return 'large'


# ═══════════════════════════════════════════════════
# 3. Patient-wise 70/15/15 Split
#    (전체 병합 후 단순 분할 절대 금지 — Data Leakage)
# ═══════════════════════════════════════════════════

def load_dataset_split(dset_path):
    """
    환자別 독립 70/15/15 시계열 분할 후 세트 단위 Concatenate.

    Returns dict with X_train/val/test, y_train/val/test,
    feature_names, n_windows, scale — or None if insufficient data.
    """
    ds = dset_path.name
    aug = dset_path / f"{ds}-time-augmented"
    pfiles = sorted(aug.glob("*.csv"))
    if not pfiles:
        return None

    log(f"\n{'=' * 60}")
    log(f"[{ds}] {len(pfiles)} subjects — loading 70/15/15 split...")

    feat_cols = get_numeric_cols(pfiles)

    # 환자별 분할 후 각 세트 리스트에 누적
    tr_X, tr_y = [], []
    va_X, va_y = [], []
    te_X, te_y = [], []
    feature_names = []

    for pf in pfiles:
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
        if len(Y) < 20:
            continue
        if not feature_names:
            feature_names = fnames

        n = len(Y)
        t1 = int(n * 0.70)   # 각 환자 내 70% 지점
        t2 = int(n * 0.85)   # 각 환자 내 85% 지점

        tr_X.append(Xm[:t1]);  tr_y.append(Y[:t1])
        va_X.append(Xm[t1:t2]); va_y.append(Y[t1:t2])
        te_X.append(Xm[t2:]);  te_y.append(Y[t2:])

    if not tr_X:
        log(f"  [SKIP] Insufficient data")
        return None

    # ── vstack + 즉시 메모리 해제 (스파이크 방지) ──
    X_train = downcast_to_float32(np.vstack(tr_X))
    y_train = np.concatenate(tr_y).astype(np.float32)
    del tr_X, tr_y;  gc.collect()

    X_val = downcast_to_float32(np.vstack(va_X))
    y_val = np.concatenate(va_y).astype(np.float32)
    del va_X, va_y;  gc.collect()

    X_test = downcast_to_float32(np.vstack(te_X))
    y_test = np.concatenate(te_y).astype(np.float32)
    del te_X, te_y;  gc.collect()

    n_total = len(y_train) + len(y_val) + len(y_test)
    scale = get_scale(n_total)

    log(f"  Scale={scale}  Total={n_total:,}  "
        f"Train={len(y_train):,} Val={len(y_val):,} Test={len(y_test):,}")
    log(f"  Feature Dim={X_train.shape[1]}")

    return {
        'name': ds,
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
        'feature_names': feature_names,
        'n_windows': n_total,
        'feature_dim': X_train.shape[1],
        'scale': scale,
    }


# ═══════════════════════════════════════════════════
# 4. CatBoost Custom Pruning Callback
#    (optuna-integration[catboost] Windows 설치 불가 대체)
# ═══════════════════════════════════════════════════

class OptunaCatBoostCallback:
    """
    CatBoost 네이티브 TrainingCallback — 라운드마다 val RMSE를
    Optuna trial에 보고하여 가망 없는 trial을 조기 종료한다.

    ※ 핵심 패턴: CatBoost C++ 래퍼가 Python 예외를 가로채므로
      콜백 내부에서 raise TrialPruned()를 하면 trial이 'failed'로
      잘못 처리된다. 대신 self.pruned 플래그를 세우고 return False로
      학습을 멈춘 뒤, objective 함수에서 TrialPruned를 raise한다.
    """
    def __init__(self, trial: optuna.Trial, eval_name: str = "validation"):
        self.trial = trial
        self.eval_name = eval_name
        self.pruned = False   # objective에서 확인할 플래그

    def after_iteration(self, info) -> bool:
        iteration = info.iteration
        metrics = info.metrics
        pool_key = self.eval_name if self.eval_name in metrics else list(metrics)[-1]
        val_rmse = list(metrics[pool_key].get("RMSE", [0]))[-1]

        self.trial.report(float(val_rmse), step=iteration)
        if self.trial.should_prune():
            self.pruned = True
            return False  # CatBoost 학습 중단 (예외 raise 금지)
        return True


# ═══════════════════════════════════════════════════
# 5. Objective Functions
# ═══════════════════════════════════════════════════

def objective_lgbm(trial: optuna.Trial, data: dict) -> float:
    X_tr, y_tr = data['X_train'], data['y_train']
    X_va, y_va = data['X_val'],   data['y_val']

    params = {
        'num_leaves':        trial.suggest_int('num_leaves', 31, 511),
        'max_depth':         trial.suggest_int('max_depth', 6, 16),
        'learning_rate':     trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        # 고정값
        'n_estimators': 1000,
        'n_jobs': N_JOBS,
        'random_state': 42,
        'verbose': -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='rmse',
        callbacks=[
            LightGBMPruningCallback(trial, "rmse"),
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ]
    )

    pred = model.predict(X_va)
    val_rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
    val_mae  = float(mean_absolute_error(y_va, pred))
    val_mape = float(mape(y_va, pred))

    trial.set_user_attr('val_MAE',  round(val_mae,  3))
    trial.set_user_attr('val_MAPE', round(val_mape, 2))
    trial.set_user_attr('actual_n_estimators', model.best_iteration_)

    del model
    gc.collect()
    return val_rmse


def objective_catboost(trial: optuna.Trial, data: dict) -> float:
    X_tr, y_tr = data['X_train'], data['y_train']
    X_va, y_va = data['X_val'],   data['y_val']
    scale = data['scale']

    max_d = CATBOOST_MAX_DEPTH[scale]
    params = {
        'depth':               trial.suggest_int('depth', 5, max_d),
        'learning_rate':       trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1.0, 20.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count':        trial.suggest_int('border_count', 64, 255),
        'min_data_in_leaf':    trial.suggest_int('min_data_in_leaf', 5, 100),
        # 고정값
        'iterations': 1000,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 50,
        'thread_count': N_JOBS,
        'random_seed': 42,
        'verbose': 0,
    }

    pruning_cb = OptunaCatBoostCallback(trial, eval_name="validation")

    train_pool = cb.Pool(X_tr, y_tr)
    val_pool   = cb.Pool(X_va, y_va)

    model = cb.CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        callbacks=[pruning_cb],
    )

    # 콜백이 pruning 플래그를 세웠으면 여기서 TrialPruned raise
    if pruning_cb.pruned:
        del model, train_pool, val_pool
        gc.collect()
        raise optuna.TrialPruned()

    pred = model.predict(X_va)
    val_rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
    val_mae  = float(mean_absolute_error(y_va, pred))
    val_mape = float(mape(y_va, pred))

    trial.set_user_attr('val_MAE',  round(val_mae,  3))
    trial.set_user_attr('val_MAPE', round(val_mape, 2))
    trial.set_user_attr('actual_n_estimators', model.best_iteration_)

    del model, train_pool, val_pool
    gc.collect()
    return val_rmse


# ═══════════════════════════════════════════════════
# 6. Final Evaluation (Best Trial → X_test)
# ═══════════════════════════════════════════════════

def evaluate_on_test(best_params: dict, model_name: str,
                     data: dict, actual_n: int) -> dict:
    """Best Trial 파라미터로 모델 재생성 → X_test 추론."""
    X_tr, y_tr = data['X_train'], data['y_train']
    X_va, y_va = data['X_val'],   data['y_val']
    X_te, y_te = data['X_test'],  data['y_test']
    scale = data['scale']

    # Train+Val 합산은 하지 않음 (Tier 3 설계 결정: Best Trial 직접 사용)
    # 단, best trial의 실제 n_estimators로 재생성해야 과적합 없이 재현 가능
    if model_name == 'LightGBM':
        params = dict(best_params)
        params['n_estimators'] = actual_n  # early stopping으로 확정된 횟수
        params['n_jobs'] = N_JOBS
        params['random_state'] = 42
        params['verbose'] = -1
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.log_evaluation(0),
                              lgb.early_stopping(50, verbose=False)])
        pred = model.predict(X_te)

    else:  # CatBoost
        params = dict(best_params)
        params['iterations'] = actual_n
        params['eval_metric'] = 'RMSE'
        params['thread_count'] = N_JOBS
        params['random_seed'] = 42
        params['verbose'] = 0
        model = cb.CatBoostRegressor(**params)
        model.fit(cb.Pool(X_tr, y_tr), eval_set=cb.Pool(X_va, y_va))
        pred = model.predict(X_te)

    test_rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    test_mae  = float(mean_absolute_error(y_te, pred))
    test_mape = float(mape(y_te, pred))

    del model
    gc.collect()
    return {
        'test_RMSE': round(test_rmse, 3),
        'test_MAE':  round(test_mae,  3),
        'test_MAPE': round(test_mape, 2),
    }


# ═══════════════════════════════════════════════════
# 7. Study Runner
# ═══════════════════════════════════════════════════

def run_study(model_name: str, objective_fn, data: dict,
              n_trials: int, timeout_sec: int) -> dict | None:
    """
    Optuna study를 생성하고 최적화를 실행한 뒤
    best trial 정보를 딕셔너리로 반환.
    """
    ds = data['name']
    log(f"  [{model_name}] n_trials={n_trials}, timeout={timeout_sec}s")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=20
    )
    sampler = optuna.samplers.TPESampler(
        seed=42, multivariate=True, n_startup_trials=10
    )
    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        sampler=sampler,
    )

    t0 = time.time()
    try:
        study.optimize(
            lambda trial: objective_fn(trial, data),
            n_trials=n_trials,
            timeout=timeout_sec,
            n_jobs=1,
            show_progress_bar=False,
            catch=(Exception,),
        )
    except KeyboardInterrupt:
        log(f"  [{model_name}] Interrupted — using best so far")

    elapsed = time.time() - t0
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed:
        log(f"  [{model_name}] No completed trials!")
        return None

    best = study.best_trial
    val_rmse = round(best.value, 3)
    val_mae  = best.user_attrs.get('val_MAE',  float('nan'))
    val_mape = best.user_attrs.get('val_MAPE', float('nan'))
    actual_n = best.user_attrs.get('actual_n_estimators', 300)
    retrain_n = math.ceil(actual_n * 1.15)

    log(f"  [{model_name}] Best trial #{best.number}  "
        f"val_RMSE={val_rmse}  val_MAE={val_mae}  val_MAPE={val_mape}%  "
        f"n_est={actual_n}  ({elapsed:.0f}s, {len(completed)} completed)")

    # ── Test set 최종 평가 ──
    test_metrics = evaluate_on_test(best.params, model_name, data, actual_n)
    log(f"  [{model_name}] Test → RMSE={test_metrics['test_RMSE']}  "
        f"MAE={test_metrics['test_MAE']}  MAPE={test_metrics['test_MAPE']}%")

    return {
        'best_trial_id':        best.number,
        'n_completed':          len(completed),
        'elapsed_sec':          round(elapsed),
        'val_RMSE':             val_rmse,
        'val_MAE':              val_mae,
        'val_MAPE':             val_mape,
        'test_RMSE':            test_metrics['test_RMSE'],
        'test_MAE':             test_metrics['test_MAE'],
        'test_MAPE':            test_metrics['test_MAPE'],
        'actual_n_estimators':  actual_n,
        'retrain_n_estimators': retrain_n,
        'params':               best.params,
    }


# ═══════════════════════════════════════════════════
# 8. Main Pipeline
# ═══════════════════════════════════════════════════

def main():
    start_time = time.time()
    datasets = discover_datasets()
    log(f"Discovered {len(datasets)} datasets.")
    log(f"Total budget: {TOTAL_BUDGET_SEC // 3600}h  N_JOBS={N_JOBS}")

    all_results = {
        'metadata': {
            'tuning_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'total_budget_sec': TOTAL_BUDGET_SEC,
            'validation_strategy': 'patient-wise 70/15/15 time-ordered split',
            'final_eval_strategy': 'best_trial_model_direct (Tier 3)',
        },
        'datasets': {}
    }

    summary_rows = []

    for dset_path in datasets:
        data = load_dataset_split(dset_path)
        if data is None:
            continue

        ds      = data['name']
        scale   = data['scale']
        n_total = data['n_windows']
        n_trials  = N_TRIALS_MAP[scale]
        # 각 모델에 절반씩 timeout 배분
        per_model_timeout = TIMEOUT_MAP[scale] // 2

        ds_result = {
            'n_windows':   n_total,
            'feature_dim': data['feature_dim'],
            'scale':       scale,
        }

        for model_name, obj_fn in [
            ('LightGBM', objective_lgbm),
            ('CatBoost', objective_catboost),
        ]:
            result = run_study(model_name, obj_fn, data,
                               n_trials, per_model_timeout)
            if result:
                ds_result[model_name] = result
                summary_rows.append({
                    'Dataset':       ds,
                    'Scale':         scale,
                    'Model':         model_name,
                    'n_trials_done': result['n_completed'],
                    'Val_RMSE':      result['val_RMSE'],
                    'Test_RMSE':     result['test_RMSE'],
                    'Test_MAE':      result['test_MAE'],
                    'Test_MAPE%':    result['test_MAPE'],
                    'actual_n_est':  result['actual_n_estimators'],
                    'elapsed_s':     result['elapsed_sec'],
                })

        all_results['datasets'][ds] = ds_result

        # 데이터셋 사용 완료 → 즉시 메모리 회수
        del data
        gc.collect()

    # ── JSON 저장 ──
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\nBest params saved → {RESULTS_JSON}")

    # ── Markdown 요약 ──
    df = pd.DataFrame(summary_rows)
    if not df.empty:
        # 데이터셋별 Best Model 강조
        best_idx = df.groupby('Dataset')['Test_RMSE'].idxmin()
        df['Best?'] = '—'
        df.loc[best_idx, 'Best?'] = '⭐'

    with open(RESULTS_MD, 'w', encoding='utf-8') as f:
        f.write("# Tier 3 — Optuna HPO 결과 요약\n\n")
        f.write(f"> 튜닝 완료: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  "
                f"| 총 소요: {(time.time() - start_time) / 3600:.1f}h\n\n")
        f.write("## 성능 비교표 (Validation / Test RMSE, mg/dL)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    log(f"\nDone. Results → {RESULTS_MD} / {RESULTS_JSON}")
    log(f"Total elapsed: {(time.time() - start_time) / 3600:.2f}h")


if __name__ == "__main__":
    main()
