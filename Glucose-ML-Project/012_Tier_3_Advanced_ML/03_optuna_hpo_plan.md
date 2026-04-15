# Tier 3: Optuna HPO (Hyperparameter Optimization) 세부 계획서
## (v2 — 베이스라인 실험 기반 현실화 개정판)

본 문서는 `02_1_Baseline_Analysis_Report.md`의 진단 결과를 반영하여, 베이스라인 모델 성능 검토 이후 본격적인 하이퍼파라미터 튜닝(`02_optuna_hpo.py`)을 수행하기 위한 구체적인 방법론과 아키텍처를 정의합니다.

---

## 1. 대상 모델 선정 (Model Selection)

베이스라인 평가(`02_Baseline_Boosting_Results.md`) 근거로, Optuna 튜닝 대상을 **LightGBM**과 **CatBoost** 두 모델로 축소한다.

| 모델 | 선정 이유 | 비고 |
|---|---|---|
| ✅ **LightGBM** | 초대형 데이터셋(GLAM, IOBP2, PEDAP)에서 압도적 속도 + 동등 RMSE | 대형 데이터 전담 |
| ✅ **CatBoost** | 12개 중 8개 데이터셋 1위, 소~중형 데이터에서 범용 최강 | 일반 데이터 전담 |
| ❌ **XGBoost** | 모든 데이터셋에서 RMSE·속도 양면 열세 — 자원 낭비 | 제외 |

---

## 2. 목적 함수 (Objective Function)

### 2.1 단일 목적 최적화 (Single-Objective) — RMSE 기준

> **⚠️ 설계 변경 (v1 대비):** v1에서 제안된 RMSE+MAE+MAPE 3목적 Pareto Front 방식을 **철회**한다.
>
> **이유:**
> 1. 3목적 Pareto Front는 수십~수백 개의 비열등 해를 반환 → 최종 파라미터 선정 기준이 모호해짐
> 2. RMSE와 MAE는 상관관계가 매우 높아 사실상 중복 정보
> 3. 로컬 오버나이트 자원(10시간) 내에서 Pareto 탐색은 단일 목적 대비 훨씬 많은 n_trials를 요구

**채택 방식:**
- **최적화 목표:** `minimize(RMSE)` — 단일 스칼라 목적함수
- **MAE, MAPE:** `trial.set_user_attr()`로 기록만 하여 사후 분석 및 리포트에 활용
- **Multi-Objective 활용 시점:** Tier 4(Cross-dataset generalization)에서 `minimize(RMSE_train) + minimize(RMSE_cross)`의 2목적 구조로 전환 예정

### 2.2 검증 전략 — Patient-wise 70/15/15 시계열 분할 (⚠️ 구현 필수 준수 사항)

#### 치명적 함정: 전체 병합 후 행(Row) 단위 분할 금지

단순히 전체 병합 데이터프레임의 70% 행에서 자르면 다음 문제가 발생한다:
- 환자 A의 데이터가 Train에만 집중, 환자 Z는 Test에만 존재하는 환자 불균형
- 경계면의 환자 시계열이 단절되어 Feature Engineering 결과가 오염됨

#### 올바른 방법: 환자(Patient ID)별 독립 분할 후 세트 단위 Concatenate

```python
# ✅ 올바른 Patient-wise 분할
train_list, val_list, test_list = [], [], []
for patient_file in patient_files:
    X, y = build_windows(patient_file)  # 해당 환자의 시계열 전체
    n = len(y)
    t1 = int(n * 0.70)  # 각 환자 내 70% 지점
    t2 = int(n * 0.85)  # 각 환자 내 85% 지점
    train_list.append((X[:t1],  y[:t1]))
    val_list.append(  (X[t1:t2], y[t1:t2]))
    test_list.append( (X[t2:],  y[t2:]))

X_train = np.vstack([x for x, _ in train_list])
X_val   = np.vstack([x for x, _ in val_list])
X_test  = np.vstack([x for x, _ in test_list])
# ... y도 동일하게 concatenate
```

```
결과적 데이터 구성 (모든 환자의 시계열 흐름이 3개 세트에 고르게 분포)
├── X_train / y_train : 전체 환자 각각의 앞 70% 시점 병합
├── X_val   / y_val   : 전체 환자 각각의 중간 15% 시점 병합 ← Optuna 목적함수
└── X_test  / y_test  : 전체 환자 각각의 마지막 15% 시점 병합 ← 최종 평가 전용
```

> **load_dataset()의 현재 80/20 분할도 동일한 원칙이 적용되어 있음** — 환자 파일별로 80%/20% 분할 후 합산하는 구조이므로 호환됨. `02_optuna_hpo.py`에서는 이를 70/15/15로 확장하면 된다.

---

## 3. 학습 환경 통제 및 자원 분배 (Resource Allocation)

### 3.1 데이터셋 크기별 동적 n_trials 할당

베이스라인에서 측정된 **실제 per-trial 학습 시간**을 기반으로 현실적인 탐색 횟수를 유도한다.

| 규모 | 데이터셋 | Windows | LGBM/trial | Cat/trial | n_trials | per-DS 예상시간 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 소형 | AIDET1D, BIGIDEAs, CGMND, Park_2025, UCHTT1DM | <50만 | ~2s | ~5s | **60** | ~10분 |
| 중형 | Bris-T1D, CGMacros×2, HUPA-UCM | 30~90만 | ~20s | ~35s | **30** | ~20분 |
| 대형 | IOBP2, PEDAP | 700만~1400만 | ~60s | ~150s | **12** | ~40분 |
| 초대형 | GLAM | 2600만 | ~180s | ~300s | **6** | ~40분 |

> **총 예상 학습 시간:** 약 7~8시간 (오버나이트 10시간 내 완료 가능)

### 3.2 Pruner 설정 — LightGBM/CatBoost 전용 통합 방식

> **⚠️ 설계 추가 (v1 대비):** v1에서 `MedianPruner`만 언급했으나, trial 단위 pruning은 각 모델과의 전용 통합 없이는 **n_estimators 내부에서의 early stopping이 불가능하여 속도 개선 효과가 미미**하다.

**LightGBM:** `optuna.integration.LightGBMPruningCallback`을 사용하여 트리 추가 단계(round)마다 중간 성능을 Optuna에 보고 → 가망 없는 trial은 조기 종료

```python
# LightGBM 내에서 Optuna Pruning 연동 예시
from optuna.integration import LightGBMPruningCallback

callbacks = [
    LightGBMPruningCallback(trial, "rmse"),
    lgb.log_evaluation(0)
]
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=callbacks)
```

**CatBoost:** `early_stopping_rounds` + Optuna의 `MedianPruner` 조합 (CatBoost는 Optuna 전용 통합 콜백 미제공 — trial 단위 pruning)

```python
# CatBoost Pruning 방식
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
```

### 3.3 전체 타임아웃 안전장치

```python
study.optimize(
    objective,
    n_trials=N_TRIALS_PER_DS,  # 데이터셋별 동적 할당
    timeout=per_ds_timeout_sec,  # 데이터셋별 시간 할당
    n_jobs=1,  # Optuna 내부는 단일 스레드, 모델 학습에 n_jobs=16 할당
    show_progress_bar=True
)
```

---

## 4. 하이퍼파라미터 탐색 공간 (Search Space)

### 4.1 LightGBM Search Space

```python
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
    'n_estimators': 1000,  # early_stopping으로 실제 최적 iter 탐색
    'n_jobs': N_JOBS,
    'random_state': 42,
    'verbose': -1,
}
```

> `n_estimators`를 크게 설정하고 `early_stopping_rounds=50`으로 실제 최적 반복 수를 자동 결정

### 4.2 CatBoost Search Space — 데이터셋 크기별 depth 상한 동적 제한

#### ⚠️ CatBoost Oblivious Tree 지수적 복잡도 경고

CatBoost는 완전 이진 대칭 트리(Oblivious Tree) 구조를 사용하므로, `depth`가 1 증가할 때마다 리프 노드 수가 $O(2^d)$로 지수 폭발한다.

| depth | 리프 수 | GLAM(2600만) 위험도 |
|:---:|:---:|:---:|
| 8 | 256 | 안전 ✅ |
| 10 | 1,024 | 주의 ⚠️ |
| **12** | **4,096** | **OOM 위험 🚨** |

**규칙: 데이터셋 윈도우 수에 따라 `depth` 탐색 상한을 동적으로 결정한다.**

```python
# 데이터셋별 CatBoost depth 상한 매핑
CATBOOST_MAX_DEPTH = {
    # 소형 (<50만)  → depth 최대 11 허용
    'small':  11,
    # 중형 (50만~500만) → depth 최대 10
    'medium': 10,
    # 대형/초대형 (500만+) → depth 최대 8  (OOM/시간 안전선)
    'large':  8,
}

def get_ds_scale(n_windows):
    if n_windows < 500_000:
        return 'small'
    elif n_windows < 5_000_000:
        return 'medium'
    else:
        return 'large'

# objective 함수 내에서
max_d = CATBOOST_MAX_DEPTH[get_ds_scale(n_windows)]
params = {
    'depth':              trial.suggest_int('depth', 5, max_d),  # 동적 상한
    'learning_rate':      trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
    'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg', 1.0, 20.0),
    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    'border_count':       trial.suggest_int('border_count', 64, 255),
    'min_data_in_leaf':   trial.suggest_int('min_data_in_leaf', 5, 100),
    # 고정값
    'iterations': 1000,
    'early_stopping_rounds': 50,
    'eval_metric': 'RMSE',
    'thread_count': N_JOBS,
    'random_seed': 42,
    'verbose': 0,
}
```

### 4.3 Sampler 선택

```python
sampler = optuna.samplers.TPESampler(
    seed=42,
    multivariate=True,  # 파라미터 간 상관관계 학습 (기본 독립 가정보다 우수)
    n_startup_trials=10  # 초기 10회는 Random Sampling으로 공간 탐색
)
```

---

## 5. 최종 평가 전략 및 HPO 결과 저장 스키마

### 5.1 최종 평가 설계 결정 (Design Decision)

> **Tier 3 채택: Option A — HPO Best Trial 모델 객체를 Test 셋에 직접 적용**

| | Option A: Best Trial 직접 사용 ✅ | Option B: Train+Val 병합 후 Retrain |
|---|---|---|
| **구현 복잡도** | 낮음 | 높음 (n_estimators 보정 필요) |
| **재현성** | 완벽 | 보정 계수(1.1~1.2x)에 따라 변동 |
| **데이터 활용** | Train 70% 학습 | Train+Val 85% 학습 |
| **Tier 3 목표 적합성** | ✅ 공정한 Tier 간 벤치마크 | ⚠️ 검증 세트 오염 위험 |
| **Tier 4 적합성** | ❌ 일반화 성능 낮음 | ✅ 권장 |

**Tier 3 최종 평가:** Best Trial 모델 객체 → `X_test` 직접 예측  
**Tier 4 최종 모델:** 최적 파라미터 + `n_estimators × 1.15` → Train+Val 85% Retrain 후 Cross-dataset 평가

> **n_estimators 보정 근거 (Tier 4용):** early_stopping은 Train 70% 기준으로 조기 종료됨. 데이터가 85%로 증가하면 트리가 더 학습할 여지가 생기므로, 저장된 `actual_n_estimators`에 경험적 계수 **1.15를 곱하여** Retrain 횟수를 설정한다.

### 5.2 결과 저장 스키마 (JSON)

- **파일명:** `hpo_best_params.json`

```json
{
  "metadata": {
    "tuning_date": "YYYY-MM-DD HH:MM",
    "total_budget_sec": 36000,
    "validation_strategy": "patient-wise 70/15/15 time-ordered split",
    "final_eval_strategy": "best_trial_model_direct (Tier 3) / retrain_85pct (Tier 4)"
  },
  "datasets": {
    "GLAM": {
      "n_windows": 26165917,
      "scale": "large",
      "LightGBM": {
        "best_trial_id": 3,
        "val_RMSE": 13.21,
        "val_MAE": 9.51,
        "val_MAPE": 9.8,
        "test_RMSE": 13.38,
        "actual_n_estimators": 412,
        "retrain_n_estimators": 474,
        "params": {
          "num_leaves": 255,
          "max_depth": 11,
          "learning_rate": 0.05,
          "min_child_samples": 30,
          "reg_lambda": 0.5
        }
      },
      "CatBoost": {
        "best_trial_id": 2,
        "val_RMSE": 13.25,
        "test_RMSE": 13.42,
        "actual_n_estimators": 387,
        "retrain_n_estimators": 445,
        "params": {
          "depth": 8,
          "learning_rate": 0.06,
          "l2_leaf_reg": 5.2
        }
      }
    }
  }
}
```

> **Co-worker 활용 가이드:**
> - **Tier 3 재현:** `best_trial_id`의 모델 객체를 저장된 `params` + `actual_n_estimators`로 재생성 후 `X_test`에 직접 예측
> - **Tier 4 Retrain:** `params` + `retrain_n_estimators`(= `actual_n_estimators × 1.15`, 정수 반올림)로 Train+Val 85% 데이터 재학습

---

## 6. 스크립트 파일 구조 (`02_optuna_hpo.py`)

```
02_optuna_hpo.py
├── [Config]  N_JOBS, TOTAL_BUDGET_SEC, 데이터셋별 n_trials/timeout 매핑 딕셔너리
├── [Func]    split_train_val_test(X, y)  → 70/15/15 시간 순서 분할
├── [Func]    objective_lgbm(trial, X_tr, y_tr, X_val, y_val)
├── [Func]    objective_catboost(trial, X_tr, y_tr, X_val, y_val)
├── [Func]    run_study(ds_name, model_name, objective_fn, data, ...)
├── [Main]    데이터셋 순회 → study 생성 → optimize → JSON 저장
└── [Output]  hpo_best_params.json
             04_HPO_Results.md  (결과 요약 테이블)
```

---

## 7. 변경 이력 (Changelog)

| 버전 | 일자 | 주요 변경 내용 |
|---|---|---|
| v1 | 2026-04-15 | 초안 작성 (Co-worker 초안) |
| v2 | 2026-04-15 | ① Multi-objective → Single-objective(RMSE) 전환 ② n_trials 데이터셋 크기별 동적 할당 ③ LightGBM Optuna 전용 Callback 통합 명시 ④ 70/15/15 시계열 검증 분할 추가 ⑤ 구체적 Search Space 정의 ⑥ TPE multivariate sampler 명시 |
| v3 | 2026-04-15 | ① **Patient-wise split** 구현 방식 명확화 및 전체 병합 후 단순 분할의 위험성 경고 추가 ② **CatBoost depth 동적 제한** — 데이터셋 크기별 max_depth 상한 매핑 테이블 추가(OOM 방지) ③ **최종 평가 전략 설계 결정** — Tier 3: Best Trial 모델 직접 사용 / Tier 4: 85% Retrain + n_estimators×1.15 방침 확정 ④ JSON 스키마에 `retrain_n_estimators` 필드 추가 |
