# 구현 설명서 (Implementation Guide)
## `run_baseline_regression.py`

### 개요
본 스크립트는 14개의 시계열 통합 데이터셋에 대하여 Ridge Regression 기반의 단변량(Univariate) 및 다변량(Multivariate) 혈당 예측 실험을 자동으로 순차 수행하고 결과 요약을 마크다운 파일로 출력합니다.

---

### 핵심 함수 설명

#### `get_dataset_global_cols(p_files)`
**역할:** 데이터셋 내 모든 환자 파일을 5행씩 샘플링하여 사용 가능한 숫자형 컬럼의 합집합을 결정하는 셰리프(Schema Negotiator) 역할을 합니다.
- `pd.to_numeric(errors='coerce')` 를 적용하여 `'Unknown'`, `'N/A'` 등 문자열 잔류값을 안전하게 처리합니다.
- `timestamp`, `person_id` 등 시스템 컬럼은 명시적으로 제외합니다.
- `glucose_value_mg_dl`을 항상 첫 번째 컬럼으로 고정하여 단변량/다변량 분기 시 슬라이싱 기준이 됩니다.

#### `generate_windows(df, global_cols, n_lookback=6, n_horizon=6)`
**역할:** 환자 한 명의 시계열 레코드를 Supervised Learning용 `(X, y)` 윈도우 행렬로 변환합니다.
- **Gap Protection:** 윈도우 내 타임스탬프 간격이 중간값(Median)의 1.5배를 초과하면 해당 윈도우를 파기(Drop)합니다. 센서 교체나 수면 중 탈락으로 인한 데이터 공백이 허위 연속성 학습으로 이어지는 것을 차단합니다.
- **글로벌 스키마 강제:** 특정 환자에게 없는 컬럼은 `0.0`으로 채워넣어 데이터셋 전체에 걸쳐 동일한 `X` 차원(Dimension)을 보장합니다.
- **Univariate `X_uni`:** `(n_lookback,)` 형태 — 과거 혈당치만 포함
- **Multivariate `X_multi`:** `(n_lookback × n_features,)` 형태 — 모든 공변량 포함 (Flattened)

#### `main()`
**역할:** 데이터셋 루프 → 환자 루프 → 80/20 연대기적 분할 → 합산 → 실험 A/B 학습 및 평가 → 결과 저장 순서로 실행됩니다.

```
[데이터셋 발견]
    ↓
[글로벌 컬럼 협상]
    ↓
[환자별 Window 생성 & Gap Drop]
    ↓
[80% Train / 20% Test 연대기 분할 (per patient)]
    ↓
[전체 환자 Train/Test 집합 병합]
    ↓
┌────────────────────────────┐
│  실험 A: Univariate Ridge  │  → RMSE_u, MAE_u, MAPE_u
└────────────────────────────┘
┌──────────────────────────────┐
│  실험 B: Multivariate Ridge  │  → RMSE_m, MAE_m, MAPE_m
└──────────────────────────────┘
    ↓
[Delta = RMSE_u - RMSE_m 계산]
    ↓
[3_Result_Summary.md 저장]
```

---

### 하이퍼파라미터
| 파라미터 | 값 | 설명 |
|---|---|---|
| `n_lookback` | 6 | 과거 관측 스텝 수 |
| `n_horizon` | 6 | 예측 목표 미래 스텝 수 |
| `gap_threshold` | 1.5x median | 윈도우 파기 기준 간격 비율 |
| `Ridge alpha` | 1.0 | L2 정규화 강도 |
| `train_ratio` | 0.80 | 환자 내 연대기적 분할 비율 |

---

### 출력 결과물
- `3_Result_Summary.md` — 데이터셋별 단변량/다변량 RMSE, MAE, MAPE 비교표 (마크다운)
- `4_Result_Report.md` — 결과 해석 및 시사점 보고서 (실험 완료 후 자동 생성)
