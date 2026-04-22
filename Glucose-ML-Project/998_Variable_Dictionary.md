# Variable Dictionary — Glucose-ML Project

이 문서는 프로젝트 전체에서 사용되는 변수를 분류하고, 각 변수의 정의·타입·값·출처·적용 범위를 기록한다.
단일 진실 원본: `global_config.py`의 `GlobalConfig`.
각 Tier Config는 이 값을 상속하며 재정의하지 않는다.

---

## 1. 공통 변수 (GlobalConfig — 모든 Tier 공유)

### 1.1 경로

| 변수명 | 타입 | 값 | 설명 |
|---|---|---|---|
| `PROJECT_ROOT` | `Path` | `global_config.py` 위치의 부모 디렉터리 | 프로젝트 루트 경로 |
| `DATA_ROOT` | `Path` | `PROJECT_ROOT / "003_Glucose-ML-collection"` | 전체 데이터셋 루트 |

### 1.2 재현성

| 변수명 | 타입 | 값 | 설명 |
|---|---|---|---|
| `SEED` | `int` | `42` | 모든 난수 생성기의 공통 seed. Subject 분리, 모델 초기화에 사용. |

### 1.3 태스크 정의 (Rule 8)

| 변수명 | 타입 | 값 | 설명 |
|---|---|---|---|
| `LOOKBACK_STEPS` | `int` | `3` | 모델 입력에 사용하는 과거 CGM 측정값의 스텝 수. |
| `PREDICTION_STEPS` | `int` | `3` | 예측 대상 시점. 마지막 룩백 이후 3스텝 뒤의 값. |

**물리적 예측 시간 (그룹별)**

| 샘플링 주기 | 예측 시간 |
|---|---|
| 1분 | 3분 뒤 |
| 5분 | 15분 뒤 |
| 15분 | 45분 뒤 |

### 1.4 데이터 분리 비율 (Rule 5)

| 변수명 | 타입 | 값 | 설명 |
|---|---|---|---|
| `TRAIN_RATIO` | `float` | `0.70` | Subject의 70%를 학습에 배정. |
| `VAL_RATIO` | `float` | `0.15` | Subject의 15%를 Early Stopping 및 하이퍼파라미터 선택에 사용. |
| `TEST_RATIO` | `float` | `0.15` | Subject의 15%를 최종 성능 보고에 1회만 사용. |

분리 방식: Subject 단위 3-way. Subject 배정은 `SEED`로 고정된 무작위 순서로 결정한다. Subject 내부 시계열 순서는 보존한다.

### 1.5 엄격 전처리 기준 (Rule 1~4)

| 변수명 | 타입 | 값 | 단위 | 설명 |
|---|---|---|---|---|
| `MIN_GLUCOSE` | `float` | `40.0` | mg/dL | 혈당 하한값. 미만 행 제거. |
| `MAX_GLUCOSE` | `float` | `400.0` | mg/dL | 혈당 상한값. 초과 행 제거. |
| `MAX_ROC_MG_DL_MIN` | `float` | `20.0` | mg/dL/min | 변화율 상한. 초과 행 제거. |
| `MIN_TIME_DIFF_SEC` | `float` | `30.0` | 초 | 측정 간격 하한. 미만이면 센서 오류로 간주. |
| `MISSING_GAP_MULT` | `float` | `3.0` | 배수 | 샘플링 주기의 3배 이상 공백을 결측 구간으로 판정. |

### 1.6 동적 스키마 (혈당 컬럼명 우선순위)

```
['glucose_value_mg_dl', 'CGM', 'Value', 'GlucValue', 'CGM (mg / dl)', 'CBG (mg / dl)']
```

순서대로 탐색하여 최초 발견된 컬럼을 대상 혈당 컬럼으로 사용한다.

### 1.7 Single Source of Truth — 폴더 우선순위

| 우선순위 | 폴더 접미사 | 설명 |
|---|---|---|
| 1 | `-extracted-glucose-files` | Glucose-ML 표준 추출 파일 |
| 2 | `-time-augmented` | 시간 피처 보강 파일 |
| 3 | (루트 CSV) | 위 두 폴더 없을 때 폴백 |
| 제외 | `-extended-features` | 다중 모달리티 파일. 혈당 단독 소스로 사용하지 않음. |

### 1.8 제외 데이터셋 목록 (Rule 9)

| 데이터셋 | 제외 사유 |
|---|---|
| `Park_2025` | `timestamp`가 식사 기준 상대 시간(분)이며 시간 순서를 복원할 수 없음. 연속 CGM 예측 태스크에 부적합. (999_Preprocessing_Rules.md Rule 9 참고) |

### 1.9 연산 설정

| 변수명 | 타입 | 값 | 설명 |
|---|---|---|---|
| `N_JOBS` | `int` | `-1` | scikit-learn / LightGBM 병렬 코어 수. -1은 전체 코어 사용. |

---

## 2. Tier별 추가 변수

### 2.1 Tier 2.5 v3 (tier2_5_v3_config.py)

GlobalConfig 상속. 아래 항목만 추가로 정의한다.

| 변수명 | 타입 | 값 | 설명 |
|---|---|---|---|
| `OUTPUT_DIR` | `Path` | `PROJECT_ROOT / "014_Tier_2.5_v3_Strict_Preprocessing"` | 로그 및 결과 저장 경로 |
| `MIN_SUBJECTS_PER_GROUP` | `int` | `5` | 그룹당 최소 Subject 수. 미만 시 해당 그룹 학습 생략. |

**RATE_GROUPS** (샘플링 주기 그룹 정의)

| 그룹명 | `rate_min` (분) | `rate_max` (분) | `pred_time_min` (분) |
|---|---|---|---|
| `1min` | 0.5 | 2.0 | 3 |
| `5min` | 2.0 | 8.0 | 15 |
| `15min` | 8.0 | 20.0 | 45 |

`pred_time_min` = `PREDICTION_STEPS` × 해당 주기. 결과 보고 시 명시.

### 2.2 Tier 3~5 (012_Tier_3_Advanced_ML)

GlobalConfig 상속으로 전환 예정. 현재 파일(`tier3_data_utils.py`)에 하드코딩된 아래 값들은 GlobalConfig 값으로 대체한다.

| 현재 변수명 | 현재 값 | 대체 변수 | 현재 파일 위치 |
|---|---|---|---|
| `n_back=6` | `GlobalConfig.LOOKBACK_STEPS` (= 3) | - | `tier3_data_utils.py:145` |
| `n_fwd=6` | `GlobalConfig.PREDICTION_STEPS` (= 3) | - | `tier3_data_utils.py:145` |
| `train_ratio=0.8` | `GlobalConfig.TRAIN_RATIO` (= 0.70) | - | `tier3_data_utils.py:269` |
| `seed=42` | `GlobalConfig.SEED` | - | 각 스크립트 |
| `N_JOBS=16` | `GlobalConfig.N_JOBS` (= -1) | - | `01_baseline_boosting.py:37` |

> `fillna(0)` (`tier3_data_utils.py:162, 308`)와 `ffill()` (`tier3_data_utils.py:169`)은
> 인슐린·식사량 등 이벤트형 공변량 컬럼에만 적용된다.
> 이벤트 부재(= 0)는 물리적으로 올바른 처리이며, 혈당 보간 금지 규칙(Rule 3)의 적용 대상이 아니다.
> 해당 호출은 유지한다.

**Tier 5 전용 추가 변수**

| 변수명 | 위치 | 설명 |
|---|---|---|
| `LOCAL_FEATURE_MAP` | `tier5_data_utils.py` | 데이터셋별 사용 가능한 로컬 동적 피처 컬럼 목록 |
| `COHORT_DEFAULTS` | `tier5_data_utils.py` | 코호트별 Q/M 정적 피처 대체값 (HbA1c, 나이, 성별) |
| `DATASET_COHORT` | `tier5_data_utils.py` | 데이터셋 → 코호트 매핑 |

### 2.3 Tier 6 (013_Tier_6_Domain_Adaptation)

GlobalConfig 상속으로 전환 예정. 현재 파일(`tier6_config.py`)에서 아래 값들을 대체한다.

| 현재 변수명 | 현재 값 | 대체 변수 |
|---|---|---|
| `RANDOM_SEED` | `42` | `GlobalConfig.SEED` |
| `TRAIN_RATIO` | `0.7` | `GlobalConfig.TRAIN_RATIO` |
| `VAL_RATIO` | `0.15` | `GlobalConfig.VAL_RATIO` |

Tier 6 전용 추가 변수 (유지)

| 변수명 | 타입 | 값 | 설명 |
|---|---|---|---|
| `LGBM_ROUNDS` | `int` | `1000` | LightGBM 최대 학습 라운드 수 |
| `LGBM_EARLY_STOPPING` | `int` | `50` | LightGBM Early Stopping 기준 라운드 수 |
| `LGBM_SP_ROUNDS` | `int` | `500` | 특수화 모델(SP) 최대 학습 라운드 수 |
| `LGBM_SP_EARLY_STOPPING` | `int` | `20` | 특수화 모델 Early Stopping 기준 |
| `N_GLOBAL_TS_FEATURES` | `int` | `20` | 전역 시계열 피처 차원 수 |
| `TCA_N_COMPONENTS` | `int` | `10` | Transfer Component Analysis 성분 수 |

### 2.4 Tier 7 (016_Tier_7_Cross_Disease)

GlobalConfig 상속. 아래 항목만 추가로 정의한다.

**EXPERIMENT_GROUPS** (샘플링 주기 × 소스/타겟 그룹 정의)

| 그룹명 | 샘플링 주기 | 소스 데이터셋 (T1D) | 타겟 데이터셋 |
|---|---|---|---|
| `5min` | 5분 | RT-CGM, IOBP2, FLAIR, SENCE, WISDM, PEDAP | CITY, Colas_2019 |
| `15min` | 15분 | ShanghaiT1DM, Bris-T1D_Open | ShanghaiT2DM |

소스와 타겟은 반드시 같은 그룹 내에서 선택한다. 그룹 간 혼합 금지 (AGENTS.md L27-28).

**TrAdaBoost 파라미터**

| 변수명 | 타입 | 값 | 설명 |
|---|---|---|---|
| `TRADABOOST_N_ITER` | `int` | `20` | TrAdaBoost 부스팅 반복 횟수 |
| `TRADABOOST_ENSEMBLE` | `int` | `10` | 앙상블에 사용할 후반 모델 수 (마지막 N개 평균) |

**기타 Tier 7 전용 변수**

| 변수명 | 타입 | 값 | 설명 |
|---|---|---|---|
| `LGBM_ROUNDS` | `int` | `2000` | LightGBM 최대 학습 라운드 수 |
| `LGBM_EARLY_STOPPING` | `int` | `100` | Early Stopping 기준 라운드 수 |
| `LEARNING_CURVE_RATIOS` | `list[float]` | `[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]` | 학습 곡선 실험: 타겟 데이터 비율 단계 |
| `SHAP_SAMPLE_N` | `int` | `2000` | SHAP 분석에 사용할 샘플 수 |

---

## 3. 파생 변수 (Feature Engineering)

`build_windows_no_interpolation` 및 `build_windows_with_features`에서 생성되는 파생 피처 목록.

| 피처명 | 수식 | 설명 |
|---|---|---|
| `glucose_t-0` ~ `glucose_t-2` | — | 룩백 혈당값 3스텝 (t-2, t-1, t) |
| `Velocity` | `g[t] - g[t-1]` | 1스텝 혈당 변화속도 |
| `Acceleration` | `v[t] - v[t-1]` | 변화속도의 변화율 |
| `Window_Mean` | `mean(g)` | 룩백 구간 평균 혈당 |
| `Window_Std` | `std(g)` | 룩백 구간 표준편차 |
| `TIR` | `sum(70≤g≤180) / n_back` | 목표 범위 내 비율 |
| `TAR` | `sum(g>180) / n_back` | 목표 범위 초과 비율 |
| `TBR` | `sum(g<70) / n_back` | 목표 범위 미달 비율 |
| `SD1` | `std(diff(g)) / sqrt(2)` | Poincaré SD1 (단기 변동성) |
| `tod_sin` | `sin(2π × hour / 24)` | 하루 주기 위상 (sin) |
| `tod_cos` | `cos(2π × hour / 24)` | 하루 주기 위상 (cos) |

Tier 3~5 추가 피처 (현재 v3에서는 미사용)

| 피처명 | 설명 |
|---|---|
| `LBGI` | Kovatchev 저혈당 위험 지수 |
| `HBGI` | Kovatchev 고혈당 위험 지수 |
| `Window_AUC` | 룩백 구간 혈당 곡선하 면적 |
| `Jerk` | 3차 미분 (가속도 변화율) |

Tier 7 추가 피처 (T2D 도메인 특화)

| 피처명 | 수식 | 설명 |
|---|---|---|
| `fasting_proxy` | `mean(g) × [hour < 6]` | 공복 시간대(00:00~06:00) 평균 혈당. T2D 기저 저항성 프록시. |
| `postmeal_rise` | `max(0, g[-1] - min(g))` | 룩백 구간 내 최솟값 대비 현재 상승폭. 식후 혈당 반응 추정. |
| `high_persist` | `mean(g > 180)` | 룩백 구간 중 고혈당(>180 mg/dL) 비율. 지속성 고혈당 패턴. |
| `in_range_frac` | `mean(70 ≤ g ≤ 180)` | 룩백 구간 중 목표 범위 내 비율. |

---

## 4. 임상 기준값 (TIR/TAR/TBR)

| 구간 | 범위 | 기준 |
|---|---|---|
| TBR (Low) | < 70 mg/dL | 저혈당 |
| TIR (Normal) | 70 ~ 180 mg/dL | 목표 범위 |
| TAR (High) | > 180 mg/dL | 고혈당 |

---

## 5. 변경 이력

| 날짜 | 변경 내용 | 관련 Rule |
|---|---|---|
| 2026-04-22 | `LOOKBACK_STEPS`, `PREDICTION_STEPS` = 3으로 통일 | Rule 8 |
| 2026-04-22 | Train/Val/Test 3-way 분리 도입 (70/15/15) | Rule 5 |
| 2026-04-22 | `Park_2025` 제외 데이터셋 등록 | Rule 9 |
| 2026-04-22 | `GlobalConfig` 생성, 단일 진실 원본 확립 | Rule 6, AGENTS.md |
| 2026-04-22 | 신규 데이터셋 7개 추가 (RT-CGM, CITY, SENCE, WISDM, FLAIR, SHD, ReplaceBG) — 총 26개 | 997_Active_Datasets.md |
| 2026-04-22 | Rule 9 재구성 타임스탬프 허용 기준 추가 (SHD, ReplaceBG) | 999_Preprocessing_Rules.md Rule 9 |
| 2026-04-22 | Tier 7 생성: T1D→T2D 전이학습 실험 (5-way 비교, TrAdaBoost) | 016_Tier_7_Cross_Disease |
