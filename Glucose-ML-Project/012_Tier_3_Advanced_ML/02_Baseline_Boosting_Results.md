# Tier 3 Baseline Boosting Results

XGBoost / LightGBM / CatBoost 디폴트 하이퍼파라미터 기저 성능 비교

## 1. 성능 비교표 (RMSE mg/dL)

| Dataset         |   Windows |   Dim |   XGB_RMSE |   LGBM_RMSE |   Cat_RMSE | Best     |   Best_RMSE |   Best_MAE |   Best_MAPE% |
|:----------------|----------:|------:|-----------:|------------:|-----------:|:---------|------------:|-----------:|-------------:|
| AIDET1D         |    471528 |    20 |      22.54 |       22.46 |      22.36 | CatBoost |       22.36 |      15.7  |         11.8 |
| BIGIDEAs        |     36409 |    92 |      14.4  |       14.28 |      14.03 | CatBoost |       14.03 |       9.07 |          7.6 |
| Bris-T1D_Open   |    818600 |   236 |      27.49 |       27.42 |      27.34 | CatBoost |       27.34 |      19.28 |         13.4 |
| CGMacros_Dexcom |    415299 |   398 |       3.22 |        3.08 |       3.22 | LightGBM |        3.08 |       1.97 |          1.4 |
| CGMacros_Libre  |    455705 |   398 |       1.63 |        1.4  |       1.34 | CatBoost |        1.34 |       0.81 |          0.8 |
| CGMND           |    114409 |    20 |      14.29 |       14.13 |      13.92 | CatBoost |       13.92 |       9.49 |          9.2 |
| GLAM            |  26165917 |    38 |      13.51 |       13.5  |      13.51 | LightGBM |       13.5  |       9.72 |         10   |
| HUPA-UCM        |    309092 |   200 |      14.72 |       14.63 |      14.42 | CatBoost |       14.42 |       9.77 |          7.9 |
| IOBP2           |  14027905 |    74 |      24.79 |       24.78 |      24.81 | LightGBM |       24.78 |      17.45 |         11.7 |
| Park_2025       |     23064 |    38 |      21.13 |       21.02 |      20.18 | CatBoost |       20.18 |      15.43 |         16.5 |
| PEDAP           |   7055811 |    56 |      28.34 |       28.33 |      28.38 | LightGBM |       28.33 |      20.28 |         14.8 |
| UCHTT1DM        |     27220 |    38 |      19.11 |       19    |      17.78 | CatBoost |       17.78 |      11.98 |         11.8 |

## 2. Best Model Feature Importance & Training Time

| Dataset         | Best_Top5                                                                                                                            | XGB_Time   | LGBM_Time   | Cat_Time   |
|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------|:-----------|:------------|:-----------|
| AIDET1D         | glucose_value_mg_dl_t-0 (60.7%), Velocity (18.8%), glucose_value_mg_dl_t-2 (3.8%), glucose_value_mg_dl_t-1 (3.2%), Window_Std (2.2%) | 2s         | 2s          | 5s         |
| BIGIDEAs        | glucose_value_mg_dl_t-0 (37.0%), Velocity (11.2%), Acceleration (5.8%), tod_cos (4.4%), tod_sin (3.8%)                               | 2s         | 1s          | 3s         |
| Bris-T1D_Open   | glucose_value_mg_dl_t-0 (55.5%), Velocity (22.2%), glucose_value_mg_dl_t-1 (4.1%), SD1 (3.3%), tod_sin (2.3%)                        | 14s        | 4s          | 52s        |
| CGMacros_Dexcom | Velocity (10.7%), glucose_value_mg_dl_t-0 (4.6%), Libre GL_t-0 (3.9%), SD1 (2.9%), Window_Std (2.7%)                                 | 131s       | 27s         | 24s        |
| CGMacros_Libre  | glucose_value_mg_dl_t-0 (20.0%), TIR (14.8%), glucose_value_mg_dl_t-1 (12.5%), glucose_value_mg_dl_t-2 (9.3%), Window_AUC (7.6%)     | 32s        | 11s         | 24s        |
| CGMND           | glucose_value_mg_dl_t-0 (28.7%), tod_cos (11.8%), Velocity (8.6%), glucose_value_mg_dl_t-5 (8.5%), tod_sin (6.4%)                    | 1s         | 1s          | 2s         |
| GLAM            | Velocity (10.0%), glucose_value_mg_dl_t-0 (7.5%), Jerk (7.5%), SD1 (7.4%), glucose_value_mg_dl_t-5 (7.2%)                            | 413s       | 179s        | 298s       |
| HUPA-UCM        | glucose_value_mg_dl_t-0 (46.4%), Velocity (20.9%), glucose_value_mg_dl_t-1 (7.6%), TIR (2.6%), glucose_value_mg_dl_t-2 (1.8%)        | 9s         | 5s          | 13s        |
| IOBP2           | Velocity (12.2%), Jerk (10.0%), Acceleration (9.9%), SD1 (9.1%), Window_Std (7.3%)                                                   | 137s       | 60s         | 230s       |
| Park_2025       | Window_AUC (24.5%), Window_Mean (20.2%), HBGI (9.9%), TIR (7.0%), TAR (5.7%)                                                         | 3s         | 3s          | 7s         |
| PEDAP           | Velocity (9.7%), tod_cos (6.5%), tod_sin (6.4%), SD1 (6.1%), glucose_value_mg_dl_t-0 (5.9%)                                          | 109s       | 47s         | 82s        |
| UCHTT1DM        | glucose_value_mg_dl_t-0 (41.1%), Velocity (17.1%), glucose_value_mg_dl_t-1 (6.6%), Window_Mean (4.5%), Window_Std (2.9%)             | 1s         | 0s          | 2s         |

---

## 3. HPO vs Default 비교 — Optuna 튜닝 효과 분석

> 총 소요: **4.6시간** (Optuna, LightGBM + CatBoost, 12개 데이터셋)

### 3.1 HPO 결과 비교표

| Dataset | Default Best | Default RMSE | HPO Best | HPO RMSE | Δ RMSE | 판정 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| AIDET1D | CatBoost | 22.36 | CatBoost | **22.35** | -0.01 | ≈ 동일 |
| BIGIDEAs | CatBoost | **14.03** | LightGBM | 14.36 | +0.33 | 악화 |
| Bris-T1D | CatBoost | 27.34 | CatBoost | **27.08** | **-0.26** | 소폭 개선 |
| CGMacros_Dex | LightGBM | 3.08 | LightGBM | **2.89** | **-0.19** | 소폭 개선 |
| CGMacros_Lib | CatBoost | **1.34** | LightGBM | 1.41 | +0.07 | 악화 |
| CGMND | CatBoost | **13.92** | CatBoost | 14.14 | +0.22 | 악화 |
| GLAM | LightGBM | **13.50** | LightGBM | 13.52 | +0.02 | ≈ 동일 |
| HUPA-UCM | CatBoost | **14.42** | CatBoost | 14.49 | +0.07 | ≈ 동일 |
| IOBP2 | LightGBM | 24.78 | LightGBM | **24.74** | -0.04 | ≈ 동일 |
| Park_2025 | CatBoost | 20.18 | CatBoost | **19.85** | **-0.33** | 소폭 개선 |
| PEDAP | LightGBM | **28.33** | LightGBM | 28.43 | +0.10 | 악화 |
| UCHTT1DM | CatBoost | 17.78 | CatBoost | **17.75** | -0.03 | ≈ 동일 |

### 3.2 통계 요약

| 지표 | 값 |
|:---|:---:|
| 개선 데이터셋 | 5/12 (42%) |
| 악화 데이터셋 | 4/12 (33%) |
| 동일(±0.1) | 3/12 (25%) |
| 평균 Δ RMSE | **-0.004 mg/dL** |
| 최대 개선 | -0.33 (Park_2025) |
| 최대 악화 | +0.33 (BIGIDEAs) |
| HPO 소요 시간 | 4.6시간 |
| **시간 대비 효과** | **≈ 0 mg/dL / hour** |

### 3.3 결론: HPO는 실질적 효과가 없다

> **4.6시간의 HPO가 산출한 평균 RMSE 개선은 0.004 mg/dL로, 이는 CGM 센서 자체의 측정 오차(±10~15 mg/dL)의 0.04%에 불과하다.** LightGBM/CatBoost의 디폴트 하이퍼파라미터가 이미 거의 최적이며, 혈당 예측 성능의 상한은 **하이퍼파라미터가 아닌 다른 요인**에 의해 결정된다.

---

## 4. 핵심 발견사항 (Key Findings)

### 4.1 성능 상한을 결정하는 요인은 하이퍼파라미터가 아니다

GBM 모델의 혈당 예측 성능은 다음 순서로 영향을 받는다:

1. **환자군(Cohort) 특성** — T1DM(RMSE ~22~28) vs ND(RMSE ~3~14) → **최대 25 mg/dL 차이**
2. **피처 풍부도** — CGMacros(398 dim, RMSE 1~3) vs AIDET1D(20 dim, RMSE 22) → **다중 모달 데이터의 압도적 효과**
3. **데이터 규모** — 대체로 대규모 데이터셋에서 더 낮은 MAPE
4. **모델 선택** (XGB vs LGBM vs CatBoost) — **최대 ~1 mg/dL 차이**
5. **하이퍼파라미터** (디폴트 vs 튜닝) — **최대 ~0.3 mg/dL 차이 (노이즈 수준)**

### 4.2 현재 파이프라인에 포함되지 않은 요소

> [!WARNING]
> **현재 Tier 3 파이프라인은 Q(설문/인구통계) 및 M(임상검사) 범주의 변수를 전혀 포함하지 않는다.** 모든 12개 데이터셋의 원본에는 나이, 성별, HbA1c, BMI 등의 환자별 정적 변수가 존재하지만(🔶), harmonized 시계열 파이프라인에서 체계적으로 제외되어 있다.
>
> 이는 다음을 의미한다:
> - 현재 모델은 **순수하게 시계열 패턴(CGM lag, velocity, window stats 등)만으로** 예측하고 있다
> - "이 환자는 HbA1c 8%의 T1DM 환자"인지 "HbA1c 5.5%의 건강한 성인"인지에 대한 정보가 **모델에 전달되지 않는다**
> - HPO로 0.004 mg/dL을 쥐어짜는 것보다, **Q/M 정적 피처를 추가하는 것이 훨씬 큰 성능 개선을 가져올 가능성**이 있다
>
> 상세 분석: `011_Intermediate_Results/tier4/method1.md` §2.4 "Q/M 정적 피처의 Local Feature 확장" 참조

### 4.3 임상 배포 관점

| 관점 | 시사점 |
|:---|:---|
| **병원 현장 배포** | 디폴트 파라미터로 충분 → HPO 인프라 불필요. 신규 병원에서 GPU 없이도 즉시 배포 가능 |
| **모델 업데이트** | 신규 환자 데이터 축적 시 HPO 없이 재학습만으로 동등한 성능 유지 가능 |
| **연구 우선순위** | HPO보다 **피처 엔지니어링(Q/M 정적 변수 추가)** 및 **전이학습(다중 데이터셋 활용)** 에 투자하는 것이 효율적 |

---

## 5. Tier 4 전이학습과의 연결

현재 Tier 3의 결론은 다음 Tier 4 연구 질문에 직접 동기를 부여한다:

| Tier 3 결론 | Tier 4 연구 질문 |
|:---|:---|
| HPO 효과 ≈ 0 mg/dL | 전이학습으로 달성 가능한 cross-dataset 성능 변동은 HPO보다 얼마나 큰가? |
| Q/M 피처 미포함 | Q/M 정적 피처(HbA1c, BMI, 나이)를 global feature로 추가하면 전이 성능이 개선되는가? (H5) |
| 디폴트 파라미터 충분 | 전이학습에서도 디폴트 파라미터가 충분한가? → Tier 4에서 HPO 생략 정당화 근거 |

