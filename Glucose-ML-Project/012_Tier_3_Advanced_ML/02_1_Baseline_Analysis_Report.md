# Tier 3 — Baseline Boosting 결과 분석 보고서

**실험일:** 2026-04-15  
**대상 모델:** XGBoost 3.2.0 / LightGBM 4.6.0 / CatBoost 1.2.10  
**Feature Space:** Tier 2.5_v2 동일 (시계열 위상 임베딩 + 역학 파생 변수)  
**하드웨어:** Intel Core i7-12700 (12코어 20스레드, n_jobs=16), 32GB RAM

---

## 1. 실험 개요

Tier 2.5_v2에서 검증된 feature engineering을 유지한 채, **부스팅 알고리즘을 디폴트 하이퍼파라미터로 투입**하는 첫 번째 실험이다. 이 단계의 목적은 성능 측정 자체가 아니라, **세 모델의 기저 성능·속도·Feature Importance 패턴의 차이를 진단**하여 Optuna HPO 전략 수립에 활용하는 것이다.

> **공통 기본값:** `n_estimators/iterations=300`, `max_depth=8`, `learning_rate=0.1`, `subsample/colsample=0.8`
> **트리 기반 부스팅은 Feature Scaling 불필요** → StandardScaler 미적용

---

## 2. RMSE 비교표 (vs. Tier 2.5_v2 Random Forest)

| 데이터셋 | 환자군 | Tier 2.5_v2 RF | XGB (Best) | LGBM (Best) | CatBoost (Best) | **Winner** | **vs. RF** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| AIDET1D | T1DM | 22.58 | 22.54 | 22.46 | **22.36** | CatBoost | ▼ 0.22 ✅ |
| BIGIDEAs | ND/PreD | 14.11 | 14.40 | 14.28 | **14.03** | CatBoost | ▼ 0.08 ✅ |
| Bris-T1D | T1DM | 27.57 | 27.49 | 27.42 | **27.34** | CatBoost | ▼ 0.23 ✅ |
| CGMacros_Dexcom | ND | 3.22 | 3.22 | **3.08** | 3.22 | LightGBM | ▼ 0.14 ✅ |
| CGMacros_Libre | ND | 1.27 | 1.63 | 1.40 | **1.34** | CatBoost | ▲ 0.07 ❌ |
| CGMND | ND | 14.02 | 14.29 | 14.13 | **13.92** | CatBoost | ▼ 0.10 ✅ |
| GLAM | GDM | 13.59 | 13.51 | **13.50** | 13.51 | LightGBM | ▼ 0.09 ✅ |
| HUPA-UCM | T1DM/ND | 15.02 | 14.72 | 14.63 | **14.42** | CatBoost | ▼ 0.60 ✅ |
| IOBP2 | T1DM | 24.95 | 24.79 | **24.78** | 24.81 | LightGBM | ▼ 0.17 ✅ |
| Park_2025 | ND | 20.32 | 21.13 | 21.02 | **20.18** | CatBoost | ▼ 0.14 ✅ |
| PEDAP | T1DM | 28.59 | 28.34 | **28.33** | 28.38 | LightGBM | ▼ 0.26 ✅ |
| UCHTT1DM | T1DM | 17.84 | 19.11 | 19.00 | **17.78** | CatBoost | ▼ 0.06 ✅ |

> **12개 데이터셋 중 11개에서 부스팅 모델이 RF를 초과하였다.**  
> 유일한 예외: CGMacros_Libre (디폴트 max_depth=8의 보수성이 원인으로 추정, HPO에서 해소 예상)

---

## 3. 모델별 특성 비교 분석

### 3.1 모델 선호 데이터셋 패턴

| 모델 | 승리 데이터셋 수 | 강세 환경 | 약세 환경 |
|---|:---:|---|---|
| **CatBoost** | **8/12** | 소~중형 데이터셋, 범주형/복합 Feature 공간 | 1,000만+ 초대형 데이터셋 (학습 시간↑) |
| **LightGBM** | **4/12** | 초대형 데이터셋(GLAM, IOBP2, PEDAP) | 소형 데이터셋 (범주 다양성 부족 시) |
| **XGBoost** | **0/12** | — | 모든 환경에서 LightGBM/CatBoost에 열세 |

**결론:** XGBoost는 디폴트 설정에서 LightGBM과 CatBoost 모두에 성능·속도 양면에서 우위를 점하지 못했다. HPO 자원 배분에서 제외하는 것이 합리적이다.

### 3.2 학습 속도 비교 (GLAM 기준: 2,600만 윈도우)

| 모델 | 학습 시간 | XGB 대비 |
|---|:---:|:---:|
| XGBoost | 413초 | 기준 |
| **LightGBM** | **179초** | **2.3배 빠름** ⚡ |
| CatBoost | 298초 | 1.4배 빠름 |

대용량 데이터에서 LightGBM이 Optuna n_trials를 가장 많이 탐색할 수 있는 구조로 가장 효율적이다.

---

## 4. Feature Importance 패턴 변화 (RF → Boosting)

부스팅 모델에서 Feature Importance 패턴이 RF와 두드러지게 다른 점이 확인됐다.

### 4.1 부스팅 모델에서 Velocity의 기여도 상승

RF에서 Velocity는 3~7%를 점유했으나, 부스팅 모델에서는 **10~22%로 대폭 상승**한다.  
이는 부스팅의 순차적 잔차 학습(Residual Fitting) 특성 때문이다 — 각 트리가 이전 트리의 **급격한 변화 오차**에 집중하도록 훈련되므로, 속도(1차 미분)를 포착하는 Velocity가 훨씬 중요해진다.

| 데이터셋 | RF Velocity | CatBoost/LGBM Velocity |
|---|:---:|:---:|
| Bris-T1D | 6.8% | **22.2%** |
| HUPA-UCM | 5.2% | **20.9%** |
| AIDET1D | 3.5% | **18.8%** |

### 4.2 Jerk와 SD1의 지위 강화

GLAM(GDM)과 IOBP2(T1DM)에서 `Jerk`와 `SD1`이 각각 7~10%를 기록하며 **Top 3 지표로 부상**했다.  
RF에서는 1~2%에 머물렀던 변수가 부스팅의 고해상도 분기(split) 탐색 과정에서 그 가치가 극대화된 것이다. 이는 Tier 2.5_v2에서 추가한 3차 역학 변수의 타당성을 재확인한다.

### 4.3 tod_sin/cos의 돌출 (PEDAP, CGMND)

- **PEDAP:** `tod_cos(6.5%)`, `tod_sin(6.4%)` — 일주기 리듬이 Velocity와 동등한 비중으로 부상  
- **CGMND:** `tod_cos(11.8%)` — 정상인 코호트에서 Velocity(8.6%)를 제치고 2위

이 패턴은 정상인군(ND)에서 신체가 생체리듬에 따라 혈당을 통제한다는 생리학적 직관과 정확히 일치한다.

---

## 5. 환자군별 부스팅 모델 특성

### 🔴 T1DM (AIDET1D, Bris-T1D, IOBP2, PEDAP, UCHTT1DM)
- **RMSE 범위:** 22~29 mg/dL (여전히 높음 — 롤러코스터 혈당 특성상 한계존재)
- **모델 전략:** Velocity + Jerk 중심의 역학 예측  
- **HPO 방향:** T1DM 데이터에서 `max_depth`를 더 깊게(12~16), `num_leaves`를 넓게 설정하면 세밀한 역학 분기가 가능해질 것으로 예상

### 🟢 ND/PreD (BIGIDEAs, CGMacros, CGMND, Park_2025)
- **RMSE 범위:** 1.3~21 (Park_2025 제외 시 1.3~14 — 매우 우수)
- **모델 전략:** 일주기 리듬 + Window_Mean 중심의 항상성 예측  
- **HPO 방향:** 소형 데이터셋(Park, BIGIDEAs)은 과적합 방지를 위해 `min_child_samples`, `l2_leaf_reg` 강화 필요

### 🟡 GDM — GLAM (2,600만 윈도우 특이 케이스)
- RF: 13.59 → Boosting Best(LightGBM): **13.50** — 미세하지만 일관된 개선
- GLAM에서만 Velocity의 압도적 독주(93% RF 기여)가 사라지고 `Velocity(10%)`, `Jerk(7.5%)`, `SD1(7.4%)`, `glucose_t-0(7.5%)`, `glucose_t-5(7.2%)`로 **Feature Importance가 균등하게 분산됨.**  
  이는 임산부 혈당이 단일 변수로 설명되지 않는 복합 기전임을 시사한다.

---

## 6. 결론 및 HPO 전략 수립 근거

| 항목 | 결론 |
|---|---|
| **튜닝 대상 모델** | LightGBM + CatBoost (XGBoost 제외) |
| **우선 튜닝 파라미터** | `max_depth`, `num_leaves`, `min_child_samples`, `learning_rate`, `l2_leaf_reg` |
| **튜닝 우선 데이터셋** | T1DM군(RMSE>20으로 개선 여지 최대), CGMacros_Libre(유일 역전 데이터셋) |
| **대형 데이터셋 전략** | GLAM(LightGBM 우선, timeout 할당↑), IOBP2는 n_trials 축소 |
| **공통 경고** | UCHTT1DM에서 CatBoost 디폴트가 RF보다 낮은 것은 **과적합 신호**일 수 있음 — `l2_leaf_reg` 강화 권장 |
