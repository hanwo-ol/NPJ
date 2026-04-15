# Tier 3: Advanced Machine Learning 설계서

## 1. 개요 (Overview)
Tier 3의 목표는 **딥러닝을 제외한 가장 진보된 머신러닝 기법**을 활용하여 연속 혈당 측정(CGM) 데이터를 기반으로 한 혈당 예측 모델의 성능을 극대화하는 것입니다. 앞선 `9_Tier_2.5_Feature_Engineering`에서 추출된 고품질 피처들을 100% 활용하며, 추후 진행될 Tier 4(전이 학습 및 타 데이터셋 적용)의 견고한 기반(Base)을 마련하는 데 중점을 둡니다.

## 2. 핵심 설계 방향 (Core Architecture)

### 2.1 최신 트리 기반 앙상블 알고리즘 (Advanced Boosting Ensembles)
전통적인 머신러닝 알고리즘(Classic ML)의 성능 한계를 극복하기 위해, 현재 정형(Tabular) 및 시계열 예측에서 가장 뛰어난 성능을 보이는 **3대 Boosting 알고리즘**을 도입합니다.
* **XGBoost (eXtreme Gradient Boosting):** 강력한 정규화(Regularization)를 제공하여 과적합을 방지하고 높은 예측 성능을 달성합니다.
* **LightGBM:** 대용량 CGM 데이터에서 학습 속도가 매우 빠르며 메모리 사용량이 적어, 수많은 파생 변수를 처리하는 데 적합합니다.
* **CatBoost:** 범주형 변수(예: 시간대, 요일, 식사 여부, 환자 ID 등)를 별도의 타겟 인코딩 없이 최적으로 처리하며, 하이퍼파라미터 튜닝 없이도 높은 기본 성능을 보장합니다.

> **Note:** 모델별 구체적인 입력 데이터 형태, 범주형 변수 처리, 로컬 메모리(32GB) 및 멀티코어(12코어 20스레드) 활용 최적화, Optuna 튜닝 전략 등에 대한 세부 계획은 [`01_Tier3_Advanced_Models_Detail.md`](./01_Tier3_Advanced_Models_Detail.md)를 참조하십시오.

### 2.2 베이지안 하이퍼파라미터 최적화 (Advanced HPO: Optuna)
수많은 하이퍼파라미터를 가지는 부스팅 모델들의 잠재력을 최대한 끌어내기 위해, Grid Search나 Random Search 대신 **Optuna** 프레임워크를 사용합니다.
* Search Space의 지능적 탐색 (Bayesian Optimization)
* 자원 절약을 위한 조기 종료(Pruning) 도입
* 모델별로 최적의 파라미터 셋트 산출 및 기록

### 2.3 시계열 특화 교차 검증 (Time-Series Cross Validation)
혈당 데이터는 시간에 따른 종속성이 강합니다. 미래의 데이터가 과거 학습에 사용되는 데이터 누수(Data Leakage)를 원천 차단합니다.
* **Walk-Forward Validation / Expanding Window:** 시간을 축으로 과거 데이터만 사용하여 모델을 학습하고, 바로 직후의 미래 데이터를 예측 및 평가하는 구조로 검증합니다.
* 이를 통해 실제 임상 환경(Real-time prediction)과 가장 유사한 모델 성능 지표를 얻을 수 있습니다.

### 2.4 스태킹 및 앙상블 융합 (Stacking & Blending Ensemble)
단일 모델의 편향(Bias)을 줄이고 Tier 4 전이학습을 위한 일반화 성능을 높이기 위해, 여러 모델의 예측을 결합합니다.
* **Base Layer:** 최적화된 XGBoost, LightGBM, CatBoost 및 우수한 성능의 Classic ML 모델.
* **Meta Layer:** Base Layer의 예측값들을 입력받아 최종 혈당값을 예측하는 가벼운 Meta Model (예: Ridge Regression, Linear Regression) 구축.

### 2.5 설명 가능한 AI (XAI: eXplainable AI)
의료 도메인에서는 단순한 성능보다 모델의 예측 근거가 매우 중요합니다.
* **SHAP (SHapley Additive exPlanations) 도입:**
  * 특정 시간대에 혈당이 급격히 상승/하락할 것으로 예측한 이유(Feature Contribution)를 분석.
  * Tier 2.5에서 생성된 어떤 피처가 전반적인 예측 성능에 가장 결정적인 영향을 미치는지 시각화 (SHAP Summary Plot, Force Plot).

---

## 3. 개발 파이프라인 및 파일 구조 제안

Tier 3 파이프라인은 논리적 순서에 따라 다음과 같은 스크립트/모듈로 구성됩니다.

```text
10_Tier_3_Advanced_ML/
│
├── 01_baseline_boosting.py      # XGB, LGBM, CatBoost의 기본 학습 및 비교 평가
├── 02_optuna_hpo.py             # Optuna를 활용한 베이지안 하이퍼파라미터 튜닝
├── 03_timeseries_cv.py          # Walk-Forward 방식의 시계열 교차 검증 적용 파이프라인
├── 04_stacking_ensemble.py      # 튜닝된 다중 모델 기반의 스태킹 앙상블 구축
├── 05_shap_analysis.py          # 최종 모델 대상 SHAP 기반 특성 중요도 및 해석 결과 추출
├── 01_Tier3_Advanced_Models_Detail.md # 각 모델별 로컬 환경 최적화 및 세부 실행 전략 문서
└── Tier3_Design.md              # 현재 설계 문서
```

## 4. 기대 효과 및 Tier 4 연계성 (Link to Tier 4)
* **극한의 성능 달성:** 딥러닝을 사용하지 않고도 딥러닝에 필적하거나 오히려 우수한 성능과 학습 속도를 확보합니다.
* **견고함(Robustness):** 시계열 교차검증과 앙상블을 통해 특정 환자나 특정 기간에 과적합되지 않는 일반화된 모델을 생성합니다.
* **전이 학습 최적화 (Tier 4 준비):** Tier 3에서 추출된 Meta Model과 최적화된 하이퍼파라미터 지식은, Tier 4에서 완전히 새로운 CGM 데이터셋이나 타 국가 환자군에 적용될 때 강력한 베이스라인이자 Transfer Learning의 초기 가중치/로직으로 기능하게 됩니다.
