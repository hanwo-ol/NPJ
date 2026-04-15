# Baseline Linear Regression 실험 계획서 (Experimental Plan)

## 1. 실험 목적
14개의 시계열 통합 데이터셋(Glucose-ML-collection)에 대하여, 가장 기초적인 모형인 **선형 회귀(Linear Regression)** 와 **릿지 회귀(Ridge Regression)** 를 활용해 예측의 절대적 하한선(Lower Bound Baseline)을 구축합니다. 이를 통해 향후 고급 ML/DL 모형 성능 평가의 비교 우위 파악 기준으로 활용합니다.

## 2. 듀얼 디자인 패러다임 (A/B Test)
본 실험은 다중 모달 변수의 영향력을 정량 측정하기 위해 동일한 프레임워크 아래 2개의 분리된 계층으로 수행됩니다.
*   **A. 단변량 예측 (Univariate):** 과거 $n$ 시점의 CGM(순수 혈당치) 데이터만 투입하여 미래 혈당 예측.
*   **B. 다변량 예측 (Multivariate):** 과거 $n$ 시점의 CGM과 통합된 외부 환경 변수(식단, 수면, 인슐린 투여량 등)를 동시에 병렬로 투입하여 예측 향상성(Delta) 도출.

## 3. 시계열 학습 방법론 (Sliding Window & Data Leakage Prevention)
*   **데이터 윈도우 생성:** 과거 6 스텝(Lookback n=6)의 데이터를 관측하여, 미래 6 스텝(Horizon t=6) 시점의 특정 혈당치를 예측 타겟(`y`)으로 삼는 지도 학습(Supervised Learning) 매트릭스를 생성합니다.
*   **결측 및 점프 방어 메커니즘:** 연속된 시간축 상 센서 탈락이나 데이터 누락이 존재하여 생성된 윈도우 시퀀스의 연속성이 끊어진 경우, 해당 구간의 윈도우를 학습에서 즉시 누락(Drop)시켜 인지 오류를 차단합니다.
*   **연대기적 분할 (Chronological Split):** 시계열 누수 방지를 위해 **개별 환자 스케일**에서 연대기적 순방향으로 **초기 80%를 Train Data, 후기 20%를 Test Data**로 나눕니다. 무작위 분할(Random split)은 엄격히 금지됩니다.

## 4. 모델 세부 튜닝 및 평가 지표
*   **스케일링 적용:** `StandardScaler`를 활용하여 혈압, 인슐린 용량, 탄수화물 그램 수 등 각기 다른 다변량 벡터의 도메인 단위를 통일합니다.
*   **L2 정규화:** 변수 간 강한 다중공선성을 억제하기 위해 Alpha 패널티를 부여한 `Ridge Regression`을 주력 Baseline 메인 엔진으로 작동시킵니다.
*   **성과 추출 (Metric Extraction):**
    *   `RMSE` (상대적 제곱 평균 오차)
    *   `MAE` (순수 절대 오차)
    *   `MAPE` (%)
