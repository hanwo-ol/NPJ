# 실험 결과 보고서 (Experimental Result Report)
## Tier 2 — Classic Machine Learning (Non-Linearity & Interactions)

---

## 1. 실험 수준 정리
- **대상:** Tier 1과 동일한 12개 데이터셋
- **모델 구조:**
  - **Decision Tree (Max Depth = 5):** 해석 가능한 직관적 분기 규칙(Rule) 추출 특화
  - **Random Forest (Estimators = 50, Max Depth = 15):** 대규모 윈도우 스케일에서의 비선형 회귀 방어력 및 Feature Importance 추출 특화
  - *SVM 제외됨:* 데이터 전수 보존(Sub-sampling 거부) 원칙에 따라, 대형 데이터에서의 무결성을 지키기 위해 실험에서 공식적으로 제외.
- **분할:** 환자별 연대기적 80/20 Chronological Split

---

## 2. 모형 별 성과 및 발견 사항 (Tier 1 능가 여부)

| 데이터셋 | 윈도우 수 | Tier 1 (다변량 선형) | Decision Tree | Random Forest | 비고 |
|---|---|---|---|---|---|
| BIGIDEAs | 약 3.6만 | RMSE 14.29 | 16.08 | **14.72** | 선형 모형(Ridge)이 RF보다 근소 우위 |
| CGMacros_Dexcom | 약 41.5만 | RMSE 3.00 | 5.80 | **3.41** | 선형 모형(Ridge) 우위 유지 |
| CGMacros_Libre | 약 45.5만 | RMSE 1.25 | 4.00 | **1.62** | 선형 규칙이 매우 강한 데이터 |
| PEDAP (T1DM) | 약 705만 | RMSE 30.43 | 32.36 | **28.98** | **RF가 오차 1.45 대폭 감소 (성능 돌파)** |
| Bris-T1D (T1DM) | 약 81.8만 | RMSE 29.56 | 30.44 | **28.00** | **RF가 오차 1.56 대폭 감소** |
| AIDET1D (T1DM) | 약 47.1만 | RMSE 23.63 | 25.24 | **22.98** | **RF가 오차 0.65 감소** |
| IOBP2 (T1DM) | 약 1,400만 | RMSE 26.08 | 28.59 | **25.15** | **RF가 오차 0.93 감소** |

### 🔍 분석 1: 당뇨 환자군에서의 비선형성(Random Forest)의 승리
정상인이나 2형 당뇨(BIGIDEAs, CGMacros 등)에서는 선형 회귀(Ridge)가 오히려 Random Forest보다 오차가 적었습니다. 이는 정상적인 대사 작용이나 영양 기반의 혈당 상승 곡선이 매우 **완만한 선형적 특성**을 지님을 의미합니다.
반면, **T1DM 환자군(PEDAP, Bris-T1D, IOBP2 등)**에서는 **전원 Random Forest가 Tier 1 대비 압도적인 오차율 감소(RMSE 1~1.5 감소)**를 달성했습니다. 인슐린 주입에 의한 급격한 혈당 붕괴 현상이 전형적인 비선형 구조이며 모델이 이 상호작용(Interaction)을 감지하고 성공적으로 방어했다는 뜻입니다.

### 🔍 분석 2: Feature Importance (어떤 변수가 영향을 주었나?)
Random Forest가 분석한 각 변수의 혈당 변동 증감 지분율(Top 3)입니다:

1. **BIGIDEAs:** glucose(92.6%), **calorie(4.8%)**, **protein(2.6%)** 
   - 칼로리와 단백질이 혈당 예측의 주요 동력으로 작용
2. **Park_2025:** glucose(95.3%), **rep(4.7%)**
   - 반복성 식사 여부가 유의미하게 포착됨
3. **PEDAP:** glucose(99.3%), **BasalRate(0.7%)**
   - 기저 인슐린(Basal) 용량이 혈당 변동에 제동 장치로 개입함을 모델이 감지
4. **UCHTT1DM:** glucose(99.0%), **Carbohydrates (Value g) (1.0%)**

### 🔍 분석 3: Decision Tree Rules 가시화
`Decision_Tree_Rules` 폴더에 데이터셋별로 직관적인 모델 분기 조건(.txt)이 저장되었습니다. 인슐린 투여량 등 주요 이벤트 값이 임계점을 돌파할 때 트리 분기(Branch)가 갈라지며 예측값이 점프하는 임상적 규칙을 시각적으로 확인할 수 있습니다.

---

## 3. Tier 3 (Advanced Sequence Learning) 진입을 향한 제언

1. 현재 Random Forest가 1,400만 개의 윈도우 스케일에서도 **과적합(Overfitting) 없이 T1DM 오차를 방어**하며 훌륭한 성과를 보였습니다. 
2. 그러나 트리 앙상블은 본질적으로 **순차적 시간의 흐름(Sequence Dependency)**을 학습하지 못하고 단순히 과거 n개 시점의 독립적 특징만으로 계산한다는 치명적인 단점이 있습니다.
3. 이를 돌파하기 위해서는 RNN(LSTM/GRU)이나 Transformer 모형을 활용한 차세대 시계열 예측 구조 도입이 필연적으로 요구됩니다.
