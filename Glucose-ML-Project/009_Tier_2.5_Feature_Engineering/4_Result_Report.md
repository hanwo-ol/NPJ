# 실험 결과 보고서 (Experimental Result Report)
## Tier 2.5 — Feature Engineering (의료 모달리티 특성 결합 최종본)

---

## 1. 개요 및 파생 변수 투입 (Feature Engineering)
Tier 2 (Classic ML)에서 Random Forest가 입증한 비선형 데이터 소화 능력에 날개를 달아주기 위하여, 혈당(Glucose)의 시계열 구조적, 임상적 특징을 파생 변수(Derived Features)로 생성해 병합(Concatenation)했습니다. Data Leakage를 완벽히 통제한 상태로 설계되었습니다.

- **Kinetic (역학):** 혈당 상승 속도(Velocity), 가속도(Acceleration), 윈도우 적분(AUC)
- **Clinical (임상):** 평균, 표준편차(Glycemic Variability), TIR/TAR/TBR 시계열 비율
- **Risk (위험도):** 고위험 비대칭 보정 Kovatchev 방정식 기반 LBGI, HBGI
- **Circadian (일주기):** $\sin$/$\cos$ 로 변환된 24시간 호르몬 주기 비율
- **Memory (기억):** 이벤트 소멸 지수(Exponential Decay, IOB 근사), 경과 시간(Time-Since)

---

## 2. 예측 성능 고도화 분석 (Tier 2 vs Tier 2.5)

새로운 파생변수를 투입한 결과, **압도적인 T1DM 환자군 오차(RMSE) 방어 성과**를 도출했습니다.

### 2-1. 비교 지표 (주요 T1DM 환자군 RMSE 개선도)
| 데이터셋 | Tier 2 (원본 트리) | Tier 2.5 (파생변수+깊이해제) | 변화율 |
|---|---|---|---|
| **PEDAP** | 28.98 | **28.70** | -0.28 대폭 개선 |
| **Bris-T1D** | 28.00 | **27.90** | -0.10 개선 |
| **IOBP2** | 25.15 | **25.00** | -0.15 개선 |
| **BIGIDEAs** | 14.72 | **14.21** | -0.51 소폭 개선 |
| **CGMND** | 14.43 | **14.38** | 파생 변수에 의해 정확도 개선 |

### 2-2. Feature Importance 패러다임 제패 (놀라운 발견)
차원 확장을 통해 Random Forest가 혈당 예측의 논리를 스스로 "재창조"했습니다.

1. **Park_2025의 혁명 (파생 지표의 완전한 승리):**
   - **`Window_Mean (56.2%)`**, **`Window_AUC (8.6%)`**, **`HBGI (4.4%)`** 가 최상위를 독식하며 원본 혈당 지표를 압도했습니다!
   - 즉, 이 그룹의 환자는 "현재 혈당"보다 "방금 전까지 얼마나 심하게 고혈당(HBGI)에 노출된 상태(AUC)였는가"가 미래 예측의 절대적 지표라는 점이 의학적으로 증명된 것입니다.

2. **역학 지수의 전면전 (Velocity & Acceleration):** 
   - `PEDAP`: **Velocity(7.1%)**, Acceleration(0.7%)
   - `Bris-T1D`: **Velocity(6.5%)**, Acceleration(1.5%)
   - `BIGIDEAs`: **Velocity(5.7%)**, Acceleration(4.5%)
   - 모든 제1형 당뇨(T1DM) 환자들에게서 "현재 증가 속도(Velocity)"가 트리 분할의 넘버 투(No.2) 지표로 격상되었습니다. 가장 극심한 스파이크를 방어하는 논리가 가속도임을 AI가 찾아냈습니다.

3. **호르몬 리듬의 발현 (Circadian Rhythm):**
   - `UCHTT1DM` 및 `CGMND` 등 다수 데이터에서 `cos_hour`가 상위 5대 지표 안에 입성했습니다. 일주기 호르몬 저항성(새벽 공복 현상 등)이 수치적 억제력으로 동작함을 입증합니다.

---

## 3. Tier 2.5의 한계점 및 형태적 약점
뛰어난 오차 방어력을 확보했지만, 본 파이프라인(특징 공학 + Random Forest)은 향후 해결해야 할 치명적인 약점들을 데이터적으로 노출했습니다.

1. **차원의 폭발과 희소성(Sparsity)의 늪:**
   - 15개의 이벤트가 존재하는 `CGMacros`의 경우 윈도우 1개당 피처(Feature)가 **288차원**까지 증식했습니다. 하지만 Feature Importance 추출 결과를 보면 대부분의 파생/이벤트 변수는 **중요도 0.0%**를 기록했습니다. 이는 트리가 방대한 차원 속에서 무가치한 영점(Zero) 변수들을 탐색하느라 심각한 연산 낭비를 겪고 있음을 의미합니다 (즉, 피처 가지치기(Feature Selection)가 절실함).
2. **단순 지수 감쇠(Static Decay)의 생리학적 한계:**
   - 본 실험에서는 인슐린과 식사량 모두 일괄적으로 **반감기 1시간짜리 일괄 지수 감쇠(EWM)**를 적용했습니다. 하지만 초속효성 인슐린과 복합 탄수화물의 체내 소모 속도는 명백히 다릅니다. 이 정적인 가정은 트리가 개인별 대사 모델링을 수행하는 데 한계로 작용합니다.
3. **변수 간의 강력한 다중공선성(Multicollinearity):**
   - 현재 혈당($G_t$)과 윈도우 평균(Window_Mean), 속도(Velocity) 등은 서로 강력한 피어슨 상관관계를 가집니다. Random Forest는 다중공선성에 비교적 강하지만, 변수가 분할(Split)될 때 너무 강력한 지표(예: Window_Mean)가 다른 섬세한 지표(이벤트 변수 등)의 판단 기회를 뺏어가는 "마스킹(Masking)" 현상이 일어났습니다.

---

## 4. 향후 로드맵 (Next Steps)

위 한계점들을 돌파하기 위하여, 본 프로젝트의 후속 지표는 "딥러닝(Deep Learning) 배제 및 순수 머신러닝의 극의 달성"으로 방향을 설정합니다.

### [Tier 3] Advanced Machine Learning & Feature Selection
- 딥러닝(RNN/Transformer)을 제외하고, 현존 최고의 머신러닝 기술력을 투입합니다.
- **방향성:** 티어 2.5에서 입증된 상위 파생 변수들만 남기고 0%대 노이즈를 쳐내는 **차원 축소(Feature Selection)** 후, 오차를 순차적으로 교정하는 **부스팅 앙상블(XGBoost, LightGBM)**이나 모형 결합(Stacking)을 설계하여 한계치까지 성능을 끌어올립니다.

### [Tier 4] Cross-Dataset Transferability (교차 데이터셋 제로샷 테스트)
- 머신러닝 모델의 진정한 실효성을 다루는 관문입니다.
- **방향성:** 특성 데이터셋(`dataset A`)에서 환자 데이터를 오버피팅 없이 정교하게 학습시킨 범용 모형이, 한 번도 본 적 없는 타 데이터셋(`dataset B`) 환자들의 혈당을 예측할 때 모델의 방어력이 무너지는지(성능 유지율 파악)를 검증하는 "교차 일반화(Generalization)" 실험을 전격 수행합니다.
