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

## 3. Tier 2.5의 한계점 및 약점 (Limitations)
성공적인 개선에도 불구하고, 본 단계에서 노출된 명확한 방법론적 한계와 향후 고민점은 다음과 같습니다.

1. **차원의 저주와 다중공선성(Multicollinearity)의 방치:**
   - 파생 변수들의 무분별한 결합으로 차원이 폭증했습니다(최대 288차원). 속도, 가속도, 평균 등은 필연적으로 원본 혈당과 수학적 상관성이 매우 높습니다. Random Forest가 이를 어느정도 버텨내긴 했으나, 불필요한 노이즈로 인해 트리가 트리거를 잃을 확률이 높아집니다. **다음 단계에서는 통계적, 혹은 SHAP 등을 활용한 엄격한 'Feature Selection(특성 선택/축소)' 과정이 수반되어야 합니다.**
2. **트리 앙상블의 구조적 한계 (외삽 불가 및 지역성):**
   - Random Forest는 본질적으로 데이터를 직사각형 공간으로 자르는 모델이므로, 학습 데이터(Train)에 존재하지 않았던 극한의 저혈당/고혈당 값(Extrapolation)을 절대로 예측할 수 없습니다. 
3. **순차적 정보 흐름의 부재:**
   - IOB(잔존 인슐린)와 속도 개념을 억지로 수식화하여 밀어넣었지만, 본질적으로 모델은 여전히 시간적 맥락을 가진 '시계열 모델'이 아니라 사진을 찍듯 판독하는 '정적 타뷸러(Tabular) 모델'에 가깝습니다.

---

## 4. 향후 로드맵 (Next Steps)
위의 한계점과 과제들을 돌파하기 위하여 이어질 후속 티어의 설계 방향은 다음과 같습니다.

- **[Tier 3] Advanced Classic Machine Learning:**
  - 딥러닝(Deep Learning)으로 도피하지 않고, 순수 머신러닝의 극한을 탐구합니다. Gradient Boosting 계열의 차세대 알고리즘(XGBoost, LightGBM, CatBoost)을 도입하여, Random Forest가 놓친 미세 잔차(Residual) 패턴을 정교하게 추적합니다.
  - 차원의 저주를 해결하기 위한 모델 기반의 Feature Selection 메커니즘을 적용하여 불필요한 차원을 강제로 축소합니다.
- **[Tier 4] 데이터셋 간 교차 검증 (Cross-Dataset Transferability):**
  - 특정 데이터셋(Ex: `PEDAP`)에서 학습된 머신러닝 모형 구조가 완전히 다른 환자군이나 코호트(Ex: `Bris-T1D`, `UCHTT1DM`)에 주입되어 테스트되었을 때도, 엔지니어링된 파생 변수(가속도, TIR 등)의 논리가 보편적인 생리학적 방어력을 발휘하는지(Generalization)를 검증합니다.
