# Glucose CGM 연구 설계: Stage 1 & Stage 2

**연구 그룹**: 김광수 교수님 — 한울 (Glucose), 현수 (Breast Cancer), 남윤 (COVID-19)
**타겟 저널**: npj Digital Medicine — *"Evaluating the Real-World Clinical Performance of AI"*
**최종 업데이트**: 2026-05-27

---

## 전체 구조

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: npj Digital Medicine 논문                          │
│  ──────────────────────────────                              │
│  "임상 AI의 Generalizability & Reproducibility 평가"         │
│                                                              │
│  방법: LightGBM + CORAL + TrAdaBoost (정적 ML)               │
│  역할: 기존 현장에서 쓰이는 수준의 ML 전이학습이               │
│        다기관 CGM 시계열에서 얼마나 재현 가능한지 평가          │
│  핵심 산출물:                                                 │
│    ① 12개 코호트 LODO generalizability 정량화                 │
│    ② Within Variation (반복 안정성)                           │
│    ③ 도메인 거리 ↔ 성능 저하 상관 분석                        │
│    ④ 정적 전이학습의 시계열 한계 식별 (잔차 ACF + 구간별 오차) │
│                                                              │
│    ④에서 식별된 한계가 Stage 2의 연구 동기가 됨              │
│              │                                                │
│              ▼                                                │
├──────────────────────────────────────────────────────────────┤
│  Stage 2: 후속 논문 (단독 or 그룹)                            │
│  ──────────────────────────────────                           │
│  "시계열 특화 전이학습으로 정적 ML의 한계 극복"                │
│                                                              │
│  방법: LSTM Pre-train/FT, PatchTST, TS2Vec 등               │
│  역할: Stage 1에서 증명된 한계를 시계열 아키텍처가             │
│        실제로 극복할 수 있는지 실증                            │
│  핵심 산출물:                                                 │
│    ① 잔차 자기상관 해소 여부                                  │
│    ② 급변 구간 오차 개선 여부                                 │
│    ③ LODO에서 target_only 초과 달성 여부                     │
└──────────────────────────────────────────────────────────────┘
```

---
---

# STAGE 1: 정적 ML 기반 Generalizability & Reproducibility 평가

## 1. 임상 시나리오 — "왜 이 연구가 현장에서 가치 있는가?"

> [!IMPORTANT]
> 모든 실험은 아래 3개 임상 시나리오 중 하나에 매핑되어야 한다. 시나리오에 매핑되지 않는 실험은 논문에 포함하지 않는다.

### 시나리오 A: Cross-Site Deployment ("다른 집단 AI를 내 환자에게 쓸 수 있나?")

```
상황: 서울 A병원이 미국 데이터로 학습된 CGM AI를 도입하려 한다.
      한국 환자 데이터는 30명뿐.
      AI 제조사는 "RMSE 15 mg/dL"이라고 주장한다.

실제로 부딪히는 질문:
  Q1. 이 AI를 우리 환자에게 그대로 쓰면 성능이 얼마나 떨어지나?
  Q2. 30명 데이터로 fine-tune하면 충분한가?
  Q3. 성능 저하를 실험 전에 예측할 수 있나? (도메인 거리)
  Q4. 어떤 환자에서 특히 위험한가? (급변 구간)

현실적 근거:
  - FDA AI/ML-SaMD 가이드라인에서 "multi-site validation" 요구
  - K-MFDS '인공지능 기반 의료기기 허가·심사 가이드라인'에서 
    "다기관 데이터 검증" 권고
  - Dexcom G7, Medtronic Guardian 4 등 실제 CGM AI 탑재 기기 존재
```

**매핑되는 실험**: LODO, 5-Way 비교, 도메인 거리 분석, 잔차 ACF 분석

---

### 시나리오 B: Cold Start ("신규 환자에게 며칠 후부터 AI를 신뢰할 수 있나?")

```
상황: 환자가 어제 처음 CGM을 착용했다.
      AI 앱이 "30분 뒤 혈당"을 예측해준다.
      이 환자의 데이터는 24시간치뿐.

실제로 부딪히는 질문:
  Q1. 범용 모델의 예측을 처음 며칠간 믿어도 되나?
  Q2. 개인화 모델이 범용 모델을 이기려면 며칠이 필요한가?
  Q3. 전이학습이 이 "교차점"을 앞당길 수 있나?

현실적 근거:
  - CGM 기기에는 "warm-up period" (1~2시간)가 있지만,
    AI 예측 모델의 "learning period"는 정의되어 있지 않음
  - 환자가 가장 불안한 시기 = 착용 초기
  - FDA: "device learning period" 개념이 존재하나 정량 기준 미비
```

**매핑되는 실험**: 학습 곡선 (기존 Tier 7), Cold Start 교차점 분석 (기존 Tier 7.1)

---

### 시나리오 C: Reproducibility Audit ("보고된 AI 성능을 재현할 수 있는가?")

```
상황: 논문 A에서 "LightGBM으로 RMSE 15 mg/dL 달성"이라고 보고.
      다른 연구자가 같은 데이터, 같은 모델로 재현하니 RMSE가 17~21.
      
실제로 부딪히는 질문:
  Q1. 같은 데이터+모델에서 seed/HP를 바꾸면 결과가 얼마나 흔들리나?
  Q2. "전이학습이 target_only보다 좋다"는 결론이 뒤집히는 빈도는?
  Q3. 어떤 모델/조건에서 재현성이 가장 취약한가?

현실적 근거:
  - Nature (2016): "70%의 연구자가 다른 사람의 실험을 재현 실패"
  - npj 컬렉션 핵심 주제: "Transparency and Reproducibility"
  - AI-SaMD 규제에서 "post-market surveillance" 요구 증가
```

**매핑되는 실험**: Within Variation (seed × HP 반복)

---

## 2. 기존 실험 결과 재활용 — 이미 가진 것

| 완료 항목 | 파일 위치 | 시나리오 매핑 | 논문에서의 역할 |
|:---|:---|:---|:---|
| 12개 코호트 전처리 | `002_Harmonize-cgm-datasets/` | 전체 | 데이터 설명 (Methods) |
| 5-Way 비교 (T1D→T2D/Mixed) | [Tier7_Results_Analysis.md](file:///c:/Users/11015/Documents/NPJ2/Glucose-ML-Project/016_Tier_7_Cross_Disease/Tier7_Results_Analysis.md) | A | Cross-domain 성능 저하 정량화 |
| Negative transfer 정량화 | Tier 7 결과표 | A | 핵심 발견 1 |
| CORAL / TrAdaBoost 비교 | [tier6_transfer_utils.py](file:///c:/Users/11015/Documents/NPJ2/Glucose-ML-Project/013_Tier_6_Domain_Adaptation/tier6_transfer_utils.py) | A | 도메인 적응 기법 효과 |
| 학습 곡선 (ShanghaiT2DM) | [experiment.log](file:///c:/Users/11015/Documents/NPJ2/Glucose-ML-Project/016_Tier_7_Cross_Disease/tier7_results/experiment.log) | B | Cold Start 분석 |
| 동일 질병 전이 대조 실험 | [Tier7.1_Conclusions_and_Next_Steps.md](file:///c:/Users/11015/Documents/NPJ2/Glucose-ML-Project/017_Tier_7.1_Clinical_Transfer/Tier7.1_Conclusions_and_Next_Steps.md) | A | 도메인 갭 vs 기법 한계 분리 |
| Clarke Error Grid + 저혈당 sensitivity | Tier 7.1 | A | 임상 안전성 |

> [!NOTE]
> 기존 Tier 7/7.1 실험 결과는 **그대로 재활용** 가능. 프레이밍만 학술적("T1D→T2D 전이 가능한가?")에서 임상적("다른 집단 AI를 내 환자에게 적용하면?")으로 전환.

---

## 3. 추가 실험 설계

### 실험 S1-1: Within Variation — Computational Reproducibility

**시나리오 매핑**: C (Reproducibility Audit)

**목적**: 같은 데이터·모델에서 학습 조건(seed, HP)을 바꾸면 결과와 임상적 결론이 얼마나 흔들리는지 정량화

```
고정 요소:
  - 데이터: ShanghaiT2DM (타겟) + T1D pool (소스)
  - 모델: LightGBM
  - 피처: 22개 (기존과 동일)
  - 비교 구조: 5-Way (source_only, target_only, mixed, coral, tradaboost)

변동 요소:
  ① Random seed: 10개 (42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999)
  ② Hyperparameter grid (선택적 확장):
     - learning_rate: {0.01, 0.05, 0.1}
     - num_leaves: {31, 63, 127}
     - max_depth: {-1, 6, 10}

실행 규모:
  - 최소: 10 seeds × 5 models = 50회
  - 확장: 10 seeds × 5 models × 9 HP combos = 450회

평가 지표:
  ① 성능 변동성
     - RMSE의 변동계수 (CV = σ/μ) — 각 모델별
     - RMSE 분포의 IQR (사분위 범위)

  ② 결론 안정성
     - 순위 일관성: 50회 중 "tradaboost > target_only"가 성립하는 비율
     - 결론 뒤집힘 빈도 (flip rate):
       "전이학습이 target_only보다 좋다"가 뒤집히는 횟수 / 전체 횟수
     - Kendall's τ: 5-Way 순위의 반복 간 일치도

  ③ 임상적 결론 안정성
     - Clarke Error Grid Zone A 비율의 변동 범위
     - 저혈당 sensitivity의 변동 범위

산출물:
  - 표: 모델별 RMSE (mean ± std), CV, flip rate
  - 그림: violin plot (5 모델 × RMSE 분포)
  - 그림: flip rate bar chart
```

**구현 난이도**: ⭐⭐ (기존 코드에 for loop 추가)

---

### 실험 S1-2: LODO — Cross-Domain Generalizability

**시나리오 매핑**: A (Cross-Site Deployment)

**목적**: 12개 코호트를 체계적으로 순환하여, "내 병원 데이터가 학습에 없었을 때" 성능이 얼마나 저하되는지 정량화

```
데이터: 12개 코호트 전체
  T1D:   RT-CGM, IOBP2, FLAIR, SENCE, WISDM, PEDAP
  T2D:   ShanghaiT2DM
  Mixed: CITY, Colas_2019
  ND:    CGMacros, Hall_2018, ShanghaiT1DM (소규모)

프로토콜:
  for i in 1..12:
    Train: D_1 ∪ ... ∪ D_{i-1} ∪ D_{i+1} ∪ ... ∪ D_12
    Test:  D_i

    모델별 학습:
      - Direct (나머지 합산 → D_i 테스트)    = Source-Only
      - D_i의 70%로만 학습                    = Target-Only (Self)
      - Direct + CORAL 정렬                   = CORAL
      - Direct + TrAdaBoost                   = TrAdaBoost

    기록: RMSE_i, MAE_i, MARD_i

주의사항:
  - 샘플링 주기(5분/15분) 그룹 분리 유지 (Glucose-ML 원칙)
  - 소스 풀 서브샘플링 비율 통일 (e.g., 최대 200만 windows)
  - 5분 그룹과 15분 그룹의 LODO를 별도로 수행

산출물:
  - 히트맵: 12 targets × 4 methods (RMSE 값, 색상 코딩)
  - 표: 코호트 특성(질병, 환자 수, 센서, 국가)별 LODO RMSE 분포
  - 산점도: (코호트 크기 or 질병 유형) vs. LODO 성능 저하량
```

**구현 난이도**: ⭐⭐⭐ (기존 5-Way 코드를 반복 호출, 데이터 분할 자동화 필요)

---

### 실험 S1-3: 도메인 거리 ↔ 성능 저하 예측 가능성

**시나리오 매핑**: A (Cross-Site Deployment) — "실험 전에 성능 저하를 예측할 수 있나?"

**목적**: 소스-타겟 간 분포 거리와 성능 저하량의 상관관계를 측정. 도메인 거리만으로 전이 성능을 사전에 예측할 수 있는지 검증.

```
입력: 실험 S1-2 (LODO)의 모든 (소스, 타겟) 쌍

각 쌍에 대해 도메인 거리 계산:
  ① MMD (Maximum Mean Discrepancy)
     - RBF 커널, 피처 공간에서 소스-타겟 분포의 평균 차이
     - 기존 tier6_transfer_utils.py의 TCA에서 MMD 행렬 L 이미 구현됨

  ② Proxy-A-Distance (PAD)
     - 소스 vs 타겟을 이진 분류하는 SVM/LR의 정확도
     - PAD = 2(1 - 2 × error)
     - 완벽히 분류 가능 = 도메인 완전히 다름 (PAD → 2)

  ③ 피처 공분산 Frobenius Norm
     - ||Cov_source - Cov_target||_F
     - CORAL이 정렬하려는 대상 자체의 크기

  ④ Wasserstein Distance (optional)
     - 최적 운송 거리, 해석 용이

성능 저하량:
  ΔPerformance_i = LODO_RMSE_i - Self_RMSE_i

상관 분석:
  - Spearman ρ(도메인 거리, ΔPerformance) — 순위 상관
  - 선형 회귀: ΔPerformance ~ α + β × 도메인 거리
  - R² 보고: "도메인 거리가 성능 저하의 X%를 설명한다"

산출물:
  - 산점도: X축 = MMD, Y축 = ΔPerformance, 점 = 코호트
  - 표: 4개 거리 측도별 Spearman ρ 및 p-value
  - 논문 핵심 Figure 후보
```

**구현 난이도**: ⭐⭐⭐ (거리 계산 코드 신규 작성, 통계 분석)

---

### 실험 S1-4: 정적 전이학습의 시계열 한계 식별

**시나리오 매핑**: A (Cross-Site Deployment) — "현재 AI가 놓치고 있는 패턴은 무엇인가?"

> [!IMPORTANT]
> 이 실험이 Stage 1과 Stage 2를 연결하는 **브릿지**이다. "한계가 있을 것이다"라는 추측이 아닌, 데이터에서 한계를 **정량적으로 증명**해야 한다.

#### S1-4a: 잔차의 시간적 자기상관 분석 (Residual Autocorrelation)

```
핵심 논리:
  - 모델이 시계열 패턴을 완벽히 학습했다면,
    예측 오차(잔차)는 랜덤해야 한다 (백색 잡음, white noise)
  - 잔차에 시간적 자기상관이 남아 있다면,
    = "연속된 시점에서 같은 방향으로 틀린다"
    = 모델이 포착하지 못한 시간적 패턴이 존재한다는 정량적 증거

프로토콜:
  대상 모델: 5-Way 전체 (source_only, target_only, mixed, coral, tradaboost)
  대상 타겟: ShanghaiT2DM, CITY, Colas_2019 (3개 모두)

  ① 잔차 시계열 추출
     for each model M, target T:
       residual[t] = y_true[t] - y_pred_M[t]
       # 주의: test set 내에서 환자별·시간순 정렬 유지

  ② 자기상관 함수(ACF) 계산
     ACF(lag) for lag = 1, 2, 3, ..., 12
     # lag=1: 1 스텝(5분 or 15분) 전 잔차와의 상관
     # lag=12: 1시간 전 잔차와의 상관

  ③ 통계 검정
     Ljung-Box Q 검정:
       H0: "잔차는 i.i.d. (시간적 구조 없음)"
       유의수준: α = 0.01
       기각 → 잔차에 시간 구조가 남아 있음 = 정적 모델의 한계

  ④ 정량적 지표
     - ACF(lag=1) 값 자체: 0에 가까우면 시간 구조 없음, 높으면 한계
     - Durbin-Watson 통계량: 2에 가까우면 자기상관 없음

기대 결과:
  - 정적 전이 모델의 ACF(lag=1) ≈ 0.3~0.5 (높은 자기상관)
  - Ljung-Box p < 0.001 (강력히 기각)
  - = "정적 모델은 혈당의 관성(momentum)을 체계적으로 놓친다"

산출물:
  - ACF 플롯: 5개 모델 × 3개 타겟 (15개 패널)
  - 표: 모델별 ACF(lag=1), Ljung-Box p-value, Durbin-Watson
```

#### S1-4b: 혈당 변동 구간별 오차 분해 (Segment-wise Error Decomposition)

```
핵심 논리:
  - 정적 모델은 "지금 혈당이 높다/낮다"는 인식 가능
  - 그러나 "지금 올라가는 중이다/내려가는 중이다(velocity)"는 
    수동 피처로만 부분 보상
  - velocity 구간별로 오차를 분해하면,
    동적 구간에서 정적 전이학습이 더 취약한지 직접 확인 가능

프로토콜:
  ① velocity 계산
     velocity[t] = (glucose[t] - glucose[t-1]) / Δt
     # 5분 그룹: Δt = 5분, 15분 그룹: Δt = 15분

  ② 구간 분류
     안정 구간:  |velocity| ≤ 1 mg/dL/Δt
     상승 구간:  velocity > 2 mg/dL/Δt
     하강 구간:  velocity < -2 mg/dL/Δt
     전이 구간:  1 < |velocity| ≤ 2 mg/dL/Δt

  ③ 구간별 오차 계산
     for each segment S in {안정, 상승, 하강, 전이}:
       for each model M:
         RMSE_M_S = RMSE(y_true[S], y_pred_M[S])

  ④ 교차 분석
     - 정적 전이 모델(CORAL)과 target_only의 오차 비율:
       ratio = RMSE_coral_S / RMSE_target_only_S
     - 안정 구간에서 ratio ≈ 1.0 이고 급변 구간에서 ratio >> 1.0이면
       = "정적 전이학습은 급변 구간에서 특히 취약하다"

기대 결과:
  ┌─────────────────────────────────────────────────┐
  │ 구간        │ target_only │ CORAL  │ 비율(ratio) │
  ├─────────────────────────────────────────────────┤
  │ 안정        │  8.2        │ 8.5    │ 1.04 (유사) │
  │ 상승 (급변) │ 25.1        │ 31.4   │ 1.25 (격차) │
  │ 하강 (급변) │ 22.8        │ 29.7   │ 1.30 (격차) │
  └─────────────────────────────────────────────────┘

산출물:
  - Grouped bar chart: 구간별 × 모델별 RMSE
  - 논문 핵심 주장: "정적 전이학습의 오차는 혈당 급변 구간에서
    target_only 대비 X% 증가하며, 이는 시간적 의존성을
    모델링하지 못하는 구조적 한계에 기인한다."
```

#### S1-4의 논문 활용 — Stage 1 Discussion에 들어갈 문단

> *"본 연구에서 적용한 CORAL 및 TrAdaBoost 기반 도메인 적응은 negative transfer를 해소하여 target_only 수준의 성능을 회복시켰으나, target_only를 유의미하게 초과하지 못했다. 이 한계의 원인을 잔차 분석으로 조사한 결과, 전이학습 모델의 예측 잔차는 시간적 자기상관이 통계적으로 유의하게 존재하였다 (Ljung-Box p < 0.001, ACF(lag=1) = X.XX). 특히, 혈당 변동 속도가 높은 구간(|velocity| > 2 mg/dL/5min)에서 정적 전이 모델의 오차는 target_only 대비 X% 증가하였다. 이는 CORAL과 TrAdaBoost가 데이터 포인트를 i.i.d.로 가정하여 시계열의 시간적 의존성을 구조적으로 무시하기 때문이다. 시간적 표현 학습(temporal representation learning)을 통합한 전이학습 프레임워크가 이 한계를 극복할 수 있는지는 후속 연구에서 검증이 필요하다."*

---

### 실험 S1-5: Regression → Classification 전환

**시나리오 매핑**: A, C (3개 파트 통일)

**목적**: 교수님 피드백 "classification 전환 필요" 반영. 3개 파트(이미지/테이블/시계열)의 재현성 지표를 통일.

```
분류 기준 (임상 표준):
  - 저혈당 (Hypoglycemia): CGM < 70 mg/dL
  - 정상 (In-Range):       70 ≤ CGM ≤ 180 mg/dL
  - 고혈당 (Hyperglycemia): CGM > 180 mg/dL

구현:
  기존 regression 파이프라인 유지
  → 예측값(연속)을 3분류로 이산화
  → 분류 지표 계산

평가 지표:
  - Accuracy, Macro F1-score, AUC-ROC (one-vs-rest)
  - Cohen's Kappa ← 3개 파트 공통 재현성 척도
  - 클래스별 Sensitivity / Specificity
  - 주의: 클래스 불균형 극심 (저혈당 1~5%, 정상 80~90%, 고혈당 5~15%)

Within Variation과 결합:
  - 10 seeds × 5 models → Kappa 분포
  - "Kappa가 seed에 따라 얼마나 흔들리는가?"
```

**구현 난이도**: ⭐⭐ (예측값 임계치 적용 + sklearn metrics)

---

## 4. Stage 1 산출물 요약 및 논문 구조

### 핵심 Figure 후보

| Figure | 내용 | 시나리오 |
|:---|:---|:---|
| **Fig 1** | 12개 코호트 LODO 히트맵 (12 targets × 4 methods) | A |
| **Fig 2** | 도메인 거리(MMD) vs. 성능 저하(ΔRMSE) 산점도 + 회귀선 | A |
| **Fig 3** | Within Variation violin plot (5 models × RMSE 분포) | C |
| **Fig 4** | 잔차 ACF 플롯 (시계열 한계 증거) | A → Stage 2 브릿지 |
| **Fig 5** | 구간별(안정/급변) 오차 분해 bar chart | A → Stage 2 브릿지 |
| **Fig 6** | 학습 곡선 (Cold Start 교차점) | B |

### Glucose 섹션 논문 구조

```
Methods
  - 12개 코호트 설명 + 피처 22개
  - 5-Way 비교 프로토콜
  - LODO 프로토콜
  - Within Variation 프로토콜
  - 도메인 거리 계산 방법
  - 잔차 분석 방법

Results
  3.x.1 Cross-Domain Generalizability (LODO) — Fig 1
  3.x.2 도메인 거리로 성능 저하 예측 가능한가? — Fig 2
  3.x.3 Computational Reproducibility — Fig 3
  3.x.4 Cold Start: 며칠 후부터 신뢰 가능한가? — Fig 6
  3.x.5 정적 전이학습의 시계열 한계 — Fig 4, 5

Discussion
  - "정적 ML 전이학습은 12개 코호트에서 ~한 generalizability를 보인다"
  - "도메인 거리(MMD)는 성능 저하의 X%를 설명한다"
  - "그러나 잔차에 시계열 구조가 남아있어, 시간적 모델링이 필요하다"
  → Stage 2의 연구 동기
```

---

## 5. Stage 1 타임라인

| 주차 | 작업 | 실험 ID | 산출물 |
|:---|:---|:---|:---|
| **W1 (5/27~6/2)** | Within Variation (10 seeds × 5-Way) | S1-1 | RMSE 분포 + flip rate 표 |
| | Classification 전환 코드 | S1-5 | 3분류 파이프라인 |
| **W2 (6/3~6/9)** | LODO (12코호트 순환) | S1-2 | 12×4 히트맵 |
| | 잔차 ACF 분석 | S1-4a | ACF 플롯 + Ljung-Box 표 |
| **W3 (6/10~6/16)** | 도메인 거리 계산 + 상관 분석 | S1-3 | 산점도 + R² |
| | 구간별 오차 분해 | S1-4b | Grouped bar chart |
| **W4 (6/17~6/23)** | 결과 통합 + 시각화 | — | 논문용 Figure 초안 |
| **W5~** | 3개 파트 통합 + 논문 작성 | — | 원고 |

---
---

# STAGE 2: 시계열 특화 전이학습으로 정적 ML의 한계 극복

## 1. 연구 동기 — Stage 1에서의 직접적 연결

Stage 2는 Stage 1에서 **데이터로 증명된** 다음 2가지 한계를 극복하는 것이 목표:

| Stage 1 발견 (증거) | Stage 2 질문 |
|:---|:---|
| 잔차 ACF(lag=1) = X.XX (자기상관 존재) | **시계열 아키텍처가 이 자기상관을 해소할 수 있는가?** |
| 급변 구간에서 CORAL 오차 +X% 증가 | **시간적 도메인 적응이 급변 구간 오차를 줄일 수 있는가?** |
| CORAL/TrAdaBoost가 target_only를 초과하지 못함 | **시계열 표현 학습이 target_only를 넘을 수 있는가?** |

> [!WARNING]
> Stage 2는 Stage 1의 잔차 분석 결과가 나온 후에 구체적 방법론을 확정해야 한다. ACF 패턴의 구조(어떤 lag에서 강한지, 코호트별 차이 등)에 따라 적합한 시계열 아키텍처가 달라진다.

---

## 2. 방법론 후보 (문헌 조사 기반)

문헌 조사 결과 ([literature_review_timeseries_transfer_learning.md](file:///C:/Users/11015/.gemini/antigravity-ide/brain/7aadef9f-848e-42b6-ae73-4370ced3e16f/literature_review_timeseries_transfer_learning.md))에서 도출된 4개 패러다임:

### Tier S2-A: 하이브리드 접근 (시계열 표현 + 기존 트리)

**난이도**: ⭐⭐ | **기존 인프라 활용도**: 높음

```
아이디어: 시계열 표현만 신경망으로 추출하고,
         분류/예측은 기존 LightGBM 파이프라인 유지

방법:
  ① TS2Vec (대조 학습)로 CGM 윈도우의 표현 벡터 추출
  ② 표현 벡터를 기존 22개 피처에 추가 (concat)
  ③ LightGBM으로 학습

장점:
  - 기존 파이프라인 최소 변경
  - TS2Vec 표현이 시간적 구조를 인코딩하므로
    ACF 잔차 한계를 일부 해소할 가능성
  - Stage 1과 직접 비교 가능 (동일 모델, 피처만 확장)

평가:
  - 잔차 ACF 재계산: TS2Vec 피처 추가 후 ACF(lag=1) 감소?
  - 급변 구간 오차 재측정: 개선되었는가?
  - LODO에서 target_only 초과 달성?
```

### Tier S2-B: LSTM Population → Personalized Fine-tuning

**난이도**: ⭐⭐⭐ | **CGM 도메인 검증**: 가장 많음

```
아이디어: CGM 전이학습에서 가장 많이 검증된 패턴.
         대규모 인구 데이터로 LSTM 사전학습 → 개별 타겟에 fine-tune

방법:
  ① 12개 코호트 전체로 Bi-LSTM 사전학습 (population model)
  ② 타겟 코호트 데이터로 fine-tune (lower learning rate)
  ③ Stage 1과 동일한 LODO 프로토콜로 평가

비교 대상:
  - Stage 1의 LightGBM LODO 결과 (baseline)
  - LSTM without pre-training (scratch)
  - LSTM with pre-training (transfer)

핵심 검증:
  "LSTM pre-train/FT가 Stage 1의 LightGBM + CORAL보다
   LODO에서 유의미하게 좋은가?"
```

### Tier S2-C: PatchTST Self-supervised Pre-training

**난이도**: ⭐⭐⭐⭐ | **최신성**: 높음 (ICLR 2023)

```
아이디어: CGM 윈도우를 패치로 분할 → 일부 마스킹 → 재구성 학습
         = 시계열의 BERT

방법:
  ① 12개 코호트 전체의 CGM 윈도우로 PatchTST 사전학습
     (마스킹 패치 재구성, 레이블 불필요)
  ② 타겟 코호트에 fine-tune
  ③ LODO 프로토콜로 평가

장점:
  - 레이블 없이 사전학습 가능 → 더 많은 데이터 활용
  - 5/13 미팅에서 교수님이 "PatchTST" 직접 언급
  - 시계열 구조를 네이티브로 학습

도전:
  - 구현 복잡도 높음 (PyTorch + Transformer)
  - 하이퍼파라미터 튜닝 부담
  - 학습 시간 증가
```

### Tier S2-D: CGM 특화 파운데이션 모델 (CGMformer / GluFormer)

**난이도**: ⭐⭐⭐⭐⭐ | **임팩트**: 가장 높음

```
아이디어: CGM 데이터에 특화된 기 학습 파운데이션 모델 활용

방법:
  - CGMformer 또는 GluFormer의 공개 가중치가 있다면:
    → fine-tune만 수행
  - 공개 가중치가 없다면:
    → 12개 코호트로 자체 사전학습 (대규모 컴퓨팅 필요)

현실적 판단:
  - 공개 가중치 확인 필요 (2026-05 기준)
  - 자체 학습 시 GPU 자원 및 시간 투자 큼
  - 논문 임팩트는 가장 크나 실현 가능성 불확실
```

---

## 3. Stage 2 실험 설계

### 핵심 원칙: Stage 1과의 동일 평가 프레임워크

Stage 2의 모든 실험은 Stage 1과 **동일한 프로토콜**로 평가해야 직접 비교 가능:

```
공통 평가 요소:
  - 동일 12개 코호트
  - 동일 LODO 프로토콜
  - 동일 평가 지표 (RMSE, MARD, Kappa)
  - 동일 잔차 분석 (ACF, 구간별 오차)

Stage 2 고유 비교:
  ┌─────────────────────────────────────────────────┐
  │ Baseline (Stage 1)          │ Proposed (Stage 2) │
  ├─────────────────────────────────────────────────┤
  │ LightGBM + CORAL            │ TS2Vec + LightGBM  │
  │ LightGBM + TrAdaBoost       │ LSTM Pre-train/FT  │
  │ LightGBM target_only        │ PatchTST FT        │
  └─────────────────────────────────────────────────┘

성공 기준:
  ① 잔차 ACF(lag=1)이 Stage 1 대비 유의하게 감소
  ② 급변 구간 오차가 Stage 1 대비 유의하게 감소
  ③ LODO에서 target_only를 유의하게 초과 (paired t-test p < 0.05)
```

### 추천 실행 순서

```
Phase 1: TS2Vec + LightGBM 하이브리드 (Tier S2-A)
  → 가장 빠르게 결과 확인 가능
  → 기존 파이프라인 변경 최소

Phase 2: LSTM Pre-train/FT (Tier S2-B)
  → CGM 도메인에서 가장 많이 검증된 방법
  → Phase 1 대비 얼마나 나은지 비교

Phase 3: PatchTST (Tier S2-C)
  → Phase 2보다 나은 결과가 필요한 경우에만
  → 교수님 관심 반영
```

---

## 4. Stage 2 예상 타임라인

| 시기 | 작업 | 산출물 |
|:---|:---|:---|
| Stage 1 완료 후 1~2주 | TS2Vec 표현 추출 + LightGBM 결합 | ACF 비교, LODO 비교 |
| +2~4주 | LSTM Pre-train/FT 구현 + LODO | 12코호트 결과 |
| +4~6주 | PatchTST 구현 (선택적) | 최종 비교 |
| +6~8주 | 논문 작성 | 원고 |

---
---

# 부록: 교수님 피드백 대응 매핑 (전체)

| 미팅 일자 | 교수님 피드백 | Stage | 대응 실험 |
|:---|:---|:---|:---|
| 3/4 | "성능보다 변동성에 초점" | 1 | S1-1 (Within Variation) |
| 4/15 | "LODO 시나리오" | 1 | S1-2 (LODO) |
| 4/15 | "classification 전환 필요" | 1 | S1-5 (3분류 전환) |
| 4/15 | "regression: GT 아닌데 의미 있냐?" | 1 | 분류 전환 + CGM FDA 인증 기기 논거 |
| 4/29 | "시나리오 더 정교하게" | 1 | 임상 시나리오 A/B/C 재정의 |
| 4/29 | "전이학습 알고리즘 확정" | 1 | CORAL + TrAdaBoost 유지 (정적) |
| 5/13 | "PatchTST" | 2 | Tier S2-C |
| 5/13 | "time series TL / domain adaptation 비교" | 2 | Stage 2 전체 |
