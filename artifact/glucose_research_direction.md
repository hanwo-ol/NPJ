# Glucose 파트 연구 방향: Generalizability & Reproducibility

**논문 타겟**: npj Digital Medicine — *"Evaluating the Real-World Clinical Performance of AI"*
**마감**: 2026/06/03 (유동적, ~3개월)
**다음 회의**: 2026-05-27

---

## 1. 큰 그림: 3개 파트는 어떻게 연결되는가?

연구 그룹 전체가 하나의 통합 논문을 쓴다면, 핵심 메시지는 이것입니다:

> **"높은 성능을 보이는 임상 AI 모델이라도, 데이터·도메인·학습 조건이 바뀌면 성능과 임상적 결론이 얼마나 흔들리는가?"**

각 파트는 **서로 다른 데이터 모달리티**에서 이 질문을 검증합니다:

```
┌─────────────────────────────────────────────────────────────┐
│            통합 논문: AI Reproducibility & Generalizability  │
├──────────────┬──────────────┬───────────────────────────────┤
│ Breast Cancer│ COVID-19     │ Glucose (CGM)                 │
│ (이미지)      │ (테이블형)    │ (시계열)                      │
│              │              │                               │
│ CNN/DL       │ TabPFN       │ LightGBM (→ 시계열 모델?)     │
│ Detection    │ Classification│ Regression → Classification  │
│              │              │                               │
│ 현수님       │ 남윤님        │ 한울님 (본인)                  │
├──────────────┴──────────────┴───────────────────────────────┤
│ 공통 평가 축:                                                │
│  ① Within Variation (같은 데이터, 다른 학습 조건)             │
│  ② Between Variation (다른 데이터셋 간 전이/일반화)           │
│  ③ 도메인 분포 거리 → 성능 저하 상관관계                      │
└─────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Glucose 파트의 독보적 강점**: 3개 파트 중 유일하게 **시계열(time series)** 데이터를 다루며, 12개 이상의 다기관·다국적 코호트를 이미 확보하고 전이학습 실험까지 완료한 상태입니다. 이는 generalizability 논문에서 **가장 풍부한 cross-domain 증거**를 제공할 수 있는 위치입니다.

---

## 2. 한울님이 이미 가진 것 vs. 추가로 해야 할 것

### 2.1 이미 완료된 작업 (재활용 가능)

| 완료 항목 | 관련 파일 | 논문에서의 역할 |
|:---|:---|:---|
| 12개 코호트 전처리 파이프라인 | `002_Harmonize-cgm-datasets/` | 데이터 설명 |
| 5-Way 비교 실험 (T1D → T2D/Mixed) | `016_Tier_7_Cross_Disease/` | **핵심 실험 1: Cross-domain generalizability** |
| Negative transfer 정량화 | Tier7 Results | 핵심 발견 |
| CORAL/TrAdaBoost 비교 | Tier6, Tier7 | 도메인 적응 기법 효과 |
| 학습 곡선 (데이터 양 vs 성능) | ShanghaiT2DM 학습 곡선 | Cold-start 시나리오 |
| 동일 질병 전이 대조 실험 | Tier7.1 (same-disease) | **핵심 발견: 기법 한계 vs 도메인 갭 분리** |
| Clarke Error Grid + 저혈당 sensitivity | Tier7.1 (clinical safety) | 임상 안전성 |

### 2.2 논문화를 위해 추가로 해야 할 것

| 우선순위 | 작업 | 이유 | 난이도 |
|:---|:---|:---|:---|
| 🥇 | **Within Variation 실험** | 교수님 요청: 같은 데이터+모델에서 seed/HP 변동 시 결과 안정성 | ⭐⭐ |
| 🥇 | **LODO 실험** | 교수님 제안 (4/15): Leave-One-Dataset-Out으로 generalizability 정량화 | ⭐⭐ |
| 🥈 | **분류 문제 전환** | 교수님 피드백 (4/15): regression → classification (저혈당/고혈당/정상 3분류) | ⭐⭐ |
| 🥈 | **도메인 거리 ↔ 성능 저하 상관 분석** | 논문 Approach 2와 직결: MMD, PAD 등으로 사전 예측 가능성 | ⭐⭐⭐ |
| 🥉 | **결과 안정성 측도 통일** | 3개 파트 공통: Correlation, ARI, Cohen's Kappa 등 | ⭐ |

---

## 3. 구체적 연구 설계 제안

### 실험 1: Within Variation — Computational Reproducibility
**질문**: "같은 CGM 데이터, 같은 LightGBM 모델이어도, 학습 조건을 바꾸면 임상적 결론이 흔들리는가?"

```
고정: 데이터셋 (ShanghaiT2DM + T1D 소스), 모델 (LightGBM), 피처 (22개)
변동 요인:
  ① Random seed: 10개 시드 × 5-Way 실험 → 50회 반복
  ② Hyperparameter: learning_rate, num_leaves, max_depth 그리드
  ③ Preprocessing: lookback 윈도우 크기 (3, 6, 12 steps)
  ④ 소스 풀 서브샘플링 비율 (1%, 5%, 10%, 50%, 100%)

평가:
  - RMSE의 변동계수(CV) = σ/μ
  - 5-Way 순위(ordering)의 Kendall's τ 일관성
  - 임상 결론 뒤집힘 빈도: "전이학습이 target_only보다 좋다"가 뒤집히는 비율
```

> [!TIP]
> 이 실험은 기존 코드를 for loop으로 감싸는 것만으로 구현 가능합니다. 가장 빠르게 결과를 낼 수 있는 실험입니다.

### 실험 2: Between Variation — Cross-Domain Generalizability (LODO)
**질문**: "한 코호트를 빼고 나머지로 학습하면, 빠진 코호트에서 성능이 얼마나 저하되는가?"

```
데이터: 12개 코호트 전체 (T1D 6개 + T2D 1개 + Mixed 2개 + ND 3개)

LODO 프로토콜:
  for each dataset D_i in {D_1, ..., D_12}:
    Train: 나머지 11개 합산
    Test:  D_i
    → RMSE_i, MARD_i 기록

비교:
  ① LODO RMSE vs. Self-trained RMSE (=기존 Tier 7의 target_only)
  ② 도메인 적응 (CORAL, TrAdaBoost) 적용 시 LODO 성능 회복 정도
  ③ 코호트 특성(질병 유형, 환자 수, 센서, 국가)별 RMSE 분포
```

> [!IMPORTANT]
> 이것이 **논문의 핵심 Figure**가 됩니다. 12개 코호트 × {LODO, Self, CORAL, TrAdaBoost} 히트맵을 그리면, 어떤 조건에서 generalizability가 깨지는지 한눈에 보입니다.

### 실험 3: 도메인 거리 → 성능 저하 예측 가능성
**질문**: "소스-타겟 간 분포 거리를 측정하면, 전이 성능 저하를 사전에 예측할 수 있는가?"

```
for each (source, target) pair in LODO:
  1. 도메인 거리 계산:
     - MMD (Maximum Mean Discrepancy)
     - Proxy-A-Distance (PAD)
     - Wasserstein Distance
     - 피처 공분산 Frobenius Norm

  2. 성능 저하량 계산:
     - ΔPerformance = LODO_RMSE - Self_RMSE

  3. 상관 분석:
     - Pearson/Spearman(도메인 거리, ΔPerformance)
     - 회귀: ΔPerformance ~ f(도메인 거리)
```

> [!NOTE]
> 이것은 토의 문서의 **Approach 2 (도메인 분포 차이 기반 성능 예측 척도)**와 정확히 일치합니다. Glucose 파트에서 이를 실증하면 논문의 방법론적 기여가 됩니다.

### 실험 4: Regression → Classification 전환 (분류 재현성)
**질문**: "CGM 예측을 저혈당/정상/고혈당 3분류로 전환하면, 재현성 패턴이 달라지는가?"

```
분류 기준 (임상 표준):
  - 저혈당: CGM < 70 mg/dL
  - 정상:   70 ≤ CGM ≤ 180 mg/dL
  - 고혈당: CGM > 180 mg/dL

평가 지표:
  - Accuracy, F1-score (macro), AUC-ROC (one-vs-rest)
  - Cohen's Kappa (3개 파트 공통 척도)
  - 클래스별 Sensitivity/Specificity

장점:
  ① 3개 파트(이미지/테이블/시계열) 모두 분류 문제로 통일 가능
  ② 교수님 피드백 "classification으로 전환 필요" 반영
  ③ 재현성 지표(Kappa, ARI)가 분류에서 더 명확하게 정의됨
```

---

## 4. 논문 구조에서 Glucose 파트의 위치

```
논문 제목(안):
"Evaluating Reproducibility and Generalizability of Clinical AI
 across Data Modalities: Image, Tabular, and Time Series"

Section 1. Introduction
  - AI 재현성 위기 + npj collection 맥락

Section 2. Methods
  2.1 공통 평가 프레임워크
    - Within/Between Variation 정의
    - 도메인 거리 ↔ 성능 저하 상관 분석 방법론
    - 공통 지표: Kappa, ARI, 변동계수

  2.2 Breast Cancer (이미지) — 현수님
  2.3 COVID-19 (테이블) — 남윤님
  2.4 Glucose/CGM (시계열) — 한울님 ◀
    - 12개 다기관 코호트 설명
    - LODO 프로토콜
    - 5-Way 전이학습 비교
    - Regression + Classification 이중 평가

Section 3. Results
  3.1 Within Variation (3개 모달리티 비교)
  3.2 Between Variation (LODO / Cross-domain)
  3.3 도메인 거리 예측 가능성
  3.4 임상 안전성 (Clarke Grid — glucose 전용)

Section 4. Discussion
  - "어떤 모달리티에서 재현성이 가장 취약한가?"
  - "도메인 거리로 성능 저하를 사전에 예측할 수 있는가?"
  - 시계열 특유의 도전 (concept drift, 시간 의존성)

Section 5. Conclusion
```

---

## 5. 타임라인 제안

| 주차 | 작업 | 산출물 |
|:---|:---|:---|
| **Week 1 (5/27~6/2)** | ① Within Variation 실험 (seed × HP 반복) | RMSE 분포 + 결론 뒤집힘 표 |
| | ② Classification 전환 코드 작성 | 3분류 파이프라인 |
| **Week 2 (6/3~6/9)** | ③ LODO 실험 (12코호트 순환) | 12×4 히트맵 |
| | ④ 도메인 거리 계산 (MMD, PAD) | 거리-성능 산점도 |
| **Week 3 (6/10~6/16)** | ⑤ 결과 통합 + 시각화 | 논문용 Figure 초안 |
| | ⑥ 3개 파트 공통 지표 정렬 | Methods 섹션 초안 |
| **Week 4~** | 논문 작성 + 피드백 반영 | 원고 |

---

## 6. 교수님 피드백 대응 매핑

| 교수님 피드백 (미팅 기록) | 대응 방안 |
|:---|:---|
| "regression: GT가 아닌데 의미가 있냐?" (4/15) | → 실험 4에서 classification 전환으로 해결. CGM 값 자체가 GT 역할(FDA 인증 기기) |
| "classification으로 전환 필요" (4/15) | → 저혈당/정상/고혈당 3분류로 전환 |
| "LODO 시나리오" (4/15) | → 실험 2에서 12개 코호트 LODO 구현 |
| "시나리오 더 정교하게" (4/29) | → LODO + Within Variation + 도메인 거리 분석으로 체계화 |
| "전이학습 알고리즘 확정할 것" (4/29) | → 기존 CORAL + TrAdaBoost 유지 (ML 수준), 논문에서는 "정적 전이학습의 한계"로 프레이밍 |
| "성능보다 변동성에 초점" (3/4) | → Within Variation 실험이 정확히 이것. CV, 결론 뒤집힘 빈도 중심 |
| "stationary하지 않음 → PatchTST" (5/13) | → 논문 Discussion에서 "시계열 특화 전이학습의 필요성"으로 Future Work 제시 |

---

## 7. 핵심 요약: "나는 뭘 해야 하는가?"

> [!IMPORTANT]
> ### 즉시 실행 (이번 주)
> 1. **Within Variation 실험**: 기존 5-Way 코드에 `for seed in range(10)` 루프를 추가하여 50회 반복 실행. RMSE 분포의 변동성과 임상 결론 뒤집힘 빈도를 측정.
> 2. **Classification 전환**: 기존 regression 타겟(CGM 값)을 {저혈당, 정상, 고혈당} 3분류로 변환하는 코드 작성.

> ### 1~2주 내
> 3. **LODO 실험**: 12개 코호트를 순환하며 Leave-One-Dataset-Out 일반화 성능 측정. → 논문의 핵심 Figure.
> 4. **도메인 거리 분석**: MMD/PAD로 소스-타겟 간 분포 거리를 사전 계산하고, 성능 저하량과의 상관관계 검증.

> ### 논문 작성 시
> 5. 기존 Tier 7/7.1 결과를 **"Cross-domain generalizability of CGM AI across 12 global cohorts"**로 재포장.
> 6. 3개 파트(이미지/테이블/시계열) 공통 프레임워크에 맞춰 결과를 정렬.

**한 줄 요약**: 기존에 이미 한 실험들은 generalizability 논문의 **Between Variation** 파트로 직결됩니다. 부족한 것은 **Within Variation**(반복 안정성)과 **LODO**(체계적 일반화 테스트), 그리고 **분류 전환**입니다.
