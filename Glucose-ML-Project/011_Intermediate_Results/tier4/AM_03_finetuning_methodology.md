# AM-03: 파인튜닝 방법론 비교 분석
# — "현재 방법 vs Nalmpatian et al. (2025) 원본 방법론"

---

## 1. 현재 사용한 방법: LightGBM Incremental Learning

### 1.1 방법 설명

현재 Tier 4 실험에서 사용한 파인튜닝은 **LightGBM의 `init_model` 파라미터**를 이용한 incremental learning이다.

```
[Source 학습]                          [Fine-tuning]
Dataset A의 train set     →           Dataset B의 train set
LightGBM 300 rounds       →           기존 모델 위에 100 rounds 추가
lr = 0.1                  →           lr = 0.05
                          →           init_model = source 모델 파일
                                      ↓
                          결과: tree 300개(source) + tree 100개(target) = 400개

```

### 1.2 작동 원리

GBM은 boosting 알고리즘이므로 각 tree는 **이전 tree들의 잔차(residual)를 학습**한다:

```
Tree 1~300 (Source 단계):
  - Dataset A의 패턴을 순차적으로 학습
  - 300번째 tree까지의 누적 예측 = ŷ_source

Tree 301~400 (Fine-tuning 단계):
  - Dataset B의 실제값과 ŷ_source의 차이(잔차)를 학습
  - 즉, "source 모델이 target에서 틀린 부분"만 보정
  - 낮은 lr(0.05)로 과적합 방지
```

### 1.3 이 방법의 특성

| 측면 | 설명 |
|:---|:---|
| **구현** | LightGBM `init_model` 파라미터 1개로 구현 (1줄) |
| **Source 지식 보존** | 기존 300개 tree는 변경 불가 — 보존됨 |
| **Target 적응** | 추가 100개 tree가 source-target 차이를 학습 |
| **Target 데이터 필요** | ✅ target의 train set이 필요 (zero-shot 아님) |
| **2단계 구조** | ❌ Global → Specialized 구조 없음 (단일 모델) |
| **유사도 활용** | ❌ source 선택에 유사도를 사용하지 않음 |
| **시계열 보존** | ✅ 원본 데이터를 그대로 사용, 리샘플링/노이즈 없음 |

### 1.4 결과 요약

- 132개 cross-dataset 쌍 중 **124개(94%) 개선, 0개(0%) 악화**
- 평균 ΔRMSE = **-6.65 mg/dL**
- Fine-tuning 후 cross RMSE(17.58) ≈ self RMSE(17.63) → **transfer gap 완전 해소**

---

## 2. Nalmpatian et al. (2025) 원본 방법론

### 2.1 Algorithm 1 — 5단계 구조

```
Step 1: Global Model (f_G)
  - K개 source 데이터셋을 풀링(pooling)
  - 공통 피처(global features)만 사용하여 GBM 학습

Step 2: Specialized Models (f_j, j=1...K)
  - 각 source 데이터셋 j에서 개별 GBM 학습
  - 공통 + 고유 피처(local features) 모두 사용
  - f_G의 출력을 초기값으로 사용하여 학습 (warm start)

Step 3: Similarity Index
  - 외부 데이터(HMD, OECD 등)로 target M과 각 source j의 유사도 계산
  - Manhattan distance → exponential normalization

Step 4: Synthetic Data Generation (합성 데이터 생성)
  - 유사도에 비례하여 source 데이터를 재표집(resample)
  - 유사도 역비례로 노이즈 주입 (비유사 source의 영향 희석)
  - target의 인구 수준 통계로 대체

Step 5: Transfer Prediction
  - D̂_M = Σ_j [ f_G(X_M^(j)_global) + f_j(X_M^(j)_local) ]
  - Global 예측 + 각 source의 Specialized 보정의 합
```

### 2.2 핵심 차별점 — 우리 방법과 비교

| 측면 | Nalmpatian (원본) | 우리 (현재) |
|:---|:---|:---|
| **모델 구조** | **2단계:** Global f_G + Specialized f_j | 단일 모델 |
| **유사도 활용** | **핵심:** 재표집 비율, 노이즈 크기 결정 | 미사용 |
| **Source 수** | **모든 K개** source의 가중 결합 | **1개** source만 사용 |
| **합성 데이터** | ✅ 유사도 비례 재표집 + 노이즈 | ❌ 없음 |
| **Local features** | ✅ 데이터셋 고유 피처 활용 | ❌ 공통 20-dim만 |
| **Target 데이터 필요** | ❌ 없어도 가능 (zero-shot) | ✅ 필요 |
| **Drift 분석** | ✅ GLM으로 오차 원인 분해 | ❌ 없음 |

---

## 3. Nalmpatian 방법론을 적용할 수 있는가?

### 3.1 적용 가능한 부분 ✅

| 단계 | 적용 가능 여부 | 구현 방안 |
|:---|:---:|:---|
| **Step 1: Global Model** | ✅ 즉시 가능 | 11개 데이터셋 풀링 → 공통 20-dim으로 LightGBM 학습 |
| **Step 2: Specialized Models** | ✅ 가능 | 각 데이터셋에서 (공통 + 고유) 피처로 LightGBM 학습, f_G 출력을 `init_model`로 사용 |
| **Step 3: Similarity Index** | ✅ **이미 완료** | `similarity_matrix.csv` 생성됨 — 혈당 통계 기반 유사도 |
| **Step 5: Transfer Prediction** | ✅ 가능 | Σ_j [ f_G(global) + f_j(local) ] 계산 |
| **Step 6: Drift Analysis** | ✅ 가능 | 전이 오차를 cohort/sensor/feature 수준에서 GLM 분석 |

### 3.2 적용이 어려운 부분 ⚠️

#### Step 4: 합성 데이터 생성 — **CGM 시계열에 직접 적용 불가**

Nalmpatian의 원본 방법은 **보험 사망률 데이터**(각 행이 독립적인 집계 레코드)를 대상으로 했다. 행을 독립적으로 재표집해도 의미가 보존된다. 그러나 **CGM 시계열 윈도우**는:

| 문제 | 설명 |
|:---|:---|
| **시간적 자기상관 파괴** | 윈도우의 lag 피처(t-0~t-5)는 연속 시간의 혈당. 독립 재표집하면 "t-5=120, t-4=250, t-3=80" 같은 비현실적 시퀀스 생성 |
| **파생 피처 무효화** | Velocity = g(t-0) - g(t-1)로 정의. 재표집된 행에서 이 값은 원본 시계열의 변화율이 아님 |
| **Data leakage** | 시간 순서 파괴 시 미래 정보가 과거로 혼입 가능 |
| **노이즈 주입의 비현실성** | 혈당에 가우시안 노이즈 추가 시 "5분 만에 200 mg/dL 변동" 같은 생리학적 불가능 궤적 생성 |
| **Prioleau et al. (2025) 위반** | 벤치마크 가이드라인이 interpolation/extrapolation조차 금지 → 합성 데이터는 더 심한 위반 |

### 3.3 Step 4 대안: Patient-Level Resampling

원본 Step 4의 **정신(spirit)**은 보존하되, **단위를 행(row) → 환자(patient)로 변경**:

```
원본 Step 4 (행 단위, 시계열 불가):
  - source j에서 n_j = total × s_j 개의 "행"을 랜덤 추출
  - 각 행에 노이즈 주입
  
대안 Step 4 (환자 단위, 시계열 보존):
  - source j에서 p_j = round(P_total × s_j) 명의 "환자"를 랜덤 선택
  - 선택된 환자의 전체 시계열을 원본 그대로 사용 (노이즈 없음)
  - 가중치: selected patient의 loss weight = s_j (유사할수록 높은 가중치)
```

**이렇게 하면:**
- 시계열 구조 100% 보존 ✅
- 유사도 기반 재표집 유지 ✅
- Prioleau et al. 가이드라인 준수 ✅
- 노이즈 없이도 s_j의 loss weight로 "비유사 source의 영향 희석" 효과 달성 ✅

---

## 4. 제안: Nalmpatian 방법론 적용 실험 설계

### 4.1 실험 구조

```
=== Nalmpatian-CGM Transfer Framework ===

For M ∈ {Dataset_1, ..., Dataset_12}:   (LODO: M을 빼고 나머지 11개로)
  
  Step 1: Global Model
    - 11개 source를 풀링 (공통 20-dim 피처)
    - LightGBM(300 rounds) 학습 → f_G
  
  Step 2: Specialized Models (×11)
    - 각 source j에서 f_G를 init_model로 사용
    - 공통 + 고유 피처(20~398 dim)로 LightGBM(100 rounds) 학습 → f_j
    ※ 고유 피처가 없는 데이터셋(AIDET1D, CGMND 등)은 f_j = f_G와 동일
  
  Step 3: Similarity Index
    - similarity_matrix.csv에서 s_j 로드
  
  Step 4: Patient-Level Resampling (대안)
    - source j에서 p_j명의 환자를 선택 (p_j ∝ s_j)
    - 선택된 환자의 원본 시계열을 그대로 사용
    - 합성 데이터셋 X_M 구성 (각 행에 출처 j 기록)
  
  Step 5: Transfer Prediction
    방법 A (원본 충실): D̂_M = Σ_j [ f_G(X_M^(j)_global) + f_j(X_M^(j)_local) ]
    방법 B (단순화):    D̂_M = Σ_j [ s_j × f_j(X_M_global) ]  (가중 앙상블)
  
  Evaluate on M's test set
```

### 4.2 우리 현재 방법과의 비교 실험

| 방법 | 코드명 | Source 수 | 유사도 | 구조 |
|:---|:---|:---:|:---:|:---|
| **Baseline** | self | 0 | - | M 자체만으로 학습 |
| **As-is** | asis | 1 | ❌ | source 모델을 M에 그대로 |
| **현재 FT** | ft_single | 1 | ❌ | source 모델 + M train으로 추가 학습 |
| **Nalmpatian 원본** | nalm_full | 11 | ✅ | Global → Specialized → 합성 → 가중합 |
| **Nalmpatian 단순** | nalm_simple | 11 | ✅ | Specialized → 유사도 가중 앙상블 |

### 4.3 검증할 연구 질문

| # | 질문 | 비교 |
|:---:|:---|:---|
| **RQ1** | 11개 source를 모두 활용하는 것이 1개 source보다 나은가? | nalm_full vs ft_single |
| **RQ2** | 유사도 가중이 균등 가중보다 나은가? | nalm_full vs 균등 가중 앙상블 |
| **RQ3** | 2단계 구조(Global→Specialized)가 단일 모델보다 나은가? | nalm_full vs nalm_simple |
| **RQ4** | Local (고유) 피처가 Global 피처만 쓰는 것보다 나은가? | nalm_full vs Global only |
| **RQ5** | Nalmpatian 프레임워크가 simple fine-tuning보다 나은가? | nalm_full vs ft_single |

### 4.4 예상 소요 시간

| 단계 | 소요 시간 | 비고 |
|:---|:---:|:---|
| Global Model (11개 풀링) | ~15분 × 12 LODO = ~3시간 | 가장 큰 비용 |
| Specialized Models (11개 × 12) | ~5분 × 132 = ~11시간 | 병렬화 가능 |
| Patient-Level Resampling | ~10분 × 12 = ~2시간 | I/O 바운드 |
| Transfer Prediction + 평가 | ~5분 × 12 = ~1시간 | 빠름 |
| **Total** | **~17시간** | Sequential 기준 |

---

## 5. 결론: 적용 가능하며, 적용해야 한다

| 관점 | 판단 |
|:---|:---|
| **학문적 가치** | Nalmpatian의 2단계 구조를 CGM 시계열에 성공적으로 적용하면, **도메인 간 전이학습 프레임워크의 일반화 가능성**을 입증하는 기여가 됨 |
| **기술적 가능성** | Step 1~3, 5~6은 즉시 구현 가능. Step 4만 patient-level로 수정 필요 |
| **현재 실험과의 관계** | 현재 `ft_single`은 **Nalmpatian의 일부(Step 2만)**를 축소 적용한 셈. 원본 방법론은 이를 포함하는 상위 구조 |
| **논문 기여 강화** | "단순 fine-tuning으로도 transfer gap이 해소되지만, Nalmpatian 프레임워크를 적용하면 source 선택의 유사도 정당화 + zero-shot 전이도 가능"이라는 계층적 결과를 보고할 수 있음 |

### 우선순위 제안

```
이미 완료: self, asis, ft_single
     ↓
다음 단계: nalm_simple (유사도 가중 앙상블) — 구현 ~2시간, 실행 ~3시간
     ↓
추가 단계: nalm_full (Global → Specialized + Patient-Level Resampling) — 구현 ~4시간, 실행 ~17시간
```

> nalm_simple부터 시작하면, 빠르게 "유사도 가중의 효과"를 확인하고, nalm_full로 확장 여부를 결정할 수 있다.

---

*Glucose-ML-Project · Agenda Material 03 · 2026-04-16*
