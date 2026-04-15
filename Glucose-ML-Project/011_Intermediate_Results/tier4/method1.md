# Tier 4 — Method 1: GBM-based Transfer Learning Framework
## Adapted from Nalmpatian et al. (2025), PLOS ONE

> **Source:** Nalmpatian A, Heumann C, Alkaya L, Jackson W. (2025). *Transfer learning for mortality risk: A case study on the United Kingdom.* PLoS ONE 20(5): e0313378.  
> DOI: [10.1371/journal.pone.0313378](https://doi.org/10.1371/journal.pone.0313378)

---

## 1. 논문 방법론 요약

### 1.1 문제 설정

Nalmpatian et al.은 **자국 내 보험포트폴리오 사망률 데이터가 전혀 없는 영국(UK)**
을 대상으로, 8개국의 사망률 포트폴리오 데이터로 사전학습된 GBM 모델을 전이(transfer)하여 사망률을 예측하는 프레임워크를 제시한다.

<img width="900" height="396" alt="image" src="https://github.com/user-attachments/assets/98b6e506-a489-45ba-bfda-8a686ffeaaee" />


핵심 아이디어:
1. **Pretrained Global Model:** K개 source countries의 pooled data로 GBM을 사전학습 (global features만 사용)
2. **Specialized Local Models:** 각 country j의 local features를 추가하여 GBM을 fine-tune (pretrained model의 출력을 초기값으로 사용)
3. **Country Similarity Index:** 외부 공공데이터(OECD, HMD 등)로 target country M과 K개 source countries 사이의 유사도를 Manhattan distance → exponential transformation으로 계량화
4. **Synthetic Data Generation:** 유사도 점수에 비례하여 K개 source countries로부터 resampling + 유사도 역비례 노이즈 주입 → target country M에 대한 합성 포트폴리오 데이터 생성
5. **Transfer Prediction:** Global model + K개 specialized models의 가중 합산으로 target M의 사망률 예측
6. **Drift Model:** GLM(Poisson)으로 전이 예측과 실제(CMI) 사이의 잔차 패턴을 분석 → 어떤 인구통계학적 요인이 전이 오차를 유발하는지 규명

### 1.2 핵심 알고리즘 (Algorithm 1)

```
Step 1: Train global GBM f_G on pooled {X_1, ..., X_K} using global features only
Step 2: For each j ∈ {1,...,K}: train specialized GBM f_j on (global ∪ local_j),
        initialized from f_G predictions (not random)
Step 3: Compute similarity scores s_j between M and each source j
        using external indicators (Manhattan distance → exp normalization)
Step 4: Generate synthetic dataset X_M:
        - Resample rows from each j proportional to s_j
        - Replace population mortality with M's HMD data
        - Add Gaussian noise (σ ∝ 1/s_j) to metric features
        - Add categorical flip noise (threshold ∝ 1/s_j)
Step 5: Predict for M:
        D̂_M = Σ_j [ f_G(X_M^(j)_global) + f_j(X_M^(j)_local) ]
```

### 1.3 핵심 설계 원리

| 원리 | 논문의 적용 | 본질적 역할 |
|:---|:---|:---|
| 2단계 모델 구조 | Global → Specialized | 공통 패턴 먼저 포착, 이후 국소 특성 보정 |
| 유사도 기반 재표집 | Country Similarity Index | 유사한 source에 더 큰 가중치 부여 |
| 노이즈 주입 | σ ∝ 1/similarity | 비유사 source의 영향을 의도적으로 희석 |
| Drift 분석 | GLM(Poisson) on residuals | 전이 오차의 원인을 factor 수준으로 분해 |

---

## 2. 우리 프로젝트와의 구조적 대응

### 2.1 도메인 매핑

| 논문 개념 | 우리 프로젝트 대응 |
|:---|:---|
| **Country j** (source country) | **Dataset j** (12개 CGM 데이터셋 중 K개) |
| **Country M** (target, no data) | **Dataset M** (hold-out 대상 데이터셋) |
| **Insurance portfolio mortality** | **30-min ahead glucose prediction** (RMSE) |
| **Global features** (age, gender 등) | **공통 harmonized features** (Tier 2.5 시계열 피처 20 dim + Q/M 정적 피처 중 커버리지 ≥50%인 변수) |
| **Local features** (occupation class 등) | **데이터셋 고유 동적 피처** (I: bolus/basal, A: HR/Steps, S: nutrition 등) **+ Q/M 정적 피처 중 해당 데이터셋에만 존재하는 변수** |
| **HMD population mortality** | **공통 혈당 통계** (Mean glucose, SD, TIR 등 — 모든 데이터셋에서 계산 가능한 환자군 수준 통계) |
| **Country Similarity Index** | **Dataset Similarity Index** (혈당 분포, TIR/TAR/TBR, 환자군 유형, 센서 유형, 샘플링 간격 등으로 구성) |
| **Drift Model (GLM-Poisson)** | **Error Attribution Model** (전이 예측 오차를 cohort/sensor/feature 수준에서 분석) |

### 2.2 구조적 적합성 평가

> [!IMPORTANT]
> 이 방법론이 우리 프로젝트에 **특히 적합한 이유**: 12개 데이터셋이 환자군(T1DM/GDM/ND), 센서(Dexcom/Libre), 국가(6개국), 피처 차원(20~398)에서 극도로 이질적이지만, **Tier 2.5의 harmonized feature 파이프라인 덕분에 공통 피처(global features)가 이미 정의**되어 있다. 논문의 "global vs local" 2단계 구조에 자연스럽게 매핑된다.

### 2.3 데이터셋 간 피처 이질성 현황

아래 표는 `dataset_datatype_figure.png`에서 확인한 피처 가용성에 기초한다:

| 데이터셋 | 공통 시계열 | 고유 동적 (I/A/S) | Q/M 정적 (원본 존재) | 고유 피처 예시 |
|:---|:---:|:---:|:---:|:---|
| AIDET1D | 20 | 0 | age, sex, DM duration, insulin regimen | (CGM only + Q) |
| BIGIDEAs | 20 | 72 | age, sex, DM duration, HbA1c, BMI, lifestyle | Food calories, carbs, protein, fat |
| Bris-T1D | 20 | 216 | age, sex | HR, steps, distance, basal, bolus, carb_bolus |
| CGMacros_Dex | 20 | 378 | age, sex, DM duration, HbA1c, BMI, lifestyle | 2nd sensor, nutrition(6), HR, steps, calories |
| CGMacros_Lib | 20 | 378 | age, sex, DM duration, HbA1c, BMI, lifestyle | (Dexcom과 동일 구조) |
| CGMND | 20 | 0 | age, sex | Sleep log (미사용) |
| GLAM | 20 | 18 | age, sex, HbA1c, BMI, OGTT, cholesterol, lifestyle | Meal event marker |
| HUPA-UCM | 20 | 180 | age, sex, DM duration | HR, steps, calories, basal, bolus, carb_bolus |
| IOBP2 | 20 | 54 | age, sex, DM duration, HbA1c, BMI | Bolus, carb_bolus |
| Park_2025 | 20 | 18 | age, sex | Carb intake |
| PEDAP | 20 | 36 | age, sex, DM duration, HbA1c, BMI | Basal, bolus, carb_bolus |
| UCHTT1DM | 20 | 18 | age, sex, DM duration, HbA1c, BMI | Bolus |

> **Global (공통) 피처 풀:** glucose lags (t-0 ~ t-5), Velocity, Acceleration, Jerk, SD1, Window_Mean/Std/AUC, TIR/TAR/TBR, LBGI/HBGI, tod_sin, tod_cos = ~20 dim (시계열) + demographics(age, sex) = ~22 dim

### 2.4 Q/M 정적 피처(Static Features)의 Local Feature 확장

> [!NOTE]
> `dataset_data_type_availability.md`에서 확인된 바와 같이, Q(설문/인구통계)와 M(임상검사) 카테고리는 **대부분 원본 데이터셋에 존재하지만 현재 harmonized 파이프라인에 미포함(🔶)**이다. 이 변수들은 **환자별 상수(static per-patient)** 이므로 해당 환자의 모든 시계열 윈도우에 broadcast하여 피처로 추가할 수 있다.

이는 논문의 구조와 정확히 대응한다:
- 논문: 보험계약자의 **직업군(occupation class)** = 계약별 상수를 모든 exposure record에 첨부
- 우리: 환자의 **HbA1c, BMI, 나이, 성별** = 환자별 상수를 모든 30분 윈도우에 첨부

#### Q/M 변수별 커버리지와 활용 전략

| 변수 | 커버리지 | 전략 | 근거 |
|:---|:---:|:---|:---|
| **Demographics (age, sex)** | 12/12 (100%) | **→ Global feature 승격** | 전체 데이터셋에 존재. 논문의 age/gender가 global feature인 것과 동일 |
| **HbA1c** | 7/12 (58%) | **→ Quasi-Global** (결측 데이터셋은 cohort 중앙값으로 대체) | 커버리지 >50%. 혈당 변동성의 장기 지표로서 예측력 높음 |
| **BMI / Weight** | 7/12 (58%) | **→ Quasi-Global** (동일 전략) | 인슐린 저항성 proxy, 커버리지 >50% |
| **DM duration** | 8/12 (67%) | **→ Quasi-Global** (ND 데이터셋은 0으로 설정 가능) | T1DM 데이터셋 대부분 보유. ND는 duration=0으로 자연스럽게 인코딩 |
| **OGTT / FPG** | 2/12 (17%) | **→ Pure Local** (CGMND, GLAM 전용) | 커버리지 극소. 해당 데이터셋의 specialized model에만 투입 |
| **Cholesterol / TG** | 1/12 (8%) | **→ Pure Local** (GLAM 전용) | GLAM만 보유. Specialized-GLAM에서만 활용 |
| **Lifestyle survey** | 4/12 (33%) | **→ Pure Local** (BIGIDEAs, CGMacros×2, GLAM) | 비정형 + 커버리지 부족 |

#### 3-Tier Feature 구조 (Hybrid 전략, 추천)

```
Tier 1 — Global Features (모든 데이터셋 공통)
├── 시계열 파생 피처 (20 dim): glucose lags, Velocity, Acceleration, ...
├── Demographics (2~3 dim): age, sex (+ race if available)
└── Quasi-Global Q/M (3 dim, 결측 시 imputation): HbA1c, BMI, DM_duration
    Total: ~25 dim

Tier 2 — Specialized Local Features (데이터셋별 고유)
├── 동적 피처 (I/A/S): insulin, activity, nutrition (0~378 dim)
└── 희소 Q/M: OGTT, cholesterol, lifestyle survey (0~3 dim)
    Total: 데이터셋별 0~381 dim

Tier 3 — Synthetic Bridge Features (합성 데이터 생성 시)
└── Target M의 Q/M 통계를 source j에서 bootstrap 될 때 대치
    (논문의 "HMD population mortality를 target 것으로 교체" 와 동일)
```

#### Imputation 전략 (결측 Q/M 처리)

| 결측 유형 | 전략 | 예시 |
|:---|:---|:---|
| **Cohort 내 결측** | 동일 cohort의 다른 데이터셋 중앙값 | Bris-T1D에 HbA1c 없음 → T1DM 평균 HbA1c (다른 T1DM 데이터셋에서) |
| **Cohort 간 결측** | 전체 데이터셋 중앙값 + cohort indicator 피처 추가 | Park_2025(ND)에 HbA1c 없음 → ND 평균 HbA1c 추정 |
| **논리적 결측** | 도메인 지식 기반 고정값 | ND 데이터셋의 DM_duration → 0 (유병 기간 없음) |

> [!IMPORTANT]
> **Q/M 정적 피처 확장의 핵심 가치:** 현재 Tier 3의 공통 20-dim 피처는 **순수하게 시계열 패턴에서만 추출**된 것이다. Q/M을 추가하면 **환자 간(inter-patient) 이질성**을 모델이 직접 포착할 수 있게 된다. 이는 특히 전이학습에서 결정적이다 — "HbA1c가 8%인 T1DM 환자"와 "HbA1c가 5.5%인 ND 환자"의 혈당 dynamics가 근본적으로 다르다는 정보를 global model이 활용할 수 있기 때문이다.

---

## 3. 제안하는 연구 설계

### 3.1 연구 질문 (Research Questions)

논문의 3가지 RQ를 CGM 예측 도메인에 재정의한다:

| # | 연구 질문 | 논문 RQ와의 대응 |
|:---|:---|:---|
| **RQ1** | *특정 환자군/데이터셋의 내부 학습 데이터 없이, 다른 데이터셋에서 학습된 모델을 전이하여 혈당을 얼마나 정확히 예측할 수 있는가?* | RQ(i): How can we estimate mortality in a country with no internal data? |
| **RQ2** | *전이 예측의 오차(Drift)에 가장 큰 영향을 미치는 요인은 무엇인가? (환자군? 센서? 피처 이질성?)* | RQ(ii): How accurate is the model & what drives drift? |
| **RQ3** | *고유 피처(insulin, nutrition, activity)가 공통 피처를 넘어서 전이 성능을 개선하는가?* | RQ(iii): How can additional variables improve predictions? |

### 3.2 실험 프로토콜

#### Phase A: Dataset Similarity Index 구축

논문의 Country Similarity Index를 데이터셋 수준으로 변환한다.

**유사도 지표 후보 (Q items):**

| # | 지표 | 출처 | 논문 대응 |
|:---|:---|:---|:---|
| 1 | Mean glucose (mg/dL) | dataset_summary_stats.csv | Overall population mortality |
| 2 | Glucose SD (mg/dL) | 동일 | - |
| 3 | TIR (%) | 동일 | - |
| 4 | TAR (%) | 동일 | Life insurance indicators |
| 5 | TBR (%) | 동일 | - |
| 6 | 환자군 인코딩 (T1DM→1, GDM→0.5, ND→0) | cohort metadata | Health care statistics |
| 7 | 센서 인코딩 (Dexcom→1, Libre→0) | sensor metadata | - |
| 8 | 샘플링 간격 (5min→1, 15min→0) | parsing config | - |
| 9 | log(N subjects) | dataset_summary_stats.csv | Exposure volume |
| 10 | log(Total readings) | 동일 | - |
| 11 | 공통 피처 차원 대비 고유 피처 비율 | feature dictionary | Insurance penetration analogy |
| 12 | 수집 기간 중앙값 (days) | 동일 | - |
| 13 | 국가/지역 인코딩 | metadata | Geographic region |

**계산:**
```
d_jM = Σ_q |z_jq - z_Mq|  (Manhattan distance, z = standardized)
s_jM = exp(-d_jM) / Σ_k exp(-d_kM)  (normalized similarity score)
```

#### Phase B: 실험 시나리오 — Leave-One-Dataset-Out (LODO)

논문이 K=8 source → M=UK(1 target)로 실험한 것과 동일한 구조를 LODO로 반복한다.

```
For M in {Dataset_1, Dataset_2, ..., Dataset_12}:
    K = {all datasets} \ {M}   (K=11 source datasets)

    Step 1: Pool all K datasets' harmonized data
            → Train global GBM f_G on 공통 20-dim features
    
    Step 2: For each j ∈ K:
            → Fine-tune specialized GBM f_j on (공통 ∪ 고유_j),
              initialized from f_G output
    
    Step 3: Compute similarity scores s_j for M vs each j
    
    Step 4: Generate synthetic dataset for M:
            → Resample from K proportional to s_j
            → Replace "population-level stats" with M's statistics
            → Add noise inversely proportional to s_j
    
    Step 5: Predict glucose for M:
            D̂_M = Σ_j [ f_G(X_M^(j)_global) + f_j(X_M^(j)_local) ]
    
    Step 6: Evaluate RMSE, MAE, MAPE vs Tier 3 baseline (in-dataset trained model)
    
    Step 7: Error Attribution (Drift) analysis via GLM
```

> [!WARNING]
> **LODO 12회 반복은 계산 비용이 크다.** Global model만 해도 ~50M 윈도우를 학습해야 하므로, LightGBM의 대규모 최적화가 필수적이다. 예상 시간: 데이터셋당 ~30~60분 × 12 = 6~12시간. Tier 3 HPO와 유사한 배치 실행이 필요하다.

#### Phase C: Drift(Error Attribution) 분석

논문의 Drift Model을 다음과 같이 적용한다:

```
Target:   ratio = RMSE_transferred / RMSE_in_dataset
Features: cohort_type, sensor_type, sampling_interval,
          N_subjects, feature_overlap_ratio, mean_glucose_gap,
          TIR_gap, geographic_distance
Model:    GLM(Gaussian/Gamma) or Random Forest
Output:   Which factors explain the degradation in transfer?
```

### 3.3 기대 실험 매트릭스

|  | AIDET1D | BIGIDEAs | Bris-T1D | CGM_Dex | CGM_Lib | CGMND | GLAM | HUPA | IOBP2 | Park | PEDAP | UCH |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tier 3 Baseline** | 22.36 | 14.03 | 27.34 | 3.08 | 1.34 | 13.92 | 13.50 | 14.42 | 24.78 | 20.18 | 28.33 | 17.78 |
| **Transfer (예상)** | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| **Δ (Degradation)** | - | - | - | - | - | - | - | - | - | - | - | - |

**성공 기준:**
- Transfer RMSE가 Baseline 대비 **≤ 20% 증가**이면 "유효한 전이"로 판정
- 가장 유사한 코호트 간(예: T1DM→T1DM) 전이가 이기종(T1DM→ND) 전이보다 유의미하게 우수한 것이 확인되면, Similarity Index 유효성 입증

---

## 4. 적용 가능성 판정: 7대 핵심 차원

### 4.1 ✅ 적용 가능한 요소

| 차원 | 적합도 | 근거 |
|:---|:---:|:---|
| **2단계 GBM 구조** | ★★★★★ | Tier 3에서 이미 LightGBM/CatBoost baseline이 존재. `init_model` 파라미터로 pretrained → specialized 전이 구현 가능 |
| **Similarity Index** | ★★★★☆ | 12개 데이터셋의 메타통계가 `dataset_summary_stats.csv`에 이미 수집됨. 추가 수집 불필요 |
| **Synthetic Data** | ★★★★☆ | 유사도 비례 재표집 + 노이즈 주입은 복잡도가 낮으며 구현 용이 |
| **Drift Model** | ★★★★★ | GLM/GBM residual 분석은 우리 파이프라인에 자연스럽게 통합 가능 |
| **LODO 실험 설계** | ★★★★☆ | 논문의 Leave-One-Country-Out과 동일 구조. 12개 데이터셋에 자연스럽게 매핑 |

### 4.2 ⚠️ 적응이 필요한 요소

| 차원 | 주의점 | 해결 방안 |
|:---|:---|:---|
| **예측 타겟의 차이** | 논문 = 사망 건수(count, Poisson), 우리 = 연속 혈당값(continuous, Gaussian) | Loss function을 MSE/MAE로 변경. GBM의 `objective='regression'` 사용 |
| **피처 차원 극심 불균형** | AIDET1D(20 dim) vs CGMacros(398 dim) → specialized model의 입력 차원이 데이터셋마다 극도로 다름 | 논문과 동일: global model은 공통 20 dim만 사용, specialized는 각자의 full dim 사용. 합성 데이터 생성 시 target M의 고유 피처 차원에 맞춰 resampling |
| **데이터 규모 불균형** | GLAM(26M) vs UCHTT1DM(27K) → 1000배 차이 | Resampling 시 exposure 균등화 (논문의 total exposure = 100M 고정과 유사하게 total windows 고정) |

### 4.3 ❌ 적용 불가 / 불필요한 요소

| 차원 | 사유 |
|:---|:---|
| **Lee-Carter 사망률 모델 (S2)** | 시계열 사망률 전용. CGM에 대응하는 메커니즘이 불필요 |
| **포아송 분포 가정** | 혈당값은 연속이므로 Gaussian/Student-t 가정이 적절 |
| **CMI mortality tables** | 외부 검증 벤치마크로서의 역할만 → 우리의 경우 Tier 3 in-dataset baseline이 그 역할 |

---

## 5. 이 방법론으로 도출되는 연구의 형태

### 5.1 논문 제목 후보

> **"Cross-Cohort Transfer Learning for Glucose Prediction: A GBM-Based Similarity-Weighted Framework Across 12 Heterogeneous CGM Datasets"**

### 5.2 핵심 기여 (Contribution)

1. **최초의 대규모 Cross-Dataset CGM 전이학습 체계 연구:** 기존 CGM 논문은 대부분 한 데이터셋 내 cross-patient 전이에 집중. **12개 이질적 데이터셋 간 cross-dataset 전이**를 체계적으로 실험한 연구는 거의 없음.

2. **Dataset Similarity Index:** CGM 데이터셋 간 유사도를 정량화하는 최초의 표준화된 지표 제시. 어떤 데이터셋이 어떤 다른 데이터셋에 가장 유용한 source knowledge를 제공하는지 지도(map) 제공.

3. **Error Attribution (Drift) Analysis:** 전이 오차의 원인을 cohort/sensor/feature/geography 차원으로 분해. 임상적으로 "어떤 환자군에게 전이학습이 안전한가?"라는 질문에 근거 기반 답변 제시.

4. **실용적 가이드라인:** 새로운 CGM 데이터셋이 등장했을 때, 기존 모델을 어떻게 최소 비용으로 적응시킬 수 있는지에 대한 구체적 프로토콜 (Algorithm + Similarity Index + Noise schedule).

### 5.3 예상 핵심 발견사항

| 가설 | 검증 수단 |
|:---|:---|
| **H1:** 동일 코호트(T1DM→T1DM) 전이가 이종 코호트(T1DM→ND) 전이보다 RMSE degradation이 유의미하게 작다 | LODO 결과의 paired t-test / Wilcoxon |
| **H2:** Similarity Index가 높은 source에서 resampled된 합성 데이터일수록 전이 성능이 좋다 | Ablation — uniform resampling vs similarity-weighted |
| **H3:** Specialized model(고유 피처 포함)이 Global model(공통 피처 only)보다 동일 코호트 내 전이 시 유의미하게 우수하다 | f_G only vs f_G + f_j |
| **H4:** Drift의 주요 원인은 cohort type > sensor type > geographic origin 순이다 | Drift GLM의 coefficient 크기 비교 |
| **H5:** Q/M 정적 피처(HbA1c, BMI, DM duration)를 global model에 추가하면, 순수 시계열 피처 대비 cross-cohort 전이 성능이 유의미하게 개선된다 | Ablation — 20 dim (시계열 only) vs ~25 dim (시계열 + Q/M quasi-global) |

---

## 6. 구현 계획

### 6.1 파일 구조

```
013_Tier_4_Transfer_Learning/
├── method1_design.md                     # 본 문서 최종판
├── 01_dataset_similarity_index.py        # Similarity Index 계산
├── 02_global_model_pretrain.py           # Global GBM 사전학습
├── 03_specialized_model_finetune.py      # K개 Specialized GBM 학습
├── 04_synthetic_data_generator.py        # 합성 데이터 생성
├── 05_transfer_predict.py                # 전이 예측 실행 (LODO)
├── 06_drift_analysis.py                  # Error Attribution GLM
├── 07_results_visualization.py           # 최종 그림/표 생성
└── results/
    ├── similarity_matrix.csv
    ├── lodo_results.csv
    └── drift_coefficients.csv
```

### 6.2 핵심 구현 세부사항

#### Pretrained → Specialized 전이 메커니즘 (LightGBM)

```python
import lightgbm as lgb

# Step 1: Global Model
global_model = lgb.train(
    params={'objective': 'regression', 'metric': 'rmse', ...},
    train_set=pooled_global_data,   # 공통 20-dim features only
    num_boost_round=500
)

# Step 2: Specialized Model for dataset j
# 핵심: init_model로 pretrained model을 전달
specialized_j = lgb.train(
    params={'objective': 'regression', 'metric': 'rmse', ...},
    train_set=dataset_j_full_data,  # 공통 + 고유 features
    init_model=global_model,        # ← 이것이 논문의 핵심 전이 메커니즘
    num_boost_round=200             # 추가 boosting rounds
)
```

> [!NOTE]  
> LightGBM의 `init_model`은 이전 모델의 잔차 위에 추가 트리를 쌓는 것이므로, 논문의 "pretrained output → specialized refinement" 구조와 정확히 일치한다. CatBoost의 `init_model` 파라미터도 동일한 기능을 제공한다.

#### 합성 데이터 생성

```python
def generate_synthetic(target_M, source_datasets, similarity_scores, total_windows=1_000_000):
    """
    논문 Algorithm 1, Step 4 구현.
    
    - target_M: 타겟 데이터셋 메타정보
    - source_datasets: K개 source 데이터셋
    - similarity_scores: s_j (K-dim vector, sum=1)
    - total_windows: 합성 데이터의 총 윈도우 수
    """
    synthetic_rows = []
    
    for j, (ds, s_j) in enumerate(zip(source_datasets, similarity_scores)):
        n_j = int(total_windows * s_j)  # 유사도에 비례한 표본 수
        sampled = ds.sample(n=n_j, replace=True)
        
        # Metric features: add Gaussian noise (σ ∝ 1/s_j)
        sigma = 1.0 / (s_j + 1e-8)
        for col in metric_columns:
            sampled[col] += np.random.normal(0, sigma * sampled[col].std() * 0.05, size=n_j)
        
        # Categorical features: random flip with probability ∝ 1/s_j
        flip_prob = min(0.5, 0.1 / (s_j + 1e-8))
        for col in categorical_columns:
            mask = np.random.random(n_j) < flip_prob
            sampled.loc[mask, col] = np.random.choice(sampled[col].unique(), mask.sum())
        
        # Replace "population-level" stats with M's stats
        for stat_col in ['Window_Mean', 'TIR', 'TAR', 'TBR']:
            sampled[stat_col] = target_M_stats[stat_col]
        
        sampled['_source_dataset'] = j  # 출처 기록 (specialized model 적용 시 필요)
        synthetic_rows.append(sampled)
    
    return pd.concat(synthetic_rows, ignore_index=True)
```

### 6.3 계산 비용 추정

| 단계 | 예상 시간 | 메모리 |
|:---|:---:|:---:|
| Similarity Index 계산 | < 1분 | 무시 |
| Global Model 학습 (~50M windows, 20 dim) | ~15분 | ~8GB |
| Specialized Models × 11 | ~30분 (합계) | ~4GB peak |
| Synthetic Data 생성 × 12 targets | ~5분 (합계) | ~6GB peak |
| Transfer Prediction × 12 LODO | ~10분 (합계) | ~4GB |
| Drift Analysis | < 5분 | 무시 |
| **Total (1 LODO round)** | **~65분** | **~8GB peak** |
| **Total (12 LODO rounds)** | **~12시간** | **~8GB peak** |

---

## 7. 한계 및 주의사항

### 7.1 방법론적 한계

1. **합성 데이터의 본질적 제약:** 노이즈 주입된 합성 데이터는 실제 target 데이터의 분포를 완벽히 모사할 수 없다. 특히 CGM 데이터의 **시간적 자기상관 구조**는 독립 resampling으로 보존되지 않는다. 논문의 사망률 데이터는 집계(aggregated) 데이터여서 이 문제가 경미했지만, 우리의 시계열 윈도우 데이터에서는 더 심각할 수 있다.

2. **고유 피처 매핑 문제:** CGMacros(영양 데이터 보유) → AIDET1D(영양 데이터 없음)로 전이할 때, specialized model이 영양 피처에 학습한 패턴을 AIDET1D에 적용할 방법이 없다. 논문에서는 합성 데이터에 source의 local features를 그대로 포함시켜 해결하지만, CGM의 경우 "가짜 영양 데이터"를 주입하는 것의 생물학적 타당성이 의문.

3. **규모 불균형:** GLAM(26M) vs UCHTT1DM(27K)는 1000배 차이. 풀링 시 GLAM이 global model을 지배하게 되어, 실질적으로 "GLAM 모델 + 소폭 보정"이 될 가능성.

### 7.2 해결 전략

| 한계 | 제안 전략 |
|:---|:---|
| 시간 자기상관 파괴 | 윈도우 단위 (이미 vectorized된 형태)로 resampling → 시계열 구조는 피처 내부에 인코딩되어 있으므로 부분적으로 보존됨 |
| 고유 피처 매핑 불가 | Global model 예측만 사용 (f_G only) + Specialized 보정은 동일 피처셋 공유 source만 사용하는 "filtered specialization" 변형 |
| 규모 불균형 | Global model 학습 시 **데이터셋별 동수 subsampling** (각 데이터셋에서 최대 N_max 윈도우만 추출) |

---

## 8. 결론: 적합성 종합 판정

> [!IMPORTANT]
> **결론: 적용 가능하며, 유의미한 연구로 발전할 수 있다.**

| 판정 차원 | 등급 | 코멘트 |
|:---|:---:|:---|
| 방법론적 적합성 | **A** | GBM 2단계 전이 + Similarity Index는 우리 파이프라인에 직접 매핑 가능 |
| 데이터 준비도 | **A** | Tier 2.5의 harmonized features + dataset_summary_stats.csv로 즉시 착수 가능 |
| 신규성(Novelty) | **A+** | Cross-dataset CGM 전이학습은 매우 드문 연구 주제. Similarity Index + Drift Analysis 조합은 CGM 분야에서 최초 |
| 실행 가능성 | **B+** | 로컬 환경(32GB RAM, RTX 3060)으로 실행 가능하나, 12 LODO rounds ~12시간 소요 |
| 임상 시사점 | **A** | "새 병원의 CGM 데이터를 수집하지 않아도 기존 다른 병원 모델로 합리적 예측 가능" → 현실적 가치 높음 |

이 방법론은 **Tier 4의 첫 번째 실험 축**으로서, Cross-Dataset Transferability의 체계적 벤치마크를 제공하고, 이후의 딥러닝 기반 전이학습(Method 2, 3, ...)과 직접 비교할 수 있는 강력한 baseline이 된다.

---

*작성일: 2026-04-15 | Glucose-ML-Project Tier 4 Literature Review Series*
