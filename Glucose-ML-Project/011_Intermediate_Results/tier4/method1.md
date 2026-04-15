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

> 아래는 Nalmpatian et al. (2025), Table "Algorithm 1: Algorithmic representation of the transfer framework"의 번역이다.

**Algorithm 1. 전이 프레임워크의 알고리즘적 표현**

**Step 1.** 국가 1, …, K의 데이터셋을 포함하는 풀링된 데이터셋에 대해 **글로벌 GBM 모델 f_G**를 학습한다. 여기서 사용되는 피처는 모든 국가에 걸쳐 공통적으로 비교 가능한 글로벌 피처만을 포함한다.

**Step 2.** 각 국가 j ∈ {1, …, K}에 대해, 해당 국가 j의 데이터셋을 사용하여 **로컬 GBM 모델 f_j**를 학습한다. 이 데이터셋에는 글로벌 피처와 국가 j에 고유한 로컬 피처가 모두 포함된다. 이 모델들은 (통상적인 무작위 초기화 대신) **사전학습된 글로벌 GBM 모델의 출력 예측값을 초기값으로 사용**하여 초기화된다.

**Step 3.** 국가 M (= UK)에 대해, 사전 정의된 유사도 메트릭에 기반한 외부 데이터를 사용하여 각 국가 j와의 **유사도 점수(similarity scores)** 를 계산한다. 이 메트릭에는 생명보험, 경제, 사망률에 특화된 요인들이 포함될 수 있다.

**Step 4.** 각 국가 j ∈ {1, …, K}에 대해, 국가 M (UK)의 합성 데이터셋을 생성하기 위해 다음 단계를 수행한다:

  - 계산된 유사도 점수를 사용하여 각 국가 j의 데이터셋에서 국가 M의 합성 데이터셋에 기여할 **익스포져(exposure)를 비례적으로 재표집(resample)** 한다. 국가 M의 총 익스포져 E_M은 각 국가 j에서 재표집된 익스포져의 합과 같으며, 총합은 100,000,000이 되도록 한다.
  - **데이터 증강(data augmentation)** 을 적용하여 피처에 노이즈를 추가함으로써 변동성을 생성하고 모델의 강건성을 향상시킨다.
  - 데이터셋 내 **인구 전체 사망률 변수를 국가 M의 것으로 교체**하여, 데이터셋을 국가 M의 사망률 조건에 정렬시킨다.
  - 재표집 및 증강된 데이터를 컴파일하여 국가 M의 **합성 데이터셋 X_M**을 구성한다. 이 데이터셋은 행-블록 행렬(row-block matrix)로, 각 블록은 서로 다른 차원을 가진 특정 국가 j의 데이터에 대응하며, 글로벌 피처와 로컬 피처를 모두 포함한다. 첫 번째 열들은 글로벌 모델이 적용될 글로벌 피처로 구성된다. **각 행의 출처(origin)를 기록**하여, 해당 국가에 대해 학습된 전문화된 GBM 모델이 이후에 적용될 수 있도록 보장한다.

**Step 5.** 글로벌 모델 f_G와 각각의 로컬 모델 f_j를 사용하여 국가 M의 합성 데이터셋에 대한 **기대값 D̂_M을 예측**한다:

```
D̂_M = Σ_j [ f_G(X_M^(j)_global) + f_j(X_M^(j)_local) ]
```

여기서 D̂_M^(j)는 국가 j로부터 도출된 국가 M 합성 데이터의 기대 사망률이다. f_G(·) 항은 국가 j에서 도출된 국가 M의 합성 데이터셋의 글로벌 피처를 사용한 글로벌 모델의 예측값이다. f_j(·) 항은 합성 데이터셋 X_M에서 국가 j로부터 유래한 부분에 적용된 국가 j의 로컬 모델에 의한 보정(adjustment)을 나타낸다. 이는 글로벌 모델의 예측이 유사도 점수에 의해 결정된 바와 같이 국가 M과 가장 유사한 국가 j의 특정 특성을 반영하도록 미세 조정(fine-tune)되는 것을 보장한다.

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

> [!CAUTION]
> **한계 1은 Prioleau et al. (2025)의 벤치마크 가이드라인과 직접 충돌하는 치명적 문제이다.** 논문의 Step 4(합성 데이터 생성)를 CGM 시계열에 그대로 적용할 수 없으며, 대안적 전이 메커니즘이 필수적이다. → **Appendix A/B/C** 참조.

1. **합성 데이터 생성의 시계열 구조 파괴 (치명적):**

   Nalmpatian et al.의 Step 4는 source 데이터셋에서 행(row)을 독립적으로 resampling하고 노이즈를 주입하여 target용 합성 데이터를 생성한다. 이 접근은 **집계된(aggregated) 보험 사망률 데이터**에서는 유효하나, **CGM 시계열 데이터**에서는 다음 이유로 적용 불가능하다:

   - **시간적 자기상관 파괴:** CGM 윈도우는 연속적 시간 의존성을 가진다. 독립 resampling은 환자 내 시간 순서를 해체하여, lag 피처(t-0 ~ t-5), Velocity, Acceleration, Jerk 등 연속 윈도우 차분으로 정의된 파생 피처의 물리적 의미를 파괴한다.
   - **Data leakage 위험:** 시간 순서가 파괴되면 미래 윈도우가 과거로 혼입될 수 있으며, 이는 Prioleau et al.이 [48]을 인용하여 경고한 "공통적 함정(common pitfalls)"에 해당한다.
   - **노이즈 주입의 비생물학성:** 혈당값에 가우시안 노이즈를 주입하면 생리학적으로 불가능한 혈당 궤적(예: 30mg/dL→300mg/dL 5분 내 변동)이 생성될 수 있다.

   **Prioleau et al. (2025)의 관련 가이드라인:**

   > *"Informed by best practice guidelines, we excluded periods of missing data from our analysis to **avoid interpolation or extrapolation approaches that can introduce errors** into the prediction model and reported accuracy."* — §5.3, p.7
   >
   > *"Models should **not be evaluated on test sets that have been smoothed or that include interpolation of missing data**. These common pitfalls can **lead to invalid estimates of accuracy**."* — §6.1, p.10

   합성 데이터 생성(resampling + noise injection)은 interpolation/extrapolation보다 더 심한 데이터 왜곡이므로, Prioleau et al.의 기준을 명백히 위반한다.

2. **고유 피처 매핑 문제:** CGMacros(영양 데이터 보유) → AIDET1D(영양 데이터 없음)로 전이할 때, specialized model이 영양 피처에 학습한 패턴을 AIDET1D에 적용할 방법이 없다. 논문에서는 합성 데이터에 source의 local features를 그대로 포함시켜 해결하지만, CGM의 경우 "가짜 영양 데이터"를 주입하는 것의 생물학적 타당성이 의문.

3. **규모 불균형:** GLAM(26M) vs UCHTT1DM(27K)는 1000배 차이. 풀링 시 GLAM이 global model을 지배하게 되어, 실질적으로 "GLAM 모델 + 소폭 보정"이 될 가능성.

### 7.2 해결 전략: 데이터 레벨 전이 → 모델 레벨 전이로 대체

논문의 핵심 가치(2단계 GBM 구조 + 유사도 가중)를 보존하되, **Step 4(합성 데이터 생성)를 모델 레벨 전이로 대체**하는 3가지 대안을 제시한다. 각 대안의 상세 설계, 구현 코드, Prioleau et al. (2025) 준수 여부 분석은 **Appendix A/B/C**에 기술한다.

| 대안 | 핵심 아이디어 | 시계열 보존 | Prioleau 준수 | 로컬 피처 활용 |
|:---|:---|:---:|:---:|:---:|
| **A. Similarity-Weighted Model Fusion** | M의 실제 데이터에 f_G + Σ s_j·f_j 적용 | ✅ 완전 | ✅ | ⚠️ 공통 피처 부분만 |
| **B. Instance-Weighted Pooled Training** | K개 풀링 학습 시 유사도 비례 가중치 | ✅ 완전 | ✅ | ❌ 공통 피처만 |
| **C. Patient-Level Transfer** | 환자 전체 시퀀스 단위로 유사도 비례 선택 | ✅ 완전 | ✅ | ✅ 가능 |

> [!IMPORTANT]
> **대안 C(Patient-Level Transfer)를 주 전략으로 권장한다.** 시계열 구조를 100% 보존하면서, 논문의 유사도 기반 재표집 개념을 환자 단위로 격상시켜 적용할 수 있다. Prioleau et al. (2025)의 모든 가이드라인을 준수한다. 상세 분석은 Appendix C 참조.

| 한계 | 해결 전략 |
|:---|:---|
| 시간 자기상관 파괴 | **Step 4를 폐기**하고 모델 레벨 전이로 대체 (Appendix A/B/C) |
| 고유 피처 매핑 불가 | Global model 예측만 사용 (f_G only) + Specialized 보정은 동일 피처셋 공유 source만 사용하는 "filtered specialization" 변형 |
| 규모 불균형 | Global model 학습 시 **데이터셋별 동수 subsampling** (각 데이터셋에서 최대 N_max 윈도우만 추출) |

---

## 8. 결론: 적합성 종합 판정

> [!IMPORTANT]
> **결론: 핵심 프레임워크(2단계 GBM + Similarity Index + Drift Analysis)는 적용 가능하며 유의미하다. 단, Step 4(합성 데이터 생성)는 CGM 시계열 특성상 모델 레벨 전이(Appendix C 권장)로 대체해야 한다.**

| 판정 차원 | 등급 | 코멘트 |
|:---|:---:|:---|
| 방법론적 적합성 | **A-** | 프레임워크 전체는 적합하나 Step 4 수정 필수 → Appendix C로 해결 |
| 데이터 준비도 | **A** | Tier 2.5의 harmonized features + dataset_summary_stats.csv로 즉시 착수 가능 |
| 신규성(Novelty) | **A+** | Cross-dataset CGM 전이학습은 매우 드문 연구 주제. Similarity Index + Drift Analysis 조합은 CGM 분야에서 최초 |
| 실행 가능성 | **B+** | 로컬 환경(32GB RAM, RTX 3060)으로 실행 가능하나, 12 LODO rounds ~12시간 소요 |
| 임상 시사점 | **A** | "새 병원의 CGM 데이터를 수집하지 않아도 기존 다른 병원 모델로 합리적 예측 가능" → 현실적 가치 높음 |
| Prioleau 가이드라인 준수 | **A** | Appendix C 채택 시 interpolation/extrapolation/smoothing 금지 원칙 완전 준수 |

이 방법론은 **Tier 4의 첫 번째 실험 축**으로서, Cross-Dataset Transferability의 체계적 벤치마크를 제공하고, 이후의 딥러닝 기반 전이학습(Method 2, 3, ...)과 직접 비교할 수 있는 강력한 baseline이 된다.

---

## Appendix A: Similarity-Weighted Model Fusion

### A.1 개요

논문의 Step 1~3은 그대로 유지하되, **Step 4(합성 데이터 생성)를 폐기**하고 Step 5를 수정하여, target M의 **실제 테스트 데이터**에 직접 모델을 적용한다.

### A.2 수정된 알고리즘

```
Step 1: [동일] Global GBM f_G를 K개 pooled dataset의 공통 피처로 학습
Step 2: [동일] 각 j ∈ K에 대해 Specialized GBM f_j를 (공통 + 고유) 피처로 학습,
        f_G 출력을 초기값으로 사용
Step 3: [동일] 유사도 점수 s_j 계산
Step 4: [폐기] 합성 데이터 생성하지 않음
Step 5: [수정] Target M의 실제 테스트 데이터에 직접 예측:

        ŷ_M = f_G(X_M_global) + Σ_j [ s_j · f_j(X_M_global) ]

        → 각 specialized model의 공통 피처 부분만 사용하여 예측,
          유사도 점수로 가중 합산
```

### A.3 구현

```python
import lightgbm as lgb
import numpy as np

def predict_transfer_A(X_M_global, global_model, specialized_models, similarity_scores):
    """
    대안 A: Similarity-Weighted Model Fusion
    
    X_M_global: target M의 실제 테스트 데이터 (공통 피처만, N × 20~25 dim)
    global_model: f_G
    specialized_models: {j: f_j} for j in K
    similarity_scores: {j: s_j} for j in K, sum=1
    """
    # Global model 예측 (전체 공통 패턴)
    y_global = global_model.predict(X_M_global)
    
    # Specialized models의 가중 합산 (로컬 보정)
    y_local_sum = np.zeros(len(X_M_global))
    for j, model_j in specialized_models.items():
        s_j = similarity_scores[j]
        # specialized model의 공통 피처 부분만으로 예측
        # (f_j는 f_G 위에 추가 트리를 쌓은 것이므로, 공통 피처로도 잔차 보정 가능)
        y_j = model_j.predict(X_M_global) - global_model.predict(X_M_global)
        y_local_sum += s_j * y_j
    
    return y_global + y_local_sum
```

### A.4 장단점

| 장점 | 단점 |
|:---|:---|
| 시계열 구조 100% 보존 — M의 실제 데이터를 그대로 사용 | Specialized model의 로컬 피처(insulin, nutrition 등) 활용 불가 |
| 구현 단순 — 학습 완료 후 inference만 수행 | f_j가 공통 피처로만 평가되므로 로컬 보정 효과 제한적 |
| Prioleau 가이드라인 완전 준수 | 유사도 가중치가 모델 예측에만 적용되어, 논문의 데이터 레벨 가중 개념이 약화 |

### A.5 Prioleau et al. (2025) 준수 여부

| 기준 | 준수 | 근거 |
|:---|:---:|:---|
| Interpolation/extrapolation 금지 | ✅ | 합성 데이터를 생성하지 않음. M의 실제 데이터만 사용 |
| Test set smoothing 금지 | ✅ | 테스트 데이터에 어떤 변형도 가하지 않음 |
| 결측 구간 제외 | ✅ | 기존 전처리 파이프라인의 결측 처리 그대로 유지 |

---

## Appendix B: Instance-Weighted Pooled Training

### B.1 개요

합성 데이터 생성 대신, K개 source 데이터셋을 **유사도 비례 인스턴스 가중치**를 부여하여 하나의 모델로 pooled 학습한다. 유사한 source의 데이터 포인트에 높은 학습 가중치를 부여하여, 논문의 "유사도 비례 재표집" 개념을 **데이터 복제 없이** 동등하게 구현한다.

### B.2 수정된 알고리즘

```
Step 1-3: [동일]
Step 4:   [대체] K개 source 데이터셋을 공통 피처로 풀링.
          각 데이터셋 j의 모든 윈도우에 가중치 w = s_j / N_j 부여.
          (s_j: 유사도 점수, N_j: 데이터셋 j의 윈도우 수)
Step 5:   가중 풀링 데이터로 단일 GBM 학습 → M의 실제 테스트에 적용
```

### B.3 구현

```python
import lightgbm as lgb
import numpy as np
import pandas as pd

def train_transfer_B(source_datasets, similarity_scores, target_M_test):
    """
    대안 B: Instance-Weighted Pooled Training
    """
    X_all, y_all, w_all = [], [], []
    
    for j, ds in source_datasets.items():
        s_j = similarity_scores[j]
        N_j = len(ds)
        weight_j = s_j / N_j  # 유사도 높을수록 + 데이터셋 크기 보정
        
        X_all.append(ds[global_features])
        y_all.append(ds['target'])
        w_all.append(np.full(N_j, weight_j))
    
    X_pool = pd.concat(X_all)
    y_pool = pd.concat(y_all)
    w_pool = np.concatenate(w_all)
    
    # LightGBM의 weight 파라미터로 유사도 가중 학습
    train_data = lgb.Dataset(X_pool, label=y_pool, weight=w_pool)
    
    model = lgb.train(
        params={'objective': 'regression', 'metric': 'rmse'},
        train_set=train_data,
        num_boost_round=500
    )
    
    # M의 실제 테스트 데이터에 직접 예측
    y_pred = model.predict(target_M_test[global_features])
    return y_pred
```

### B.4 장단점

| 장점 | 단점 |
|:---|:---|
| 시계열 구조 100% 보존 — 각 데이터셋 내 윈도우 순서 유지 | 공통 피처만 사용 가능 (로컬 피처 차원이 다르므로 풀링 불가) |
| LightGBM `weight` 한 줄로 구현 — 논문의 재표집을 수학적으로 동등하게 구현 | 단일 모델 → Global + Specialized 2단계 구조 포기 |
| 규모 불균형 자동 해결 (w = s_j / N_j) | |

### B.5 Prioleau et al. (2025) 준수 여부

| 기준 | 준수 | 근거 |
|:---|:---:|:---|
| Interpolation/extrapolation 금지 | ✅ | 실제 데이터만 사용. 가중치 변경은 데이터 자체를 변형하지 않음 |
| Test set smoothing 금지 | ✅ | 테스트 셋에 어떤 변형도 가하지 않음 |
| 결측 구간 제외 | ✅ | 기존 전처리 파이프라인 유지 |

---

## Appendix C: Patient-Level Transfer (권장)

### C.1 개요

논문의 Step 4(행 단위 재표집)를 **환자(subject) 단위 재표집**으로 격상시킨다. 개별 윈도우가 아닌 **환자의 전체 시계열 시퀀스**를 재표집의 단위로 사용하여, 각 환자 내부의 시간적 자기상관 구조를 100% 보존한다.

### C.2 핵심 논리: 왜 환자 단위가 Prioleau et al.에 부합하는가?

Prioleau et al. (2025)이 금지하는 것: **데이터 포인트 수준의 인위적 조작**
- ❌ Interpolation: 실측되지 않은 시점의 혈당값을 추정하여 삽입
- ❌ Extrapolation: 관측 범위 밖의 혈당값을 추정
- ❌ Smoothing: 기존 혈당값을 인위적으로 변형
- ❌ Synthetic resampling + noise: 기존 데이터 포인트를 복제하고 노이즈 주입 (위 3가지의 상위호환)

Prioleau et al. (2025)이 금지하지 **않는** 것: **학습 데이터 구성 전략**
- ✅ 어떤 환자의 데이터를 학습에 포함할지 선택 (subject selection)
- ✅ 어떤 데이터셋을 학습에 포함할지 선택 (dataset selection)
- ✅ 학습/검증/테스트 분할 방식 결정

**환자 단위 전이의 핵심:** 각 환자의 시계열 데이터를 **한 글자도 변형하지 않는다.** 단지 "어떤 환자들을 학습 풀에 포함시킬 것인가"의 선택 문제로 전환한다. 이는 의학 연구에서 코호트를 구성할 때 포함/배제 기준(inclusion/exclusion criteria)을 적용하는 것과 동일한 수준의 조작이다.

### C.3 수정된 알고리즘

```
Step 1: [동일] Global GBM f_G를 K개 pooled dataset의 공통 피처로 학습

Step 2: [동일] 각 j ∈ K에 대해 Specialized GBM f_j를 (공통 + 고유) 피처로 학습,
        f_G 출력을 초기값으로 사용

Step 3: [동일] 유사도 점수 s_j 계산

Step 4: [수정 — Patient-Level Resampling]
        각 source 데이터셋 j에서, 유사도 점수 s_j에 비례한 수의 환자를
        무작위 선택한다. 선택된 환자의 전체 시계열 시퀀스를 변형 없이
        그대로 학습 풀에 포함한다.

        구체적으로:
        (a) 총 학습 환자 수 N_total 설정 (예: 전체 환자의 80%)
        (b) 데이터셋 j에서 선택할 환자 수: n_j = round(N_total × s_j)
        (c) 데이터셋 j에서 n_j명을 무작위 선택 (replace=True if n_j > N_j)
        (d) 선택된 환자의 모든 시계열 윈도우를 원본 그대로 유지
        (e) 데이터 수준의 어떤 변형(노이즈, 보간, 평활)도 적용하지 않음

Step 5: [수정 — 2단계 예측]
        (a) 선택된 환자 풀로 similarity-aware 모델 학습 (또는 f_G + f_j 구조 유지)
        (b) Target M의 실제 테스트 데이터에 직접 적용하여 예측
```

### C.4 구현

```python
import lightgbm as lgb
import numpy as np
import pandas as pd

def train_transfer_C(source_datasets, source_patient_ids, similarity_scores, 
                     target_M_test, N_total_patients=500):
    """
    대안 C: Patient-Level Transfer
    
    source_datasets: {j: DataFrame} — 각 데이터셋의 전체 윈도우 데이터
    source_patient_ids: {j: list} — 각 데이터셋의 환자 ID 목록
    similarity_scores: {j: s_j} — 유사도 점수 (sum=1)
    target_M_test: DataFrame — target M의 실제 테스트 데이터
    N_total_patients: int — 학습 풀에 포함할 총 환자 수
    """
    selected_data = []
    
    for j, patient_ids in source_patient_ids.items():
        s_j = similarity_scores[j]
        n_j = max(1, round(N_total_patients * s_j))  # 유사도 비례 환자 수
        
        # 환자 단위 선택 (n_j > len(patient_ids)이면 중복 허용)
        selected_ids = np.random.choice(
            patient_ids, 
            size=min(n_j, len(patient_ids)), 
            replace=False
        )
        
        # 선택된 환자의 전체 시퀀스를 원본 그대로 포함
        ds = source_datasets[j]
        patient_data = ds[ds['patient_id'].isin(selected_ids)]
        selected_data.append(patient_data)
    
    # 풀링 (공통 피처만)
    pool = pd.concat(selected_data)
    
    # Step 1: Global model 학습
    global_model = lgb.train(
        params={'objective': 'regression', 'metric': 'rmse'},
        train_set=lgb.Dataset(pool[global_features], pool['target']),
        num_boost_round=500
    )
    
    # Step 5: M의 실제 테스트 데이터에 적용
    y_pred = global_model.predict(target_M_test[global_features])
    return y_pred, global_model
```

### C.5 Prioleau et al. (2025) 준수 여부 — 조항별 상세 검증

| Prioleau et al. 기준 | 준수 | 상세 근거 |
|:---|:---:|:---|
| **Interpolation 금지** (§5.3) | ✅ | 관측되지 않은 시점의 혈당값을 어떤 방식으로도 생성하지 않음 |
| **Extrapolation 금지** (§5.3) | ✅ | 관측 범위 밖의 혈당값을 추정하지 않음 |
| **Test set smoothing 금지** (§6.1) | ✅ | target M의 테스트 데이터에 어떤 변형도 가하지 않음 |
| **결측 구간 제외** (§5.3) | ✅ | 기존 Tier 2.5 전처리의 결측 제거 결과를 그대로 사용 |
| **데이터 노이즈 주입 금지** (implied) | ✅ | 선택된 환자 데이터에 노이즈를 추가하지 않음 |
| **시계열 시간 순서 보존** (implied) | ✅ | 환자의 전체 시퀀스를 원본 그대로 유지. 윈도우 순서 변경 없음 |
| **Train/Test 분리** ([48]) | ✅ | LODO 구조에서 M의 데이터는 학습에 포함되지 않음 |

> [!TIP]
> **대안 C가 논문의 원래 의도에 가장 충실한 이유:** Nalmpatian et al.의 Step 4가 하는 것은 본질적으로 "유사한 source에서 더 많은 데이터를 가져와 target을 대리한다"이다. 행 단위를 환자 단위로 격상시키면 이 본질을 보존하면서 시계열 무결성도 확보한다. 보험 사망률에서 "행 = 독립적인 계약자 그룹"이었던 것이, CGM에서는 "환자 = 독립적인 시계열 단위"에 대응한다. **재표집의 단위(grain)만 데이터 특성에 맞게 조정**한 것이다.

### C.6 장단점

| 장점 | 단점 |
|:---|:---|
| 시계열 구조 100% 보존 | 환자 수가 적은 데이터셋(BIGIDEAs 16명)에서 선택 해상도가 낮음 |
| Prioleau 가이드라인 완전 준수 | 동일 환자 중복 선택 시(replace=True) 학습 데이터 편향 가능 |
| 논문의 2단계 구조 + 유사도 가중 개념 보존 | 환자 간 이질성이 큰 경우, 환자 단위 선택이 충분히 세밀하지 않을 수 있음 |
| Specialized model에 로컬 피처 투입 가능 | 유사도 점수가 데이터셋 수준이므로, 환자 개인의 유사도는 반영하지 않음 |
| 생물학적 타당성 유지 — 실제 환자의 실제 혈당 궤적만 사용 | |

---

*작성일: 2026-04-15 (Updated) | Glucose-ML-Project Tier 4 Literature Review Series*
