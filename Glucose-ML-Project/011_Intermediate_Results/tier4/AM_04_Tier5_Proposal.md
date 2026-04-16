# Tier 5 — Nalmpatian Full Framework: 환자 수준 전이학습

> **배경:** Tier 4에서 확인된 3가지 한계를 직접 해결하는 실험 설계

---

## 0. Tier 4 → Tier 5: 무엇을 바꾸는가?

| Tier 4 한계 | Tier 5 해결책 | 기대 효과 |
|:---|:---|:---|
| **공통 20-dim만 사용** (인슐린·식사·활동량 미반영) | **Specialized Local Model**: 데이터셋별 고유 동적 피처(I/A/S) 투입 | 인슐린·식사 정보가 있는 데이터셋에서 성능 향상 |
| **Q/M 정적 피처 미포함** (나이·HbA1c·BMI 미반영) | **Global Feature 확장**: age, sex, HbA1c, BMI, DM_duration을 Global Feature에 추가 | 환자 간 이질성을 모델이 직접 포착 |
| **Patient-Level Resampling 근사** (환자 ID 미보존) | **Patient ID 보존 파이프라인**: 로더에서 환자 ID를 끝까지 유지, 환자 단위 선택 가능 | 정확한 환자 단위 리샘플링 |

---

## 1. 피처 구조 재설계

### 1.1 현재 (Tier 4) vs 제안 (Tier 5)

```
Tier 4 — 공통 피처만 (20 dim)
┌─────────────────────────────────────────────┐
│ glucose_t-5 ~ t-0 (6) + 파생 (14) = 20 dim │  모든 데이터셋 동일
└─────────────────────────────────────────────┘

Tier 5 — 3-Tier Feature 구조
┌─────────────────────────────────────────────────────────────────────┐
│ Tier A: Global Features (공통 25 dim)                               │
│ ├── 시계열 파생 (20 dim): glucose lags, Velocity, TIR, LBGI, ...   │
│ ├── Demographics (2 dim): age, sex                                  │
│ └── Quasi-Global Q/M (3 dim): HbA1c, BMI, DM_duration              │
│     ※ 결측 시 cohort별 중앙값으로 imputation                         │
├─────────────────────────────────────────────────────────────────────┤
│ Tier B: Local Dynamic Features (데이터셋별 0~378 dim)               │
│ ├── I (인슐린): basal, bolus, carb_bolus (lag 포함)                 │
│ ├── A (활동량): HR, steps, calories (lag 포함)                      │
│ └── S (영양):   carbs, protein, fat, meal_marker (lag 포함)         │
│     ※ 해당 데이터셋에만 존재. 없으면 0 dim                           │
├─────────────────────────────────────────────────────────────────────┤
│ Tier C: Cohort Indicator (3 dim)                                    │
│ └── cohort_T1DM, cohort_ND, cohort_GDM (one-hot)                   │
│     ※ 코호트 유형을 명시적으로 인코딩                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Q/M 변수 가용성과 Imputation 전략

| 변수 | 보유 데이터셋 | 커버리지 | 결측 시 전략 |
|:---|:---|:---:|:---|
| **age** | 12/12 | 100% | — (결측 없음) |
| **sex** | 12/12 | 100% | — (결측 없음) |
| **HbA1c** | AIDET1D(×), BIGIDEAs, Bris(×), CGM_D, CGM_L, CGMND(×), GLAM, HUPA(×), IOBP2, Park(×), PEDAP, UCHTT1DM | 7/12 (58%) | Cohort별 중앙값: T1DM→7.8, ND→5.4, GDM→5.9 |
| **BMI** | BIGIDEAs, CGM_D, CGM_L, GLAM, IOBP2, PEDAP, UCHTT1DM | 7/12 (58%) | Cohort별 중앙값: T1DM→25.5, ND→24.0, GDM→28.0 |
| **DM_duration** | AIDET1D, BIGIDEAs, CGM_D, CGM_L, HUPA, IOBP2, PEDAP, UCHTT1DM | 8/12 (67%) | ND→0년, 결측 T1DM→cohort 중앙값 |

> **핵심:** HbA1c=8.5인 환자와 HbA1c=5.2인 환자의 혈당 dynamics는 근본적으로 다르다. 이 정보가 Global Feature에 포함되면, Global Model이 **코호트 간 전이에서 환자 유형을 구분**할 수 있다.

### 1.3 Local Dynamic Features — 실제 가용 현황

| 데이터셋 | 인슐린 (I) | 활동량 (A) | 영양 (S) | 추가 dim (추정) |
|:---|:---:|:---:|:---:|:---:|
| AIDET1D | ❌ | ❌ | ❌ | **0** |
| BIGIDEAs | ❌ | ❌ | calories, carbs, protein, fat | **24** (6×4lag) |
| Bris-T1D | basal, bolus, carb_bolus | HR, steps, distance | ❌ | **36** (6×6lag) |
| CGMacros_Dex | ❌ | HR, steps, calories | carbs, protein, fat + nutrition(6) | **60+** |
| CGMacros_Lib | ❌ | HR, steps, calories | 위와 동일 | **60+** |
| CGMND | ❌ | ❌ | ❌ | **0** |
| GLAM | ❌ | ❌ | meal_marker | **6** (1×6lag) |
| HUPA-UCM | basal, bolus, carb_bolus | HR, steps, calories | ❌ | **36** (6×6lag) |
| IOBP2 | bolus, carb_bolus | ❌ | ❌ | **12** (2×6lag) |
| Park_2025 | ❌ | ❌ | carb_intake | **6** (1×6lag) |
| PEDAP | basal, bolus, carb_bolus | ❌ | ❌ | **18** (3×6lag) |
| UCHTT1DM | bolus | ❌ | ❌ | **6** (1×6lag) |

> **4개 데이터셋(AIDET1D, CGMND, + 부분적 GLAM, Park)은 Local Feature가 거의 없음** — 이 데이터셋들의 Specialized Model은 사실상 Global Model과 동일할 것.  
> **Bris-T1D, CGMacros, HUPA-UCM은 풍부한 Local Feature 보유** — Specialized Model의 효과를 가장 잘 검증할 수 있는 대상.

---

## 2. Tier 5 실험 구조

### 2.1 Nalmpatian Full Framework (LODO)

```
For M ∈ {Dataset_1, ..., Dataset_12}:   (M = 평가 대상, 나머지 = source)
  
  ┌─────────────────────────────────────────────────────────┐
  │ Step 1: Global Model (f_G)                               │
  │                                                          │
  │   11개 source 데이터셋을 풀링                               │
  │   Global Features 25 dim 사용                             │
  │   (시계열 20 + age + sex + HbA1c + BMI + DM_dur)          │
  │   LightGBM(300 rounds, lr=0.1) 학습                      │
  │                                                          │
  │   → f_G: 모든 데이터셋의 공통 혈당 패턴 포착               │
  └──────────────────────┬──────────────────────────────────┘
                         │ init_model로 전달
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 2: Specialized Models (f_j, j=1...11)               │
  │                                                          │
  │   각 source j에서:                                        │
  │   Global 25 dim + Local Dynamic j dim                    │
  │   (인슐린, 활동량, 영양 등 해당 데이터셋 고유 피처)          │
  │   f_G를 init_model로 사용하여 LightGBM(100 rounds) 추가   │
  │                                                          │
  │   → f_j: source j의 고유 패턴(인슐린 효과 등)을 추가 학습   │
  │                                                          │
  │   ※ Local 피처가 없는 데이터셋은 f_j ≈ f_G                 │
  │   ※ Local 피처가 풍부한 데이터셋은 f_j가 f_G보다 강력       │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 3: Similarity Index (s_j)                           │
  │                                                          │
  │   Tier 4의 혈당 통계 유사도 + Q/M 유사도 결합               │
  │   s_jM = exp(-d_jM) / Σ_k exp(-d_kM)                    │
  │                                                          │
  │   새로운 유사도 요소 (Tier 4 대비 추가):                    │
  │   - cohort_type 일치 (T1DM↔T1DM = 1.0, ND↔T1DM = 0.3)   │
  │   - HbA1c 차이 (|HbA1c_j - HbA1c_M|)                    │
  │   - Local 피처 겹침 비율 (공유하는 I/A/S 변수 수)           │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 4: Patient-Level Resampling (합성 데이터 구축)        │
  │                                                          │
  │   각 source j에서 p_j명의 환자를 선택 (p_j ∝ s_j)         │
  │   선택된 환자의 전체 시계열을 원본 그대로 보존                │
  │                                                          │
  │   ※ patient_ids를 보존하므로 정확한 환자 단위 선택 가능     │
  │   ※ 노이즈 주입 없음 (시계열 구조 보존, Prioleau 준수)     │
  │                                                          │
  │   합성 데이터셋 X_M 구성:                                  │
  │   - 각 행에 source_dataset_id 기록                        │
  │   - Global Features는 모든 행에 동일 구조                  │
  │   - Local Features는 source별로 다른 차원                  │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 5: Transfer Prediction                              │
  │                                                          │
  │   방법 A (Nalmpatian 원본):                               │
  │     D̂_M = Σ_j [ f_G(X_M^(j)_global) + f_j(X_M^(j)) ]   │
  │     → Global 예측 + Source별 Specialized 보정의 합         │
  │                                                          │
  │   방법 B (Pooled + FT, Tier 4 최고 성능):                  │
  │     f_G_ft = f_G를 M의 train set으로 fine-tune             │
  │     D̂_M = f_G_ft(X_M_global)                             │
  │     → Global Model의 target 적응                          │
  │                                                          │
  │   방법 C (결합, 제안):                                     │
  │     f_G_ft를 학습 후,                                     │
  │     M에 Local 피처가 있으면 f_M_specialized도 학습          │
  │     D̂_M = f_G_ft(global) + f_M_spec(global + local)      │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │ Step 6: Drift Analysis (오차 원인 분석)                    │
  │                                                          │
  │   ratio_jM = RMSE_transfer(j→M) / RMSE_self(M)           │
  │                                                          │
  │   GLM(Gaussian):                                         │
  │     ratio ~ cohort_match + sensor_match + HbA1c_gap      │
  │           + local_feature_overlap + log(n_source)         │
  │                                                          │
  │   → "어떤 요인이 전이 오차를 유발하는가?" 정량적 답변       │
  └─────────────────────────────────────────────────────────┘
```

---

## 3. 기술적 구현 계획

### 3.1 데이터 로더 개선 (`tier5_data_utils.py`)

현재 `load_dataset_global_features()`를 확장하여:

```python
def load_dataset_tier5(dset_path, seed=42):
    """
    Returns:
      {
        'name': str,
        'X_global_train':  np.array (N, 25),   # 시계열 20 + Q/M 5
        'X_local_train':   np.array (N, L_j),   # 데이터셋별 고유 동적 피처
        'y_train':         np.array (N,),
        'patient_ids_train': np.array (N,),     # ← 환자 ID 보존 (NEW)
        'static_features': {                    # ← 환자별 Q/M (NEW)
            pid_0: {'age': 35, 'sex': 1, 'HbA1c': 7.2, 'BMI': 24.5, 'DM_dur': 12},
            pid_1: {...},
            ...
        },
        'local_feature_names': ['basal_t-5', ..., 'bolus_t-0', ...],
        ...동일 구조로 val, test...
      }
    """
```

**핵심 변경:**
1. `patient_ids`를 X와 함께 반환 (현재는 split 후 버림)
2. Q/M 정적 피처를 CSV에서 읽어 환자별로 매핑
3. Local dynamic 피처(I/A/S)를 windowing에 포함
4. 결측 imputation 로직 내장

### 3.2 Q/M 정적 피처 추출 소스

| 데이터셋 | Q/M 소스 파일 | 추출 방법 |
|:---|:---|:---|
| AIDET1D | `questionnaire/` 또는 `demographics.csv` | CSV 파싱 |
| BIGIDEAs | `participant_info.csv` | 직접 매핑 |
| Bris-T1D | 데이터 딕셔너리 참조 | 제한적 (age, sex만) |
| CGMacros | `participant_demographics.csv` | 풍부한 Q/M |
| CGMND | `demographics.csv` | age, sex만 |
| GLAM | `clinical_data.csv` | HbA1c, BMI, OGTT 등 |
| HUPA-UCM | `patients.csv` | age, sex, DM_dur |
| IOBP2 | `subject_info.csv` | age, sex, HbA1c, BMI |
| Park_2025 | `metadata.csv` | age, sex |
| PEDAP | `participant_info.csv` | age, sex, HbA1c, BMI |
| UCHTT1DM | `demographics.csv` | age, sex, HbA1c, BMI |

> ⚠️ **각 데이터셋의 실제 파일 구조를 확인하여 Q/M 추출 코드를 데이터셋별로 작성해야 함** — 이는 구현의 가장 큰 노동 집약적 부분.

### 3.3 Local Dynamic Feature 추출

현재 `build_windows_with_features(df, feat_cols)`에서 `feat_cols`에 glucose만 넘기는 것을 확장:

```python
# Tier 4 (현재)
feat_cols = ['glucose_value_mg_dl']

# Tier 5 (확장)
feat_cols = ['glucose_value_mg_dl']  # 항상 포함

# 데이터셋별 추가
if dataset_name in ['Bris-T1D_Open', 'HUPA-UCM']:
    feat_cols += ['basal_rate', 'bolus_dose', 'carb_bolus']
if dataset_name in ['CGMacros_Dexcom', 'CGMacros_Libre']:
    feat_cols += ['heart_rate', 'steps', 'active_calories', 'carbs', 'protein', 'fat']
if dataset_name in ['IOBP2', 'PEDAP']:
    feat_cols += ['bolus_dose', 'carb_bolus']
# ... etc
```

> 이렇게 하면 `build_windows_with_features`의 windowing 로직이 자동으로 각 피처의 lookback(6개)을 생성.

---

## 4. 비교 실험 매트릭스

### 4.1 Tier 5에서 비교할 방법론

| # | 코드명 | Global Features | Local Features | Q/M Static | Patient ID | 비고 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| T4① | self_20 | 20dim | ❌ | ❌ | ❌ | Tier 4 baseline |
| T5① | **self_25** | **25dim** | ❌ | ✅ | ❌ | Q/M 추가 효과 측정 |
| T5② | **self_25+L** | **25dim** | ✅ | ✅ | ❌ | Local 피처 추가 효과 측정 |
| T4⑥ | pool_ft_20 | 20dim | ❌ | ❌ | ❌ | Tier 4 최고 |
| T5③ | **pool_ft_25** | **25dim** | ❌ | ✅ | ❌ | Q/M 추가 Global Model |
| T5④ | **nalm_full** | **25dim** | ✅ | ✅ | ✅ | **Nalmpatian 완전 구현** |
| T5⑤ | **nalm_full+drift** | **25dim** | ✅ | ✅ | ✅ | + Drift 오차 분석 |

### 4.2 검증할 연구 질문

| RQ | 질문 | 비교 |
|:---:|:---|:---|
| **RQ1** | Q/M 정적 피처(HbA1c, BMI, DM_dur)가 전이 성능을 개선하는가? | T4①(20dim) vs T5①(25dim) |
| **RQ2** | Local 동적 피처(인슐린, 영양)가 추가 개선을 주는가? | T5①(25) vs T5②(25+L) |
| **RQ3** | Q/M가 Global Model에 추가되면 풀링 전이학습이 개선되는가? | T4⑥(pool_20) vs T5③(pool_25) |
| **RQ4** | Nalmpatian Full(Global→Specialized)이 단순 Pooled FT보다 나은가? | T5③ vs T5④ |
| **RQ5** | 전이 오차의 주요 원인은 무엇인가? | T5⑤ Drift Analysis |
| **RQ6** | Local 피처가 풍부한 데이터셋(Bris, CGMacros, HUPA)에서 Specialized Model 효과가 더 큰가? | T5④의 데이터셋별 분석 |

### 4.3 예상 결과 가설

```
성능 순서 예상:

self_20 (17.63)  ← Tier 4 baseline
   ↓ Q/M 추가 (-0.3~0.5 예상)
self_25 (17.1~17.3)
   ↓ Local 피처 추가 (데이터셋에 따라 0~-2.0)
self_25+L (15.5~17.0 — 데이터셋별 편차 클 것)
   ↓ 풀링 전이학습 (-0.4 예상)
pool_ft_25 (16.7~17.0)
   ↓ Specialized Model 
nalm_full (16.5~17.0)
```

> **핵심 가설:** Q/M 추가가 가장 큰 개선을 줄 것이다. HbA1c가 7.8인 환자와 5.4인 환자를 구분할 수 있으면, 코호트 간 전이에서 모델이 환자 유형에 맞게 예측을 조정할 수 있기 때문.

---

## 5. 구현 일정 추정

| 단계 | 작업 | 예상 시간 | 난이도 |
|:---:|:---|:---:|:---:|
| **1** | Q/M 정적 피처 추출 (12개 데이터셋별 파싱) | 3~4시간 | ⭐⭐⭐ (노동 집약) |
| **2** | Local Dynamic 피처 추출 로직 확장 | 2~3시간 | ⭐⭐ |
| **3** | `tier5_data_utils.py` 작성 (로더 + imputation) | 2시간 | ⭐⭐ |
| **4** | `08_tier5_nalmpatian_full.py` 작성 | 3시간 | ⭐⭐⭐ |
| **5** | 실험 실행 (T5①~⑤) | 4~6시간 (자동) | ⭐ |
| **6** | Drift Analysis (GLM) | 1시간 | ⭐⭐ |
| **7** | 결과 분석 및 보고서 작성 | 2시간 | ⭐⭐ |
| **합계** | | **~17~21시간** | |

> ⚠️ **가장 큰 블로커: Step 1 (Q/M 추출)**. 12개 데이터셋의 원본 파일 구조가 각각 다르므로, 데이터셋별로 개별 파싱 코드를 작성해야 한다.

---

## 6. 논문 기여 관점 — Tier 5가 추가하는 것

| Tier 4 기여 | Tier 5 추가 기여 |
|:---|:---|
| C1: HPO 무의미 | — (유지) |
| C2: 전이 격차 정량화 | **C2+: Q/M 추가 시 전이 격차가 줄어드는 정도 정량화** |
| C3: FT로 94% 해소 | **C3+: Specialized Model이 FT를 넘어서는 경우 식별** |
| C4: Pooled FT > self | **C4+: Pooled FT + Q/M > Pooled FT, 그 차이의 원인** |
| C5: 풀링 > 앙상블 | — (유지) |
| C6: Trend 개선 | **C6+: Local 피처(인슐린)가 Trend 예측을 추가 개선하는가?** |
| C7: 코호트 유형이 핵심 | **C7+: Drift Analysis로 정량적 증거 제시** |
| — | **C8 (신규): 환자 수준 전이학습 프레임워크의 일반화 가능성 (Nalmpatian→CGM)** |
| — | **C9 (신규): 인슐린·영양 피처의 전이 가치 (transferable vs dataset-specific)** |

---

## 7. 결정이 필요한 사항

### 안건 7.1: Q/M 추출 범위

**선택지:**
- **A. 최소 (age + sex만):** 12/12 커버리지. 구현 1시간. 하지만 개선 효과 작을 것.
- **B. 중간 (age + sex + HbA1c + BMI + DM_dur):** 7~8/12 + imputation. 구현 3시간. **추천.**
- **C. 최대 (OGTT, cholesterol, lifestyle 포함):** 1~4/12. 구현 5시간. 불균형 심함.

### 안건 7.2: Local Dynamic Feature 범위

**선택지:**
- **A. 인슐린만 (I):** 8/12 보유. 가장 일관적. LightGBM에서 자연스러움.
- **B. 인슐린 + 활동량 (I+A):** 4/12 보유. CGMacros, Bris, HUPA에서만 효과.
- **C. 전체 (I+A+S):** 12/12 중 일부. 구현 복잡. **→ 단, 없는 데이터셋은 Global만 사용하므로 문제없음.**

> **추천:** Q/M은 B (중간), Local은 C (전체) — 있는 것은 다 넣고, 없으면 Global만 쓰게 설계.

### 안건 7.3: 우선 실행 순서

```
빠른 승리 (Quick Win):
  T5① self_25 — Q/M만 추가. 구현 3시간.
  → "HbA1c, BMI 추가만으로 얼마나 좋아지는가?" 즉시 확인

중기:
  T5③ pool_ft_25 — Q/M + 풀링. 기존 코드 약간 수정.
  → Tier 4 최고 성능에 Q/M 추가 효과 확인

장기:
  T5④ nalm_full — Local + Patient-Level 전체 구현.
  → 논문의 핵심 기여
```

---

*Glucose-ML-Project · Tier 5 Proposal · 2026-04-16*
