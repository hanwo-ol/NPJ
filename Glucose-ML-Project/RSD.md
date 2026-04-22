# Research Scenario Design — Glucose-ML Project (v2)
# 전이학습 가치 중심 재정의

---

## 재설계 원칙

전이학습 실험이 의미를 갖기 위한 3가지 필수 조건:

| 조건 | 질문 |
|---|---|
| **①  필요성** | 타겟 도메인에서 직접 학습하면 왜 안 되는가? |
| **②  도메인 갭** | 소스와 타겟의 분포 차이는 무엇인가? |
| **③  전이 이익** | 어떤 지표로 전이의 효과를 측정할 것인가? |

> CGM 예측 자체가 목적이 아님.  
> **"전이학습이 없으면 특정 집단에 AI를 배포할 수 없다"** 는 상황을 만드는 것이 핵심.

---

## Scenario A — 희귀·소규모 임상 집단 적응 (Clinical Scarcity Adaptation)

### 배경
특수 임상 집단(소아, 고령 중증 저혈당, 임신부)은 **데이터가 구조적으로 희소**하다.
이 집단에게 AI가 가장 절실하지만, 그 집단만으로는 모델을 훈련할 수 없다.

### 세부 전이 경로

| Source (대규모·일반) | Target (소규모·특수) | 도메인 갭 | 임상 필요성 |
|---|---|---|---|
| RT-CGM (448명) + IOBP2 (440명) + GLAM (886명) | **PEDAP** (103명, 소아 T1D) | 성장호르몬·인슐린 감수성 차이, 빠른 혈당 변동 | 소아 저혈당 = 뇌 발달 위협 |
| GLAM + RT-CGM + FLAIR | **SHD** (200명, 고령 중증 저혈당) | 저혈당 무감각, 신장 기능 저하, 서행 반응 | 고령 저혈당 = 사망 위험 ↑ |
| 대규모 T1D pool | **ShanghaiT1DM** (12명) | 아시아 식단, 탄수화물 구성 차이 | 소규모 단일 병원 코호트 현실 |

### 실험 설계 (ML 수준)

```
[Step 1] Source-Only Baseline
  → 소스 전체로 학습, 타겟 test set 평가
  (예: zero-shot 전이)

[Step 2] Target-Only Baseline
  → 타겟 데이터만으로 학습
  (성능의 하한 — 데이터 부족 문제를 직접 보여줌)

[Step 3] TL via TrAdaBoost (ML 수준 전이)
  → 소스 인스턴스에 가중치 재조정 → 타겟 분포에 맞게 학습

[Step 4] Fine-tune (Stacking / Warm-Start)
  → 소스 LightGBM 학습 → 타겟 소량으로 추가 적응
```

### 평가 지표
- **성능 격차 (Performance Gap)**: Source-Only RMSE - Target-Only RMSE
- **전이 이익 (Transfer Gain)**: Source-Only AUROC vs. TL AUROC (저혈당 이벤트 기준)
- **적응 효율 (Adaptation Efficiency)**: 타겟 데이터 N개에 따른 성능 학습 곡선

### TL이 필요한 이유를 수치로 증명
> PEDAP 103명으로만 3-step 예측 모델을 훈련하면 train/val/test split 후  
> 훈련 샘플 수 ≈ 72명 × 평균 256일 = 충분해 보이지만,  
> **저혈당 이벤트 레이블은 극히 희소** → 불균형 비율 > 50:1  
> → TL이 없으면 minority class 예측 불가

---

## Scenario B — 기기 간 교차 전이 (Cross-Device Sensor Transfer)

### 배경
동일 환자가 **센서를 교체**하거나 병원마다 다른 CGM을 사용할 때,
기존 모델은 새 센서의 bias·noise 특성을 모른다.

### 도메인 갭의 정체

| 특성 | Dexcom G6 | FreeStyle Libre |
|---|---|---|
| 샘플링 주기 | 5분 | 15분 |
| MARD | ≈9% | ≈9.4% |
| 간질액 지연 | ~6분 | ~10분 |
| 보정 방식 | Factory calibrated | Factory calibrated |

이 차이는 **피처 분포 자체의 shift** → CORAL 또는 feature normalization으로 접근

### 실험 설계

```
Source: Dexcom 기반 (CGMacros_Dexcom, IOBP2, RT-CGM, FLAIR)
Target: Libre 기반 (CGMacros_Libre, ShanghaiT1DM, ShanghaiT2DM)

비교:
  - No adaptation (source model → target 직접 적용)
  - Z-score 정규화 후 적용
  - CORAL (covariance alignment)
  - Source + Target mixed training
```

### TL 가치
- 센서를 바꿀 때마다 재학습이 필요하다면 임상 배포 불가
- 전이가 성공하면: **장치 독립적(device-agnostic) AI** 주장 가능
- 전이가 실패하면: 센서별 특화 모델의 필요성을 데이터로 증명 → 그것도 기여

---

## Scenario C — 저혈당 위험 예측: 희귀 이벤트 전이 (Rare Event TL)

### 배경
저혈당 이벤트는 **클래스 불균형이 극단적** (발생 비율 ≈ 1–5%).
일반 인구 CGM 시계열로 학습 → 저혈당 다발 집단 예측은 recall ≈ 0.

CGM이 GT가 아니어도 이 시나리오는 유효:
- 비교 기준이 "실제 혈당 < 70"이 아니라 **"다음 CGM 값 < 70"**
- SHD·CITY·WISDM·SENCE는 `DiabHypoEvent` 테이블을 실제 보유 → **임상 이벤트 레이블 검증 가능**

### 실험 설계

```
Task: t+3 (15분 후) CGM < 70 mg/dL 이진 분류

Source: 일반 T1D (저혈당 비율 낮음)
  → GLAM, RT-CGM, IOBP2, PEDAP

Target: 고위험 집단 (저혈당 비율 높음)
  → SHD (Severe Hypo Dataset)
  → WISDM seniors
  → SHD의 DiabHypoEvent로 실제 임상 이벤트 검증

전이 방법 (ML 수준):
  1. TrAdaBoost: 소스 인스턴스 재가중
  2. Cost-sensitive LightGBM: scale_pos_weight 조정
  3. Positive Unlabeled (PU) Learning: 일반 → 고위험 레이블 전이
```

### 핵심 평가 지표
- Sensitivity at 90% Specificity (임상 경보 기준)
- **Lead time**: 저혈당 발생 몇 분 전에 탐지 가능한가
- Target-only vs. TL 모델의 False Alert Rate 비교

---

## Scenario D — 질병 간 전이: T1D → T2D (Cross-Disease Transfer)

### 배경
T1D 코호트는 ML 연구에서 가장 많이 연구됨 (정확한 인슐린 기록, 명확한 집단).
T2D는 훨씬 많은 환자이나 **이질성이 높고 데이터 수집이 어렵다**.

### 도메인 갭의 정체
| 특성 | T1D | T2D |
|---|---|---|
| 혈당 패턴 | 급격한 변동, 명확한 인슐린 반응 | 완만한 변동, 인슐린 저항성 패턴 |
| 공복혈당 | 낮음 (적극 관리) | 높음 (기저 저항성) |
| 식후 혈당 피크 | 더 높고 빠름 | 더 완만하고 오래 지속 |

### 실험 설계

```
Source: T1D 대규모 (GLAM 886 + RT-CGM 448 + IOBP2 440)
Target: T2D (ShanghaiT2DM 100명) + Mixed (CITY 153명)

평가:
  1. T1D 모델 → T2D zero-shot: 기저 성능
  2. T2D only: 성능 하한 (n=100 한계)
  3. TrAdaBoost / Domain-Weighted LightGBM
  4. 혈당 패턴 피처의 공유/비공유 분석 (SHAP)
```

### TL 가치
- "T1D 모델은 T2D에 쓸 수 없다"는 가정을 실험으로 검증
- 어떤 피처가 공유되고 어떤 피처가 질병 특이적인지 → **도메인 갭의 해부학**

---

## Scenario E — 개인화 사전 적응 (N=1 Cold Start)

### 배경
실제 임상에서 신환 환자는 **CGM 착용 3~7일 이내에 AI 보조가 필요**하다.
7일치 데이터로 모델을 훈련하면 test set이 존재하지 않는다.

이것이 ML 수준 전이학습의 **가장 현실적이고 강력한 존재 이유**.

### 실험 설계

```
[Phase 1] Population prior 학습
  → 전체 26개 데이터셋 pool에서 LightGBM 학습
  → "평균적인 T1D 혈당 패턴"을 학습

[Phase 2] 개인화 미세 조정 (ML 수준)
  - Option A: Warm-start (LightGBM n_iter_no_change 활용)
  - Option B: Source 예측을 피처로 추가하여 target 학습
  - Option C: TrAdaBoost로 유사 환자를 source에서 선택

[Phase 3] 적응 효율 평가
  → 타겟 환자 데이터 N일 (1d / 3d / 7d / 14d / 30d)에 따른
     성능 학습 곡선 → "몇 일이면 충분한가?" 답변
```

### 평가 전략
개인별 leave-one-patient-out 교차 검증:
- Patient i를 타겟, 나머지 전체를 소스
- 타겟 환자 데이터 1/3/7/14일 점진 추가
- 목표: **7일 이내에 patient-specific 모델이 population 모델을 능가하는 시점 탐지**

---

## 시나리오 통합 프레임: "전이가 없으면 AI를 못 쓰는 집단"

```
┌─────────────────────────────────────────────────────┐
│         Transfer Learning is NECESSARY when:          │
│                                                       │
│  A. Target has too few samples                        │
│     → Scenario A (pediatric, elderly hypo)           │
│     → Scenario E (new patient, cold start)           │
│                                                       │
│  B. Target has rare events (extreme imbalance)        │
│     → Scenario C (hypoglycemia alert in SHD)         │
│                                                       │
│  C. Target domain is physically different            │
│     → Scenario B (device change: Dexcom→Libre)       │
│     → Scenario D (disease: T1D→T2D)                  │
└─────────────────────────────────────────────────────┘
```

---

## 시나리오 우선순위 (전이학습 가치 기준)

| 시나리오 | TL 필요성 | 도메인 갭 명확성 | 데이터 즉시 사용 가능 | 추천 |
|---|---|---|---|---|
| **A: 소규모 임상 집단** | ★★★ | ★★★ | ✓ (PEDAP, SHD 확보) | **1순위** |
| **C: 저혈당 희귀 이벤트** | ★★★ | ★★★ | ✓ (DiabHypoEvent 보유) | **1순위** |
| **E: N=1 개인화 cold start** | ★★★ | ★★ | ✓ (전체 pool 사용) | **2순위** |
| **D: T1D → T2D** | ★★ | ★★★ | ✓ (ShanghaiT2DM, CITY) | **2순위** |
| **B: 기기 간 전이** | ★★ | ★★ | ✓ (Dexcom/Libre 분리됨) | **3순위** |

---

## 실험 구현 우선순위

```
Week 1-2: Scenario A (저혈당 집단 간 transfer)
  → Source: RT-CGM + IOBP2, Target: SHD / PEDAP
  → 지표: AUROC, Sensitivity@90%Spec
  → 전이 방법: TrAdaBoost (기존 Tier 6 인프라 활용)

Week 3-4: Scenario C (희귀 이벤트 transfer)
  → SHD의 DiabHypoEvent로 임상 레이블 검증
  → 전이 방법: cost-sensitive LightGBM + TrAdaBoost

Week 5-6: Scenario E (개인화 cold start 학습 곡선)
  → Leave-one-patient-out × N일 학습 곡선
  → 전이 방법: warm-start / source-as-feature stacking

Week 7-8: Scenario D (T1D → T2D)
  → SHAP 기반 피처 공유/비공유 분석
```

---

## 공통 평가 프레임워크

모든 시나리오에서 동일 비교 구조:

```
┌──────────────┬────────────────────────────────────┐
│  Baseline    │  설명                               │
├──────────────┼────────────────────────────────────┤
│ Source-Only  │ 소스로만 학습 → 타겟 평가 (zero-shot) │
│ Target-Only  │ 타겟 데이터만 학습 (하한, 희소 상황)  │
│ Oracle       │ 타겟 전체로 학습 (상한, 현실 불가)    │
│ TL model     │ 제안 방법 (TrAdaBoost / warm-start)  │
└──────────────┴────────────────────────────────────┘

목표: Source-Only < TL < Oracle, TL >> Target-Only
```
