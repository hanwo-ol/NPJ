# Figure Description — 005_dataset_datatype_figure.png
## Data Type Availability Across 12 CGM Datasets

> **파일:** `005_dataset_datatype_figure.png`  
> **생성 스크립트:** `plot_datatype_availability.py`  
> **대응 표:** `002_dataset_data_type_availability.md`  
> **참조 논문:** Prioleau et al. (2025), *npj Digital Medicine* — Figure 1 스타일

---

## 그림 개요

12개 CGM 데이터셋에 포함된 **데이터 유형(G/I/A/S/Q/M) × 세부 변수**의 가용성을 히트맵 형태로 시각화한 그림.  
Prioleau et al. (2025)의 Figure 1 분류 체계를 적용하여, 각 셀이 **3가지 상태** 중 하나를 나타낸다.

---

## 읽는 법

### 셀 상태 (3단계)

| 색상/기호 | 의미 | 설명 |
|:---:|:---|:---|
| 🟩 초록 + `V` | **Used in model** | harmonized 데이터에 포함, 실제 모델 학습에 투입됨 |
| 🟧 주황 + `O` | **In original dataset** | 원본 데이터에 존재하지만, harmonization 과정에서 제외됨 (미사용) |
| ⬛ 어두운 배경 | **Not in dataset** | 해당 데이터셋 원본 자체에 해당 정보가 없음 |

### 축 구조

- **X축 (가로, 하단):** 12개 데이터셋명. 각 열 상단에 **코호트 라벨** (T1DM / ND / GDM / T1DM-ND / ND-PreD) 이 색상 배지로 표시
- **Y축 (세로, 왼쪽):** 세부 변수명 (25개), 6대 카테고리로 그룹핑

### 코호트 색상 (상단 배지 및 우측 범례)

| 색상 | 코호트 |
|:---|:---|
| 🔴 코랄/살몬 | T1DM — Type 1 Diabetes |
| 🟠 주황 | T1DM/ND — Mixed cohort |
| 🟣 보라 | GDM — Gestational DM |
| 🔵 파랑 | ND/PreD — Pre-diabetes |
| 🟢 초록 | ND — Non-diabetic |

---

## 6대 데이터 카테고리 상세 설명

### G — CGM (연속혈당측정)

| 변수 | 가용 데이터셋 | 관찰 |
|:---|:---|:---|
| **CGM glucose value** | ✅ 12/12 전체 | 프로젝트의 핵심 입력. 모든 데이터셋이 100% 보유 |
| **2nd CGM sensor** | ✅ CGMacros 2개만 | Dexcom↔Libre 교차센서. 이 2개 데이터셋에서만 이중 센서 비교 가능 |

### I — Insulin (인슐린 전달 시스템)

| 변수 | 가용 데이터셋 | 관찰 |
|:---|:---|:---|
| **Basal insulin rate** | ✅ Bris-T1D, HUPA-UCM, PEDAP (3개) | T1DM 인슐린 펌프/주사 데이터셋에 한정 |
| **Bolus insulin dose** | ✅ Bris-T1D, HUPA-UCM, IOBP2, PEDAP, UCHTT1DM (5개) | T1DM 전반에 가장 널리 보유 |
| **Carb input for bolus** | ✅ Bris-T1D, HUPA-UCM, IOBP2, PEDAP (4개) | 볼루스 계산 시 입력한 탄수화물량 |
| **Insulin regimen** | ✅ AIDET1D (1개) | 환자별 MDI/Pump 방식 정보 (Q와 I의 경계) |

> **핵심:** 인슐린 데이터는 **T1DM 중심 6개 데이터셋에만 존재**. GDM(GLAM)과 ND 데이터셋은 인슐린 치료를 하지 않으므로 본질적으로 미보유.

### A — Activity (활동 추적)

| 변수 | 가용 데이터셋 | 관찰 |
|:---|:---|:---|
| **Heart rate (HR)** | ✅ Bris-T1D, CGMacros×2, HUPA-UCM (4개) | 웨어러블 착용 데이터셋에 한정 |
| **Step count** | ✅ 동일 4개 | HR과 동일 분포 |
| **Distance walked** | ✅ Bris-T1D (1개) | 가장 풍부한 활동 데이터셋 |
| **Active calories** | ✅ Bris-T1D, CGMacros×2, HUPA-UCM (4개) | — |
| **Sleep log** | ✅ CGMND (1개) | 수면-각성 시간 자가보고 |

> **핵심:** 활동 데이터의 커버리지 비대칭 — 4개 데이터셋에 집중. 나머지 8개는 활동 정보 전무.

### S — Self-report (자가보고 / 식이)

| 변수 | 가용 데이터셋 | 관찰 |
|:---|:---|:---|
| **Meal event marker** | ✅ GLAM (1개) | 식사 발생 여부만 기록 (영양소 정량 없음) |
| **Food calories** | ✅ BIGIDEAs, CGMacros×2 (3개) | 정량적 식이 추적 |
| **Carbohydrate intake** | ✅ BIGIDEAs, CGMacros×2, Park_2025 (4개) | 탄수화물은 혈당 반응의 직접 원인 |
| **Protein intake** | ✅ BIGIDEAs, CGMacros×2 (3개) | — |
| **Fat intake** | ✅ BIGIDEAs, CGMacros×2 (3개) | — |
| **Dietary fiber** | ✅ CGMacros×2 (2개) | BIGIDEAs는 원본에 존재하나 미사용(🔶) |
| **Meal photo** | 🔶 CGMacros×2 (2개) | 이미지 파일 존재하나 수치형 아님 → 미사용 |

> **핵심:** 정량 식이 데이터(칼로리, 탄수화물, 단백질, 지방)는 **BIGIDEAs + CGMacros 3개 데이터셋에 집중**. GLAM은 식사 이벤트 마커만 보유.

### Q — Questionnaire (설문 / 인구통계)

| 변수 | 가용 데이터셋 | 관찰 |
|:---|:---|:---|
| **Demographics (age, sex, race)** | 🔶 12/12 전체 | **모든 데이터셋에 원본 존재하나, 현재 harmonized 파이프라인에 미포함** |
| **Diabetes duration** | 🔶 8/12 | T1DM 데이터셋 + 일부 ND 보유 |
| **Insulin delivery method** | ✅ AIDET1D (1개) | MDI vs Pump 정보 |
| **Lifestyle survey** | 🔶 4/12 | BIGIDEAs, CGMacros×2, GLAM |

> **핵심:** Q 카테고리는 **히트맵에서 대부분 주황(🔶)**. 원본에는 존재하지만 시계열 harmonization에서 제외. **Tier 4에서 정적 피처(static feature)로 투입할 핵심 후보.**

### M — Medical Record (임상검사)

| 변수 | 가용 데이터셋 | 관찰 |
|:---|:---|:---|
| **HbA1c** | 🔶 7/12 | BIGIDEAs, CGMacros×2, GLAM, IOBP2, PEDAP, UCHTT1DM |
| **OGTT / Fasting glucose** | 🔶 2/12 | CGMND, GLAM |
| **Cholesterol / Triglycerides** | 🔶 1/12 | GLAM만 |
| **BMI / Weight** | 🔶 7/12 | HbA1c와 유사한 분포 |

> **핵심:** M 카테고리는 **히트맵에서 전체가 주황(🔶) 또는 어두운 배경**. 현재 모델에 단 하나도 투입되지 않았지만, HbA1c와 BMI는 7/12 데이터셋에 존재 → Tier 4 Static Feature 후보.

---

## 그림 전체에서 드러나는 패턴

### 1. 대각선적 희소성 (Diagonal Sparsity)

각 데이터 카테고리(I/A/S)가 특정 데이터셋 그룹에만 집중되어 있어, 히트맵 전체에 녹색 셀이 **흩어져** 분포한다. 12개 데이터셋 전체가 공유하는 변수는 **오직 CGM glucose value 1개뿐**.

### 2. 코호트-피처 상관

- **T1DM 데이터셋:** I(인슐린) 행에 녹색 집중
- **ND/GDM 데이터셋:** S(식이) 행에 녹색 집중
- **웨어러블 착용 데이터셋:** A(활동) 행에 녹색 집중

→ 코호트 유형이 보유 피처 종류를 결정하는 구조적 의존성 존재.

### 3. Q/M 밴드 — 미활용 자원의 시각화

히트맵 하단의 Q/M 영역이 **일관된 주황(🔶) 밴드**를 형성. 이는:
- 원본 데이터에는 풍부한 임상/인구통계 정보가 있으나
- harmonization 파이프라인에서 체계적으로 제외되었음을 시각적으로 보여준다
- **Tier 4의 Static Feature 확장**으로 이 밴드를 녹색(✅)으로 전환하는 것이 핵심 기회

### 4. 데이터셋 풍부도 랭킹

| 순위 | 데이터셋 | 초록(✅) 셀 수 | 보유 카테고리 |
|:---:|:---|:---:|:---|
| 1 | **CGMacros (Dexcom/Libre)** | ~14 | G + A + S (+ Q/M 🔶 다수) |
| 2 | **Bris-T1D** | ~10 | G + I + A |
| 3 | **HUPA-UCM** | ~9 | G + I + A |
| 4 | **BIGIDEAs** | ~7 | G + S |
| 5 | **PEDAP** | ~5 | G + I |
| ... | AIDET1D, CGMND, Park_2025 | 1~3 | G 중심 |

---

## Tier 4 전이학습과의 관계

이 그림은 `tier4/method1.md`의 **§2.3 피처 이질성 현황** 표의 시각적 근거이다:

- **Global features (공통):** 히트맵 최상단의 `CGM glucose value` 행 = 전체 녹색(✅) → 이로부터 파생된 20-dim 공통 시계열 피처가 global model의 입력
- **Local features (고유):** 녹색이 일부 열에만 존재하는 I/A/S 행 → 각 데이터셋의 specialized model에만 투입
- **Q/M 정적 피처:** 주황 밴드 → Quasi-Global 승격 또는 Pure Local로 활용

---

*Glucose-ML-Project · 011_Intermediate_Results*
