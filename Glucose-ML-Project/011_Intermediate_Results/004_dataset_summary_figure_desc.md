# Figure Description — 004_dataset_summary_figure.png
## Dataset Characteristics — 12 CGM Datasets Used in Analysis

> **파일:** `004_dataset_summary_figure.png`  
> **생성 스크립트:** `plot_dataset_summary.py`  
> **데이터 출처:** `dataset_summary_stats.csv`  
> **대응 표:** `003_dataset_summary_table.md` (Table 1)  
> **참조 논문:** Prioleau et al. (2025), *npj Digital Medicine* — Table 1 / Figure 2 스타일

---

## 그림 개요

12개 CGM 데이터셋의 핵심 통계량을 **8개 서브플롯 (A~H)** 으로 시각화한 종합 대시보드.  
각 막대의 색상은 **환자 코호트(Patient Cohort)** 를 나타낸다:

| 색상 | 코호트 | 데이터셋 |
|:---|:---|:---|
| 🔴 빨강/코랄 | T1DM (제1형 당뇨) | IOBP2, PEDAP, AIDET1D, Bris-T1D, UCHTT1DM |
| 🟠 주황 | T1DM/ND (혼합) | HUPA-UCM |
| 🟣 보라 | GDM (임신성 당뇨) | GLAM |
| 🔵 파랑 | ND/PreD (전당뇨) | BIGIDEAs |
| 🟢 초록 | ND (비당뇨) | CGMND, Park_2025, CGMacros_Dexcom, CGMacros_Libre |

---

## 서브플롯별 상세 설명

### (A) Number of Subjects — 피험자 수

- **Y축:** 피험자 수 (로그 스케일)
- **표시:** 각 막대 위에 실제 피험자 수 라벨
- **핵심 관찰:**
  - **GLAM이 886명으로 최대** — GDM 코호트로 대규모 전향적 연구
  - **IOBP2 (440명)**, **PEDAP (103명)** 은 T1DM 중 대규모
  - **BIGIDEAs (16명)**, **Park_2025 (38명)** 등 소규모 데이터셋 다수
- **시사점:** 데이터셋 간 피험자 수가 16~886명으로 약 55배 차이. 전이학습 시 규모 균등화(subsampling) 필요

### (B) Total CGM Readings — 총 CGM 측정포인트 수

- **Y축:** 총 reading 수 (백만 단위, 로그 스케일)
- **핵심 관찰:**
  - **GLAM (~26.6M)**, **IOBP2 (~14.3M)**, **PEDAP (~7.2M)** 이 3대 대규모 데이터셋
  - **UCHTT1DM (~29K)**, **Park_2025 (~24K)** 는 극소규모
  - 최대/최소 비율 약 **1000:1**
- **시사점:** 학습 데이터 불균형이 극심. Pooling 시 대규모 데이터셋이 모델을 지배할 가능성

### (C) Prediction Windows (30-min) — 예측 윈도우 수

- **Y축:** 30분 예측 윈도우 수 (백만 단위, 로그 스케일)
- **설명:** 각 윈도우는 Lookback 6 steps + Forecast 6 steps (= 30분 간격 예측 단위)
- **핵심 관찰:**
  - (B)의 패턴과 거의 동일 — 윈도우 수 ≈ reading 수 (sliding window이므로)
  - GLAM(26.2M), IOBP2(14.0M), PEDAP(7.1M) 순
- **시사점:** 이 값이 실제 모델 학습에 투입되는 데이터 포인트 수

### (D) Mean Glucose ± SD — 평균 혈당 및 표준편차

- **Y축:** 혈당값 (mg/dL)
- **막대:** 평균 혈당, **에러바:** ±1 SD
- **참조선:**
  - 🔴 빨간 점선: TAR 임계값 (180 mg/dL)
  - 🟢 초록 점선: TBR 임계값 (70 mg/dL)
  - 🟡 노란 점선: 식후 참조값 (140 mg/dL)
- **핵심 관찰:**
  - **T1DM 데이터셋 (빨강):** 평균 ~150~170 mg/dL, SD 크다 (혈당 변동성 높음)
  - **ND 데이터셋 (초록):** 평균 ~100~115 mg/dL, SD 작다 (안정적)
  - **GDM (GLAM, 보라):** 평균 ~100 mg/dL, SD 가장 작음
  - **CGMacros_Dexcom은 ND인데 평균 141.5 mg/dL** — 다른 ND 대비 이상치적으로 높음
- **시사점:** 코호트 간 평균 혈당 차이가 ~70 mg/dL (100 vs 170). 전이학습 시 도메인 시프트의 핵심 원인

### (E) Glucose Range Distribution (%) — 혈당 범위 분포

- **Y축:** 비율 (%, 누적 100%)
- **3가지 영역:**
  - 🔴 **TAR (>180 mg/dL):** 고혈당 비율
  - 🟢 **TIR (70~180 mg/dL):** 정상 범위 비율
  - 🔵 **TBR (<70 mg/dL):** 저혈당 비율
- **핵심 관찰:**
  - **IOBP2:** TAR 37.1% — T1DM 중 고혈당 비율 최고
  - **GLAM, BIGIDEAs, CGMND:** TIR 94~98% — 거의 전체가 정상 범위
  - **UCHTT1DM:** TBR 5.9% — 저혈당 비율 최고 (6.8일 단기 관찰)
- **시사점:** T1DM은 TAR 20~37%로 고혈당 예측이 핵심 과제, ND는 TIR 90%+ 로 변동 예측이 미미

### (F) Glucose Distribution (IQR ± 1.5SD) — 혈당 분포 박스플롯

- **Y축:** 혈당값 (mg/dL)
- **형태:** 박스(IQR) + 위스커(±1.5×IQR) + 중앙값 라인
- **참조선:** 70, 140, 180 mg/dL
- **핵심 관찰:**
  - **T1DM 데이터셋:** 박스가 넓고 위스커가 길게 확장 → 혈당 분포 이질적
  - **ND/GDM 데이터셋:** 박스가 좁고 70~180 범위 내에 집중
  - **IOBP2:** 위스커 상단이 ~300 mg/dL까지 도달 — 극단적 고혈당 사례 포함
- **시사점:** (D)의 평균±SD를 보완하는 분포 형태 정보. Outlier의 실제 범위를 확인 가능

### (G) Data Collection Duration — 데이터 수집 기간

- **Y축:** 일(days) per subject
- **두 막대:** 빨간색 = Mean(평균), 연한색 = Max(최대)
- **핵심 관찰:**
  - **IOBP2:** 최대 ~1952일 (5.3년) — 가장 긴 종단 관찰
  - **HUPA-UCM:** 최대 ~574일
  - **UCHTT1DM:** 6.8일 — 가장 짧은 관찰 기간 (임상 시험)
  - **BIGIDEAs:** 9.9일
- **시사점:** 종단 관찰 기간이 7일~5년으로 약 280배 차이. 계절 패턴/장기 변동 포착 능력이 데이터셋마다 크게 다름

### (H) CGM Sampling Interval — CGM 센서 샘플링 간격

- **Y축:** 분(minutes) — 5분 또는 15분
- **색상:** 초록 = 5분, 주황 = 15분
- **핵심 관찰:**
  - **10개 데이터셋:** 5분 간격 (Dexcom 계열)
  - **GLAM, CGMacros_Libre:** 15분 간격 (FreeStyle Libre)
- **시사점:** 5→15분은 시간 해상도 3배 차이. Harmonization 시 15분 데이터셋은 보간(interpolation) 또는 window size 조정 필요

---

## 그림 전체 시사점 요약

1. **규모 이질성:** 피험자 수(16~886), 총 reading 수(24K~26.6M), 수집 기간(7일~5년) 모두 극심한 편차
2. **코호트 이질성:** T1DM 평균 ~160 mg/dL vs ND 평균 ~110 mg/dL — 도메인 시프트의 주요 원인
3. **센서 이질성:** 5분(Dexcom) vs 15분(Libre) — 시간 해상도 불일치
4. **이 그림은 Table 1 (`003_dataset_summary_table.md`)의 시각화 버전**으로, 논문의 Figure 2에 대응

---

*Glucose-ML-Project · 011_Intermediate_Results*
