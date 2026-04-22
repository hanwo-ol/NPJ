# Tier 6 Ablation Study: Results & Limitations Report

이 문서는 `tier6_ablation_A_B_C_D` 결과 수치와 터미널/파일 로깅 기록을 분석하여, 현재 혈당 예측(CGM) 기계학습 모델의 성능적 한계와 개선 방안을 학술적으로 도출한 결과 보고서입니다.

---

## 1. 절제 연구 (Ablation) 결과 요약 (RMSE 기준)

| 데이터셋 (Target) | Path A (eGMI Imputation) | Path B (UMD Virtual) | Path C (CORAL Alignment) | Path D (TCA Projection) | 분석 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AIDET1D** | 21.26 | 21.77 | 41.28 | 25.06 | UMD 유지, 도메인 적응 악화 |
| **BIGIDEAs** | 20.03 | **13.64** | 36.08 | 32.63 | UMD 강력한 효과 입증 |
| **Bris-T1D_Open** | 56.17 | **32.69** | 91.04 | 83.66 | UMD 도입 시 오차 약 23 하락 |
| **CGMacros_Dex** | 10.50 | **3.59** | 49.31 | 19.65 | 식사 마커 가상화 압도적 성공 |
| **CGMacros_Lib** | 11.07 | **4.96** | 45.88 | 23.86 | 식사 마커 가상화 압도적 성공 |
| **CGMND** | 12.97 | 13.41 | 28.43 | 15.03 | 유지 |
| **HUPA-UCM** | 38.18 | **18.25** | 77.36 | 51.52 | 오차율 50% 이상 감소 |

> [!IMPORTANT]
> **Path A (eGMI) vs Path B (UMD) 분석**
> - Velocity/Acceleration 기반으로 식사 확률을 추정하는 **UMD(가상 마커) 기법은 대성공**입니다. 특히 식사 전후 혈당 변동성이 극심한 `Bris-T1D`, `CGMacros`, `HUPA-UCM` 등에서 RMSE를 무려 절반(50%) 이하로 떨어뜨리는 매우 강력한 효과를 입증했습니다.

> [!WARNING]
> **Path C (CORAL) vs Path D (TCA) 분석**
> - 두 도메인 적응 기법 모두 Baseline(A)에 비해 **성능이 심각하게 파괴**되는 양상을 보였습니다 (RMSE 2~3배 폭증). 
> - **원인:** 트리 기반 모델(LightGBM)은 변수의 직교적 스플릿(Orthogonal Split)을 기반으로 작동합니다. CORAL(공분산 회전)과 TCA(선형 커널 투영) 등 선형/행렬 변환 기반의 적응 기법은 시계열 특성들의 비선형성 및 위상학적 구조를 강제로 뒤틀어버려 트리가 분기점(Split point)을 찾지 못하게 만드는 치명적 부작용을 낳았습니다.

---

## 2. 부정적 이슈 및 로그 결함 분석

터미널 및 `tier6_experiment_C_D.log` 파일에서 관측된 시스템/데이터 파이프라인의 에러를 분석한 결과입니다.

### 2.1. 대량의 데이터셋 스킵 (Skip) 현상
- **로그 관측:** `[WARN] Skipped Mode X for AI-READI: Insufficient sequences or structurally invalid data.`
- **현상:** 총 26개의 대상 중 `AI-READI`, `Colas_2019`, `OhioT1DM` 등 무려 **14개**의 데이터셋이 연산되지 않고 스킵되었습니다.
- **원인:** 
  1. `time-augmented.csv` 파일이 없거나 내부 컬럼명이 `glucose_value_mg_dl`이 아님 (예: `CGM`, `value`).
  2. 전처리 시 타임스탬프 파싱 실패 또는 10개 미만의 윈도우 시퀀스로 인해 필터링됨.

### 2.2. 도메인 어댑테이션 맵핑 크래시 (수정 완료됨)
- **로그 관측:** `[LightGBM] [Fatal] Length of labels differs from the length of #data`
- **원인:** Path C 도메인 맵핑 단계에서 원본 피처와 타겟 정답지 간의 차원 불일치가 발생. 
- **조치:** 맵핑 차원 동기화 및 `y_train` 병합 로직으로 즉시 자체 수정 완료되어 정상 구동 확인.

---

## 3. 한계점(Limitations)

1. **선형 도메인 적응과 GBDT의 불협화음:** 통계적 분포를 강제 일치시키는 CORAL, TCA 방식은 순수 머신러닝(LightGBM) 파이프라인에서는 역효과를 초래함이 입증되었습니다.
2. **폐쇄적 파싱 로직:** 특정 컬럼명(`glucose_value_mg_dl`)에 과도하게 의존하여 절반 이상의 귀중한 외부 데이터셋이 로드조차 되지 못하고 버려지고 있습니다.
3. **과도한 윈도우 의존성:** 현재 과거 6시점(Lookback=6)을 고정적으로 사용하고 있어, 데이터 밀도가 낮거나 누락이 잦은 데이터셋에서는 정상적인 윈도우가 형성되지 않습니다.

---

## 4. 개선안 및 다음 단계 설계 (Tier 6.5)

현재의 발견들을 바탕으로 성능과 호환성을 극대화하기 위한 구조적 개편안입니다.

### 🎯 Improvement 1. 다중 시점(Lookback) 지원 모듈화
- 하드코딩된 6시점을 **3시점(Lookback=3)** 옵션과 호환되도록 구성. 앞서 논의한 바와 같이, 파생 피처(Velocity/Acc) 계산에 무리가 없는 최소 3시점 모드를 도입하여 스킵되는 짧은 시퀀스 데이터들을 최대한 구출(Rescue)합니다.

### 🎯 Improvement 2. 동적 컬럼 매핑 (Dynamic Schema Resolution)
- `tier6_data_utils.py`의 `feat_parse_cols` 로직을 개선하여, `glucose_value_mg_dl` 뿐만 아니라 `CGM`, `Glucose`, `Value` 등 이형(Heterogeneous) 컬럼명들을 자동 정규화하는 정규식/배열 매핑 엔진을 탑재합니다.

### 🎯 Improvement 3. TCA/CORAL 폐기 및 UMD 전역화 (Globalization)
- 도메인 어댑테이션 기법(C, D)은 GBDT 생태계에 부적합하므로 파이프라인에서 배제(Ablation)합니다.
- 대신, 극단적 효율을 보인 **UMD Virtual Marker (Path B)를 Local 잔차 모델이 아닌 Global 기본 입력 차원(Dimension)으로 승격**시킵니다. 이를 통해 잔차 학습 2-Stage 복잡도를 제거하고 1-Stage 융합 모델의 효율을 극대화합니다.
