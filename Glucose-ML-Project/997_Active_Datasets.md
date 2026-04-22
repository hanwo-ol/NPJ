# Active Datasets — Glucose-ML Project

이 문서는 실험에 실제로 사용되는 데이터셋 목록을 기록한다.

마지막 업데이트: 2026-04-22

---

## 데이터 소재 원칙

모든 실험 데이터셋은 전처리 완료 후 `003_Glucose-ML-collection` 에 적재되며, 이 디렉터리를 단일 진실 원천(SSoT)으로 사용한다.

신규 데이터셋 7개 (DiaData / Jaeb Center)의 원본 파일은 `../new-data/` 에 보관된다. 이 디렉터리는 버전 관리 대상에서 제외되며, 전처리 스크립트(`015_New_Dataset_Preprocessing/preprocess_new_datasets.py`)를 통해 `003_Glucose-ML-collection` 에 반입된다.

---

## 요약

| 항목 | 수 |
|---|---|
| 전체 실험 대상 | 26 |
| 기존 데이터셋 | 19 |
| 신규 추가 (DiaData / Jaeb) | 7 |
| 제외 (파일 없음) | 6 |
| 제외 (설계 부적합) | 1 |

---

## 실험 사용 데이터셋 (26개)

| # | 데이터셋 | 피험자 수 | 주기 | 그룹 | 질병 유형 | 출처 | 타임스탬프 유형 | 비고 |
|---|---|---|---|---|---|---|---|---|
| 1 | AIDET1D | 29 | 5분 | 5min | T1D | 기존 | 절대 시각 | |
| 2 | AZT1D | 25 | 5분 | 5min | T1D | 기존 | 절대 시각 | |
| 3 | BIGIDEAs | 16 | 5분 | 5min | ND | 기존 | 절대 시각 | |
| 4 | Bris-T1D_Open | 20 | 15분 | 15min | T1D | 기존 | 절대 시각 | |
| 5 | CGMacros_Dexcom | 45 | 1분 | 1min | Mixed | 기존 | 절대 시각 | 원본 Dexcom 5분 측정값 기반 1분 리샘플링 |
| 6 | CGMacros_Libre | 45 | 1분 | 1min | Mixed | 기존 | 절대 시각 | 원본 Libre 15분 측정값 기반 1분 리샘플링 |
| 7 | CGMND | 45 | 5분 | 5min | ND | 기존 | 절대 시각 | |
| 8 | Colas_2019 | 208 | 5분 | 5min | Mixed | 기존 | 절대 시각 | 건강인에서 T2D 발병 위험군으로 진행한 종단 연구 |
| 9 | D1NAMO | 9 | 5분 | 5min | T1D | 기존 | 절대 시각 | |
| 10 | GLAM | 886 | 5분 | 5min | ND | 기존 | 절대 시각 | Glucose Levels Across Maternity. 건강한 임신부 코호트. HbA1c < 6.5% |
| 11 | Hall_2018 | 57 | 5분 | 5min | Mixed | 기존 | 절대 시각 | 정상혈당 + 전당뇨 + T2D 혼재 코호트 |
| 12 | HUPA-UCM | 25 | 5분 | 5min | T1D | 기존 | 절대 시각 | G+I+A+S 멀티모달 (심박수, 걸음수, 인슐린, 칼로리) |
| 13 | IOBP2 | 440 | 5분 | 5min | T1D | 기존 | 절대 시각 | iLet 바이오닉 췌장 임상시험 (Jaeb) |
| 14 | PEDAP | 103 | 5분 | 5min | T1D | 기존 | 절대 시각 | 소아 T1D |
| 15 | PhysioCGM | 9 | 5분 | 5min | T1D | 기존 | 절대 시각 | ECG/PPG/EDA/가속도계 포함 멀티모달 |
| 16 | ShanghaiT1DM | 12 | 15분 | 15min | T1D | 기존 | 절대 시각 | |
| 17 | ShanghaiT2DM | 100 | 15분 | 15min | T2D | 기존 | 절대 시각 | |
| 18 | T1D-UOM | 17 | 5분 | 5min | T1D | 기존 | 절대 시각 | 소수점 패턴은 mmol/L→mg/dL 변환 아티팩트. 보간 아님. |
| 19 | UCHTT1DM | 20 | 5분 | 5min | T1D | 기존 | 절대 시각 | |
| 20 | RT-CGM | 448 | 5분 | 5min | T1D | 신규 (DiaData) | 절대 시각 (익명화, 2000년 기준) | 다중 기간 파일 분할. 다인종 포함. |
| 21 | CITY | 153 | 5분 | 5min | Mixed | 신규 (DiaData) | 절대 시각 (SAS 익명화) | T1D/T2D/ND 혼재 |
| 22 | SENCE | 143 | 5분 | 5min | T1D | 신규 (DiaData) | 절대 시각 (SAS 익명화) | |
| 23 | WISDM | 203 | 5분 | 5min | T1D | 신규 (DiaData) | 절대 시각 (SAS 익명화) | 고령 T1D 코호트 |
| 24 | FLAIR | 113 | 5분 | 5min | T1D | 신규 (DiaData) | 절대 시각 | Unusable 플래그 필터링 적용 |
| 25 | SHD | 200 | 5분 | 5min | T1D | 신규 (DiaData) | 재구성 (2000-01-01 기준) | 중증 저혈당 코호트. 입원 기준 상대 일수+시각 재구성. |
| 26 | ReplaceBG | 226 | 5분 | 5min | T1D | 신규 (DiaData) | 재구성 (2000-01-01 기준) | RecordType==CGM 필터링. BGM 동시 보유 (HDeviceBGM). |

---

## 그룹별 요약

| 그룹 | 데이터셋 수 | 예측 시간 (3스텝) | 주요 데이터셋 |
|---|---|---|---|
| 1min | 2 | 3분 뒤 | CGMacros_Dexcom, CGMacros_Libre |
| 5min | 21 | 15분 뒤 | RT-CGM, IOBP2, GLAM, WISDM 등 |
| 15min | 3 | 45분 뒤 | ShanghaiT1DM, ShanghaiT2DM, Bris-T1D_Open |

---

## 질병 유형별 요약

| 질병 유형 | 데이터셋 수 | 주요 데이터셋 |
|---|---|---|
| T1D | 17 | RT-CGM, IOBP2, FLAIR, SENCE, WISDM, PEDAP, SHD, ReplaceBG 등 |
| T2D | 1 | ShanghaiT2DM |
| ND | 3 | BIGIDEAs, CGMND, GLAM |
| Mixed | 5 | CGMacros_Dexcom, CGMacros_Libre, CITY, Colas_2019, Hall_2018 |

---

## 제외 데이터셋 (7개)

### 파일 없음 (6개)

`extracted-glucose-files` 및 기타 소스 폴더가 존재하지 않는 데이터셋.

| 데이터셋 | 비고 |
|---|---|
| AI-READI | 파일 없음 |
| DiaTrend | 파일 없음 |
| OhioT1DM | 파일 없음 |
| T1DEXI | 파일 없음 |
| T1DEXIP | 파일 없음 |
| T1DiabetesGranada | 파일 없음 |

### 설계 부적합 제외 (1개)

| 데이터셋 | 제외 사유 | 근거 |
|---|---|---|
| Park_2025 | `timestamp`가 식사 기준 상대 시간이며 재구성 불가. 연속 CGM 예측 태스크에 부적합. | `global_config.py` `EXCLUDED_DATASETS`, 999_Preprocessing_Rules.md Rule 9 |

---

## 데이터 소스 우선순위

모든 사용 데이터셋은 `extracted-glucose-files` 폴더를 혈당 시계열 소스로 사용한다.
`time-augmented`, `extended-features` 폴더는 피처 보강 목적으로만 접근하며, 혈당 시계열 로드에는 사용하지 않는다.

근거: `AGENTS.md` Single Source of Truth 원칙, `global_config.py` `GLUCOSE_SUBFOLDER_PRIORITY`.
