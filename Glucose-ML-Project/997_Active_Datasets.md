# Active Datasets — Glucose-ML Project

이 문서는 실험에 실제로 사용되는 데이터셋 목록을 기록한다.
스캔 기준: `003_Glucose-ML-collection` 전체 디렉터리.
마지막 업데이트: 2026-04-22

---

## 요약

| 항목 | 수 |
|---|---|
| 전체 데이터셋 | 33 |
| **실험 사용** | **26** |
| 제외 (파일 없음) | 6 |
| 제외 (설계 부적합) | 1 |

---

## 실험 사용 데이터셋 (26개)

### 기존 데이터셋 (19개)

| # | 데이터셋 | 소스 | 피험자 | 주기 | 그룹 | 비고 |
|---|---|---|---|---|---|---|
| 1 | AIDET1D | extracted-glucose-files | — | 5분 | 5min | |
| 2 | AZT1D | extracted-glucose-files | — | 5분 | 5min | |
| 3 | BIGIDEAs | extracted-glucose-files | — | 5분 | 5min | |
| 4 | Bris-T1D_Open | extracted-glucose-files | — | 15분 | 15min | |
| 5 | CGMacros_Dexcom | extracted-glucose-files | — | 1분 | 1min | 원본 1분 주기 배포 (PhysioNet, 2025). Dexcom 5분 측정값 기반 리샘플링. |
| 6 | CGMacros_Libre | extracted-glucose-files | — | 1분 | 1min | 원본 1분 주기 배포 (PhysioNet, 2025). Libre 15분 측정값 기반 리샘플링. |
| 7 | CGMND | extracted-glucose-files | — | 5분 | 5min | |
| 8 | Colas_2019 | extracted-glucose-files | — | 5분 | 5min | |
| 9 | D1NAMO | extracted-glucose-files | — | 5분 | 5min | |
| 10 | GLAM | extracted-glucose-files | — | 5분 | 5min | |
| 11 | Hall_2018 | extracted-glucose-files | — | 5분 | 5min | |
| 12 | HUPA-UCM | extracted-glucose-files | — | 5분 | 5min | |
| 13 | IOBP2 | extracted-glucose-files | — | 5분 | 5min | |
| 14 | PEDAP | extracted-glucose-files | — | 5분 | 5min | |
| 15 | PhysioCGM | extracted-glucose-files | — | 5분 | 5min | |
| 16 | ShanghaiT1DM | extracted-glucose-files | — | 15분 | 15min | |
| 17 | ShanghaiT2DM | extracted-glucose-files | — | 15분 | 15min | |
| 18 | T1D-UOM | extracted-glucose-files | — | 5분 | 5min | 소수점 패턴은 mmol/L → mg/dL 단위 변환 아티팩트. 보간 아님. |
| 19 | UCHTT1DM | extracted-glucose-files | — | 5분 | 5min | |

### 신규 추가 데이터셋 (7개, DiaData / Jaeb Center)

전처리 스크립트: `015_New_Dataset_Preprocessing/preprocess_new_datasets.py`
원본 출처: [Beyza-Cinar/DiaData](https://github.com/Beyza-Cinar/DiaData), Jaeb Center for Health Research

| # | 데이터셋 | 피험자 수 | 주기 | 그룹 | 타임스탬프 방식 | 비고 |
|---|---|---|---|---|---|---|
| 20 | RT-CGM | 448 | 5분 | 5min | 절대 시각 (익명화, 기준 2000년) | T1D, 다중 기간 파일 분할 |
| 21 | CITY | 153 | 5분 | 5min | SAS 익명화 (`ddMMMyyy:HH:MM:SS`) | T1D/T2D |
| 22 | SENCE | 143 | 5분 | 5min | SAS 익명화 | T1D |
| 23 | WISDM | 203 | 5분 | 5min | SAS 익명화 | T1D, 고령 환자 |
| 24 | FLAIR | 113 | 5분 | 5min | 절대 시각 | T1D, Unusuable 플래그 필터링 적용 |
| 25 | SHD | 200 | 5분 | 5min | 입원 기준 상대 일수+시각 → 재구성 | T1D 중증 저혈당 코호트 |
| 26 | ReplaceBG | 226 | 5분 | 5min | 입원 기준 상대 일수+시각 → 재구성 | T1D, RecordType==CGM 필터링 |

---

## 그룹별 요약

| 그룹 | 데이터셋 수 | 신규 추가 | 예측 시간 (3스텝) |
|---|---|---|---|
| 1min | 2 | 0 | 3분 뒤 |
| 5min | 21 | +7 | 15분 뒤 |
| 15min | 3 | 0 | 45분 뒤 |

---

## 타임스탬프 유형별 분류

| 유형 | 데이터셋 | 처리 방식 |
|---|---|---|
| 절대 시각 (실제) | FLAIR 등 대부분 | 그대로 사용 |
| 절대 시각 (익명화, 2000년 기준) | RT-CGM, CITY, SENCE, WISDM | 그대로 사용 (연속성 보장됨) |
| 상대 일수 + 시각 재구성 | SHD, ReplaceBG | `datetime(2000,1,1) + days + time` 으로 복원 |
| 절대 시각 아님 (식사 기준) | Park_2025 | **제외** |

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
| Park_2025 | `timestamp` 컬럼이 식사 기준 상대 시간이며 절대 시각이 아님. 연속 CGM 예측 태스크에 부적합. | `global_config.py` `EXCLUDED_DATASETS`, Rule 9 |

---

## 데이터 소스 원칙

모든 사용 데이터셋은 `extracted-glucose-files` 폴더를 소스로 사용한다.
`time-augmented`, `extended-features` 폴더는 피처 보강 목적으로만 접근하며, 혈당 시계열 로드에는 사용하지 않는다.

근거: `AGENTS.md` Single Source of Truth 원칙, `global_config.py` `GLUCOSE_SUBFOLDER_PRIORITY`.
