# 변수 정의서 (Feature Dictionary)
## Glucose-ML-Project — Tier 1 ~ Tier 2.5_v2

---

## [구조 안내]

본 문서는 각 Tier에서 모델에 투입된 변수(Feature)를 아래 3계층으로 분류합니다.

| 계층 | 설명 |
|---|---|
| **Raw** | 원본 데이터셋에서 직접 추출한 원시 변수 |
| **Engineered** | 원시 변수로부터 수식에 따라 새롭게 파생된 변수 |
| **Target (Y)** | 모델이 예측하는 목적 변수 |

**공통 윈도우 구조:** 모든 Tier에서 Lookback 6 step + Forecast 6 step (= 30분 예측 horizon, 5분 간격 기준)

---

## Part A. 공통 원본 변수 (Raw Features by Dataset)

각 데이터셋에서 모델에 투입되는 원시(Raw) 공변량 목록입니다. 데이터셋마다 가용한 공변량이 다르며, 누락된 공변량 컬럼은 `0.0`으로 채워집니다.

| 데이터셋 | 공변량(Covariates) 컬럼명 | 임상적 의미 |
|---|---|---|
| **AIDET1D** | `glucose_value_mg_dl` | CGM 혈당값 (mg/dL) |
| **BIGIDEAs** | `glucose_value_mg_dl`, `calorie`, `carbs`, `protein`, `fat` | CGM 혈당값 + 음식 영양소별 섭취량 |
| **Bris-T1D_Open** | `glucose_value_mg_dl`, `basal`, `bolus`, `carbs`, `hr`, `steps`, `slp_score` | CGM 혈당값, 기저/볼루스 인슐린, 탄수화물, 심박수, 보행수, 수면점수 |
| **CGMacros_Dexcom** | `glucose_value_mg_dl`, `Libre GL`, `calories`, `protein`, `fat`, `carbs`, `fiber`, `sugars`, `sodium`, `potassium`, `vitamin_c`, `vitamin_d`, `calcium`, `iron`, `heart_rate`, `steps` | CGM(Dexcom) + Libre 교차 + 15종 영양소 + 활동 지표 |
| **CGMacros_Libre** | `glucose_value_mg_dl`, `Dexcom GL`, `calories`, `protein`, `fat`, `carbs`, `fiber`, `sugars`, `sodium`, `potassium`, `vitamin_c`, `vitamin_d`, `calcium`, `iron`, `heart_rate`, `steps` | CGM(Libre) + Dexcom 교차 + 15종 영양소 + 활동 지표 |
| **CGMND** | `glucose_value_mg_dl` | CGM 혈당값만 (정상인 코호트, 공변량 없음) |
| **GLAM** | `glucose_value_mg_dl`, `event_marker` | CGM 혈당값 + 식사 이벤트 마커(0/1) |
| **HUPA-UCM** | `glucose_value_mg_dl`, `basal`, `bolus`, `carbs`, `hr`, `steps` | CGM 혈당값, 인슐린(기저/볼루스), 탄수화물, 심박수, 보행수 |
| **IOBP2** | `glucose_value_mg_dl`, `bolus` | CGM 혈당값, 볼루스 인슐린 |
| **Park_2025** | `glucose_value_mg_dl`, `carbs` | CGM 혈당값, 탄수화물 섭취량 |
| **PEDAP** | `glucose_value_mg_dl`, `basal`, `bolus`, `carbs` | CGM 혈당값, 기저/볼루스 인슐린, 탄수화물 |
| **UCHTT1DM** | `glucose_value_mg_dl`, `bolus` | CGM 혈당값, 볼루스 인슐린 |

---

## Part B. 공통 목적 변수 (Target Variable Y)

모든 Tier에서 동일하게 적용됩니다.

| 변수명 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `Y` | $G_{t+H}$ , $H = 6\text{ steps}$ | mg/dL | Lookback 윈도우의 마지막 시점($t$)으로부터 30분 후(5분 × 6)의 CGM 혈당값 |

---

## Part C. Tier별 입력 변수 정의

---

### 【Tier 1】 Baseline Linear Regression

**모델:** Ridge Regression (규제 항 포함 선형 회귀)

#### C.1-a. 단변량 모드 (Univariate)
오직 혈당 과거값(Lookback)만을 사용합니다.

| 변수명 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `glucose_value_mg_dl_t-5` | $G_{t-5}$ | mg/dL | 현재 기준 25분 전 혈당값 |
| `glucose_value_mg_dl_t-4` | $G_{t-4}$ | mg/dL | 현재 기준 20분 전 혈당값 |
| `glucose_value_mg_dl_t-3` | $G_{t-3}$ | mg/dL | 현재 기준 15분 전 혈당값 |
| `glucose_value_mg_dl_t-2` | $G_{t-2}$ | mg/dL | 현재 기준 10분 전 혈당값 |
| `glucose_value_mg_dl_t-1` | $G_{t-1}$ | mg/dL | 현재 기준 5분 전 혈당값 |
| `glucose_value_mg_dl_t-0` | $G_t$ | mg/dL | 현재 시점 혈당값 |

#### C.1-b. 다변량 모드 (Multivariate)
혈당 Lookback에 데이터셋 공변량 Lookback을 Concatenation합니다.

| 변수명 패턴 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `[covariate]_t-k` | $C_{t-k}$ ($k=0..5$) | 원본 단위 | 각 공변량의 $t-k$ 스텝 래그 값 (원시값 그대로 투입) |

> **총 Feature Dimension:** $6 \times (1 + N_{\text{covariates}})$ 차원

---

### 【Tier 2】 Classic Machine Learning

**모델:** Decision Tree (depth=5) + Random Forest (trees=100, max_depth=15, n_jobs=-1)

Tier 1과 동일한 원시 원도우(Raw Lookback)를 사용하나, 모델 자체의 비선형 처리 능력에 의존하여 파생 변수를 추가하지 않습니다.

| 변수명 패턴 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `glucose_value_mg_dl_t-k` | $G_{t-k}$ ($k=0..5$) | mg/dL | Lookback 혈당 시퀀스 |
| `[covariate]_t-k` | $C_{i,t-k}$ | 원본 단위 | 모든 공변량의 Lookback 시퀀스 |

> **총 Feature Dimension:** $6 \times (1 + N_{\text{covariates}})$ 차원 (Tier 1 Multivariate와 동일)

---

### 【Tier 2.5】 Feature Engineering — v1

**모델:** Decision Tree (depth=5) + Random Forest (trees=50, max_depth=20, n_jobs=-1)
**핵심 변화:** 이벤트 기억 파생 변수 + 역학/임상 파생 변수를 원시 시퀀스에 Concatenation

#### C.3-a. 원시 이벤트 파생 변수 (Raw Event Engineering)
각 공변량($C_i$)에 대해 아래 2종의 파생 변수를 생성 후 Lookback 윈도우에 포함합니다.

| 변수명 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `[covariate]_time_since` | $\tau_{i,t} = \sum_{s=0}^{t} \mathbf{1}[C_{i,s}=0]$, $\text{clip}(0, 288)$ | steps | 해당 공변량의 마지막 발생 이벤트로부터 경과한 스텝 수 (최대 24시간=288스텝) |
| `[covariate]_decay` | $\hat{C}_{i,t} = \text{EWM}(C_i, \text{halflife}=12)$ | 원본 단위 | Exponential Weighted Mean: 30분(12스텝) 반감기를 가진 지수 이동 평균 (IOB/COB 근사) |

#### C.3-b. 역학(Kinetic) 파생 변수 (Lookback 혈당 시퀀스 기반)

| 변수명 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `Velocity` | $v_t = G_t - G_{t-1}$ | mg/dL/step | 혈당 상승/하강 속도 (1차 후방 차분) |
| `Acceleration` | $a_t = v_t - v_{t-1} = G_t - 2G_{t-1} + G_{t-2}$ | mg/dL/step² | 혈당 변화 가속도 (2차 후방 차분) |
| `Window_AUC` | $\int_{t-5}^{t} G \, ds \approx \text{trapz}(G_{t-5:t})$ | mg/dL·step | 사다리꼴 공식에 의한 Lookback 구간 혈당 적분 |

#### C.3-c. 임상(Clinical) 파생 변수

| 변수명 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `Window_Mean` | $\bar{G} = \frac{1}{6}\sum_{k=0}^{5} G_{t-k}$ | mg/dL | Lookback 윈도우 혈당 평균 |
| `Window_Std` | $\sigma_G = \text{std}(G_{t-5:t})$ | mg/dL | Lookback 윈도우 혈당 표준편차 (당 변동성) |
| `TIR` | $\frac{1}{6}\sum \mathbf{1}[70 \le G \le 180]$ | ratio [0,1] | Time In Range: 혈당 정상 범위 내 비율 |
| `TAR` | $\frac{1}{6}\sum \mathbf{1}[G > 180]$ | ratio [0,1] | Time Above Range: 고혈당 구간 비율 |
| `TBR` | $\frac{1}{6}\sum \mathbf{1}[G < 70]$ | ratio [0,1] | Time Below Range: 저혈당 구간 비율 |

#### C.3-d. Kovatchev 위험 지수 (Risk Index)

| 변수명 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `LBGI` | $\bar{\text{LBGI}} = \frac{1}{6}\sum 10 \cdot [\min(f(G), 0)]^2$ | (무차원) | Low Blood Glucose Index: 저혈당 고위험 비대칭 위험 지수 평균 |
| `HBGI` | $\bar{\text{HBGI}} = \frac{1}{6}\sum 10 \cdot [\max(f(G), 0)]^2$ | (무차원) | High Blood Glucose Index: 고혈당 고위험 비대칭 위험 지수 평균 |
| | $f(G) = 1.509 \cdot (\ln G)^{1.084} - 5.381$ | | Kovatchev 변환 함수 |

#### C.3-e. 일주기 리듬 (Circadian)

| 변수명 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `sin_hour` | $\sin(2\pi \cdot h / 24)$ | [-1, 1] | 현재 시각($h$: 0~23)의 일주기 사인 성분 |
| `cos_hour` | $\cos(2\pi \cdot h / 24)$ | [-1, 1] | 현재 시각($h$: 0~23)의 일주기 코사인 성분 |

> **총 Feature Dimension 공식:**
> $$D_{2.5} = 6 \times (1 + N_{\text{covariates}} \times 3) + 12$$
> - $N_{\text{covariates}} \times 3$: 각 공변량당 원시값 + time_since + decay 파생 3종
> - $+12$: 역학(3) + 임상(5) + 위험(2) + 일주기(2) 전역 파생 변수

---

### 【Tier 2.5_v2】 Feature Engineering — 고주파 위상 임베딩

**모델:** Decision Tree (depth=5) + Random Forest (trees=50, max_depth=30, min_samples_leaf=20, n_jobs=-1)
**핵심 변화:** (Tier 2.5 모든 변수 유지) + 핵심 공변량 다중 주파수 위상 임베딩 + Jerk + SD1

Tier 2.5의 모든 변수를 Superset으로 유지하며, 아래의 변수들을 추가합니다.

#### C.4-a. 핵심 공변량 위상 임베딩 (Multi-Frequency Positional Encoding)

키워드 필터 (`ins`, `carb`, `meal`, `cal`, `dose`)에 해당하는 공변량에만 적용하여 Feature Dilution을 방지합니다.

| 변수명 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `[core_cov]_sin_30m` | $\sin(2\pi \cdot \tau / 6)$ | [-1, 1] | 30분(6스텝) 주기 – 단순당 즉각 흡수 반응 위상 |
| `[core_cov]_cos_30m` | $\cos(2\pi \cdot \tau / 6)$ | [-1, 1] | 30분 주기 코사인 좌표 |
| `[core_cov]_sin_1h` | $\sin(2\pi \cdot \tau / 12)$ | [-1, 1] | 1시간(12스텝) 주기 – 속효성 인슐린 최대 활성 위상 |
| `[core_cov]_cos_1h` | $\cos(2\pi \cdot \tau / 12)$ | [-1, 1] | 1시간 주기 코사인 좌표 |
| `[core_cov]_sin_2h` | $\sin(2\pi \cdot \tau / 24)$ | [-1, 1] | 2시간(24스텝) 주기 – 식사 소화 안정화 위상 |
| `[core_cov]_cos_2h` | $\cos(2\pi \cdot \tau / 24)$ | [-1, 1] | 2시간 주기 코사인 좌표 |

> $\tau$: 해당 공변량의 마지막 이벤트로부터 경과 스텝 수 (`time_since`)

#### C.4-b. 고위험 동역학 파생 변수 (Advanced Kinetics)

| 변수명 | 수식 | 단위 | 설명 |
|---|---|---|---|
| `Jerk` | $j_t = a_t - a_{t-1} = G_t - 3G_{t-1} + 3G_{t-2} - G_{t-3}$ | mg/dL/step³ | 가가속도: 가속도의 변화율 (3차 후방 차분). 돌발 스파이크 변곡점 탐지기 |
| `SD1` | $\text{SD1} = \frac{1}{\sqrt{2}} \cdot \text{std}(\Delta G_{t-5:t})$ | mg/dL | Poincaré 단기 잔물결 변동성: 스텝-바이-스텝 혈당 변화량의 표준편차를 $\sqrt{2}$로 정규화한 비선형 생리 안정성 지수 |

> **총 Feature Dimension 공식:**

$$D_{2.5\_v2} = D_{2.5} + N_{\text{core cov}} \times 6 + 2$$

> - $N_{\text{core cov}} \times 6$: 핵심 공변량 1개당 $\sin/\cos \times 3$ 주기
> - $+2$: Jerk, SD1

---

## Part D. Feature Dimension 요약

| 데이터셋 | Raw Covariates ($N_c$) | Tier 2.5 Dim | Tier 2.5_v2 Dim | 핵심 공변량 수 |
|---|:---:|:---:|:---:|:---:|
| AIDET1D | 0 | 18 | 20 | 0 |
| BIGIDEAs | 3 | 54 | 92 | 1 (`calorie`) |
| Bris-T1D_Open | 6 | 126 | 236 | 3 (`bolus`, `basal`, `carbs`) |
| CGMacros_Dexcom | 15 | 288 | 398 | 3 (`calories`, `carbs`, `sugars`) |
| CGMacros_Libre | 15 | 288 | 398 | 3 |
| CGMND | 0 | 18 | 20 | 0 |
| GLAM | 1 | 36 | 38 | 1 (`event_marker`) |  
| HUPA-UCM | 5 | 126 | 200 | 3 (`bolus`, `basal`, `carbs`) |
| IOBP2 | 1 | 36 | 74 | 1 (`bolus`) |
| Park_2025 | 1 | 36 | 38 | 1 (`carbs`) |
| PEDAP | 3 | 54 | 56 | 1 (`bolus`) |
| UCHTT1DM | 1 | 36 | 38 | 1 (`bolus`) |

---

## Part E. 데이터 누출(Data Leakage) 방지 설계

모든 Tier에서 아래 원칙을 엄수하여 시계열 데이터 무결성을 유지합니다.

| 규칙 | 내용 |
|---|---|
| **Lookback-only 슬라이싱** | 파생 변수 계산 시 현재 시점($t$) 및 이전 데이터만 사용. 미래($t+1$~) 정보 접근 원천 차단 |
| **갭 필터링** | 윈도우 내 타임스탬프 간격이 $1.5 \times \text{median\_gap}$ 초과하는 경우 해당 윈도우를 학습/평가에서 제외 |
| **`time_since` 클리핑** | 이벤트 경과 시간을 최대 288스텝(24시간)으로 클리핑하여 단순 카운터 발산 방지 |
| **train/test 분할** | 환자별로 시간 순서를 유지하여 앞 80%를 train, 뒤 20%를 test로 분할 (시간 역전 방지) |
