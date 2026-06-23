# 시계열 파생 변수 수학적 정의 및 표기법 명세서 (문서 2)
**— 연속혈당측정치 동역학, 임상 위험 지수 및 공변량 지수 감쇄 공식의 수학적 정의 —**

본 문서는 Glucose-ML 프로젝트의 피처 엔지니어링 파이프라인에서 산출되는 모든 시계열 파생 변수(Derived Features)의 수학적 표기법(Notation)과 구체적인 연산 공식을 명시합니다.

---

## 1. 기본 노테이션 (Basic Notation)

- $G_\tau$: 특정 시점 $\tau$에서의 CGM 혈당 수치 (mg/dL)
- $L$: 과거 룩백 시퀀스 길이 (`LOOKBACK_STEPS`로 정의되며, 기본값은 3)
- $t$: 현재 예측 시점 (예측을 시작하는 기준 시점)
- $\mathbf{G}_t$: 시점 $t$ 기준의 과거 룩백 혈당 시퀀스 벡터
  $$\mathbf{G}_t = [G_{t-L}, G_{t-L+1}, \dots, G_{t-1}]$$

---

## 2. 시계열 동역학 파생 변수 (Kinematics Features)

룩백 시퀀스의 미분 변위를 모사하여 혈당이 변동하는 속도와 물리적 관성을 수치화합니다.

### 2.1. 속도 (Velocity)
- **정의:** 혈당의 1차 후방 차분(First-order backward difference)입니다.
- **공식:**
  $$v_\tau = G_\tau - G_{\tau-1}, \quad \tau \in [t-L+1, t-1]$$
- **피처 투입 값 (마지막 시점 속도):**
  $$v_{t-1} = G_{t-1} - G_{t-2}$$

### 2.2. 가속도 (Acceleration)
- **정의:** 혈당의 2차 후방 차분(Second-order backward difference)입니다.
- **공식:**
  $$a_\tau = v_\tau - v_{\tau-1}, \quad \tau \in [t-L+2, t-1]$$
- **피처 투입 값 (마지막 시점 가속도):**
  $$a_{t-1} = v_{t-1} - v_{t-2}$$

### 2.3. 저크 (Jerk)
- **정의:** 혈당의 3차 후방 차분(Third-order backward difference)입니다. 단, 룩백 스텝 $L \ge 4$ 인 경우에만 유효하게 정의됩니다.
- **공식:**
  $$j_\tau = a_\tau - a_{\tau-1}, \quad \tau \in [t-L+3, t-1]$$
- **피처 투입 값 (마지막 시점 저크):**
  $$j_{t-1} = a_{t-1} - a_{t-2}$$

---

## 3. 시계열 통계 및 변동성 파생 변수 (Statistical & Variability Features)

### 3.1. 윈도우 평균 (Window Mean)
- **공식:**
  $$\mu_t = \frac{1}{L} \sum_{i=1}^{L} G_{t-i}$$

### 3.2. 윈도우 표준편차 (Window Std)
- **공식:**
  $$\sigma_t = \sqrt{\frac{1}{L} \sum_{i=1}^{L} (G_{t-i} - \mu_t)^2}$$

### 3.3. Poincaré SD1 (단기 변동성)
인접한 혈당 변위(차분값)들의 표준편차를 $\sqrt{2}$로 나누어 단기 요동 세기를 대각선 축 대비로 나타낸 지표입니다.
- **공식:**
  - Let $D_\tau = G_\tau - G_{\tau-1}$ for $\tau \in [t-L+1, t-1]$.
  - $\mathbf{D}_t = [D_{t-L+1}, \dots, D_{t-1}]$는 차분 시퀀스 벡터입니다.
  $$SD1_t = \frac{\text{std}(\mathbf{D}_t)}{\sqrt{2}}$$
  $$\text{where } \text{std}(\mathbf{D}_t) = \sqrt{\frac{1}{L-1} \sum_{\tau=t-L+1}^{t-1} (D_\tau - \mu_D)^2}, \quad \mu_D = \frac{1}{L-1} \sum_{\tau=t-L+1}^{t-1} D_\tau$$

---

## 4. 임상 관리 대역 지표 (Clinical Ranges)

룩백 윈도우 내에서 환자의 상태가 유효 임상 구간에 얼마나 오래 머물렀는지 분포 비율로 지표화합니다. 지시 함수(Indicator function) $\mathbb{I}(\cdot)$을 사용합니다.

### 4.1. TIR (Time in Range, 적정 혈당 체류 비율)
- **공식:**
  $$\text{TIR}_t = \frac{1}{L} \sum_{i=1}^{L} \mathbb{I}(70 \le G_{t-i} \le 180)$$

### 4.2. TAR (Time Above Range, 고혈당 체류 비율)
- **공식:**
  $$\text{TAR}_t = \frac{1}{L} \sum_{i=1}^{L} \mathbb{I}(G_{t-i} > 180)$$

### 4.3. TBR (Time Below Range, 저혈당 체류 비율)
- **공식:**
  $$\text{TBR}_t = \frac{1}{L} \sum_{i=1}^{L} \mathbb{I}(G_{t-i} < 70)$$

---

## 5. 비대칭 생리학적 위험 지수 (Kovatchev Risk Indices)

혈당의 생리학적 위험 비대칭성(저혈당은 $70 \text{ mg/dL}$ 이하로 떨어지면 급사 위험이 있으나 고혈당은 상대적으로 수치가 넓게 분포함)을 보정하기 위한 수학적 변환 지표입니다.

### 5.1. 혈당의 대수 비대칭 변환 함수
- **공식:**
  $$g = \max(G, 1.0) \quad \text{(0 또는 음수 입력 방지 장치)}$$
  $$f(g) = 1.509 \times \left( (\ln(g))^{1.084} - 5.381 \right)$$

### 5.2. 혈당 영역별 가중치 함수
- **저혈당 가중치 ($rl$):**
  $$rl(g) = \begin{cases} 10 \times (f(g))^2, & \text{if } f(g) < 0 \\ 0, & \text{if } f(g) \ge 0 \end{cases}$$
- **고혈당 가중치 ($rh$):**
  $$rh(g) = \begin{cases} 10 \times (f(g))^2, & \text{if } f(g) > 0 \\ 0, & \text{if } f(g) \le 0 \end{cases}$$

### 5.3. LBGI (저혈당 위험 지수) 및 HBGI (고혈당 위험 지수)
- **공식:**
  $$\text{LBGI}_t = \frac{1}{L} \sum_{i=1}^{L} rl(G_{t-i})$$
  $$\text{HBGI}_t = \frac{1}{L} \sum_{i=1}^{L} rh(G_{t-i})$$

---

## 6. 누적 혈당 면적 (Area Under Curve - AUC)

과거 시계열 동안 환자가 축적한 총 대사 활성 에너지를 사다리꼴 공식(Trapezoidal Integration)을 통해 근사하여 표현합니다.
- **공식 ($dt = 1$ 가정):**
  $$\text{AUC}_t = \sum_{i=1}^{L-1} \frac{G_{t-L+i-1} + G_{t-L+i}}{2}$$

---

## 7. 일주기 시간 위상 인코딩 (Circadian Encoding)

호르몬 일주기 패턴을 모사하기 위해 하루의 시간 정보(24시간)를 사인과 코사인 공간에 매핑합니다.
- **노테이션:** $H_t$는 룩백 윈도우 마지막 시점의 당일 시각(Hour, float 단위, $0.0 \le H_t < 24.0$)을 나타냅니다.
- **공식:**
  $$\text{tod\_sin}_t = \sin\left(\frac{2 \pi H_t}{24}\right)$$
  $$\text{tod\_cos}_t = \cos\left(\frac{2 \pi H_t}{24}\right)$$

---

## 8. 이벤트 공변량 전처리 변수 (Event Covariate Features)

불연속적 처방 요인(인슐린, 식사량 등)이 체내에 잔류하며 일으키는 화학적 활성 감쇄와 누적 지연 효과를 물리적으로 모사합니다. $E_\tau$는 시점 $\tau$에서의 원시 이벤트 입력치(인슐린 단위 또는 탄수화물 중량)를 나타냅니다.

### 8.1. 이벤트 경과 시간 (Time-Since-Event)
마지막 이벤트 발생 시점으로부터 경과한 시간 단계를 계산하고, 최대 24시간 범위(5분 주기 기준 288 스텝)에서 클리핑합니다.
- **공식:**
  $$\text{TSE}_t(E) = \min\left( t - 1 - \max \{ \tau < t \mid E_\tau > 0 \}, 288 \right)$$
  - 단, 환자가 인슐린이나 식사 이벤트를 개시한 적이 없는 기동(Initialization) 단계에서는 $\text{TSE}_t(E) = 288$로 고정합니다.

### 8.2. 지수식 물리 감쇄 (Exponential Decay)
체내 약동학적 반감기를 EWMA(Exponentially Weighted Moving Average) 방식으로 근사하여 인슐린 활성량(IOB) 및 탄수화물 흡수 잔량(COB)을 정량화합니다.
- **공식:**
  $$\text{Decay}_t(E) = \alpha E_{t-1} + (1 - \alpha) \text{Decay}_{t-1}(E)$$
  $$\text{where } \alpha = 1 - e^{-\frac{\ln(2)}{\lambda}}$$
  - 반감기 파라미터 $\lambda$: 1시간(5분 주기 기준 12 스텝)으로 설정하므로 $\lambda = 12$ 이며, $\alpha \approx 0.0561$ 입니다.

### 8.3. 핵심 공변량 주파수 다중 인코딩 (Multi-Frequency Positional Encoding)
이벤트 경과 시간($\text{TSE}$)에 대해 30분, 1시간, 2시간 주기의 다중 주파수 코디네이트 위상을 주입하여 선형 결합 구조가 혈당 지연 곡선을 쉽게 포착하도록 유도합니다.
- **공식 (주파수 주기 $P \in \{6, 12, 24\}$ 스텝):**
  $$\text{pe\_sin}_t(E, P) = \sin\left(\frac{2 \pi \text{TSE}_t(E)}{P}\right)$$
  $$\text{pe\_cos}_t(E, P) = \cos\left(\frac{2 \pi \text{TSE}_t(E)}{P}\right)$$

---

## 9. 교차 질환 및 영역 특화 보조 변수 (Cross-Dataset Auxiliary Features)

### 9.1. Fasting Proxy (야간 기저 대리 지표)
야간 공복(새벽 0시부터 6시 사이) 혈당 수준을 수치화합니다.
- **공식:**
  $$\text{FastingProxy}_t = \mathbb{I}(0 \le \text{Hour}_t < 6) \times \mu_t$$

### 9.2. Postmeal Rise (식후 상승 대리 지표)
식사 자극으로 인한 혈당 스파이크의 단기 변동 크기를 간접 수치화합니다.
- **공식:**
  $$\text{PostmealRise}_t = \max(0.0, G_{t-1} - \min(\mathbf{G}_t))$$

### 9.3. High Persistence (고혈당 지속 비율)
- **공식:**
  $$\text{HighPersistence}_t = \frac{1}{L} \sum_{i=1}^{L} \mathbb{I}(G_{t-i} > 180)$$

### 9.4. UMD (Unannounced Meal Detection, 무알림 식사 감지 확률 변수)
식후 혈당 변화 패턴(속도와 가속도의 동시 증가 양상)에 근거해 자가 기록이 누락된 식사 시점을 감지하기 위한 확률 대리 지표입니다.
- **공식:**
  $$\text{UMD}_t = \begin{cases} \min\left(1.0, \max\left(0.0, (v_{t-1} - 1.5) \times 0.5 + a_{t-1} \times 0.2\right)\right), & \text{if } v_{t-1} > 1.5 \text{ and } a_{t-1} > 0 \\ 0.0, & \text{otherwise} \end{cases}$$

---

> [!NOTE]
> **시계열 파생 변수의 데이터 누출(Data Leakage) 배제 및 안정성 검증**
>
> 본 문서에 명세된 모든 파생 변수들은 모델 학습 및 평가 시 데이터 누출(Data Leakage)을 완벽하게 차단하도록 설계되었습니다.
>
> 1. **인과적(Causal) 연산 구조:** 모든 동역학, 임상 범위 지표, 생리학적 위험도 지수, AUC 등은 미래 시점($t$ 이상)을 참조하지 않고, 오직 과거 룩백 시퀀스 $\mathbf{G}_t$ 영역(시점 $t-1$ 이하)의 데이터만 사용하여 계산됩니다. 이에 따라 미래 목표 변수($t$ 시점 이상)의 정보 유출이 원천 차단됩니다.
> 2. **이벤트 공변량 누적의 안전성:** 경과 시간($\text{TSE}$) 및 지수식 물리 감쇄($\text{Decay}$) 등의 특징 추출 연산은 시간의 흐름(과거에서 미래 방향)에 따라 순차적으로만 계산(ffill, EWMA)되므로 미래 정보의 역류가 발생하지 않습니다.
> 3. **인스턴스 단위 정규화(RevIN):** 전역 스케일러(StandardScaler 등)를 사전 사용할 때 발생하는 테스트 셋 정보 누출을 예방하기 위해, 각 윈도우 샘플 내의 통계량만을 독립적으로 사용하는 RevIN 모듈을 도입하여 분포 누출을 차단합니다.
> 4. **환자 단위 분리 원칙(Subject-Level Split):** 슬라이딩 윈도우의 시간적 겹침으로 인한 데이터 누출을 방지하기 위해, 데이터셋 분리 시 환자(Subject)를 기준으로 완전히 공간 격리(Train 70% / Val 15% / Test 15%)를 적용합니다.
