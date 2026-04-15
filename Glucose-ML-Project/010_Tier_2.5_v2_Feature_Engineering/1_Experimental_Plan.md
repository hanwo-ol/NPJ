# [Tier 2.5_v2] High-Frequency Phase Encoding & Advanced Kinetics

## 1. 개요 (Overview)
Tier 2.5의 "시간 감쇠(Exponential Decay)"가 가진 정적이고 둔탁한 한계를 돌파하기 위해, 트랜스포머의 **Positional Encoding 철학**을 도입합니다. 이벤트 발생 후 경과 시간을 다중 주파수(30분, 1시간, 2시간)의 삼각함수로 임베딩하여, 트리가 '인슐린 활성' 또는 '단순당 소화'의 **특정 생리학적 위상(Phase)**을 геометри적으로 인식하게 만듭니다. 또한 Jerk(가가속도) 및 Poincare 지표를 통해 미세한 스파이크 전조를 탐지합니다.

## [확정] 모형 과적합 방지 통제 (Hyperparameter Guardrails)
수백 개로 확장되는 차원 속에서 삼각함수 임베딩이 생성하는 미세한 '물결 모양'의 불필요한 분기를 방지하기 위해 과적합 제어 장치를 도입합니다.
- **`max_depth=30`:** 트리 깊이를 대폭 개방하되 무제한 허용(None)은 금지.
- **`min_samples_leaf=20`:** 특정 위상(Phase) 분기가 최소 20개의 통계적 근거를 가질 때만 노드가 쪼개지도록 강제하여 노이즈 과적합을 차단합니다.

## Proposed Changes (동역학 및 임상 변수 추가)

### A. 핵심 공변량 국한-다중 고주파수 이벤트 위상 임베딩 (Multi-Frequency Phase Encoding)
5~15분 간격 시계열의 "급성 반응"을 놓치지 않기 위해 밀도 높은 주기로 스케일 다운합니다.
**단, 차원의 희석(Feature Dilution)을 막기 위해 '속효성 인슐린', '탄수화물', '칼로리' 등 혈당과 직결된 "주요 핵심 공변량"에만 위상 임베딩을 적용**하며 나머지 부수 영양소는 이전의 단일 변수로 유지합니다.

* $P_1 = 6 \text{ steps}$ (30분 주기): $\sin(2\pi \frac{t}{6}), \cos(2\pi \frac{t}{6})$
* $P_2 = 12 \text{ steps}$ (1시간 주기): $\sin(2\pi \frac{t}{12}), \cos(2\pi \frac{t}{12})$
* $P_3 = 24 \text{ steps}$ (2시간 주기): $\sin(2\pi \frac{t}{24}), \cos(2\pi \frac{t}{24})$

이로써 이벤트마다 선형적 증가 수치가 아닌 **6개의 기하학적 매핑(Spatial Coordinate)**이 생성됩니다.

### B. 고위험 시계열 동역학 변수 추가
단순한 속도(Velocity)를 넘어선 극단적 이상 징후 센서 2종을 도입합니다.

1. **Jerk (가가속도):** 
   - 수학적 정의: $\Delta a_t = a_t - a_{t-1} = (G_t - 3G_{t-1} + 3G_{t-2} - G_{t-3})$
   - 임상적 의미: 혈당 곡선이 완만하게 타다가 갑자기 투석하듯 치솟는 "변곡점 돌파 속도" 측정.
2. **Poincaré SD1 (단기 잔물결 변동성):**
   - 수학적 정의: $\text{SD1} = \frac{1}{\sqrt{2}} \times \text{Std}(\Delta G)$ (윈도우 내 속도 벡터의 표준편차)
   - 임상적 의미: 현재 환자의 항상성(Homeostasis)이 걷잡을 수 없이 파괴되어 잔진동(Fluctuation)이 심한 불안정 상태인가를 판독.

---

> [!NOTE]
> 선생님께서 승인/보완해주신 모든 수학적 제약 조건(min_samples_leaf, 핵심 공변량 필터링, 정석적 3차 후방 차분 Jerk 및 Poincare SD1 수식)을 위 설계에 확정 병합하였습니다. 곧바로 인코딩 코드 구현에 착수합니다.
