# 수학적 정식화 (Mathematical Formulation)
## Tier 2: Classic Machine Learning Pipeline

본 문서는 시계열 슬라이딩 윈도우(Sliding Window) 환경에서 다중 모달 공변량(Multi-modal Covariates)을 결합하여 혈당을 예측하는 Tier 2 머신러닝 모형들의 구조를 수학적으로 정식화(Formulation)한 것입니다.

---

### 1. 전제 표기법 (Notation Setup)

어떤 환자의 $t$ 시점에서 측정된 혈당 수치를 $G_t$라 하고, 동시에 측정된 $K$개의 외부 공변량(인슐린, 영양소, 심박조율 등) 벡터를 $\mathbf{C}_t \in \mathbb{R}^K$라고 정의합니다. 

실험에서 채택한 과거 관측 범위(Lookback)를 $L$, 예측하고자 하는 미래 시점(Horizon)을 $H$라 할 때, 하나의 윈도우 인스턴스 $i$에 대한 입력 데이터 행렬 $\mathbf{X}^{(i)}$와 타겟 스칼라 $Y^{(i)}$는 다음과 같이 펼쳐진 형태(Flattened Vector)로 사상됩니다.

$$ \mathbf{X}^{(i)} = \Big[ G_{t-L+1}, \, \mathbf{C}_{t-L+1}, \dots, \, G_t, \, \mathbf{C}_t \Big] \in \mathbb{R}^{L \times (1 + K)} $$
$$ Y^{(i)} = G_{t+H} $$

- 학습 데이터셋은 $\mathcal{D} = \{ (\mathbf{X}^{(1)}, Y^{(1)}), (\mathbf{X}^{(2)}, Y^{(2)}), \dots, (\mathbf{X}^{(N)}, Y^{(N)}) \}$ 으로 구성됩니다.

---

### 2. Decision Tree Regressor (의사결정나무 회귀)

Decision Tree 모델 $f_{DT}$는 입력 차원 $\mathbb{R}^{L \times (1+K)}$ 공간을 축에 평행한 $M$개의 서로소 구역(Disjoint Regions) $R_1, R_2, \dots, R_M$으로 분기(Recursive Binary Splitting)하여 나눕니다.

입력 $\mathbf{X}$가 주어졌을 때의 혈당 예측값은 해당 $\mathbf{X}$가 속한 구역 $R_m$에 있는 타겟 값들의 평균 $\hat{c}_m$으로 결정됩니다.

$$ \hat{Y} = f_{DT}(\mathbf{X}) = \sum_{m=1}^M \hat{c}_m \cdot \mathcal{I}(\mathbf{X} \in R_m) $$
여기서 $\mathcal{I}$는 지시 함수(Indicator function)이며, $\hat{c}_m$은 다음과 같습니다.
$$ \hat{c}_m = \frac{1}{N_m} \sum_{\mathbf{X}^{(i)} \in R_m} Y^{(i)} $$

**분기 탐색 기준(Splitting Criterion):**
임의의 노드에서 특성 $j$ (예: $t-1$ 시점의 인슐린 용량)와 임계값 $s$를 기준으로 데이터를 $D_{left}$와 $D_{right}$로 쪼갤 때, 분기 후의 **잔차 제곱합(SSR: Sum of Squared Residuals)을 최소화**하도록 최적의 $(j, s)$ 쌍을 찾습니다.

$$ \min_{j,s} \Bigg[ \min_{c_{left}} \sum_{\mathbf{X}^{(i)} \in D_{left}(j,s)} (Y^{(i)} - c_{left})^2 \,+\, \min_{c_{right}} \sum_{\mathbf{X}^{(i)} \in D_{right}(j,s)} (Y^{(i)} - c_{right})^2 \Bigg] $$
*이 제약 조건 하에서 깊이 제한(Max Depth) $d \le 5$까지 분기를 반복하며, 이는 우리가 추출한 `Decision_Tree_Rules`로 물리적 번역이 됩니다.*

---

### 3. Random Forest Regressor (랜덤 포레스트)

Random Forest $f_{RF}$는 $B$개(본 실험에서는 $B=50$)의 독립적인 의사결정나무 $T_1, \dots, T_B$를 구성하고 이를 평균 내어 예측 편향(Variance)과 과적합을 억제하는 앙상블(Bagging) 모형입니다.

$$ \hat{Y} = f_{RF}(\mathbf{X}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{X}; \Theta_b) $$

여기서 $\Theta_b$는 각 트리 $b$를 학습시키기 위해 원래 데이터 $\mathcal{D}$에서 복원 추출한 **부트스트랩 표본(Bootstrap Sample)**과 분기마다 무작위로 선택된 특성 풀(Feature pool subset)을 규정하는 확률 벡터입니다.

**변수 중요도(Feature Importance - MDI, Mean Decrease in Impurity):**
의료 데이터의 핵심인 "어떤 인자가 혈당에 영향을 주는가?"를 평가하기 위해 노드 불순도(Impurity) 감소량을 사용합니다. 특정 공변량 $j$ (예: 칼로리)가 갖는 중요도 $Imp(j)$는, 모든 트리 $B$에서 변수 $j$가 기준으로 사용된 노드 $t$들이 만들어낸 불순도 감소폭 $\Delta I(t)$의 합으로 나타냅니다.

$$ Imp(j) = \frac{1}{B} \sum_{b=1}^B \sum_{t \in T_b : v(t)=j} p(t) \, \Delta I(t) $$
*(여기서 $p(t)$는 노드 $t$에 도달한 샘플의 비율, $v(t)$는 노드 $t$의 분기 변수를 뜻합니다.)*
본 실험에서 출력된 Top 3 Feature 퍼센티지는 위에서 산출된 $Imp(j)$ 값들을 합산 정규화(Normalization, $\sum Imp=1$)한 수치입니다.
