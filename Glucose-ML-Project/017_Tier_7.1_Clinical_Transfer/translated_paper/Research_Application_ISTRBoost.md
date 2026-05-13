# 우리 연구 (Glucose-ML Clinical Transfer) 적용 방안: S-TrAdaBoost.R2

S-TrAdaBoost.R2 논문 *"ISTRBoost: Importance Sampling Transfer Regression using Boosting"* (Gupta et al., 2022, arXiv:2204.12044)은 **소스 데이터가 타겟보다 압도적으로 클 때 발생하는 가중치 편향(Weight Skewness)을 Importance Sampling으로 교정**하여, 기존 Two-Stage TrAdaBoost.R2(TTR2)의 Negative Transfer 문제를 완화하는 부스팅 기반 전이 회귀 방법론을 제시합니다.

현재 진행 중인 **Glucose-ML-Project (017_Tier_7.1_Clinical_Transfer)** 파이프라인의 Tier 7 TrAdaBoost.R2를 이 논문의 방법론으로 업그레이드하는 적용 방안은 다음과 같습니다.

## 1. 문제 진단: 현재 Tier 7 파이프라인의 구조적 위험

* **논문이 지적하는 문제:** TTR2는 소스 데이터가 타겟보다 클 때, 초기 가중치가 소스 쪽으로 쏠려 부스팅 전체가 소스에 과적합됩니다. 논문 Fig.1에서 Concrete 데이터셋의 타겟 비율이 35% → 63%로 증가할 때 TTR2의 R²가 비전이(AdaBoost.R2)보다 낮아지는 **Negative Transfer**를 실증합니다.
* **Glucose-ML 현황:** 우리 Tier 7의 소스(T1D 6개 데이터셋, ~1,450명) 대 타겟(ShanghaiT2DM, ~100명) 비율은 약 **14:1**입니다. 이는 논문이 경고하는 "소스 >> 타겟" 시나리오에 정확히 해당하며, 현재 TTR2 기반 파이프라인이 Negative Transfer 위험에 노출되어 있음을 의미합니다.

## 2. Importance Sampling으로 소스 사전 필터링

* **논문의 접근:** 부스팅 시작 전에, 타겟 데이터 평균($\bar{x}^T$)과의 유클리드 거리(L2 norm)를 기준으로 소스 인스턴스를 정렬하고, 상위 $p$개만 선별합니다. 나머지 $(n-p)$개는 부스팅에 투입하기 전에 폐기합니다.

$$X_{ES} = \text{argsort}_{x_i \in X_S} \|x^S_i - \bar{x}^T\|, \quad |X_{ES}| = p \ll n$$

* **Glucose-ML 적용:**
  - 소스 6개 데이터셋의 전체 인스턴스(~1,450명 × 윈도우 수)에서, 타겟(ShanghaiT2DM)의 평균 피처 벡터와 가장 가까운 상위 $p$개 인스턴스만 선별합니다.
  - **피처 공간:** Tier 2.5 v3에서 정의된 10여 가지 파생 피처(`Window_Mean`, `Window_Std`, `Velocity`, `LBGI`, `HBGI` 등)의 공간에서 L2 거리를 계산합니다.
  - **$p$의 결정:** 논문에서 명시적 가이드라인이 없으므로, 타겟 인스턴스 수의 2~5배 범위($p = 2m \sim 5m$)를 실험적으로 탐색합니다.
  - **거리 함수 검토:** 논문은 L2(유클리드)를 사용했으나, 우리 피처 스케일이 매우 다른 경우(TIR: 0~100% vs. LBGI: 0~20) 정규화된 거리 또는 마할라노비스 거리로의 대체를 검토합니다.

## 3. k-Center Sampling으로 타겟에 분산 주입 (Variance Injection)

* **논문의 접근:** 소스 데이터에 k-Means 클러스터링을 적용하여 $k$개의 대표 중심점을 추출하고, 이 중심점에 가장 가까운 소스 인스턴스를 타겟 데이터셋에 추가합니다. 이는 타겟의 커버리지를 넓히기 위한 통제된 노이즈 주입입니다.

* **Glucose-ML 적용 (시계열 무결성 보호):**
  - 논문의 k-Center Sampling은 인스턴스 단위로 작동하므로, 시계열 원본을 변형하지 않습니다. 소스의 **실제 윈도우 데이터를 그대로** 타겟에 추가하는 것이므로, 가우스 노이즈 합성과 달리 **생리학적 무결성이 보존**됩니다.
  - $k$의 결정: 타겟 인스턴스 수 $m$과 동일하게 설정($k = m$)하여 타겟 크기를 2배로 확장하는 것을 초기값으로 실험합니다.
  - 최종 타겟 크기: $q = m + k$

## 4. 비제약적 가중치 업데이트 (Unconstrained Weight Update)

* **논문의 접근 (TTR2와의 핵심 차이):** TTR2는 2단계로 나뉘어 (1) 소스 가중치를 줄인 후 (2) 소스를 동결하고 타겟만 업데이트합니다. S-TrAdaBoost.R2는 이 동결을 제거하고 소스와 타겟을 **매 반복마다 동시에** 업데이트하되, 소스를 더 강하게 벌점화합니다:

$$w^{t+1}_i = \begin{cases} w^t_i \cdot \bar{\beta}_t^{e^t_i \cdot \alpha} / Z_t & \text{(소스: } 1 \leq i \leq p\text{)} \\ w^t_i \cdot \beta_t^{(1-e^t_i) \cdot \alpha} / Z_t & \text{(타겟: } p < i \leq p+q\text{)} \end{cases}$$

여기서 $\bar{\beta}_t = \eta_t / (1 - \eta_t)$, $\eta_t = \sum_{k=1}^{p+q} w^t_i e^t_i$, $\beta_t = q/(p+q) + t/((S-1)(1-q/(p+q)))$

* **Glucose-ML 적용:**
  - LightGBM을 기본 학습기(Base Learner)로 유지합니다. 논문은 Decision Tree를 사용했으나, LightGBM은 단일 Decision Tree의 앙상블이므로 기본 학습기로서 호환됩니다.
  - **하이퍼파라미터:** 논문의 기본값(S=30, F=10, α=0.1)을 초기값으로 사용하고, 타겟 검증 세트에서 Bayesian Optimization으로 튜닝합니다.

## 5. 전체 파이프라인 구성

```
[기존 Tier 7]
소스(T1D 1,450명) + 타겟(T2D 100명) → TrAdaBoost.R2 (TTR2) → 예측

[제안 Tier 7.1]
소스(T1D 1,450명)
  ↓ [Importance Sampling] 타겟 평균 기준 L2 거리 → 상위 p개 선별
  ↓ 소스 축소: X_ES (p개, p << 1,450)

타겟(T2D 100명)
  ↓ [k-Center Sampling] 소스 대표점에 가까운 소스 인스턴스 k개 추가
  ↓ 타겟 확장: X_VT (m + k개)

X_ES + X_VT → S-TrAdaBoost.R2 (비제약 가중치 업데이트, S=30 반복)
  ↓
최종 예측 모델 hf = argmin_i error_i
  ↓
[드리프트 모델] (PLOS ONE 논문에서 차용) → 사후 오차 진단 리포트
```

## 6. 실험 설계: 5-Way 비교

| # | 방법 | 설명 | 기대 역할 |
|---|---|---|---|
| 1 | Source-Only | 소스만으로 LightGBM 학습 | 하한선 (Negative Transfer 최대) |
| 2 | Target-Only | 타겟만으로 LightGBM 학습 | 소규모 데이터 한계 확인 |
| 3 | Mixed | 소스+타겟 전부 섞어 학습 | 단순 풀링 기준선 |
| 4 | TTR2 | 기존 TrAdaBoost.R2 | 현재 Tier 7 기준선 |
| 5 | **S-TrAdaBoost.R2** | Importance Sampling + 비제약 업데이트 | **제안 방법** |

**평가 지표:** RMSE, R², MAE (20-fold CV)

**검증 포인트:**
1. S-TrAdaBoost.R2가 TTR2보다 RMSE가 낮은가? (논문 기준 75% 확률)
2. Source-Only 대비 Negative Transfer가 완화되었는가?
3. 타겟 데이터를 10%, 30%, 50%로 변화시킬 때 성능 곡선이 단조 증가하는가? (Higher Asymptote 달성 여부)

## 7. 수식 비교: 논문 vs. 우리 연구

### 7.1 논문의 Importance Sampling 수식

**노테이션 정의:**

| 기호 | 의미 |
|---|---|
| $X_S$ | 소스 데이터셋, $n$개 인스턴스 |
| $X_T$ | 타겟 데이터셋, $m$개 인스턴스 |
| $\bar{x}^T$ | 타겟 인스턴스의 평균 벡터 |
| $X_{ES}$ | Importance Sampling 후 축소된 소스 ($p$개, $p \ll n$) |
| $X_{VT}$ | k-Center Sampling 후 확장된 타겟 ($q = m + k$개) |
| $e^t_i$ | 반복 $t$에서 인스턴스 $i$의 조정된 오차 |
| $\beta_t$ | 타겟 가중치 업데이트 계수 |
| $\bar{\beta}_t$ | 소스 가중치 업데이트 계수 |
| $Z_t$ | 정규화 상수 (타겟 가중치의 합) |
| $S$ | 부스팅 반복 횟수 |
| $\alpha$ | 학습률 |

**Importance Sampling:**

$$X_{ES} = \text{top-}p \left( \|x^S_i - \bar{x}^T\|_2 \right), \quad \forall x_i \in X_S$$

**k-Center Sampling:**

$$X_C = \text{k-Means}(X_S, k) \quad \rightarrow \quad X_{VT} = \text{nearest}(X_S, X_C) \cup X_T$$

**가중치 업데이트:**

$$w^{t+1}_i = \begin{cases} w^t_i \cdot \bar{\beta}_t^{e^t_i \cdot \alpha} / Z_t & \text{소스 } (1 \leq i \leq p) \\ w^t_i \cdot \beta_t^{(1-e^t_i) \cdot \alpha} / Z_t & \text{타겟 } (p < i \leq p+q) \end{cases}$$

$$\text{여기서 } \bar{\beta}_t = \frac{\eta_t}{1 - \eta_t}, \quad \eta_t = \sum_{i=1}^{p+q} w^t_i e^t_i, \quad \beta_t = \frac{q}{p+q} + \frac{t}{(S-1)} \cdot \left(1 - \frac{q}{p+q}\right)$$

**최종 가설 선택:**

$$h_f = \arg\min_{t} \text{error}_t$$

---

### 7.2 우리 연구의 대응 수식 (Glucose-ML Transfer)

**노테이션 정의:**

| 기호 | 의미 |
|---|---|
| $X_S$ | 소스: T1D 6개 데이터셋의 전체 인스턴스 (~1,450명 × 윈도우) |
| $X_T$ | 타겟: ShanghaiT2DM (~100명 × 윈도우) |
| $\mathbf{f}_i$ | 인스턴스 $i$의 CGM 파생 피처 벡터 (Tier 2.5 v3 기준) |
| $\bar{\mathbf{f}}^T$ | 타겟 인스턴스의 평균 피처 벡터 |
| $X_{ES}$ | Importance Sampling 후 축소된 소스 ($p$개) |
| $X_{VT}$ | k-Center Sampling 후 확장된 타겟 ($q$개) |

**Importance Sampling (소스 축소):**

$$X_{ES} = \text{top-}p \left( \|\mathbf{f}^S_i - \bar{\mathbf{f}}^T\|_2 \right), \quad \mathbf{f}_i = \left[\text{W\_Mean}_i, \text{W\_Std}_i, \text{Vel}_i, \text{LBGI}_i, \text{HBGI}_i, ...\right]$$

**k-Center Sampling (타겟 확장):**

$$X_C = \text{k-Means}(X_S, k) \quad \rightarrow \quad X_{VT} = \text{nearest}(X_S, X_C) \cup X_T$$

**가중치 업데이트:** 논문과 동일한 수식 적용. 기본 학습기만 Decision Tree → **LightGBM**으로 교체.

---

### 7.3 노테이션 1:1 대응 비교표

| 구성 요소 | 논문 (S-TrAdaBoost.R2) | 우리 연구 (Glucose-ML) | 변환 이유 |
|---|---|---|---|
| **소스 단위** | UCI 데이터셋의 분할 | T1D 6개 병원/코호트 | 실제 다중 소스 환경 |
| **타겟 단위** | UCI 데이터셋의 분할 | ShanghaiT2DM | 실제 타겟 병원 |
| **피처 공간** | UCI 원본 피처 | CGM 파생 피처 (Tier 2.5 v3) | 시계열 → 정적 피처 변환 |
| **Importance Sampling 거리** | L2 (유클리드) | L2 또는 정규화 L2 | 피처 스케일 차이 고려 |
| **k-Center 군집화** | k-Means on 소스 | 동일: k-Means on 소스 | 구조 그대로 차용 |
| **기본 학습기** | Decision Tree | **LightGBM** | 결측치 처리 + 속도 + 성능 |
| **하이퍼파라미터** | S=30, F=10, α=0.1 | 초기값 동일 → Bayesian Opt | 데이터 특성에 맞춰 튜닝 |
| **후속 단계** | (없음) | **드리프트 모델 사후 진단** | PLOS ONE 논문에서 차용 |

---

## 8. 최종 결론: 이 논문에서 우리 연구에 실질적으로 가져올 수 있는 것

이 논문(S-TrAdaBoost.R2)은 현재 Tier 7 파이프라인의 TTR2를 **직접적으로 업그레이드**할 수 있는 유일한 논문입니다:

| 논문의 방법론 | 우리 연구에서의 판정 | 사유 |
|---|---|---|
| **Importance Sampling** | **✅ 핵심 도입 대상** | 14:1 소스/타겟 불균형에서 가중치 편향 교정 |
| **k-Center Sampling** | **✅ 도입 대상** | 타겟 커버리지 확장. 시계열 무결성 보존됨 |
| **비제약 가중치 업데이트** | **✅ 핵심 도입 대상** | TTR2의 2단계 동결 제거 → 일반화 개선 |
| 복잡도 지표 (CFE, DL, DI) | ⚠️ 참고 | 데이터셋 난이도 사전 진단에 활용 가능 |
| UCI 벤치마크 실험 설계 | ⚠️ 참고 | 우리 5-Way 비교의 기본 틀로 차용 |

**PLOS ONE(드리프트 모델)과의 결합:** S-TrAdaBoost.R2는 전이 학습의 **"방법"**을 개선하고, 드리프트 모델은 전이 학습의 **"결과 진단"**을 수행합니다. 두 논문을 결합하면 "Importance Sampling으로 소스를 정제 → 비제약 부스팅으로 전이 → 드리프트 GLM으로 사후 진단"이라는 **완결된 3단계 파이프라인**을 구성할 수 있습니다.

### 주의 사항 및 열린 질문

1. **$p$ (소스 선별 수)의 결정:** 논문에 가이드라인 없음. 타겟의 2~5배 범위에서 그리드 서치 필요
2. **$k$ (분산 주입 수)의 결정:** 초기값 $k = m$ (타겟 크기)으로 설정 후 실험적 조정
3. **L2 거리의 적절성:** 피처 스케일이 크게 다른 경우 정규화 또는 마할라노비스 거리 검토
4. **LightGBM과의 호환:** 논문은 단일 Decision Tree를 기본 학습기로 사용. LightGBM은 내부적으로 여러 트리를 사용하므로, S-TrAdaBoost.R2의 각 부스팅 라운드에서 LightGBM **단일 트리** 또는 **소규모 앙상블**을 사용할지 결정 필요
5. **arXiv 프리프린트 인용:** 정식 피어 리뷰 미완료 상태이므로, 논문 인용 시 "프리프린트" 명시 필요
