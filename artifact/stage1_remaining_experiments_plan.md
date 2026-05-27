# Stage 1 잔여 실험 계획서: CGM AI의 일반화 가능성 및 재현성 검증

본 문서는 Glucose CGM 연구 프로젝트의 Stage 1 잔여 실험에 대한 상세 계획, 수학적 정의, 실행 프로토콜을 수립한다. 본 실험들은 리뷰어 피드백 대응 및 *npj Digital Medicine* 게재 기준 충족을 목적으로 설계되었다.

## 0. 이전 실험 요약 및 본 실험의 배경

### 선행 실험 결과
Stage 1의 선행 실험(Tier 7, 7.1)에서는 LightGBM 기반 5-Way 비교(Source-Only, Target-Only, Mixed, CORAL, TrAdaBoost)를 통해 T1D 소스에서 T2D/Mixed 타겟으로의 교차 질환 전이를 수행하였다. CORAL 및 TrAdaBoost가 negative transfer를 해소하여 Target-Only 수준의 성능을 회복하였으나, Target-Only를 유의미하게 초과하지는 못하였다. 이 결과는 "정적 머신러닝 기반 도메인 적응이 CGM 혈당 예측에서 충분한 전이 이득을 제공하지 못한다"는 것을 시사하지만, 아직 이것이 우연(시드/하이퍼파라미터에 의한 변동)인지 구조적 한계인지 구분되지 않았다.

### 본 실험의 필요성
본 문서의 5개 잔여 실험은 위 결과가 단순 우연이 아닌 **정적 모델의 구조적 한계**임을 다각도로 입증하기 위해 설계되었다. 각 실험의 논증 역할은 다음과 같다:

| 실험 | 논증 역할 |
|---|---|
| S1-1 (Within Variation) | 전이학습이 Target-Only를 초과하지 못한다는 결과가 시드/하이퍼파라미터를 바꿔도 일관되게 재현되는지 확인하여, 이 성능 천장이 우연이 아닌 안정적 현상임을 입증 |
| S1-2 (LODO) | 성능 한계가 특정 코호트 조합에서만 발생하는 것이 아니라, 학습에 포함되지 않은 새로운 코호트에 모델을 적용할 때 전반적으로 발생함을 확인 |
| S1-3 (도메인 거리) | 성능 저하가 소스-타겟 간 분포 거리와 상관함을 정량적으로 입증 |
| S1-4 (시계열 한계) | 잔차에 시간적 자기상관이 남아있음을 증명하여, 정적 모델이 시간 의존성을 구조적으로 포착하지 못함을 입증 |
| S1-5 (분류 전환) | 회귀 수준의 한계가 임상적 분류 결론에서도 동일하게 나타나는지 확인 |

특히 **S1-4는 Stage 1과 Stage 2를 잇는 브릿지 실험**이다. 정적 모델의 예측 잔차에 유의한 시간적 자기상관이 존재한다면, 이는 CORAL/TrAdaBoost가 데이터를 i.i.d.로 가정하여 혈당의 시간적 관성(momentum)을 구조적으로 무시하고 있다는 정량적 증거가 된다. 이 증거가 Stage 2에서 시계열 아키텍처(RNN/Transformer 기반)로의 전환을 정당화하는 핵심 근거이다.

상세 내용은 다음 문서를 참조한다:
* 전체 연구 설계: [Stage 1/2 연구 설계서](stage1_stage2_research_design.md)
* 교수님 미팅 피드백: [연구 토의 내용](연구%20토의%20내용.md)
* 선행 실험 코드 및 결과: [Tier 7 교차 질환 전이](../Glucose-ML-Project/016_Tier_7_Cross_Disease/), [Tier 7.1 임상 안전성](../Glucose-ML-Project/017_Tier_7.1_Clinical_Transfer/)

---

## 1. S1-1: Within Variation (계산적 재현성)

### 1.1 목적
동일한 데이터 구성에서 난수 시드와 하이퍼파라미터 설정을 변경했을 때, 모델 성능과 임상적 결론이 얼마나 변동하는지 정량화한다.

### 1.2 프로토콜
* **고정 데이터**: 소스 풀(T1D 풀)과 타겟 코호트(ShanghaiT2DM).
* **기본 모델**: LightGBM.
* **시드 스위프**: 10개 난수 시드 (42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999).
* **하이퍼파라미터 스위프 (선택적 확장)**:
  * 학습률: 0.01, 0.05, 0.1
  * 리프 노드 수: 31, 63, 127
  * 최대 트리 깊이: 무제한, 6, 10
* **평가 구조**: Source-Only, Target-Only, Mixed, CORAL, TrAdaBoost로 구성된 5-Way 비교.

### 1.3 평가 지표

#### 성능 변동성
* **변동계수 (CV)**:
  $$\text{CV} = \frac{\sigma_{\text{RMSE}}}{\mu_{\text{RMSE}}}$$
* **사분위 범위 (IQR)**: 시드 간 RMSE 분포의 사분위 범위.

#### 결론 안정성
* **뒤집힘 빈도 (Flip Rate, FR)**: "전이학습이 Target-Only보다 우수하다"는 임상적 결론이 뒤집히는 비율:
  $$\text{FR} = \frac{\sum_{s} \mathbb{I}(\text{RMSE}_{\text{TL}, s} > \text{RMSE}_{\text{Target\_Only}, s})}{\text{Total Seeds}}$$
* **순위 안정성 (Kendall's $\tau$)**: 시드 간 5개 모델 순위의 쌍별 순위 상관계수.

#### 임상적 결론 안정성
* **Clarke Error Grid Zone A 변동성**: 10개 시드에 걸쳐 Zone A 비율의 범위(최대 - 최소)와 표준편차. 모델별로 산출.
* **저혈당 민감도 변동성**: 10개 시드에 걸쳐 저혈당 민감도(70 mg/dL 미만 검출률)의 범위와 표준편차. 모델별로 산출.

---

## 2. S1-2: Leave-One-Dataset-Out (LODO) 일반화 가능성

### 2.1 목적
학습 과정에서 완전히 배제된 새로운 임상 기관/사이트에 CGM AI를 배포할 때 발생하는 교차 도메인 일반화 손실을 평가한다.

### 2.2 프로토콜
프로젝트의 샘플링 주기 분리 원칙(AGENTS.md)에 따라, 26개 활성 데이터셋(997_Active_Datasets.md)을 3개 주기 그룹으로 분리하고, 각 그룹의 LODO를 독립적으로 수행한다. 서로 다른 샘플링 주기의 데이터셋은 동일 모델에 혼합하지 않는다.

#### 1분 주기 그룹 (2개 데이터셋)
* **Mixed**: CGMacros_Dexcom, CGMacros_Libre.

#### 5분 주기 그룹 (21개 데이터셋)
* **T1D (15개)**: AIDET1D, AZT1D, D1NAMO, HUPA-UCM, IOBP2, PEDAP, PhysioCGM, T1D-UOM, UCHTT1DM, RT-CGM, SENCE, WISDM, FLAIR, SHD, ReplaceBG.
* **ND (3개)**: BIGIDEAs, CGMND, GLAM.
* **Mixed (3개)**: Colas_2019, CITY, Hall_2018.

#### 15분 주기 그룹 (3개 데이터셋)
* **T1D**: Bris-T1D_Open, ShanghaiT1DM.
* **T2D**: ShanghaiT2DM.

각 주기 그룹 내의 타겟 데이터셋 $D_i$에 대해 다음을 수행한다:
1. **학습 세트**: 동일 주기 그룹 내 $D_i$를 제외한 나머지 데이터셋을 합산.
2. **테스트 세트**: $D_i$의 테스트 파티션 (피험자 단위 15% 분할, Rule 5).
3. **모델**: Source-Only, Target-Only ($D_i$의 학습 분할로만 학습), CORAL, TrAdaBoost.

### 2.3 주의사항
* **소스 풀 서브샘플링**: 합산된 소스 풀이 200만 윈도우를 초과할 경우, 데이터셋 비례 서브샘플링을 적용하여 코호트 다양성을 유지하면서 학습 시간을 단축한다.
* **1분 그룹 제한사항**: CGMacros_Dexcom과 CGMacros_Libre는 원본 5분/15분 센서 데이터를 리샘플링하여 생성된 데이터이다. 따라서 이 그룹의 LODO는 교차 사이트 일반화 검증이 아닌 내부 일관성 점검(internal consistency check)의 성격을 가진다. 이 제한사항은 논문에 반드시 명시한다.

### 2.4 평가 지표
* 각 타겟에 대한 LODO RMSE, MAE, MARD.
* 주기 그룹별 $N \times M$ 행렬(타겟 데이터셋 vs. 전이 방법) 히트맵.

---

## 3. S1-3: 도메인 거리 vs. 성능 저하 상관 분석

### 3.1 목적
소스와 타겟 피처 분포 간의 수학적 거리가, 배포 전에 전이 성능 저하를 예측하는 대리 지표(proxy)로 활용 가능한지 검증한다.

### 3.2 프로토콜
LODO 분할에서 생성된 각 (소스 풀, 타겟 데이터셋) 쌍에 대해, 22차원 피처 공간에서 다음 거리 지표를 산출한다:

1. **Maximum Mean Discrepancy (MMD)**:
   $$\text{MMD}^2(X_S, X_T) = \frac{1}{n_s^2} \sum_{i,j} k(x_i^s, x_j^s) - \frac{2}{n_s n_t} \sum_{i,j} k(x_i^s, x_j^t) + \frac{1}{n_t^2} \sum_{i,j} k(x_i^t, x_j^t)$$
   RBF 커널 $k(x, y) = \exp(-\gamma ||x-y||^2)$을 사용한다.
2. **Proxy-A-Distance (PAD)**:
   소스와 타겟 샘플을 구분하는 선형 분류기를 학습하여, 분류기 오류율이 $\epsilon$일 때:
   $$\text{PAD} = 2(1 - 2\epsilon)$$
3. **공분산 프로베니우스 노름**:
   $$\text{Dist}_{\text{Cov}} = ||\Sigma_S - \Sigma_T||_F$$
4. **Wasserstein Distance**: 소스와 타겟 분포 간 최적 수송 비용을 산출한다.

### 3.3 상관 분석
* 선형 회귀 수행: $\Delta \text{RMSE} = \alpha + \beta \cdot \text{Distance}$ (여기서 $\Delta \text{RMSE} = \text{LODO\_RMSE} - \text{Self\_RMSE}$).
* Spearman의 $\rho$, Pearson의 $r$, 결정계수 $R^2$를 보고한다.

---

## 4. S1-4: 정적 모델의 시계열 한계 식별

### 4.1 S1-4a: 잔차 자기상관 (ACF) 분석
* **핵심 논리**: 모델이 시계열 역학을 성공적으로 포착했다면, 예측 잔차는 백색 잡음(white noise)에 가까워야 한다(즉, 시간적 상관 없음). 잔차에 유의한 자기상관이 존재한다면, 이는 모델이 포착하지 못한 시간적 구조가 남아있다는 정량적 증거이다.
* **수식**:
  각 모델 $M$과 타겟 $T$에 대해, 환자별 시간순 잔차 시퀀스 $e_t = y_t - \hat{y}_t$를 추출한다. 시차 $k = 1, 2, \dots, 12$에 대해 ACF를 계산한다:
  $$\text{ACF}(k) = \frac{\sum_{t=k+1}^N (e_t - \bar{e})(e_{t-k} - \bar{e})}{\sum_{t=1}^N (e_t - \bar{e})^2}$$
* **통계 검정**:
  유의수준 $\alpha = 0.01$에서 Ljung-Box Q 검정을 수행한다:
  $$Q = N(N+2) \sum_{k=1}^m \frac{\hat{\rho}_k^2}{N-k}$$
  $p < 0.01$이면 귀무가설 $H_0$(잔차가 i.i.d.)을 기각한다. 이는 잔차에 시간적 자기상관이 존재함을 의미한다.
* **요약 지표**: ACF(lag=1) 값 및 Durbin-Watson 통계량.

### 4.2 S1-4b: 구간별 오차 분해 (Segment-wise Error Decomposition)
* **핵심 논리**: 정적 전이학습 모델이 혈당 급변 구간에서 특히 취약한지 분석한다.
* **수식**:
  샘플링 스텝 단위의 혈당 변화 속도(velocity)를 산출한다:
  $$v_t = \frac{g_t - g_{t-1}}{\Delta t}$$
  여기서 $\Delta t$는 1 스텝(5분 그룹에서는 5분, 15분 그룹에서는 15분)이다. $v_t$의 단위는 **mg/dL/step**이며, mg/dL/min이 아니다. 스텝 기반 정의를 사용함으로써 주기 그룹 간에 동일한 임계값으로 일관된 구간 분류가 가능하다.
* **구간 정의** (mg/dL/step 기준):
  * **안정 구간**: $|v_t| \le 1.0$ mg/dL/step
  * **급상승 구간**: $v_t > 2.0$ mg/dL/step
  * **급하강 구간**: $v_t < -2.0$ mg/dL/step
  * **전이 구간**: $1.0 < |v_t| \le 2.0$ mg/dL/step
* **물리 시간 환산**: 5분 그룹에서 2.0 mg/dL/step = 0.4 mg/dL/min. 15분 그룹에서 2.0 mg/dL/step = 0.13 mg/dL/min.
* **평가 지표**: 각 구간 내에서 Target-Only 대비 CORAL/TrAdaBoost의 RMSE를 비교한다.

---

## 5. S1-5: 회귀 예측의 3분류 전환

### 5.1 목적
연속 혈당 예측값을 이산적 임상 분류로 전환하여, 범주형 결과에 대한 재현성을 분석한다 (교수님 미팅 피드백 반영).

### 5.2 분류 기준 (임상 표준)
* **저혈당 (Hypoglycemia)**: CGM < 70 mg/dL
* **정상 범위 (In-Range)**: 70 <= CGM <= 180 mg/dL
* **고혈당 (Hyperglycemia)**: CGM > 180 mg/dL

### 5.3 분류 평가 지표
* **정확도 (Accuracy)**: 전체 분류 정확률.
* **Cohen's Kappa ($\kappa$)**: 평가자 간 일치도의 표준 측도. 3개 연구 그룹(영상, 테이블, 시계열) 공통 재현성 지표로 사용한다.
* **Macro-averaged F1-score**.
* **AUC-ROC (one-vs-rest)**: 3분류 문제에 대한 클래스별 AUC.
* **저혈당/고혈당 특이적 민감도 및 특이도**.

### 5.4 클래스 불균형 고려사항
CGM 혈당 값은 3개 클래스 간 극심한 불균형을 보인다: 저혈당은 전체 샘플의 약 1~5%, 정상 범위는 80~90%, 고혈당은 5~15%를 차지한다. 이에 대해 다음과 같이 대응한다:
* 클래스별 지표를 매크로 평균 지표와 함께 보고한다.
* 정상 범위 클래스의 압도적 비율로 인한 정확도 과대평가를 방지하기 위해, 가중 F1-score 및 가중 Kappa 변형을 고려한다.

### 5.5 S1-1과의 결합: Kappa 시드 분포 분석
3분류 전환 평가는 Within Variation 실험(S1-1)과 결합하여 수행한다. 10개 난수 시드와 5개 모델 각각에 대해:
1. 이산화된 예측값으로부터 Cohen's Kappa를 산출한다.
2. 모델별 10개 시드 Kappa 분포를 수집한다.
3. Kappa의 평균, 표준편차, 범위를 보고한다.
4. 핵심 질문: "Kappa는 시드에 따라 RMSE보다 더 흔들리는가, 덜 흔들리는가?" Kappa의 CV가 RMSE의 CV보다 높다면, 임상적 분류 결론이 회귀 수준의 결론보다 불안정함을 의미한다.

---

## 6. 실행 파이프라인

잔여 실험은 Tier 8 (재현성) 디렉터리 내에 독립 모듈로 구현한다. 각 모듈은 하나의 실험에 대응하며, 자기완결적 출력 파일을 생성한다.

| 모듈 | 실험 | 주요 산출물 |
|---|---|---|
| Within Variation | S1-1, S1-5 | RMSE 분포(평균, 표준편차, CV, 뒤집힘 빈도), Clarke Zone A 변동성, 저혈당 민감도 변동성, 모델별 Kappa 분포 |
| LODO | S1-2 | 주기 그룹별 RMSE/MAE/MARD 히트맵 (타겟 데이터셋 vs. 전이 방법) |
| 도메인 거리 분석 | S1-3 | 쌍별 MMD, PAD, 공분산 프로베니우스 노름; 성능 저하와의 Spearman/Pearson 상관 |
| 시계열 한계 분석 | S1-4 | ACF 플롯, Ljung-Box p-value, Durbin-Watson 통계량; 구간별 RMSE 그룹 막대 차트 |

---

## 7. 일정

* **1주차**: Within Variation 실험 (S1-1) 및 3분류 전환 (S1-5) 구현.
* **2주차**: LODO 실험 (S1-2) 및 잔차 시계열 한계 분석 (S1-4) 구현.
* **3주차**: 도메인 거리 상관 분석 (S1-3) 구현 및 회귀선 산점도 생성.
* **4주차**: 전체 지표 통합, violin chart, heatmap, 회귀 산점도 등 논문용 Figure 초안 작성.
