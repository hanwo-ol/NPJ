# [Research Application Guide] Robust Angle-based Transfer Learning 적용 방안

**논문:** "Robust angle-based transfer learning in high dimensions" (Gu et al., 2023)  
**목적:** 기존 유클리드(L2) 거리의 한계를 극복하는 '방향성(Angle) 기반' 전이 학습과, 검증 데이터가 필요 없는 다중 소스 앙상블 기법을 Glucose-ML 파이프라인(Tier 7.1)에 적용하기 위함입니다.

---

## 1. 이 논문이 왜 필요한가? (기존 방식의 맹점)

### 1.1 "경향은 맞는데 스케일이 다르다면?"
기존 선형 전이학습(예: transLASSO)은 타겟 병원의 파라미터($\beta$)와 소스 병원의 파라미터($w$) 간의 **거리(L2 Distance)**가 무조건 가까워야 한다고 강제합니다. 
하지만 병원마다 혈당 측정 기기의 캘리브레이션이나, 환자군의 기본 인슐린 저항성 베이스라인이 다르다면 어떨까요? 두 병원 환자의 "식후 혈당이 오르는 경향성(방향)"은 완벽히 똑같아도, 절대적인 수치(크기)가 달라서 L2 거리는 매우 멀게 측정될 수 있습니다. 기존 방식은 이럴 때 전이 학습을 포기하거나 모델에 엉뚱한 제약을 가하게 됩니다.

> **근거 문헌:** 본 논문(Gu et al., 2023)은 타겟과 소스 변수의 스케일이나 결과값의 정의가 다를 때 거리 기반 전이학습은 비효율적이라고 지적하며, 이를 뒷받침하기 위해 아래 문헌들을 인용합니다.
> * **Miglioretti, D. L. (2003)**: *Latent transition regression for mixed outcomes.* Biometrics, 59, 710–720.
> * **Stearns, F. W. (2010)**: *One hundred years of pleiotropy: a retrospective.* Genetics, 186, 767–773.
> * 스케일 보정을 제안한 **Liang, M. et al. (2020)** (*Learning a high-dimensional classification rule using auxiliary outcomes*, arXiv)의 연구도 존재하지만, 본 논문은 해당 방법이 소스 원본 데이터 전체를 요구한다는 한계가 있음을 지적합니다.

### 1.2 "6개 병원의 데이터를 그냥 섞어도 될까?"
우리는 1,450명의 T1D 소스 데이터를 가지고 있지만, 이들은 사실 6개의 각기 다른 코호트(병원)에서 수집되었습니다. 이들을 '하나의 큰 덩어리'로 무작정 풀링(Pooling)해서 학습시키면, 품질이 낮거나 타겟(ShanghaiT2DM)과 생리학적 성격이 완전히 다른 코호트가 섞여 전체 모델의 성능을 깎아먹을 수 있습니다.

> **근거 문헌:** 외부 데이터를 단순히 풀링할 때 발생하는 '데이터 이질성(Data Heterogeneity)'의 위험성 및 특성 차이로 인한 왜곡은 아래 문헌에서 강하게 경고된 바 있습니다.
> * **Chen, W.-C. et al. (2020)**: *Propensity score-integrated composite likelihood approach for augmenting the control arm of a randomized controlled trial by incorporating real-world data.* Journal of Biopharmaceutical Statistics.
> 
> 또한, 다기관 데이터 수집 시 병원이나 측정 시기에 따라 변수 정의와 캘리브레이션이 달라지는 현장의 위험성은 다음 문헌들을 통해 널리 제기되었습니다.
> * **Mansukhani, M. P. et al. (2019)**: *Effect of varying definitions of hypopnea on the diagnosis and clinical outcomes of sleep-disordered breathing: a systematic review and meta-analysis.* Journal of Clinical Sleep Medicine.
> * **Mitchell, B. L. et al. (2021)**: *Polygenic risk scores derived from varying definitions of depression and risk of depression.* JAMA psychiatry.

---

## 2. 핵심 해결책 1: 각도 기반 페널티 (AngleTL)

**논문의 접근:** 
파라미터 간의 유클리드(L2) 거리를 재는 대신, 두 파라미터 벡터가 가리키는 **각도(방향, Cosine Similarity)**를 계산하여 일치시킵니다. 

**임상적/기술적 의미:**
- 크기(Scale)가 달라도 혈당 변동의 방향만 맞으면 외부 지식을 적극적으로 차용합니다.
- 타겟 데이터의 자체 시그널이 강하면 소스의 영향력을 줄이고, 타겟 데이터가 부족하거나 노이즈가 많으면 소스 모델의 방향성에 크게 의존하도록 수식이 자동으로 밸런스를 조절합니다. 이 덕분에 기존 전이학습의 고질적 문제인 **Negative Transfer(외부 데이터 유입으로 인한 성능 저하)를 원천 차단**합니다.

---

## 3. 핵심 해결책 2: 다중 소스 스펙트럴 앙상블 (Algorithm 2)

**논문의 접근:**
여러 개의 소스 모델이 있을 때, 어떤 소스를 얼마나 믿을지 가중치를 정해야 합니다. 논문의 Algorithm 2는 소중한 타겟 환자의 데이터(Validation Set)를 낭비하지 않고도, 소스 모델 파라미터들의 **주성분 분석(PCA)**을 통해 최적의 합의점(Consensus) 가중치를 찾아내는 비지도(Unsupervised) 방식을 제안합니다.

**구체적 예시 (Glucose-ML 적용):**
1. 6개의 T1D 병원 데이터를 각각 따로 학습시켜 6개의 소스 모델 가중치($\hat{w}_1 \dots \hat{w}_6$)를 얻습니다.
2. 이 6개의 벡터를 모아 PCA를 돌립니다.
3. 첫 번째 주성분(First Principal Component)은 "6개 병원이 공통적으로 가장 강하게 가리키는 생리학적 방향"을 나타냅니다.
4. 이 주성분에 대한 기여도(Loadings)를 바탕으로 6개 병원의 가중치를 결정합니다. 
   - *예: 글로벌 스탠다드에 부합하는 병원 A는 가중치 0.5, 지나치게 튀는 아웃라이어 병원 B는 가중치 0.05.*
5. 이 과정을 거치면 6개 병원을 통째로 섞는 것보다 훨씬 안전하고 똑똑한 '다중 코호트 앙상블 모델'이 탄생합니다.

---

## 4. 수식 표현

### 4.1 AngleTL 목적 함수
$$ \hat{\beta}_{\lambda,\eta} = \arg \min_{\beta} \frac{1}{n}\|Y - X\beta\|_2^2 + \lambda\|\beta\|_2^2 - 2\eta \hat{w}^\top \beta $$

- $\frac{1}{n}\|Y - X\beta\|_2^2$: 타겟 데이터 예측 오차 (이 값이 작아야 함)
- $\lambda\|\beta\|_2^2$: 릿지(Ridge) 정규화 (과적합 방지)
- **$- 2\eta \hat{w}^\top \beta$ (핵심 노벨티)**: 타겟 $\beta$와 소스 $\hat{w}$의 내적. 두 벡터가 같은 방향을 가리킬수록 이 내적 값이 커져서 전체 비용 함수(Cost)를 낮춰줍니다. (거리가 아니라 각도를 일치시키도록 강제)

### 4.2 Algorithm 2 (PCA 기반 스펙트럴 가중치)
1. 6개 소스 모델 정규화: $\bar{w}_k = \frac{\hat{w}_k}{\|\hat{w}_k\|_2}$
2. 행렬 구성: $\bar{W} = [\bar{w}_1, \dots, \bar{w}_6]^\top$
3. PCA 수행: $\bar{W}\bar{W}^\top$ 행렬의 첫 번째 고유벡터(Eigenvector) $u_1$ 도출
4. 가중치 할당: $u_1$의 절대값을 가중치 $\hat{s}$로 사용해 최종 소스 모델 $\hat{w} = \sum \hat{s}_k \bar{w}_k$ 생성

---

## 5. Glucose-ML 파이프라인 도입 전략 (How to Use)

이 논문은 선형(Ridge) 모델 기반이므로 메인 엔진(LightGBM + ISTRBoost)을 완전히 대체할 수는 없습니다. 하지만 **강력한 대조군 및 서브 파이프라인 고도화** 목적으로 다음과 같이 활용합니다.

### 전략 1: SOTA(최신) 선형 전이학습 Baseline 등판
- 5-Way 비교 실험 설계에서 단순한 Target-only, Source-only 모델 대신, **이 논문의 AngleTL을 강력한 대조군으로 투입**합니다.
- "가장 진보된 형태의 선형 전이학습 모델(AngleTL)조차 우리가 만든 비선형 인스턴스 전이학습(ISTRBoost)에 미치지 못한다"는 것을 입증하여 연구의 논리적 타당성을 대폭 끌어올립니다.

### 전략 2: 6개 병원 지능형 앙상블 (Spectral Weighting)
- 1,450명 T1D 데이터를 무작정 풀링하는 기존 방식을 버리고, 위에서 설명한 Algorithm 2를 적용해 '동적 가중치 기반 다중 소스 앙상블'을 수행하는 파이프라인 분기를 생성합니다. 이를 통해 각 병원 데이터의 퀄리티와 특성을 자동으로 보정합니다.

### 전략 3: LightGBM용 Custom Loss 설계 (Knowledge Distillation)
- 파라미터 벡터($\beta$)가 존재하지 않는 트리 모델(LightGBM)의 한계를 극복하기 위해, 논문의 "방향성 보존" 철학을 **"예측값 출력의 방향성"**으로 치환하여 적용합니다.
- **Custom Loss 아이디어:** `MSE(타겟 정답, 타겟 예측)` - $\eta \times$ `상관관계(타겟 예측, 글로벌 소스 예측)`
- 이 방식을 도입하면 크기(Scale)는 타겟 데이터의 특성에 완벽히 맞추되, 혈당이 오르고 내리는 추세(Trend/Angle)는 글로벌 대규모 데이터의 지식을 철저히 따르도록 모델을 훈련시킬 수 있습니다.

---

## 6. 임상의(Doctor) 관점에서의 활용 시나리오

**상황:**
우리 연구팀이 서양인 제1형 당뇨(T1D) 환자 수천 명으로 만든 글로벌 예측 AI를, 한국의 한 노인 전문 병원(주로 제2형 당뇨)에 도입하려 합니다.

**문제:**
노인 T2D 환자들은 상대적으로 완만한 혈당 곡선을 가집니다. 글로벌 AI를 그대로 가져다 쓰면, 모델이 서양 T1D의 극단적인 특성에 맞춰져 있어 스케일이 맞지 않아 "곧 혈당이 급격히 떨어집니다!"라는 가짜 알람(False Alarm)을 남발합니다. 

**AngleTL (방향성 전이) 알고리즘이 해결하는 방식:**
의사는 AI에게 이렇게 지시하는 것과 같습니다. *"저 글로벌 AI가 혈당이 오를지 내릴지 알려주는 **'추세 방향(Trend/Angle)'**만 빼오고, 정확히 얼마나 오를지 **'절대 수치(Scale)'**는 우리 병원 환자 100명 데이터만 보고 다시 맞춰라."*

이 알고리즘은 글로벌 데이터의 강력한 패턴 감지 능력(각도)은 물려받으면서도, 절대적인 스케일은 철저히 로컬 병원(타겟)에 맞게 재조정합니다. 그 결과 의사를 피곤하게 하던 가짜 알람은 사라지고, 혈당 추세 예측의 정확도만 남아 환자 안전을 지키는 데 실질적인 도움을 줍니다.
