# [Tier 2] Classic Machine Learning (Non-Linearity & Interactions)

본 문서는 Tier 1 (Linear Regression)에서 발견된 선형 분석의 구조적 한계를 돌파하기 위한 Tier 2 (Decision Tree, Random Forest, SVM) 실험에 대한 기술적 구현 설계서입니다.

Tier 1 재설계 완료에 따라 보완된 공변량(식별 가능한 이벤트)을 활용하여, 비선형적인 상호작용(Interaction)과 혈당 역학 시계열 규칙을 포착하는 것에 주안점을 둡니다.

## User Review Required

> [!IMPORTANT]
> **SVM(Support Vector Machine)의 완전 제외 결정**
> 의료 데이터의 무결성을 지키면서 대용량 데이터를 처리하기 위해, 데이터를 모델에 맞추어 인위적으로 삭감(Sub-sampling)하는 대신 **SVM을 파이프라인에서 완전히 제외**합니다. 2,600만 개(GLAM) 스케일을 온전히 감당할 수 있는 **Decision Tree**와 **Random Forest**만을 채택함으로써 학술적 방어력을 극대화합니다.

> [!WARNING]
> **다변량 전용 실행 (Multivariate Only Run)**
> Tier 2 모형들은 근본적으로 다중 변수간 분기를 통해 이득을 얻는 모형입니다. 따라서 단변량/다변량을 전부 계산하던 Tier 1과 달리, Tier 2부터는 파이프라인 효율성을 위해 **"오직 다변량(Multivariate) 데이터 집합에 대해서만" 모형들을 학습**하도록 설계합니다 (Covariates가 없었던 세트는 단변량과 동일하게 작동합니다). 동의하시는지요? 

## Proposed Changes

### 1단계: 모델 별 설계 전략 (Model Schemas)

1. **Decision Tree Regressor (해석력 극대화)**
   - `max_depth = 5`: 시각화 및 직관적 해석이 가능하도록 트리 깊이를 제한합니다.
   - 역할: 데이터셋별 최상위 노드 피처(가장 강력한 분기 요소)를 파악하고, `Rule-based` 혈당 증감 원인을 텍스트/이미지 목적으로 추출합니다.

2. **Random Forest Regressor (성능 한계선 돌파)**
   - `n_estimators = 50, max_depth = 15, n_jobs = -1`
   - 역할: Zero-fill로 채워진 불연속 이벤트(인슐린, 식사량)를 방어적이고 가장 강력하게 소화하여 비선형 오차 감소율(%)을 수치화합니다. `feature_importances_`를 추출하여 각 그룹별로 "어떤 변수가 가장 유의미했는가"를 테이블로 도출합니다.

3. **제거됨: Support Vector Regressor (차원 확장)**
   - 대규모 윈도우 스케일링에서의 $O(n^3)$ 시간 복잡도로 인한 데이터 무결성 훼손 방지를 위해 제외.

---

### 2단계: 파일 처리 구조

#### [NEW] `C:\Users\user\Documents\NPJ2\Glucose-ML-Project\8_Classic_ML\run_classic_ml.py`
기존 Tier 1 엔진의 벡터화를 그대로 계승하고 다음 로직을 이식합니다:

- **서브샘플링 로직:** 학습(`Train_Xm`) 사이즈가 초과할 경우 `np.random.choice`를 이용해 SVM 전용으로만 한도 내로 삭감.
- **다중 모형 루프:** 데이터셋별로 (DT, RF) 순차 fit → 3가지 Metric 측정.
- **Top-3 Feature 로깅:** RF의 Feature Importance 배열을 바탕으로 (Ex. 식사시간 15%, 혈당 80%, 심박수 5%) 산술 요약을 도출해 결과로 저장.

#### [NEW] `C:\Users\user\Documents\NPJ2\Glucose-ML-Project\8_Classic_ML\1_Experimental_Plan.md`
위 설계가 확정되면 이 내용을 작업 디렉토리에 복사하여 공식 저장합니다.

## Open Questions

> [!NOTE]
> 논의에 의해 **1) Decision Tree 결과는 텍스트 분기 조건식(.txt)으로 저장**, **2) 하이퍼파라미터는 지정된 기본값 유지**로 합의하여 진행합니다.

## Verification Plan

### Automated Tests
- `run_classic_ml.py` 구동을 통한 3개 모델 벤치마크 루프 검증.
- 대형 셋(GLAM 등) 도달 시 SVM 샘플링 발동 여부 모니터링 로그 확인.

### Manual Verification
- 추출된 RF Feature Importance 중 T1DM 환자군의 식단/인슐린 의존도가 일반 셋과 어떻게 다르게 측정되는지 결과 확인.
