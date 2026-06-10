# Time-Series Preservation in Transfer Learning: 4-Model Analysis

본 보고서는 시계열 데이터 고유의 핵심 특성(시간 순서, 절대 수치 스케일, 주기성 및 인과성)을 훼손하지 않으면서 다른 환자나 이종 데이터셋의 지식을 성공적으로 전이할 수 있는 4개 핵심 모델에 대한 상세 분석 문서입니다. 

각 모델에 대한 개별 마스터 분석 본문은 아래 링크를 참고해 주십시오:
1. **TS2Vec**: [2106.10466_review_artifact.md](file:///c:/Users/user/Documents/NPJ2/Glucose-ML-Project/020_Tier_9_Temporal_Models/paper_artifact/2106.10466_review_artifact.md)
2. **CoST**: [2202.01575_review_artifact.md](file:///c:/Users/user/Documents/NPJ2/Glucose-ML-Project/020_Tier_9_Temporal_Models/paper_artifact/2202.01575_review_artifact.md)
3. **Non-stationary Transformer**: [2205.14415_review_artifact.md](file:///c:/Users/user/Documents/NPJ2/Glucose-ML-Project/020_Tier_9_Temporal_Models/paper_artifact/2205.14415_review_artifact.md)
4. **Adversarial Multi-Source Transfer Learning**: [2006.15940_review_artifact.md](file:///c:/Users/user/Documents/NPJ2/Glucose-ML-Project/020_Tier_9_Temporal_Models/paper_artifact/2006.15940_review_artifact.md)

---

## 1. 4대 모델 비교 요약 매트릭스

| 모델명 | 주요 전이 메커니즘 | 보존되는 시계열 특성 | 수치 붕괴 방지 장치 | L=3 / Edge 제약 우회 설계안 |
|---|---|---|---|---|
| **TS2Vec**<br>(arXiv:2106.10466) | 맥락적 일관성 대조 학습<br>(Contextual Consistency) | 시간 연속성, 국소 흐름, 미시적 수치 스케일 | 입력 레이어 이후 **잠재 벡터 마스킹** (원시 데이터 미훼손) | 12시간 링 버퍼 기반 대조 학습 + 실시간 L=3 비마스킹 바이패스 경로 구성 |
| **CoST**<br>(arXiv:2202.01575) | 추세-주기 물리적 분리 학습<br>(Trend-Seasonal Disentanglement) | 시간적 인과성, 위상/진폭 주기 패턴 | 인과 그래프(BSTS) 기반 분리 인코더, **주파수 진폭/위상 대조 손실 분리** | 12시간 링 버퍼 기반 Fourier 분해 + 실시간 L=3 Linear skip 및 실수형 MLP 근사 |
| **Non-stationary<br>Transformer**<br>(arXiv:2205.14415) | 가역적 인스턴스 정규화<br>(RevIN) | 정상적 변동 패턴, **환자별 절대 수치 스케일** | 입력단 Normalization 및 출력단 **De-normalization** 대칭 매핑 | 12시간 링 버퍼 기반 rolling 통계량 추정 및 표준편차 하한 바인딩 설정 |
| **Adversarial Multi-<br>Source Transfer**<br>(arXiv:2006.15940) | 경사도 반전 레이어(GRL)<br>도메인 적대적 학습 | 시간적 변동 궤적(Trajectory) | 도메인 분류기의 학습을 **예측 피처 추출과 적대 결합**해 노이즈만 소거 | 경량 1-Layer Causal FCN 및 단층 도메인 분류기 축소를 통한 CPU 연산 제어 |

---

## 2. 모델별 심층 분석

### 2.1 TS2Vec: 맥락적 일관성을 통한 미시 표상 전이
기존 시계열 대조 학습은 시점 변경(Permutation)이나 크기 스케일링(Scaling) 같은 강한 CV식 데이터 증강을 적용하여 시계열 고유의 수치적 물리 법칙과 시간 순서를 파괴하는 경향이 있었습니다.

* **시계열 보존 핵심 매커니즘**:
  * **원시 데이터 손상 없는 잠재 영역 마스킹(Timestamp Masking)**: 입력층 프로젝터(nn.Linear)를 거쳐 고차원 임베딩 벡터로 변환된 단계에서 무작위 베르누이 마스크(p=0.5)를 씌웁니다. 이로 인해 임펄스성 노이즈가 유입되어도 원시 수치 스케일은 원형 그대로 보존됩니다.
  * **랜덤 크롭 일관성(Contextual Consistency)**: 무작위로 추출한 2개의 중첩 구간에서 겹치는 부분의 시점별 특징을 일치시키도록 학습하여, 시간 궤적에 따른 데이터의 형태학적 동질성을 강건하게 전이합니다.
* **Glucose-ML 적용 방안**:
  * 환자의 과거 장기 데이터를 자가 지도 대조 학습(Self-Supervised Learning)하여 강건한 임베딩 특징을 사전 구축한 후, 신규 환자에 대해 Ridge 회귀 헤드 가중치만 고속 업데이트(Closed-form)하여 시간적 해상도를 100% 보존한 채 전이합니다.

---

### 2.2 CoST: 물리 기반 추세-주기 분리를 통한 인과 전이
시계열의 완만한 추세(Trend)와 규칙적인 주기성(Seasonal)이 얽힌 상태에서 전이를 시도하면, 특정 성분의 변동(예: 환자의 컨디션 난조로 인한 추세 시프트)이 다른 불변 성분(식사에 의한 주기적 변동)의 전이 지식까지 훼손하는 부정적 전이(Negative Transfer)를 유발합니다.

* **시계열 보존 핵심 매커니즘**:
  * **인과 분리 인코딩**: Causal Conv로 구축된 Trend Feature Disentangler(TFD)는 시간 순서 및 인과 관계를 엄밀히 보존하고, DFT와 복소수 가중치를 쓰는 Seasonal Feature Disentangler(SFD)는 시간 지연 없는 주기 성분을 포착합니다.
  * **위상 및 진폭 분리 대조**: 주파수 영역에서 진폭 $|F|$과 위상 $\phi(F)$의 대조 학습 손실을 각각 독립 부과하여, 특정 대사 신호가 지닌 위상차(인슐린 투여 후 약효가 나타나는 시차)와 진폭(혈당 상승 높이)의 고유한 주파수 성질을 파괴하지 않고 모델에 전이합니다.
* **Glucose-ML 적용 방안**:
  * 환자의 일일 주기 혈당 반응(Seasonal)과 장기 공변량 변동(Trend)을 별도 분리 인코딩하도록 백본을 고정해 두고, 실시간 추론 시에는 단말에서 무거운 복소수 연산을 피하기 위해 실수 진폭 기반의 MLP로 간소화하여 전이 효율을 극대화합니다.

---

### 2.3 Non-stationary Transformer: 역정규화를 통한 스케일 특성 복원
시계열 데이터셋은 수집된 기관이나 환자 개인에 따라 혈당의 평균 및 진폭 범위(분산)가 극도로 다릅니다. 이 비정상성(Non-stationarity)을 제거하지 않으면 전이학습 시 모델이 훈련 셋의 특정 범위만 암기하여 타겟 환자에게 적용할 수 없게 되며, 무작위 정규화를 해버리면 혈당 임상 한계치(예: 70mg/dL 미만 저혈당, 180mg/dL 초과 고혈당) 같은 절대적 경계 특성이 소실됩니다.

* **시계열 보존 핵심 매커니즘**:
  * **가역적 인스턴스 정규화(RevIN)**: 인코더 입력단에서 각 시퀀스의 평균 $\mu$와 표준편차 $\sigma$를 제거하여 패턴만 전달하므로, 모델은 절대 수치에 방해받지 않고 '변동 추이' 정보만 학습하여 전이할 수 있습니다.
  * **예측단 역정규화(De-normalization)**: 모델의 최종 출력 벡터에 대해, 입력단에서 제거했던 타겟 환자 고유의 $\mu$와 $\sigma$를 다시 대칭곱 및 가산해 줍니다. 이를 통해 타 환자로부터 전이받은 변동 형태는 유지하면서, 최종 예측 수치는 정확히 타겟 환자의 혈당 스케일 상에 정합시킵니다.
* **Glucose-ML 적용 방안**:
  * 3스텝 룩백($L=3$) 하에서 trimmed 통계량 계산 시 발생하는 분모 제로화 및 수치 발산을 막기 위해, 12시간 단말 링 버퍼 기반의 통계량을 백바이어스로 융합하여 안전한 복원 연산을 보증합니다.

---

### 2.4 Adversarial Multi-Source Transfer: 적대적 특성 소거를 통한 궤적 보존
다중 소스 환자 데이터로 일반화 모델을 구축할 때, 단순 지도 학습은 모델이 소스 환자의 특정 노이즈 패턴이나 센서 바이어스까지 암기하게 만들어 미지의 신규 타겟 환자에 대한 일반화 성능을 급격히 떨어뜨립니다.

* **시계열 보존 핵심 매커니즘**:
  * **환자 식별성 정보의 표상 소거(Patient-Invariant Feature Learning)**: 피처 추출기와 환자 도메인 분류기 사이에 GRL(Gradient Reversal Layer)을 두어, 역전파 시 분류기 손실 함수 부호를 반전시킵니다. 이로 인해 피처 인코더는 '환자를 구별하게 만드는 지표'는 전면 삭제하고, 시계열의 순수한 생리적 공통 궤적(Temporal Trajectory) 정보만 보존하여 전이 가능한 공간을 만듭니다.
* **Glucose-ML 적용 방안**:
  * LODO(Leave-One-Dataset-Out) 크로스 밸리데이션 학습 중 모델이 특정 환자의 대사 속도 바이어스를 강하게 학습하는 것을 방지하여 제로샷 일반화 성능을 확보하되, 모바일 단말 CPU 학습 연산 오버헤드를 막기 위해 도메인 분류기는 단층 선형 레이어로 극소화하여 탑재합니다.
