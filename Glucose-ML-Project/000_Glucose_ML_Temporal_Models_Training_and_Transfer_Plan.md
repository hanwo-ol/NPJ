# 시계열 Temporal 모델 본학습 및 전이학습 수행 계획서 (문서 3)
**— 9종 모델별 LODO 사전 학습 프로토콜 및 타겟 미세 조정 전략 —**

본 계획서는 [000_Glucose_ML_Temporal_Models_Common_Specification.md](file:///c:/Users/user/Documents/NPJ2/Glucose-ML-Project/000_Glucose_ML_Temporal_Models_Common_Specification.md)에 기술된 9대 시계열 모델(DLinear, N-BEATS, N-HiTS, TSMixer, TiDE, FITS, PatchMLP, XLinear, SOFTS)의 훈련, 평가, 다기관 전이학습(Transfer Learning)을 진행하기 위한 구체적인 수행 절차와 설계 사항을 정의합니다.

---

## 1. 본학습 (Main Training) 수행 계획

개별 기관 데이터셋 단위로 기본 예측 베이스라인 성능을 도출하기 위한 표준 최적화 루프입니다.

### 1.1. 데이터 구성 및 배치 파이프라인
- **데이터셋 분할:** [000_Glucose_ML_Preprocessing_Pipeline.md](file:///c:/Users/user/Documents/NPJ2/Glucose-ML-Project/000_Glucose_ML_Preprocessing_Pipeline.md)의 시계열 무결성 규칙(Rule 5)에 따라, 전체 피험자 식별자(Subject ID)를 고정 시드(SEED=42)로 무작위 셔플한 뒤 Train 70%, Val 15%, Test 15% 비율로 분배합니다. 
- **셔플 정책:** 개별 환자 내부의 시계열 시간 순서는 보존(환자별 정렬)하여 데이터 로더에 피드합니다. 훈련(Train) DataLoader는 미니배치 단위 셔플을 켜며, 검증(Val) 및 평가(Test) 로더는 셔플링을 배제하여 안정적인 정적 검증을 보장합니다.
- **연산 입력 규격:** [data_loader.py](file:///c:/Users/user/Documents/NPJ2/Glucose-ML-Project/021_Tier_9.1_Temporal_Models/data_loader.py)를 통해 캐시로부터 추출한 `[Batch_Size, LOOKBACK_STEPS=3, Input_Channels]` 크기의 3D 시계열 텐서를 입력으로 주입합니다.

### 1.2. 최적화 및 훈련 루프
- **손실 함수 (Loss Function):** MAE(L1 Loss) 및 MSE(L2 Loss)를 표준 적용합니다. 추가로 저혈당 예측 신뢰성을 담보하기 위해 저혈당 대역 페널티 손실함수를 융합 가동합니다.
- **최적화 알고리즘:** 가중치 감쇄를 포함한 AdamW 옵티마이저를 사용합니다.
  - 초기 학습률 (Learning Rate): $1 \times 10^{-3}$
  - 가중치 감쇄 (Weight Decay): $1 \times 10^{-4}$
- **정상성 유지:** Reversible Instance Normalization (RevIN) 모듈을 활성화하여 훈련 중 비정상성 통계에 따른 경사도 폭주(Gradient Exploding)를 예방합니다.
- **과적합 제어:** 검증 세트의 MAE 손실을 매 에포크마다 추적하여, 10회 연속으로 손실 개선이 이루어지지 않을 경우 조기 종료(Early Stopping, patience=10)를 작동시킵니다.

---

## 2. 전이학습 (Transfer Learning) 수행 계획

다기관/다코호트 이질성을 수용하고 모델의 범용 일반화 성능을 높이기 위한 LODO 기반 전이학습 로직 설계입니다.

### 2.1. LODO Pre-training (사전 학습 단계)
1. **타겟 제외 그룹 풀링:** 특정 타겟 데이터셋 $D_{Target}$을 평가할 때, 해당 데이터셋이 포함된 동일 주기 그룹(예: 5분 주기 그룹)에 속한 나머지 모든 소스 데이터셋 $D_{Source}$의 Train 세트(70%)를 일괄 병합합니다.
2. **사전 학습 최적화:** 병합된 대형 사전 학습 데이터 풀을 사용해 시계열 모델의 초기 글로벌 가중치 $\mathbf{\Theta}_{Global}$을 수렴될 때까지 학습합니다. (최대 100 에포크 가동, 데이터셋 비례 샘플링을 적용해 학습 자원 균형 조절).
3. **가중치 동결 저장:** 학습이 완료된 최적 가중치 파일 `global_pretrain_[ModelName].pth`를 결과 디렉터리에 아티팩트로 저장합니다.

### 2.2. Zero-shot Transfer (제로샷 평가 단계)
- 사전 학습된 모델 가중치 $\mathbf{\Theta}_{Global}$을 그대로 로드하여, 타겟 데이터셋 $D_{Target}$의 평가(Test, 15%) 세트에 직접 추론을 실행합니다. 
- 미세 조정 없이도 다른 기관 데이터로 학습된 글로벌 모델이 낯선 코호트에서 얼마나 높은 일반화 견고성을 발휘하는지 RMSE, MAE, MARD 및 Clarke Error Grid 구역 비율을 측정해 기록합니다.

### 2.3. Fine-tuning & Adaptation (미세 조정 단계)
- 사전 학습 가중치 $\mathbf{\Theta}_{Global}$을 타겟 데이터셋 $D_{Target}$의 Train 세트(70%) 데이터를 활용하여 추가 미세 조정을 수행합니다.
- **학습률 제어:** 사전 학습 지식을 보존하고 가중치 파괴를 방지하기 위해 Fine-tuning 단계에서의 학습률은 사전 학습 시의 1/10 수준인 $1 \times 10^{-4}$로 낮춰 설정합니다.
- 에포크 수를 10~20회 내외로 제한하고 Early Stopping(patience=3)을 민감하게 적용하여 타겟 소규모 집단으로의 급격한 오버피팅을 방지합니다.

---

## 3. 모델군별 전이학습 어댑테이션 전략

모델의 구조적 속성에 맞춰 사전 학습 가중치 동결 및 파인튜닝 가중치 마스킹을 다르게 적용합니다.

```
       [사전 학습 글로벌 모델 가중치]
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│   가중치 동결    │      │   미세 조정      │
│ (Feature/Encoder)│      │  (Fusion/Decoder)│
└──────────────────┘      └──────────────────┘
```

### 3.1. Rank 1: Linear & Mixer 계열 (DLinear, TSMixer, XLinear)
- **DLinear / XLinear:** 
  - 파라미터 개수가 극히 적고 가중치 자체가 시계열 선형 트렌드 궤적을 투영합니다.
  - 사전 학습 모델의 파라미터 전체를 파인튜닝하는 **전면 미세 조정(Full Fine-tuning)** 전략을 취해도 오버피팅 위험이 낮으며, 빠른 수렴을 확보할 수 있습니다.
- **TSMixer:**
  - **전략:** 시간 축 결합 가중치는 일반적 시계열 주기성을 기억하므로 동결(Freeze)하고, 공변량 결합을 담당하는 Feature-Mixing MLP와 최종 출력 사영 레이어(`time_project`, `projection`)만 선택적으로 업데이트하는 **부분 미세 조정(Partial Tuning)**을 적용합니다.

### 3.2. Rank 2: Basis & Seq2Seq 계열 (TiDE, N-HiTS, N-BEATS)
- **TiDE:**
  - **전략:** 과거 입력을 인코딩하는 Encoder MLP 영역은 소스 데이터의 풍부한 시계열 표현을 담고 있으므로 완전히 동결(Freeze)합니다. 반면, 미래 예측 영역을 생성하는 Decoder MLP 및 공변량 사영 레이어와 최종 Linear Residual Connection 가중치만 Fine-tuning 시 업데이트(Adaptation)합니다.
- **N-HiTS / N-BEATS:**
  - **전략:** 다중 계층 스택 구조를 활용합니다. 장기적이고 거친 물리 패턴(저주파 트렌드)을 해석하는 하위 블록들은 동결하고, 식후 요동이나 단기 노이즈 보정을 타겟으로 하는 상위 고주파 및 잔차 보정 블록들만 파인튜닝 시 활성화하여 학습을 최적화합니다.

### 3.3. Rank 3: Frequency & Segment 계열 (FITS, SOFTS, PatchMLP)
- **FITS:**
  - 주파수 영역 복소 선형 가중치가 10K 내외로 매우 컴팩트합니다. 따라서 별도의 동결 없이 전체 주파수 보간 레이어를 동시에 조정하는 **전면 미세 조정(Full Fine-tuning)**을 이행합니다.
- **SOFTS:**
  - **전략:** 스타 토폴로지 내 전역 채널 정보를 집약하는 글로벌 코어(Global MLP Mixer) 가중치는 사전 학습의 일반적 물리 상호작용 지식을 담고 있으므로 동결합니다. 개별 채널의 로컬 인코더 및 채널 융합 레이어(`fusion`)의 파라미터만 미세 조정합니다.
- **PatchMLP:**
  - **전략:** 로컬 시간 의미를 분할하는 패치 임베딩 레이어(`patch_embed`)는 동결하여 시계열 패치 토큰화 지식을 그대로 재활용하고, 패치 간 결합을 연산하는 `patch_mixing` 및 `hidden_mixing` 파라미터만 파인튜닝 단계에서 훈련합니다.

---

## 4. 모델 전이학습 평가 지표 및 정합성 기준

전이학습 실행 시 각 모델의 예측 성능 평가는 다음 3단계 벤치마크 지표를 비교 분석하여 종합 우위를 판단합니다.

1. **Self-Adaptation Ratio (자가 학습 대비 비율):** 
   - $$\text{SAR} = \frac{\text{Fine-tuned Target RMSE}}{\text{Self-trained Target RMSE}}$$
   - SAR 수치가 1.0 미만일 경우, 글로벌 사전 학습 정보를 활용해 단일 기관 데이터만 썼을 때보다 예측 오차가 통계적으로 유의미하게 향상되었음을 검증하는 일차 기준이 됩니다.
2. **Zero-shot Transfer Loss:** 
   - 파인튜닝 전 글로벌 가중치 그대로 평가했을 때 타겟 데이터셋에서 발생하는 MAE/RMSE를 측정하여, 코호트 간 장비 및 기기 이질성 하에서의 모델 기본 일반화 견고성을 파악합니다.
3. **임상 범위 Zone 정합성:**
   - Clarke Error Grid 구역 매핑을 통해 Zone A 및 Zone B에 속하는 샘플 비율이 전이학습 적용 전후로 어떻게 개선되는지 비교하고, Zone A+B 비율이 98% 이상을 안정적으로 수렴하는지 통계 검증합니다.
