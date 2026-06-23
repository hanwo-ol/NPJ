# 다기관 CGM 시계열 예측 Temporal 모델 패밀리 공통 설계서
**— DLinear, N-BEATS, N-HiTS, TSMixer, TiDE, FITS, PatchMLP, XLinear, SOFTS 모델의 통합 아키텍처 및 구현 명세 —**

본 설계서는 [000_Glucose_ML_Preprocessing_Pipeline.md](file:///c:/Users/user/Documents/NPJ2/Glucose-ML-Project/000_Glucose_ML_Preprocessing_Pipeline.md)의 엄격한 전처리 파이프라인 및 피처 사양을 준수하며, 최신 시계열 예측 기계학습 모델 군(MLP 및 Linear 계열 9종)을 다기관 연속혈당측정(CGM) 데이터에 일관되게 적용하기 위한 공통 인터페이스, 데이터 토폴로지, 모듈러 아키텍처 및 구현 명세를 정의합니다.

---

## 1. 공통 데이터 인터페이스 사양

모든 Temporal 모델은 데이터 로딩 단계에서 생성되는 동일한 텐서 규격과 피처 차원을 공유하여 공정한 벤치마크 및 플러그앤플레이(Plug-and-Play) 형태의 모델 교체가 가능해야 합니다.

### 1.1. 입력 텐서 규격 (Input Tensor Shape)
- **형태:** `[Batch_Size, Lookback_Steps, Input_Channels]`
- **룩백 스텝 (Lookback_Steps):** 
  - 기본값: 3스텝 (단기 예측, 약 15분 관측 버퍼).
  - 장기 패턴 학습용: 144스텝 (12시간 버퍼, 5분 주기 센서 기준).
- **입력 채널 (Input_Channels):**
  - **단변량 모드 (Single Channel):** 혈당 단독 (`glucose_value_mg_dl`) $\rightarrow 1$ 채널.
  - **공통 피처 모드 (Global Features):** 동역학/임상/위상 파생 변수 14종 결합 $\rightarrow 14$ 채널.
  - **다변량 외생 변수 모드 (Global + Local):** 공통 피처 14종 + 각 코호트 고유 채널(인슐린, 식사 탄수화물, 심박수, 걸음 수 등) 조합 $\rightarrow 14 + K$ 채널.

### 1.2. 출력 텐서 규격 (Output Tensor Shape)
- **형태:** `[Batch_Size, Horizon_Steps, Target_Channels]`
- **예측 지평선 (Horizon_Steps):**
  - 기본값: 3스텝 (5분 주기 기준 15분 후 예측, 15분 주기 기준 45분 후 예측).
  - 중장기 타겟: 12스텝 (5분 주기 기준 1시간 후 예측).
- **대상 채널 (Target_Channels):**
  - 최종 예측 목표는 미래 시점의 혈당 실측값 단독이므로 $1$ 채널 (`glucose_value_mg_dl`)로 고정합니다.

---

## 2. 공통 전처리 및 정상화 모듈 (RevIN Block)

CGM 시계열 데이터의 장기 추세 변동(Concept Drift) 및 환자/기기 간 스케일 차이에 대응하기 위해, 모든 모델의 입출력 단에 **가역적 인스턴스 정규화 (Reversible Instance Normalization, RevIN)** 레이어를 공통 장착합니다.

```
       [입력 시퀀스: X] 
              │
              ▼
    ┌───────────────────┐
    │     RevIN.mean    │ ─── 평균(mu) 추출
    │     RevIN.std     │ ─── 표준편차(sigma) 추출
    └───────────────────┘
              │
              ▼  정상성 텐서(Normalised X)
    ┌───────────────────┐
    │   Predictive Model│ (DLinear, TSMixer, PatchMLP 등 9종 중 택1)
    └───────────────────┘
              │
              ▼  정상성 예측 텐서(Normalised Y_pred)
    ┌───────────────────┐
    │ RevIN.denormalise │ ◄── 평균(mu) 및 표준편차(sigma) 재주입
    └───────────────────┘
              │
              ▼
    [최종 예측 혈당: Y_pred (mg/dL)]
```

### 2.1. 정규화 공식
$$\hat{X} = \frac{X - \text{mean}(X, \text{axis}=1)}{\sqrt{\text{var}(X, \text{axis}=1) + \epsilon}}$$
- 입력 데이터 $X$를 시간 축에 대해 인스턴스 정규화하여 데이터 내의 정상성(Stationarity) 정보만 모델 학습에 투입합니다.

### 2.2. 역정규화 공식
$$\hat{Y}_{final} = \hat{Y}_{model} \cdot \sqrt{\text{var}(X, \text{axis}=1) + \epsilon} + \text{mean}(X, \text{axis}=1)$$
- 모델이 예측한 정규화 상태의 출력값 $\hat{Y}_{model}$에 입력 시점의 스케일($\mu, \sigma$)을 다시 주입하여 물리 혈당 단위계($\text{mg/dL}$)로 안전하게 복원합니다.

---

## 3. 세부 모델별 아키텍처 설계 명세

### 3.1. DLinear
- **개념:** 시계열 분해(Decomposition) 기법과 일차 선형 레이어(Linear Layer)의 결합형 구조입니다.
- **아키텍처 흐름:**
  1. 입력 시퀀스를 이동 평균 필터(Moving Average)를 통해 저주파 성분인 추세(Trend, $X_{Trend}$)와 고주파 성분인 잔차(Seasonal, $X_{Seasonal}$)로 분해합니다.
     $$X_{Trend} = \text{AvgPool1D}(X, \text{kernel\_size})$$
     $$X_{Seasonal} = X - X_{Trend}$$
  2. 분해된 두 성분에 대해 개별 선형 레이어 $\mathbf{W}_{Trend}$와 $\mathbf{W}_{Seasonal}$을 적용하여 독립적으로 예측 지평선 크기로 사영합니다.
  3. 두 사영 결과를 합산하여 최종 결과를 산출합니다.
     $$\hat{Y} = \mathbf{W}_{Trend} X_{Trend} + \mathbf{W}_{Seasonal} X_{Seasonal}$$
- **특징:** 채널 독립성(Channel Independence) 방식을 채택하여 변수 간 혼선 없이 개별 시계열 트렌드만 일률 매핑하므로 극도로 경량화되어 있습니다.

### 3.2. N-BEATS
- **개념:** 자가 어텐션이나 순환 연결 없이 다층 퍼셉트론(MLP) 블록의 이중 잔차 연결(Doubly Residual Stacking)만으로 과거 이력을 분해 예측하는 모델입니다.
- **아키텍처 흐름:**
  1. 여러 개의 FC 블록(Block)을 스택(Stack) 형태로 직렬 연결합니다.
  2. 각 블록 $l$은 입력을 받아 과거 재구성 값인 백캐스트(Backcast, $\hat{X}_l$)와 미래 예측 값인 포캐스트(Forecast, $\hat{Y}_l$)를 생성합니다.
  3. 다음 블록의 입력은 이전 블록의 백캐스트 잔차로 제공됩니다.
     $$X_{l} = X_{l-1} - \hat{X}_{l-1}$$
  4. 모든 블록에서 도출한 포캐스트 출력을 누적 합산하여 최종 예측을 생성합니다.
     $$\hat{Y} = \sum_{l} \hat{Y}_l$$
- **특징:** 해석 가능 모드(Interpretable Mode) 선택 시 블록의 기저 함수(Basis)를 다항식(Polynomial) 및 푸리에(Fourier) 함수로 제안하여 혈당의 기저 변동 패턴과 급격한 섭식 요동을 기하학적으로 분해할 수 있습니다.

### 3.3. N-HiTS
- **개념:** N-BEATS의 이중 잔차 스태킹을 계층적 풀링(Hierarchical Pooling) 및 보간(Interpolation) 연산으로 확장한 모델입니다.
- **아키텍처 흐름:**
  1. 각 스택마다 서로 다른 보폭의 풀링 레이어($\text{MaxPool1D}$ 등)를 통과시켜 입력 시계열의 해상도를 다르게 구성(Multi-rate sampling)합니다.
  2. 저해상도 블록은 거친 트렌드(장기 변동)를 담당하고, 고해상도 블록은 촘촘한 세부 변동(식사 자극 등 초단기 변동)을 파악하도록 설계합니다.
  3. 저해상도 블록의 포캐스트 출력물은 계층적 선형 보간(Hierarchical Interpolation)을 거쳐 원래 해상도의 스텝 수로 복원된 후 합산됩니다.
- **특징:** 과거 및 미래 예측 지평선이 길어질 때 발생하는 계산 비용을 획기적으로 줄이며 과적합 방지 능력이 탁월합니다.

### 3.4. TSMixer
- **개념:** 채널 독립성 가정을 탈피하고 변수 간(Cross-channel) 및 시간 간(Cross-time) 상관성을 순차 믹싱(Mixing)하는 MLP 기반 다변량 시계열 모델입니다.
- **아키텍처 흐름:**
  1. **Time-Mixing MLP:** 입력 텐서의 채널을 고정한 채 시간 차원(`Lookback_Steps`)에 대해 가중치 행렬을 곱해 시간 축 정보를 결합합니다.
  2. **Feature-Mixing MLP:** 시간 스텝을 고정한 채 변수 채널 차원(`Input_Channels`)에 대해 가중치 행렬을 곱해 공변량 간 상호작용을 통합합니다.
  3. 위 두 과정을 잔차 연결(Residual Connection) 및 레이어 정규화(Layer Normalization)와 결합하여 여러 층으로 적층(Stacking)합니다.
- **특징:** 인슐린 및 식사와 혈당 간의 다변량 인과관계를 어텐션 연산 장치 없이 초고속으로 동시 학습할 수 있습니다.

### 3.5. TiDE
- **개념:** 과거 시점의 입력과 미래의 외생적 공변량(Future Covariates) 정보를 인코더-디코더 구조의 MLP 네트워크로 통합 전파하는 아키텍처입니다.
- **아키텍처 흐름:**
  1. **Feature Projection:** 과거 및 미래 시점의 모든 공변량(예: 시간 인코딩, 식사 예정량 등)을 저차원의 조밀한 벡터로 매핑합니다.
  2. **Encoder:** 과거 혈당 및 매핑된 공변량 벡터들을 평탄화(Flatten)하여 조밀한 특징 표현(Dense Representation)으로 인코딩합니다.
  3. **Decoder:** 인코더의 특징 표현을 해석하여 미래 시점별 표현 벡터로 디코딩합니다.
  4. **Temporal Decoder:** 미래 표현 벡터에 미래 공변량을 병합하여 각 예측 시점의 아웃풋을 산출합니다.
  5. **Residual Connection:** 입력단 혈당 데이터에서 출력 예측단으로 이어지는 다이렉트 선형 잔차 커넥션을 결합하여 최종 값을 보정합니다.
- **특징:** 인슐린 주입 패턴과 같은 미래에 이미 결정된 치료 스케줄링(Known Future Covariates) 정보를 예측 파이프라인에 효과적으로 주입할 수 있습니다.

### 3.6. FITS
- **개념:** 시계열 데이터를 주파수 영역(Frequency Domain)으로 전환하여 저주파 필터링 및 복소 주파수 성분 보간을 적용하는 극도로 가벼운 선형 모델입니다.
- **아키텍처 흐름:**
  1. 입력 시퀀스 $X$에 대해 1차원 이산 푸리에 변환(Real Fast Fourier Transform, $\text{rFFT}$)을 실행하여 주파수 영역 복소수 텐서로 변환합니다.
     $$X_{freq} = \text{rFFT}(X)$$
  2. 고주파 영역의 센서 잡음을 필터링하기 위해 저주파 차원만 슬라이싱하여 추출(Cut-off)합니다.
  3. 주파수 영역에서 학습 가능한 복소 가중치 행렬(Complex Linear Layer)을 결합하여, 과거 주파수 해상도를 미래 해상도로 크기 조정(Frequency Interpolation / Padding)합니다.
  4. 변환 완료된 미래 주파수 성분에 역 푸리에 변환($\text{irFFT}$)을 취해 시간 도메인의 미래 혈당 예측값으로 환원합니다.
- **특징:** 파라미터 수가 룩백이나 예측 지평선 크기가 아닌 주파수 분해능에 비례하므로 연산 효율이 극대화되며, 초경량 엣지 장치 구동에 최적입니다.

### 3.7. PatchMLP
- **개념:** 단일 타임스텝 단위의 입력이 노이즈에 취약한 점을 개선하기 위해, 시계열을 로컬 세그먼트인 패치(Patch) 단위로 분할하여 추세적 흐름을 파악하는 MLP 아키텍처입니다.
- **아키텍처 흐름:**
  1. 입력 시퀀스를 겹치거나(Overlap) 독립된 크기 $P$ (Patch Length)와 이동 간격 $S$ (Stride)를 기준으로 다수의 패치 토큰으로 분할합니다.
  2. 각 패치 시퀀스를 MLP 임베딩 레이어를 통해 벡터 스페이스로 사영하여 국소적 의미 정보(Semantic Context)를 추출합니다.
  3. 추출된 다중 패치들을 패치 축과 특징 축 기준으로 각각 MLP 교차 믹싱을 수행합니다.
  4. 예측단 사영 레이어를 통해 각 채널별 패치 단위 예측을 수행한 후, 원래 시퀀스 차원으로 복원합니다.
- **특징:** CGM 신호의 단기 노이즈 변동에 대한 강건성(Robustness)이 우수하며 시계열 수렴도가 뛰어납니다.

### 3.8. XLinear
- **개념:** DLinear가 갖는 채널 독립성(CI) 제약을 제거하여, 채널 간의 상호 교차 작용(Cross-channel Interaction)을 선형적으로 처리할 수 있도록 확장한 하이브리드 선형 아키텍처입니다.
- **아키텍처 흐름:**
  1. 입력 시퀀스를 이동 평균 기반의 Trend와 Seasonal 성분으로 분리하는 기조는 DLinear와 동일합니다.
  2. 선형 투사 수행 시, 시간 차원에 대한 학습 파라미터 행렬 외에 채널 차원에 대한 선형 사영 행렬 $\mathbf{W}_{channel}$을 추가 정의합니다.
  3. 두 영역의 텐서곱(Kronecker Product) 또는 이중 Linear 순차 연산을 결합하여, 과거 $i$번째 채널의 트렌드가 미래 $j$번째 채널의 혈당 예측에 미치는 선형적 전파 경로를 연산합니다.
     $$\hat{Y} = \text{CrossLinear}_{Trend}(X_{Trend}) + \text{CrossLinear}_{Seasonal}(X_{Seasonal})$$
- **특징:** 연산 속도는 DLinear급의 고속을 유지하면서도 타 채널(인슐린, 식사량 등)이 혈당 변화에 기여하는 동적 상관성을 선형 범위 내에서 명시적으로 학습합니다.

### 3.9. SOFTS
- **개념:** 채널 간 정보 결합(Channel Interaction) 시 자가 어텐션의 고비용($O(N^2)$) 문제를 해결하기 위해, 원패스(One-pass) 방식으로 다변량 정보를 병합/분산하는 스케일러블 주파수/MLP 믹싱 모델입니다.
- **아키텍처 흐름:**
  1. 입력 시퀀스를 채널별로 인코딩한 뒤, 채널 축 전체를 하나로 병합하는 공통 표현 풀링(Global Core Representation) 연산을 수행합니다.
  2. 병합 과정에 주파수 영역 푸리에 변환(FFT) 또는 피드포워드 MLP 결합을 사용하여 모든 채널의 동역학적 특성을 요약합니다.
  3. 요약된 글로벌 특징 정보망을 개별 채널 디코더에 다시 분배(Broadcast & Multiply)하여 상호 보정합니다.
  4. 복원된 각 채널 특징 벡터에서 최종 혈당 예측 타겟을 사영 도출합니다.
- **특징:** 채널 수가 수십 개 이상으로 증가해도 계산 복잡도가 채널 수 $N$에 선형 비례하여 증가하므로 다차원 변수를 보유한 다기관 CGM 실험에 적합합니다.

---

## 4. 모듈러 컴포넌트 공유 설계

구현 단순화 및 코드 재사용성을 높이기 위해, 각 모델의 고유 연산부를 제외한 보조 레이어들은 공유 모듈로 통합 구현합니다.

```
                    [ Raw input tensor ]
                             │
                             ▼
            ┌──────────────────────────────────┐
            │       RevIN 정상화 모듈          │ (정상성 복원)
            └──────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Decomposition  │ │    rFFT 모듈    │ │   Patch Split   │ (표현 모듈 분기)
│ (DLinear, 등)   │ │ (FITS, SOFTS)   │ │  (PatchMLP)     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 4.1. 공통 분해 컴포넌트 (Decomposition Module)
- **대상 모델:** DLinear, XLinear, TSMixer, TiDE (선택 적용)
- **클래스 정의:** `SeriesDecomposition(nn.Module)`
  - `moving_avg` 커널 크기를 파라미터로 설정하여 추세 및 잔차 텐서 쌍을 리턴하는 단일 헬퍼 클래스를 구현합니다.

### 4.2. 공통 주파수 연산 모듈 (Frequency Processing Module)
- **대상 모델:** FITS, SOFTS
- PyTorch의 `torch.fft.rfft` 및 `torch.fft.irfft`를 표준 래핑하여 복소수 곱 연산 및 저주파 필터 컷오프 기능을 수행하는 유틸리티로 모듈화합니다.

### 4.3. 공통 패칭 연산 모듈 (Patching Module)
- **대상 모델:** PatchMLP
- 입력 텐서 `[B, L, C]`를 `[B, C, N_Patches, Patch_Length]`로 뷰(View) 및 언폴드(Unfold) 처리하여 변환하는 공유 유틸리티를 설계합니다.

---

## 5. 학습, 검증 및 임상 지표 최적화 사양

### 5.1. 손실 함수 (Loss Function)
공통 오차 최적화를 위해 기본 MAE(L1 Loss) 및 MSE(L2 Loss)를 결합하여 사용하되, 임상적 안전성 강화를 위해 저혈당 대역 페널티 손실 함수를 선택적으로 추가합니다.
- **저혈당 페널티 손실 (Hypoglycemic Penalty Loss):**
  실제 혈당이 $70 \text{ mg/dL}$ 미만일 때 예측값이 실제보다 높게 예측되는 경우(과대평가로 인한 치료 누락 방지), 오차에 페널티 가중치 $\gamma > 1.0$을 추가 곱하여 계산합니다.
  $$\mathcal{L} = \begin{cases} 
  |Y - \hat{Y}| \times \gamma, & \text{if } Y < 70 \text{ and } \hat{Y} > Y \\ 
  |Y - \hat{Y}|, & \text{otherwise} 
  \end{cases}$$

### 5.2. 평가 메트릭 (Evaluation Metrics)
1. **일반 예측 오차:** RMSE, MAE, MAPE
2. **임상 안전성 평가 (Clarke Error Grid):**
   - 예측값과 실측값을 비교하여 안전 구역인 Zone A 및 Zone B에 위치하는 비율(%)을 검출합니다.
   - Zone A+B 비율이 최소 98% 이상을 충족하는지 검증합니다.
3. **위험 변동 평가:** 저혈당($<70 \text{ mg/dL}$) 대역에서의 예측 민감도(Sensitivity) 및 특이도(Specificity).

---

## 6. 온디바이스(Edge CPU) 배포 및 실시간 개인화 규격

엣지 컴퓨팅 기반의 모바일 디바이스 또는 인슐린 펌프 내부 임베디드 칩 구동을 전제로 한 최적화 표준 설계입니다.

### 6.1. 경량화 및 직렬화 표준
- **연산 엔진 고정:** 모바일 디바이스 상의 과도한 메모리 점유 및 배터리 소모를 방지하기 위해 PyTorch의 Autograd(역전파 엔진) 및 동적 그래프 빌더를 완전히 비활성화한 후 추론 모드로 전환합니다.
- **ONNX 및 TFLite 컴파일:** 공통 구현 모델들을 ONNX(Open Neural Network Exchange) 형식으로 변환한 뒤, 대상 엣지 타겟에 맞춰 FP16 양자화(Quantization)를 거쳐 TFLite 바이너리로 컴파일하여 구동합니다.

### 6.2. 엣지 실시간 개인화 레이어 (On-Device Personalization Block)
글로벌 데이터로 선학습된 공통 모델 위에, 실시간 환자의 특이 반응(인슐린 저항성 변화, 호르몬 요동)을 장치 단에서 즉시 학습하여 보정하는 초경량 레이어 설계입니다.

```
       [공통 글로벌 모델 예측값: Y_global]
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │     개인화 선형 레이어: W_personal    │ ◄── 환자 최신 잔차 데이터로
    └───────────────────────────────────────┘     On-Device 실시간 폐형태 해 업데이트
                        │
                        ▼
      [최종 보정 예측 혈당: Y_final (mg/dL)]
```

- **설계 구조:**
  - 글로벌 모델 추론 출력단 바로 뒤에 학습 가능한 1차원 가중치 파라미터 $\mathbf{W}_{personal}$과 바이어스 $\mathbf{b}_{personal}$을 가진 단일 선형 레이어를 연결합니다.
  - 디바이스 내부 메모리 버퍼에 최근 1시간(12개 샘플)의 예측 오차 잔차 데이터를 유지합니다.
- **폐형태 해(Closed-form Solution) 초고속 갱신:**
  - 디바이스의 제한된 CPU 파워로 경사하강법(GD)을 반복 연산하면 하드웨어 자원이 낭비되므로, `Moore-Penrose Pseudo-Inverse` 의사역행렬 공식을 사용하여 행렬 연산 1회로 최적 가중치 해를 즉시 산출합니다.
  - 수식:
    $$\mathbf{W}_{personal} = \mathbf{Y}_{true} \mathbf{X}_{pred}^{T} (\mathbf{X}_{pred} \mathbf{X}_{pred}^{T})^{-1}$$
  - 이 수식을 통해 디바이스 내부에서 복잡한 최적화 알고리즘 구동 없이, 실시간 버퍼가 갱신될 때마다 가중치를 즉각 업데이트하여 개인 최적화 혈당 경로 보정을 구현합니다.
