# [Tier 2.5] Feature Engineering & Classic ML

본 문서는 Tier 2의 Random Forest와 Decision Tree 성능을 극대화하기 위해, 원천 데이터에 명시적인 특징 공학(Feature Engineering)을 결합하여 예측 인지력을 증폭시키는 Tier 2.5 설계안입니다. 
제안하신 바와 같이, 머신러닝 모형은 딥러닝과 달리 시계열의 모멘텀이나 분산을 자체적으로 추출하는 데 한계가 있으므로, 이를 파생 변수(Derived Features)로 만들어 **원본 시계열(Raw CGM)과 함께 동시 투입(Concatenation)**하는 것이 핵심입니다.

## User Review Required

> [!CAUTION]
> 파생 변수를 추가하면 윈도우 데이터의 차원(Dimension)이 증가합니다. 
> 기존 윈도우 `X`가 [최근 6개의 원본 혈당, 최근 6개의 공변량]이었다면, 
> Tier 2.5 윈도우는 **[최근 6개의 원본 혈당, 속도, 가속도, 평균, 편차, 시간대... + 최근공변량]** 형태로 대폭 확장됩니다.
> 이 방식이 의도하신 "원본 변수 CGM이 같이 투입되는 형태"가 맞는지 확인해 주세요!

## Proposed Changes

### [확정] 통합 파생 변수 (Feature Engineering List)

현행 타겟 혈당값($G_{t+H}$)이 결코 누출(Data Leakage)되지 않도록 철저히 과거 윈도우(Lookback, $t-L+1 \sim t$) 내에서만 다음 변수들을 생성해 원본과 병합합니다.

1. **역학(Kinetic) 변수**
   - **Velocity (속도):** $\Delta G_t = G_t - G_{t-1}$
   - **Acceleration (가속도):** $\Delta^2 G_t = \Delta G_t - \Delta G_{t-1}$
   - **Window AUC:** 윈도우 내 혈당 곡선 하 면적 (사다리꼴 적분 활용, 누적 대사 스트레스)

2. **통계적 변동성 및 임상 지표**
   - **Window Mean / Std:** 윈도우 내 평균 및 표준편차
   - **TIR / TAR / TBR:** 직전 윈도우 내 정상 범위(70~180), 고혈당(>180), 저혈당(<70) 체류 비율(%)
   - **LBGI / HBGI (Kovatchev Risk Index):** 저혈당/고혈당 위험의 비대칭적 비선형 변환 지수. 저혈당 스파이크 민감도 확보.

3. **일주기 리듬 (Circadian Rhythm)**
   - **Hour of Day:** $\sin(\text{hour})$ 및 $\cos(\text{hour})$ 로 변환하여 호르몬 주기 반영.

4. **이벤트 메모리 변수 (Event Decay & Time-Since)**
   - **Time-Since-Event:** 최근 식사/인슐린 이벤트로부터 경과한 스텝 수 (단절 시 최대 지정값으로 클리핑).
   - **Exponential Decay (IOB/COB 근사):** 이벤트 값에 지수적 감쇠(decay)를 적용하여 잔존 인슐린(IOB)이나 체내 잔존 탄수화물(COB) 효과를 연속적인 모멘텀으로 변환.

### 시스템 아키텍처 (`run_tier2.5_feature_eng.py`)

기존 `8_Classic_ML/run_classic_ml.py` 엔진을 계승 및 개조하여 `9_Tier_2.5_Feature_Engineering/` 폴더에서 실험을 진행합니다.

1. **Vectorized Feature Factory:**
   - 기존의 `build_windows` 함수 내부에서 NumPy의 벡터 연산을 활용해 속도, 편차, 시간 변수를 계산하여 병합 행렬(Concatenated Matrix)을 만듭니다.
2. **동일 모형 벤치마크 루프 (단, 하이퍼파라미터 봉인 해제):**
   - 확장된 차원을 소화하기 위해 **Random Forest의 `max_depth`를 20 (또는 None)**으로 상향 조정. 시각화용 Decision Tree는 `depth=5` 유지.
3. **결과 비교 대조군:**
   - Tier 2의 결과와 Tier 2.5의 결과를 1:1로 비교할 수 있는 산출표를 만들고, 파생 변수가 실제 `Feature Importance`에서 상위권으로 치고 올라오는지(`Top-3 Features`) 관찰합니다.

> [!NOTE]
> 제안해주신 모든 추가 도메인 지표(TIR, LBGI, Exponential Decay, Time-Since) 및 모형 방어구조(Data Leakage 방지, max_depth 상향)를 전부 승인 및 수용하여 위 설계에 최종 병합하였습니다. 구현에 곧바로 착수하겠습니다.

## Verification Plan
1. `run_tier2.5_feature_eng.py` 실행 후, Output 행렬의 `shape`가 (기존 변수 개수 + 5)로 정확히 확장되었는지 점검.
2. 각 데이터셋 별 RMSE 계산 후, Tier 2 결과 수치(`8_Classic_ML/3_Result_Summary.md`) 대비 오차 감소분 입증.
