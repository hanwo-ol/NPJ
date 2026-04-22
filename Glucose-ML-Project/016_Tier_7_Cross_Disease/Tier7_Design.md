# Tier 7: Cross-Disease Transfer — T1D → T2D

## 실험 목적

RSD.md Scenario D 구현.

**핵심 질문**: T1D CGM 시계열에서 학습한 혈당 패턴 표현이 T2D에 전이될 수 있는가?

전이학습의 필요성:
- ShanghaiT2DM = 100명 → 직접 학습 시 과적합, 희소
- T2D는 전 세계 당뇨의 90%이나 CGM ML 연구에서 소외
- T1D 대규모 코호트가 T2D 모델의 사전 지식으로 기능할 수 있다면 공중보건적 가치 큼

---

## 도메인 갭 정의

| 특성 | T1D (Source) | T2D (Target) |
|---|---|---|
| 혈당 변동성 | 높음 (CV > 36%) | 낮음~중간 |
| 공복혈당 | 낮음~정상 | 높음 (기저 저항성) |
| 식후 피크 | 빠르고 높음 | 완만하고 지속 |
| 인슐린 반응 | 즉각적 | 지연됨 |
| 샘플링 | 5분 (대부분) | 15분 (Shanghai) |

---

## 4-Way 비교 실험 설계

```
┌──────────────────────────────────────────────────────────┐
│  T2D Test Set (15%) — 공통 평가 대상, 모든 모델 동일       │
├──────────────────────────────────────────────────────────┤
│  1. Source-Only    │ T1D 전체 → T2D test (zero-shot)     │
│  2. Target-Only    │ T2D train(70%) → T2D test           │
│  3. Mixed          │ T1D + T2D train → T2D test          │
│  4. TrAdaBoost     │ 재가중 T1D + T2D train → T2D test   │
│  5. CORAL          │ 분포 정렬 후 학습 → T2D test         │
│  [Oracle]          │ T2D 10-fold CV (참고용 상한)         │
└──────────────────────────────────────────────────────────┘
```

목표 순서: Source-Only < Target-Only < Mixed ≤ CORAL ≤ TrAdaBoost < Oracle

---

## 소스 데이터셋 (T1D)

| 데이터셋 | 환자 수 | 샘플링 | 선택 이유 |
|---|---|---|---|
| RT-CGM | 448 | 5분 | 대규모, 다인종 |
| IOBP2 | 440 | 5분 | 대규모 Jaeb 임상 |
| FLAIR | 113 | 5분 | Jaeb 임상 표준 |
| SENCE | 143 | 5분 | Jaeb 임상 표준 |
| WISDM | 203 | 5분 | 고령 T1D (다양성) |
| PEDAP | 103 | 5분 | 소아 T1D (다양성) |

**총 1,450명 T1D**

## 타겟 데이터셋 (T2D)

| 데이터셋 | 환자 수 | 샘플링 | 분할 |
|---|---|---|---|
| ShanghaiT2DM | 100 | 15분 | 70/15/15 |

---

## 피처 설계 (15분 예측 표준화)

모든 데이터셋을 **15분 예측 horizon**으로 통일:
- 5분 샘플링: t, t-5, t-10, t-15, t-20, t-25 → 예측 t+15 (t+3)
- 15분 샘플링: t, t-15, t-30 → 예측 t+15 (t+1)

### 추출 피처 (tier3_data_utils 호환)
- Lag features: glucose[t-1], ..., glucose[t-k]
- Delta features: Δglucose, Δ²glucose
- Window stats: mean, std, min, max over lookback
- Risk features: LBGI, HBGI (Kovatchev)
- Time features: hour_sin, hour_cos, is_night

### 추가 T2D 도메인 피처 (신규)
- `fasting_proxy`: 자정~06:00 구간 평균 (공복혈당 proxy)
- `postmeal_rise`: 로컬 최솟값 이후 60분 상승폭 (식후 반응)
- `resistance_index`: 고혈당 지속 시간 비율

---

## TrAdaBoost 구현 사양 (ML 수준)

```python
알고리즘:
  1. 초기 가중치: 소스 1/Ns, 타겟 1/Nt
  2. 반복 N번:
     a. 현재 가중치로 LightGBM 학습 (소스 + 타겟 훈련)
     b. 타겟 훈련 세트 오차 계산
     c. 소스 가중치 업데이트: 큰 오차 → 가중치 감소 (소스 아웃라이어 제거 효과)
     d. 타겟 가중치 유지 (타겟 샘플은 항상 중요)
  3. 후반 N/2 모델 앙상블 (초반 불안정 제거)
```

---

## 평가 지표

| 지표 | 임계값 | 의미 |
|---|---|---|
| RMSE (mg/dL) | < 20 임상 허용 | 주 지표 |
| MAE (mg/dL) | — | 보조 |
| MARD (%) | < 15% 임상 허용 | CGM 수준 비교 |
| TIR-RMSE | — | 70-180 구간 내 오차 |

---

## 출력 파일

```
tier7_results/
├── 4way_comparison.csv       — 4-way 모델 RMSE 비교표
├── learning_curve.png        — 타겟 데이터량 vs. 성능
├── shap_source.png           — 소스 학습 모델의 SHAP
├── shap_target.png           — 타겟 학습 모델의 SHAP
├── shap_delta.png            — 피처 중요도 변화 (어떤 피처가 전이됨)
└── domain_gap_analysis.md    — 실험 결과 해석
```
