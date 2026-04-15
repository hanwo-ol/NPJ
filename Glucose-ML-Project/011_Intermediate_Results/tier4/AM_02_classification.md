# AM-02: Classification으로의 문제 전환
# — "CGM Regression 대신 (또는 함께) Classification으로 무엇을 할 수 있는가?"

---

## 배경

현재 파이프라인은 **"30분 후 CGM 값(mg/dL)을 예측하는 regression"** 으로 설계되어 있다. 그러나 AM-01에서 논의한 바와 같이 CGM은 ground truth가 아니며, 절대 수치의 정확도보다 **임상적으로 의미 있는 범주/경향의 예측**이 더 중요할 수 있다.

Classification으로 문제를 전환(또는 보조 분석으로 추가)할 경우, 다음과 같은 추가 이점이 있다:

1. **CGM ≠ GT 문제 완화** — 절대 수치가 아닌 범주/방향을 예측하므로 센서 오차에 강건
2. **일관성 지표 자연 적용** — Cohen's Kappa, ARI 등 분류 일치도 지표가 직접 사용 가능
3. **임상적 해석 용이** — "RMSE가 3 mg/dL 줄었다"보다 "저혈당 예측 sensitivity가 85%→92%로 올랐다"가 임상가에게 직관적

---

## 문제 정의 1: Glycemic Range Classification (3-class)

### 정의

> "30분 후 혈당이 **저혈당 / 정상 / 고혈당** 중 어디에 속할 것인가?"

| 클래스 | 범위 (mg/dL) | 임상적 의미 |
|:---:|:---:|:---|
| **TBR** (Time Below Range) | < 70 | 저혈당 — 즉각 조치 필요 (간식 섭취, 인슐린 중단) |
| **TIR** (Time In Range) | 70 ~ 180 | 정상 범위 — 현 상태 유지 |
| **TAR** (Time Above Range) | > 180 | 고혈당 — 인슐린 조절 또는 활동량 증가 필요 |

### 구현 방법

기존 regression target(t+6 값)을 3개 범주로 변환만 하면 된다:

```
y_class = "TBR" if y < 70 else ("TAR" if y > 180 else "TIR")
```

### CGM ≠ GT 관점

절대값 오차(±15 mg/dL)가 있더라도, **경계(70, 180) 근처의 샘플에서만** 분류 오류 발생. 전체 샘플 중 경계 ±15 mg/dL 구간은 소수이므로 전체 accuracy에 미치는 영향이 제한적이다.

### 장점과 단점

| 장점 | 단점 |
|:---|:---|
| 파이프라인 수정 최소 | **극심한 class imbalance** — T1DM에서도 TBR 2~5%, ND에서는 TBR ≈ 0% |
| Cohen's Kappa 직접 적용 가능 | 3개 범주가 임상적으로 너무 넓을 수 있음 |
| ADA/EASD 국제 기준 범주와 일치 | 경계 근처의 "근접 저혈당"(예: 75 mg/dL)이 TIR로 분류됨 |

### 평가 지표

- **Macro F1-score** (class imbalance 대응)
- **Cohen's Kappa** (모델 간 일치도 → Within/Between 비교에 핵심)
- **Confusion Matrix** (TBR→TIR 오분류 vs TBR→TAR 오분류의 임상적 의미 구분)

---

## 문제 정의 2: Hypoglycemia Prediction (Binary)

### 정의

> "30분 이내에 **저혈당(< 70 mg/dL)이 발생할 것인가?** (Yes / No)"

### 변형

| 변형 | 기준 | 임상적 의미 |
|:---|:---:|:---|
| Level 1 저혈당 | < 70 mg/dL | 주의 필요 |
| Level 2 저혈당 | < 54 mg/dL | **심각 — 인지 장애, 의식 소실 가능** |

### CGM ≠ GT 관점

저혈당은 **놓치면 위험하고, 오경보는 비교적 안전**한 비대칭 비용 구조. CGM의 오차로 인한 false positive(실제 75이지만 CGM이 68로 읽음)는 환자에게 간식을 먹게 하는 정도의 비용이므로, **보수적 오경보가 허용**된다.

### 장점과 단점

| 장점 | 단점 |
|:---|:---|
| **임상적 가치 최고** — 가장 actionable한 예측 | **극심한 class imbalance** — 양성 비율 1~5% |
| Sensitivity/NPV가 직관적 | ND 데이터셋에는 저혈당 이벤트가 거의 없어 학습 불가 |
| Between Variation 질문: "A에서 학습한 저혈당 예측기가 B에서 sensitivity 유지?" | AUPRC가 필요하나 AUROC만 보고하면 과대 평가 |

### 평가 지표

- **AUPRC** (Precision-Recall AUC — 희귀 이벤트에 적합, AUROC보다 정직한 평가)
- **Sensitivity @ fixed specificity** (예: Specificity 95%에서의 Sensitivity)
- **NPV** (Negative Predictive Value — "안전하다고 예측한 것 중 실제로 안전한 비율")

### Between Variation 관점

**"A 데이터셋에서 학습한 저혈당 예측기가 B 데이터셋에서도 sensitivity를 유지하는가?"** — 이것만으로도 npj Digital Medicine급 연구 질문이 될 수 있다.

---

## 문제 정의 3: Glucose Trend Classification (3~5 class)

### 정의

> "30분 후 혈당이 현재 대비 **급상승 / 상승 / 안정 / 하강 / 급하강** 중 어디인가?"

| 클래스 | Δglucose (t+6 − t+0) | CGM 기기 화살표 | 비율 (추정) |
|:---:|:---:|:---:|:---:|
| ↑↑ 급상승 | > +30 mg/dL | ⬆⬆ | ~5~10% |
| ↑ 상승 | +15 ~ +30 | ⬆ | ~15~20% |
| → 안정 | -15 ~ +15 | ➡ | ~40~50% |
| ↓ 하강 | -30 ~ -15 | ⬇ | ~15~20% |
| ↓↓ 급하강 | < -30 | ⬇⬇ | ~5~10% |

### CGM ≠ GT 관점

**이 문제에서 CGM ≠ GT 문제가 가장 약화된다.** 예측하는 것이 절대값이 아닌 **변화량(Δ)**이므로, CGM의 constant offset(예: CGM이 혈당보다 항상 10 높게 읽음)이 상쇄된다.

```
Δ = CGM(t+6) - CGM(t+0)
  = (BG(t+6) + offset) - (BG(t+0) + offset)
  = BG(t+6) - BG(t+0)  ← offset 상쇄
```

### 장점과 단점

| 장점 | 단점 |
|:---|:---|
| **CGM ≠ GT 문제 최소화** — 변화량 기반, offset 상쇄 | 경계 기준(±15, ±30)의 정당화 필요 |
| **Class balance가 비교적 균등** — 5개 클래스가 대칭 분포 | 5-class는 평가지표 해석이 복잡 |
| 상용 CGM 기기의 "trend arrow"와 직접 대응 | Δ의 크기가 sensor interval(5분 vs 15분)에 따라 다를 수 있음 |
| 순위 안정성(rank stability) 측정에 적합 | — |

### 평가 지표

- **Weighted Kappa** (순서형 분류에 적합 — "급상승을 안정으로 예측"이 "급상승을 상승으로 예측"보다 더 나쁨을 반영)
- **Adjacent Accuracy** (한 단계 이내 정답 비율)
- **Macro F1-score**

---

## 문제 정의 4: Critical Event Window Classification (Binary, 시간 확장)

### 정의

> "향후 1~2시간 내에 **Level 2 저혈당(< 54) 또는 심각한 고혈당(> 250)** 이 발생하는가?"

### 기존 파이프라인과의 차이

| | 기존 (Regression) | Critical Event (Classification) |
|:---|:---|:---|
| Target | t+6 **한 개** 포인트의 값 | t+1 ~ t+12(또는 t+24) 중 **하나라도** 위험 범위 진입 |
| 질문 | "30분 후 혈당이 얼마?" | "향후 2시간 동안 안전한가?" |
| 임상 대응 | 수치 확인 | Go/No-Go 판단 |

### CGM ≠ GT 관점

여러 포인트의 **최솟값/최댓값 기반**이므로, 단일 포인트의 센서 노이즈에 덜 민감하다.

### 장점과 단점

| 장점 | 단점 |
|:---|:---|
| 임상적으로 가장 자연스러운 질문 | **파이프라인 수정 필요** — target 윈도우 확장 |
| "향후 N시간 안전" 예측은 AID 시스템의 직접적 요구 | 확장된 윈도우에서 class balance 재계산 필요 |
| 시간 범위를 파라미터로 두면 다양한 horizon 분석 가능 | — |

---

## 문제 정의 5: Patient-Level Glycemic Control Classification (환자 단위)

### 정의

> "이 환자의 전반적인 혈당 조절 수준은 **양호 / 보통 / 불량** 중 무엇인가?"

| 클래스 | 기준 (ADA/EASD 권고 기반) | 분류 근거 |
|:---:|:---|:---|
| **양호** | TIR > 70% AND TBR < 4% | ADA 2023 권고 목표 충족 |
| **보통** | TIR 50~70% | 목표 미달, 개선 필요 |
| **불량** | TIR < 50% OR TBR > 4% | 적극적 개입 필요 |

### 기존 파이프라인과의 차이

**완전히 다른 문제 구조.** 윈도우 단위가 아닌 **환자 단위**로 집계(aggregate)된 분류이므로:
- 입력: 환자의 전체 CGM 시퀀스에서 추출한 통계량 + Q/M 정적 피처(나이, HbA1c, BMI)
- 출력: 환자 1명에 대한 1개 분류 레이블
- **Q/M 피처가 핵심 역할** — regression에서는 미사용이었던 정적 변수가 여기서 주력 피처가 됨

### 장점과 단점

| 장점 | 단점 |
|:---|:---|
| Q/M 피처 자연 활용 | **대규모 파이프라인 재설계 필요** |
| 환자 단위 → 표본 수가 "윈도우 수"가 아닌 "환자 수"(12~886명) | 표본 수 극감 → GBM 학습 어려울 수 있음 |
| Between Variation: "A의 환자 특성으로 B의 환자를 분류 가능?" | ADA 기준이 모든 당뇨 유형에 동일 적용 가능한지 불확실 |

---

## 종합 비교

| 문제 | CGM≠GT 완화 | Class Balance | 파이프라인 수정 | Within/Between 적용 | 임상 가치 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 1. Range (3-class) | ⭐⭐ | ❌ 불균형 심각 | 최소 | Kappa 직접 적용 | ⭐⭐⭐ |
| 2. Hypo (binary) | ⭐⭐ | ❌❌ 극심 | 최소 | Sensitivity 전이 | ⭐⭐⭐⭐ |
| **3. Trend (5-class)** | **⭐⭐⭐** | **⭐⭐ 균등** | **최소** | **Weighted Kappa** | **⭐⭐⭐⭐** |
| 4. Critical Event | ⭐⭐⭐ | ❌ | 중간 | 안전시간 일반화 | ⭐⭐⭐ |
| 5. Patient-Level | ⭐ | ⭐⭐ | 대규모 | Q/M 활용 | ⭐⭐ |

---

## 권장: Regression + Classification 병행 전략

기존 regression 파이프라인을 유지하면서, **동일한 예측값으로부터** classification 지표를 추가 산출하는 전략을 권장한다:

```
                        ┌─ RMSE, MAE  .................. (기존)
                        │
예측값 ŷ(t+6) ──────────┼─ Range class (< 70 / 70~180 / > 180) ── Kappa, F1
                        │
                        ├─ Hypo binary (< 70 여부) ──── Sensitivity, AUPRC
                        │
                        └─ Trend class (Δ = ŷ - x(t+0)) ─ Weighted Kappa
```

### 이 전략의 이유

1. **파이프라인 수정 없음** — regression 모델은 그대로, post-hoc으로 범주화만 추가
2. **벤치마크 비교 유지** — RMSE 보고로 기존 논문들과의 비교 가능성 유지
3. **평가 지표 안건 해결** — Within/Between 일관성을 Cohen's Kappa로 직접 측정 가능
4. **CGM ≠ GT 완화** — Trend classification에서 offset 상쇄 효과
5. **추가 Table/Figure** — 논문의 풍부도 증가 (regression 결과 + classification 결과)

### 회의 안건 연결

이 내용은 `meeting_agenda.md`의 다음 안건과 직접 연결된다:
- **안건 1.2** (평가 지표) — Classification 추가 시 Cohen's Kappa, Weighted Kappa가 주 일관성 지표로 확정
- **안건 3.2** (결과 형태) — Regression Table + Classification Table 병행 보고

---

*Glucose-ML-Project · Agenda Material 02 · 2026-04-15*
