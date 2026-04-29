# Tier 7.1 임상 전이 분석 가이드

## 분석 기준 전제

본 연구에서 혈당 예측 모델의 평가 기준(reference)은 **CGM 간질액 포도당 측정값**이며, 정맥혈 기준(YSI)이 아니다.
이 전제에 대한 근거를 아래 세 가지로 정리한다.

### 1) Clarke Error Grid의 원전

Clarke Error Grid는 자가혈당측정기(SMBG)의 임상 정확도를 정맥혈 대비 평가하기 위해 설계되었다.

- Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL.
  "Evaluating clinical accuracy of systems for self-monitoring of blood glucose."
  *Diabetes Care*. 1987;10(5):622-628. doi:10.2337/diacare.10.5.622

### 2) CGM 값이 임상 의사결정의 기준이 된 근거

현재 임상 현장에서는 정맥혈이 아닌 CGM 값을 기반으로 인슐린 투여량 결정, 저혈당 경보,
Time-in-Range 목표 설정 등 치료 판단이 이루어진다. 이를 공식화한 국제 합의문:

- Danne T, Nimri R, Battelino T, et al.
  "International Consensus on Use of Continuous Glucose Monitoring."
  *Diabetes Care*. 2017;40(12):1631-1640. doi:10.2337/dc17-1600

이 합의문은 CGM 데이터를 임상 의사결정에 직접 사용할 것을 권고하며,
이에 따라 CGM 값 자체가 치료 판단의 실질적 기준(practical reference)이 되었다.

### 3) CGM을 reference로 Clarke Grid를 적용한 선행 연구

CGM 기반 혈당 예측 연구에서 CGM 측정값을 Clarke Grid의 reference로 사용하는 것은
분야의 표준 관행이다. 이를 실제로 수행한 주요 논문:

- Perez-Gandia C, Facchinetti A, Sparacino G, et al.
  "Artificial neural network algorithm for online glucose prediction from continuous glucose monitoring."
  *Diabetes Technology & Therapeutics*. 2010;12(1):81-88. doi:10.1089/dia.2009.0076
  (CGM 데이터로 15/30/45분 예측, Clarke Grid로 평가. Zone A+B 92.3%)

- Li K, Liu C, Zhu T, Herrero P, Georgiou P.
  "GluNet: A Deep Learning Framework for Accurate Glucose Forecasting."
  *IEEE Journal of Biomedical and Health Informatics*. 2020;24(2):414-423. doi:10.1109/JBHI.2019.2931842
  (OhioT1DM 데이터셋, CGM reference, Clarke Grid 보고)

- Zhu T, Li K, Herrero P, Georgiou P.
  "Dilated Recurrent Neural Networks for Glucose Forecasting in Type 1 Diabetes."
  *Journal of Healthcare Informatics Research*. 2020;4:308-324. doi:10.1007/s41666-020-00068-2
  (CGM reference, Clarke Grid + SEG 보고)

### 한계 명시

위 선행 연구들이 CGM을 reference로 Clarke Grid를 적용했으나,
"CGM을 reference로 사용할 때 Zone 비율이 정맥혈 reference 대비 어떻게 달라지는가"를
직접 비교한 체계적 연구는 확인하지 못하였다.

따라서 본 분석에서 Clarke Zone 비율의 **절대값 해석**에는 주의가 필요하다.
다만, 본 분석의 주 목적은 "어떤 모델이 절대적으로 안전한가"가 아니라,
**동일한 reference(CGM) 하에서 모델 간 상대 비교**를 통해
"전이학습이 임상 안전성 지표를 개선하는가, 악화시키는가"를 판별하는 것이다.
이 목적에서 reference의 종류는 모델 간 비교 결과에 영향을 주지 않는다.

---

> 모든 실험 완료 후, 이 문서의 순서대로 분석을 수행한다.
> 각 항목의 판단 기준이 사전 정의되어 있으므로, 결과에 맞춰 해석만 채우면 된다.

---

## 분석 TODO

- [ ] 1. 단계 0 검증: y_pred 저장 정합성 확인
- [ ] 2. 단계 1 분석: Clarke Error Grid 해석
- [ ] 3. 단계 1 분석: 저혈당 구간 안전성 판단
- [ ] 4. 단계 1 분석: 구간별 RMSE 패턴 확인
- [ ] 5. 단계 2 분석: 동일 질병 전이 핵심 가설 판별
- [ ] 6. 단계 3 분석: Cold Start 교차점 확인
- [ ] 7. 단계 3 분석: 도메인 갭별 Cold Start 패턴 비교
- [ ] 8. 종합: 3개 단계 결과를 연결한 최종 결론 도출
- [ ] 9. Tier7.1_Results_Analysis.md 보고서 작성

---

## 1. 단계 0 검증: 저장 정합성

**확인 사항:**
- `tier7_results/predictions/` 하위에 3개 타겟 폴더 존재하는가?
- 각 폴더에 5개 `.npz` 파일 (source_only, target_only, mixed, coral, tradaboost) 존재하는가?
- 각 `.npz`의 `y_true` shape와 `y_pred` shape가 일치하는가?
- `5way_all_targets.csv`의 RMSE와 `.npz`에서 재계산한 RMSE가 소수점 2자리까지 일치하는가?

**판단:** 불일치 시 해당 타겟 재실행 필요. 일치하면 다음 단계로 진행.

---

## 2. 단계 1 — Clarke Error Grid 해석

### Clarke Error Grid란?

Clarke Error Grid는 혈당 예측(또는 측정) 장치의 **임상적 안전성**을 평가하는 표준 도구이다.
1987년 Clarke 등이 제안하였으며, 현재까지 CGM 기기와 혈당 예측 모델의 FDA 인허가 및 논문 심사에서
사실상 표준으로 사용된다.

이 방법은 "예측 오차의 크기"가 아니라 "예측 오차가 의사의 판단을 어떻게 바꾸는가"를 기준으로 평가한다.
예를 들어, 실제 혈당이 200 mg/dL인 환자에게 AI가 210으로 예측하면 오차는 있지만 의사의 치료 결정은 동일하다 (Zone A).
반면, AI가 60으로 예측하면 의사는 불필요하게 포도당을 투여하거나 인슐린을 중단할 수 있다 (Zone E).

### Zone 분류 수식

$R$ = 실제 혈당 (reference, mg/dL), $P$ = 예측 혈당 (predicted, mg/dL) 일 때:

**Zone A** (임상적으로 정확 — 올바른 임상 판단으로 이어짐):

$$
\text{Zone A} = \begin{cases}
P < 70 & \text{if } R < 70 \\
|P - R| \leq 0.2 \cdot R & \text{if } R \geq 70
\end{cases}
$$

즉, 실제 혈당이 70 미만이면 예측도 70 미만이어야 하고, 70 이상이면 오차가 실제값의 20% 이내여야 한다.

**Zone E** (가장 위험 — 정반대 판단 유발):

$$
\text{Zone E} = (R \geq 180 \text{ and } P \leq 70) \;\text{ or }\; (R \leq 70 \text{ and } P \geq 180)
$$

실제로는 고혈당인데 저혈당으로 예측하거나, 그 반대인 경우이다.

**Zone C** (과잉 치료 유발):

$$
\text{Zone C}: \; R \geq 70 \text{ and } P > 1.2 \cdot R \text{ and } P \geq 180
$$
$$
\text{or}: \; R \geq 70 \text{ and } P < 0.8 \cdot R \text{ and } P \leq 70
$$

**Zone D** (위험한 미감지):

$$
\text{Zone D}: \; (R < 70 \text{ and } 70 \leq P \leq 180) \;\text{ or }\; (R > 180 \text{ and } 70 \leq P \leq 180)
$$

**Zone B** (양성 오류 — 해롭지 않은 오차):

$$
\text{Zone B} = \text{A, C, D, E 어디에도 속하지 않는 나머지}
$$

### 왜 RMSE와 별도로 필요한가?

RMSE가 같아도 Zone 분포가 다를 수 있다. RMSE는 오차의 평균 크기를 보여주지만,
Clarke Grid는 그 오차가 **환자에게 해를 끼치는가**를 보여준다.
이것이 RMSE와 별도로 Clarke Grid를 분석하는 이유이다.

**파일:** `tier71_results/clinical/clarke_zones.csv`

### 판단 기준

| Zone | 의미 | 허용 기준 |
|---|---|---|
| A | 임상적으로 정확 | >= 95% 면 우수, >= 85% 면 수용 가능 |
| A+B | 임상적으로 안전 | >= 99% 면 안전, < 95% 면 위험 |
| C+D+E | 위험한 오류 | < 1% 면 안전, > 5% 면 심각 |

### 분석 질문

**Q1. 모델 간 Zone A 차이가 유의미한가?**
- `target_only`와 `tradaboost`의 Zone A 차이가 1%p 이상이면 유의미
- `source_only`의 Zone A가 다른 모델보다 5%p 이상 낮으면 cross-disease 위험 확인

**Q2. Colas_2019에서 source_only의 Zone C+D+E 비율은?**
- Tier 7에서 MARD 19%였으므로 Zone C+D+E가 높을 것으로 예상
- 이 수치가 구체적으로 몇 %인지가 "T1D 모델을 건강인에 적용하면 위험하다"의 정량적 근거

**Q3. CORAL/TrAdaBoost가 Zone A를 개선하는가?**
- `mixed` 대비 `coral`/`tradaboost`의 Zone A 증가 여부
- 증가하면: 전이 기법이 임상 안전성도 개선
- 불변이면: RMSE 개선이 임상적으로는 무의미

---

## 3. 단계 1 — 저혈당 안전성

### 왜 저혈당 구간을 별도로 분석하는가?

혈당 70 mg/dL 미만은 **저혈당(hypoglycemia)**으로 정의되며, 즉각적인 의료 대응이 필요한 위험 상태이다.
경미한 저혈당은 어지러움과 발한을 유발하고, 심한 저혈당(<54 mg/dL)은 의식 상실, 경련, 사망에 이를 수 있다.

**70 mg/dL 임계값의 근거:**

- International Hypoglycaemia Study Group.
  "Glucose Concentrations of Less Than 3.0 mmol/L (54 mg/dL) Should Be Reported in Clinical Trials."
  *Diabetes Care*. 2017;40(1):155-157. doi:10.2337/dc16-2215

이 합의문은 저혈당을 Level 1 (<70 mg/dL), Level 2 (<54 mg/dL), Level 3 (의식 변화)으로 분류하며,
70 mg/dL은 치료 조정이 필요한 "경고값(alert value)"으로 국제적으로 합의되었다.

AI 혈당 예측 모델의 핵심 가치 중 하나는 저혈당을 **사전에 경고**하는 것이다.
따라서 모델이 저혈당을 놓치는 비율(sensitivity)은 환자 안전에 직결된다.

문제는, RMSE가 낮아도 저혈당 구간에서는 오히려 성능이 나쁠 수 있다는 점이다.
전체 혈당 데이터의 대부분은 70~180 mg/dL 범위(정상)이므로, RMSE는 정상 구간의 성능에 지배된다.
저혈당은 전체의 1~5%에 불과하여 RMSE에 거의 영향을 주지 않는다.

따라서 저혈당 구간만 분리하여 "이 모델이 저혈당을 얼마나 잘 감지하는가"를 별도 평가해야 한다.
전이학습이 평균 RMSE를 개선하더라도, 저혈당 감지 능력이 악화된다면 환자 안전 관점에서 사용할 수 없다.

**파일:** `tier71_results/clinical/hypo_analysis.csv`

### 판단 기준

| 지표 | 의미 | 기준 |
|---|---|---|
| Sensitivity | 실제 저혈당을 얼마나 잡아내는가 | >= 0.80 이면 수용, < 0.50 이면 위험 |
| Specificity | 거짓 경보 비율 | >= 0.90 이면 수용 |
| PPV | 경보가 울리면 진짜일 확률 | >= 0.30 이면 수용 (저혈당은 희귀 이벤트) |

**판단 기준의 한계:** 위 수치 기준(sensitivity >= 0.80 등)은 본 연구에서 설정한 분석 기준이며,
저혈당 예측 sensitivity에 대한 공식적 국제 합의 수치는 확인하지 못하였다.
다만, 이진 분류 기반 의료 AI 평가에서 sensitivity 0.80은 일반적으로 수용 가능한 수준으로 간주된다.

### 분석 질문

**Q4. 전이학습이 저혈당 sensitivity를 악화시키는가?**
- `tradaboost`의 sensitivity가 `target_only`보다 낮으면: 평균 RMSE는 개선되지만 위험 구간은 악화
- 이 경우 "전이학습이 안전하다"고 말할 수 없음

**Q5. source_only가 저혈당을 전혀 감지하지 못하는 타겟이 있는가?**
- Colas_2019에서 sensitivity < 0.30이면: T1D 모델이 건강인의 저혈당을 구조적으로 놓침
- 건강인의 저혈당 패턴(서서히 하강)이 T1D(급격 하강)와 다르기 때문

---

## 4. 단계 1 — 구간별 RMSE

### 구간 정의의 근거

본 분석에서 사용하는 혈당 구간(<70, 70-180, 180-250, >250 mg/dL)은
국제 합의 기반 CGM 데이터 해석 표준에서 정의된 구간이다:

- Battelino T, Danne T, Bergenstal RM, et al.
  "Clinical Targets for Continuous Glucose Monitoring Data Interpretation:
  Recommendations From the International Consensus on Time in Range."
  *Diabetes Care*. 2019;42(8):1593-1603. doi:10.2337/dci19-0028

이 합의문은 Time in Range(TIR) 지표의 목표를 정의하며,
70-180 mg/dL을 정상 범위, <70을 저혈당, >250을 심각한 고혈당으로 분류한다.

**파일:** `tier71_results/clinical/range_rmse.csv`

### 구간별 RMSE 분해의 한계

구간별 RMSE를 별도 보고하는 것은 직관적이고 실용적이지만,
이를 표준적으로 요구하는 공식 가이드라인은 확인하지 못하였다.
Clarke Error Grid가 구간별 임상 위험을 이미 반영하므로,
구간별 RMSE는 **보완적 분석**으로서 의미가 있다.

### 분석 질문

**Q6. 어떤 구간에서 전이학습의 이익/손실이 집중되는가?**
- `tradaboost`가 `target_only` 대비 RMSE를 개선하는 구간은?
- 개선이 70-180 구간(정상)에 집중되고 <70 구간(저혈당)에서는 악화된다면: 임상적으로 문제

**Q7. source_only의 구간별 RMSE 패턴은?**
- <70 구간의 RMSE가 비정상적으로 높으면: T1D 모델이 저혈당 패턴을 잘못 학습
- >250 구간의 RMSE가 낮으면: T1D 모델이 고혈당은 잘 예측 (T1D 특성상 고혈당 데이터 풍부)

---

## 5. 단계 2 — 핵심 가설 판별

### 방법론적 근거: 동일 도메인 전이 실험

Negative transfer 진단의 표준 방법은 target-only 베이스라인 대비 전이학습 성능을 비교하는 것이다.
이를 체계화한 연구:

- Wang Z, Dai Z, Poczos B, Carbonell J.
  "Characterizing and Avoiding Negative Transfer."
  *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
  2019:11285-11294. doi:10.1109/CVPR.2019.01155

이 논문은 도메인 간 분포 차이(divergence)와 타겟 데이터 양에 따라 negative transfer가 발생하는 조건을
공식화하였다. 본 실험의 "동일 질병 전이(T2D->T2D)"는 이 프레임워크에서 divergence를 최소화한
대조 실험(ablation)에 해당한다.

동일 질병에서도 negative transfer가 발생하면 기법 자체의 한계이고,
동일 질병에서는 해소되면 도메인 갭이 원인이라는 판별 논리는
Wang et al.의 divergence factor 분석과 일치한다.

**파일:** `tier71_results/same_disease/comparison.csv`

### 핵심 판별 로직

`comparison.csv`의 `beats_target_only` 열을 확인한다:

| intra_T2D에서 beats_target_only | 의미 | 결론 |
|---|---|---|
| **True** (CORAL 또는 TrAdaBoost) | 같은 질병에서는 전이가 target_only를 넘는다 | **도메인 갭이 원인**. T1D->T2D에서 못 넘은 것은 질병 유형 차이 때문. |
| **False** (모든 모델) | 같은 질병에서도 전이가 target_only를 못 넘는다 | **기법 한계**. CORAL/TrAdaBoost 자체가 target_only를 초과할 능력 없음. |

### 추가 분석

**Q8. intra_T2D의 negative transfer는 존재하는가?**
- `mixed > target_only` 이면: 같은 질병에서도 negative transfer 발생 (환자 이질성)
- `mixed <= target_only` 이면: 같은 질병에서는 단순 혼합도 안전

**Q9. intra_T2D의 source_only MARD는 cross-disease 대비 얼마나 낮은가?**
- cross: source_only MARD 10.6% (T1D->T2D)
- intra: source_only MARD ? (T2D->T2D)
- 차이가 도메인 갭의 질병 유형 기여분

---

## 6. 단계 3 — Cold Start 교차점

### Cold Start란?

Cold Start(콜드 스타트)는 **새로운 환자가 CGM을 처음 착용했을 때, AI가 이 환자의 혈당을 예측할 수 있는가?**라는 문제이다.

AI 모델은 데이터가 있어야 학습한다. 그런데 새 환자는 데이터가 없거나 극히 적다.
이때 선택지는 세 가지이다:

1. **개인 모델 (personal_only)**: 이 환자의 데이터(예: 1일치)만으로 모델을 학습한다.
   데이터가 너무 적으므로 초기에는 성능이 매우 나쁘지만, 데이터가 쌓이면 점점 좋아진다.

2. **범용 모델 (population)**: 다른 환자들의 데이터로 미리 학습된 모델을 이 환자에게 그대로 적용한다.
   이 환자의 개인 특성을 반영하지 못하지만, 처음부터 어느 정도 작동한다.

3. **전이학습 모델 (tradaboost)**: 범용 모델을 기반으로, 이 환자의 적은 데이터를 활용하여 조정한다.
   범용 모델의 안정성과 개인 데이터의 특이성을 결합한다.

**교차점(crossover)**은 개인 모델이 범용 모델을 넘어서는 시점이다.
교차점이 7일이라면, "이 환자는 CGM 착용 후 7일까지는 범용 모델을 쓰고, 7일 이후부터 개인 모델로 전환하는 것이 최적"이라는 의미이다.
전이학습이 이 교차점을 앞당길 수 있다면, 더 빨리 개인화된 예측을 제공할 수 있다.

### 14일 기준의 근거

본 실험에서 축적 일수를 [1, 3, 7, 14]일로 설정한 근거:

- Danne T, Nimri R, Battelino T, et al.
  "International Consensus on Use of Continuous Glucose Monitoring."
  *Diabetes Care*. 2017;40(12):1631-1640. doi:10.2337/dc17-1600

이 합의문에서 CGM 데이터의 안정적 해석을 위해 최소 **14일**의 착용 기간과 70% 이상의 착용률을 권고한다.
따라서 14일을 상한으로 설정하고, 그 이전 시점(1, 3, 7일)에서의 모델 성능 변화를 추적한다.

### Cold Start 실험 설계의 한계

"개인 모델 vs 범용 모델의 교차점"을 정량적으로 비교한 혈당 예측 논문은 확인하지 못하였다.
이 실험 설계는 추천 시스템(Netflix, Amazon 등)의 cold start 문제에서 차용한 개념이며,
혈당 예측 분야에서는 비교적 새로운 분석 프레임워크이다.
따라서 본 결과는 탐색적(exploratory) 분석으로 해석해야 한다.

**파일:** `tier71_results/cold_start/crossover_{target}.csv`, `cold_start_summary_{target}.csv`

### 핵심 질문

**Q10. personal_only가 population을 넘는 교차점은 며칠인가?**

| 교차점 | 의미 |
|---|---|
| D = 1~3일 | 개인 데이터가 매우 빠르게 유효해짐. 개인화 가치 높음. |
| D = 7~14일 | 상당 기간 범용 모델에 의존해야 함. 전이학습 가치 높음. |
| 교차점 없음 (14일 내) | 2주 이내에는 개인화 불가. 범용 모델 필수. |

**Q11. TrAdaBoost는 personal과 population 사이에서 항상 최적인가?**
- 모든 D에서 `tradaboost`가 다른 두 모델 이하이면: 전이학습이 Cold Start의 보편적 해법
- 특정 D에서만 우위이면: 전이학습의 적용 구간이 제한적

---

## 7. 단계 3 — 도메인 갭별 Cold Start 비교

### 3개 타겟의 Cold Start 패턴을 비교한다

**Q12. 도메인 갭이 클수록 교차점이 늦어지는가?**

| 타겟 | 도메인 갭 (Tier 7) | 예상 교차점 |
|---|---|---|
| CITY | 0.2%p (작음) | 빠름 (1~3일) |
| ShanghaiT2DM | 0.4%p (중간) | 중간 (3~7일) |
| Colas_2019 | 12.4%p (극단적) | 느림 또는 없음 |

예상대로 나오면: "도메인 갭이 클수록 개인화까지 더 오래 걸린다"
예상과 다르면: 도메인 갭 외 요인 (데이터 밀도, 환자 수 등) 존재

**Q13. Colas_2019에서 population이 personal을 압도하는가?**
- 건강인 집단에서 다른 건강인의 데이터가 개인 데이터보다 유용하다면:
  건강인의 혈당 패턴이 균일하여 개인화 필요성 자체가 낮을 수 있음
- 이는 Tier 7의 "CITY에서 전이학습 필요성이 낮다"와 일관된 결론

---

## 8. 종합 결론 도출

### 3개 단계의 결과를 연결하여 답해야 할 최종 질문들

**A. "T1D 모델을 T2D에 쓰면 임상적으로 위험한가?"**
- 단계 1의 Zone C+D+E 비율과 저혈당 sensitivity로 정량적 답변

**B. "전이학습이 임상 안전성을 개선하는가, 평균 RMSE만 개선하는가?"**
- 단계 1의 모델별 Zone A 비교 + 저혈당 sensitivity 비교로 판별

**C. "전이학습이 target_only를 넘을 수 있는 조건이 존재하는가?"**
- 단계 2의 beats_target_only 결과로 판별
- True면 후속 연구 가치 확인, False면 기법 자체의 한계 인정

**D. "신규 환자에게 AI 예측을 며칠 후부터 신뢰할 수 있는가?"**
- 단계 3의 교차점으로 답변
- 전이학습 적용 시 교차점이 앞당겨지면 임상적 실용성 입증

---

## 9. 보고서 작성 체크리스트

- [ ] 단계 1 결과를 Tier7_Results_Analysis.md의 4.5절에 통합
- [ ] 단계 2 결과를 별도 섹션(4.6 또는 5.3)에 추가
- [ ] 단계 3 결과를 별도 섹션(4.7 또는 5.4)에 추가
- [ ] 종합 결론에서 A~D 질문에 대한 답변 기술
- [ ] 모든 시각화(PNG)를 보고서에 삽입
- [ ] 후속 연구 방향을 결과에 기반하여 업데이트
