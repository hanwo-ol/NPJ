# Tier 7.1 실험 결과 분석 가이드

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

**파일:** `tier71_results/clinical/hypo_analysis.csv`

### 판단 기준

| 지표 | 의미 | 기준 |
|---|---|---|
| Sensitivity | 실제 저혈당을 얼마나 잡아내는가 | >= 0.80 이면 수용, < 0.50 이면 위험 |
| Specificity | 거짓 경보 비율 | >= 0.90 이면 수용 |
| PPV | 경보가 울리면 진짜일 확률 | >= 0.30 이면 수용 (저혈당은 희귀 이벤트) |

### 분석 질문

**Q4. 전이학습이 저혈당 sensitivity를 악화시키는가?**
- `tradaboost`의 sensitivity가 `target_only`보다 낮으면: 평균 RMSE는 개선되지만 위험 구간은 악화
- 이 경우 "전이학습이 안전하다"고 말할 수 없음

**Q5. source_only가 저혈당을 전혀 감지하지 못하는 타겟이 있는가?**
- Colas_2019에서 sensitivity < 0.30이면: T1D 모델이 건강인의 저혈당을 구조적으로 놓침
- 건강인의 저혈당 패턴(서서히 하강)이 T1D(급격 하강)와 다르기 때문

---

## 4. 단계 1 — 구간별 RMSE

**파일:** `tier71_results/clinical/range_rmse.csv`

### 분석 질문

**Q6. 어떤 구간에서 전이학습의 이익/손실이 집중되는가?**
- `tradaboost`가 `target_only` 대비 RMSE를 개선하는 구간은?
- 개선이 70-180 구간(정상)에 집중되고 <70 구간(저혈당)에서는 악화된다면: 임상적으로 문제

**Q7. source_only의 구간별 RMSE 패턴은?**
- <70 구간의 RMSE가 비정상적으로 높으면: T1D 모델이 저혈당 패턴을 잘못 학습
- >250 구간의 RMSE가 낮으면: T1D 모델이 고혈당은 잘 예측 (T1D 특성상 고혈당 데이터 풍부)

---

## 5. 단계 2 — 핵심 가설 판별

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
