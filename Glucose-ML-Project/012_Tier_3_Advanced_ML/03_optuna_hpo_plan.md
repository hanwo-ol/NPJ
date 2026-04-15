# Tier 3: Optuna HPO (Hyperparameter Optimization) 세부 계획서

본 문서는 `Tier3_Design.md`의 연장선으로, 베이스라인 모델 성능 검토 이후 본격적인 하이퍼파라미터 튜닝(`02_optuna_hpo.py`)을 수행하기 위한 구체적인 방법론과 아키텍처를 정의합니다. 동료 연구자(Co-worker)들이 본 파이프라인의 목적과 결과를 명확히 이해하고 후속 연구(Tier 4)에 연계할 수 있도록 상세히 문서화되었습니다.

---

## 1. 대상 모델 선정 및 축소 (Model Selection)

베이스라인 부스팅 평가 결과(`02_Baseline_Boosting_Results.md`)를 근거로, Optuna 튜닝 대상 모델을 **LightGBM**과 **CatBoost** 두 가지로 축소하여 진행합니다. (XGBoost 제외)

**선정 이유:**
1.  **LightGBM:** 대용량 데이터셋(GLAM, IOBP2, PEDAP 등)에서 압도적인 학습 속도와 최소의 메모리 사용량을 보이며, 동시에 가장 낮은 RMSE를 기록하는 등 대규모 데이터 처리에 최적화되어 있습니다.
2.  **CatBoost:** 중소형 데이터셋 및 다양한 Feature Space에서 전반적으로 1위(8/12개 데이터셋)를 차지하는 뛰어난 범용성과 기본 성능을 입증했습니다.
3.  **선택과 집중:** 한정된 로컬 자원(10시간 오버나이트) 내에서 탐색 효율을 극대화하기 위해, 학습 시간이 가장 오래 걸리고 상대적으로 우위가 적었던 XGBoost를 튜닝 공간에서 제외하여 나머지 두 모델의 탐색 공간(Search Space)에 자원을 집중합니다.

---

## 2. 목적 함수 (Objective Function): 순수 다중 목적 최적화 (Multi-objective)

혈당 예측 모델은 특정 오차(예: 평균 오차)만 줄이는 것이 아니라, 이상치(Extreme values)에 대한 강건함도 함께 확보해야 합니다. 따라서 단일 지표 스칼라 통합 방식을 지양하고, Optuna의 **순수 다중 목적 최적화 (Multi-objective Optimization)** 기능을 활용합니다.

*   **최적화 대상 지표 (Directions = `["minimize", "minimize", "minimize"]`):**
    1.  **RMSE (Root Mean Squared Error):** 큰 예측 오차(급격한 혈당 변동을 놓친 경우)에 더 큰 패널티를 부여.
    2.  **MAE (Mean Absolute Error):** 전반적인 평균 오차의 절대적인 크기 최소화.
    3.  **MAPE (Mean Absolute Percentage Error):** 저혈당 등 값이 작을 때의 상대적인 오차율을 최소화하여 임상적 위험 회피.
*   **결과 도출 (Pareto Front):** Optuna는 세 지표 간의 Trade-off를 분석하여, 어느 한 지표를 손해보지 않고서는 다른 지표를 개선할 수 없는 최적의 파라미터 집합(Pareto Front)을 반환합니다. 향후 시계열 교차검증 스크립트에서 이 Pareto Front 중 '가장 균형 잡힌 모델' 또는 '특정 지표(예: RMSE) 기준 최고 모델'을 선택하여 사용할 수 있습니다.

---

## 3. 학습 환경 통제 및 자원 분배 전략 (Resource Allocation & Pruning)

로컬 PC(32GB RAM, 12코어)에서 오버나이트(Overnight) 학습을 수행하기 위해, 실무 관행(Optuna KDD Paper 등)에 기반한 엄격한 자원 통제 스케줄러를 도입합니다.

### 3.1 동적 타임아웃(Timeout) 할당 알고리즘
전체 실험의 하드 리미트(Total Budget)를 **10시간(36,000초)**으로 고정합니다. 이 10시간을 각 데이터셋의 크기(Window 수 기준)에 비례하여 동적으로 분배합니다.
*   수천만 윈도우를 가진 GLAM은 전체 시간의 상당 부분(예: 3~4시간)을 할당받고, 중소형 데이터셋은 몇 분 ~ 수십 분만 할당받도록 스케줄링하여 무한 루프나 특정 데이터셋의 자원 독점을 방지합니다. (※ 필요시 로그 스케일 가중치 적용)

### 3.2 Trial 및 Pruner 설정
*   **n_trials = 100 (고정):** 트리 모델의 하이퍼파라미터 차원(약 5~8개)을 고려할 때 $10 \times D$ 법칙에 의거, 100번의 탐색을 최대 목표로 삼습니다.
*   **Optuna Pruner 도입:** 탐색 중간 단계(Early-stopping)에서 성능 향상 가망이 없는 하위(Bottom) 파라미터 조합은 `optuna.pruners.MedianPruner` 등을 통해 즉각 폐기(Pruning)합니다. 이를 통해 할당된 타임아웃 내에서 유효한 탐색 횟수(Trials)를 비약적으로 늘립니다.

---

## 4. HPO 결과 저장 스키마 (JSON Format & Documentation)

튜닝 완료 후 도출된 Pareto Front 최적 파라미터들은 향후 다른 스크립트(`03_timeseries_cv.py` 등)나 Co-worker들이 쉽게 재사용할 수 있도록 명시적인 JSON 구조로 저장합니다.

*   **파일명:** `012_Tier_3_Advanced_ML/hpo_results_best_params.json`
*   **저장 스키마 (Schema Design):**
    ```json
    {
      "metadata": {
        "tuning_date": "YYYY-MM-DD HH:MM",
        "total_time_budget_sec": 36000,
        "n_trials_per_dataset": 100
      },
      "datasets": {
        "GLAM": {
          "LightGBM": {
            "pareto_front": [
              {
                "trial_id": 12,
                "metrics": {"RMSE": 12.5, "MAE": 9.1, "MAPE": 8.5},
                "params": {"learning_rate": 0.05, "num_leaves": 63, "max_depth": 10, "...": "..."}
              },
              {
                "trial_id": 45,
                "metrics": {"RMSE": 12.8, "MAE": 8.9, "MAPE": 8.4},
                "params": {"...": "..."}
              }
            ]
          },
          "CatBoost": {
             "...": "..."
          }
        },
        "BIGIDEAs": {
          "...": "..."
        }
      }
    }
    ```
*   **Co-worker 활용 가이드:** 향후 교차 검증 및 스태킹 앙상블 진행 시, 이 JSON 파일을 로드하여 각 데이터셋/모델별로 `pareto_front` 리스트 내 첫 번째 딕셔너리의 `params`를 주입하면 즉시 최적화된 모델을 재현할 수 있습니다.
