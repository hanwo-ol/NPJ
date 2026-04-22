"""
Tier 7: Cross-Disease Transfer — Configuration
================================================
Scenario D: T1D (Source) → T2D/Mixed (Target)

AGENTS.md 준수 사항:
  - TRAIN_RATIO, SEED, LOOKBACK_STEPS 등 공통 파라미터는 GlobalConfig에서 상속.
  - 샘플링 주기별로 실험 그룹을 분리하여 서로 다른 주기 데이터를 동일 모델에 혼합하지 않음.
  - Tier-specific 값(출력 경로, 그룹 정의, LightGBM 파라미터)만 재정의.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from global_config import GlobalConfig


class Tier7Config(GlobalConfig):

    # ─── 출력 경로 ─────────────────────────────────────────────────────────────
    OUT_DIR = Path(__file__).parent / "tier7_results"
    OUT_DIR.mkdir(exist_ok=True)

    # ─── 실험 그룹 정의 (샘플링 주기별 분리, AGENTS.md L27-28) ───────────────
    #
    # PREDICTION_STEPS = GlobalConfig.PREDICTION_STEPS = 3
    # 물리적 예측 시간:
    #   5분 주기 그룹  → 3 steps × 5분 = 15분 뒤 예측
    #   15분 주기 그룹 → 3 steps × 15분 = 45분 뒤 예측
    #
    # 소스와 타겟은 반드시 같은 주기 그룹에 속해야 한다.

    EXPERIMENT_GROUPS = {
        # 그룹명: (sampling_min, source_datasets, target_datasets)
        '5min': (
            5,
            {   # Source: T1D 대규모 (5분)
                'RT-CGM': 5,
                'IOBP2':  5,
                'FLAIR':  5,
                'SENCE':  5,
                'WISDM':  5,
                'PEDAP':  5,
            },
            {   # Target: Mixed/T2D 경향 (5분)
                # CITY: T1D+T2D+ND 혼합 → T2D 경향 집단
                # Colas_2019: 건강인→T2D 발병 위험군
                'CITY':       5,
                'Colas_2019': 5,
            },
        ),
        '15min': (
            15,
            {   # Source: T1D (15분)
                'ShanghaiT1DM': 15,
                'Bris-T1D_Open': 15,
            },
            {   # Target: T2D (15분) — primary cross-disease 실험
                'ShanghaiT2DM': 15,
            },
        ),
    }

    # ─── LightGBM 파라미터 ────────────────────────────────────────────────────
    LGBM_PARAMS = {
        'objective':         'regression',
        'metric':            'rmse',
        'learning_rate':     0.05,
        'num_leaves':        63,
        'feature_fraction':  0.8,
        'bagging_fraction':  0.8,
        'bagging_freq':      5,
        'min_child_samples': 20,
        'verbose':           -1,
        'seed':              GlobalConfig.SEED,
        'n_jobs':            GlobalConfig.N_JOBS,
    }
    LGBM_ROUNDS         = 2000
    LGBM_EARLY_STOPPING = 100

    # ─── TrAdaBoost 파라미터 ──────────────────────────────────────────────────
    TRADABOOST_N_ITER   = 20
    TRADABOOST_ENSEMBLE = 10

    # ─── 학습 곡선: 타겟 데이터 비율 ──────────────────────────────────────────
    LEARNING_CURVE_RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    # ─── SHAP 분석 샘플 수 ────────────────────────────────────────────────────
    SHAP_SAMPLE_N = 2000
