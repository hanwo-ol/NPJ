"""
Tier 6: Centralized Configuration
=================================
GlobalConfig를 상속하여 Tier 6 전용 설정만 추가로 정의한다.
SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, N_JOBS는 GlobalConfig에서 상속된다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from global_config import GlobalConfig


class Tier6Config(GlobalConfig):

    # ─── Tier 6 전용 경로 ─────────────────────────────────────────────────────
    OUT_DIR  = Path(__file__).parent / "tier6_results"
    LOG_FILE = OUT_DIR / "tier6_experiment_C_D.log"
    OUT_DIR.mkdir(exist_ok=True)

    # ─── LightGBM 전역 모델 파라미터 ──────────────────────────────────────────
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': GlobalConfig.SEED,
        'n_jobs': GlobalConfig.N_JOBS,
    }
    LGBM_ROUNDS         = 1000
    LGBM_EARLY_STOPPING = 50

    # ─── LightGBM 특수화 모델 (SP) 파라미터 ───────────────────────────────────
    LGBM_SP_PARAMS = {
        'objective': 'regression',
        'learning_rate': 0.01,
        'verbose': -1,
        'seed': GlobalConfig.SEED,
        'n_jobs': GlobalConfig.N_JOBS,
    }
    LGBM_SP_ROUNDS         = 500
    LGBM_SP_EARLY_STOPPING = 20

    # ─── 피처 설정 ────────────────────────────────────────────────────────────
    N_GLOBAL_TS_FEATURES = 20   # 전역 시계열 피처 차원 (글루코스 룩백 + 파생)
    TCA_N_COMPONENTS     = 10   # Transfer Component Analysis 성분 수
