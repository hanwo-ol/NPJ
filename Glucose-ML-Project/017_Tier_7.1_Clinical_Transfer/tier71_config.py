"""
Tier 7.1: Clinical Transfer Analysis — Configuration
=====================================================
Tier 7의 후속 분석을 위한 설정.

- 단계 1: 임상 안전성 분석 (Clarke Error Grid, 저혈당 분석)
- 단계 2: 동일 질병 전이 (T2D -> T2D)
- 단계 3: N=1 Cold Start 개인화

GlobalConfig를 상속하며, Tier-specific 값만 재정의한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from global_config import GlobalConfig


class Tier71Config(GlobalConfig):

    # --- 출력 경로 ----------------------------------------------------------
    OUT_DIR = Path(__file__).parent / "tier71_results"
    OUT_DIR.mkdir(exist_ok=True)

    # --- Tier 7 예측값 경로 (단계 0에서 저장) --------------------------------
    TIER7_PRED_DIR = (Path(__file__).parent.parent
                      / "016_Tier_7_Cross_Disease" / "tier7_results" / "predictions")

    # --- Clarke Error Grid 임계값 ------------------------------------------
    CLARKE_THRESHOLD_MG_DL = 20.0   # Zone A: +-20 mg/dL (< 70 mg/dL)
    CLARKE_THRESHOLD_PCT   = 0.20   # Zone A: +-20%     (>= 70 mg/dL)

    # --- 혈당 구간 정의 (mg/dL) ---------------------------------------------
    HYPO_THRESHOLD    = 70.0
    NORMAL_LOWER      = 70.0
    NORMAL_UPPER      = 180.0
    HYPER_THRESHOLD_1 = 180.0
    HYPER_THRESHOLD_2 = 250.0

    # --- 동일 질병 전이 설정 (단계 2) ----------------------------------------
    SAME_DISEASE_DATASET  = "ShanghaiT2DM"
    SAME_DISEASE_N_SOURCE = 50   # 100명 중 50명을 소스로

    # --- LightGBM 파라미터 (Tier 7과 동일) ----------------------------------
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

    # --- TrAdaBoost 파라미터 ------------------------------------------------
    TRADABOOST_N_ITER   = 20
    TRADABOOST_ENSEMBLE = 10

    # --- Cold Start 설정 (단계 3) -------------------------------------------
    COLD_START_DAYS    = [1, 3, 7, 14]
    COLD_START_TARGETS = ["ShanghaiT2DM", "CITY", "Colas_2019"]
    # 샘플링 주기 매핑 (주기별 그룹 분리 준수)
    COLD_START_SAMPLING = {
        "ShanghaiT2DM": 15,
        "CITY":          5,
        "Colas_2019":    5,
    }

    # --- Cold Start 병렬 처리 -----------------------------------------------
    COLD_START_MAX_WORKERS = 4  # 환자 단위 병렬 스레드 수
