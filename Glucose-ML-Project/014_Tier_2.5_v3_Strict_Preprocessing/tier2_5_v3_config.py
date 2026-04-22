"""
Tier 2.5 v3 Configuration
==========================
GlobalConfig를 상속하여 Tier 2.5 v3 전용 경로 및 그룹 정의만 추가한다.
공통 파라미터(LOOKBACK_STEPS, PREDICTION_STEPS, 분리 비율 등)는
GlobalConfig에서 상속되므로 여기서 재정의하지 않는다.
"""

import sys
from pathlib import Path

# GlobalConfig는 프로젝트 루트에 위치한다.
sys.path.insert(0, str(Path(__file__).parent.parent))
from global_config import GlobalConfig


class Tier2_5_v3_Config(GlobalConfig):

    # ─── Tier-specific Paths ──────────────────────────────────────────────────
    OUTPUT_DIR = GlobalConfig.PROJECT_ROOT / "014_Tier_2.5_v3_Strict_Preprocessing"

    # ─── Sampling Rate Groups ─────────────────────────────────────────────────
    # 그룹 분리는 성능 보고용이다. 모든 그룹이 동일한 LOOKBACK_STEPS / PREDICTION_STEPS를 사용한다.
    # pred_time_min: PREDICTION_STEPS × 해당 주기(분). 결과 보고 시 명시한다.
    RATE_GROUPS = {
        '1min':  {'rate_min': 0.5,  'rate_max': 2.0,  'pred_time_min': 3},
        '5min':  {'rate_min': 2.0,  'rate_max': 8.0,  'pred_time_min': 15},
        '15min': {'rate_min': 8.0,  'rate_max': 20.0, 'pred_time_min': 45},
    }
    MIN_SUBJECTS_PER_GROUP = 5


import os
os.makedirs(Tier2_5_v3_Config.OUTPUT_DIR, exist_ok=True)
