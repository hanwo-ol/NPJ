"""
Tier 8: Reproducibility & Limitations — Configuration
=======================================================
Stage 1 잔여 실험 (S1-1 ~ S1-5) 전용 설정.

AGENTS.md 준수 사항:
  - GlobalConfig 상속. SEED, LOOKBACK_STEPS 등 공통 파라미터 재사용.
  - 샘플링 주기별 그룹 분리 유지.
  - 26개 활성 데이터셋 전체를 LODO 대상으로 포함.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from global_config import GlobalConfig


class Tier8Config(GlobalConfig):

    # ─── 출력 경로 ─────────────────────────────────────────────────────────────
    OUT_DIR = Path(__file__).parent / "tier8_results"
    OUT_DIR.mkdir(exist_ok=True)

    # ─── E드라이브 NPZ 캐시 ──────────────────────────────────────────────────
    CACHE_DIR = Path("E:/glucose_cache")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 26개 활성 데이터셋 정의 (997_Active_Datasets.md 기준) ────────────────
    # (데이터셋명, 샘플링주기분, 질환유형, 피험자수)
    DATASET_REGISTRY = {
        # === 1min 그룹 ===
        'CGMacros_Dexcom': {'freq_min': 1,  'disease': 'Mixed', 'n_subjects': 45},
        'CGMacros_Libre':  {'freq_min': 1,  'disease': 'Mixed', 'n_subjects': 45},
        # === 5min 그룹 ===
        'AIDET1D':         {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 29},
        'AZT1D':           {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 25},
        'BIGIDEAs':        {'freq_min': 5,  'disease': 'ND',    'n_subjects': 16},
        'CGMND':           {'freq_min': 5,  'disease': 'ND',    'n_subjects': 45},
        'Colas_2019':      {'freq_min': 5,  'disease': 'Mixed', 'n_subjects': 208},
        'D1NAMO':          {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 9},
        'GLAM':            {'freq_min': 5,  'disease': 'ND',    'n_subjects': 886},
        'Hall_2018':       {'freq_min': 5,  'disease': 'Mixed', 'n_subjects': 57},
        'HUPA-UCM':        {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 25},
        'IOBP2':           {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 440},
        'PEDAP':           {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 103},
        'PhysioCGM':       {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 9},
        'T1D-UOM':         {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 17},
        'UCHTT1DM':        {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 20},
        'RT-CGM':          {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 448},
        'CITY':            {'freq_min': 5,  'disease': 'Mixed', 'n_subjects': 153},
        'SENCE':           {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 143},
        'WISDM':           {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 203},
        'FLAIR':           {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 113},
        'SHD':             {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 200},
        'ReplaceBG':       {'freq_min': 5,  'disease': 'T1D',   'n_subjects': 226},
        # === 15min 그룹 ===
        'Bris-T1D_Open':   {'freq_min': 15, 'disease': 'T1D',   'n_subjects': 20},
        'ShanghaiT1DM':    {'freq_min': 15, 'disease': 'T1D',   'n_subjects': 12},
        'ShanghaiT2DM':    {'freq_min': 15, 'disease': 'T2D',   'n_subjects': 100},
    }

    # ─── 주기 그룹별 데이터셋 이름 목록 ──────────────────────────────────────
    @classmethod
    def datasets_by_group(cls, freq_min: int) -> list:
        return [name for name, info in cls.DATASET_REGISTRY.items()
                if info['freq_min'] == freq_min]

    @classmethod
    def group_names(cls) -> list:
        return sorted(set(info['freq_min'] for info in cls.DATASET_REGISTRY.values()))

    # ─── LightGBM 파라미터 (Tier 7과 동일) ───────────────────────────────────
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

    # ─── S1-1: Within Variation ──────────────────────────────────────────────
    WITHIN_VAR_SEEDS = [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    WITHIN_VAR_TARGET = 'ShanghaiT2DM'
    WITHIN_VAR_SOURCE_GROUP = 15  # 15min 그룹

    # ─── S1-2: LODO ──────────────────────────────────────────────────────────
    MAX_SOURCE_WINDOWS = 1_000_000

    # ─── TrAdaBoost ──────────────────────────────────────────────────────────
    TRADABOOST_N_ITER   = 20
    TRADABOOST_ENSEMBLE = 10

    # ─── S1-5: 3분류 임상 임계값 ─────────────────────────────────────────────
    HYPO_THRESHOLD  = 70.0   # mg/dL
    HYPER_THRESHOLD = 180.0  # mg/dL
