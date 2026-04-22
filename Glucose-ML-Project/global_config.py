"""
Global Configuration — Glucose-ML Project
==========================================
모든 Tier에서 공유하는 고정 하이퍼파라미터와 실험 설계 기준을 정의한다.
각 Tier의 Config 클래스는 이 파일에서 상속하거나 명시적으로 참조한다.

규칙 문서: 999_Preprocessing_Rules.md, AGENTS.md
"""

from pathlib import Path


class GlobalConfig:

    # ─── Paths ────────────────────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).parent
    DATA_ROOT    = PROJECT_ROOT / "003_Glucose-ML-collection"

    # ─── Random Seed ──────────────────────────────────────────────────────────
    SEED = 42

    # ─── Task Definition (Rule 8, 999_Preprocessing_Rules.md) ────────────────
    # 예측 태스크는 샘플 스텝 수로 정의한다.
    # 물리적 예측 시간은 그룹(샘플링 주기)에 따라 달라진다.
    #   5min 주기 → 15분 뒤 예측
    #   15min 주기 → 45분 뒤 예측
    LOOKBACK_STEPS   = 3
    PREDICTION_STEPS = 3

    # ─── Train / Val / Test Split (Temporal Integrity Rule, Rule 5) ───────────
    # Subject(환자) 단위 3-way 분리.
    # 각 Subject의 시계열 순서는 보존된다(내부 shuffle 없음).
    # Subject 배정만 무작위로 결정한다(seed 고정).
    #
    #   Train  : 70%  — 모델 파라미터 학습
    #   Val    : 15%  — Early Stopping, 하이퍼파라미터 선택
    #   Test   : 15%  — 최종 성능 보고 (1회만 사용)
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15   # = 1 - TRAIN_RATIO - VAL_RATIO

    # ─── Strict Preprocessing (Rule 1~4, 999_Preprocessing_Rules.md) ─────────
    MIN_GLUCOSE          = 40.0    # mg/dL, 하한 (Rule 1)
    MAX_GLUCOSE          = 400.0   # mg/dL, 상한 (Rule 1)
    MAX_ROC_MG_DL_MIN    = 20.0    # mg/dL/min, 변화율 상한 (Rule 2)
    MIN_TIME_DIFF_SEC    = 30.0    # 초, 최소 측정 간격 (Rule 2)
    MISSING_GAP_MULT     = 3.0     # 샘플링 주기 × 3 이상 공백 → 결측 (Rule 3)

    # ─── Dynamic Schema (Dataset Dictionary) ─────────────────────────────────
    # 혈당 컬럼명 우선순위 배열. 순서대로 탐색하여 최초 발견된 컬럼을 사용한다.
    GLUCOSE_COL_PRIORITY = [
        'glucose_value_mg_dl',
        'CGM',
        'Value',
        'GlucValue',
        'CGM (mg / dl)',
        'CBG (mg / dl)',
    ]

    # ─── Single Source of Truth (AGENTS.md) ──────────────────────────────────
    # extracted-glucose-files 우선, 없으면 time-augmented, 없으면 top-level CSV.
    GLUCOSE_SUBFOLDER_PRIORITY = [
        '-extracted-glucose-files',
        '-time-augmented',
    ]
    EXCLUDED_SUBFOLDER_KEYWORDS = ['-extended-features']

    # ─── Dataset Exclusion List (Rule 9, 999_Preprocessing_Rules.md) ─────────
    # timestamp가 절대 시각이 아니거나, extracted-glucose-files 자체가
    # 선형 보간된 경우 등록한다.
    EXCLUDED_DATASETS = ['Park_2025']

    # ─── Compute Settings ─────────────────────────────────────────────────────
    N_JOBS = -1   # scikit-learn / LightGBM 공통. -1 = 모든 코어 사용.
