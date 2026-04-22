"""
Tier 7: Data Utilities
=======================
샘플링 주기가 동일한 그룹 내에서만 데이터를 혼합한다 (AGENTS.md L27-28).
GlobalConfig.LOOKBACK_STEPS, PREDICTION_STEPS를 사용하여 윈도우를 생성한다.
GlobalConfig.MIN_GLUCOSE, MAX_GLUCOSE로 혈당 범위를 필터링한다.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '012_Tier_3_Advanced_ML'))

from global_config import GlobalConfig
from tier7_config import Tier7Config
from tier3_data_utils import kovatchev_risk_index, log


# ─── 피처 추출 ────────────────────────────────────────────────────────────────

def extract_features(glucose: np.ndarray,
                     timestamps: pd.DatetimeIndex) -> dict:
    """
    GlobalConfig.LOOKBACK_STEPS 크기의 윈도우에서 피처를 추출한다.
    피처 차원은 샘플링 주기와 무관하게 동일하다 — 그룹 내 혼합이 가능.
    """
    g = glucose  # shape: (LOOKBACK_STEPS,)
    k = len(g)

    feats = {}

    # 1. Lag features (LOOKBACK_STEPS개 고정)
    for step in range(1, k + 1):
        feats[f'lag_{step}'] = float(g[-step])

    # 2. Delta features
    feats['delta_1']  = float(g[-1] - g[-2]) if k >= 2 else 0.0
    feats['delta_2']  = float(g[-1] - g[-3]) if k >= 3 else 0.0
    feats['delta_sq'] = feats['delta_1'] ** 2

    # 3. Window statistics
    feats['win_mean'] = float(np.mean(g))
    feats['win_std']  = float(np.std(g))
    feats['win_min']  = float(np.min(g))
    feats['win_max']  = float(np.max(g))
    feats['win_cv']   = (feats['win_std'] / feats['win_mean']
                         if feats['win_mean'] > 0 else 0.0)

    # 4. Risk indices (Kovatchev) — 배열 반환이므로 mean으로 스칼라 변환
    lbgi_arr, hbgi_arr = kovatchev_risk_index(g)
    feats['lbgi'] = float(np.mean(lbgi_arr))
    feats['hbgi'] = float(np.mean(hbgi_arr))

    # 5. Time features
    if timestamps is not None and len(timestamps) > 0:
        ts   = timestamps[-1]
        hour = ts.hour + ts.minute / 60.0
        feats['hour_sin'] = float(np.sin(2 * np.pi * hour / 24))
        feats['hour_cos'] = float(np.cos(2 * np.pi * hour / 24))
        feats['is_night'] = float(hour < 6 or hour >= 22)
        feats['dow_sin']  = float(np.sin(2 * np.pi * ts.dayofweek / 7))
        feats['dow_cos']  = float(np.cos(2 * np.pi * ts.dayofweek / 7))
    else:
        feats.update({'hour_sin': 0.0, 'hour_cos': 0.0,
                      'is_night': 0.0, 'dow_sin': 0.0, 'dow_cos': 0.0})

    # 6. T2D 도메인 피처
    if timestamps is not None and len(timestamps) > 0:
        hour_val = timestamps[-1].hour
        feats['fasting_proxy']  = float(0 <= hour_val < 6) * feats['win_mean']
        feats['postmeal_rise']  = float(max(0.0, g[-1] - float(np.min(g))))
        feats['high_persist']   = float(np.mean(g > 180))
        feats['in_range_frac']  = float(np.mean((g >= 70) & (g <= 180)))
    else:
        feats.update({'fasting_proxy': 0.0, 'postmeal_rise': 0.0,
                      'high_persist': 0.0, 'in_range_frac': 0.0})

    return feats


def build_windows(df: pd.DataFrame) -> tuple:
    """
    단일 환자 DataFrame → (X_array, y_array) 슬라이딩 윈도우.
    LOOKBACK_STEPS, PREDICTION_STEPS는 GlobalConfig에서 참조.
    """
    lb = GlobalConfig.LOOKBACK_STEPS
    ph = GlobalConfig.PREDICTION_STEPS
    min_len = lb + ph

    glucose    = df['glucose_value_mg_dl'].values.astype(float)
    timestamps = pd.DatetimeIndex(df['timestamp'])

    if len(glucose) < min_len:
        return None, None

    X_rows, y_rows = [], []
    for i in range(lb, len(glucose) - ph + 1):
        g_win  = glucose[i - lb: i]
        ts_win = timestamps[i - lb: i]
        y_val  = glucose[i + ph - 1]

        feats = extract_features(g_win, ts_win)
        X_rows.append(feats)
        y_rows.append(y_val)

    if not X_rows:
        return None, None

    X = pd.DataFrame(X_rows).values.astype(np.float32)
    y = np.array(y_rows, dtype=np.float32)
    return X, y


# ─── 단일 데이터셋 로딩 ───────────────────────────────────────────────────────

def load_dataset(ds_name: str, max_patients: int = None) -> tuple:
    """
    datasets-extracted-glucose-files 에서 환자별 CSV 로딩.
    GlobalConfig.MIN_GLUCOSE / MAX_GLUCOSE로 혈당 범위 필터링.
    """
    ds_dir = (GlobalConfig.DATA_ROOT / ds_name
              / f"{ds_name}-extracted-glucose-files")
    if not ds_dir.exists():
        log(f"  [SKIP] {ds_name}: directory not found")
        return None, None

    files = sorted(ds_dir.glob("*.csv"))
    if max_patients:
        files = files[:max_patients]

    X_list, y_list = [], []
    for f in tqdm(files, desc=f"  {ds_name}", leave=False):
        try:
            df = pd.read_csv(f, low_memory=False)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['glucose_value_mg_dl'] = pd.to_numeric(
                df['glucose_value_mg_dl'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'glucose_value_mg_dl'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            mask = ((df['glucose_value_mg_dl'] >= GlobalConfig.MIN_GLUCOSE) &
                    (df['glucose_value_mg_dl'] <= GlobalConfig.MAX_GLUCOSE))
            df = df[mask].reset_index(drop=True)

            X, y = build_windows(df)
            if X is not None:
                X_list.append(X)
                y_list.append(y)
        except Exception as e:
            log(f"    [ERROR] {f.name}: {e}")

    if not X_list:
        return None, None

    return np.vstack(X_list), np.concatenate(y_list)


# ─── 그룹별 소스 풀 / 타겟 분할 ──────────────────────────────────────────────

def load_source_pool(group_name: str) -> tuple:
    """
    지정 그룹의 소스(T1D) 데이터셋 전체를 로딩하여 합산 반환.
    """
    _, source_defs, _ = Tier7Config.EXPERIMENT_GROUPS[group_name]
    log(f"Loading source pool [{group_name}]...")
    X_all, y_all = [], []
    for ds_name in source_defs:
        X, y = load_dataset(ds_name)
        if X is not None:
            X_all.append(X)
            y_all.append(y)
            log(f"  {ds_name}: {len(X):,} windows")
    if not X_all:
        raise RuntimeError(f"No source data for group '{group_name}'")
    Xs = np.vstack(X_all)
    ys = np.concatenate(y_all)
    log(f"  Total source [{group_name}]: {len(Xs):,} windows")
    return Xs, ys


def load_target_split(ds_name: str) -> dict:
    """
    타겟 데이터셋을 환자 단위 70/15/15로 분할하여 반환.
    GlobalConfig.TRAIN_RATIO / VAL_RATIO 사용.
    """
    ds_dir = (GlobalConfig.DATA_ROOT / ds_name
              / f"{ds_name}-extracted-glucose-files")
    files  = sorted(ds_dir.glob("*.csv"))
    n      = len(files)

    n_train = int(n * GlobalConfig.TRAIN_RATIO)
    n_val   = int(n * GlobalConfig.VAL_RATIO)

    splits = {
        'train': files[:n_train],
        'val':   files[n_train: n_train + n_val],
        'test':  files[n_train + n_val:],
    }

    result = {}
    for split_name, split_files in splits.items():
        X_list, y_list = [], []
        for f in split_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['glucose_value_mg_dl'] = pd.to_numeric(
                    df['glucose_value_mg_dl'], errors='coerce')
                df = df.dropna(subset=['timestamp', 'glucose_value_mg_dl'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                mask = ((df['glucose_value_mg_dl'] >= GlobalConfig.MIN_GLUCOSE) &
                        (df['glucose_value_mg_dl'] <= GlobalConfig.MAX_GLUCOSE))
                df = df[mask].reset_index(drop=True)
                X, y = build_windows(df)
                if X is not None:
                    X_list.append(X)
                    y_list.append(y)
            except Exception:
                pass
        X_s = np.vstack(X_list)    if X_list else np.empty((0, 1), dtype=np.float32)
        y_s = np.concatenate(y_list) if y_list else np.empty(0,    dtype=np.float32)
        result[split_name] = (X_s, y_s)
        log(f"  {ds_name} {split_name}: {len(X_s):,} windows ({len(split_files)} patients)")

    return result
