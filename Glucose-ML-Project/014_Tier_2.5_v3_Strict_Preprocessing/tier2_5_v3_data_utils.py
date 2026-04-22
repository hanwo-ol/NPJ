import os
import pandas as pd
import numpy as np
from pathlib import Path
from tier2_5_v3_config import Tier2_5_v3_Config


# ─────────────────────────────────────────────────────────
# Single Source of Truth: Folder Resolution
# ─────────────────────────────────────────────────────────
def resolve_glucose_files(ds_path: Path, ds_name: str) -> list:
    """
    Returns the list of glucose CSV files to load from a dataset directory,
    following the Single Source of Truth priority defined in Config.
    Prevents duplicate loading when multiple folder variants coexist.
    """
    # Priority 1: check for designated glucose subfolders in order
    for suffix in Tier2_5_v3_Config.GLUCOSE_SUBFOLDER_PRIORITY:
        subfolder = ds_path / (ds_name + suffix)
        if subfolder.exists():
            files = [f for f in subfolder.glob("*.csv")
                     if "metadata" not in f.name.lower()]
            if files:
                return files

    # Fallback: top-level CSVs only (not recursing into subfolders)
    top_files = [f for f in ds_path.glob("*.csv")
                 if "metadata" not in f.name.lower()]
    if top_files:
        return top_files

    # Last resort: rglob but exclude known auxiliary subfolders
    all_files = []
    for f in ds_path.rglob("*.csv"):
        if "metadata" in f.name.lower():
            continue
        excluded = any(kw in str(f) for kw in Tier2_5_v3_Config.EXCLUDED_SUBFOLDER_KEYWORDS)
        if not excluded:
            all_files.append(f)
    return all_files


# ─────────────────────────────────────────────────────────
# Schema Resolution
# ─────────────────────────────────────────────────────────
def load_and_resolve_schema(csv_path: Path):
    """Loads a CSV and dynamically resolves the target glucose column."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None, None

    for candidate in Tier2_5_v3_Config.GLUCOSE_COL_PRIORITY:
        if candidate in df.columns:
            return df, candidate

    return df, None


# ─────────────────────────────────────────────────────────
# Strict Filters
# ─────────────────────────────────────────────────────────
def apply_strict_filters(df: pd.DataFrame, target_col: str):
    """
    Applies the strict 2507 paper rules:
    1. Non-numeric removal
    2. Range Filter (40-400 mg/dL)
    3. RoC Filter (dt < 30s or dG/dt > 20 mg/dL/min)
    """
    stats = {'total_initial': len(df), 'dropped_non_numeric': 0,
             'dropped_range': 0, 'dropped_roc': 0}

    if 'timestamp' not in df.columns:
        return df, stats

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 1. Non-numeric
    initial_valid = df[target_col].notna().sum()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    stats['dropped_non_numeric'] = initial_valid - df[target_col].notna().sum()

    # 2. Range Filter
    mask_range = (df[target_col] < Tier2_5_v3_Config.MIN_GLUCOSE) | \
                 (df[target_col] > Tier2_5_v3_Config.MAX_GLUCOSE)
    stats['dropped_range'] = int(mask_range.sum())
    df.loc[mask_range, target_col] = np.nan

    # 3. RoC Filter
    dt_sec = df['timestamp'].diff().dt.total_seconds()
    dG = df[target_col].diff()

    with np.errstate(divide='ignore', invalid='ignore'):
        roc_mg_dl_min = np.abs(dG / (dt_sec / 60.0))

    # First row has NaN dt, so mask_time[0]=False (NaN < 30 is False) — intentional.
    mask_time = dt_sec < Tier2_5_v3_Config.MIN_TIME_DIFF_SEC
    mask_roc  = roc_mg_dl_min > Tier2_5_v3_Config.MAX_ROC_MG_DL_MIN
    mask_invalid = mask_time | mask_roc

    stats['dropped_roc'] = int((mask_invalid & df[target_col].notna()).sum())
    df.loc[mask_invalid, target_col] = np.nan

    return df, stats


# ─────────────────────────────────────────────────────────
# Sampling Rate Detection
# ─────────────────────────────────────────────────────────
def detect_sampling_rate_min(ts: np.ndarray) -> float:
    """
    Detects the sampling rate in minutes from a sorted timestamp array.
    Returns 0.0 if it cannot be determined.
    """
    if len(ts) < 2:
        return 0.0
    gaps_sec = np.diff(ts.astype('int64')) / 1e9
    positive_gaps = gaps_sec[gaps_sec > 0]
    if len(positive_gaps) == 0:
        return 0.0
    median_gap_sec = float(np.median(positive_gaps))
    return median_gap_sec / 60.0


# ─────────────────────────────────────────────────────────
# Derived Features
# ─────────────────────────────────────────────────────────
def extract_derived_features(X_raw: np.ndarray, n_back: int) -> np.ndarray:
    """Kinematic and variability features on a NaN-free glucose window."""
    N = X_raw.shape[0]
    if N == 0:
        return np.empty((0, 8))

    derived = np.zeros((N, 8))

    v_t  = X_raw[:, -1] - X_raw[:, -2] if n_back >= 2 else np.zeros(N)
    v_t1 = X_raw[:, -2] - X_raw[:, -3] if n_back >= 3 else np.zeros(N)
    a_t  = v_t - v_t1 if n_back >= 3 else np.zeros(N)

    w_mean = np.mean(X_raw, axis=1)
    w_std  = np.std(X_raw, axis=1)
    v_window = np.diff(X_raw, axis=1) if n_back >= 2 else np.zeros((N, 1))
    sd1 = np.std(v_window, axis=1) / np.sqrt(2.0)

    tir = np.sum((X_raw >= 70) & (X_raw <= 180), axis=1) / n_back
    tar = np.sum(X_raw > 180, axis=1) / n_back
    tbr = np.sum(X_raw < 70,  axis=1) / n_back

    derived[:, 0] = v_t;   derived[:, 1] = a_t
    derived[:, 2] = w_mean; derived[:, 3] = w_std
    derived[:, 4] = tir;   derived[:, 5] = tar
    derived[:, 6] = tbr;   derived[:, 7] = sd1
    return derived


# ─────────────────────────────────────────────────────────
# Window Builder (No Interpolation + Full-Window Validity)
# ─────────────────────────────────────────────────────────
def build_windows_no_interpolation(df: pd.DataFrame, target_col: str,
                                   sampling_rate_min: float):
    """
    Builds sliding windows with fixed step counts (LOOKBACK_STEPS, PREDICTION_STEPS).
    sampling_rate_min is used only to compute the 3x gap threshold for missing period detection.
    Drops any window that fails Full-Window Validity check (Rule 7).
    """
    if 'timestamp' not in df.columns or sampling_rate_min <= 0:
        return np.empty((0, 0)), np.empty(0), [], {'skipped_windows': 0}

    n_back = Tier2_5_v3_Config.LOOKBACK_STEPS
    n_fwd  = Tier2_5_v3_Config.PREDICTION_STEPS
    total_len = n_back + n_fwd

    ts   = df['timestamp'].values
    gluc = df[target_col].values
    n_rows = len(df)

    if n_rows < total_len:
        return np.empty((0, 0)), np.empty(0), [], {'skipped_windows': 0}

    max_allowed_gap_sec = sampling_rate_min * 60.0 * Tier2_5_v3_Config.MISSING_GAP_MULTIPLIER
    gaps_sec = np.diff(ts.astype('int64')) / 1e9

    # is_missing_gap[k] = 1 means "gap between row k-1 and row k is too large"
    is_missing_gap = (gaps_sec >= max_allowed_gap_sec).astype(np.int32)
    is_missing_gap = np.concatenate([[0], is_missing_gap])

    is_nan_gluc = np.isnan(gluc).astype(np.int32)

    cs_gap = np.cumsum(is_missing_gap)
    cs_nan = np.cumsum(is_nan_gluc)

    n_cands = n_rows - total_len + 1
    if n_cands <= 0:
        return np.empty((0, 0)), np.empty(0), [], {'skipped_windows': 0}

    starts = np.arange(n_cands)

    # Full-Window Validity (Rule 7): check NaN and gap across entire span [start, start+total_len-1]
    full_nan_count = cs_nan[starts + total_len - 1] - cs_nan[starts] + is_nan_gluc[starts]
    full_gap_count = cs_gap[starts + total_len - 1] - cs_gap[starts]

    valid_mask = (full_nan_count == 0) & (full_gap_count == 0)

    idx = starts[valid_mask]
    skipped_count = n_cands - len(idx)

    if len(idx) == 0:
        return np.empty((0, 0)), np.empty(0), [], {'skipped_windows': skipped_count}

    back_idx   = idx[:, None] + np.arange(n_back)
    target_idx = idx + total_len - 1

    X_raw = gluc[back_idx]
    Y     = gluc[target_idx]

    derived_feats = extract_derived_features(X_raw, n_back)

    t_idx    = idx + n_back - 1
    hours    = pd.to_datetime(ts[t_idx]).hour.values
    sin_hour = np.sin(2 * np.pi * hours / 24.0).reshape(-1, 1)
    cos_hour = np.cos(2 * np.pi * hours / 24.0).reshape(-1, 1)

    X_enhanced = np.hstack([X_raw, derived_feats, sin_hour, cos_hour])

    feature_names = [f"glucose_t-{n_back-step-1}" for step in range(n_back)]
    feature_names += ['Velocity', 'Acceleration', 'Window_Mean', 'Window_Std',
                      'TIR', 'TAR', 'TBR', 'SD1', 'tod_sin', 'tod_cos']

    return X_enhanced, Y, feature_names, {'skipped_windows': skipped_count}
