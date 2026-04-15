"""
Tier 3: Shared Data Loading & Feature Engineering Utilities
============================================================
Tier 2.5_v2 파이프라인에서 검증된 feature engineering 로직을 공유 모듈로 추출.
01~05 스크립트 전체에서 import하여 사용한다.

동일한 데이터 전처리 → 동일한 Feature 생성 → 공정한 모델 비교 보장.
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════
# 1. Utility Functions
# ═══════════════════════════════════════════════════

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)"""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    valid = y_true != 0
    if valid.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100


def log(msg):
    """Unicode-safe print wrapper."""
    try:
        print(msg, flush=True)
    except Exception:
        print(msg.encode('ascii', 'replace').decode(), flush=True)


def downcast_to_float32(X):
    """float64 → float32 다운캐스팅 (메모리 50% 절약)"""
    if isinstance(X, np.ndarray):
        return X.astype(np.float32)
    return X


# ═══════════════════════════════════════════════════
# 2. Mathematical Domain Feature Extraction
#    (Tier 2.5_v2 동일 로직)
# ═══════════════════════════════════════════════════

def kovatchev_risk_index(glucose_array):
    """Kovatchev의 LBGI/HBGI 비대칭 위험 지수 계산."""
    g = np.clip(glucose_array, 1.0, None)
    f_g = 1.509 * ((np.log(g) ** 1.084) - 5.381)
    rl = np.where(f_g < 0, f_g, 0)
    rh = np.where(f_g > 0, f_g, 0)
    lbgi = 10 * (rl ** 2)
    hbgi = 10 * (rh ** 2)
    return lbgi, hbgi


def extract_derived_features(X_raw, n_back, k_covariates):
    """
    혈당 Lookback 시퀀스로부터 역학/임상/위상 파생 변수 14종 추출.
    Returns: (N, 14) array
      [0] Velocity, [1] Acceleration, [2] Window_Mean, [3] Window_Std,
      [4] TIR, [5] TAR, [6] TBR, [7] LBGI, [8] HBGI, [9] Window_AUC,
      [10] Jerk, [11] SD1, [12] tod_sin, [13] tod_cos
    ※ tod_sin/tod_cos는 별도로 주입하므로 이 함수에서는 12종만 반환.
    """
    N = X_raw.shape[0]
    if N == 0:
        return np.empty((0, 12))

    n_cols_per_step = 1 + k_covariates
    g_seq = X_raw[:, 0::n_cols_per_step]  # glucose columns only

    derived = np.zeros((N, 12), dtype=np.float32)

    # Kinematics (1st ~ 3rd order backward difference)
    v_t = g_seq[:, -1] - g_seq[:, -2]
    v_t1 = g_seq[:, -2] - g_seq[:, -3]
    v_t2 = g_seq[:, -3] - g_seq[:, -4]

    a_t = v_t - v_t1
    a_t1 = v_t1 - v_t2
    jerk = a_t - a_t1

    # Statistical summaries
    w_mean = np.mean(g_seq, axis=1)
    w_std = np.std(g_seq, axis=1)

    # Poincaré SD1 (short-term variability)
    v_window = np.diff(g_seq, axis=1)
    sd1 = np.std(v_window, axis=1) / np.sqrt(2.0)

    # Clinical ranges
    tir = np.sum((g_seq >= 70) & (g_seq <= 180), axis=1) / n_back
    tar = np.sum(g_seq > 180, axis=1) / n_back
    tbr = np.sum(g_seq < 70, axis=1) / n_back

    # Risk indices & AUC
    lbgi_seq, hbgi_seq = kovatchev_risk_index(g_seq)
    lbgi = np.mean(lbgi_seq, axis=1)
    hbgi = np.mean(hbgi_seq, axis=1)
    auc = np.trapz(g_seq, axis=1)

    derived[:, 0] = v_t
    derived[:, 1] = a_t
    derived[:, 2] = w_mean
    derived[:, 3] = w_std
    derived[:, 4] = tir
    derived[:, 5] = tar
    derived[:, 6] = tbr
    derived[:, 7] = lbgi
    derived[:, 8] = hbgi
    derived[:, 9] = auc
    derived[:, 10] = jerk
    derived[:, 11] = sd1

    return derived


# ═══════════════════════════════════════════════════
# 3. Schema Detection & Windowing
# ═══════════════════════════════════════════════════

def get_numeric_cols(p_files, sample_n=10):
    """환자 파일들을 샘플링하여 numeric-coercible 컬럼 합집합을 구성."""
    numeric_cols = set()
    for pf in p_files[:sample_n]:
        hdr = pd.read_csv(pf, low_memory=False)
        for c in hdr.columns:
            if pd.to_numeric(hdr[c], errors='coerce').notna().any():
                numeric_cols.add(c)
    exclude = {'timestamp', 'person_id', 'DeviceDtTm', 'Unnamed: 0', 'subject'}
    numeric_cols -= exclude
    numeric_cols = sorted(numeric_cols)
    if 'glucose_value_mg_dl' in numeric_cols:
        numeric_cols.remove('glucose_value_mg_dl')
    return ['glucose_value_mg_dl'] + numeric_cols


def build_windows_with_features(df, feature_cols, n_back=6, n_fwd=6):
    """
    Tier 2.5_v2 완전 동일 윈도우 빌더.
    - 이벤트 decay + time_since + 핵심 공변량 multi-frequency sin/cos 임베딩
    - 역학 파생 변수 12종 + tod_sin/cos 2종
    Returns: (X_enhanced, Y, feature_names)
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    ts = df['timestamp'].values
    gluc = df['glucose_value_mg_dl'].values.astype(float)

    covariate_cols = [c for c in feature_cols if c != 'glucose_value_mg_dl']
    engineered_cols = ['glucose_value_mg_dl']

    core_keywords = ['ins', 'carb', 'meal', 'cal', 'dose']

    for c in covariate_cols:
        val = df[c].fillna(0).astype(float)
        engineered_cols.append(c)

        # Time-Since-Event
        is_event = val > 0
        not_event = ~is_event
        b = not_event.cumsum()
        time_since = (b - b.where(is_event).ffill().fillna(0)).values
        time_since = np.clip(time_since, 0, 288)
        df[f"{c}_time_since"] = time_since
        engineered_cols.append(f"{c}_time_since")

        # Exponential Decay (IOB/COB approximation)
        decay = val.ewm(halflife=12, ignore_na=True).mean().values
        df[f"{c}_decay"] = decay
        engineered_cols.append(f"{c}_decay")

        # Core Covariate Multi-Frequency Positional Encoding
        if any(k in c.lower() for k in core_keywords):
            for P_steps, label in [(6, '30m'), (12, '1h'), (24, '2h')]:
                phase = 2.0 * np.pi * (time_since / P_steps)
                df[f"{c}_sin_{label}"] = np.sin(phase)
                df[f"{c}_cos_{label}"] = np.cos(phase)
                engineered_cols.extend([f"{c}_sin_{label}", f"{c}_cos_{label}"])

    feat = df[engineered_cols].values.astype(np.float32)
    total = n_back + n_fwd
    n_rows = len(df)
    if n_rows < total + 1:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32), []

    # Gap detection — reject windows with timestamp discontinuities
    gaps = np.diff(ts.astype('int64')) / 1e9
    median_gap = np.median(gaps[gaps > 0]) if (gaps > 0).any() else 0
    if median_gap == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32), []

    threshold = median_gap * 1.5
    ok = gaps <= threshold

    bad = (~ok).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(bad)])
    n_cands = n_rows - total
    if n_cands <= 0:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32), []

    starts = np.arange(n_cands)
    window_bad_count = cs[starts + total - 1] - cs[starts]
    valid = window_bad_count == 0
    idx = starts[valid]
    if len(idx) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32), []

    back_idx = idx[:, None] + np.arange(n_back)
    target_idx = idx + total - 1

    X_raw = feat[back_idx].reshape(len(idx), -1)
    Y = gluc[target_idx].astype(np.float32)

    # Derived kinetic/clinical features
    k_covariates_engineered = len(engineered_cols) - 1
    derived_feats = extract_derived_features(X_raw, n_back, k_covariates_engineered)

    # Time-of-day circadian encoding
    t_idx = idx + n_back - 1
    hours = pd.to_datetime(ts[t_idx]).hour.values
    sin_hour = np.sin(2 * np.pi * hours / 24.0).reshape(-1, 1).astype(np.float32)
    cos_hour = np.cos(2 * np.pi * hours / 24.0).reshape(-1, 1).astype(np.float32)
    derived_feats = np.hstack([derived_feats, sin_hour, cos_hour])

    # Final concatenation: [raw_lookback | derived_features]
    X_enhanced = np.hstack([X_raw, derived_feats])

    # Feature name metadata
    final_feature_names = []
    for step in range(n_back):
        for col_name in engineered_cols:
            final_feature_names.append(f"{col_name}_t-{n_back - step - 1}")

    derived_names = [
        'Velocity', 'Acceleration', 'Window_Mean', 'Window_Std',
        'TIR', 'TAR', 'TBR', 'LBGI', 'HBGI', 'Window_AUC',
        'Jerk', 'SD1', 'tod_sin', 'tod_cos'
    ]
    final_feature_names.extend(derived_names)

    return X_enhanced, Y, final_feature_names


# ═══════════════════════════════════════════════════
# 4. Dataset Discovery & Loading
# ═══════════════════════════════════════════════════

def discover_datasets(data_root=None):
    """003_Glucose-ML-collection 내의 time-augmented 데이터셋 목록을 탐색."""
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent / "003_Glucose-ML-collection"
    else:
        data_root = Path(data_root).resolve()

    datasets = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and (d / f"{d.name}-time-augmented").exists()
    ])
    return datasets


def load_dataset(dset_path, train_ratio=0.8):
    """
    단일 데이터셋을 로드하여 train/test 분할까지 수행.

    Returns:
        dict with keys:
            'name', 'X_train', 'y_train', 'X_test', 'y_test',
            'feature_names', 'n_windows', 'feature_dim'
        or None if the dataset should be skipped.
    """
    ds = dset_path.name
    aug = dset_path / f"{ds}-time-augmented"
    pfiles = sorted(aug.glob("*.csv"))
    if not pfiles:
        return None

    log(f"\n{'=' * 60}")
    log(f"[{ds}] {len(pfiles)} subjects")

    feat_cols = get_numeric_cols(pfiles)
    log(f"  Raw features detected: {len(feat_cols)}")

    train_m, test_m = [], []
    train_y, test_y = [], []
    feature_names = []

    pfiles_iter = tqdm(pfiles, desc=f"Loading {ds}", leave=False)

    for pf in pfiles_iter:
        df = pd.read_csv(pf, low_memory=False)
        if 'timestamp' not in df.columns or 'glucose_value_mg_dl' not in df.columns:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'glucose_value_mg_dl'])

        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        Xm, Y, fnames = build_windows_with_features(df, feat_cols)
        if len(Y) < 10:
            continue
        if not feature_names:
            feature_names = fnames

        sp = int(len(Y) * train_ratio)
        train_m.append(Xm[:sp])
        test_m.append(Xm[sp:])
        train_y.append(Y[:sp])
        test_y.append(Y[sp:])

    if not train_m:
        log(f"  [SKIP] Insufficient data")
        return None

    X_train = np.vstack(train_m)
    X_test = np.vstack(test_m)
    y_train = np.concatenate(train_y)
    y_test = np.concatenate(test_y)

    # Explicit cleanup of intermediate lists
    del train_m, test_m, train_y, test_y
    gc.collect()

    log(f"  Feature Dim: {X_train.shape[1]}  |  "
        f"Train: {len(y_train):,}  Test: {len(y_test):,}")

    return {
        'name': ds,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'n_windows': len(y_train) + len(y_test),
        'feature_dim': X_train.shape[1],
    }


def extract_top_features(importances, feature_names, top_k=5):
    """Feature importance 배열로부터 상위 k개 변수명+기여도 문자열 반환."""
    ranked = np.argsort(importances)[::-1]
    parts = []
    for ri in ranked[:top_k]:
        parts.append(f"{feature_names[ri]} ({importances[ri] * 100:.1f}%)")
    return ", ".join(parts)
