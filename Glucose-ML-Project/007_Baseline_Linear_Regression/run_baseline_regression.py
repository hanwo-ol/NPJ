import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ───── Utility ─────
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    valid = y_true != 0
    if valid.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100

def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode(), flush=True)

# ───── Schema negotiation ─────
def get_numeric_cols(p_files, sample_n=10):
    """Sample a handful of patient files and build the union of numeric-coercible columns."""
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

# ───── Vectorised window builder ─────
def build_windows(df, feature_cols, n_back=6, n_fwd=6):
    """
    Vectorised sliding-window construction.
    Returns X_uni (n_samples, n_back),
            X_multi (n_samples, n_back * n_features),
            Y (n_samples,)
    Only windows whose timestamps are perfectly equidistant are kept.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    ts = df['timestamp'].values                         # datetime64
    gluc = df['glucose_value_mg_dl'].values.astype(float)
    feat = df[feature_cols].values.astype(float)        # (rows, n_features)

    total = n_back + n_fwd
    n_rows = len(df)
    if n_rows < total + 1:
        return np.empty((0, n_back)), np.empty((0, n_back * len(feature_cols))), np.empty(0)

    # Pre-compute all inter-row gaps in seconds (vectorised)
    gaps = np.diff(ts.astype('int64')) / 1e9             # seconds
    median_gap = np.median(gaps[gaps > 0]) if (gaps > 0).any() else 0
    if median_gap == 0:
        return np.empty((0, n_back)), np.empty((0, n_back * len(feature_cols))), np.empty(0)

    threshold = median_gap * 1.5

    # Boolean mask: True where the gap to the NEXT row is acceptable
    ok = gaps <= threshold                               # length n_rows-1

    # For each candidate start index i, the window [i .. i+total-1] is valid
    # iff ALL gaps ok[i], ok[i+1], …, ok[i+total-2] are True.
    # Use a cumsum trick for O(n) evaluation.
    bad = (~ok).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(bad)])            # length n_rows
    n_cands = n_rows - total
    if n_cands <= 0:
        return np.empty((0, n_back)), np.empty((0, n_back * len(feature_cols))), np.empty(0)

    starts = np.arange(n_cands)
    window_bad_count = cs[starts + total - 1] - cs[starts]
    valid = window_bad_count == 0
    idx = starts[valid]

    if len(idx) == 0:
        return np.empty((0, n_back)), np.empty((0, n_back * len(feature_cols))), np.empty(0)

    # Gather windows via fancy indexing
    back_idx = idx[:, None] + np.arange(n_back)          # (n_valid, n_back)
    target_idx = idx + total - 1

    X_uni = gluc[back_idx]                               # (n_valid, n_back)
    X_multi = feat[back_idx].reshape(len(idx), -1)       # (n_valid, n_back*n_feat)
    Y = gluc[target_idx]                                 # (n_valid,)
    return X_uni, X_multi, Y

# ───── Main ─────
def main():
    root = Path("../3_Glucose-ML-collection").resolve()
    out_md = Path("3_Result_Summary.md")

    datasets = sorted([
        d for d in root.iterdir()
        if d.is_dir() and (d / f"{d.name}-time-augmented").exists()
    ])

    results = []

    for dset in datasets:
        ds = dset.name
        aug = dset / f"{ds}-time-augmented"
        pfiles = sorted(aug.glob("*.csv"))
        if not pfiles:
            continue

        log(f"\n{'='*60}")
        log(f"[{ds}] {len(pfiles)} subjects")

        feat_cols = get_numeric_cols(pfiles)
        n_feat = len(feat_cols)
        log(f"  Features detected: {n_feat}  (covariate cols excl. glucose: {n_feat-1})")

        train_u, test_u = [], []
        train_m, test_m = [], []
        train_y, test_y = [], []

        for pf in pfiles:
            df = pd.read_csv(pf, low_memory=False)
            if 'timestamp' not in df.columns or 'glucose_value_mg_dl' not in df.columns:
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'glucose_value_mg_dl'])

            # Enforce column schema & coerce to numeric
            for c in feat_cols:
                if c not in df.columns:
                    df[c] = 0.0
                else:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

            Xu, Xm, Y = build_windows(df, feat_cols)
            if len(Y) < 10:
                continue

            sp = int(len(Y) * 0.8)
            train_u.append(Xu[:sp]); test_u.append(Xu[sp:])
            train_m.append(Xm[:sp]); test_m.append(Xm[sp:])
            train_y.append(Y[:sp]);  test_y.append(Y[sp:])

        if not train_u:
            log(f"  [SKIP] Skipped (insufficient records)")
            continue

        Tr_u = np.vstack(train_u); Te_u = np.vstack(test_u)
        Tr_m = np.vstack(train_m); Te_m = np.vstack(test_m)
        Tr_y = np.concatenate(train_y); Te_y = np.concatenate(test_y)

        log(f"  Windows  train={len(Tr_y):,}  test={len(Te_y):,}")

        # ── Experiment A: Univariate Ridge ──
        sc_u = StandardScaler()
        model_u = Ridge(alpha=1.0)
        model_u.fit(sc_u.fit_transform(Tr_u), Tr_y)
        pred_u = model_u.predict(sc_u.transform(Te_u))
        rmse_u = np.sqrt(mean_squared_error(Te_y, pred_u))
        mae_u  = mean_absolute_error(Te_y, pred_u)
        mape_u = mape(Te_y, pred_u)

        # ── Experiment B: Multivariate Ridge ──
        sc_m = StandardScaler()
        model_m = Ridge(alpha=1.0)
        model_m.fit(sc_m.fit_transform(Tr_m), Tr_y)
        pred_m = model_m.predict(sc_m.transform(Te_m))
        rmse_m = np.sqrt(mean_squared_error(Te_y, pred_m))
        mae_m  = mean_absolute_error(Te_y, pred_m)
        mape_m = mape(Te_y, pred_m)

        delta = rmse_u - rmse_m

        log(f"  Univar  RMSE={rmse_u:.2f}  MAE={mae_u:.2f}  MAPE={mape_u:.1f}%")
        log(f"  Multivar RMSE={rmse_m:.2f}  MAE={mae_m:.2f}  MAPE={mape_m:.1f}%")
        log(f"  Delta RMSE = {delta:+.2f}  {'(improved)' if delta > 0 else '(no gain)'}")

        results.append({
            'Dataset': ds,
            'Subjects': len(pfiles),
            'Windows': len(Tr_y) + len(Te_y),
            'Covariates': n_feat - 1,
            'Uni_RMSE': round(rmse_u, 2),
            'Uni_MAE': round(mae_u, 2),
            'Uni_MAPE%': round(mape_u, 1),
            'Multi_RMSE': round(rmse_m, 2),
            'Multi_MAE': round(mae_m, 2),
            'Multi_MAPE%': round(mape_m, 1),
            'RMSE_Delta': round(delta, 2),
        })

    # ── Write summary ──
    rdf = pd.DataFrame(results)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# 실험 결과 단순 정리 (Result Summary)\n\n")
        f.write("Ridge Linear Regression — Univariate vs Multivariate 비교\n")
        f.write("- Lookback = 6 steps, Horizon = 6 steps\n")
        f.write("- Train/Test = 80/20 chronological split per patient\n\n")
        f.write(rdf.to_markdown(index=False))
        f.write("\n\n> **RMSE_Delta > 0** → 다변량 투입 시 오차가 감소(성능 향상)\n")

    log(f"\n{'='*60}")
    log(f"Done. {len(results)} datasets processed → 3_Result_Summary.md")

if __name__ == "__main__":
    main()
