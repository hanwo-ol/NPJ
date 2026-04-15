import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

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

# ───── Mathematical Domain Features ─────
def kovatchev_risk_index(glucose_array):
    """Calculate LBGI and HBGI for an array of glucose values."""
    # Ensure no zeroes or negatives for log
    g = np.clip(glucose_array, 1.0, None)
    f_g = 1.509 * ((np.log(g)**1.084) - 5.381)
    
    rl = np.where(f_g < 0, f_g, 0)
    rh = np.where(f_g > 0, f_g, 0)
    
    lbgi = 10 * (rl**2)
    hbgi = 10 * (rh**2)
    return lbgi, hbgi

def extract_derived_features(X_raw, n_back, k_covariates):
    """
    X_raw: (N_windows, n_back * (1 + k_covariates))
    Returns exactly matching the number of windows, pure vectorized.
    """
    N = X_raw.shape[0]
    if N == 0:
        return np.empty((0, 10)) # 10 statistical/kinetic features
        
    # X_raw is arranged as: [G_t-5, C1_t-5, C2_t-5..., G_t-4, C1_t-4..., ... G_t, C1_t...]
    n_cols_per_step = 1 + k_covariates
    
    # Extract only the glucose sequence for all windows: (N, n_back)
    g_seq = X_raw[:, 0::n_cols_per_step]
    
    derived = np.zeros((N, 10))
    
    # 1. Velocity (Delta G): G_t - G_{t-1}
    vel = g_seq[:, -1] - g_seq[:, -2]
    # 2. Acceleration: Velocity_t - Velocity_{t-1}
    accel = vel - (g_seq[:, -2] - g_seq[:, -3])
    
    # 3. Window Mean
    w_mean = np.mean(g_seq, axis=1)
    # 4. Window Std
    w_std = np.std(g_seq, axis=1)
    
    # 5. TIR (70-180) %
    tir = np.sum((g_seq >= 70) & (g_seq <= 180), axis=1) / n_back
    # 6. TAR (>180) %
    tar = np.sum(g_seq > 180, axis=1) / n_back
    # 7. TBR (<70) %
    tbr = np.sum(g_seq < 70, axis=1) / n_back
    
    # 8-9. LBGI, HBGI over the window (Using the mean risk index of the window)
    lbgi_seq, hbgi_seq = kovatchev_risk_index(g_seq)
    lbgi = np.mean(lbgi_seq, axis=1)
    hbgi = np.mean(hbgi_seq, axis=1)
    
    # 10. Window AUC (Trapezoidal rule, dt=1)
    auc = np.trapz(g_seq, axis=1)
    
    derived[:, 0] = vel
    derived[:, 1] = accel
    derived[:, 2] = w_mean
    derived[:, 3] = w_std
    derived[:, 4] = tir
    derived[:, 5] = tar
    derived[:, 6] = tbr
    derived[:, 7] = lbgi
    derived[:, 8] = hbgi
    derived[:, 9] = auc
    
    return derived

# ───── Schema & Vectorized Window ─────
def get_numeric_cols(p_files, sample_n=10):
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
    df = df.sort_values('timestamp').reset_index(drop=True)
    ts = df['timestamp'].values
    
    gluc = df['glucose_value_mg_dl'].values.astype(float)
    
    # Engineer Event Decays & Time-Since for covariates BEFORE windowing
    covariate_cols = [c for c in feature_cols if c != 'glucose_value_mg_dl']
    engineered_cols = ['glucose_value_mg_dl']
    
    for c in covariate_cols:
        val = df[c].fillna(0).astype(float)
        engineered_cols.append(c)
        
        # Binary event marker mask
        is_event = val > 0
        
        # Feature: Time-Since-Event (clip at 24 hours = 288 steps of 5-min)
        not_event = ~is_event
        b = not_event.cumsum()
        time_since = (b - b.where(is_event).ffill().fillna(0)).values
        time_since = np.clip(time_since, 0, 288) # Clip at 24h
        df[f"{c}_time_since"] = time_since
        engineered_cols.append(f"{c}_time_since")
        
        # Feature: Exponential Decay (IOB approx, halflife=12 steps = 60 mins)
        decay = val.ewm(halflife=12, ignore_na=True).mean().values
        df[f"{c}_decay"] = decay
        engineered_cols.append(f"{c}_decay")
    
    feat = df[engineered_cols].values.astype(float)
    
    total = n_back + n_fwd
    n_rows = len(df)
    if n_rows < total + 1:
        return np.empty((0, 0)), np.empty(0), []

    gaps = np.diff(ts.astype('int64')) / 1e9
    median_gap = np.median(gaps[gaps > 0]) if (gaps > 0).any() else 0
    if median_gap == 0:
        return np.empty((0, 0)), np.empty(0), []

    threshold = median_gap * 1.5
    ok = gaps <= threshold

    bad = (~ok).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(bad)])
    n_cands = n_rows - total
    if n_cands <= 0:
        return np.empty((0, 0)), np.empty(0), []

    starts = np.arange(n_cands)
    window_bad_count = cs[starts + total - 1] - cs[starts]
    valid = window_bad_count == 0
    idx = starts[valid]

    if len(idx) == 0:
        return np.empty((0, 0)), np.empty(0), []

    back_idx = idx[:, None] + np.arange(n_back)
    target_idx = idx + total - 1

    X_raw = feat[back_idx].reshape(len(idx), -1)
    Y = gluc[target_idx]
    
    # ── Calculate Derived Kinetic/Statistical Features for this window ──
    # Note: Using ONLY the lookback glucose values to absolutely prevent data leakage
    k_covariates_engineered = len(engineered_cols) - 1
    derived_feats = extract_derived_features(X_raw, n_back, k_covariates_engineered)
    
    # ── Time of Day Feature ──
    # Calculate for the 'current' time t (which is the last element of the lookback window)
    t_idx = idx + n_back - 1
    hours = pd.to_datetime(ts[t_idx]).hour.values
    sin_hour = np.sin(2 * np.pi * hours / 24.0).reshape(-1, 1)
    cos_hour = np.cos(2 * np.pi * hours / 24.0).reshape(-1, 1)
    derived_feats = np.hstack([derived_feats, sin_hour, cos_hour])
    
    # Concatenate everything
    X_enhanced = np.hstack([X_raw, derived_feats])
    
    # Naming metadata for the concatenated array
    final_feature_names = []
    for step in range(n_back):
        for c in engineered_cols:
            final_feature_names.append(f"{c}_t-{n_back-step-1}")
            
    derived_names = ['Velocity', 'Acceleration', 'Window_Mean', 'Window_Std', 'TIR', 'TAR', 'TBR', 'LBGI', 'HBGI', 'Window_AUC', 'sin_hour', 'cos_hour']
    final_feature_names.extend(derived_names)
    
    return X_enhanced, Y, final_feature_names

def extract_top_features(rf_model, feature_names, top_k=5):
    """Extract aggregated/ranked names directly."""
    importances = rf_model.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]
    
    top_features = []
    for ri in ranked_indices[:top_k]:
        top_features.append(f"{feature_names[ri]} ({importances[ri]*100:.1f}%)")
        
    return ", ".join(top_features)

# ───── Main ─────
def main():
    root = Path("../3_Glucose-ML-collection").resolve()
    out_md = Path("3_Result_Summary.md")
    tree_out_dir = Path("Decision_Tree_Rules")
    tree_out_dir.mkdir(exist_ok=True)

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
        log(f"  Raw features detected: {n_feat}  (covariate cols excl. glucose: {n_feat-1})")

        train_m, test_m = [], []
        train_y, test_y = [], []
        
        feature_names = []

        try:
            from tqdm import tqdm
            pfiles_iter = tqdm(pfiles, desc=f"Loading {ds}")
        except ImportError:
            pfiles_iter = pfiles
            
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

            sp = int(len(Y) * 0.8)
            train_m.append(Xm[:sp]); test_m.append(Xm[sp:])
            train_y.append(Y[:sp]);  test_y.append(Y[sp:])

        if not train_m:
            log(f"  [SKIP] Skipped (insufficient continuous records)")
            continue

        Tr_m = np.vstack(train_m); Te_m = np.vstack(test_m)
        Tr_y = np.concatenate(train_y); Te_y = np.concatenate(test_y)

        log(f"  Final Feature Dimension: {Tr_m.shape[1]}")
        log(f"  Windows  train={len(Tr_y):,}  test={len(Te_y):,}")

        sc_m = StandardScaler()
        Tr_m_sc = sc_m.fit_transform(Tr_m)
        Te_m_sc = sc_m.transform(Te_m)

        # ── 1. Decision Tree (Explainability, depth=5) ──
        log(f"  Training Decision Tree (depth=5)...")
        model_dt = DecisionTreeRegressor(max_depth=5, random_state=42)
        model_dt.fit(Tr_m_sc, Tr_y)
        pred_dt = model_dt.predict(Te_m_sc)
        
        rmse_dt = np.sqrt(mean_squared_error(Te_y, pred_dt))
        mape_dt = mape(Te_y, pred_dt)
        
        tree_rules = export_text(model_dt, feature_names=feature_names, max_depth=3)
        with open(tree_out_dir / f"{ds}_TreeRules_Tier2.5.txt", 'w', encoding='utf-8') as f:
            f.write(f"Decision Tree Rules for {ds} (Feature Engineering Tier)\n\n")
            f.write(tree_rules)

        # ── 2. Random Forest (Performance, hyperparams loosened to max_depth=20) ──
        log(f"  Training Random Forest (trees=50, max_depth=20)...")
        # n_jobs=-1 handles parallel execution across all CPU cores securely.
        # verbose=2 added to track parallel progression of Tree Building!
        model_rf = RandomForestRegressor(n_estimators=50, max_depth=20, n_jobs=-1, random_state=42, verbose=2)
        model_rf.fit(Tr_m_sc, Tr_y)
        pred_rf = model_rf.predict(Te_m_sc)
        
        rmse_rf = np.sqrt(mean_squared_error(Te_y, pred_rf))
        mae_rf  = mean_absolute_error(Te_y, pred_rf)
        mape_rf = mape(Te_y, pred_rf)
        
        top_5_feats = extract_top_features(model_rf, feature_names, top_k=5)

        log(f"  DT RMSE={rmse_dt:.2f}  MAPE={mape_dt:.1f}%")
        log(f"  RF RMSE={rmse_rf:.2f}  MAE={mae_rf:.2f}  MAPE={mape_rf:.1f}%")
        log(f"  Top Features: {top_5_feats}")

        results.append({
            'Dataset': ds,
            'Windows': len(Tr_y) + len(Te_y),
            'Feature_Dim': Tr_m.shape[1],
            'DT_RMSE': round(rmse_dt, 2),
            'RF_RMSE': round(rmse_rf, 2),
            'RF_MAE': round(mae_rf, 2),
            'RF_MAPE%': round(mape_rf, 1),
            'Top_5_Important_Features': top_5_feats
        })

    try:
        import tabulate
    except ImportError:
        pass
        
    rdf = pd.DataFrame(results)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# 실험 결과 요약 (Tier 2.5 Feature Engineering)\n\n")
        f.write("고난도 임상 특징 파생변수 투입에 따른 Random Forest(max_depth=20) 벤치마크 결과\n\n")
        f.write(rdf.to_markdown(index=False))

    log(f"\n{'='*60}")
    log(f"Done. {len(results)} datasets processed -> 3_Result_Summary.md")

if __name__ == "__main__":
    main()
