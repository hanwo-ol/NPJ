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

# ───── Schema negotiation ─────
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

# ───── Vectorised window builder ─────
def build_windows(df, feature_cols, n_back=6, n_fwd=6):
    df = df.sort_values('timestamp').reset_index(drop=True)
    ts = df['timestamp'].values
    feat = df[feature_cols].values.astype(float)
    gluc = df['glucose_value_mg_dl'].values.astype(float)

    total = n_back + n_fwd
    n_rows = len(df)
    if n_rows < total + 1:
        return np.empty((0, n_back * len(feature_cols))), np.empty(0)

    gaps = np.diff(ts.astype('int64')) / 1e9
    median_gap = np.median(gaps[gaps > 0]) if (gaps > 0).any() else 0
    if median_gap == 0:
        return np.empty((0, n_back * len(feature_cols))), np.empty(0)

    threshold = median_gap * 1.5
    ok = gaps <= threshold

    bad = (~ok).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(bad)])
    n_cands = n_rows - total
    if n_cands <= 0:
        return np.empty((0, n_back * len(feature_cols))), np.empty(0)

    starts = np.arange(n_cands)
    window_bad_count = cs[starts + total - 1] - cs[starts]
    valid = window_bad_count == 0
    idx = starts[valid]

    if len(idx) == 0:
        return np.empty((0, n_back * len(feature_cols))), np.empty(0)

    back_idx = idx[:, None] + np.arange(n_back)
    target_idx = idx + total - 1

    X_multi = feat[back_idx].reshape(len(idx), -1)
    Y = gluc[target_idx]
    return X_multi, Y

def extract_top_features(rf_model, feature_cols, n_back=6, top_k=3):
    """Map the flattened feature importances back to their original column names."""
    importances = rf_model.feature_importances_
    # Flattened order: [feat1_t1, feat2_t1, ... featN_t1, feat1_t2, feat2_t2... ]
    # We aggregate importance by original feature column
    n_feat = len(feature_cols)
    agg_imp = np.zeros(n_feat)
    
    for i, imp in enumerate(importances):
        feat_idx = i % n_feat
        agg_imp[feat_idx] += imp
        
    ranked_indices = np.argsort(agg_imp)[::-1]
    
    top_features = []
    for ri in ranked_indices[:top_k]:
        top_features.append(f"{feature_cols[ri]} ({agg_imp[ri]*100:.1f}%)")
        
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
        log(f"  Features detected: {n_feat}  (covariate cols excl. glucose: {n_feat-1})")

        train_m, test_m = [], []
        train_y, test_y = [], []

        for pf in pfiles:
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

            Xm, Y = build_windows(df, feat_cols)
            if len(Y) < 10:
                continue

            sp = int(len(Y) * 0.8)
            train_m.append(Xm[:sp]); test_m.append(Xm[sp:])
            train_y.append(Y[:sp]);  test_y.append(Y[sp:])

        if not train_m:
            log(f"  [SKIP] Skipped (insufficient continuous records)")
            continue

        Tr_m = np.vstack(train_m); Te_m = np.vstack(test_m)
        Tr_y = np.concatenate(train_y); Te_y = np.concatenate(test_y)

        log(f"  Windows  train={len(Tr_y):,}  test={len(Te_y):,}")

        # Scale features
        sc_m = StandardScaler()
        Tr_m_sc = sc_m.fit_transform(Tr_m)
        Te_m_sc = sc_m.transform(Te_m)

        # ── 1. Decision Tree (Explainability) ──
        log(f"  Training Decision Tree (depth=5)...")
        model_dt = DecisionTreeRegressor(max_depth=5, random_state=42)
        model_dt.fit(Tr_m_sc, Tr_y)
        pred_dt = model_dt.predict(Te_m_sc)
        
        rmse_dt = np.sqrt(mean_squared_error(Te_y, pred_dt))
        mape_dt = mape(Te_y, pred_dt)
        
        # Save Tree Rules
        flat_feature_names = []
        for step in range(6): # n_back=6
            for col in feat_cols:
                flat_feature_names.append(f"{col}_t-{5-step}")
                
        tree_rules = export_text(model_dt, feature_names=flat_feature_names, max_depth=3)
        with open(tree_out_dir / f"{ds}_TreeRules.txt", 'w', encoding='utf-8') as f:
            f.write(f"Decision Tree Rules for {ds}\n\n")
            f.write(tree_rules)

        # ── 2. Random Forest (Performance & Feature Importance) ──
        log(f"  Training Random Forest (trees=50, max_depth=15)...")
        # n_jobs=-1 handles parallel execution across all CPU cores securely
        model_rf = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
        model_rf.fit(Tr_m_sc, Tr_y) # Sklearn RF takes unscaled or scaled, standard scaler doesn't hurt RF
        pred_rf = model_rf.predict(Te_m_sc)
        
        rmse_rf = np.sqrt(mean_squared_error(Te_y, pred_rf))
        mae_rf  = mean_absolute_error(Te_y, pred_rf)
        mape_rf = mape(Te_y, pred_rf)
        
        # Top 3 feature extraction
        top_3_feats = extract_top_features(model_rf, feat_cols)

        log(f"  DT RMSE={rmse_dt:.2f}  MAPE={mape_dt:.1f}%")
        log(f"  RF RMSE={rmse_rf:.2f}  MAE={mae_rf:.2f}  MAPE={mape_rf:.1f}%")
        log(f"  Top Features: {top_3_feats}")

        results.append({
            'Dataset': ds,
            'Windows': len(Tr_y) + len(Te_y),
            'Covariates': n_feat - 1,
            'DT_RMSE': round(rmse_dt, 2),
            'DT_MAPE%': round(mape_dt, 1),
            'RF_RMSE': round(rmse_rf, 2),
            'RF_MAE': round(mae_rf, 2),
            'RF_MAPE%': round(mape_rf, 1),
            'Top_3_Important_Features': top_3_feats
        })

    # ── Write summary ──
    try:
        import tabulate
    except ImportError:
        pass
        
    rdf = pd.DataFrame(results)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# 실험 결과 요약 (Tier 2 ML Baseline)\n\n")
        f.write("Decision Tree 및 Random Forest 벤치마크 결과표 (SVM 데이터 무결성 보존을 위해 제거됨)\n\n")
        f.write(rdf.to_markdown(index=False))

    log(f"\n{'='*60}")
    log(f"Done. {len(results)} datasets processed -> 3_Result_Summary.md")

if __name__ == "__main__":
    main()
