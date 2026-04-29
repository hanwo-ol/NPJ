"""
Tier 7.1: Same-Disease Transfer (T2D -> T2D)
==============================================
단계 2: target_only 초과 조건 탐색

ShanghaiT2DM(100명)을 50:50으로 분할하여 동일 질병 내 전이 실험을 수행한다.
결과를 Tier 7의 cross-disease(T1D->T2D) 결과와 비교하여,
"도메인 갭이 원인인지, 전이 기법 한계인지"를 판별한다.

실행: python 017_Tier_7.1_Clinical_Transfer/tier71_same_disease.py
"""

import sys
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '016_Tier_7_Cross_Disease'))
sys.path.insert(0, str(Path(__file__).parent.parent / '013_Tier_6_Domain_Adaptation'))

from global_config import GlobalConfig
from tier71_config import Tier71Config
from tier7_data_utils import build_windows, log
from tier7_tradaboost import TrAdaBoostRegressor
from tier6_transfer_utils import apply_coral


# --- 평가 지표 -------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mard(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

def evaluate(y_true, y_pred, label: str) -> dict:
    r, m, d = rmse(y_true, y_pred), mae(y_true, y_pred), mard(y_true, y_pred)
    print(f"  [{label:15s}]  RMSE={r:.2f}  MAE={m:.2f}  MARD={d:.1f}%")
    return {'model': label, 'rmse': r, 'mae': m, 'mard': d}


# --- LightGBM 학습 --------------------------------------------------------

def train_lgbm(X_tr, y_tr, X_val, y_val, sample_weight=None) -> lgb.Booster:
    p      = dict(Tier71Config.LGBM_PARAMS)
    ds_tr  = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    return lgb.train(
        p, ds_tr,
        num_boost_round=Tier71Config.LGBM_ROUNDS,
        valid_sets=[ds_val],
        callbacks=[
            lgb.early_stopping(Tier71Config.LGBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )


# --- 환자 단위 데이터 로딩 -------------------------------------------------

def load_patient_data(ds_name: str) -> list:
    """환자별 (X, y) 리스트를 반환한다."""
    ds_dir = (GlobalConfig.DATA_ROOT / ds_name
              / f"{ds_name}-extracted-glucose-files")
    files  = sorted(ds_dir.glob("*.csv"))
    patients = []

    for f in tqdm(files, desc=f"  Loading {ds_name}", leave=False, ncols=70):
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
                patients.append((X, y, f.stem))
        except Exception:
            pass

    return patients


# --- 5-way 실험 (환자 분할 방식) -------------------------------------------

def run_same_disease_experiment():
    """ShanghaiT2DM 내부 분할 5-way 비교."""
    out_dir = Tier71Config.OUT_DIR / "same_disease"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ds_name  = Tier71Config.SAME_DISEASE_DATASET
    n_source = Tier71Config.SAME_DISEASE_N_SOURCE

    print(f"\n{'='*58}")
    print(f"  Same-Disease Transfer: {ds_name} ({n_source}:{100-n_source})")
    print(f"{'='*58}")

    # 환자 로딩
    patients = load_patient_data(ds_name)
    n_total  = len(patients)
    print(f"  Total patients: {n_total}")

    # 환자 분할: source / target
    rng = np.random.default_rng(GlobalConfig.SEED)
    indices = rng.permutation(n_total)
    src_idx = indices[:n_source]
    tgt_idx = indices[n_source:]

    # Source 데이터 합산
    Xs = np.vstack([patients[i][0] for i in src_idx])
    ys = np.concatenate([patients[i][1] for i in src_idx])
    print(f"  Source: {len(Xs):,} windows ({len(src_idx)} patients)")

    # Target 데이터 분할 (70/15/15)
    tgt_patients = [patients[i] for i in tgt_idx]
    n_tgt   = len(tgt_patients)
    n_train = int(n_tgt * GlobalConfig.TRAIN_RATIO)
    n_val   = int(n_tgt * GlobalConfig.VAL_RATIO)

    train_p = tgt_patients[:n_train]
    val_p   = tgt_patients[n_train:n_train + n_val]
    test_p  = tgt_patients[n_train + n_val:]

    X_tr = np.vstack([p[0] for p in train_p])
    y_tr = np.concatenate([p[1] for p in train_p])
    X_val = np.vstack([p[0] for p in val_p]) if val_p else np.empty((0, Xs.shape[1]))
    y_val = np.concatenate([p[1] for p in val_p]) if val_p else np.empty(0)
    X_te = np.vstack([p[0] for p in test_p]) if test_p else np.empty((0, Xs.shape[1]))
    y_te = np.concatenate([p[1] for p in test_p]) if test_p else np.empty(0)

    print(f"  Target train: {len(X_tr):,} ({len(train_p)} patients)")
    print(f"  Target val:   {len(X_val):,} ({len(val_p)} patients)")
    print(f"  Target test:  {len(X_te):,} ({len(test_p)} patients)")

    if len(X_te) == 0:
        print("  [SKIP] No test data.")
        return

    results = []

    # 독립 모델 3종 병렬
    print("\n  [1-3/5] Source-Only / Target-Only / Mixed (parallel)...")

    def _train_source():
        return train_lgbm(Xs, ys, X_val, y_val)

    def _train_target():
        return train_lgbm(X_tr, y_tr, X_val, y_val)

    def _train_mixed():
        return train_lgbm(np.vstack([Xs, X_tr]),
                          np.concatenate([ys, y_tr]),
                          X_val, y_val)

    task_map = {
        'source_only': _train_source,
        'target_only': _train_target,
        'mixed':       _train_mixed,
    }
    model_cache = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fn): name for name, fn in task_map.items()}
        for future in tqdm(as_completed(futures), total=3,
                           desc="    Training", leave=False, ncols=70):
            name = futures[future]
            model_cache[name] = future.result()

    for label in ['source_only', 'target_only', 'mixed']:
        preds = model_cache[label].predict(X_te)
        results.append(evaluate(y_te, preds, label))
        np.savez(pred_dir / f"{label}.npz", y_true=y_te, y_pred=preds)

    # CORAL
    print("  [4/5] CORAL...")
    Xs_c    = apply_coral(Xs, X_tr)
    m_coral = train_lgbm(np.vstack([Xs_c, X_tr]),
                         np.concatenate([ys, y_tr]),
                         X_val, y_val)
    preds_coral = m_coral.predict(X_te)
    results.append(evaluate(y_te, preds_coral, 'coral'))
    np.savez(pred_dir / "coral.npz", y_true=y_te, y_pred=preds_coral)

    # TrAdaBoost
    print(f"  [5/5] TrAdaBoost ({Tier71Config.TRADABOOST_N_ITER} iterations)...")
    tada = TrAdaBoostRegressor()
    tada.fit(Xs, ys, X_tr, y_tr, X_val, y_val)
    preds_tada = tada.predict(X_te)
    results.append(evaluate(y_te, preds_tada, 'tradaboost'))
    np.savez(pred_dir / "tradaboost.npz", y_true=y_te, y_pred=preds_tada)

    # 결과 저장
    df = pd.DataFrame(results)
    df.insert(0, 'target', ds_name)
    df.insert(0, 'experiment', 'intra_T2D')
    df.to_csv(out_dir / "5way_intra_T2D.csv", index=False, encoding='utf-8-sig')
    print(f"\nSaved: {out_dir / '5way_intra_T2D.csv'}")

    # Cross-disease 결과와 비교
    cross_csv = (Tier71Config.TIER7_PRED_DIR.parent / "5way_all_targets.csv")
    if cross_csv.exists():
        df_cross = pd.read_csv(cross_csv)
        df_cross_shd = df_cross[df_cross['target'] == ds_name].copy()
        df_cross_shd.insert(0, 'experiment', 'cross_T1D_to_T2D')

        df_intra = df.copy()
        df_comp  = pd.concat([df_cross_shd, df_intra], ignore_index=True)

        # beats_target_only 판별
        for exp in df_comp['experiment'].unique():
            mask_exp = df_comp['experiment'] == exp
            tgt_rmse = df_comp.loc[mask_exp & (df_comp['model'] == 'target_only'),
                                   'rmse'].values
            if len(tgt_rmse) > 0:
                df_comp.loc[mask_exp, 'beats_target_only'] = \
                    df_comp.loc[mask_exp, 'rmse'] < tgt_rmse[0]

        df_comp.to_csv(out_dir / "comparison.csv",
                       index=False, encoding='utf-8-sig')
        print(f"Saved: {out_dir / 'comparison.csv'}")
        print("\n=== Cross-Disease vs Same-Disease Comparison ===")
        print(df_comp.to_string(index=False))

    # 시각화
    plot_same_disease(df, out_dir)


def plot_same_disease(df: pd.DataFrame, out_dir: Path):
    """동일 질병 전이 5-way 막대 그래프."""
    ORDER  = ['source_only', 'target_only', 'mixed', 'coral', 'tradaboost']
    COLORS = {
        'source_only': '#8b949e', 'target_only': '#e63946',
        'mixed': '#f4a261', 'coral': '#457b9d', 'tradaboost': '#2a9d8f',
    }

    sub = df.set_index('model').reindex(ORDER).dropna().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    bars = ax.bar(sub['model'], sub['rmse'],
                  color=[COLORS[m] for m in sub['model']],
                  width=0.6, edgecolor='none')
    for bar, row in zip(bars, sub.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f"{row.rmse:.2f}",
                ha='center', va='bottom', color='#e6edf3', fontsize=9)

    ax.set_ylabel('RMSE (mg/dL)', color='#e6edf3')
    ax.set_title('Same-Disease Transfer: T2D -> T2D (ShanghaiT2DM 50:50)',
                 color='#e6edf3', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#e6edf3')
    ax.spines[:].set_color('#30363d')
    ax.yaxis.grid(True, color='#30363d', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_dir / "5way_intra_T2D.png", dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_dir / '5way_intra_T2D.png'}")


# --- 구동부 ----------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tier 7.1 Step 2: Same-Disease Transfer (T2D->T2D)')
    parser.parse_args()
    run_same_disease_experiment()
