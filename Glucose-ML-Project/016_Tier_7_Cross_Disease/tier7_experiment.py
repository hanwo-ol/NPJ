"""
Tier 7: Main Experiment — 5-Way Comparison (Group-Aware)
=========================================================
T1D (Source) → T2D/Mixed (Target)

개선사항:
  - tqdm: TrAdaBoost 반복, Oracle fold, 타겟 루프에 진행 표시 추가
  - 병렬처리: Oracle 10-fold (joblib), 독립 모델 3종 (concurrent.futures)
  - 로그파일: 터미널 출력을 tier7_results/experiment.log 에 동시 기록
  - 학습 곡선: _run_learning_curve 복원
"""

import sys
import warnings
import argparse
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '013_Tier_6_Domain_Adaptation'))

from tier7_config import Tier7Config
from tier7_data_utils import load_source_pool, load_target_split
from tier7_tradaboost import TrAdaBoostRegressor
from tier6_transfer_utils import apply_coral


# ─── 로거 설정 ────────────────────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    """터미널 + 파일 동시 출력 로거."""
    log_path = Tier7Config.OUT_DIR / "experiment.log"
    logger   = logging.getLogger("tier7")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─── 평가 지표 ────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mard(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

def evaluate(y_true, y_pred, label: str, logger: logging.Logger) -> dict:
    r, m, d = rmse(y_true, y_pred), mae(y_true, y_pred), mard(y_true, y_pred)
    logger.info(f"  [{label:15s}]  RMSE={r:.2f}  MAE={m:.2f}  MARD={d:.1f}%")
    return {'model': label, 'rmse': r, 'mae': m, 'mard': d}


# ─── LightGBM 학습 헬퍼 ───────────────────────────────────────────────────────

def train_lgbm(X_tr, y_tr, X_val, y_val,
               sample_weight=None, params=None) -> lgb.Booster:
    p      = params or dict(Tier7Config.LGBM_PARAMS)
    ds_tr  = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    return lgb.train(
        p, ds_tr,
        num_boost_round=Tier7Config.LGBM_ROUNDS,
        valid_sets=[ds_val],
        callbacks=[
            lgb.early_stopping(Tier7Config.LGBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )


# ─── Oracle: 병렬 10-fold CV ──────────────────────────────────────────────────

def _oracle_fold(fold_idx, X_all, y_all, tr_i, te_i):
    """단일 fold 학습/예측 (스레드 풀에서 실행)."""
    Xk, yk = X_all[tr_i], y_all[tr_i]
    n_v    = max(1, int(len(Xk) * 0.2))
    m_k    = train_lgbm(Xk[:-n_v], yk[:-n_v], Xk[-n_v:], yk[-n_v:])
    return te_i, m_k.predict(X_all[te_i])


def run_oracle_parallel(X_all, y_all, n_splits=10) -> np.ndarray:
    """Oracle 10-fold CV를 ThreadPoolExecutor로 병렬 실행."""
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=Tier7Config.SEED)
    folds  = list(kf.split(X_all))
    preds  = np.zeros_like(y_all)

    # LightGBM 자체가 n_jobs=-1로 멀티코어 사용하므로 스레드 수는 fold 수로 제한
    max_workers = min(n_splits, 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_oracle_fold, i, X_all, y_all, tr_i, te_i): i
            for i, (tr_i, te_i) in enumerate(folds)
        }
        for future in tqdm(as_completed(futures), total=n_splits,
                           desc="    Oracle fold", leave=False, ncols=70):
            te_i, fold_preds = future.result()
            preds[te_i] = fold_preds

    return preds


# ─── 그룹 단위 5-way 실험 ─────────────────────────────────────────────────────

def run_group(group_name: str, target_ds: str,
              logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"\n{'='*58}")
    logger.info(f"  Group [{group_name}]  Target: {target_ds}")
    logger.info(f"{'='*58}")

    Xs, ys = load_source_pool(group_name)
    tgt    = load_target_split(target_ds)
    X_tr, y_tr   = tgt['train']
    X_val, y_val = tgt['val']
    X_te, y_te   = tgt['test']

    if len(X_te) == 0:
        logger.info(f"  [SKIP] No test data for {target_ds}")
        return pd.DataFrame()

    logger.info(f"  Source: {len(Xs):,} | Train: {len(X_tr):,} | Test: {len(X_te):,}")
    results = []

    # ── 독립 모델 3종 병렬 학습 (Source-Only / Target-Only / Mixed) ─────────
    logger.info("\n  [1-3/6] Source-Only / Target-Only / Mixed  (parallel)...")

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
        results.append(evaluate(y_te, model_cache[label].predict(X_te),
                                label, logger))

    # ── 4. CORAL ─────────────────────────────────────────────────────────────
    logger.info("  [4/6] CORAL alignment...")
    Xs_c    = apply_coral(Xs, X_tr)
    m_coral = train_lgbm(np.vstack([Xs_c, X_tr]),
                         np.concatenate([ys, y_tr]),
                         X_val, y_val)
    results.append(evaluate(y_te, m_coral.predict(X_te), 'coral', logger))

    # ── 5. TrAdaBoost ────────────────────────────────────────────────────────
    logger.info(f"  [5/6] TrAdaBoost ({Tier7Config.TRADABOOST_N_ITER} iterations)...")
    tada = TrAdaBoostRegressor()
    # TrAdaBoostRegressor.fit() 내부에서 tqdm 진행 표시 적용 (아래에서 수정)
    tada.fit(Xs, ys, X_tr, y_tr, X_val, y_val)
    results.append(evaluate(y_te, tada.predict(X_te), 'tradaboost', logger))

    # ── 6. Oracle (병렬 10-fold) ──────────────────────────────────────────────
    logger.info("  [6/6] Oracle 10-fold CV (parallel)...")
    X_all_t = np.vstack([X_tr, X_val, X_te])
    y_all_t = np.concatenate([y_tr, y_val, y_te])
    oracle_preds = run_oracle_parallel(X_all_t, y_all_t)
    results.append(evaluate(y_all_t, oracle_preds, 'oracle', logger))

    df = pd.DataFrame(results)
    df.insert(0, 'target', target_ds)
    df.insert(0, 'group',  group_name)
    return df


# ─── 학습 곡선 ────────────────────────────────────────────────────────────────

def run_learning_curve(group_name: str, target_ds: str,
                       Xs, ys, X_tr, y_tr, X_val, y_val, X_te, y_te,
                       logger: logging.Logger, out: Path):
    logger.info(f"\n  [Learning Curve] {target_ds} — target ratio sweep...")
    records = []

    for ratio in tqdm(Tier7Config.LEARNING_CURVE_RATIOS,
                      desc="    ratio", leave=False, ncols=70):
        n      = max(10, int(len(X_tr) * ratio))
        Xt_sub = X_tr[:n]
        yt_sub = y_tr[:n]

        r_src  = rmse(y_te, train_lgbm(Xs, ys, X_val, y_val).predict(X_te))
        r_tgt  = rmse(y_te, train_lgbm(Xt_sub, yt_sub, X_val, y_val).predict(X_te))
        tada   = TrAdaBoostRegressor()
        tada.fit(Xs, ys, Xt_sub, yt_sub, X_val, y_val)
        r_tada = rmse(y_te, tada.predict(X_te))

        records.append({'ratio': ratio, 'n_target': n,
                        'source_only': r_src, 'target_only': r_tgt,
                        'tradaboost':  r_tada})

    df_lc = pd.DataFrame(records)
    df_lc.to_csv(out / f"learning_curve_{target_ds}.csv",
                 index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot(df_lc['n_target'], df_lc['source_only'],
            's--', color='#8b949e', label='Source-Only')
    ax.plot(df_lc['n_target'], df_lc['target_only'],
            'o-',  color='#e63946', label='Target-Only')
    ax.plot(df_lc['n_target'], df_lc['tradaboost'],
            '^-',  color='#2a9d8f', label='TrAdaBoost', linewidth=2)
    ax.set_xlabel('Target Training Windows', color='#e6edf3')
    ax.set_ylabel('RMSE (mg/dL)', color='#e6edf3')
    ax.set_title(f'Learning Curve: {target_ds}', color='#e6edf3', fontsize=11)
    ax.tick_params(colors='#e6edf3')
    ax.spines[:].set_color('#30363d')
    ax.yaxis.grid(True, color='#30363d', linewidth=0.5)
    ax.legend(facecolor='#1c2128', edgecolor='#30363d',
              labelcolor='#e6edf3', fontsize=9)
    plt.tight_layout()
    fig.savefig(out / f"learning_curve_{target_ds}.png", dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Saved: learning_curve_{target_ds}.csv / .png")


# ─── 시각화 ───────────────────────────────────────────────────────────────────

def plot_results(df_all: pd.DataFrame, out: Path, logger: logging.Logger):
    ORDER  = ['source_only', 'target_only', 'mixed', 'coral', 'tradaboost', 'oracle']
    COLORS = {
        'source_only': '#8b949e', 'target_only': '#e63946',
        'mixed':       '#f4a261', 'coral':       '#457b9d',
        'tradaboost':  '#2a9d8f', 'oracle':      '#238636',
    }
    for tgt in df_all['target'].unique():
        sub = (df_all[df_all['target'] == tgt]
               .set_index('model').reindex(ORDER).dropna().reset_index())
        grp = sub['group'].iloc[0] if not sub.empty else ''

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
        ax.set_title(f"Tier 7: T1D → T2D  [{grp}]  Target: {tgt}",
                     color='#e6edf3', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#e6edf3')
        ax.spines[:].set_color('#30363d')
        ax.yaxis.grid(True, color='#30363d', linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        fig.savefig(out / f"5way_{tgt}.png", dpi=150,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"  Saved: 5way_{tgt}.png")


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def run_experiment(groups_filter: list = None,
                   run_lc: bool = False) -> pd.DataFrame:
    logger  = setup_logger()
    out     = Tier7Config.OUT_DIR
    all_res = []

    group_targets = [
        (gname, tgt_ds)
        for gname, (_, _, tgt_defs) in Tier7Config.EXPERIMENT_GROUPS.items()
        if not groups_filter or gname in groups_filter
        for tgt_ds in tgt_defs
    ]

    for group_name, target_ds in tqdm(group_targets,
                                      desc="Experiments", ncols=70):
        df = run_group(group_name, target_ds, logger)
        if not df.empty:
            all_res.append(df)

        if run_lc:
            tgt    = load_target_split(target_ds)
            Xs, ys = load_source_pool(group_name)
            run_learning_curve(
                group_name, target_ds,
                Xs, ys,
                *tgt['train'], *tgt['val'], *tgt['test'],
                logger, out,
            )

    if not all_res:
        logger.info("No results generated.")
        return pd.DataFrame()

    df_all = pd.concat(all_res, ignore_index=True)
    df_all.to_csv(out / "5way_all_targets.csv", index=False, encoding='utf-8-sig')
    logger.info(f"\nSaved: 5way_all_targets.csv")
    logger.info(df_all.to_string(index=False))
    plot_results(df_all, out, logger)
    return df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tier 7: Cross-Disease Transfer')
    parser.add_argument('--groups', nargs='+', choices=['5min', '15min'],
                        default=None,
                        help='실행할 샘플링 주기 그룹 (기본: 전체)')
    parser.add_argument('--learning-curve', action='store_true',
                        help='학습 곡선 실험 추가 실행')
    args = parser.parse_args()
    run_experiment(groups_filter=args.groups, run_lc=args.learning_curve)
