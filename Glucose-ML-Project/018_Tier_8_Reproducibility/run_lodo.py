"""
S1-2: LODO (Leave-One-Dataset-Out) 다기관 교차 검증
=====================================================
주기 그룹별(1min, 5min, 15min)로, 각 데이터셋을 순환적으로 타겟으로 설정하고
나머지를 소스로 사용하여 4-Way 비교(Source-Only, Target-Only, CORAL, TrAdaBoost)를 수행한다.

핵심 질문:
  - 해석 B(데이터 특이성): 다양한 데이터셋 조합에서도 동일한 현상이 나타나는가?
  - 해석 C(질환 유형 차이): T1D→T1D 전이에서도 Target-Only 초과 실패가 나타나는가?

결과물:
  - LODO 결과 CSV (전체 + 질환 유형별 하위 분석)
  - 히트맵 (타겟 x 모델 RMSE)
  - 환자별 시간순 잔차 NPZ (S1-4용)
  - 소스/타겟 피처 행렬 서브샘플 NPZ (S1-3용)
"""

import sys
import warnings
import logging
import time
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

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '016_Tier_7_Cross_Disease'))
sys.path.insert(0, str(Path(__file__).parent.parent / '013_Tier_6_Domain_Adaptation'))

from tier8_config import Tier8Config
from tier8_data_utils import (load_cached, load_cached_split,
                               build_lodo_source)
from tier7_tradaboost import TrAdaBoostRegressor
from tier6_transfer_utils import apply_coral


# ─── 로거 ─────────────────────────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    log_path = Tier8Config.OUT_DIR / "s1_2_lodo.log"
    logger = logging.getLogger("s1_2")
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


# ─── LightGBM 학습 ────────────────────────────────────────────────────────────

def train_lgbm(X_tr, y_tr, X_val, y_val, seed=None):
    params = dict(Tier8Config.LGBM_PARAMS)
    if seed is not None:
        params['seed'] = seed
    ds_tr = lgb.Dataset(X_tr, label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    return lgb.train(
        params, ds_tr,
        num_boost_round=Tier8Config.LGBM_ROUNDS,
        valid_sets=[ds_val],
        callbacks=[
            lgb.early_stopping(Tier8Config.LGBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )


# ─── 단일 LODO iteration ──────────────────────────────────────────────────────

def run_lodo_iteration(target_ds: str, freq_min: int,
                       logger: logging.Logger,
                       out_dir: Path) -> list:
    """
    단일 타겟에 대해 4-Way 비교를 수행하고 결과를 반환한다.
    S1-3/S1-4용 중간 산출물도 저장한다.
    """
    info = Tier8Config.DATASET_REGISTRY[target_ds]
    disease = info['disease']
    n_subj = info['n_subjects']

    logger.info(f"\n{'='*58}")
    logger.info(f"  Target: {target_ds}  ({disease}, {n_subj} subjects, {freq_min}min)")
    logger.info(f"{'='*58}")

    # ── 데이터 로딩 (캐시) ──
    try:
        splits = load_cached_split(target_ds)
        X_tr, y_tr = splits['train']
        X_val, y_val = splits['val']
        X_te, y_te = splits['test']
    except Exception as e:
        logger.info(f"  [SKIP] {target_ds}: split load failed - {e}")
        return []

    if len(X_te) == 0 or len(X_tr) == 0:
        logger.info(f"  [SKIP] {target_ds}: insufficient data "
                     f"(train={len(X_tr)}, test={len(X_te)})")
        return []

    try:
        X_src, y_src = build_lodo_source(target_ds, freq_min)
    except Exception as e:
        logger.info(f"  [SKIP] {target_ds}: source pool failed - {e}")
        return []

    logger.info(f"  Source: {len(X_src):,} | Train: {len(X_tr):,} | "
                f"Val: {len(X_val):,} | Test: {len(X_te):,}")

    results = []
    predictions = {}

    # ── Source-Only / Target-Only / Mixed 병렬 ──
    def _train(name, X, y):
        m = train_lgbm(X, y, X_val, y_val)
        return name, m

    tasks = [
        ('source_only', X_src, y_src),
        ('target_only', X_tr, y_tr),
        ('mixed', np.vstack([X_src, X_tr]), np.concatenate([y_src, y_tr])),
    ]

    model_cache = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_train, n, X, y): n for n, X, y in tasks}
        for f in as_completed(futures):
            name, model = f.result()
            model_cache[name] = model

    for name in ['source_only', 'target_only', 'mixed']:
        preds = model_cache[name].predict(X_te)
        predictions[name] = preds
        r, m, d = rmse(y_te, preds), mae(y_te, preds), mard(y_te, preds)
        logger.info(f"    [{name:15s}]  RMSE={r:.2f}  MAE={m:.2f}  MARD={d:.1f}%")
        results.append({
            'group': f'{freq_min}min', 'target': target_ds,
            'disease': disease, 'n_subjects': n_subj,
            'model': name, 'rmse': r, 'mae': m, 'mard': d,
        })

    # ── CORAL ──
    try:
        Xs_c = apply_coral(X_src, X_tr)
        m_coral = train_lgbm(np.vstack([Xs_c, X_tr]),
                             np.concatenate([y_src, y_tr]),
                             X_val, y_val)
        preds_coral = m_coral.predict(X_te)
        predictions['coral'] = preds_coral
        r, m, d = rmse(y_te, preds_coral), mae(y_te, preds_coral), mard(y_te, preds_coral)
        logger.info(f"    [{'coral':15s}]  RMSE={r:.2f}  MAE={m:.2f}  MARD={d:.1f}%")
        results.append({
            'group': f'{freq_min}min', 'target': target_ds,
            'disease': disease, 'n_subjects': n_subj,
            'model': 'coral', 'rmse': r, 'mae': m, 'mard': d,
        })
    except Exception as e:
        logger.info(f"    [CORAL FAILED] {e}")

    # ── TrAdaBoost ──
    try:
        tada = TrAdaBoostRegressor(
            n_iterations=Tier8Config.TRADABOOST_N_ITER,
            n_ensemble=Tier8Config.TRADABOOST_ENSEMBLE,
        )
        tada.fit(X_src, y_src, X_tr, y_tr, X_val, y_val)
        preds_tada = tada.predict(X_te)
        predictions['tradaboost'] = preds_tada
        r, m, d = rmse(y_te, preds_tada), mae(y_te, preds_tada), mard(y_te, preds_tada)
        logger.info(f"    [{'tradaboost':15s}]  RMSE={r:.2f}  MAE={m:.2f}  MARD={d:.1f}%")
        results.append({
            'group': f'{freq_min}min', 'target': target_ds,
            'disease': disease, 'n_subjects': n_subj,
            'model': 'tradaboost', 'rmse': r, 'mae': m, 'mard': d,
        })
    except Exception as e:
        logger.info(f"    [TrAdaBoost FAILED] {e}")

    # ── S1-4용: 잔차 저장 ──
    residual_dir = out_dir / "residuals"
    residual_dir.mkdir(parents=True, exist_ok=True)
    for name, preds in predictions.items():
        np.savez(residual_dir / f"{target_ds}_{name}.npz",
                 y_true=y_te, y_pred=preds)

    # ── S1-3용: 소스/타겟 피처 서브샘플 저장 ──
    domain_dir = out_dir / "domain_features"
    domain_dir.mkdir(parents=True, exist_ok=True)
    n_sample = min(10000, len(X_src), len(X_tr))
    rng = np.random.default_rng(Tier8Config.SEED)
    src_idx = rng.choice(len(X_src), size=min(n_sample, len(X_src)), replace=False)
    tgt_idx = rng.choice(len(X_tr), size=min(n_sample, len(X_tr)), replace=False)
    np.savez(domain_dir / f"{target_ds}_features.npz",
             X_src=X_src[src_idx], X_tgt=X_tr[tgt_idx])

    return results


# ─── 히트맵 생성 ──────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, out_dir: Path, group_name: str):
    """타겟 x 모델 RMSE 히트맵."""
    pivot = df.pivot_table(index='target', columns='model', values='rmse')

    # 정렬: Target-Only RMSE 기준
    if 'target_only' in pivot.columns:
        pivot = pivot.sort_values('target_only')

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color='#e6edf3', fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))

    # 타겟 이름에 질환 유형 표기
    ylabels = []
    for tgt in pivot.index:
        info = Tier8Config.DATASET_REGISTRY.get(tgt, {})
        disease = info.get('disease', '?')
        n_subj = info.get('n_subjects', '?')
        ylabels.append(f"{tgt} ({disease}, n={n_subj})")
    ax.set_yticklabels(ylabels, color='#e6edf3', fontsize=8)

    # 셀 내 값 표시
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                       color='white', fontsize=7, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('RMSE (mg/dL)', color='#e6edf3')
    cbar.ax.tick_params(colors='#e6edf3')

    ax.set_title(f'S1-2 LODO: {group_name} Group — RMSE Heatmap',
                 color='#e6edf3', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_{group_name}.png", dpi=150,
                facecolor='#0d1117', bbox_inches='tight')
    plt.close()


# ─── 메인 실행 ────────────────────────────────────────────────────────────────

def main():
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("  S1-2: LODO (Leave-One-Dataset-Out)")
    logger.info("=" * 60)

    out_dir = Tier8Config.OUT_DIR / "s1_2_lodo"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_start = time.perf_counter()

    for freq_min in Tier8Config.group_names():
        datasets = Tier8Config.datasets_by_group(freq_min)
        logger.info(f"\n{'#'*60}")
        logger.info(f"  Group: {freq_min}min — {len(datasets)} datasets")
        logger.info(f"{'#'*60}")

        for target_ds in tqdm(datasets, desc=f"LODO {freq_min}min", ncols=70):
            results = run_lodo_iteration(target_ds, freq_min, logger, out_dir)
            all_results.extend(results)

            # 중간 저장 (장시간 실행 대비)
            df_interim = pd.DataFrame(all_results)
            df_interim.to_csv(out_dir / "lodo_results_interim.csv",
                              index=False, encoding='utf-8-sig')

    elapsed = time.perf_counter() - t_start

    # ── 최종 결과 저장 ──
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "lodo_results.csv", index=False, encoding='utf-8-sig')
    logger.info(f"\n\nSaved {len(df)} rows to lodo_results.csv")
    logger.info(f"Total time: {elapsed/60:.1f} min")

    # ── 요약 통계 ──
    logger.info("\n" + "=" * 60)
    logger.info("  Summary: RMSE by Model (All Groups)")
    logger.info("=" * 60)

    summary = df.groupby('model')['rmse'].agg(['mean', 'std', 'count'])
    logger.info(f"\n{summary.to_string()}")

    # ── 질환 유형별 하위 분석 ──
    logger.info("\n" + "=" * 60)
    logger.info("  Subgroup Analysis: Target Disease Type")
    logger.info("=" * 60)

    for disease in df['disease'].unique():
        sub = df[df['disease'] == disease]
        logger.info(f"\n  --- Disease: {disease} ({len(sub)//sub['model'].nunique()} targets) ---")
        sub_summary = sub.groupby('model')['rmse'].agg(['mean', 'std'])
        logger.info(f"{sub_summary.to_string()}")

        # Flip Rate
        pivoted = sub.pivot_table(index='target', columns='model', values='rmse')
        if 'target_only' in pivoted.columns:
            for tl in ['coral', 'tradaboost']:
                if tl in pivoted.columns:
                    flips = (pivoted[tl] > pivoted['target_only']).sum()
                    total = len(pivoted)
                    logger.info(f"    Flip Rate ({tl} > target_only): "
                                f"{flips}/{total} = {flips/total:.1%}")

    # ── 히트맵 생성 ──
    logger.info("\nGenerating heatmaps...")
    for freq_min in Tier8Config.group_names():
        sub = df[df['group'] == f'{freq_min}min']
        if len(sub) > 0:
            plot_heatmap(sub, out_dir, f'{freq_min}min')
            logger.info(f"  Saved heatmap_{freq_min}min.png")

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
