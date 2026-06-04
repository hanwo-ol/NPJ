"""
S1-1: Within Variation (계산적 재현성) + S1-5: 3분류 전환
===========================================================
10개 시드 × 5개 모델(Source-Only, Target-Only, Mixed, CORAL, TrAdaBoost)
고정 데이터: 15min 그룹 T1D 소스 → ShanghaiT2DM 타겟

S1-5는 S1-1의 예측값을 3분류(저혈당/정상/고혈당)로 이산화하여
Cohen's Kappa의 시드별 분포를 분석한다.
"""

import sys
import warnings
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mean_squared_error, cohen_kappa_score, f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '016_Tier_7_Cross_Disease'))
sys.path.insert(0, str(Path(__file__).parent.parent / '013_Tier_6_Domain_Adaptation'))

from tier8_config import Tier8Config
from tier8_data_utils import load_cached, load_cached_split, build_lodo_source
from tier7_tradaboost import TrAdaBoostRegressor
from tier6_transfer_utils import apply_coral


# ─── 로거 설정 ────────────────────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    log_path = Tier8Config.OUT_DIR / "s1_1_within_variation.log"
    logger = logging.getLogger("s1_1")
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

def train_lgbm(X_tr, y_tr, X_val, y_val, seed, sample_weight=None):
    params = dict(Tier8Config.LGBM_PARAMS)
    params['seed'] = seed
    ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
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


# ─── 3분류 변환 (S1-5) ────────────────────────────────────────────────────────

def to_3class(values):
    """연속 혈당값을 저혈당/정상/고혈당 3분류로 변환."""
    classes = np.ones(len(values), dtype=int)  # 1 = 정상
    classes[values < Tier8Config.HYPO_THRESHOLD] = 0   # 저혈당
    classes[values > Tier8Config.HYPER_THRESHOLD] = 2   # 고혈당
    return classes


# ─── 단일 시드 실험 ────────────────────────────────────────────────────────────

def run_single_seed(seed, X_src, y_src, X_tr, y_tr, X_val, y_val,
                    X_te, y_te, logger):
    """단일 시드에서 5개 모델 학습 + 평가."""
    logger.info(f"\n  === Seed {seed} ===")
    results = []
    predictions = {}

    # ── 독립 모델 3종 병렬 학습 ──
    def _train(name, X, y):
        m = train_lgbm(X, y, X_val, y_val, seed)
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
        r = rmse(y_te, preds)
        m = mae(y_te, preds)
        d = mard(y_te, preds)
        logger.info(f"    [{name:15s}]  RMSE={r:.2f}  MAE={m:.2f}  MARD={d:.1f}%")
        results.append({
            'seed': seed, 'model': name,
            'rmse': r, 'mae': m, 'mard': d,
        })

    # ── CORAL ──
    Xs_c = apply_coral(X_src, X_tr)
    m_coral = train_lgbm(np.vstack([Xs_c, X_tr]),
                         np.concatenate([y_src, y_tr]),
                         X_val, y_val, seed)
    preds_coral = m_coral.predict(X_te)
    predictions['coral'] = preds_coral
    r = rmse(y_te, preds_coral)
    m = mae(y_te, preds_coral)
    d = mard(y_te, preds_coral)
    logger.info(f"    [{'coral':15s}]  RMSE={r:.2f}  MAE={m:.2f}  MARD={d:.1f}%")
    results.append({'seed': seed, 'model': 'coral', 'rmse': r, 'mae': m, 'mard': d})

    # ── TrAdaBoost ──
    tada = TrAdaBoostRegressor(
        n_iterations=Tier8Config.TRADABOOST_N_ITER,
        n_ensemble=Tier8Config.TRADABOOST_ENSEMBLE,
    )
    tada.fit(X_src, y_src, X_tr, y_tr, X_val, y_val)
    preds_tada = tada.predict(X_te)
    predictions['tradaboost'] = preds_tada
    r = rmse(y_te, preds_tada)
    m = mae(y_te, preds_tada)
    d = mard(y_te, preds_tada)
    logger.info(f"    [{'tradaboost':15s}]  RMSE={r:.2f}  MAE={m:.2f}  MARD={d:.1f}%")
    results.append({'seed': seed, 'model': 'tradaboost', 'rmse': r, 'mae': m, 'mard': d})

    return results, predictions


# ─── S1-5: 분류 지표 계산 ─────────────────────────────────────────────────────

def compute_classification_metrics(y_true, predictions_dict, seed):
    """단일 시드의 모든 모델에 대해 3분류 지표 계산."""
    y_true_cls = to_3class(y_true)
    rows = []
    for model_name, y_pred in predictions_dict.items():
        y_pred_cls = to_3class(y_pred)
        kappa = cohen_kappa_score(y_true_cls, y_pred_cls)
        f1_macro = f1_score(y_true_cls, y_pred_cls, average='macro',
                            zero_division=0)
        f1_weighted = f1_score(y_true_cls, y_pred_cls, average='weighted',
                               zero_division=0)
        # 저혈당 민감도 (클래스 0)
        hypo_mask = y_true_cls == 0
        hypo_sens = (np.sum((y_pred_cls == 0) & hypo_mask) / np.sum(hypo_mask)
                     if np.sum(hypo_mask) > 0 else np.nan)
        # 고혈당 민감도 (클래스 2)
        hyper_mask = y_true_cls == 2
        hyper_sens = (np.sum((y_pred_cls == 2) & hyper_mask) / np.sum(hyper_mask)
                      if np.sum(hyper_mask) > 0 else np.nan)

        rows.append({
            'seed': seed, 'model': model_name,
            'kappa': kappa,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'hypo_sensitivity': hypo_sens,
            'hyper_sensitivity': hyper_sens,
        })
    return rows


# ─── 메인 실행 ────────────────────────────────────────────────────────────────

def main():
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("  S1-1: Within Variation + S1-5: Classification")
    logger.info("=" * 60)

    target_ds = Tier8Config.WITHIN_VAR_TARGET
    freq_min = Tier8Config.WITHIN_VAR_SOURCE_GROUP
    seeds = Tier8Config.WITHIN_VAR_SEEDS

    # ── 데이터 로딩 (1회, 캐시에서) ──
    logger.info(f"\nLoading source pool (15min group, excluding {target_ds})...")
    X_src, y_src = build_lodo_source(target_ds, freq_min)
    logger.info(f"  Source: {len(X_src):,} windows")

    logger.info(f"Loading target split ({target_ds})...")
    splits = load_cached_split(target_ds)
    X_tr, y_tr = splits['train']
    X_val, y_val = splits['val']
    X_te, y_te = splits['test']
    logger.info(f"  Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_te):,}")

    # ── 시드 루프 ──
    all_regression_results = []
    all_classification_results = []

    for seed in tqdm(seeds, desc="Seeds", ncols=70):
        reg_results, preds = run_single_seed(
            seed, X_src, y_src, X_tr, y_tr, X_val, y_val, X_te, y_te, logger
        )
        all_regression_results.extend(reg_results)

        cls_results = compute_classification_metrics(y_te, preds, seed)
        all_classification_results.extend(cls_results)

    # ── 결과 저장 ──
    out_dir = Tier8Config.OUT_DIR / "s1_1_within_variation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_reg = pd.DataFrame(all_regression_results)
    df_reg.to_csv(out_dir / "regression_results.csv",
                  index=False, encoding='utf-8-sig')
    logger.info(f"\nSaved regression results: {len(df_reg)} rows")

    df_cls = pd.DataFrame(all_classification_results)
    df_cls.to_csv(out_dir / "classification_results.csv",
                  index=False, encoding='utf-8-sig')
    logger.info(f"Saved classification results: {len(df_cls)} rows")

    # ── 요약 통계 ──
    logger.info("\n" + "=" * 60)
    logger.info("  S1-1 Summary: RMSE by Model")
    logger.info("=" * 60)

    summary = df_reg.groupby('model')['rmse'].agg(['mean', 'std', 'min', 'max'])
    summary['cv'] = summary['std'] / summary['mean']
    logger.info(f"\n{summary.to_string()}")

    # Flip Rate: 전이학습이 Target-Only보다 나쁜 시드의 비율
    logger.info("\n  Flip Rate (TL RMSE > Target-Only RMSE):")
    for tl_model in ['coral', 'tradaboost', 'mixed']:
        pivoted = df_reg.pivot(index='seed', columns='model', values='rmse')
        if tl_model in pivoted.columns and 'target_only' in pivoted.columns:
            flips = (pivoted[tl_model] > pivoted['target_only']).sum()
            fr = flips / len(seeds)
            logger.info(f"    {tl_model}: {flips}/{len(seeds)} = {fr:.1%}")

    # S1-5 Kappa 요약
    logger.info("\n" + "=" * 60)
    logger.info("  S1-5 Summary: Kappa by Model")
    logger.info("=" * 60)

    kappa_summary = df_cls.groupby('model')['kappa'].agg(['mean', 'std', 'min', 'max'])
    kappa_summary['cv'] = kappa_summary['std'] / kappa_summary['mean']
    logger.info(f"\n{kappa_summary.to_string()}")

    # ── Violin Plot (RMSE) ──
    logger.info("\nGenerating violin plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    models = ['source_only', 'target_only', 'mixed', 'coral', 'tradaboost']
    colors = ['#8b949e', '#e63946', '#457b9d', '#2a9d8f', '#e9c46a']

    data_for_violin = []
    labels = []
    for model in models:
        vals = df_reg[df_reg['model'] == model]['rmse'].values
        if len(vals) > 0:
            data_for_violin.append(vals)
            labels.append(model)

    parts = ax.violinplot(data_for_violin, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, color='#e6edf3', fontsize=9)
    ax.set_ylabel('RMSE (mg/dL)', color='#e6edf3')
    ax.set_title('S1-1: Within Variation — RMSE across 10 Seeds',
                 color='#e6edf3', fontsize=12)
    ax.tick_params(colors='#e6edf3')
    ax.spines[:].set_color('#30363d')
    ax.yaxis.grid(True, color='#30363d', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "violin_rmse.png", dpi=150, facecolor='#0d1117')
    plt.close()

    # ── Violin Plot (Kappa) ──
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('#0d1117')
    ax2.set_facecolor('#161b22')

    data_kappa = []
    labels_k = []
    for model in models:
        vals = df_cls[df_cls['model'] == model]['kappa'].values
        if len(vals) > 0:
            data_kappa.append(vals)
            labels_k.append(model)

    parts2 = ax2.violinplot(data_kappa, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax2.set_xticks(range(1, len(labels_k) + 1))
    ax2.set_xticklabels(labels_k, color='#e6edf3', fontsize=9)
    ax2.set_ylabel("Cohen's Kappa", color='#e6edf3')
    ax2.set_title('S1-5: Kappa Distribution across 10 Seeds',
                  color='#e6edf3', fontsize=12)
    ax2.tick_params(colors='#e6edf3')
    ax2.spines[:].set_color('#30363d')
    ax2.yaxis.grid(True, color='#30363d', linewidth=0.5)

    fig2.tight_layout()
    fig2.savefig(out_dir / "violin_kappa.png", dpi=150, facecolor='#0d1117')
    plt.close()

    logger.info("\nDone. All results saved to: " + str(out_dir))


if __name__ == '__main__':
    main()
