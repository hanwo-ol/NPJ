"""
Tier 7: SHAP Feature Transfer Analysis
=======================================
어떤 피처가 T1D → T2D 전이에서 공유되고, 어떤 피처가 도메인 특이적인가를 분석.

출력:
  - shap_source.png  : T1D source 학습 모델의 SHAP 중요도
  - shap_target.png  : T2D target 학습 모델의 SHAP 중요도
  - shap_delta.png   : 중요도 변화 (공유 피처 vs 도메인 특이 피처)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from tier7_config import Tier7Config
from tier7_data_utils import load_source_pool, load_target_split
from tier7_experiment import train_lgbm

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] shap not installed — running feature importance fallback")


DARK_BG  = '#0d1117'
AX_BG    = '#161b22'
TXT_COL  = '#e6edf3'
GRID_COL = '#30363d'


def _lgbm_importance(model: lgb.Booster, feature_names: list) -> pd.Series:
    imp = pd.Series(
        model.feature_importance(importance_type='gain'),
        index=feature_names
    )
    return imp / imp.sum()   # 정규화


def _plot_importance(imp: pd.Series, title: str, color: str, path: Path):
    imp_sorted = imp.sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(AX_BG)
    ax.barh(imp_sorted.index, imp_sorted.values, color=color, edgecolor='none')
    ax.set_xlabel('Normalized Importance (gain)', color=TXT_COL)
    ax.set_title(title, color=TXT_COL, fontsize=11, fontweight='bold')
    ax.tick_params(colors=TXT_COL, labelsize=8)
    ax.spines[:].set_color(GRID_COL)
    ax.xaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def run_shap_analysis():
    print("\n" + "="*60)
    print("  Tier 7: SHAP Feature Transfer Analysis")
    print("="*60)

    Xs, ys = load_source_pool()
    tgt    = load_target_split()
    X_tr, y_tr   = tgt['train']
    X_val, y_val = tgt['val']
    X_te, y_te   = tgt['test']

    # 피처 이름 재구성 (tier7_data_utils 와 동일 순서)
    # build_windows → extract_features 반환 순서에 맞춤
    lag_names = [f'lag_{i}' for i in range(1, 9)]
    base_names = (lag_names +
                  ['delta_1', 'delta_2', 'delta_sq',
                   'win_mean', 'win_std', 'win_min', 'win_max', 'win_cv',
                   'lbgi', 'hbgi',
                   'hour_sin', 'hour_cos', 'is_night', 'dow_sin', 'dow_cos',
                   'fasting_proxy', 'postmeal_rise', 'high_persist', 'in_range_frac'])

    out = Tier7Config.OUT_DIR

    # ── Source 모델 (T1D) ─────────────────────────────────────────────────────
    print("\nTraining source model (T1D)...")
    m_src = train_lgbm(Xs, ys, X_val, y_val)
    imp_src = _lgbm_importance(m_src, base_names)
    _plot_importance(imp_src,
                     "Feature Importance — Source Model (T1D pool)",
                     '#2a9d8f',
                     out / "shap_source.png")

    # ── Target 모델 (T2D) ─────────────────────────────────────────────────────
    print("Training target model (T2D only)...")
    m_tgt = train_lgbm(X_tr, y_tr, X_val, y_val)
    imp_tgt = _lgbm_importance(m_tgt, base_names)
    _plot_importance(imp_tgt,
                     "Feature Importance — Target Model (T2D only)",
                     '#e63946',
                     out / "shap_target.png")

    # ── Delta 분석 ────────────────────────────────────────────────────────────
    delta = (imp_src - imp_tgt).sort_values()
    df_delta = pd.DataFrame({
        'feature':    delta.index,
        'imp_source': imp_src[delta.index].values,
        'imp_target': imp_tgt[delta.index].values,
        'delta':      delta.values,
    })
    df_delta.to_csv(out / "shap_delta.csv", index=False, encoding='utf-8-sig')

    # Plot delta
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(AX_BG)
    colors = ['#e63946' if d < 0 else '#2a9d8f' for d in delta.values]
    ax.barh(delta.index, delta.values, color=colors, edgecolor='none')
    ax.axvline(0, color=TXT_COL, linewidth=0.8, linestyle='--')
    ax.set_xlabel('Δ Importance  (Source − Target)', color=TXT_COL)
    ax.set_title(
        'Feature Importance Shift: T1D → T2D\n'
        'Green = more important in T1D (source-specific) | '
        'Red = more important in T2D (domain-specific)',
        color=TXT_COL, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TXT_COL, labelsize=8)
    ax.spines[:].set_color(GRID_COL)
    ax.xaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out / "shap_delta.png", dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f"\nSaved: shap_source.png, shap_target.png, shap_delta.png")
    print("\n[Feature Transfer Summary]")
    shared   = df_delta[df_delta['delta'].abs() < 0.02]['feature'].tolist()
    src_spec = df_delta[df_delta['delta'] > 0.03]['feature'].tolist()
    tgt_spec = df_delta[df_delta['delta'] < -0.03]['feature'].tolist()
    print(f"  Shared features  : {shared}")
    print(f"  T1D-specific     : {src_spec}")
    print(f"  T2D-specific     : {tgt_spec}")


if __name__ == '__main__':
    run_shap_analysis()
