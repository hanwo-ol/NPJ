"""
S1-3: 도메인 거리 분석
========================
LODO에서 저장된 소스/타겟 피처 서브샘플을 사용하여
4가지 도메인 거리 지표를 산출하고, RMSE 차이(Delta = TL - Target-Only)와의
상관관계를 분석한다.

거리 지표:
  - MMD (Maximum Mean Discrepancy, RBF kernel)
  - PAD (Proxy A-Distance, 선형 SVM)
  - Covariance Frobenius Distance
  - Sliced Wasserstein Distance
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from tier8_config import Tier8Config


# ─── 거리 지표 ────────────────────────────────────────────────────────────────

def compute_mmd_rbf(X_src, X_tgt, gamma=None, n_subsample=5000):
    """RBF 커널 기반 MMD."""
    rng = np.random.default_rng(42)
    if len(X_src) > n_subsample:
        X_src = X_src[rng.choice(len(X_src), n_subsample, replace=False)]
    if len(X_tgt) > n_subsample:
        X_tgt = X_tgt[rng.choice(len(X_tgt), n_subsample, replace=False)]

    if gamma is None:
        # median heuristic
        D = cdist(X_src[:500], X_tgt[:500], 'sqeuclidean')
        gamma = 1.0 / np.median(D[D > 0])

    K_ss = np.exp(-gamma * cdist(X_src, X_src, 'sqeuclidean'))
    K_tt = np.exp(-gamma * cdist(X_tgt, X_tgt, 'sqeuclidean'))
    K_st = np.exp(-gamma * cdist(X_src, X_tgt, 'sqeuclidean'))

    mmd = np.mean(K_ss) + np.mean(K_tt) - 2 * np.mean(K_st)
    return float(max(0, mmd))


def compute_pad(X_src, X_tgt, n_subsample=5000):
    """Proxy A-Distance (선형 SVM 분류 정확도 기반)."""
    rng = np.random.default_rng(42)
    if len(X_src) > n_subsample:
        X_src = X_src[rng.choice(len(X_src), n_subsample, replace=False)]
    if len(X_tgt) > n_subsample:
        X_tgt = X_tgt[rng.choice(len(X_tgt), n_subsample, replace=False)]

    X = np.vstack([X_src, X_tgt])
    y = np.concatenate([np.zeros(len(X_src)), np.ones(len(X_tgt))])

    clf = LinearSVC(max_iter=2000, dual='auto')
    acc = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
    pad = 2 * (1 - 2 * abs(acc - 0.5))
    return float(pad)


def compute_cov_frobenius(X_src, X_tgt):
    """공분산 행렬 Frobenius 거리."""
    cov_s = np.cov(X_src.T)
    cov_t = np.cov(X_tgt.T)
    return float(np.linalg.norm(cov_s - cov_t, 'fro'))


def compute_sliced_wasserstein(X_src, X_tgt, n_projections=200, n_subsample=10000):
    """Sliced Wasserstein Distance."""
    rng = np.random.default_rng(42)
    if len(X_src) > n_subsample:
        X_src = X_src[rng.choice(len(X_src), n_subsample, replace=False)]
    if len(X_tgt) > n_subsample:
        X_tgt = X_tgt[rng.choice(len(X_tgt), n_subsample, replace=False)]

    d = X_src.shape[1]
    # 랜덤 단위 벡터 생성
    directions = rng.standard_normal((n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    distances = []
    for theta in directions:
        proj_s = X_src @ theta
        proj_t = X_tgt @ theta
        proj_s.sort()
        proj_t.sort()
        # 길이 맞추기 (짧은 쪽을 보간)
        n = min(len(proj_s), len(proj_t))
        s_interp = np.interp(np.linspace(0, 1, n),
                             np.linspace(0, 1, len(proj_s)), proj_s)
        t_interp = np.interp(np.linspace(0, 1, n),
                             np.linspace(0, 1, len(proj_t)), proj_t)
        distances.append(np.mean(np.abs(s_interp - t_interp)))

    return float(np.mean(distances))


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  S1-3: Domain Distance Analysis")
    print("=" * 60)

    out_dir = Tier8Config.OUT_DIR / "s1_3_domain_distance"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = Tier8Config.OUT_DIR / "s1_2_lodo" / "domain_features"
    lodo_csv = Tier8Config.OUT_DIR / "s1_2_lodo" / "lodo_results.csv"

    df_lodo = pd.read_csv(lodo_csv)

    results = []
    for npz_file in sorted(feature_dir.glob("*.npz")):
        target_ds = npz_file.stem.replace("_features", "")
        print(f"\n  {target_ds}...")

        data = np.load(npz_file)
        X_src = data['X_src'].astype(np.float64)
        X_tgt = data['X_tgt'].astype(np.float64)

        info = Tier8Config.DATASET_REGISTRY.get(target_ds, {})

        # 거리 지표 산출
        mmd = compute_mmd_rbf(X_src, X_tgt)
        pad = compute_pad(X_src, X_tgt)
        cov_frob = compute_cov_frobenius(X_src, X_tgt)
        swd = compute_sliced_wasserstein(X_src, X_tgt)

        print(f"    MMD={mmd:.4f}  PAD={pad:.4f}  "
              f"CovFrob={cov_frob:.2f}  SWD={swd:.4f}")

        # LODO 결과에서 Delta RMSE 추출
        sub = df_lodo[df_lodo['target'] == target_ds]
        tgt_only_rmse = sub[sub['model'] == 'target_only']['rmse'].values
        coral_rmse = sub[sub['model'] == 'coral']['rmse'].values
        tada_rmse = sub[sub['model'] == 'tradaboost']['rmse'].values

        delta_coral = float(coral_rmse[0] - tgt_only_rmse[0]) if len(coral_rmse) > 0 and len(tgt_only_rmse) > 0 else np.nan
        delta_tada = float(tada_rmse[0] - tgt_only_rmse[0]) if len(tada_rmse) > 0 and len(tgt_only_rmse) > 0 else np.nan

        results.append({
            'target': target_ds,
            'disease': info.get('disease', '?'),
            'n_subjects': info.get('n_subjects', 0),
            'mmd': mmd,
            'pad': pad,
            'cov_frobenius': cov_frob,
            'sliced_wasserstein': swd,
            'delta_coral': delta_coral,
            'delta_tradaboost': delta_tada,
            'target_only_rmse': float(tgt_only_rmse[0]) if len(tgt_only_rmse) > 0 else np.nan,
        })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "domain_distances.csv", index=False, encoding='utf-8-sig')
    print(f"\nSaved {len(df)} rows")

    # ── 상관관계 분석 ──
    print("\n" + "=" * 60)
    print("  Correlation: Distance vs. Delta RMSE")
    print("=" * 60)

    dist_cols = ['mmd', 'pad', 'cov_frobenius', 'sliced_wasserstein']
    delta_cols = ['delta_coral', 'delta_tradaboost']

    for dc in delta_cols:
        print(f"\n  --- {dc} ---")
        for dist in dist_cols:
            valid = df[[dist, dc]].dropna()
            if len(valid) >= 5:
                r_p, p_p = pearsonr(valid[dist], valid[dc])
                r_s, p_s = spearmanr(valid[dist], valid[dc])
                print(f"    {dist:25s}  Pearson r={r_p:.3f} (p={p_p:.3f})  "
                      f"Spearman rho={r_s:.3f} (p={p_s:.3f})")

    # ── 산점도 ──
    print("\nGenerating scatter plots...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('#0d1117')

    disease_colors = {'T1D': '#2a9d8f', 'T2D': '#e63946',
                      'ND': '#457b9d', 'Mixed': '#e9c46a'}

    for row_idx, dc in enumerate(delta_cols):
        for col_idx, dist in enumerate(dist_cols):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('#161b22')

            for disease, color in disease_colors.items():
                sub = df[df['disease'] == disease]
                if len(sub) > 0:
                    ax.scatter(sub[dist], sub[dc], c=color, s=50,
                              alpha=0.8, label=disease, edgecolors='white',
                              linewidth=0.5)

            ax.axhline(y=0, color='#8b949e', linewidth=0.8, linestyle='--')
            ax.set_xlabel(dist, color='#e6edf3', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(dc, color='#e6edf3', fontsize=9)
            ax.tick_params(colors='#e6edf3')
            ax.spines[:].set_color('#30363d')

            if row_idx == 0 and col_idx == 0:
                ax.legend(facecolor='#1c2128', edgecolor='#30363d',
                         labelcolor='#e6edf3', fontsize=8)

    fig.suptitle('S1-3: Domain Distance vs. Transfer Learning Gain',
                 color='#e6edf3', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_distance_vs_delta.png",
                dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()

    print("Done.")


if __name__ == '__main__':
    main()
