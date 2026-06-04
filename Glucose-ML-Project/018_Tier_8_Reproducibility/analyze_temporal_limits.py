"""
S1-4: 시계열 한계 분석 (잔차 시간 구조)
==========================================
LODO에서 저장된 예측값(y_true, y_pred)으로부터 잔차를 추출하고,
잔차의 시간적 자기상관을 분석하여 정적 모델의 구조적 한계를 진단한다.

분석 항목:
  - ACF (자기상관함수): lag 1~20
  - Ljung-Box 검정: "잔차가 백색 잡음인가?"
  - 구간별 RMSE: 혈당 변화 속도(velocity) 구간별 오차 분해
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from tier8_config import Tier8Config


# ─── ACF 계산 ─────────────────────────────────────────────────────────────────

def compute_acf(residuals, max_lag=20):
    """자기상관함수 계산."""
    n = len(residuals)
    mean = np.mean(residuals)
    var = np.var(residuals)
    if var == 0:
        return np.zeros(max_lag + 1)

    acf_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            cov = np.sum((residuals[:n-lag] - mean) *
                         (residuals[lag:] - mean)) / n
            acf_vals.append(cov / var)
    return np.array(acf_vals)


# ─── Ljung-Box 검정 ───────────────────────────────────────────────────────────

def ljung_box_test(residuals, max_lag=20):
    """
    Ljung-Box 검정.
    H0: 잔차는 백색 잡음 (시간적 상관 없음)
    H1: 잔차에 시간적 상관이 존재
    p < 0.05 → H0 기각 → 잔차에 시간적 패턴이 남아있음
    """
    n = len(residuals)
    acf_vals = compute_acf(residuals, max_lag)

    Q = 0.0
    for k in range(1, max_lag + 1):
        Q += (acf_vals[k] ** 2) / (n - k)
    Q *= n * (n + 2)

    # 자유도 = max_lag
    p_value = 1.0 - stats.chi2.cdf(Q, df=max_lag)
    return float(Q), float(p_value)


# ─── 구간별 RMSE ──────────────────────────────────────────────────────────────

def velocity_rmse(y_true, y_pred):
    """
    혈당 변화 속도(velocity) 구간별 RMSE 분해.
    velocity = y_true[t] - y_true[t-1]
    """
    velocity = np.diff(y_true)
    residuals = (y_true[1:] - y_pred[1:])

    bins = [
        ('rapid_drop',   velocity < -5),
        ('slow_drop',    (velocity >= -5) & (velocity < -1)),
        ('stable',       (velocity >= -1) & (velocity <= 1)),
        ('slow_rise',    (velocity > 1) & (velocity <= 5)),
        ('rapid_rise',   velocity > 5),
    ]

    results = []
    for name, mask in bins:
        if np.sum(mask) > 0:
            rmse_bin = float(np.sqrt(np.mean(residuals[mask] ** 2)))
            n_bin = int(np.sum(mask))
            frac = float(np.sum(mask)) / len(velocity)
            results.append({
                'velocity_bin': name,
                'rmse': rmse_bin,
                'n_samples': n_bin,
                'fraction': frac,
            })
    return results


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  S1-4: Temporal Limits Analysis (Residual ACF)")
    print("=" * 60)

    out_dir = Tier8Config.OUT_DIR / "s1_4_temporal_limits"
    out_dir.mkdir(parents=True, exist_ok=True)

    residual_dir = Tier8Config.OUT_DIR / "s1_2_lodo" / "residuals"

    acf_results = []
    ljungbox_results = []
    velocity_results = []

    models = ['source_only', 'target_only', 'coral', 'tradaboost']
    max_lag = 20

    for target_ds in sorted(Tier8Config.DATASET_REGISTRY.keys()):
        info = Tier8Config.DATASET_REGISTRY[target_ds]

        for model in models:
            npz_path = residual_dir / f"{target_ds}_{model}.npz"
            if not npz_path.exists():
                continue

            data = np.load(npz_path)
            y_true = data['y_true'].astype(np.float64)
            y_pred = data['y_pred'].astype(np.float64)
            residuals = y_true - y_pred

            # ACF
            acf_vals = compute_acf(residuals, max_lag)
            for lag in range(max_lag + 1):
                acf_results.append({
                    'target': target_ds,
                    'disease': info['disease'],
                    'model': model,
                    'lag': lag,
                    'acf': acf_vals[lag],
                })

            # Ljung-Box
            Q, p = ljung_box_test(residuals, max_lag)
            ljungbox_results.append({
                'target': target_ds,
                'disease': info['disease'],
                'n_subjects': info['n_subjects'],
                'model': model,
                'Q_statistic': Q,
                'p_value': p,
                'significant': p < 0.05,
                'acf_lag1': acf_vals[1],
                'acf_lag2': acf_vals[2],
                'acf_lag3': acf_vals[3],
            })

            # 구간별 RMSE
            vel_rows = velocity_rmse(y_true, y_pred)
            for row in vel_rows:
                row['target'] = target_ds
                row['disease'] = info['disease']
                row['model'] = model
                velocity_results.append(row)

        print(f"  {target_ds}: done")

    # ── 결과 저장 ──
    df_acf = pd.DataFrame(acf_results)
    df_acf.to_csv(out_dir / "acf_results.csv", index=False, encoding='utf-8-sig')

    df_lb = pd.DataFrame(ljungbox_results)
    df_lb.to_csv(out_dir / "ljungbox_results.csv", index=False, encoding='utf-8-sig')

    df_vel = pd.DataFrame(velocity_results)
    df_vel.to_csv(out_dir / "velocity_rmse.csv", index=False, encoding='utf-8-sig')

    print(f"\nSaved {len(df_acf)} ACF rows, {len(df_lb)} Ljung-Box rows, "
          f"{len(df_vel)} velocity rows")

    # ── Ljung-Box 요약 ──
    print("\n" + "=" * 60)
    print("  Ljung-Box Summary")
    print("=" * 60)

    for model in models:
        sub = df_lb[df_lb['model'] == model]
        n_sig = sub['significant'].sum()
        n_total = len(sub)
        mean_acf1 = sub['acf_lag1'].mean()
        print(f"  {model:15s}: {n_sig}/{n_total} significant (p<0.05), "
              f"mean ACF(lag=1) = {mean_acf1:.4f}")

    # ── 질환별 ACF(lag=1) ──
    print("\n  ACF(lag=1) by Disease (target_only model):")
    sub_to = df_lb[df_lb['model'] == 'target_only']
    for disease in sub_to['disease'].unique():
        d_sub = sub_to[sub_to['disease'] == disease]
        print(f"    {disease:8s}: mean={d_sub['acf_lag1'].mean():.4f}, "
              f"sig={d_sub['significant'].sum()}/{len(d_sub)}")

    # ── ACF 플롯: 전체 모델 비교 ──
    print("\nGenerating ACF plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    model_colors = {
        'source_only': '#8b949e', 'target_only': '#e63946',
        'coral': '#2a9d8f', 'tradaboost': '#e9c46a',
    }

    for model in models:
        sub = df_acf[(df_acf['model'] == model) & (df_acf['lag'] > 0)]
        mean_acf = sub.groupby('lag')['acf'].mean()
        ax.plot(mean_acf.index, mean_acf.values, 'o-',
                color=model_colors[model], label=model, linewidth=2,
                markersize=5, alpha=0.9)

    # 95% 신뢰구간 (백색 잡음 기준)
    n_avg = np.mean([info['n_subjects'] for info in
                     Tier8Config.DATASET_REGISTRY.values()]) * 100
    ci = 1.96 / np.sqrt(n_avg)
    ax.axhline(y=ci, color='#8b949e', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=-ci, color='#8b949e', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0, color='#8b949e', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Lag', color='#e6edf3')
    ax.set_ylabel('ACF', color='#e6edf3')
    ax.set_title('S1-4: Mean Residual ACF across All Targets',
                 color='#e6edf3', fontsize=12)
    ax.tick_params(colors='#e6edf3')
    ax.spines[:].set_color('#30363d')
    ax.legend(facecolor='#1c2128', edgecolor='#30363d',
             labelcolor='#e6edf3', fontsize=9)
    ax.yaxis.grid(True, color='#30363d', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "acf_mean_all_models.png", dpi=150,
                facecolor='#0d1117')
    plt.close()

    # ── 구간별 RMSE 바 차트 ──
    print("Generating velocity RMSE chart...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('#0d1117')
    ax2.set_facecolor('#161b22')

    vel_order = ['rapid_drop', 'slow_drop', 'stable', 'slow_rise', 'rapid_rise']
    x = np.arange(len(vel_order))
    width = 0.2

    for i, model in enumerate(models):
        sub = df_vel[df_vel['model'] == model]
        means = []
        for vbin in vel_order:
            v = sub[sub['velocity_bin'] == vbin]['rmse']
            means.append(v.mean() if len(v) > 0 else 0)
        ax2.bar(x + i * width, means, width, label=model,
               color=model_colors[model], alpha=0.85)

    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(vel_order, color='#e6edf3', fontsize=9, rotation=15)
    ax2.set_ylabel('RMSE (mg/dL)', color='#e6edf3')
    ax2.set_title('S1-4: RMSE by Glucose Velocity Bin',
                  color='#e6edf3', fontsize=12)
    ax2.tick_params(colors='#e6edf3')
    ax2.spines[:].set_color('#30363d')
    ax2.legend(facecolor='#1c2128', edgecolor='#30363d',
              labelcolor='#e6edf3', fontsize=9)
    ax2.yaxis.grid(True, color='#30363d', linewidth=0.5)

    fig2.tight_layout()
    fig2.savefig(out_dir / "velocity_rmse_bars.png", dpi=150,
                 facecolor='#0d1117')
    plt.close()

    print("Done.")


if __name__ == '__main__':
    main()
