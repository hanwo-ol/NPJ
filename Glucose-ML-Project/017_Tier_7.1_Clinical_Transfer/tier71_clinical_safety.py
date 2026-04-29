"""
Tier 7.1: Clinical Safety Analysis
====================================
단계 1: Clarke Error Grid + 저혈당 구간 + 구간별 RMSE

Tier 7에서 저장된 예측값(.npz)을 불러와 임상적 안전성을 분석한다.
- Clarke Error Grid 5-Zone 분류
- 저혈당(<70 mg/dL) sensitivity / specificity
- 구간별 RMSE 분해

실행: python 017_Tier_7.1_Clinical_Transfer/tier71_clinical_safety.py
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from tier71_config import Tier71Config


# --- Clarke Error Grid Zone 분류 -------------------------------------------

def classify_clarke_zone(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Clarke Error Grid의 5개 Zone을 분류한다.
    반환: Zone 문자열 배열 ('A', 'B', 'C', 'D', 'E')
    """
    n = len(y_true)
    zones = np.empty(n, dtype='U1')

    for i in range(n):
        ref = y_true[i]
        pred = y_pred[i]
        diff = pred - ref

        # Zone A: 임상적으로 정확
        if ref < 70:
            if pred < 70:
                zones[i] = 'A'
                continue
        else:
            if abs(diff) <= ref * Tier71Config.CLARKE_THRESHOLD_PCT:
                zones[i] = 'A'
                continue

        # Zone E: 반대 방향 의사결정 (가장 위험)
        if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zones[i] = 'E'
            continue

        # Zone C: 불필요한 교정 초과
        if ref >= 70 and pred >= ref + ref * 0.2 and pred >= 180:
            zones[i] = 'C'
            continue
        if ref >= 70 and pred <= ref - ref * 0.2 and pred <= 70:
            zones[i] = 'C'
            continue

        # Zone D: 위험한 미감지
        if ref < 70 and pred >= 70 and pred <= 180:
            zones[i] = 'D'
            continue
        if ref > 180 and pred <= 180 and pred >= 70:
            zones[i] = 'D'
            continue

        # Zone B: 나머지 (양성 오류)
        zones[i] = 'B'

    return zones


def compute_zone_percentages(zones: np.ndarray) -> dict:
    """Zone 배열에서 각 Zone의 비율(%)을 계산한다."""
    n = len(zones)
    return {f"zone_{z}_pct": float(np.sum(zones == z)) / n * 100
            for z in ['A', 'B', 'C', 'D', 'E']}


# --- 저혈당 분석 -----------------------------------------------------------

def compute_hypo_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         threshold: float = 70.0) -> dict:
    """저혈당 구간(<threshold)의 sensitivity, specificity, PPV, NPV."""
    actual_hypo = y_true < threshold
    pred_hypo   = y_pred < threshold

    tp = np.sum(actual_hypo & pred_hypo)
    fn = np.sum(actual_hypo & ~pred_hypo)
    fp = np.sum(~actual_hypo & pred_hypo)
    tn = np.sum(~actual_hypo & ~pred_hypo)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv         = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        'hypo_sensitivity': sensitivity,
        'hypo_specificity': specificity,
        'hypo_ppv':         ppv,
        'hypo_npv':         npv,
        'n_hypo_events':    int(tp + fn),
        'n_total':          len(y_true),
    }


# --- 구간별 RMSE -----------------------------------------------------------

def compute_range_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """혈당 구간별 RMSE를 계산한다."""
    ranges = {
        'below70':    y_true < 70,
        '70_180':     (y_true >= 70) & (y_true <= 180),
        '180_250':    (y_true > 180) & (y_true <= 250),
        'above250':   y_true > 250,
    }
    result = {}
    for name, mask in ranges.items():
        n = np.sum(mask)
        if n > 0:
            result[f'rmse_{name}'] = float(np.sqrt(
                np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        else:
            result[f'rmse_{name}'] = np.nan
        result[f'n_{name}'] = int(n)
    return result


# --- 시각화 ----------------------------------------------------------------

def plot_clarke_grid(y_true: np.ndarray, y_pred: np.ndarray,
                     zones: np.ndarray, title: str, save_path: Path):
    """Clarke Error Grid scatter plot."""
    zone_colors = {
        'A': '#2a9d8f', 'B': '#457b9d',
        'C': '#f4a261', 'D': '#e63946', 'E': '#6c1d45',
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    for z in ['E', 'D', 'C', 'B', 'A']:
        mask = zones == z
        if np.sum(mask) > 0:
            ax.scatter(y_true[mask], y_pred[mask],
                       c=zone_colors[z], s=2, alpha=0.4, label=f"Zone {z}")

    ax.plot([0, 400], [0, 400], '--', color='#8b949e', linewidth=0.8)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Reference (mg/dL)', color='#e6edf3')
    ax.set_ylabel('Predicted (mg/dL)', color='#e6edf3')
    ax.set_title(title, color='#e6edf3', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#e6edf3')
    ax.spines[:].set_color('#30363d')

    pcts = compute_zone_percentages(zones)
    info = f"A:{pcts['zone_A_pct']:.1f}% B:{pcts['zone_B_pct']:.1f}% " \
           f"C:{pcts['zone_C_pct']:.1f}% D:{pcts['zone_D_pct']:.1f}% " \
           f"E:{pcts['zone_E_pct']:.1f}%"
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            color='#e6edf3', fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#1c2128', edgecolor='#30363d'))

    ax.legend(loc='lower right', facecolor='#1c2128', edgecolor='#30363d',
              labelcolor='#e6edf3', fontsize=8, markerscale=4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


def plot_clinical_summary(df_zones: pd.DataFrame, df_hypo: pd.DataFrame,
                          save_path: Path):
    """모델별 Zone A 비율 + 저혈당 sensitivity 비교 막대 그래프."""
    targets = df_zones['target'].unique()
    n_targets = len(targets)

    fig, axes = plt.subplots(2, n_targets, figsize=(5 * n_targets, 8))
    fig.patch.set_facecolor('#0d1117')
    if n_targets == 1:
        axes = axes.reshape(2, 1)

    colors = {
        'source_only': '#8b949e', 'target_only': '#e63946',
        'mixed': '#f4a261', 'coral': '#457b9d', 'tradaboost': '#2a9d8f',
    }
    order = ['source_only', 'target_only', 'mixed', 'coral', 'tradaboost']

    for col, tgt in enumerate(targets):
        # Zone A
        ax = axes[0, col]
        ax.set_facecolor('#161b22')
        sub = df_zones[(df_zones['target'] == tgt) &
                       (df_zones['model'].isin(order))]
        sub = sub.set_index('model').reindex(order).dropna().reset_index()
        if not sub.empty:
            ax.bar(sub['model'], sub['zone_A_pct'],
                   color=[colors.get(m, '#8b949e') for m in sub['model']],
                   width=0.6)
            for i, row in sub.iterrows():
                ax.text(i, row['zone_A_pct'] + 0.3, f"{row['zone_A_pct']:.1f}",
                        ha='center', color='#e6edf3', fontsize=8)
        ax.set_ylabel('Zone A %', color='#e6edf3')
        ax.set_title(f'{tgt}', color='#e6edf3', fontsize=10)
        ax.tick_params(colors='#e6edf3', labelrotation=30)
        ax.spines[:].set_color('#30363d')

        # Hypo Sensitivity
        ax2 = axes[1, col]
        ax2.set_facecolor('#161b22')
        sub_h = df_hypo[(df_hypo['target'] == tgt) &
                        (df_hypo['model'].isin(order))]
        sub_h = sub_h.set_index('model').reindex(order).dropna().reset_index()
        if not sub_h.empty:
            ax2.bar(sub_h['model'], sub_h['hypo_sensitivity'] * 100,
                    color=[colors.get(m, '#8b949e') for m in sub_h['model']],
                    width=0.6)
            for i, row in sub_h.iterrows():
                val = row['hypo_sensitivity'] * 100
                if not np.isnan(val):
                    ax2.text(i, val + 0.3, f"{val:.1f}",
                             ha='center', color='#e6edf3', fontsize=8)
        ax2.set_ylabel('Hypo Sensitivity %', color='#e6edf3')
        ax2.tick_params(colors='#e6edf3', labelrotation=30)
        ax2.spines[:].set_color('#30363d')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


# --- 메인 분석 로직 --------------------------------------------------------

def run_clinical_analysis():
    """모든 타겟의 예측값을 로드하여 임상 분석을 수행한다."""
    pred_root = Tier71Config.TIER7_PRED_DIR
    out_dir   = Tier71Config.OUT_DIR / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ['source_only', 'target_only', 'mixed', 'coral', 'tradaboost']
    targets = [d.name for d in pred_root.iterdir() if d.is_dir()]

    if not targets:
        print(f"[ERROR] No predictions found in {pred_root}")
        print("        Run tier7_experiment.py first (Step 0).")
        return

    all_zones = []
    all_hypo  = []
    all_range = []

    for target in tqdm(targets, desc="Targets", ncols=70):
        target_dir = pred_root / target

        for model in tqdm(models, desc=f"  {target}", leave=False, ncols=70):
            npz_path = target_dir / f"{model}.npz"
            if not npz_path.exists():
                continue

            data   = np.load(npz_path)
            y_true = data['y_true']
            y_pred = data['y_pred']

            # Clarke Error Grid
            zones = classify_clarke_zone(y_true, y_pred)
            zone_pcts = compute_zone_percentages(zones)
            zone_pcts.update({'target': target, 'model': model,
                              'n_samples': len(y_true)})
            all_zones.append(zone_pcts)

            # 저혈당 분석
            hypo = compute_hypo_metrics(y_true, y_pred,
                                        Tier71Config.HYPO_THRESHOLD)
            hypo.update({'target': target, 'model': model})
            all_hypo.append(hypo)

            # 구간별 RMSE
            range_rmse = compute_range_rmse(y_true, y_pred)
            range_rmse.update({'target': target, 'model': model})
            all_range.append(range_rmse)

        # Clarke Grid scatter plot (타겟별)
        for model in models:
            npz_path = target_dir / f"{model}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                zones = classify_clarke_zone(data['y_true'], data['y_pred'])
                plot_clarke_grid(
                    data['y_true'], data['y_pred'], zones,
                    f"Clarke Grid: {target} / {model}",
                    out_dir / f"clarke_{target}_{model}.png")

    # CSV 저장
    df_zones = pd.DataFrame(all_zones)
    cols_z = ['target', 'model', 'zone_A_pct', 'zone_B_pct', 'zone_C_pct',
              'zone_D_pct', 'zone_E_pct', 'n_samples']
    df_zones[cols_z].to_csv(out_dir / "clarke_zones.csv",
                            index=False, encoding='utf-8-sig')
    print(f"Saved: {out_dir / 'clarke_zones.csv'}")

    df_hypo = pd.DataFrame(all_hypo)
    cols_h = ['target', 'model', 'hypo_sensitivity', 'hypo_specificity',
              'hypo_ppv', 'hypo_npv', 'n_hypo_events', 'n_total']
    df_hypo[cols_h].to_csv(out_dir / "hypo_analysis.csv",
                           index=False, encoding='utf-8-sig')
    print(f"Saved: {out_dir / 'hypo_analysis.csv'}")

    df_range = pd.DataFrame(all_range)
    cols_r = ['target', 'model', 'rmse_below70', 'rmse_70_180',
              'rmse_180_250', 'rmse_above250',
              'n_below70', 'n_70_180', 'n_180_250', 'n_above250']
    df_range[cols_r].to_csv(out_dir / "range_rmse.csv",
                            index=False, encoding='utf-8-sig')
    print(f"Saved: {out_dir / 'range_rmse.csv'}")

    # 종합 시각화
    plot_clinical_summary(df_zones, df_hypo,
                          out_dir / "clinical_summary.png")
    print(f"Saved: {out_dir / 'clinical_summary.png'}")

    # 결과 출력
    print("\n=== Clarke Error Grid Zone A (%) ===")
    pivot = df_zones.pivot(index='model', columns='target',
                           values='zone_A_pct')
    print(pivot.to_string())

    print("\n=== Hypo Sensitivity ===")
    pivot_h = df_hypo.pivot(index='model', columns='target',
                            values='hypo_sensitivity')
    print(pivot_h.to_string())


# --- 구동부 ----------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tier 7.1 Step 1: Clinical Safety Analysis')
    parser.parse_args()
    run_clinical_analysis()
