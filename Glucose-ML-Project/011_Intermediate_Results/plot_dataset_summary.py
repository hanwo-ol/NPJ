"""
Dataset Summary Visualization
===============================
논문 Table 1 / Figure 2 스타일로 12개 데이터셋 요약 표와 시각화 생성.

출력물:
  1. dataset_summary_table.md   (Markdown 표)
  2. dataset_summary_figure.png (멀티패널 그림)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ─── 데이터 로드 ───
ROOT = Path(__file__).resolve().parent
CSV  = ROOT / "dataset_summary_stats.csv"
df   = pd.read_csv(CSV)

# ─── 정렬: 환자군 순서 (T1DM → T2DM → ND → GDM) 후 피험자수 내림차순
cohort_order = {"T1DM": 0, "T1DM/ND": 1, "GDM": 2, "ND/PreD": 3, "ND": 4}
df["_co"] = df["Cohort"].map(cohort_order).fillna(9)
df = df.sort_values(["_co", "N_Subjects"], ascending=[True, False]).reset_index(drop=True)
df.drop(columns=["_co"], inplace=True)

# ─── 표시용 약칭 ───
short_names = {
    "AIDET1D":        "AIDET1D",
    "BIGIDEAs":       "BIGIDEAs",
    "Bris-T1D_Open":  "Bris-T1D",
    "CGMacros_Dexcom":"CGMacros\n(Dexcom)",
    "CGMacros_Libre": "CGMacros\n(Libre)",
    "CGMND":          "CGMND",
    "GLAM":           "GLAM",
    "HUPA-UCM":       "HUPA-UCM",
    "IOBP2":          "IOBP2",
    "Park_2025":      "Park 2025",
    "PEDAP":          "PEDAP",
    "UCHTT1DM":       "UCHTT1DM",
}
df["ShortName"] = df["Dataset"].map(short_names).fillna(df["Dataset"])

# ─── 환자군별 색상 ───
cohort_colors = {
    "T1DM":    "#E05A5A",
    "T1DM/ND": "#E08A5A",
    "GDM":     "#A05AE0",
    "ND/PreD": "#5A9AE0",
    "ND":      "#5AC85A",
}
df["Color"] = df["Cohort"].map(cohort_colors).fillna("#AAAAAA")

N = len(df)
x = np.arange(N)
labels = df["ShortName"].tolist()

# ══════════════════════════════════════════════════
# Figure Layout: 3열 2행 (6개 패널)
# ══════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0F1117")

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.55, wspace=0.35,
    left=0.07, right=0.97,
    top=0.91, bottom=0.10,
)

AXIS_BG      = "#1A1D27"
GRID_COLOR   = "#2E3040"
TEXT_COLOR   = "#E8EAF0"
TICK_COLOR   = "#9BA3BC"
TITLE_COLOR  = "#FFFFFF"
SPINE_COLOR  = "#2E3040"

def style_ax(ax, title, ylabel="", xlabel=""):
    ax.set_facecolor(AXIS_BG)
    ax.set_title(title, color=TITLE_COLOR, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, color=TICK_COLOR, fontsize=9)
    ax.set_xlabel(xlabel, color=TICK_COLOR, fontsize=9)
    ax.tick_params(colors=TICK_COLOR, labelsize=8)
    ax.yaxis.label.set_color(TICK_COLOR)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COLOR)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5,
                        color=TICK_COLOR, linespacing=1.1)

# ─── 패널 A: 피험자 수 ───
ax_a = fig.add_subplot(gs[0, 0])
bars = ax_a.bar(x, df["N_Subjects"], color=df["Color"], edgecolor='none', width=0.65)
for bar, v in zip(bars, df["N_Subjects"]):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
              str(v), ha='center', va='bottom', color=TEXT_COLOR, fontsize=7.5, fontweight='bold')
ax_a.set_yscale('log')
style_ax(ax_a, "(A)  Number of Subjects", ylabel="Subjects (log scale)")

# ─── 패널 B: 총 CGM 읽기 횟수 ───
ax_b = fig.add_subplot(gs[0, 1])
readings_M = df["N_Readings"] / 1e6
bars_b = ax_b.bar(x, readings_M, color=df["Color"], edgecolor='none', width=0.65)
ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}M"))
style_ax(ax_b, "(B)  Total CGM Readings", ylabel="Readings (millions)")
ax_b.set_yscale('log')

# ─── 패널 C: 예측 윈도우 수 ───
ax_c = fig.add_subplot(gs[0, 2])
windows_M = df["N_Windows"] / 1e6
bars_c = ax_c.bar(x, windows_M, color=df["Color"], edgecolor='none', width=0.65)
ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}M"))
style_ax(ax_c, "(C)  Prediction Windows (30-min)", ylabel="Windows (millions)")
ax_c.set_yscale('log')

# ─── 패널 D: 평균 혈당 ± SD ───
ax_d = fig.add_subplot(gs[1, 0])
ax_d.bar(x, df["Glucose_mean"], color=df["Color"], edgecolor='none', width=0.65, alpha=0.85)
ax_d.errorbar(x, df["Glucose_mean"], yerr=df["Glucose_std"],
               fmt='none', ecolor=TEXT_COLOR, elinewidth=1.2, capsize=3, alpha=0.7)
# 임상 기준선
ax_d.axhline(70,  color='#FFD700', linewidth=1.0, linestyle='--', alpha=0.6, label='TBR threshold (70)')
ax_d.axhline(180, color='#FF6B6B', linewidth=1.0, linestyle='--', alpha=0.6, label='TAR threshold (180)')
ax_d.axhline(140, color='#AAD4FF', linewidth=0.8, linestyle=':', alpha=0.5, label='PostPrandial ref (140)')
ax_d.legend(fontsize=6.5, facecolor=AXIS_BG, edgecolor=SPINE_COLOR,
             labelcolor=TEXT_COLOR, loc='upper left')
style_ax(ax_d, "(D)  Mean Glucose ± SD", ylabel="Glucose (mg/dL)")
ax_d.set_ylim(50, 230)

# ─── 패널 E: TIR / TAR / TBR 스택 바 ───
ax_e = fig.add_subplot(gs[1, 1])
bar_tbr = ax_e.bar(x, df["TBR_pct"], color='#4A90D9', edgecolor='none', width=0.65, label='TBR (<70)')
bar_tir = ax_e.bar(x, df["TIR_pct"], bottom=df["TBR_pct"],
                    color='#5AC85A', edgecolor='none', width=0.65, label='TIR (70–180)')
bar_tar = ax_e.bar(x, df["TAR_pct"], bottom=df["TBR_pct"] + df["TIR_pct"],
                    color='#E05A5A', edgecolor='none', width=0.65, label='TAR (>180)')
ax_e.axhline(100, color=GRID_COLOR, linewidth=0.6)
ax_e.legend(fontsize=7.5, facecolor=AXIS_BG, edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR,
             loc='lower right', ncol=1)
style_ax(ax_e, "(E)  Glucose Range Distribution (%)", ylabel="Proportion (%)")
ax_e.set_ylim(0, 110)

# ─── 패널 F: 혈당 분포 박스플롯 ───
ax_f = fig.add_subplot(gs[1, 2])
for i, row in df.iterrows():
    # 정규분포 근사로 박스 시뮬레이션 (실제 분위수가 없으므로)
    mu  = row["Glucose_mean"]
    sd  = row["Glucose_std"]
    med = row["Glucose_median"]
    q1  = max(row["Glucose_min"], mu - 0.674 * sd)
    q3  = min(row["Glucose_max"], mu + 0.674 * sd)
    w1  = max(row["Glucose_min"], mu - 1.5 * sd)
    w3  = min(row["Glucose_max"], mu + 1.5 * sd)
    c   = row["Color"]

    # 박스
    ax_f.add_patch(mpatches.FancyBboxPatch(
        (i - 0.3, q1), 0.6, q3 - q1,
        boxstyle="square,pad=0", facecolor=c, edgecolor=SPIN_C if False else TEXT_COLOR,
        linewidth=0.5, alpha=0.7
    ))
    # 중앙선
    ax_f.plot([i - 0.3, i + 0.3], [med, med], color=TEXT_COLOR, linewidth=1.5)
    # 수염
    ax_f.plot([i, i], [w1, q1], color=c, linewidth=1.0, alpha=0.7)
    ax_f.plot([i, i], [q3, w3], color=c, linewidth=1.0, alpha=0.7)
    ax_f.plot([i - 0.15, i + 0.15], [w1, w1], color=c, linewidth=0.8, alpha=0.7)
    ax_f.plot([i - 0.15, i + 0.15], [w3, w3], color=c, linewidth=0.8, alpha=0.7)

ax_f.axhline(70,  color='#FFD700', linewidth=0.8, linestyle='--', alpha=0.5)
ax_f.axhline(180, color='#FF6B6B', linewidth=0.8, linestyle='--', alpha=0.5)
style_ax(ax_f, "(F)  Glucose Distribution (IQR±1.5SD)", ylabel="Glucose (mg/dL)")
ax_f.set_ylim(30, 320)

# ─── 패널 G: 수집 기간 ───
ax_g = fig.add_subplot(gs[2, 0])
ax_g.bar(x, df["Duration_days_mean"], color=df["Color"], edgecolor='none', width=0.65, alpha=0.85,
          label='Mean')
ax_g.bar(x, df["Duration_days_max"], color=df["Color"], edgecolor='none', width=0.65,
          alpha=0.30, label='Max')
ax_g.legend(fontsize=7.5, facecolor=AXIS_BG, edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR)
style_ax(ax_g, "(G)  Data Collection Duration", ylabel="Days per subject")

# ─── 패널 H: 샘플링 간격 ───
ax_h = fig.add_subplot(gs[2, 1])
interval_colors = df["Interval_min"].map({5: "#5AC85A", 15: "#E0A05A"})
ax_h.bar(x, df["Interval_min"], color=interval_colors, edgecolor='none', width=0.65)
ax_h.set_yticks([5, 15])
ax_h.set_yticklabels(['5 min', '15 min'], color=TICK_COLOR, fontsize=9)
style_ax(ax_h, "(H)  CGM Sampling Interval", ylabel="Interval (min)")
ax_h.set_ylim(0, 20)
patch5  = mpatches.Patch(color='#5AC85A', label='5 min')
patch15 = mpatches.Patch(color='#E0A05A', label='15 min')
ax_h.legend(handles=[patch5, patch15], fontsize=8, facecolor=AXIS_BG,
             edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR)

# ─── 패널 I: 환자군 레전드 패널 ───
ax_i = fig.add_subplot(gs[2, 2])
ax_i.set_facecolor(AXIS_BG)
for sp in ax_i.spines.values():
    sp.set_edgecolor(SPINE_COLOR)
ax_i.set_xticks([])
ax_i.set_yticks([])

# 범례 텍스트
legend_items = [
    ("#E05A5A", "T1DM — Type 1 Diabetes"),
    ("#E08A5A", "T1DM/ND — Mixed cohort"),
    ("#A05AE0", "GDM — Gestational DM"),
    ("#5A9AE0", "ND/PreD — Pre-diabetes"),
    ("#5AC85A", "ND — Non-diabetic"),
]
for j, (col, lbl) in enumerate(legend_items):
    y_pos = 0.82 - j * 0.16
    ax_i.add_patch(mpatches.FancyBboxPatch(
        (0.06, y_pos - 0.04), 0.12, 0.09,
        boxstyle="round,pad=0.01",
        facecolor=col, edgecolor='none', transform=ax_i.transAxes
    ))
    ax_i.text(0.23, y_pos + 0.005, lbl, transform=ax_i.transAxes,
              color=TEXT_COLOR, fontsize=9, va='center')

ax_i.text(0.5, 0.97, "Patient Cohort", transform=ax_i.transAxes,
           color=TITLE_COLOR, fontsize=11, fontweight='bold', ha='center', va='top')

# ─── 전체 제목 ───
fig.suptitle(
    "Dataset Characteristics — 12 CGM Datasets Used in Analysis\n"
    "Glucose-ML-Project · Tier 1 ~ Tier 3",
    color=TITLE_COLOR, fontsize=14, fontweight='bold', y=0.975
)

out_fig = ROOT / "dataset_summary_figure.png"
fig.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Figure saved: {out_fig}")

# ══════════════════════════════════════════════════
# Markdown Table (논문 Table 1 스타일)
# ══════════════════════════════════════════════════
def fmt_windows(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

def fmt_readings(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

rows_md = []
for _, row in df.iterrows():
    rows_md.append({
        "Dataset": row["Dataset"],
        "Cohort": row["Cohort"],
        "Country": row["Country"],
        "N": row["N_Subjects"],
        "Readings": fmt_readings(row["N_Readings"]),
        "Windows": fmt_windows(int(row["N_Windows"])),
        "Duration\n(days, max)": row["Duration_days_max"],
        "Sensor": row["Sensor"],
        "Interval": f"{int(row['Interval_min'])} min",
        "Mean G\n(mg/dL)": f"{row['Glucose_mean']}±{row['Glucose_std']}",
        "TIR (%)": row["TIR_pct"],
        "TAR (%)": row["TAR_pct"],
        "TBR (%)": row["TBR_pct"],
    })

md_df = pd.DataFrame(rows_md)
md_out = ROOT / "dataset_summary_table.md"

with open(md_out, "w", encoding="utf-8") as f:
    f.write("# Table 1. Characteristics of the 12 CGM Datasets\n\n")
    f.write("> **Abbreviations:** N = number of subjects; Readings = total CGM time-points; "
            "Windows = 30-min prediction windows (Lookback 6 + Forecast 6 steps); "
            "TIR = Time In Range (70–180 mg/dL); TAR = Time Above Range (>180); "
            "TBR = Time Below Range (<70).\n\n")
    f.write(md_df.to_markdown(index=False))
    f.write("\n\n---\n\n")
    f.write("## Dataset Scale Breakdown\n\n")

    # 규모별 요약
    for cohort in ["T1DM", "T1DM/ND", "GDM", "ND/PreD", "ND"]:
        sub = df[df["Cohort"] == cohort]
        if sub.empty:
            continue
        f.write(f"### {cohort}\n")
        f.write(f"- Datasets: {', '.join(sub['Dataset'].tolist())}\n")
        f.write(f"- Total subjects: **{sub['N_Subjects'].sum():,}**\n")
        f.write(f"- Total readings: **{fmt_readings(sub['N_Readings'].sum())}**\n")
        f.write(f"- Mean glucose: "
                f"{sub['Glucose_mean'].mean():.1f} ± {sub['Glucose_std'].mean():.1f} mg/dL\n")
        f.write(f"- Pooled TIR: {sub['TIR_pct'].mean():.1f}% | "
                f"TAR: {sub['TAR_pct'].mean():.1f}% | "
                f"TBR: {sub['TBR_pct'].mean():.1f}%\n\n")

print(f"Table saved: {md_out}")
print("\nAll outputs complete.")
