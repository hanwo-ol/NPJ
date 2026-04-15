"""
Dataset Data Type Availability - Figure 1 Style Heatmap
=========================================================
Prioleau et al. 2025 Figure 1 style: G/I/A/S/Q/M data type
availability heatmap across 12 datasets at sub-variable level.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

OUT = Path(__file__).parent / "dataset_datatype_figure.png"

# ══════════════════════════════════════════════════
# Data Definitions
# ══════════════════════════════════════════════════

DATASETS = [
    ("AIDET1D",         "T1DM",    "Turkey"),
    ("BIGIDEAs",        "ND/PreD", "USA"),
    ("Bris-T1D",        "T1DM",    "Australia"),
    ("CGMacros\n(Dex)", "ND",      "USA"),
    ("CGMacros\n(Lib)", "ND",      "USA"),
    ("CGMND",           "ND",      "Multi"),
    ("GLAM",            "GDM",     "Spain"),
    ("HUPA-UCM",        "T1DM/ND", "Spain"),
    ("IOBP2",           "T1DM",    "USA"),
    ("Park 2025",       "ND",      "Korea"),
    ("PEDAP",           "T1DM",    "USA"),
    ("UCHTT1DM",        "T1DM",    "USA"),
]
N_DS = len(DATASETS)

COHORT_COLORS = {
    "T1DM":    "#E05A5A",
    "T1DM/ND": "#E08A5A",
    "GDM":     "#A05AE0",
    "ND/PreD": "#5A9AE0",
    "ND":      "#5AC85A",
}

# Sub-variables: (category, display label)
VARIABLES = [
    # G
    ("G", "CGM glucose value"),
    ("G", "2nd CGM sensor (cross)"),
    # I
    ("I", "Basal insulin rate"),
    ("I", "Bolus insulin dose"),
    ("I", "Carb input for bolus"),
    ("I", "Insulin regimen type"),
    # A
    ("A", "Heart rate (HR)"),
    ("A", "Step count"),
    ("A", "Distance walked"),
    ("A", "Active calories / METs"),
    ("A", "Sleep log"),
    # S
    ("S", "Meal event marker"),
    ("S", "Food calories"),
    ("S", "Carbohydrate intake"),
    ("S", "Protein intake"),
    ("S", "Fat intake"),
    ("S", "Dietary fiber"),
    ("S", "Meal photo (path only)"),
    # Q
    ("Q", "Demographics (age/sex)"),
    ("Q", "Diabetes duration"),
    ("Q", "Insulin delivery method"),
    ("Q", "Lifestyle survey"),
    # M
    ("M", "HbA1c"),
    ("M", "OGTT / Fasting glucose"),
    ("M", "Cholesterol / TG"),
    ("M", "BMI / Weight"),
]
N_VAR = len(VARIABLES)

# State codes: 2=used(green), 1=available-not-used(orange), 0=not-in-dataset(dark)
# Rows=variables, Cols=datasets
# [AIDET1D, BIGIDEAs, Bris-T1D, CGMacros-D, CGMacros-L, CGMND, GLAM, HUPA-UCM, IOBP2, Park25, PEDAP, UCHTT1DM]
DATA = np.array([
    # G
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],   # CGM glucose
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],   # 2nd sensor
    # I
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0],   # Basal
    [0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 2, 2],   # Bolus
    [0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 2, 0],   # Carb input
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Insulin regimen
    # A
    [0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0],   # HR
    [0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0],   # Steps
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Distance
    [0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0],   # Calories/METs
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],   # Sleep log
    # S
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],   # Meal event marker
    [0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],   # Food calories
    [0, 2, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0],   # Carb intake
    [0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],   # Protein
    [0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],   # Fat
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],   # Fiber
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],   # Meal photo
    # Q
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # Demographics
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],   # Diabetes duration
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Insulin delivery method
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],   # Lifestyle survey
    # M
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],   # HbA1c
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],   # OGTT/FPG
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   # Cholesterol/TG
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],   # BMI/Weight
], dtype=float)

CAT_BOUNDS = {
    "G": (0, 2),
    "I": (2, 6),
    "A": (6, 11),
    "S": (11, 18),
    "Q": (18, 22),
    "M": (22, 26),
}
CAT_COLORS = {
    "G": "#2196F3",
    "I": "#FF5722",
    "A": "#4CAF50",
    "S": "#FF9800",
    "Q": "#9C27B0",
    "M": "#607D8B",
}

# ══════════════════════════════════════════════════
# Draw Figure
# ══════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#0F1117")

ax_cat  = fig.add_axes([0.01, 0.08, 0.065, 0.82])
ax_main = fig.add_axes([0.085, 0.08, 0.71, 0.82])
ax_leg  = fig.add_axes([0.81, 0.08, 0.17, 0.82])

BG     = "#0F1117"
AX_BG  = "#1A1D27"
TICK_C = "#9BA3BC"

for ax in [ax_cat, ax_main, ax_leg]:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)

# ── Main Heatmap ──
palette = {2: "#2ECC71", 1: "#F39C12", 0: "#1E222D"}

for vi in range(N_VAR):
    for di in range(N_DS):
        val = int(DATA[vi, di])
        color = palette[val]
        rect = FancyBboxPatch(
            (di - 0.42, N_VAR - vi - 1 - 0.42),
            0.84, 0.84,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor=BG, linewidth=0.8,
        )
        ax_main.add_patch(rect)
        sym = {2: "v", 1: "o", 0: ""}[val]
        if sym:
            fc = "#FFFFFF" if val == 2 else "#C8860A"
            ax_main.text(di, N_VAR - vi - 1, sym,
                         ha='center', va='center',
                         color=fc, fontsize=8, fontweight='bold')

ax_main.set_xlim(-0.55, N_DS - 0.45)
ax_main.set_ylim(-0.55, N_VAR - 0.45)
ax_main.set_facecolor(AX_BG)

# x-axis labels + cohort color bars
ax_main.set_xticks(range(N_DS))
ax_main.set_xticklabels(
    [d[0] for d in DATASETS],
    rotation=0, ha='center', fontsize=7.5, color=TICK_C, linespacing=1.2
)
ax_main.tick_params(axis='x', length=0, pad=6)
ax_main.tick_params(axis='y', length=0)
ax_main.set_yticks([])

for di, (name, cohort, _) in enumerate(DATASETS):
    col = COHORT_COLORS.get(cohort, "#888888")
    rect = FancyBboxPatch(
        (di - 0.42, N_VAR - 0.38), 0.84, 0.32,
        boxstyle="round,pad=0.02",
        facecolor=col, edgecolor='none', alpha=0.85, clip_on=False
    )
    ax_main.add_patch(rect)
    ax_main.text(di, N_VAR - 0.22, cohort,
                 ha='center', va='center',
                 color='white', fontsize=5.5, fontweight='bold')

# Category separator lines
for cat, (start, end) in CAT_BOUNDS.items():
    y = N_VAR - end - 0.5
    ax_main.axhline(y, color="#3A3F52", linewidth=1.4, xmin=0, xmax=1)

# Vertical grid
for di in range(N_DS + 1):
    ax_main.axvline(di - 0.5, color=BG, linewidth=0.5, alpha=0.6)

# ── Category Label Panel ──
ax_cat.set_xlim(0, 1)
ax_cat.set_ylim(-0.55, N_VAR - 0.45)
ax_cat.set_xticks([])
ax_cat.set_yticks([])
ax_cat.set_facecolor(BG)

for cat, (start, end) in CAT_BOUNDS.items():
    mid_y = N_VAR - (start + end) / 2 - 1
    height = (end - start) * 0.88
    rect = FancyBboxPatch(
        (0.05, N_VAR - end + 0.08), 0.38, height,
        boxstyle="round,pad=0.05",
        facecolor=CAT_COLORS[cat], edgecolor='none', alpha=0.9,
    )
    ax_cat.add_patch(rect)
    ax_cat.text(0.24, mid_y + 0.5, cat,
                ha='center', va='center',
                color='white', fontsize=12, fontweight='bold')

# Variable name labels
for vi, (cat, varname) in enumerate(VARIABLES):
    ax_cat.text(0.50, N_VAR - vi - 1, varname,
                ha='left', va='center',
                color=TICK_C, fontsize=7.5)

# ── Legend Panel ──
ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, 1)
ax_leg.set_xticks([])
ax_leg.set_yticks([])
ax_leg.set_facecolor(BG)

# Availability legend
ax_leg.text(0.5, 0.97, "Availability", ha='center', va='top',
             color='white', fontsize=10, fontweight='bold')
leg_items = [
    ("#2ECC71", "v", "Used in model\n(in harmonized data)"),
    ("#F39C12", "o", "In original dataset\n(not harmonized)"),
    ("#1E222D", "",  "Not in dataset"),
]
for j, (col, sym, lbl) in enumerate(leg_items):
    y = 0.88 - j * 0.115
    rect = FancyBboxPatch((0.04, y - 0.038), 0.20, 0.075,
                           boxstyle="round,pad=0.01",
                           facecolor=col, edgecolor='#444', linewidth=0.5)
    ax_leg.add_patch(rect)
    if sym:
        ax_leg.text(0.14, y, sym, ha='center', va='center',
                    color='white', fontsize=9, fontweight='bold')
    ax_leg.text(0.27, y, lbl, ha='left', va='center',
                color=TICK_C, fontsize=7.5, linespacing=1.4)

# Cohort legend
ax_leg.text(0.5, 0.58, "Patient Cohort", ha='center', va='top',
             color='white', fontsize=10, fontweight='bold')
cohort_legend = [
    ("T1DM",    "#E05A5A", "Type 1 Diabetes"),
    ("T1DM/ND", "#E08A5A", "T1DM + Non-diabetic"),
    ("GDM",     "#A05AE0", "Gestational DM"),
    ("ND/PreD", "#5A9AE0", "Pre-diabetes"),
    ("ND",      "#5AC85A", "Non-diabetic"),
]
for j, (label, col, desc) in enumerate(cohort_legend):
    y = 0.50 - j * 0.088
    rect = FancyBboxPatch((0.04, y - 0.030), 0.20, 0.060,
                           boxstyle="round,pad=0.01",
                           facecolor=col, edgecolor='none')
    ax_leg.add_patch(rect)
    ax_leg.text(0.14, y, label, ha='center', va='center',
                color='white', fontsize=6.5, fontweight='bold')
    ax_leg.text(0.27, y, desc, ha='left', va='center',
                color=TICK_C, fontsize=7.5)

# Category legend
ax_leg.text(0.5, 0.07, "Data Categories", ha='center', va='top',
             color='white', fontsize=9.5, fontweight='bold')
cat_leg = [
    ("G", "CGM"),       ("I", "Insulin"),
    ("A", "Activity"),  ("S", "Self-report"),
    ("Q", "Survey"),    ("M", "Lab / Clinical"),
]
for j, (k, v) in enumerate(cat_leg):
    col = CAT_COLORS[k]
    xi = 0.04 + (j % 2) * 0.50
    yi = 0.01 - (j // 2) * 0.06
    rect = FancyBboxPatch((xi, yi - 0.018), 0.20, 0.036,
                           boxstyle="round,pad=0.01",
                           facecolor=col, edgecolor='none', alpha=0.9)
    ax_leg.add_patch(rect)
    ax_leg.text(xi + 0.10, yi, k, ha='center', va='center',
                color='white', fontsize=7, fontweight='bold')
    ax_leg.text(xi + 0.22, yi, v, ha='left', va='center',
                color=TICK_C, fontsize=7)

# ── Title ──
fig.suptitle(
    "Figure 1.  Data Type Availability Across 12 CGM Datasets\n"
    "Glucose-ML-Project  |  Modeled after Prioleau et al. 2025, Fig. 1",
    color='white', fontsize=13, fontweight='bold', y=0.998
)

fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {OUT}")
