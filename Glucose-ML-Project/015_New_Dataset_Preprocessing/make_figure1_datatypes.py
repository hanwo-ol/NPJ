"""
Figure 1 스타일 데이터 카테고리 분류 시각화
원본 논문: 2507.14077v1 (GlucoseML) Figure 1 기준

6개 카테고리:
  G: Continuous Glucose Monitor (CGM)
  I: Insulin Delivery System (인슐린 투여 / 탄수화물)
  A: Activity Tracker (심박수 / 걸음수 / 수면 / 가속도계 등)
  S: Self-Report / Mobile App (식사·운동·약물 사용자 로그)
  Q: Questionnaire / Survey (인구통계·설문)
  M: Medical Record / Clinical Measurement (HbA1c·검사 등)
"""

import sys
sys.path.insert(0, r"C:\Users\user\Documents\NPJ2\Glucose-ML-Project")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT_DIR = Path(r"C:\Users\user\Documents\NPJ2\Glucose-ML-Project\015_New_Dataset_Preprocessing")

# ─── 분류 정의 ────────────────────────────────────────────────────────────────
# (disease_type, source, G, I, A, S, Q, M)
# 근거:
#   G  : CGM 파일 직접 확인
#   I  : extended-features 또는 Data Tables (Insulin/Basal/Bolus)
#   A  : extended-features (heart_rate/steps/accelerometry) 또는 Data Tables (FrailVisHearAssess)
#   S  : extended-features (calories/carb_input) 또는 Data Tables (LifeDiabSelfMgmt)
#   Q  : Data Tables (*Screening / *Survey* / *Roster) 또는 Known from paper
#   M  : Data Tables (*Lab* / *HbA1c / *SampleResults) 또는 Known from paper

DATASETS = {
    # name: (disease, source, G, I, A, S, Q, M)
    # ─ Existing ─────────────────────────────────────────────────────────────
    'AIDET1D':        ('T1D',   'Existing', 1,0,0,0,0,0),  # CGM-only from JAEB
    'AZT1D':          ('T1D',   'Existing', 1,0,0,0,0,0),  # CGM-only
    'BIGIDEAs':       ('ND',    'Existing', 1,0,1,0,1,1),  # wearable+demo (paper confirmed)
    'Bris-T1D_Open':  ('T1D',   'Existing', 1,0,0,0,0,0),  # CGM-only
    'CGMacros_Dexcom':('Mixed', 'Existing', 1,0,1,1,1,1),  # food+activity+lab (paper confirmed)
    'CGMacros_Libre': ('Mixed', 'Existing', 1,0,1,1,1,1),  # food+activity+lab
    'CGMND':          ('ND',    'Existing', 1,0,0,0,0,0),  # CGM-only
    'Colas_2019':     ('Mixed', 'Existing', 1,0,0,0,0,1),  # CGM + OGTT/HbA1c
    'D1NAMO':         ('T1D',   'Existing', 1,1,0,1,0,0),  # CGM+insulin+food logs
    'GLAM':           ('ND',    'Existing', 1,0,0,0,1,1),  # CGM+surveys+obstetric
    'Hall_2018':      ('Mixed', 'Existing', 1,0,0,0,0,1),  # CGM + lab measurements
    'HUPA-UCM':       ('T1D',   'Existing', 1,1,1,1,0,0),  # CGM+insulin+HR/steps+calories
    'IOBP2':          ('T1D',   'Existing', 1,1,0,0,1,1),  # bionic pancreas (JAEB)
    'PEDAP':          ('T1D',   'Existing', 1,1,0,0,1,1),  # pediatric (JAEB)
    'PhysioCGM':      ('T1D',   'Existing', 1,0,1,0,0,0),  # ECG/PPG/EDA/accelerometry
    'ShanghaiT1DM':   ('T1D',   'Existing', 1,0,0,1,1,1),  # paper confirmed
    'ShanghaiT2DM':   ('T2D',   'Existing', 1,0,0,1,1,1),  # paper confirmed
    'T1D-UOM':        ('T1D',   'Existing', 1,0,0,0,0,0),  # CGM-only
    'UCHTT1DM':       ('T1D',   'Existing', 1,1,1,1,0,0),  # paper confirmed
    # ─ New (DiaData / Jaeb) ──────────────────────────────────────────────────
    'RT-CGM':         ('T1D',   'New',      1,0,0,0,1,1),  # tblASurvey*+tblALabHbA1c
    'CITY':           ('Mixed', 'New',      1,1,0,0,1,1),  # Insulin+CentralLab+DiabScreening
    'SENCE':          ('T1D',   'New',      1,1,0,0,1,1),  # Insulin+STASampleResults+DiabScreening
    'WISDM':          ('T1D',   'New',      1,1,0,0,1,1),  # Insulin+CentralLab+DiabScreening
    'FLAIR':          ('T1D',   'New',      1,1,0,0,1,1),  # FLAIRInsulin/InsulinDelivery+Surveys+STASampleResults
    'SHD':            ('T1D',   'New',      1,0,0,0,1,1),  # BBGAttitudeScale+BMedChart+BSampleResults
    'ReplaceBG':      ('T1D',   'New',      1,1,0,1,1,1),  # HDeviceBasal/Bolus+HQuestHypoFear+HLocalHbA1c
}

CATEGORIES  = ['G', 'I', 'A', 'S', 'Q', 'M']
CAT_LABELS  = {
    'G': 'G — Continuous Glucose Monitor',
    'I': 'I — Insulin Delivery System',
    'A': 'A — Activity Tracker',
    'S': 'S — Self-Report / Mobile App',
    'Q': 'Q — Questionnaire / Survey',
    'M': 'M — Medical Record / Clinical',
}
CAT_COLORS = {
    'G': '#2A9D8F',
    'I': '#E9C46A',
    'A': '#F4A261',
    'S': '#E76F51',
    'Q': '#457B9D',
    'M': '#A8DADC',
}
DISEASE_COLORS = {
    'T1D':   '#c0392b',
    'T2D':   '#2980b9',
    'ND':    '#27ae60',
    'Mixed': '#e67e22',
}

# ─── 정렬: 집단 유형 → 데이터 풍부도 ────────────────────────────────────────
ds_names   = list(DATASETS.keys())
ds_types   = [DATASETS[d][0] for d in ds_names]
ds_sources = [DATASETS[d][1] for d in ds_names]
matrix     = np.array([list(DATASETS[d][2:]) for d in ds_names], dtype=float)  # shape (26, 6)

richness = matrix.sum(axis=1)
order = sorted(range(len(ds_names)), key=lambda i: (ds_types[i], -richness[i]))
ds_names_s   = [ds_names[i] for i in order]
ds_types_s   = [ds_types[i] for i in order]
ds_sources_s = [ds_sources[i] for i in order]
matrix_s     = matrix[order]

N = len(ds_names_s)
C = len(CATEGORIES)

# ─── Figure 1: Matrix heatmap ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 13))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

CIRCLE_R = 0.38
for ri, ds in enumerate(ds_names_s):
    y = N - 1 - ri
    for ci, cat in enumerate(CATEGORIES):
        val = matrix_s[ri, ci]
        if val:
            circle = plt.Circle((ci, y), CIRCLE_R,
                                 color=CAT_COLORS[cat], zorder=3, alpha=0.92)
            ax.add_patch(circle)
            ax.text(ci, y, cat, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white', zorder=4)
        else:
            circle = plt.Circle((ci, y), CIRCLE_R,
                                 color='#21262d', zorder=2)
            ax.add_patch(circle)

    # Row label (dataset name)
    label_color = DISEASE_COLORS[ds_types_s[ri]]
    weight = 'bold' if ds_sources_s[ri] == 'New' else 'normal'
    ax.text(-0.65, y, ds, ha='right', va='center',
            fontsize=9, color=label_color, fontweight=weight)

    # Richness bar on right
    bar_w = richness[order[ri]] * 0.06
    ax.barh(y, bar_w, left=C + 0.3, height=0.55,
            color=label_color, alpha=0.6, zorder=2)
    ax.text(C + 0.35 + bar_w, y,
            f"{int(richness[order[ri]])}",
            va='center', ha='left', fontsize=8, color='#8b949e')

# Column headers
for ci, cat in enumerate(CATEGORIES):
    ax.text(ci, N + 0.1, cat, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=CAT_COLORS[cat])
    ax.text(ci, N + 0.55, CAT_LABELS[cat].split(' — ')[1],
            ha='center', va='bottom', fontsize=7, color='#8b949e',
            rotation=20, rotation_mode='anchor')

# Section dividers by disease type
prev_type = None
for ri, ds in enumerate(ds_names_s):
    y = N - 1 - ri
    if ds_types_s[ri] != prev_type and prev_type is not None:
        ax.axhline(y + 0.5, color='#30363d', linewidth=1.2, zorder=1)
    prev_type = ds_types_s[ri]

# Disease type labels on left margin
prev_type = None
block_start = 0
for ri, ds in enumerate(ds_names_s):
    if ds_types_s[ri] != prev_type:
        if prev_type is not None:
            mid = (block_start + ri - 1) / 2
            ax.text(-1.8, N - 1 - mid, prev_type,
                    ha='center', va='center', fontsize=10,
                    color=DISEASE_COLORS[prev_type], fontweight='bold',
                    rotation=90)
        block_start = ri
        prev_type = ds_types_s[ri]
# Last block
mid = (block_start + N - 1) / 2
ax.text(-1.8, N - 1 - mid, prev_type,
        ha='center', va='center', fontsize=10,
        color=DISEASE_COLORS[prev_type], fontweight='bold', rotation=90)

ax.set_xlim(-2.2, C + 1.5)
ax.set_ylim(-0.7, N + 1.3)
ax.axis('off')

# Legends
disease_patches = [mpatches.Patch(color=c, label=l) for l, c in DISEASE_COLORS.items()]
new_patch       = mpatches.Patch(facecolor='gray', label='Bold = New dataset', linewidth=0)
legend1 = ax.legend(handles=disease_patches + [new_patch],
                    loc='lower right', fontsize=8,
                    facecolor='#161b22', edgecolor='#30363d',
                    labelcolor='#e6edf3', title='Disease Type',
                    title_fontsize=8)
ax.add_artist(legend1)

ax.set_title('Figure 1 — Data Type Availability Across 26 Datasets\n'
             'Glucose-ML Project (GlucoseML collection + DiaData extension)',
             color='#e6edf3', fontsize=12, fontweight='bold', y=1.0)

fig.tight_layout()
out_path = OUT_DIR / "figure1_data_types.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")

# ─── CSV export ───────────────────────────────────────────────────────────────
import pandas as pd
rows = []
for ds, (dtype, src, *cats) in DATASETS.items():
    rows.append({
        'dataset': ds, 'disease_type': dtype, 'source': src,
        **{c: bool(v) for c, v in zip(CATEGORIES, cats)},
        'n_categories': sum(cats)
    })
df = pd.DataFrame(rows)
csv_path = OUT_DIR / "dataset_data_types.csv"
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"Saved: {csv_path}")
print(df.to_string(index=False))
