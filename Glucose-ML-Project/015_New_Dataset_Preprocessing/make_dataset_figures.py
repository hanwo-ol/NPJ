"""
Figure 2 + Table 1 스타일 시각화
원본 논문: 2507.14077v1 (GlucoseML)

Figure 2 구성:
  A: 데이터셋별 피험자 수 + 집단 구성 (T1D / T2D / ND / Mixed)
  B: 총 CGM 측정 일수 (per dataset)
  C: 충분한 CGM 데이터를 보유한 날의 비율 (≥70% completeness)
  D: 혈당 역학 — TBR / TIR / TAR 분포

Table 1: 데이터셋 메타데이터 표 (CSV + 화면 출력)
"""

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\user\Documents\NPJ2\Glucose-ML-Project")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
from global_config import GlobalConfig

# ─── Config ───────────────────────────────────────────────────────────────────
OUT_DIR  = Path(r"C:\Users\user\Documents\NPJ2\Glucose-ML-Project\015_New_Dataset_Preprocessing")
DATA_ROOT = GlobalConfig.DATA_ROOT
RATE_MIN  = 3.0   # 'missing gap' = 3x sampling interval (in minutes, so 15min for 5min CGM)

# ─── Dataset metadata (집단 타입, 샘플링 주기, 출처) ────────────────────────
# Format: name → (disease_type, sampling_min, source_label)
DATASET_META = {
    # Original datasets (기존)
    'AIDET1D':        ('T1D',   5,  'Existing'),
    'AZT1D':          ('T1D',   5,  'Existing'),
    'BIGIDEAs':       ('ND',    5,  'Existing'),
    'Bris-T1D_Open':  ('T1D',   15, 'Existing'),
    'CGMacros_Dexcom':('Mixed', 1,  'Existing'),
    'CGMacros_Libre': ('Mixed', 1,  'Existing'),
    'CGMND':          ('ND',    5,  'Existing'),
    'Colas_2019':     ('Mixed', 5,  'Existing'),   # ND → T2D 발병 위험군 혼재
    'D1NAMO':         ('T1D',   5,  'Existing'),
    'GLAM':           ('ND',    5,  'Existing'),   # Glucose Levels Across Maternity (건강한 임신부)
    'Hall_2018':      ('Mixed', 5,  'Existing'),   # 정상혈당 + 전당뇨 + T2D 혼재
    'HUPA-UCM':       ('T1D',   5,  'Existing'),
    'IOBP2':          ('T1D',   5,  'Existing'),
    'PEDAP':          ('T1D',   5,  'Existing'),
    'PhysioCGM':      ('T1D',   5,  'Existing'),   # 순수 T1D 코호트 (10명)
    'ShanghaiT1DM':   ('T1D',   15, 'Existing'),
    'ShanghaiT2DM':   ('T2D',   15, 'Existing'),
    'T1D-UOM':        ('T1D',   5,  'Existing'),
    'UCHTT1DM':       ('T1D',   5,  'Existing'),
    # New datasets (신규)
    'RT-CGM':         ('T1D',   5,  'New'),
    'CITY':           ('Mixed', 5,  'New'),
    'SENCE':          ('T1D',   5,  'New'),
    'WISDM':          ('T1D',   5,  'New'),
    'FLAIR':          ('T1D',   5,  'New'),
    'SHD':            ('T1D',   5,  'New'),
    'ReplaceBG':      ('T1D',   5,  'New'),
}

DISEASE_COLORS = {
    'T1D':   '#E63946',
    'T2D':   '#457B9D',
    'ND':    '#2A9D8F',
    'Mixed': '#F4A261',
}

# ─── Step 1: 통계 계산 ────────────────────────────────────────────────────────
print("Computing per-dataset statistics...")
records = []
for ds_name, (dtype, rate_min, source) in tqdm(DATASET_META.items()):
    ds_dir = DATA_ROOT / ds_name / f"{ds_name}-extracted-glucose-files"
    if not ds_dir.exists():
        continue
    files = sorted(ds_dir.glob("*.csv"))
    if not files:
        continue

    n_subjects = len(files)
    all_tir, all_tar, all_tbr = [], [], []
    total_days, adequate_days = 0, 0
    total_rows = 0

    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['glucose_value_mg_dl'] = pd.to_numeric(df['glucose_value_mg_dl'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'glucose_value_mg_dl'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        except:
            continue

        total_rows += len(df)
        # CGM days
        df['date'] = df['timestamp'].dt.date
        gap_min = rate_min * 3   # missing gap threshold
        max_readings_per_day = 24 * 60 / rate_min

        for date, grp in df.groupby('date'):
            total_days += 1
            completeness = len(grp) / max_readings_per_day
            if completeness >= 0.70:
                adequate_days += 1
                g = grp['glucose_value_mg_dl'].values
                tir = np.mean((g >= 70) & (g <= 180))
                tar = np.mean(g > 180)
                tbr = np.mean(g < 70)
                all_tir.append(tir)
                all_tar.append(tar)
                all_tbr.append(tbr)

    if not all_tir:
        continue

    records.append({
        'dataset':      ds_name,
        'disease_type': dtype,
        'sampling_min': rate_min,
        'source':       source,
        'n_subjects':   n_subjects,
        'total_days':   total_days,
        'adequate_days':adequate_days,
        'pct_adequate': adequate_days / total_days * 100 if total_days > 0 else 0,
        'mean_tir':     np.mean(all_tir) * 100,
        'mean_tar':     np.mean(all_tar) * 100,
        'mean_tbr':     np.mean(all_tbr) * 100,
        'total_rows':   total_rows,
        'avg_days_pp':  total_days / n_subjects if n_subjects > 0 else 0,
    })

df_stats = pd.DataFrame(records).sort_values('n_subjects', ascending=False)
df_stats.to_csv(OUT_DIR / "dataset_summary_stats.csv", index=False, encoding='utf-8-sig')
print(f"  {len(df_stats)} datasets computed")

# ─── Figure 2 ─────────────────────────────────────────────────────────────────
print("Rendering Figure 2...")

fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('#0d1117')
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

LABEL_COLOR = '#e6edf3'
GRID_COLOR  = '#30363d'
AX_COLOR    = '#161b22'

def style_ax(ax, title):
    ax.set_facecolor(AX_COLOR)
    ax.tick_params(colors=LABEL_COLOR, labelsize=9)
    ax.set_title(title, color=LABEL_COLOR, fontsize=11, fontweight='bold', pad=10)
    ax.spines[:].set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

# ── A: 피험자 수 ──────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
style_ax(ax_a, 'A  Population & Sample Size')

df_a = df_stats.sort_values('n_subjects', ascending=True)
y_pos = np.arange(len(df_a))
bar_colors = [DISEASE_COLORS[d] for d in df_a['disease_type']]
bars = ax_a.barh(y_pos, df_a['n_subjects'], color=bar_colors, height=0.7, edgecolor='none')

# Hatching for new datasets
for bar, src in zip(bars, df_a['source']):
    if src == 'New':
        bar.set_hatch('///')
        bar.set_edgecolor('#ffffff50')

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(df_a['dataset'], fontsize=8, color=LABEL_COLOR)
ax_a.set_xlabel('Number of Subjects', color=LABEL_COLOR, fontsize=9)
ax_a.tick_params(axis='x', colors=LABEL_COLOR)

# Legend: disease type
legend_patches = [mpatches.Patch(color=c, label=l) for l, c in DISEASE_COLORS.items()]
legend_patches.append(mpatches.Patch(facecolor='#aaa', hatch='///', edgecolor='white', label='New dataset'))
ax_a.legend(handles=legend_patches, loc='lower right', fontsize=7.5,
            facecolor='#1c2128', edgecolor=GRID_COLOR, labelcolor=LABEL_COLOR)

# ── B: 총 CGM 기간 ────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
style_ax(ax_b, 'B  Total CGM Days per Dataset')

df_b = df_stats.sort_values('total_days', ascending=True)
y_pos = np.arange(len(df_b))
bar_colors_b = [DISEASE_COLORS[d] for d in df_b['disease_type']]
bars_b = ax_b.barh(y_pos, df_b['total_days'] / 1000, color=bar_colors_b, height=0.7, edgecolor='none')
for bar, src in zip(bars_b, df_b['source']):
    if src == 'New':
        bar.set_hatch('///')
        bar.set_edgecolor('#ffffff50')

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(df_b['dataset'], fontsize=8, color=LABEL_COLOR)
ax_b.set_xlabel('Total CGM Days (thousands)', color=LABEL_COLOR, fontsize=9)
ax_b.tick_params(axis='x', colors=LABEL_COLOR)

# Avg days/person annotation
for i, (_, row) in enumerate(df_b.iterrows()):
    ax_b.text(row['total_days'] / 1000 + 0.5, i,
              f"{row['avg_days_pp']:.0f}d/pt",
              va='center', ha='left', fontsize=6.5, color='#8b949e')

# ── C: 충분한 데이터 비율 ────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
style_ax(ax_c, 'C  Days with Adequate CGM Data (≥70%)')

df_c = df_stats.sort_values('pct_adequate', ascending=True)
y_pos = np.arange(len(df_c))

# Stacked bar: adequate vs inadequate
ax_c.barh(y_pos, df_c['pct_adequate'], color='#2A9D8F', height=0.7, label='Adequate (≥70%)', edgecolor='none')
ax_c.barh(y_pos, 100 - df_c['pct_adequate'], left=df_c['pct_adequate'],
          color='#E63946', height=0.7, alpha=0.5, label='Inadequate (<70%)', edgecolor='none')
ax_c.set_xlim(0, 100)
ax_c.axvline(70, color='#F4A261', linewidth=1.5, linestyle='--', alpha=0.8)
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(df_c['dataset'], fontsize=8, color=LABEL_COLOR)
ax_c.set_xlabel('% CGM Days', color=LABEL_COLOR, fontsize=9)
ax_c.tick_params(axis='x', colors=LABEL_COLOR)
ax_c.legend(fontsize=7.5, facecolor='#1c2128', edgecolor=GRID_COLOR, labelcolor=LABEL_COLOR, loc='lower right')

# ── D: 혈당 역학 TBR/TIR/TAR ─────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
style_ax(ax_d, 'D  Glucose Dynamics (TBR / TIR / TAR)')

df_d = df_stats.sort_values('mean_tir', ascending=True)
y_pos = np.arange(len(df_d))

ax_d.barh(y_pos, df_d['mean_tbr'], color='#E63946', height=0.7, label='TBR (<70)', edgecolor='none')
ax_d.barh(y_pos, df_d['mean_tir'], left=df_d['mean_tbr'],
          color='#2A9D8F', height=0.7, label='TIR (70–180)', edgecolor='none')
ax_d.barh(y_pos, df_d['mean_tar'],
          left=df_d['mean_tbr'] + df_d['mean_tir'],
          color='#F4A261', height=0.7, label='TAR (>180)', edgecolor='none')

ax_d.axvline(70, color='white', linewidth=1, linestyle='--', alpha=0.4)
ax_d.set_xlim(0, 100)
ax_d.set_yticks(y_pos)
ax_d.set_yticklabels(df_d['dataset'], fontsize=8, color=LABEL_COLOR)
ax_d.set_xlabel('% CGM Readings', color=LABEL_COLOR, fontsize=9)
ax_d.tick_params(axis='x', colors=LABEL_COLOR)
ax_d.legend(fontsize=7.5, facecolor='#1c2128', edgecolor=GRID_COLOR, labelcolor=LABEL_COLOR,
            loc='lower right')

fig.suptitle('Glucose-ML Project — Dataset Overview\n(Glucose-ML Collection + DiaData Extension)',
             color=LABEL_COLOR, fontsize=14, fontweight='bold', y=0.98)

fig_path = OUT_DIR / "figure2_dataset_overview.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {fig_path}")

# ─── Table 1 ──────────────────────────────────────────────────────────────────
print("\nRendering Table 1...")

table1_cols = ['dataset', 'disease_type', 'sampling_min', 'source',
               'n_subjects', 'total_days', 'avg_days_pp',
               'pct_adequate', 'mean_tir', 'mean_tar', 'mean_tbr']
df_t1 = df_stats[table1_cols].copy()
df_t1.columns = [
    'Dataset', 'Type', 'Rate\n(min)', 'Source',
    'N\nSubjects', 'Total CGM\nDays', 'Avg Days\n/Subject',
    'Adequate\nDays (%)', 'TIR\n(%)', 'TAR\n(%)', 'TBR\n(%)'
]
df_t1 = df_t1.sort_values('N\nSubjects', ascending=False)
for col in ['Total CGM\nDays', 'Avg Days\n/Subject']:
    df_t1[col] = df_t1[col].round(0).astype(int)
for col in ['Adequate\nDays (%)', 'TIR\n(%)', 'TAR\n(%)', 'TBR\n(%)']:
    df_t1[col] = df_t1[col].round(1)

# Matplotlib table figure
fig2, ax2 = plt.subplots(figsize=(20, len(df_t1) * 0.45 + 1.5))
fig2.patch.set_facecolor('#0d1117')
ax2.set_facecolor('#0d1117')
ax2.axis('off')

col_labels = list(df_t1.columns)
cell_text  = df_t1.values.tolist()

table = ax2.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Color headers
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#238636')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Row coloring by disease type
type_col_idx = list(df_t1.columns).index('Type')
for i, row in enumerate(df_t1.itertuples(), start=1):
    dt = row.Type
    base_color = {'T1D': '#1a0a0a', 'T2D': '#0a0f1a', 'ND': '#0a1a15', 'Mixed': '#1a1205'}[dt]
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_facecolor(base_color)
        cell.set_text_props(color='#e6edf3')
        cell.set_edgecolor('#30363d')

# Highlight new datasets
source_col_idx = list(df_t1.columns).index('Source')
for i, row in enumerate(df_t1.itertuples(), start=1):
    if row.Source == 'New':
        for j in range(len(col_labels)):
            cell = table[i, j]
            current = cell.get_facecolor()
            # slightly brighter
            cell.set_facecolor(tuple(min(1.0, c + 0.08) for c in current))
            cell.set_edgecolor('#58a6ff')

ax2.set_title('Table 1: Dataset Summary — Glucose-ML Project',
              color='#e6edf3', fontsize=13, fontweight='bold', pad=15)

t1_path = OUT_DIR / "table1_dataset_summary.png"
fig2.savefig(t1_path, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close()
print(f"  Saved: {t1_path}")

# ─── Print summary ────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(df_stats[['dataset','disease_type','n_subjects','total_days','avg_days_pp',
                'pct_adequate','mean_tir','mean_tar','mean_tbr']].to_string(index=False))
