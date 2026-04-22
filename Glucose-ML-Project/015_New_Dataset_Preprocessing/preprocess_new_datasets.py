"""
New Dataset Preprocessing Pipeline
====================================
대상: RT-CGM, CITY, SENCE, WISDM, FLAIR, SHD, ReplaceBG
출력: 003_Glucose-ML-collection/{name}/{name}-extracted-glucose-files/{ptid}.csv
컬럼: timestamp, glucose_value_mg_dl

전처리 규칙: 999_Preprocessing_Rules.md Rule 1~4
  - Range: 40~400 mg/dL
  - RoC: 30초 미만 간격 제거, 20 mg/dL/min 초과 제거
  - 보간 없음
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, r"C:\Users\user\Documents\NPJ2\Glucose-ML-Project")
from global_config import GlobalConfig

RAW_ROOT  = Path(r"C:\Users\user\Documents\NPJ2\new-data")
OUT_ROOT  = GlobalConfig.DATA_ROOT
MIN_G     = GlobalConfig.MIN_GLUCOSE
MAX_G     = GlobalConfig.MAX_GLUCOSE
MAX_ROC   = GlobalConfig.MAX_ROC_MG_DL_MIN
MIN_SEC   = GlobalConfig.MIN_TIME_DIFF_SEC


# ─── Helper: Validate and save per-subject files ───────────────────────────────

def apply_quality_filter(df):
    """
    Rule 1: Range filter (40~400 mg/dL)
    Rule 2: RoC filter (30s gap, 20 mg/dL/min)
    """
    df = df.copy()
    df['glucose_value_mg_dl'] = pd.to_numeric(df['glucose_value_mg_dl'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'glucose_value_mg_dl'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Rule 1: Range
    df = df[(df['glucose_value_mg_dl'] >= MIN_G) & (df['glucose_value_mg_dl'] <= MAX_G)]

    # Rule 2: RoC
    dt = df['timestamp'].diff().dt.total_seconds()
    dg = df['glucose_value_mg_dl'].diff().abs()
    roc = dg / (dt / 60.0)
    valid = (dt.isna()) | ((dt >= MIN_SEC) & (roc <= MAX_ROC))
    df = df[valid].reset_index(drop=True)

    return df[['timestamp', 'glucose_value_mg_dl']]


def save_subjects(df_all, ds_name, ptid_col='PtID'):
    out_dir = OUT_ROOT / ds_name / f"{ds_name}-extracted-glucose-files"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = df_all[ptid_col].unique()
    saved, skipped = 0, 0

    for ptid in tqdm(subjects, desc=f"  {ds_name}"):
        sub = df_all[df_all[ptid_col] == ptid].copy()
        sub = apply_quality_filter(sub)
        if len(sub) < 10:
            skipped += 1
            continue
        out_file = out_dir / f"{ptid}.csv"
        sub.to_csv(out_file, index=False)
        saved += 1

    print(f"  {ds_name}: {saved} subjects saved, {skipped} skipped (<10 rows)")
    return saved


# ─── SAS timestamp parser ───────────────────────────────────────────────────────
def parse_sas_dt(s):
    """29JAN2000:16:11:21.000 → datetime"""
    try:
        return pd.to_datetime(s, format='%d%b%Y:%H:%M:%S.%f')
    except:
        try:
            return pd.to_datetime(s, format='%d%b%Y:%H:%M:%S')
        except:
            return pd.NaT


# ─── 1. RT-CGM ─────────────────────────────────────────────────────────────────
def process_rt_cgm():
    print("\n[1/7] RT-CGM")
    data_dir = RAW_ROOT / "RT_CGM_Randomized_Clinical_Trial" / "RT_CGM_Randomized_Clinical_Trial" / "DataTables"
    cgm_files = sorted(data_dir.glob("tblADataRTCGM*.csv"))
    if not cgm_files:
        print("  ERROR: no CGM files found")
        return 0

    frames = []
    for f in cgm_files:
        df = pd.read_csv(f, encoding='latin-1', low_memory=False)
        df = df.rename(columns={'Glucose': 'glucose_value_mg_dl', 'DeviceDtTm': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        frames.append(df[['PtID', 'timestamp', 'glucose_value_mg_dl']])

    all_data = pd.concat(frames, ignore_index=True)
    return save_subjects(all_data, "RT-CGM")


# ─── 2. CITY ──────────────────────────────────────────────────────────────────
def process_jaeb_cgm_analysis(raw_dir, ds_name):
    """CITY, SENCE, WISDM: cgmAnalysis Ext.txt (pipe-delimited)"""
    print(f"\n[?] {ds_name}")
    data_dir = raw_dir / "Data Tables"
    cgm_file = data_dir / "cgmAnalysis Ext.txt"
    if not cgm_file.exists():
        cgm_file = data_dir / "cgmAnalysis RCT.txt"
    if not cgm_file.exists():
        print(f"  ERROR: no cgmAnalysis file in {data_dir}")
        return 0

    df = pd.read_csv(cgm_file, sep='|', encoding='latin-1', low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  Columns: {list(df.columns)[:6]}")

    df['timestamp'] = df['DeviceDtTm'].apply(parse_sas_dt)
    df = df.rename(columns={'Value': 'glucose_value_mg_dl'})
    return save_subjects(df[['PtID', 'timestamp', 'glucose_value_mg_dl']], ds_name)


# ─── 3. FLAIR ─────────────────────────────────────────────────────────────────
def process_flair():
    print("\n[5/7] FLAIR")
    data_dir = RAW_ROOT / "FLAIRPublicDataSet" / "Data Tables"
    cgm_file = data_dir / "FLAIRDeviceCGM.txt"

    df = pd.read_csv(cgm_file, sep='|', encoding='latin-1', low_memory=False)
    # Filter unusable
    if 'Unusuable' in df.columns:
        df = df[df['Unusuable'] == False]
    df['timestamp'] = pd.to_datetime(df['DataDtTm'], errors='coerce')
    df = df.rename(columns={'CGM': 'glucose_value_mg_dl'})
    return save_subjects(df[['PtID', 'timestamp', 'glucose_value_mg_dl']], "FLAIR")


# ─── 4. SHD ───────────────────────────────────────────────────────────────────
def process_shd():
    print("\n[6/7] SHD")
    data_dir = RAW_ROOT / "SevereHypoDataset-c14d3739-6a20-449c-bbae-c02ff1764a91" / "Data Tables"
    cgm_file = data_dir / "BDataCGM.txt"

    df = pd.read_csv(cgm_file, sep='|', encoding='latin-1', low_memory=False)
    df = df.rename(columns={'Glucose': 'glucose_value_mg_dl'})

    # Reconstruct timestamp: base 2000-01-01 + DeviceDaysFromEnroll days + DeviceTm
    base = datetime(2000, 1, 1)
    def reconstruct_ts(row):
        try:
            days = int(row['DeviceDaysFromEnroll'])
            t = datetime.strptime(str(row['DeviceTm']).strip(), '%H:%M:%S')
            return base + timedelta(days=days, hours=t.hour, minutes=t.minute, seconds=t.second)
        except:
            return pd.NaT

    df['timestamp'] = df.apply(reconstruct_ts, axis=1)
    df = df.dropna(subset=['glucose_value_mg_dl'])
    return save_subjects(df[['PtID', 'timestamp', 'glucose_value_mg_dl']], "SHD")


# ─── 5. ReplaceBG ─────────────────────────────────────────────────────────────
def process_replacebg():
    print("\n[7/7] ReplaceBG")
    data_dir = RAW_ROOT / "Replace-BG Dataset" / "Data Tables"
    cgm_file = data_dir / "HDeviceCGM.txt"

    df = pd.read_csv(cgm_file, sep='|', encoding='latin-1', low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Filter CGM readings only (exclude calibration/BGM records)
    if 'RecordType' in df.columns:
        df = df[df['RecordType'] == 'CGM'].copy()

    # Glucose column
    df = df.rename(columns={'GlucoseValue': 'glucose_value_mg_dl'})
    if 'glucose_value_mg_dl' not in df.columns:
        print("  ERROR: no glucose column found")
        return 0

    # Reconstruct timestamp from DeviceDtTmDaysFromEnroll + DeviceTm
    base = datetime(2000, 1, 1)
    def recon(row):
        try:
            days = int(row['DeviceDtTmDaysFromEnroll'])
            t = datetime.strptime(str(row['DeviceTm']).strip(), '%H:%M:%S')
            return base + timedelta(days=days, hours=t.hour, minutes=t.minute, seconds=t.second)
        except:
            return pd.NaT
    df['timestamp'] = df.apply(recon, axis=1)
    df = df.dropna(subset=['glucose_value_mg_dl'])
    return save_subjects(df[['PtID', 'timestamp', 'glucose_value_mg_dl']], "ReplaceBG")



# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = {}

    results['RT-CGM']    = process_rt_cgm()
    results['CITY']      = process_jaeb_cgm_analysis(
        RAW_ROOT / "CITYPublicDataset-344bea7d-8085-4deb-8038-6cb747a744e3", "CITY")
    results['SENCE']     = process_jaeb_cgm_analysis(
        RAW_ROOT / "SENCEPublicDataset-ed021673-573d-436c-9b15-49dbad67bd35", "SENCE")
    results['WISDM']     = process_jaeb_cgm_analysis(
        RAW_ROOT / "WISDMPublicDataset-18f24ae5-b4fb-4e93-bec6-7021086419fa", "WISDM")
    results['FLAIR']     = process_flair()
    results['SHD']       = process_shd()
    results['ReplaceBG'] = process_replacebg()

    print("\n" + "="*50)
    print("=== Preprocessing Summary ===")
    print(f"{'Dataset':<15} {'Subjects Saved':>15}")
    print("-"*32)
    for ds, n in results.items():
        print(f"  {ds:<13} {n:>15}")
    print(f"\nTotal: {sum(results.values())} subjects")
