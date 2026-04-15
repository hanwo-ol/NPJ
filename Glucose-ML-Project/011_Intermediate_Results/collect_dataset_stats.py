"""
Dataset Summary Statistics Collector
=====================================
12개 분석 대상 데이터셋에 대해 논문 Table 1 / Figure 2 스타일 통계를 수집.

수집 항목:
  - 환자(Subject) 수
  - 총 CGM 읽기 횟수 (rows)
  - 총 예측 윈도우 수 (23,064 ~ 26M 등)
  - 데이터 수집 기간 (일수)
  - 평균 혈당 ± 표준편차 (mg/dL)
  - 혈당 중앙값, 최소, 최대
  - TIR(70-180), TAR(>180), TBR(<70) %
  - 센서 샘플링 간격 (5분 or 15분)
  - 환자군 (T1DM / T2DM / ND / GDM)
"""

import gc
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "012_Tier_3_Advanced_ML"))

# ─── 메타데이터 (임상 정보: 논문/데이터 설명서 기반) ───
DATASET_META = {
    "AIDET1D": {
        "cohort": "T1DM", "country": "Turkey",
        "sensor": "Dexcom G5", "interval_min": 5,
        "covariates_desc": "CGM only",
        "ref": "AIDET"
    },
    "BIGIDEAs": {
        "cohort": "ND/PreD", "country": "USA",
        "sensor": "Dexcom G6", "interval_min": 5,
        "covariates_desc": "CGM + Nutrition (4종)",
        "ref": "BIGIDEAs"
    },
    "Bris-T1D_Open": {
        "cohort": "T1DM", "country": "Australia",
        "sensor": "Libre/Dexcom", "interval_min": 5,
        "covariates_desc": "CGM + Insulin + Carbs + HR + Steps + Sleep",
        "ref": "Bris-T1D"
    },
    "CGMacros_Dexcom": {
        "cohort": "ND", "country": "USA",
        "sensor": "Dexcom G6", "interval_min": 5,
        "covariates_desc": "2 CGM sensors + Nutrition (15종) + Activity",
        "ref": "CGMacros"
    },
    "CGMacros_Libre": {
        "cohort": "ND", "country": "USA",
        "sensor": "FreeStyle Libre", "interval_min": 15,
        "covariates_desc": "2 CGM sensors + Nutrition (15종) + Activity",
        "ref": "CGMacros"
    },
    "CGMND": {
        "cohort": "ND", "country": "Multiple",
        "sensor": "Various", "interval_min": 5,
        "covariates_desc": "CGM only",
        "ref": "CGMND"
    },
    "GLAM": {
        "cohort": "GDM", "country": "Spain",
        "sensor": "Libre", "interval_min": 15,
        "covariates_desc": "CGM + Meal event marker",
        "ref": "GLAM"
    },
    "HUPA-UCM": {
        "cohort": "T1DM/ND", "country": "Spain",
        "sensor": "Dexcom", "interval_min": 5,
        "covariates_desc": "CGM + Insulin + Carbs + HR + Steps",
        "ref": "HUPA-UCM"
    },
    "IOBP2": {
        "cohort": "T1DM", "country": "USA",
        "sensor": "Dexcom", "interval_min": 5,
        "covariates_desc": "CGM + Bolus insulin",
        "ref": "IOBP2"
    },
    "Park_2025": {
        "cohort": "ND", "country": "Korea",
        "sensor": "Dexcom G7", "interval_min": 5,
        "covariates_desc": "CGM + Carbs",
        "ref": "Park_2025"
    },
    "PEDAP": {
        "cohort": "T1DM", "country": "USA",
        "sensor": "Dexcom", "interval_min": 5,
        "covariates_desc": "CGM + Basal + Bolus + Carbs",
        "ref": "PEDAP"
    },
    "UCHTT1DM": {
        "cohort": "T1DM", "country": "USA",
        "sensor": "Dexcom", "interval_min": 5,
        "covariates_desc": "CGM + Bolus insulin",
        "ref": "UCHTT1DM"
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLECTION_ROOT = PROJECT_ROOT / "003_Glucose-ML-collection"
OUTPUT_CSV = PROJECT_ROOT / "011_Intermediate_Results" / "dataset_summary_stats.csv"


def compute_stats(ds_name: str) -> dict | None:
    aug_dir = COLLECTION_ROOT / ds_name / f"{ds_name}-time-augmented"
    pfiles = sorted(aug_dir.glob("*.csv"))
    if not pfiles:
        return None

    all_glucose = []
    durations = []
    n_subjects = len(pfiles)

    for pf in tqdm(pfiles, desc=ds_name, leave=False):
        df = pd.read_csv(pf, low_memory=False)
        if 'glucose_value_mg_dl' not in df.columns:
            continue
        g = pd.to_numeric(df['glucose_value_mg_dl'], errors='coerce').dropna().values
        all_glucose.append(g)

        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
            if len(ts) >= 2:
                duration_days = (ts.max() - ts.min()).total_seconds() / 86400
                durations.append(duration_days)

    if not all_glucose:
        return None

    G = np.concatenate(all_glucose)
    dur = np.array(durations) if durations else np.array([np.nan])

    meta = DATASET_META.get(ds_name, {})
    interval = meta.get("interval_min", 5)
    n_readings = len(G)
    # 윈도우 수는 베이스라인 결과에서 실측값 사용
    window_counts = {
        "AIDET1D": 471528, "BIGIDEAs": 36409, "Bris-T1D_Open": 818600,
        "CGMacros_Dexcom": 415299, "CGMacros_Libre": 455705, "CGMND": 114409,
        "GLAM": 26165917, "HUPA-UCM": 309092, "IOBP2": 14027905,
        "Park_2025": 23064, "PEDAP": 7055811, "UCHTT1DM": 27220,
    }

    return {
        "Dataset": ds_name,
        "Cohort": meta.get("cohort", "—"),
        "Country": meta.get("country", "—"),
        "N_Subjects": n_subjects,
        "N_Readings": n_readings,
        "N_Windows": window_counts.get(ds_name, "—"),
        "Duration_days_mean": round(np.mean(dur), 1),
        "Duration_days_max":  round(np.max(dur), 1),
        "Glucose_mean": round(np.mean(G), 1),
        "Glucose_std":  round(np.std(G), 1),
        "Glucose_median": round(np.median(G), 1),
        "Glucose_min": round(np.min(G), 1),
        "Glucose_max": round(np.max(G), 1),
        "TIR_pct": round(np.mean((G >= 70) & (G <= 180)) * 100, 1),
        "TAR_pct": round(np.mean(G > 180) * 100, 1),
        "TBR_pct": round(np.mean(G < 70) * 100, 1),
        "Interval_min": interval,
        "Sensor": meta.get("sensor", "—"),
        "Covariates": meta.get("covariates_desc", "—"),
    }


def main():
    rows = []
    for ds_name in DATASET_META:
        print(f"Processing {ds_name}...")
        row = compute_stats(ds_name)
        if row:
            rows.append(row)
            print(f"  N={row['N_Subjects']}, readings={row['N_Readings']:,}, "
                  f"mean_G={row['Glucose_mean']}±{row['Glucose_std']}, "
                  f"TIR={row['TIR_pct']}%")
        else:
            print(f"  SKIP (no data)")
        gc.collect()

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUTPUT_CSV}")
    print(df[["Dataset", "Cohort", "N_Subjects", "Glucose_mean",
              "Glucose_std", "TIR_pct", "TAR_pct", "TBR_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
