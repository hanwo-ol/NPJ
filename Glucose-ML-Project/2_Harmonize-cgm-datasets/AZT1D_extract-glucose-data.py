import os
import pandas as pd
from pathlib import Path

def process_azt1d():
    print("Starting custom extraction for AZT1D...")
    
    glucose_ml_dir = Path("C:/Users/user/Documents/NPJ2/Glucose-ML-Project")
    
    # Raw Data Path
    raw_dir = glucose_ml_dir / "1_Auto-scripts/Original-Glucose-ML-datasets/AZT1D_raw_data/AZT1D_Unzipped/AZT1D 2025/CGM Records"
    
    # Target Path for main pipeline
    target_glucose_dir = glucose_ml_dir / "3_Glucose-ML-collection/AZT1D/AZT1D-extracted-glucose-files"
    target_glucose_dir.mkdir(parents=True, exist_ok=True)
    
    # Target Path for extended features
    target_ext_dir = glucose_ml_dir / "3_Glucose-ML-collection/AZT1D/AZT1D-extended-features"
    target_ext_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        print("Error: Raw data directory not found.")
        return
        
    count = 0
    for subject_dir in raw_dir.iterdir():
        if not subject_dir.is_dir() or "Subject" not in subject_dir.name:
            continue
            
        subject_id = subject_dir.name  # like "Subject 1"
        csv_file = subject_dir / f"{subject_id}.csv"
        
        if not csv_file.exists():
            continue
            
        # Parse CSV
        df = pd.read_csv(csv_file, low_memory=False)
        
        if "EventDateTime" not in df.columns or "CGM" not in df.columns:
            print(f"Skipping {subject_id}: required columns not found.")
            continue
            
        # Drop fully missing CGM
        df = df.dropna(subset=["CGM"]).copy()
        
        # 1. Format for standard Glucose-ML Pipeline
        df_ml = df[["EventDateTime", "CGM"]].copy()
        df_ml.rename(columns={"EventDateTime": "timestamp", "CGM": "glucose_value_mg_dl"}, inplace=True)
        # Sort and drop exact duplicates
        df_ml["timestamp"] = pd.to_datetime(df_ml["timestamp"], errors="coerce")
        df_ml = df_ml.dropna(subset=["timestamp"])
        df_ml = df_ml.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        
        out_glucose_file = target_glucose_dir / f"{subject_id}.csv"
        df_ml.to_csv(out_glucose_file, index=False)
        
        # 2. Save Extended Multi-modal Features
        df_ext = df.copy()
        df_ext.rename(columns={"EventDateTime": "timestamp"}, inplace=True)
        df_ext["timestamp"] = pd.to_datetime(df_ext["timestamp"], errors="coerce")
        out_ext_file = target_ext_dir / f"{subject_id}_extended.csv"
        df_ext.to_csv(out_ext_file, index=False)
        count += 1
        
    print(f"AZT1D extraction finished successfully. Total processed: {count} participants.")

if __name__ == "__main__":
    process_azt1d()
