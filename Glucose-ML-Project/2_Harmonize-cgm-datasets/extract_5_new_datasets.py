import os
import pandas as pd
from pathlib import Path

def process_dataset(project_name, file_path, id_col, time_col, val_col, diabetes_type):
    print(f"Starting extraction for {project_name}...")
    glucose_ml_dir = Path("C:/Users/user/Documents/NPJ2/Glucose-ML-Project")
    
    target_dir = glucose_ml_dir / f"3_Glucose-ML-collection/{project_name}"
    target_glucose_dir = target_dir / f"{project_name}-extracted-glucose-files"
    target_ext_dir = target_dir / f"{project_name}-extended-features"
    
    target_glucose_dir.mkdir(parents=True, exist_ok=True)
    target_ext_dir.mkdir(parents=True, exist_ok=True)
    
    if not file_path.exists():
        print(f"Error: Raw data file {file_path} not found.")
        return
        
    print(f"Reading file for {project_name} into memory...")
    try:
        df = pd.read_csv(file_path, sep='|', low_memory=False)
    except Exception as e:
        print(f"Failed to read {project_name}: {e}")
        return
        
    if id_col not in df.columns or time_col not in df.columns or val_col not in df.columns:
        print(f"Error: Missing columns in {project_name}")
        return
        
    df = df.dropna(subset=[val_col]).copy()
    
    # Standard format
    df_ml = df[[id_col, time_col, val_col]].copy()
    df_ml.rename(columns={id_col: "person_id", time_col: "timestamp", val_col: "glucose_value_mg_dl"}, inplace=True)
    df_ml["timestamp"] = pd.to_datetime(df_ml["timestamp"], errors="coerce")
    df_ml['glucose_value_mg_dl'] = pd.to_numeric(df_ml['glucose_value_mg_dl'], errors="coerce")
    df_ml = df_ml.dropna(subset=["timestamp", "glucose_value_mg_dl"])
    
    unique_ids = set()
    count = 0
    
    print(f"Writing glucose files for {project_name}...")
    grouped = df_ml.groupby("person_id")
    for person_id, group in grouped:
        person_str = str(person_id)
        if not person_str or person_str.lower() == 'nan': continue
        unique_ids.add(person_str)
        
        group = group.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        out_glucose_file = target_glucose_dir / f"{person_str}.csv"
        group.drop(columns=["person_id"]).to_csv(out_glucose_file, index=False)
        count += 1
        
    print(f"Writing extended features for {project_name}...")
    df_ext = df.copy()
    df_ext.rename(columns={id_col: "person_id", time_col: "timestamp"}, inplace=True)
    df_ext["timestamp"] = pd.to_datetime(df_ext["timestamp"], errors="coerce")
    
    ext_grouped = df_ext.groupby("person_id")
    for person_id, group in ext_grouped:
        person_str = str(person_id)
        if not person_str or person_str.lower() == 'nan': continue
        
        out_ext_file = target_ext_dir / f"{person_str}_extended.csv"
        group.drop(columns=["person_id"]).to_csv(out_ext_file, index=False)
    
    print(f"{project_name} extraction finished. Participants: {count}")
    
    metadata = pd.DataFrame({"person_id": list(unique_ids), "diabetes_type": diabetes_type})
    metadata.to_csv(target_dir / f"{project_name}-metadata.csv", index=False)

def main():
    glucose_ml_dir = Path("C:/Users/user/Documents/NPJ2/Glucose-ML-Project")
    raw_base = glucose_ml_dir / "1_Auto-scripts/Original-Glucose-ML-datasets"
    
    datasets = [
        ("CGMND", raw_base / "CGMND_Data_Tables/CGMNDDeviceCGM.txt", "DeidentID", "DeviceDtTm", "Value", "NonDiabetic"),
        ("AIDET1D", raw_base / "AIDET1D_Data_Tables/AIDEDeviceCGM.txt", "PtID", "DataDtTm", "GlucValue", "Type 1"),
        ("GLAM", raw_base / "GLAM_Data_Tables/tblGLAMDeviceCGM.txt", "PtID", "DeviceDtTm", "Value", "GDM"),
        ("PEDAP", raw_base / "PEDAP_Data_Files/PEDAPDexcomClarityCGM.txt", "PtID", "DeviceDtTm", "CGM", "Type 1"),
        ("IOBP2", raw_base / "IOBP2_Data_Tables/IOBP2DeviceCGM.txt", "PtID", "DeviceDtTm", "Value", "Type 1")
    ]
    
    for proj, fpath, id_col, t_col, v_col, dtype in datasets:
        process_dataset(proj, fpath, id_col, t_col, v_col, dtype)
        print("-" * 40)

if __name__ == "__main__":
    main()
