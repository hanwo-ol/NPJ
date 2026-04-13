import os
import pandas as pd
from pathlib import Path

raw_dir = Path("C:/Users/user/Documents/NPJ2/Glucose-ML-Project/1_Auto-scripts/Original-Glucose-ML-datasets")
out_dir = Path("C:/Users/user/Documents/NPJ2/Glucose-ML-Project/6_Case-study/Processed-Data")
out_dir.mkdir(parents=True, exist_ok=True)

keywords = ["insulin", "carbs", "cho", "meal", "bolus", "basal", "step", "hr", "heart", "sleep", "activity", "kcal", "dose", "food", "diet", "eda", "temp"]
target_datasets = ["AZT1D", "Bris-T1D_Open", "D1NAMO", "T1D-UOM"]

for dataset_dir in raw_dir.iterdir():
    if not dataset_dir.is_dir():
        continue
    
    dataset_name = dataset_dir.name.replace("_raw_data", "")
    if dataset_name not in target_datasets:
        continue

    out_file = out_dir / f"candidate_{dataset_name}.txt"
    representative_files = list(dataset_dir.rglob("*.csv")) + list(dataset_dir.rglob("*.xlsx")) + list(dataset_dir.rglob("*.xml"))
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"--- Feature Candidate Report for {dataset_name} (Supplementary) ---\n\n")
        
        if not representative_files:
            f.write("No CSV or XLSX files found. If this dataset used to fail silently, it means the automated download failed.\n")
            f.write("A manual download check on Zenodo reported: '403 Forbidden' restricted access.\n")
            continue
            
        scanned_headers = set()
        candidates = set()
        
        # Analyze up to 15 distinct files 
        for file_path in representative_files[:15]:
            try:
                if file_path.suffix.lower() == ".csv":
                    try:
                        df = pd.read_csv(file_path, nrows=0, low_memory=False, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, nrows=0, low_memory=False, encoding='iso-8859-1')
                        f.write(f"(Note: File {file_path.name} required ISO-8859-1 decoding fallback.)\n")
                elif file_path.suffix.lower() == ".xlsx":
                    df = pd.read_excel(file_path, nrows=0)
                else: 
                    # If XML, pandas mapping is different, we skip raw header parsing in this script constraint
                    continue
                
                cols = list(df.columns)
                if not cols:
                    continue
                    
                col_tuple = tuple(cols)
                if col_tuple in scanned_headers:
                    continue # Skip files with identical schema
                scanned_headers.add(col_tuple)
                
                f.write(f"File Schema Source: {file_path.name}\n")
                f.write(f"Columns: {', '.join([str(c) for c in cols])}\n")
                
                local_candidates = [c for c in cols if any(k in str(c).lower() for k in keywords)]
                if local_candidates:
                    for c in local_candidates:
                        candidates.add(c)
                f.write("\n")
                
            except Exception as e:
                f.write(f"Error reading {file_path.name}: {str(e)}\n\n")
                
        if candidates:
            f.write("============================================================\n")
            f.write("★ Highly Recommended Extra Features for Multi-Modal Models:\n")
            for c in sorted(candidates):
                f.write(f"  - {c}\n")
            f.write("============================================================\n")
            
print("Done extracting candidates for targets.")
