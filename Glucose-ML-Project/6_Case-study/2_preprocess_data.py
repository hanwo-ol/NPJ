from pathlib import Path
import pandas as pd
import numpy as np

def process_files(project_df, extracted_glucose_file_path, project, max_days, minimum_coverage):
    manifest_rows = []
    
    out_folder = Path("Processed-Data") / project
    out_folder.mkdir(parents=True, exist_ok=True)
    
    log_file = Path("Processed-Data") / f"log_{project}.txt"
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"--- Processing Dataset: {project} ---\n")

    for i, row in project_df.iterrows():
        person_id = str(row["person_id"])
        diabetes_type = str(row["diabetes_type"])
        split_category = str(row["split_assignment"])
        dataset = str(row["dataset"])

        person_data = extracted_glucose_file_path / f"{person_id}.csv"

        if not person_data.exists():
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{person_id}] Alert: Raw data file not found. Skipping.\n")
            continue
                
        df = pd.read_csv(person_data)
        df = df[["timestamp", "glucose_value_mg_dl"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["glucose_value_mg_dl"] = pd.to_numeric(df["glucose_value_mg_dl"], errors="coerce")
        
        n_rows_raw = len(df)
        df = df.dropna()
        
        # Sort and drop exact duplicate timestamps (no resampling / no interpolating)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        n_rows_cleaned = len(df)

        if n_rows_cleaned < 2:
            manifest_rows.append({
                "person_id": person_id, "diabetes_type": diabetes_type,
                "split_assignment": split_category, "dataset": dataset,
                "passed": "no_valid_rows", "num_rows_raw": n_rows_raw,
                "num_rows_processed": 0, "num_valid_days": 0
            })
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{person_id}] Excluded: Not enough active rows after dropping NaNs.\n")
            continue

        # Dynamic Coverage Calculation based on Native Sampling Rate
        time_diffs = df["timestamp"].diff()
        median_gap = time_diffs.median()
        
        if pd.isna(median_gap) or median_gap.total_seconds() <= 0:
            expected_readings = 288 # fallback to 5 mins default
        else:
            expected_readings = pd.Timedelta("1D") / median_gap
            
        df["date"] = df["timestamp"].dt.date
        day_counts = (df.groupby("date")["glucose_value_mg_dl"].count().reset_index())
        day_counts.columns = ["date", "non_missing_count"]
        day_counts["coverage"] = day_counts["non_missing_count"] / expected_readings

        valid_days = day_counts[day_counts["coverage"] >= minimum_coverage]["date"].tolist()
        valid_days = valid_days[:max_days]

        df = df[df["date"].isin(valid_days)].copy()
        df = df.drop(columns=["date"])

        n_rows_processed = len(df)
        n_valid_days = len(valid_days)

        output_file = out_folder / f"{person_id}.csv"

        if n_rows_processed == 0:
            status = "no"
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{person_id}] Excluded: No days met the {minimum_coverage*100}% coverage threshold based on {median_gap} native interval ({expected_readings:.1f} expected/day).\n")
        else:
            df.to_csv(output_file, index=False)
            status = "yes"
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{person_id}] Passed: Native Median Gap {median_gap}. Valid Days: {n_valid_days}/{len(day_counts)}. Rows survived: {n_rows_cleaned} -> {n_rows_processed}\n")

        manifest_rows.append({
            "person_id": person_id,
            "diabetes_type": diabetes_type,
            "split_assignment": split_category,
            "dataset": dataset,
            "passed": status,
            "num_rows_raw": n_rows_raw,
            "num_rows_processed": n_rows_processed,
            "num_valid_days": n_valid_days
        })

    return manifest_rows

def main():
    max_days = 15
    minimum_coverage = 0.7

    script_path = Path(__file__).resolve()
    glucose_ml_dir = script_path.parent.parent

    split_participants_df = pd.read_csv("participant_splits.csv",dtype={"person_id": str})
    
    manifest_path = Path("Processed-Data/preprocessing_manifest.csv")
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        final_rows = manifest_df.to_dict("records")
        existing_projects = manifest_df["dataset"].unique()
        all_projects = split_participants_df["dataset"].unique()
        project_ids = [p for p in all_projects if p not in existing_projects]
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        final_rows = []
        project_ids = split_participants_df["dataset"].unique()

    for project in project_ids:
        # Clear/Create log file for this project
        log_file = Path("Processed-Data") / f"log_{project}.txt"
        if log_file.exists():
            log_file.unlink()
            
        project_df = split_participants_df[split_participants_df["dataset"]==project]
        project_path = f'{project}-extracted-glucose-files'
        extracted_glucose_file_path = glucose_ml_dir / "3_Glucose-ML-collection" / project/ project_path

        project_rows = process_files(project_df, extracted_glucose_file_path, project, max_days, minimum_coverage)
        for row in project_rows:
            final_rows.append(row)

    manifest_df = pd.DataFrame(final_rows)
    manifest_df.to_csv("Processed-Data/preprocessing_manifest.csv", index=False)

    print("Done!")

if __name__ == "__main__":
    main()