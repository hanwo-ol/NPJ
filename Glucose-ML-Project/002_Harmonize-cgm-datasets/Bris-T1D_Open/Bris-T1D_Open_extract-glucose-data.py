import os
import pandas as pd
import sys
from pathlib import Path
LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_brist1d_data(df, subject_id, output_dir):
    '''
    Cleans and standardizes Bris-T1D CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format with additional quality checks.
    - Writes per-subject CSV files containing timestamped glucose values
    - Convert glucose records from raw mmol/L units to the project-standard mg/dL units.
    - Captures extended columns (meals, bolus, heart rate, etc.) to an _extended CSV.
    '''
    
    # Rename columns & convert timestamp data to the standardized names used throughout the project.
    df.rename(columns={"bg": "glucose_value_mg_dl"}, inplace=True)

    # Standardize time stamp data.
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extended Output (preserve everything except we ensure glucose is converted if bg exists)
    ext_dir = str(output_dir).replace("Bris-T1D_Open-extracted-glucose-files", "Bris-T1D_Open-extended-features")
    if "Standardized-datasets" in str(output_dir):
        ext_dir = str(output_dir) + "-extended-features"
    os.makedirs(ext_dir, exist_ok=True)
    
    # Convert glucose records from mmol/L to mg/dL
    df["glucose_value_mg_dl"] = (df["glucose_value_mg_dl"] * 18).round(1)
    
    # Save extended DataFrame
    df_ext = df.dropna(subset=['timestamp'])
    df_ext.to_csv(os.path.join(ext_dir, f"{subject_id}_extended.csv"), index=False)

    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    #Drop rows missing timestamps or glucose values.
    subj_df = df[["timestamp", "glucose_value_mg_dl"]]

    #Make blank csv output file.
    outfile = os.path.join(output_dir, f"{subject_id}.csv")
    # Populate the output file
    subj_df.to_csv(outfile, index=False)



def main():
    '''
    Processes raw data from the Bris-T1D dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    
    Input: Raw data directory.
    Output: Standardized CSV files for each subject. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each subject output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''
    if len(sys.argv) != 2:
        print("Invalid command. Usage: python Bris-T1D_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = Path(sys.argv[1])

    # Create output directories directly in target 3_Glucose-ML-collection struct
    glucose_ml_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = glucose_ml_dir / "3_Glucose-ML-collection/Bris-T1D_Open/Bris-T1D_Open-extracted-glucose-files"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through raw directory contents and pull the subject ID from the raw file name.
    count = 0
    for subject in input_path.rglob("**/processed_state/*.csv"):
        subject_id = subject.stem #pull the subject ID from the raw file name.
        df=pd.read_csv(subject)
        clean_brist1d_data(df, subject_id, output_dir)
        count += 1
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')
if __name__ == "__main__":
    main()