import os
import pandas as pd
from pathlib import Path
import sys

LIME_GREEN = "\033[92m"
LIGHT_RED = "\033[91m"
R = "\033[0m"

def clean_uchtt1dm_data(df, subject, subject_id, output_dir):
    '''
    Cleans and standardizes UCHTT1DM CGM data by:
    - Renaming columns to project-standard names
    - Converting timestamps to pandas datetime format
    - Writing per-subject CSV files containing timestamped glucose values
    '''

    # Rename the columns
    df = df.rename(columns={'Unnamed: 0': 'timestamp', 'Value (mg/dl)': 'glucose_value_mg_dl'})

    #Convert timestamp column to Pandas readable format.
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Drop rows missing timestamps or glucose values
    df = df.dropna(subset=["timestamp", "glucose_value_mg_dl"])

    subj_df = df[["timestamp", "glucose_value_mg_dl"]]

    # Create an output csv for each subject using the subj_df variable.
    outfile = os.path.join(output_dir, f"{subject_id}.csv")
    subj_df.to_csv(outfile, index=False)
    
    # Extended Output Configuration
    raw_dir = Path(subject.parent)
    ext_dir = str(output_dir).replace("UCHTT1DM-extracted-glucose-files", "UCHTT1DM-extended-features")
    if "Standardized-datasets" in str(output_dir):
        ext_dir = str(output_dir) + "-extended-features"
    os.makedirs(ext_dir, exist_ok=True)
    
    # Check for external files in the same directory
    for file in raw_dir.glob("*.xlsx"):
        if file.name != "Glucose.xlsx" and not file.name.startswith("~$"):
            try:
                xf = pd.read_excel(file)
                # Convert Spaces to no-spaces for standard naming
                clean_name = file.stem.replace(" ", "_").replace(".", "")
                xf.to_csv(os.path.join(ext_dir, f"{subject_id}_{clean_name}.csv"), index=False)
            except Exception as e:
                pass
            


def main():
    '''
    Processes raw data from the UCHTT1DM dataset by pulling timestamp & glucose data and 
    standardizing column names to match project conventions.
    Input: Raw data directory of .xml files.
    Output: Standardized CSV files for each participant. Creates a directory "Standardized-datasets" that will contain the generated output.

    Each participant output file has 2 column's:
     1) "timestamp" = the CGM generated timestamp in which the associated glucose reading was recorded.
     2) "glucose_value_mg_dl" = the glucose reading in mg/dL units.
    '''

    if len(sys.argv) != 2:
        print("Invalid command. Usage: python UCHTT1DM_extract-glucose-data.py <input_folder>")
        print("Tip: Make sure to only pass 1 argument & that data exists in input directory")
        sys.exit(1)

    # Path to directory containing the raw data files.
    input_path = Path(sys.argv[1])

    # Create output directories directly in target 3_Glucose-ML-collection struct
    glucose_ml_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = glucose_ml_dir / "3_Glucose-ML-collection/UCHTT1DM/UCHTT1DM-extracted-glucose-files"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through raw directory contents and pull the subject ID from the raw file name.
    count = 0
    for subject in input_path.rglob("**/Glucose.xlsx"):
        df=pd.read_excel(subject)
        subject_id = subject.parent.name #pull the subject ID from the raw file name.
        clean_uchtt1dm_data(df, subject, subject_id, output_dir)
        count += 1
    print(f'{LIME_GREEN}Glucose-ML{R}: Standardized CGM records for {LIGHT_RED}{count}{R} subjects.')

if __name__ == "__main__":
    main()