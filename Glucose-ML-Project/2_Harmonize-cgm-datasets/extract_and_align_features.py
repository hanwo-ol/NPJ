import os
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def process_state_events(cgm_df, event_df, event_time_col, cgm_time_col='timestamp'):
    """
    1. 상태(State) 유지형 이벤트
    merge_asof with direction='backward'
    """
    event_df = event_df.copy()
    event_df.rename(columns={event_time_col: cgm_time_col}, inplace=True)
    event_df[cgm_time_col] = pd.to_datetime(event_df[cgm_time_col], errors='coerce')
    event_df = event_df.dropna(subset=[cgm_time_col]).sort_values(cgm_time_col)
    
    # Drop exact duplicates to avoid merge_asof issues
    event_df = event_df.drop_duplicates(subset=[cgm_time_col], keep='last')
    
    cgm_df = cgm_df.sort_values(cgm_time_col)
    
    merged = pd.merge_asof(cgm_df, event_df, on=cgm_time_col, direction='backward')
    return merged

def process_discrete_events(cgm_df, event_df, event_time_col, cgm_time_col='timestamp', tol_minutes=5):
    """
    2. 이산(Discrete) 일회성 이벤트
    merge_asof with direction='backward' and tolerance='5m'
    """
    event_df = event_df.copy()
    event_df.rename(columns={event_time_col: cgm_time_col}, inplace=True)
    event_df[cgm_time_col] = pd.to_datetime(event_df[cgm_time_col], errors='coerce')
    event_df = event_df.dropna(subset=[cgm_time_col]).sort_values(cgm_time_col)
    
    # Drop exact duplicates to avoid merge_asof issues
    event_df = event_df.drop_duplicates(subset=[cgm_time_col], keep='last')
    
    cgm_df = cgm_df.sort_values(cgm_time_col)
    
    # Merge using backward direction and tolerance
    merged = pd.merge_asof(cgm_df, event_df, on=cgm_time_col, direction='backward', tolerance=pd.Timedelta(minutes=tol_minutes))
    
    # Fill missing discrete events with 0 (No Event Occurred) vs 1 (or the discrete dosage amount)
    # The columns added from event_df are those not in cgm_df
    new_cols = [col for col in event_df.columns if col != cgm_time_col]
    for col in new_cols:
        merged[col] = merged[col].fillna(0)
        
    return merged

def process_dataset(project_name, dataset_mappings, glucose_ml_dir):
    print(f"\n[{project_name}] Processing extended/aligned features...")
    
    glucose_dir = glucose_ml_dir / f"3_Glucose-ML-collection/{project_name}/{project_name}-extracted-glucose-files"
    output_dir  = glucose_ml_dir / f"3_Glucose-ML-collection/{project_name}/{project_name}-time-augmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not glucose_dir.exists():
        print(f"[{project_name}] Base CGM dir not found. Skipping.")
        return
        
    loaded_mappings = []
    for map_dict in dataset_mappings:
        event_file = Path(map_dict['file'])
        if not event_file.exists(): 
            continue
            
        if event_file.is_file():
            try:
                sep = '|' if str(event_file).endswith('.txt') else ','
                edf = pd.read_csv(event_file, sep=sep, low_memory=False)
                
                # --- IOBP2 MealDose Custom Time Construction ---
                if 'IOBP2MealDose' in event_file.name:
                    edf['MealDoseHr'] = edf['MealDoseHr'].fillna(0).astype(int).astype(str).str.zfill(2)
                    edf['MealDoseMin'] = edf['MealDoseMin'].fillna(0).astype(int).astype(str).str.zfill(2)
                    # For AmPm: AM = 0, PM = 1 (or string) -> just assume base format if naive, or standard parse
                    # A robust but simple parse: just use Date if time is too complex, but let's try direct concat
                    time_str = edf['MealDoseHr'] + ':' + edf['MealDoseMin']
                    edf['DeviceDtTm'] = pd.to_datetime(edf['MealDoseDt'] + ' ' + time_str, errors='coerce')
                    
                loaded_mappings.append({'map_dict': map_dict, 'edf': edf, 'is_dir': False})
            except Exception as e:
                print(f"Error reading {event_file.name}: {e}")
                continue
        else:
            loaded_mappings.append({'map_dict': map_dict, 'edf': None, 'is_dir': True})

    for p_file in glucose_dir.glob("*.csv"):
        person_id = p_file.stem
        cgm_df = pd.read_csv(p_file)
        cgm_df['timestamp'] = pd.to_datetime(cgm_df['timestamp'], errors='coerce')
        cgm_df = cgm_df.dropna(subset=['timestamp']).sort_values('timestamp')
        
        augmented_df = cgm_df.copy()
        
        # Look for mappings
        for lm in loaded_mappings:
            map_dict = lm['map_dict']
            is_dir = lm['is_dir']
            
            if is_dir:
                suffix = map_dict.get('suffix', '')
                subj_file = Path(map_dict['file']) / f"{person_id}{suffix}"
                if not subj_file.exists():
                    continue
                try:
                    edf = pd.read_csv(subj_file)
                except Exception:
                    continue
            else:
                edf = lm['edf']
            
            id_col = map_dict.get('id_col', 'NONE')
            if not is_dir:
                if id_col not in edf.columns: continue
                edf[id_col] = edf[id_col].astype(str)
                subject_edf = edf[edf[id_col] == person_id].copy()
            else:
                subject_edf = edf.copy()
            
            if subject_edf.empty:
                continue
                
            event_time_col = map_dict['time_col']
            if event_time_col not in subject_edf.columns: continue
            
            if len(map_dict['cols']) == 0:
                # If no columns specified, it means the occurrence of the timestamp itself is the event
                subject_edf['EventMarker'] = 1.0
                target_cols = [event_time_col, 'EventMarker']
            else:
                target_cols = [event_time_col] + map_dict['cols']
            
            # Ensure target cols actually exist
            available_cols = [c for c in target_cols if c in subject_edf.columns]
            subject_edf = subject_edf[available_cols]
            
            if map_dict['type'] == 'state':
                augmented_df = process_state_events(augmented_df, subject_edf, event_time_col)
            elif map_dict['type'] == 'discrete':
                augmented_df = process_discrete_events(augmented_df, subject_edf, event_time_col, tol_minutes=5)
                
        out_file = output_dir / f"{person_id}.csv"
        augmented_df.to_csv(out_file, index=False)
        
    print(f"[{project_name}] Time-augmented feature integration complete.")

def main():
    glucose_ml_dir = Path("C:/Users/user/Documents/NPJ2/Glucose-ML-Project")
    raw_base = glucose_ml_dir / "1_Auto-scripts/Original-Glucose-ML-datasets"
    
    # We define configurations based on the Candidates we investigated
    configs = {
        'AIDET1D': [
            {'type': 'state', 'file': raw_base / "AIDET1D_Data_Tables/AIDEInsulin.txt", 'id_col': 'PtID', 'time_col': 'InsTypeStartDt', 'cols': ['InsInjectionFreqInt', 'InsRoute']}
        ],
        'GLAM': [
            {'type': 'discrete', 'file': raw_base / "GLAM_Data_Tables/GLAMMealLog.txt", 'id_col': 'PtID', 'time_col': 'MealDateTime', 'cols': []},
            {'type': 'state', 'file': raw_base / "GLAM_Data_Tables/GLAMSteroidUse.txt", 'id_col': 'PtID', 'time_col': 'SteroidStartDtTm', 'cols': ['SteroidType']}
        ],
        'PEDAP': [
            {'type': 'state', 'file': raw_base / "PEDAP_Data_Files/PEDAPTandemBASALDELIVERY.txt", 'id_col': 'PtID', 'time_col': 'DeviceDtTm', 'cols': ['BasalRate']},
            {'type': 'discrete', 'file': raw_base / "PEDAP_Data_Files/PEDAPTandemBolusDelivered.txt", 'id_col': 'PtID', 'time_col': 'DeviceDtTm', 'cols': ['BolusType']},
            {'type': 'discrete', 'file': raw_base / "PEDAP_Data_Files/PEDAPMealExerciseLog.txt", 'id_col': 'PtID', 'time_col': 'MealStartTm', 'cols': ['CarbSize']}
        ],
        'IOBP2': [
            {'type': 'state', 'file': raw_base / "IOBP2_Data_Tables/IOBP2BasalRtChg.txt", 'id_col': 'PtID', 'time_col': 'DeviceDtTm', 'cols': ['BasalRate']},
            {'type': 'discrete', 'file': raw_base / "IOBP2_Data_Tables/IOBP2MealDose.txt", 'id_col': 'PtID', 'time_col': 'DeviceDtTm', 'cols': ['MealDoseAnnounceAmt']}
        ],
        'BIGIDEAs': [
            {'type': 'discrete', 'file': glucose_ml_dir / "3_Glucose-ML-collection/BIGIDEAs/BIGIDEAs-extended-features", 'suffix': '_Food.csv', 'time_col': 'time_begin', 'cols': ['calorie', 'carbohydrate', 'protein', 'fat']},
            {'type': 'state', 'file': glucose_ml_dir / "3_Glucose-ML-collection/BIGIDEAs/BIGIDEAs-extended-features", 'suffix': '_HR.csv', 'time_col': 'datetime', 'cols': ['hr']}
        ],
        'UCHTT1DM': [
            {'type': 'discrete', 'file': glucose_ml_dir / "3_Glucose-ML-collection/UCHTT1DM/UCHTT1DM-extended-features", 'suffix': '_Carbohidrates.csv', 'time_col': 'Unnamed: 0', 'cols': ['Value (g)']},
            {'type': 'discrete', 'file': glucose_ml_dir / "3_Glucose-ML-collection/UCHTT1DM/UCHTT1DM-extended-features", 'suffix': '_Insulin.csv', 'time_col': 'Unnamed: 0', 'cols': ['Dose (U)']},
            {'type': 'state', 'file': glucose_ml_dir / "3_Glucose-ML-collection/UCHTT1DM/UCHTT1DM-extended-features", 'suffix': '_Heart_Rate.csv', 'time_col': 'Unnamed: 0', 'cols': ['Value']}
        ]
    }
    
        # Process the complex asynchronous ones
    for project, mappings in configs.items():
        process_dataset(project, mappings, glucose_ml_dir)
        
    # Process natively aligned datasets (already merged at extraction)
    native_projects = [
        'ShanghaiT1DM', 'ShanghaiT2DM', 'Bris-T1D_Open', 'CGMacros_Dexcom',
        'CGMacros_Libre', 'HUPA-UCM', 'Park_2025'
    ]
    
    import shutil
    for project in native_projects:
        print(f"\n[{project}] Finalizing native extended alignments...")
        ext_dir = glucose_ml_dir / f"3_Glucose-ML-collection/{project}/{project}-extended-features"
        aug_dir = glucose_ml_dir / f"3_Glucose-ML-collection/{project}/{project}-time-augmented"
        aug_dir.mkdir(parents=True, exist_ok=True)
        
        if not ext_dir.exists():
            continue
            
        for ext_file in ext_dir.glob("*_extended.csv"):
            subj_id = ext_file.name.replace("_extended.csv", "")
            out_file = aug_dir / f"{subj_id}.csv"
            shutil.copy2(ext_file, out_file)
            
        print(f"[{project}] Native augmented final integration complete.")
        
    print("Done generating augmented time-aligned datasets!")
    
if __name__ == "__main__":
    main()
