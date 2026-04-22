"""
Tier 6: Data Utilities (eGMI Imputation & UMD Marker Generation)
================================================================
Contains data loading logic for Tier 6 Domain Adaptation & Ablation runs.
Imports from Tier 3 base logic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tier6_config import Tier6Config

# Ensure backend tier 3 utilities are accessible
sys.path.append(str(Path(__file__).parent.parent / "012_Tier_3_Advanced_ML"))
try:
    from tier3_data_utils import (
        discover_datasets, log, mape, downcast_to_float32,
        build_windows_with_features
    )
except ImportError:
    print("Cannot import tier3_data_utils. Make sure 012_Tier_3_Advanced_ML is adjacent.")

LOCAL_FEATURE_MAP = {
    'AIDET1D': [],
    'BIGIDEAs': ['calorie', 'protein'],
    'Bris-T1D_Open': ['insulin', 'carbs', 'hr', 'steps'],
    'CGMacros_Dexcom': ['HR', 'Calories (Activity)', 'Carbs', 'Protein', 'Fat'],
    'CGMacros_Libre':  ['HR', 'Calories (Activity)', 'Carbs', 'Protein', 'Fat'],
    'CGMND': [],
    'GLAM': [],
    'HUPA-UCM': ['heart_rate', 'steps', 'basal_rate', 'bolus_volume_delivered', 'carb_input'],
    'IOBP2': [],
    'Park_2025': [],
    'PEDAP': ['BasalRate'],
    'UCHTT1DM': ['Value (g)'],
}

COHORT_DEFAULTS = {
    'T1DM': {'age': 35, 'sex': 0.5, 'hba1c': 7.8},
    'ND':   {'age': 40, 'sex': 0.5, 'hba1c': 5.4},
    'GDM':  {'age': 30, 'sex': 1.0, 'hba1c': 5.9},
}

DATASET_COHORT = {
    'AIDET1D': 'T1DM', 'BIGIDEAs': 'ND', 'Bris-T1D_Open': 'T1DM',
    'CGMacros_Dexcom': 'T1DM', 'CGMacros_Libre': 'T1DM',
    'CGMND': 'ND', 'GLAM': 'GDM', 'HUPA-UCM': 'T1DM',
    'IOBP2': 'T1DM', 'Park_2025': 'ND', 'PEDAP': 'T1DM', 'UCHTT1DM': 'T1DM',
}

def load_metadata_mask(dset_path):
    """
    Load static metadata. Returns a dict of pid -> meta, 
    but for HbA1c, leaves it as None if missing.
    """
    ds = dset_path.name
    meta_file = dset_path / f"{ds}-metadata.csv"
    cohort = DATASET_COHORT.get(ds, 'T1DM')
    defaults = COHORT_DEFAULTS[cohort]

    if not meta_file.exists():
        return {}, defaults

    df = pd.read_csv(meta_file)
    static = {}
    for _, row in df.iterrows():
        pid = str(row.get('person_id', ''))
        
        # Age
        age_raw = row.get('age', None)
        if pd.notna(age_raw):
            try:
                age = float(age_raw)
            except:
                if '-' in str(age_raw):
                    parts = str(age_raw).split('-')
                    age = (float(parts[0]) + float(parts[1])) / 2
                else:
                    age = defaults['age']
        else:
            age = defaults['age']

        # Sex
        gender_raw = str(row.get('gender', '')).lower()
        sex = 1.0 if 'female' in gender_raw else (0.0 if 'male' in gender_raw else defaults['sex'])

        # HbA1c
        hba1c_raw = row.get('hba1c_%', None)
        if pd.notna(hba1c_raw):
            try:
                hba1c = float(hba1c_raw)
            except:
                hba1c = None
        else:
            hba1c = None

        static[pid] = {'age': age, 'sex': sex, 'hba1c': hba1c}
        
    return static, defaults

def generate_umd_marker(X_global_ts, velocity_idx=6, accel_idx=7):
    """
    Unannounced Meal Detection (UMD) Virtual Marker.
    Uses Velocity and Acceleration to guess if a positive transient instability occurred.
    """
    vel = X_global_ts[:, velocity_idx]
    acc = X_global_ts[:, accel_idx]
    
    # Simple rule-based UMD marker (probabilistic surrogate)
    # High velocity (>2.0 mg/dL/min approx) and positive acceleration
    prob = np.zeros_like(vel)
    mask = (vel > 1.5) & (acc > 0)
    prob[mask] = np.clip((vel[mask] - 1.5) * 0.5 + acc[mask] * 0.2, 0, 1)
    return prob.reshape(-1, 1)

def load_dataset_tier6(dset_path, use_egmi=False, use_umd_local=False):
    """
    Tier 6 DataLoader supporting Ablation features.
    - use_egmi: If True, calculates HbA1c via eGMI if missing. If False, uses cohort median.
    - use_umd_local: If True, discards actual local features and replaces with a 1-dim UMD marker.
    """
    ds = dset_path.name
    aug = dset_path / f"{ds}-time-augmented"
    pfiles = sorted(aug.glob("*.csv"))
    if not pfiles:
        return None

    local_cols = LOCAL_FEATURE_MAP.get(ds, [])
    # If we force UMD local, we don't need to parse actual local features from CSV
    if use_umd_local:
        feat_parse_cols = ['glucose_value_mg_dl']
    else:
        feat_parse_cols = ['glucose_value_mg_dl'] + local_cols

    metadata, defaults = load_metadata_mask(dset_path)
    
    all_X, all_Y, patient_ids = [], [], []
    patient_static = {}
    patient_mean_glucose = {}

    for pid_int, pf in enumerate(pfiles):
        df = pd.read_csv(pf, low_memory=False)
        if 'timestamp' not in df.columns or 'glucose_value_mg_dl' not in df.columns:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'glucose_value_mg_dl'])

        for c in feat_parse_cols:
            if c not in df.columns:
                df[c] = 0.0
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        Xm, Y, fnames = build_windows_with_features(df, feat_parse_cols)
        if len(Y) < 10:
            continue

        all_X.append(Xm)
        all_Y.append(Y)
        patient_ids.extend([pid_int] * len(Y))

        # Metadata matching
        pid_str = pf.stem
        static = metadata.get(pid_str, None)
        if static is None:
            for key in metadata:
                if key in pid_str or pid_str in key:
                    static = metadata[key]; break
        if static is None:
            static = {'age': defaults['age'], 'sex': defaults['sex'], 'hba1c': None}
        
        patient_static[pid_int] = static
        # Calculate mean glucose for eGMI
        patient_mean_glucose[pid_int] = df['glucose_value_mg_dl'].mean()

    if not all_X: return None

    X = np.vstack(all_X)
    Y = np.concatenate(all_Y)
    patient_ids = np.array(patient_ids)

    unique_pids = np.unique(patient_ids)
    rng = np.random.RandomState(Tier6Config.SEED)
    rng.shuffle(unique_pids)
    
    n_train = int(len(unique_pids) * Tier6Config.TRAIN_RATIO)
    n_val = int(len(unique_pids) * Tier6Config.VAL_RATIO)
    train_mask = np.isin(patient_ids, unique_pids[:n_train])
    val_mask = np.isin(patient_ids, unique_pids[n_train:n_train+n_val])
    test_mask = np.isin(patient_ids, unique_pids[n_train+n_val:])

    n_global_ts = Tier6Config.N_GLOBAL_TS_FEATURES
    X_global_ts = X[:, :n_global_ts]

    # Q/M Imputation
    qm = np.zeros((len(Y), 3))
    for pid_int in unique_pids:
        mask = patient_ids == pid_int
        s = patient_static.get(pid_int)
        qm[mask, 0] = s['age']
        qm[mask, 1] = s['sex']
        
        # HbA1c Imputation logic
        if s['hba1c'] is not None:
            qm[mask, 2] = s['hba1c']
        else:
            if use_egmi:
                # eGMI = 3.31 + 0.02392 * Mean_Glucose
                egmi_val = 3.31 + 0.02392 * patient_mean_glucose[pid_int]
                qm[mask, 2] = egmi_val
            else:
                qm[mask, 2] = defaults['hba1c']  # Median Imputation

    # Cohort One-Hot
    cohort = DATASET_COHORT.get(ds, 'T1DM')
    cohort_vec = np.zeros((len(Y), 3))
    if cohort == 'T1DM': cohort_vec[:, 0] = 1
    elif cohort == 'ND': cohort_vec[:, 1] = 1
    elif cohort == 'GDM': cohort_vec[:, 2] = 1

    X_global = np.hstack([X_global_ts, qm, cohort_vec])
    
    # Local Features Path
    if use_umd_local:
        # Generate UMD Virtual Marker dynamically from X_global_ts
        X_local = generate_umd_marker(X_global_ts)
        local_fnames = ["UMD_Meal_Prob"]
    else:
        if X.shape[1] > n_global_ts:
            X_local = X[:, n_global_ts:]
            local_fnames = [f'{c}_lag{i}' for c in local_cols for i in range(6)]
        else:
            X_local = np.zeros((len(Y), 0))
            local_fnames = []

    return {
        'name': ds,
        'cohort': cohort,
        'X_global_train': downcast_to_float32(X_global[train_mask]),
        'X_global_val':   downcast_to_float32(X_global[val_mask]),
        'X_global_test':  downcast_to_float32(X_global[test_mask]),
        'X_local_train':  downcast_to_float32(X_local[train_mask]),
        'X_local_val':    downcast_to_float32(X_local[val_mask]),
        'X_local_test':   downcast_to_float32(X_local[test_mask]),
        'X_full_train':   downcast_to_float32(np.hstack([X_global, X_local])[train_mask]),
        'X_full_val':     downcast_to_float32(np.hstack([X_global, X_local])[val_mask]),
        'X_full_test':    downcast_to_float32(np.hstack([X_global, X_local])[test_mask]),
        'y_train': Y[train_mask],
        'y_val':   Y[val_mask],
        'y_test':  Y[test_mask],
        'patient_ids_train': patient_ids[train_mask],
        'patient_ids_val':   patient_ids[val_mask],
        'patient_ids_test':  patient_ids[test_mask],
        'n_global_dim': X_global.shape[1],
        'n_local_dim': X_local.shape[1],
        'local_fnames': local_fnames,
    }
