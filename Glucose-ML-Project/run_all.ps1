param ()

Write-Host "Started Full Glucose-ML Pipeline Execution..."

# 1. Download all datasets
Write-Host "1. Downloading all 11 open-access datasets..."
cd C:\Users\user\Documents\NPJ2\Glucose-ML-Project\1_Auto-scripts
python auto-download-open-datasets.py d1namo bigideas shanghai uchtt1dm hupa-ucm cgmacros t1d-uom bris-t1d_open azt1d park_2025 physiocgm

# 2. Harmonize
Write-Host "2. Harmonizing all 11 open-access datasets..."
python auto-harmonize-CGM-datasets.py d1namo bigideas shanghai uchtt1dm hupa-ucm cgmacros t1d-uom bris-t1d_open azt1d park_2025 physiocgm

# 3. Process via Case-Study Pipeline
Write-Host "3. Running 6_Case-study Machine Learning Preprocessing Pipeline..."
cd C:\Users\user\Documents\NPJ2\Glucose-ML-Project\6_Case-study
python 1_split_participants.py
python 2_preprocess_data.py
python 3_calculate_features.py

Write-Host "All pipelines executed successfully."
