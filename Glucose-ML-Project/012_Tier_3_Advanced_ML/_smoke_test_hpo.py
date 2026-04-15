"""Quick smoke test for 02_optuna_hpo.py — Park_2025, 2 trials each model."""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import everything from the HPO script
from tier3_data_utils import discover_datasets
from importlib.util import spec_from_file_location, module_from_spec
import pathlib

spec = spec_from_file_location("hpo", pathlib.Path("02_optuna_hpo.py"))
hpo = module_from_spec(spec)
spec.loader.exec_module(hpo)

# Park_2025 (smallest dataset — fastest test)
datasets = hpo.discover_datasets()
park_path = [d for d in datasets if d.name == 'Park_2025'][0]
data = hpo.load_dataset_split(park_path)

n_tr = len(data['y_train'])
n_va = len(data['y_val'])
n_te = len(data['y_test'])
print(f"Split OK: train={n_tr:,}  val={n_va:,}  test={n_te:,}")
assert n_tr > n_va > 0 and n_te > 0, "Split ratio error"

# LightGBM 2 trials
r_lgbm = hpo.run_study('LightGBM', hpo.objective_lgbm, data, n_trials=2, timeout_sec=120)
assert r_lgbm is not None, "LightGBM study returned None"
print(f"LightGBM OK  val_RMSE={r_lgbm['val_RMSE']}  test_RMSE={r_lgbm['test_RMSE']}")

# CatBoost 2 trials
r_cat = hpo.run_study('CatBoost', hpo.objective_catboost, data, n_trials=2, timeout_sec=120)
assert r_cat is not None, "CatBoost study returned None"
print(f"CatBoost OK  val_RMSE={r_cat['val_RMSE']}  test_RMSE={r_cat['test_RMSE']}")

print("\nSmoke test PASSED!")
