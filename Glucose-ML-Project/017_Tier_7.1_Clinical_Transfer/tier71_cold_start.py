"""
Tier 7.1: N=1 Cold Start Personalization
==========================================
단계 3: 신규 환자 개인화 전이학습

Leave-One-Patient-Out 설계:
  각 환자 p에 대해, 처음 D일의 데이터로 개인 모델을 학습하고,
  나머지 데이터에서 평가한다. 전이학습(TrAdaBoost)과 비교한다.

3개 타겟 모두 실행: ShanghaiT2DM, CITY, Colas_2019

최적화:
  - population 모델: 환자당 1회만 학습 (D값 무관)
  - population 데이터: 전체 배열에서 인덱스로 제외 (vstack 반복 방지)
  - 환자 단위 병렬 처리 (ThreadPoolExecutor)

실행: python 017_Tier_7.1_Clinical_Transfer/tier71_cold_start.py
      python 017_Tier_7.1_Clinical_Transfer/tier71_cold_start.py --targets ShanghaiT2DM
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '016_Tier_7_Cross_Disease'))

from global_config import GlobalConfig
from tier71_config import Tier71Config
from tier7_data_utils import build_windows
from tier7_tradaboost import TrAdaBoostRegressor


# --- 평가 지표 -------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mard(y_true, y_pred):
    mask = y_true > 0
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


# --- LightGBM 학습 --------------------------------------------------------

def train_lgbm_safe(X_tr, y_tr, X_val, y_val, sample_weight=None):
    """학습 데이터가 부족하면 None을 반환한다."""
    if len(X_tr) < 20 or len(X_val) < 5:
        return None

    p      = dict(Tier71Config.LGBM_PARAMS)
    ds_tr  = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    try:
        return lgb.train(
            p, ds_tr,
            num_boost_round=Tier71Config.LGBM_ROUNDS,
            valid_sets=[ds_val],
            callbacks=[
                lgb.early_stopping(Tier71Config.LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
    except Exception:
        return None


# --- 환자 단위 데이터 로딩 -------------------------------------------------

def load_patient_with_timestamps(ds_name: str) -> list:
    """
    환자별 (X, y, timestamps, patient_id) 리스트를 반환한다.
    timestamps는 cold start의 일수 분할에 사용된다.
    """
    ds_dir = (GlobalConfig.DATA_ROOT / ds_name
              / f"{ds_name}-extracted-glucose-files")
    files  = sorted(ds_dir.glob("*.csv"))
    patients = []

    for f in tqdm(files, desc=f"  Loading {ds_name}", leave=False, ncols=70):
        try:
            df = pd.read_csv(f, low_memory=False)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['glucose_value_mg_dl'] = pd.to_numeric(
                df['glucose_value_mg_dl'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'glucose_value_mg_dl'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            mask = ((df['glucose_value_mg_dl'] >= GlobalConfig.MIN_GLUCOSE) &
                    (df['glucose_value_mg_dl'] <= GlobalConfig.MAX_GLUCOSE))
            df = df[mask].reset_index(drop=True)

            X, y = build_windows(df)
            if X is not None and len(X) > 0:
                lb = GlobalConfig.LOOKBACK_STEPS
                ph = GlobalConfig.PREDICTION_STEPS
                timestamps = pd.DatetimeIndex(df['timestamp'])
                win_ts = [timestamps[i - 1]
                          for i in range(lb, len(timestamps) - ph + 1)]
                patients.append({
                    'X': X, 'y': y,
                    'timestamps': win_ts,
                    'patient_id': f.stem,
                    'start_date': win_ts[0].date() if win_ts else None,
                })
        except Exception:
            pass

    return patients


# --- 단일 환자 Cold Start 실험 --------------------------------------------

def run_patient_cold_start(patient: dict, pop_model: lgb.Booster,
                           pop_X: np.ndarray, pop_y: np.ndarray,
                           val_X: np.ndarray, val_y: np.ndarray,
                           days_list: list) -> list:
    """
    단일 환자에 대해 D일별 3-way 비교를 수행한다.

    최적화: population 모델은 호출자가 1회 학습하여 전달한다.
    TrAdaBoost만 D값마다 재학습한다 (개인 데이터 양이 달라지므로).
    """
    records = []
    pid     = patient['patient_id']
    X_all   = patient['X']
    y_all   = patient['y']
    ts_all  = patient['timestamps']
    start   = patient['start_date']

    if start is None or len(X_all) < 30:
        return records

    for D in days_list:
        cutoff = pd.Timestamp(start) + pd.Timedelta(days=D)
        train_mask = np.array([t < cutoff for t in ts_all])
        test_mask  = ~train_mask

        n_train = np.sum(train_mask)
        n_test  = np.sum(test_mask)

        if n_train < 10 or n_test < 10:
            continue

        X_p_tr = X_all[train_mask]
        y_p_tr = y_all[train_mask]
        X_p_te = X_all[test_mask]
        y_p_te = y_all[test_mask]

        # Val: 환자 개인 데이터의 마지막 20%
        n_val_p = max(5, int(n_train * 0.2))
        X_p_val = X_p_tr[-n_val_p:]
        y_p_val = y_p_tr[-n_val_p:]
        X_p_fit = X_p_tr[:-n_val_p]
        y_p_fit = y_p_tr[:-n_val_p]

        base = {'target': '', 'patient_id': pid, 'days': D,
                'n_train_windows': int(n_train),
                'n_test_windows': int(n_test)}

        # (a) personal_only: 환자 D일만으로 학습
        m_personal = train_lgbm_safe(X_p_fit, y_p_fit, X_p_val, y_p_val)
        if m_personal is not None:
            preds_p = m_personal.predict(X_p_te)
            records.append({**base, 'model': 'personal_only',
                            'rmse': rmse(y_p_te, preds_p),
                            'mae': mae(y_p_te, preds_p),
                            'mard': mard(y_p_te, preds_p)})
        else:
            records.append({**base, 'model': 'personal_only',
                            'rmse': np.nan, 'mae': np.nan, 'mard': np.nan})

        # (b) population: 사전 학습된 모델로 zero-shot 예측 (재학습 없음)
        if pop_model is not None:
            preds_pop = pop_model.predict(X_p_te)
            records.append({**base, 'model': 'population',
                            'rmse': rmse(y_p_te, preds_pop),
                            'mae': mae(y_p_te, preds_pop),
                            'mard': mard(y_p_te, preds_pop)})
        else:
            records.append({**base, 'model': 'population',
                            'rmse': np.nan, 'mae': np.nan, 'mard': np.nan})

        # (c) tradaboost: population(source) + 환자 D일(target) -> TrAdaBoost
        tada = TrAdaBoostRegressor()
        try:
            tada.fit(pop_X, pop_y, X_p_fit, y_p_fit, X_p_val, y_p_val)
            preds_t = tada.predict(X_p_te)
            records.append({**base, 'model': 'tradaboost',
                            'rmse': rmse(y_p_te, preds_t),
                            'mae': mae(y_p_te, preds_t),
                            'mard': mard(y_p_te, preds_t)})
        except Exception:
            records.append({**base, 'model': 'tradaboost',
                            'rmse': np.nan, 'mae': np.nan, 'mard': np.nan})

    return records


# --- 타겟별 Cold Start 실험 ------------------------------------------------

def run_cold_start_for_target(target_name: str, days_list: list,
                              out_dir: Path):
    """
    단일 타겟에 대해 Leave-One-Patient-Out Cold Start를 실행한다.

    메모리 최적화:
      - 전체 데이터를 1회 합산하고, 환자별 인덱스 범위로 제외한다.
      - np.vstack을 환자 수만큼 반복하지 않는다.
    """
    print(f"\n{'='*58}")
    print(f"  Cold Start: {target_name}")
    print(f"{'='*58}")

    patients = load_patient_with_timestamps(target_name)
    n_total  = len(patients)
    print(f"  Total patients: {n_total}")

    if n_total < 5:
        print("  [SKIP] Not enough patients.")
        return

    # 전체 데이터 1회 합산 + 환자별 인덱스 범위 기록
    all_X_list = [p['X'] for p in patients]
    all_y_list = [p['y'] for p in patients]
    all_X = np.vstack(all_X_list)
    all_y = np.concatenate(all_y_list)

    # 환자별 시작/끝 인덱스
    boundaries = []
    offset = 0
    for p in patients:
        n = len(p['X'])
        boundaries.append((offset, offset + n))
        offset += n

    print(f"  Total windows: {len(all_X):,}")

    # Val 환자 인덱스 (마지막 15%)
    n_val = max(1, int(n_total * 0.15))
    val_indices = list(range(n_total - n_val, n_total))

    all_records = []

    for idx in tqdm(range(n_total), desc=f"  LOPO {target_name}", ncols=70):
        # 환자 idx 제외한 population 데이터 (인덱스 마스킹)
        start_i, end_i = boundaries[idx]
        pop_mask = np.ones(len(all_X), dtype=bool)
        pop_mask[start_i:end_i] = False
        pop_X = all_X[pop_mask]
        pop_y = all_y[pop_mask]

        # Val 데이터 (제외 환자가 val에 속하면 재조정)
        val_idx_adjusted = [v for v in val_indices if v != idx]
        if not val_idx_adjusted:
            val_idx_adjusted = [v for v in range(n_total) if v != idx][-n_val:]

        val_mask = np.zeros(len(all_X), dtype=bool)
        for vi in val_idx_adjusted:
            s, e = boundaries[vi]
            val_mask[s:e] = True
        val_X = all_X[val_mask]
        val_y = all_y[val_mask]

        # Population 모델 1회 학습 (D값과 무관)
        pop_model = train_lgbm_safe(pop_X, pop_y, val_X, val_y)

        # D일별 3-way 비교
        records = run_patient_cold_start(
            patients[idx], pop_model, pop_X, pop_y,
            val_X, val_y, days_list)

        for r in records:
            r['target'] = target_name
        all_records.extend(records)

    if not all_records:
        print("  No results.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / f"cold_start_{target_name}.csv",
              index=False, encoding='utf-8-sig')
    print(f"  Saved: cold_start_{target_name}.csv")

    # 집계
    summary = (df.groupby(['target', 'days', 'model'])
               .agg(rmse_median=('rmse', 'median'),
                    rmse_mean=('rmse', 'mean'),
                    rmse_q25=('rmse', lambda x: x.quantile(0.25)),
                    rmse_q75=('rmse', lambda x: x.quantile(0.75)),
                    n_patients=('patient_id', 'nunique'))
               .reset_index())
    summary.to_csv(out_dir / f"cold_start_summary_{target_name}.csv",
                   index=False, encoding='utf-8-sig')
    print(f"  Saved: cold_start_summary_{target_name}.csv")

    # 교차점 분석
    crossover = find_crossover(summary, target_name)
    if crossover is not None:
        crossover.to_csv(out_dir / f"crossover_{target_name}.csv",
                         index=False, encoding='utf-8-sig')
        print(f"  Saved: crossover_{target_name}.csv")

    # 시각화
    plot_cold_start_curve(summary, target_name, out_dir)

    print(f"\n  Summary for {target_name}:")
    print(summary.to_string(index=False))


def find_crossover(summary: pd.DataFrame, target_name: str):
    """personal_only가 population을 넘는 교차점 일수를 찾는다."""
    try:
        pop = summary[summary['model'] == 'population'].set_index('days')['rmse_median']
        per = summary[summary['model'] == 'personal_only'].set_index('days')['rmse_median']
        common_days = sorted(set(pop.index) & set(per.index))
        for d in common_days:
            if per[d] < pop[d]:
                return pd.DataFrame([{
                    'target': target_name,
                    'crossover_days': d,
                    'personal_rmse': per[d],
                    'population_rmse': pop[d],
                }])
    except Exception:
        pass
    return None


def plot_cold_start_curve(summary: pd.DataFrame, target_name: str,
                          out_dir: Path):
    """Cold Start 학습 곡선 시각화."""
    colors = {
        'personal_only': '#e63946',
        'population':    '#8b949e',
        'tradaboost':    '#2a9d8f',
    }
    labels = {
        'personal_only': 'Personal Only',
        'population':    'Population (zero-shot)',
        'tradaboost':    'TrAdaBoost Transfer',
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    for model in ['personal_only', 'population', 'tradaboost']:
        sub = summary[summary['model'] == model].sort_values('days')
        if sub.empty:
            continue
        ax.plot(sub['days'], sub['rmse_median'],
                'o-', color=colors[model], label=labels[model], linewidth=2)
        ax.fill_between(sub['days'], sub['rmse_q25'], sub['rmse_q75'],
                        alpha=0.15, color=colors[model])

    ax.set_xlabel('Accumulated Days', color='#e6edf3')
    ax.set_ylabel('RMSE Median (mg/dL)', color='#e6edf3')
    ax.set_title(f'Cold Start: {target_name}',
                 color='#e6edf3', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#e6edf3')
    ax.spines[:].set_color('#30363d')
    ax.yaxis.grid(True, color='#30363d', linewidth=0.5)
    ax.set_xticks(Tier71Config.COLD_START_DAYS)
    ax.legend(facecolor='#1c2128', edgecolor='#30363d',
              labelcolor='#e6edf3', fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"cold_start_curve_{target_name}.png",
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: cold_start_curve_{target_name}.png")


# --- 구동부 ----------------------------------------------------------------

def run_all_cold_start(target_filter: list = None):
    """전체 또는 지정 타겟에 대해 Cold Start 실험을 수행한다."""
    out_dir = Tier71Config.OUT_DIR / "cold_start"
    out_dir.mkdir(parents=True, exist_ok=True)
    days = Tier71Config.COLD_START_DAYS

    targets = target_filter or Tier71Config.COLD_START_TARGETS

    for target_name in targets:
        run_cold_start_for_target(target_name, days, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tier 7.1 Step 3: N=1 Cold Start Personalization')
    parser.add_argument('--targets', nargs='+',
                        choices=['ShanghaiT2DM', 'CITY', 'Colas_2019'],
                        default=None,
                        help='실행할 타겟 (기본: 전체 3개)')
    args = parser.parse_args()
    run_all_cold_start(target_filter=args.targets)
