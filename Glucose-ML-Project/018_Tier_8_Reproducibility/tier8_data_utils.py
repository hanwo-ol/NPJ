"""
Tier 8: Data Utilities — Cache & LODO Data Loader
===================================================
E드라이브 NPZ 캐시 생성/로딩 및 LODO 데이터 구성을 담당한다.
Tier 7의 피처 추출/윈도우 생성 로직을 재활용한다.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '016_Tier_7_Cross_Disease'))
sys.path.insert(0, str(Path(__file__).parent.parent / '013_Tier_6_Domain_Adaptation'))

from tier8_config import Tier8Config
from tier7_data_utils import load_dataset, load_target_split


# ─── NPZ 캐시 관리 ────────────────────────────────────────────────────────────

def cache_path(ds_name: str) -> Path:
    """데이터셋의 NPZ 캐시 경로."""
    return Tier8Config.CACHE_DIR / f"{ds_name}.npz"


def is_cached(ds_name: str) -> bool:
    return cache_path(ds_name).exists()


def build_cache(ds_name: str, force: bool = False) -> bool:
    """단일 데이터셋의 윈도우를 NPZ로 캐싱. 이미 존재하면 건너뜀."""
    cp = cache_path(ds_name)
    if cp.exists() and not force:
        return True

    X, y = load_dataset(ds_name)
    if X is None:
        print(f"  [SKIP] {ds_name}: no data")
        return False

    np.savez(cp, X=X, y=y)
    size_mb = cp.stat().st_size / (1024 ** 2)
    print(f"  [CACHED] {ds_name}: {len(X):,} windows, {size_mb:.1f} MB")
    return True


def build_all_caches(force: bool = False):
    """26개 전체 데이터셋 캐시 생성."""
    print("=" * 60)
    print("  Building NPZ caches for all 26 datasets")
    print("=" * 60)

    t0 = time.perf_counter()
    success, fail = 0, 0

    for ds_name in tqdm(Tier8Config.DATASET_REGISTRY.keys(),
                        desc="Caching", ncols=70):
        if build_cache(ds_name, force=force):
            success += 1
        else:
            fail += 1

    elapsed = time.perf_counter() - t0
    total_mb = sum(
        cache_path(ds).stat().st_size / (1024 ** 2)
        for ds in Tier8Config.DATASET_REGISTRY
        if cache_path(ds).exists()
    )
    print(f"\nDone: {success} cached, {fail} failed, "
          f"{total_mb:.0f} MB total, {elapsed:.0f} sec")


def load_cached(ds_name: str) -> tuple:
    """NPZ 캐시에서 (X, y) 로딩. 캐시 없으면 CSV 폴백."""
    cp = cache_path(ds_name)
    if cp.exists():
        data = np.load(cp)
        return data['X'], data['y']
    else:
        print(f"  [WARN] Cache miss for {ds_name}, falling back to CSV")
        return load_dataset(ds_name)


# ─── 환자별 분할 캐시 ─────────────────────────────────────────────────────────

def cache_path_split(ds_name: str) -> Path:
    """환자 단위 분할 캐시 경로."""
    return Tier8Config.CACHE_DIR / f"{ds_name}_split.npz"


def build_split_cache(ds_name: str, force: bool = False) -> bool:
    """단일 데이터셋의 환자 단위 분할(train/val/test)을 NPZ로 캐싱."""
    cp = cache_path_split(ds_name)
    if cp.exists() and not force:
        return True

    splits = load_target_split(ds_name)
    X_tr, y_tr = splits['train']
    X_val, y_val = splits['val']
    X_te, y_te = splits['test']

    np.savez(cp,
             X_train=X_tr, y_train=y_tr,
             X_val=X_val, y_val=y_val,
             X_test=X_te, y_test=y_te)
    return True


def build_all_split_caches(force: bool = False):
    """26개 전체 데이터셋 분할 캐시 생성."""
    print("\n" + "=" * 60)
    print("  Building split caches (train/val/test)")
    print("=" * 60)

    t0 = time.perf_counter()
    for ds_name in tqdm(Tier8Config.DATASET_REGISTRY.keys(),
                        desc="Split caching", ncols=70):
        build_split_cache(ds_name, force=force)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.0f} sec")


def load_cached_split(ds_name: str) -> dict:
    """분할 캐시에서 (train/val/test) 로딩."""
    cp = cache_path_split(ds_name)
    if cp.exists():
        data = np.load(cp)
        return {
            'train': (data['X_train'], data['y_train']),
            'val':   (data['X_val'],   data['y_val']),
            'test':  (data['X_test'],  data['y_test']),
        }
    else:
        print(f"  [WARN] Split cache miss for {ds_name}, falling back to CSV")
        return load_target_split(ds_name)


# ─── LODO 데이터 구성 ─────────────────────────────────────────────────────────

def _proportional_sample(X: np.ndarray, y: np.ndarray,
                         n_target: int, seed: int) -> tuple:
    """데이터셋 비례 서브샘플링."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_target, len(X)), replace=False)
    return X[idx], y[idx]


def build_lodo_source(target_ds: str, freq_min: int,
                      seed: int = None) -> tuple:
    """
    LODO: target_ds를 제외한 동일 주기 그룹의 나머지 데이터셋을 합산.
    MAX_SOURCE_WINDOWS 초과 시 데이터셋 비례 서브샘플링 적용.
    """
    if seed is None:
        seed = Tier8Config.SEED

    group_datasets = Tier8Config.datasets_by_group(freq_min)
    source_datasets = [ds for ds in group_datasets if ds != target_ds]

    # 각 데이터셋의 전체 윈도우 로딩 (캐시 사용)
    ds_arrays = {}
    for ds in source_datasets:
        X, y = load_cached(ds)
        if X is not None and len(X) > 0:
            ds_arrays[ds] = (X, y)

    if not ds_arrays:
        raise RuntimeError(f"No source data for group {freq_min}min "
                           f"(excluding {target_ds})")

    total_raw = sum(len(X) for X, _ in ds_arrays.values())
    cap = Tier8Config.MAX_SOURCE_WINDOWS

    if cap is not None and total_raw > cap:
        sampled = []
        for ds, (X, y) in ds_arrays.items():
            n_alloc = max(1, int(cap * len(X) / total_raw))
            Xs, ys = _proportional_sample(X, y, n_alloc, seed)
            sampled.append((Xs, ys))
        X_src = np.vstack([s[0] for s in sampled])
        y_src = np.concatenate([s[1] for s in sampled])
    else:
        X_src = np.vstack([X for X, _ in ds_arrays.values()])
        y_src = np.concatenate([y for _, y in ds_arrays.values()])

    return X_src, y_src


# ─── 엔트리포인트: 전체 캐시 생성 ─────────────────────────────────────────────

if __name__ == '__main__':
    build_all_caches()
    build_all_split_caches()
