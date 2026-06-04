"""
S1-2 LODO 재개: 미완료 데이터셋만 처리
========================================
중단된 LODO를 이어서 실행한다.
이미 완료된 타겟은 건너뛰고, 미완료 타겟만 처리한 뒤 기존 결과에 합산한다.
"""

import sys
import warnings
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '016_Tier_7_Cross_Disease'))
sys.path.insert(0, str(Path(__file__).parent.parent / '013_Tier_6_Domain_Adaptation'))

from tier8_config import Tier8Config
from run_lodo import run_lodo_iteration, setup_logger, plot_heatmap


def main():
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("  S1-2 LODO: Resuming incomplete targets")
    logger.info("=" * 60)

    out_dir = Tier8Config.OUT_DIR / "s1_2_lodo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 기존 결과 로딩
    interim_path = out_dir / "lodo_results_interim.csv"
    if interim_path.exists():
        df_existing = pd.read_csv(interim_path)
        done_targets = set(df_existing['target'].unique())
        logger.info(f"  Existing results: {len(df_existing)} rows, "
                     f"{len(done_targets)} targets done")
    else:
        df_existing = pd.DataFrame()
        done_targets = set()

    # 미완료 타겟 식별
    all_results = df_existing.to_dict('records') if len(df_existing) > 0 else []
    t_start = time.perf_counter()

    for freq_min in Tier8Config.group_names():
        datasets = Tier8Config.datasets_by_group(freq_min)
        remaining = [ds for ds in datasets if ds not in done_targets]

        if not remaining:
            logger.info(f"\n  Group {freq_min}min: all {len(datasets)} done, skipping")
            continue

        logger.info(f"\n{'#'*60}")
        logger.info(f"  Group: {freq_min}min — {len(remaining)} remaining "
                     f"(of {len(datasets)})")
        logger.info(f"{'#'*60}")

        for target_ds in tqdm(remaining, desc=f"LODO {freq_min}min", ncols=70):
            results = run_lodo_iteration(target_ds, freq_min, logger, out_dir)
            all_results.extend(results)

            # 중간 저장
            df_interim = pd.DataFrame(all_results)
            df_interim.to_csv(interim_path, index=False, encoding='utf-8-sig')

    elapsed = time.perf_counter() - t_start

    # 최종 결과
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "lodo_results.csv", index=False, encoding='utf-8-sig')
    logger.info(f"\n\nTotal: {len(df)} rows, {df['target'].nunique()} targets")
    logger.info(f"Resume time: {elapsed/60:.1f} min")

    # 요약
    logger.info("\n" + "=" * 60)
    logger.info("  Summary: RMSE by Model (All Groups)")
    logger.info("=" * 60)
    summary = df.groupby('model')['rmse'].agg(['mean', 'std', 'count'])
    logger.info(f"\n{summary.to_string()}")

    # 질환 유형별 하위 분석
    logger.info("\n" + "=" * 60)
    logger.info("  Subgroup Analysis: Target Disease Type")
    logger.info("=" * 60)
    for disease in df['disease'].unique():
        sub = df[df['disease'] == disease]
        logger.info(f"\n  --- Disease: {disease} ---")
        sub_summary = sub.groupby('model')['rmse'].agg(['mean', 'std'])
        logger.info(f"{sub_summary.to_string()}")
        pivoted = sub.pivot_table(index='target', columns='model', values='rmse')
        if 'target_only' in pivoted.columns:
            for tl in ['coral', 'tradaboost']:
                if tl in pivoted.columns:
                    flips = (pivoted[tl] > pivoted['target_only']).sum()
                    total = len(pivoted)
                    logger.info(f"    Flip Rate ({tl} > target_only): "
                                f"{flips}/{total} = {flips/total:.1%}")

    # 히트맵
    logger.info("\nGenerating heatmaps...")
    for freq_min in Tier8Config.group_names():
        sub = df[df['group'] == f'{freq_min}min']
        if len(sub) > 0:
            plot_heatmap(sub, out_dir, f'{freq_min}min')
            logger.info(f"  Saved heatmap_{freq_min}min.png")

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
