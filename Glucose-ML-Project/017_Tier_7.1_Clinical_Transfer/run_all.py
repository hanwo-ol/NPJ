"""
Tier 7.1: 전체 실험 순차 실행
================================
단계 0 → 1 → 2 → 3 순서로 실행한다.
분석은 모든 실험 완료 후 한번에 수행한다.

실행: python 017_Tier_7.1_Clinical_Transfer/run_all.py
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '016_Tier_7_Cross_Disease'))


def run_step0():
    """단계 0: Tier 7 재실행 (y_pred 저장)."""
    print("\n" + "=" * 60)
    print("  STEP 0: Tier 7 re-run (saving predictions)")
    print("=" * 60)
    from tier7_experiment import run_experiment
    run_experiment(groups_filter=['15min'])
    run_experiment(groups_filter=['5min'])


def run_step1():
    """단계 1: 임상 안전성 분석."""
    print("\n" + "=" * 60)
    print("  STEP 1: Clinical Safety Analysis")
    print("=" * 60)
    from tier71_clinical_safety import run_clinical_analysis
    run_clinical_analysis()


def run_step2():
    """단계 2: 동일 질병 전이."""
    print("\n" + "=" * 60)
    print("  STEP 2: Same-Disease Transfer (T2D -> T2D)")
    print("=" * 60)
    from tier71_same_disease import run_same_disease_experiment
    run_same_disease_experiment()


def run_step3():
    """단계 3: N=1 Cold Start."""
    print("\n" + "=" * 60)
    print("  STEP 3: N=1 Cold Start Personalization")
    print("=" * 60)
    from tier71_cold_start import run_all_cold_start
    run_all_cold_start()


STEPS = {
    0: ('Tier 7 re-run (y_pred save)', run_step0),
    1: ('Clinical Safety Analysis',     run_step1),
    2: ('Same-Disease Transfer',         run_step2),
    3: ('N=1 Cold Start',                run_step3),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tier 7.1: Run all experiments sequentially')
    parser.add_argument('--steps', nargs='+', type=int,
                        choices=[0, 1, 2, 3], default=None,
                        help='실행할 단계 (기본: 전체 0-3)')
    parser.add_argument('--skip-step0', action='store_true',
                        help='단계 0 건너뛰기 (이미 y_pred 저장 완료 시)')
    args = parser.parse_args()

    steps_to_run = args.steps or [0, 1, 2, 3]
    if args.skip_step0 and 0 in steps_to_run:
        steps_to_run.remove(0)

    total_start = time.time()

    for step_num in sorted(steps_to_run):
        name, func = STEPS[step_num]
        step_start = time.time()
        func()
        elapsed = time.time() - step_start
        print(f"\n  Step {step_num} ({name}) completed in {elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  ALL DONE in {total_elapsed/60:.1f} min")
    print(f"{'=' * 60}")
