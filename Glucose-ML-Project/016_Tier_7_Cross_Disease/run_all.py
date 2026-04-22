"""
Tier 7: Run All
================
실행 순서: 데이터 로딩 → 5-way 실험 → SHAP 분석
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tier7_experiment import run_experiment
from tier7_shap_analysis import run_shap_analysis

if __name__ == '__main__':
    print("╔══════════════════════════════════════╗")
    print("║   Tier 7: Cross-Disease Transfer     ║")
    print("║   T1D (Source) → T2D (Target)        ║")
    print("╚══════════════════════════════════════╝\n")

    df = run_experiment()
    run_shap_analysis()

    print("\n[DONE] All Tier 7 outputs in: 016_Tier_7_Cross_Disease/tier7_results/")
