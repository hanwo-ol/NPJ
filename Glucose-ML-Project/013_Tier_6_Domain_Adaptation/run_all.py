"""
Tier 6: Orchestrator
====================
Runs all sequential ablation modes (A, B, C, D) without manual intervention.
"""

import subprocess
import sys
from pathlib import Path

def run_all_ablations():
    base_dir = Path(__file__).parent
    main_script = base_dir / "main.py"
    
    modes = ['A', 'B', 'C', 'D']
    
    print("=" * 60)
    print("Starting Complete Tier 6 Ablation Pipeline")
    print("=" * 60)
    
    for mode in modes:
        print(f"\n[{mode}] Executing Ablation Mode {mode} ...")
        
        try:
            # We use check=True to raise an error if a subprocess fails
            subprocess.run([sys.executable, str(main_script), "--ablation_mode", mode], check=True)
            print(f"[{mode}] Successfully completed Mode {mode}")
        except subprocess.CalledProcessError as e:
            print(f"[{mode}] FAILED with exit code {e.returncode}")
            print("Halting orchestration sequence.")
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print("All Tier 6 Ablation Experiments Finished Successfully!")
    print("Results are available in the 'tier6_results' directory.")
    print("=" * 60)

if __name__ == "__main__":
    run_all_ablations()
