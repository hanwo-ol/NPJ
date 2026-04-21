"""
Tier 6: Main Entry Point
========================
Argparse and Execution Controller ONLY.
Adheres to AGENTS.md coding rules.
"""

import argparse
from tier6_trainer import execute_ablation

def main():
    parser = argparse.ArgumentParser(description="Tier 6: Domain Adaptation & Feature Engineering Ablation Experiment")
    
    # Configure Ablation Component Switches instead of deleting old features
    parser.add_argument('--ablation_mode', type=str, choices=['A', 'B', 'C', 'D'], required=True,
                        help='A: eGMI Imputation, B: UMD Virtual Marker, C: CORAL Alignment, D: TCA Alignment')
    
    parser.add_argument('--full_pipeline', action='store_true',
                        help='Run the final synergetic pipeline blending all components.')

    args = parser.parse_args()

    # Formulate Config Vector
    config = {
        'ablation_mode': args.ablation_mode,
        'full_pipeline': args.full_pipeline
    }

    # Execute
    if config['full_pipeline']:
        print("Final Combinatorial pipeline not yet fully merged. Please run individual mode.")
    else:
        execute_ablation(config)

if __name__ == "__main__":
    main()
