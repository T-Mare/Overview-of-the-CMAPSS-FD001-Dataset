import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import config to get available models and percentages
from Utilities import config

# Import and run the main orchestrator
tree_path = os.path.join(
    project_root,
    "CodeBase_Experiments",
    "1_CMAPSS_ML_Degradation_Experiment", 
    "2_Tree_Based_Models"
)
sys.path.insert(0, tree_path)

from run_all_tree import main, MODELS

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Tree-Based Baseline Experiments Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models with all data percentages:
  python run_baseline_tree.py --models all

  # Run only Random Forest:
  python run_baseline_tree.py --models RF

  # Run XGBoost and LightGBM:
  python run_baseline_tree.py --models XGB LGBM

  # Run specific percentages:
  python run_baseline_tree.py --models all --percentages 100 50 10

  # Full control:
  python run_baseline_tree.py --models RF XGB --percentages 100 90 80 70 60 50
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['RF', 'XGB', 'LGBM', 'all'],
        default=['all'],
        help='Models to run: RF (Random Forest), XGB (XGBoost), LGBM (LightGBM), or all (default: all)'
    )
    
    parser.add_argument(
        '--percentages',
        nargs='+',
        type=int,
        default=None,
        help=f'Data percentages to run (default: {config.DATA_PERCENTAGES})'
    )
    
    args = parser.parse_args()
    
    # Determine which models to run
    if 'all' in args.models:
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = args.models
    
    # Validate percentages
    if args.percentages is not None:
        for pct in args.percentages:
            if pct not in config.DATA_PERCENTAGES:
                print(f" WARNING: {pct}% not in configured percentages: {config.DATA_PERCENTAGES}")
    
    percentages_to_run = args.percentages
    
    # Print configuration
    print("\n" + "="*40)
    print("TREE-BASED BASELINE EXPERIMENTS - WRAPPER")
    print("="*40)
    print(f"\nOrchestrator: {tree_path}/run_all_tree.py")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Percentages: {percentages_to_run if percentages_to_run else 'all'}")
    print()
    
    # Run experiments
    main(models_to_run=models_to_run, percentages_to_run=percentages_to_run)

