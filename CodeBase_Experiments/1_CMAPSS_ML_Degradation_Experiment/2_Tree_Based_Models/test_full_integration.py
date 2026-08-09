import sys
import os
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("\n" + "="*40)
print("FULL INTEGRATION TEST - TREE-BASED MODELS")
print("="*40)
print("\nThis test runs the COMPLETE pipeline with reduced scope:")
print("- Tuning: 1 model (RF) × 5 trials (instead of 3 × 50)")
print("- Baseline: 1 model (RF) × 3 percentages (instead of 3 × 10)")
print("- Generates ALL output files (plots, CSVs, models, etc.)")
print("\nExpected runtime: 10-15 minutes")
print("\nPurpose: Verify NOTHING is missing before overnight run")
print("="*40)

# Get user confirmation
response = input("\nContinue with full integration test? (y/n): ")
if response.lower() != 'y':
    print("Test cancelled.")
    sys.exit(0)

# Create test results directory
test_results_dir = Path(project_root) / 'Results' / 'Test_Run'
if test_results_dir.exists():
    print(f"\n Test results directory exists: {test_results_dir}")
    response = input("Delete and recreate? (y/n): ")
    if response.lower() == 'y':
        shutil.rmtree(test_results_dir)
        print("Deleted old test results")

print("\n" + "="*40)
print("STEP 1: HYPERPARAMETER TUNING (5 trials)")
print("="*40)

# Temporarily modify results path for test
from Utilities import config
original_results_path = config.RESULTS_BASE_PATH
config.RESULTS_BASE_PATH = str(test_results_dir)

try:
    # Run tuning
    sys.path.insert(0, str(Path(__file__).parent))
    from tune_tree_models import tune_random_forest, load_data as load_tuning_data
    
    data = load_tuning_data()
    
    print("\n[TUNING] Random Forest with 5 trials...")
    tuning_results = tune_random_forest(data, n_trials=5, n_jobs=1)
    
    print("\n Tuning complete!")
    print(f"Best val RMSE: {tuning_results['best_value']:.4f}")
    print(f"Best params: {tuning_results['best_params']}")
    
    # List generated files
    tuning_dir = test_results_dir / 'Hyperparameter_Tuning' / 'RF'
    print(f"\n Tuning outputs generated in: {tuning_dir}")
    for file in sorted(tuning_dir.rglob('*')):
        if file.is_file():
            print(f"- {file.relative_to(tuning_dir)}")
    
    # Expected files
    expected_tuning_files = [
        'best_params_RF.json',
        'optuna_trials_RF.csv',
        'optimization_history_RF.png',
        'param_importances_RF.png',
        'param_relationships_RF.png'
    ]
    
    print("\n Checking expected files:")
    for fname in expected_tuning_files:
        fpath = tuning_dir / fname
        if fpath.exists():
            print(f"{fname}")
        else:
            print(f"{fname} MISSING!")
    
    print("\n" + "="*40)
    print("STEP 2: BASELINE EXPERIMENTS (RF × 3 percentages)")
    print("="*40)
    
    # Run baseline experiments
    from run_all_tree import main as run_baseline
    
    print("\n[BASELINE] Running RF on 100%, 50%, 10%...")
    run_baseline(models_to_run=['RF'], percentages_to_run=[100, 50, 10])
    
    print("\n Baseline experiments complete!")
    
    # List generated files
    baseline_dir = test_results_dir / 'Phase1_Baseline' / 'FD001' / 'RF'
    print(f"\n Baseline outputs generated in: {baseline_dir}")
    
    # Show directory structure
    print("\nDirectory structure:")
    for subdir in ['predictions', 'plots', 'models']:
        subpath = baseline_dir / subdir
        if subpath.exists():
            files = list(subpath.iterdir())
            print(f"{subdir}/ ({len(files)} files)")
            for f in sorted(files)[:5]:  # Show first 5
                print(f"- {f.name}")
            if len(files) > 5:
                print(f"... and {len(files)-5} more")
    
    # Check metrics file
    metrics_file = baseline_dir / 'RF_metrics_summary.csv'
    if metrics_file.exists():
        print(f"\n Metrics file: {metrics_file}")
        import pandas as pd
        df = pd.read_csv(metrics_file)
        print(f"Rows: {len(df)} (expected: 9 for 3 percentages × 3 splits)")
        print(f"Columns: {list(df.columns)}")
    
    # Expected baseline files
    print("\n Checking expected outputs:")
    
    expected_checks = {
        'Metrics CSV': baseline_dir / 'RF_metrics_summary.csv',
        'Combined results': test_results_dir / 'Phase1_Baseline' / 'FD001' / 'tree_all_results.csv',
    }
    
    for name, fpath in expected_checks.items():
        if fpath.exists():
            print(f"{name}: {fpath.name}")
        else:
            print(f"{name} MISSING: {fpath}")
    
    # Check predictions (should be 9: train/val/test × 3 percentages)
    pred_dir = baseline_dir / 'predictions'
    if pred_dir.exists():
        pred_files = list(pred_dir.glob('*.csv'))
        print(f"Predictions: {len(pred_files)} files (expected: 9)")
        if len(pred_files) != 9:
            print(f"Expected 9, got {len(pred_files)}")
    
    # Check plots (should be 6: 2 plot types × 3 percentages)
    plot_dir = baseline_dir / 'plots'
    if plot_dir.exists():
        plot_files = list(plot_dir.glob('*.png'))
        print(f"Plots: {len(plot_files)} files (expected: 6)")
        if len(plot_files) != 6:
            print(f"Expected 6, got {len(plot_files)}")
    
    # Check models (should be 3: 1 per percentage)
    model_dir = baseline_dir / 'models'
    if model_dir.exists():
        model_files = list(model_dir.glob('*.pkl'))
        print(f"Models: {len(model_files)} files (expected: 3)")
        if len(model_files) != 3:
            print(f"Expected 3, got {len(model_files)}")
    
    print("\n" + "="*40)
    print("INTEGRATION TEST COMPLETE!")
    print("="*40)
    
    print(f"\n All test outputs saved to: {test_results_dir}")
    print("\n Next steps:")
    print("1. Review test outputs to verify everything looks correct")
    print("2. Check plots visually")
    print("3. Check metrics CSVs for reasonable values")
    print("4. If everything looks good  run full experiments:")
    print("- Tuning: python tune_tree_models.py --models all --n_trials 50")
    print("- Baseline: python run_all_tree.py")
    
    print("\n To delete test results:")
    print(f"rm -rf {test_results_dir}")  # or on Windows: rmdir /s /q
    
    print("\n" + "="*40)
    print("FILE CHECKLIST")
    print("="*40)
    
    all_files = []
    for root, dirs, files in os.walk(test_results_dir):
        for file in files:
            fpath = Path(root) / file
            rel_path = fpath.relative_to(test_results_dir)
            all_files.append(str(rel_path))
    
    print(f"\nTotal files generated: {len(all_files)}")
    print("\nAll files:")
    for fpath in sorted(all_files):
        print(f"{fpath}")
    
    print("\n If you see all these file types, the pipeline is working correctly!")
    print("="*40)

finally:
    # Restore original results path
    config.RESULTS_BASE_PATH = original_results_path

