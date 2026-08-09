import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Import utilities
from Utilities import config
from Utilities import optuna_tuner

# Import model libraries
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

def load_data():
    print("\n" + "="*40)
    print("LOADING DATA")
    print("="*40)
    
    # Load data (features and IDs are in separate files)
    data_path = Path(config.NON_WINDOWED_DATA_PATH)
    
    train_features = pd.read_csv(data_path / 'FD001_train_features.csv')
    train_ids = pd.read_csv(data_path / 'FD001_train_ids.csv')
    
    val_features = pd.read_csv(data_path / 'FD001_val_features.csv')
    val_ids = pd.read_csv(data_path / 'FD001_val_ids.csv')
    
    X_train_full = train_features.values
    y_train_full = train_ids['RUL'].values
    
    X_val = val_features.values
    y_val = val_ids['RUL'].values
    
    feature_names = list(train_features.columns)
    
    print(f"\n Data loaded successfully:")
    print(f"Training samples: {len(X_train_full):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Features: {len(feature_names)}")
    print(f"Feature names: {feature_names[:5]}... (showing first 5)")
    
    return {
        'X_train_full': X_train_full,
        'y_train_full': y_train_full,
        'X_val': X_val,
        'y_val': y_val,
        'feature_names': feature_names
    }

def tune_random_forest(data, n_trials=None, n_jobs=None):
    print("\n" + "="*40)
    print("TUNING: RANDOM FOREST")
    print("="*40)
    
    # Fixed parameters from config (not tuned)
    fixed_params = config.RANDOM_FOREST_FIXED_PARAMS.copy()
    
    # Run tuning
    results = optuna_tuner.run_tuning(
        model_class=RandomForestRegressor,
        model_name='RF',
        search_space=config.RANDOM_FOREST_SEARCH_SPACE,
        X_train=data['X_train_full'],
        y_train=data['y_train_full'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        n_trials=n_trials,
        n_jobs=n_jobs,
        fixed_params=fixed_params
    )
    
    return results

def tune_xgboost(data, n_trials=None, n_jobs=None):
    print("\n" + "="*40)
    print("TUNING: XGBOOST")
    print("="*40)
    
    # Fixed parameters from config (not tuned)
    fixed_params = config.XGBOOST_FIXED_PARAMS.copy()
    
    # Run tuning
    results = optuna_tuner.run_tuning(
        model_class=xgb.XGBRegressor,
        model_name='XGB',
        search_space=config.XGBOOST_SEARCH_SPACE,
        X_train=data['X_train_full'],
        y_train=data['y_train_full'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        n_trials=n_trials,
        n_jobs=n_jobs,
        fixed_params=fixed_params
    )
    
    return results

def tune_lightgbm(data, n_trials=None, n_jobs=None):
    print("\n" + "="*40)
    print("TUNING: LIGHTGBM")
    print("="*40)
    
    # Fixed parameters from config (not tuned)
    fixed_params = config.LIGHTGBM_FIXED_PARAMS.copy()
    
    # Run tuning
    results = optuna_tuner.run_tuning(
        model_class=lgb.LGBMRegressor,
        model_name='LGBM',
        search_space=config.LIGHTGBM_SEARCH_SPACE,
        X_train=data['X_train_full'],
        y_train=data['y_train_full'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        n_trials=n_trials,
        n_jobs=n_jobs,
        fixed_params=fixed_params
    )
    
    return results

def save_summary(all_results, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary dataframe
    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append({
            'model': model_name,
            'best_val_rmse': results['best_value'],
            'n_trials': len(results['study'].trials),
            'best_params': str(results['best_params'])
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('best_val_rmse')
    
    # Save to CSV
    output_file = save_dir / 'tuning_summary.csv'
    df_summary.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print("TUNING SUMMARY")
    print("="*40)
    print(df_summary.to_string(index=False))
    print(f"\n Summary saved to: {output_file}")
    
    return df_summary

def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for tree-based models using Optuna'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['RF', 'XGB', 'LGBM', 'all'],
        default=['all'],
        help='Models to tune (default: all)'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=None,
        help=f'Number of Optuna trials (default: {config.OPTUNA_N_TRIALS})'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=None,
        help=f'Number of parallel jobs (default: {config.OPTUNA_N_JOBS})'
    )
    
    args = parser.parse_args()
    
    # Determine which models to tune
    if 'all' in args.models:
        models_to_tune = ['RF', 'XGB', 'LGBM']
    else:
        models_to_tune = args.models
    
    print("\n" + "="*40)
    print("TREE-BASED MODELS - HYPERPARAMETER TUNING")
    print("="*40)
    print(f"\nConfiguration:")
    print(f"Models: {', '.join(models_to_tune)}")
    print(f"Trials per model: {args.n_trials if args.n_trials else config.OPTUNA_N_TRIALS}")
    print(f"Parallel jobs: {args.n_jobs if args.n_jobs else config.OPTUNA_N_JOBS}")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data = load_data()
    
    # Dictionary to store results
    all_results = {}
    
    # Tune each model
    if 'RF' in models_to_tune:
        try:
            results = tune_random_forest(data, n_trials=args.n_trials, n_jobs=args.n_jobs)
            all_results['RF'] = results
        except Exception as e:
            print(f"\n ERROR tuning Random Forest: {e}")
            import traceback
            traceback.print_exc()
    
    if 'XGB' in models_to_tune:
        try:
            results = tune_xgboost(data, n_trials=args.n_trials, n_jobs=args.n_jobs)
            all_results['XGB'] = results
        except Exception as e:
            print(f"\n ERROR tuning XGBoost: {e}")
            import traceback
            traceback.print_exc()
    
    if 'LGBM' in models_to_tune:
        try:
            results = tune_lightgbm(data, n_trials=args.n_trials, n_jobs=args.n_jobs)
            all_results['LGBM'] = results
        except Exception as e:
            print(f"\n ERROR tuning LightGBM: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    if all_results:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning'
        save_summary(all_results, save_dir)
    
    # Final message
    print("\n" + "="*40)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*40)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {config.RESULTS_BASE_PATH}/Hyperparameter_Tuning/")
    print(f"\nNext steps:")
    print(f"1. Review tuning results and plots")
    print(f"2. Check best_params_*.json files")
    print(f"3. Run baseline experiments with optimized hyperparameters")
    print(f"4. Use: python run_all_tree.py --models all --percentages all")
    print()

if __name__ == "__main__":
    main()

