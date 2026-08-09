import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Import utilities
from Utilities.config import (
    NON_WINDOWED_DATA_PATH, 
    RESULTS_BASE_PATH, 
    RANDOM_SEED, 
    RUL_BINS,
    ENGINE_COUNTS_ALL
)
from Utilities.Plots_Metrics import (
    rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc,
    plot_actual_vs_predicted, plot_rmse_by_bins
)

# Import models
from RandomForest.rf_model import train_model as train_rf, save_model as save_rf, MODEL_INFO as RF_INFO
from XGBoost.xgb_model import train_model as train_xgb, save_model as save_xgb, MODEL_INFO as XGB_INFO
from LightGBM.lgbm_model import train_model as train_lgbm, save_model as save_lgbm, MODEL_INFO as LGBM_INFO

# ==========
# CONFIGURATION
# ==========

# Models to run
MODELS = {
    'RF': {
        'info': RF_INFO,
        'train_func': train_rf,
        'save_func': save_rf
    },
    'XGB': {
        'info': XGB_INFO,
        'train_func': train_xgb,
        'save_func': save_xgb
    },
    'LGBM': {
        'info': LGBM_INFO,
        'train_func': train_lgbm,
        'save_func': save_lgbm
    }
}

# Which engine counts to generate plots for (to avoid 100+ plots)
PLOT_ENGINE_COUNTS = [80, 40, 10, 5, 1]

# ==========
# HELPER FUNCTIONS
# ==========

def load_data():
    print("\n" + "="*40)
    print("LOADING DATA")
    print("="*40)
    
    data_path = Path(NON_WINDOWED_DATA_PATH)
    
    # Load features and IDs separately
    train_features = pd.read_csv(data_path / 'FD001_train_features.csv')
    train_ids = pd.read_csv(data_path / 'FD001_train_ids.csv')
    
    val_features = pd.read_csv(data_path / 'FD001_val_features.csv')
    val_ids = pd.read_csv(data_path / 'FD001_val_ids.csv')
    
    test_features = pd.read_csv(data_path / 'FD001_test_features.csv')
    test_ids = pd.read_csv(data_path / 'FD001_test_ids.csv')
    
    X_train_full = train_features.values
    y_train_full = train_ids['RUL'].values
    
    X_val = val_features.values
    y_val = val_ids['RUL'].values
    
    X_test = test_features.values
    y_test = test_ids['RUL'].values
    
    feature_names = list(train_features.columns)
    
    print(f"\n Data loaded successfully:")
    print(f"Full train: {len(X_train_full):,} samples, {len(train_ids['engine'].unique())} engines")
    print(f"Validation: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")
    print(f"Features: {len(feature_names)}")
    
    return {
        'X_train_full': X_train_full,
        'y_train_full': y_train_full,
        'train_ids': train_ids,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names
    }

def create_engine_subset(data, n_engines_target):
    unique_engines = data['train_ids']['engine'].unique()
    total_engines = len(unique_engines)
    
    # Handle case where target equals or exceeds total
    if n_engines_target >= total_engines:
        return data['X_train_full'], data['y_train_full'], total_engines
    
    # Sample exact number of engines
    np.random.seed(RANDOM_SEED)
    selected_engines = np.random.choice(unique_engines, size=n_engines_target, replace=False)
    selected_engines = np.sort(selected_engines)  # Sort for reproducibility
    
    # Get indices for selected engines
    mask = data['train_ids']['engine'].isin(selected_engines)
    X_subset = data['X_train_full'][mask]
    y_subset = data['y_train_full'][mask]
    
    n_samples = len(y_subset)
    print(f"Selected {n_engines_target} engines: {list(selected_engines)}  {n_samples:,} samples")
    
    return X_subset, y_subset, n_engines_target

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    
    metrics = {
        'rmse': rmse(y, y_pred),
        'mae': mae(y, y_pred),
        'r2': r2(y, y_pred),
        'cmapss_score': cmapss_score(y, y_pred),
        'auc_rmse': rmse_by_bins_with_auc(y, y_pred, RUL_BINS)[1]
    }
    
    return metrics, y_pred

def save_metrics(all_metrics, model_name, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(all_metrics)
    output_file = save_dir / f'{model_name}_metrics_summary.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Metrics saved: {output_file}")
    
    return output_file

def save_predictions(y_true, y_pred, n_engines, split_name, model_name, save_dir):
    save_dir = Path(save_dir) / 'predictions'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    output_file = save_dir / f'{split_name}_{n_engines}engines.csv'
    df.to_csv(output_file, index=False)
    
    return output_file

def generate_plots(y_true, y_pred, n_engines, split_name, model_name, save_dir):
    plots_dir = Path(save_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Actual vs Predicted
    # Function signature: plot_actual_vs_predicted(y_actual, y_pred, dataset_name, model_name, save_path, show)
    plot_actual_vs_predicted(
        y_true, y_pred,
        dataset_name=f'{split_name.upper()}, {n_engines} engines',
        model_name=model_name,
        save_path=plots_dir / f'actual_vs_pred_{split_name}_{n_engines}engines.png',
        show=False
    )
    
    # 2. RMSE by Bins
    # First calculate RMSE by bins
    bin_rmse_list, auc_rmse = rmse_by_bins_with_auc(y_true, y_pred, RUL_BINS)
    
    # Function signature: plot_rmse_by_bins(edges, rmse_bins, auc_rmse_norm, model_name, save_path, show)
    plot_rmse_by_bins(
        edges=RUL_BINS,
        rmse_bins=bin_rmse_list,
        auc_rmse_norm=auc_rmse,
        model_name=model_name,
        save_path=plots_dir / f'rmse_by_bins_{split_name}_{n_engines}engines.png',
        show=False
    )

def run_single_experiment(model_name, model_config, data, n_engines):
    model_info = model_config['info']
    train_func = model_config['train_func']
    save_func = model_config['save_func']
    
    print(f"\n  [ENGINES: {n_engines}]")
    
    # Create subset by engine
    X_train, y_train, n_engines_used = create_engine_subset(data, n_engines)
    
    print(f"Training samples: {len(X_train):,}")
    
    # Train model (will load tuned hyperparameters)
    start_time = time.time()
    model = train_func(X_train, y_train, use_tuned_params=True)
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} seconds")
    
    # Evaluate on all splits
    results = {}
    
    # Train
    metrics_train, preds_train = evaluate_model(model, X_train, y_train)
    print(f"Train RMSE: {metrics_train['rmse']:.4f}, R²: {metrics_train['r2']:.4f}")
    
    # Val
    metrics_val, preds_val = evaluate_model(model, data['X_val'], data['y_val'])
    print(f"Val   RMSE: {metrics_val['rmse']:.4f}, R²: {metrics_val['r2']:.4f}")
    
    # Test
    metrics_test, preds_test = evaluate_model(model, data['X_test'], data['y_test'])
    print(f"Test  RMSE: {metrics_test['rmse']:.4f}, R²: {metrics_test['r2']:.4f}")
    
    # Save results
    results_dir = Path(RESULTS_BASE_PATH) / 'Phase1_Baseline' / 'FD001' / model_name
    
    # Save metrics
    metrics_list = []
    for split_name, metrics in [('train', metrics_train), ('val', metrics_val), ('test', metrics_test)]:
        metrics_list.append({
            'model_name': model_name,
            'n_engines': n_engines_used,
            'n_train_samples': len(X_train) if split_name == 'train' else None,
            'split': split_name,
            'training_time_sec': training_time if split_name == 'train' else None,
            **metrics
        })
    
    # Save predictions
    save_predictions(y_train, preds_train, n_engines, 'train', model_name, results_dir)
    save_predictions(data['y_val'], preds_val, n_engines, 'val', model_name, results_dir)
    save_predictions(data['y_test'], preds_test, n_engines, 'test', model_name, results_dir)
    
    # Generate plots for selected engine counts
    if n_engines in PLOT_ENGINE_COUNTS:
        print(f"Generating plots...")
        generate_plots(data['y_test'], preds_test, n_engines, 'test', model_name, results_dir)
    
    # Save model
    model_save_path = results_dir / 'models' / f'model_{n_engines}engines.pkl'
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    save_func(model, model_save_path)
    
    return metrics_list

def main(models_to_run=None, engine_counts_to_run=None):
    print("\n" + "="*40)
    print("TREE-BASED MODELS - BASELINE EXPERIMENTS (ENGINE-BASED)")
    print("="*40)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine which models and engine counts to run
    if models_to_run is None:
        models_to_run = list(MODELS.keys())
    if engine_counts_to_run is None:
        engine_counts_to_run = ENGINE_COUNTS_ALL
    
    print(f"\nModels to run: {', '.join(models_to_run)}")
    print(f"Engine counts: {engine_counts_to_run}")
    
    # Load data
    data = load_data()
    
    # Store all results
    all_results = []
    
    # Loop through each model
    for model_key in models_to_run:
        if model_key not in MODELS:
            print(f"\n WARNING: Model '{model_key}' not found. Skipping.")
            continue
        
        model_config = MODELS[model_key]
        model_info = model_config['info']
        
        print(f"\n{'='*40}")
        print(f"MODEL: {model_info['name']} ({model_key})")
        print(f"{'='*40}")
        
        model_results = []
        
        # Run experiments for each engine count
        for n_engines in engine_counts_to_run:
            try:
                metrics_list = run_single_experiment(model_key, model_config, data, n_engines)
                model_results.extend(metrics_list)
                all_results.extend(metrics_list)
            except Exception as e:
                print(f"\n ERROR at {n_engines} engines: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save model-specific summary
        results_dir = Path(RESULTS_BASE_PATH) / 'Phase1_Baseline' / 'FD001' / model_key
        save_metrics(model_results, model_key, results_dir)
        
        print(f"\n {model_key} complete!")
    
    # Save combined results
    if all_results:
        combined_dir = Path(RESULTS_BASE_PATH) / 'Phase1_Baseline' / 'FD001'
        df_all = pd.DataFrame(all_results)
        output_file = combined_dir / 'tree_all_results.csv'
        df_all.to_csv(output_file, index=False)
        print(f"\n Combined results saved: {output_file}")
        
        # Print summary
        print("\n" + "="*40)
        print("EXPERIMENT SUMMARY - TEST SET PERFORMANCE")
        print("="*40)
        
        df_test = df_all[df_all['split'] == 'test'].copy()
        summary = df_test.groupby('model_name').agg({
            'rmse': 'mean',
            'r2': 'mean',
            'cmapss_score': 'mean'
        }).round(4)
        print(summary)
    
    print("\n" + "="*40)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*40)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {RESULTS_BASE_PATH}/Phase1_Baseline/FD001/")
    print()

if __name__ == "__main__":
    # Parse command line arguments (optional, for direct running)
    parser = argparse.ArgumentParser(description='Run tree-based model baseline experiments')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()), 
                       default=None, help='Models to run (default: all)')
    parser.add_argument('--engines', nargs='+', type=int,
                       default=None, help='Engine counts to run (default: all)')
    
    args = parser.parse_args()
    
    main(models_to_run=args.models, engine_counts_to_run=args.engines)

