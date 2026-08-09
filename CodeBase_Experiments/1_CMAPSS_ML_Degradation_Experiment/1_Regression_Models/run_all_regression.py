import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Import utilities
from Utilities.config import (
    NON_WINDOWED_DATA_PATH, 
    RESULTS_BASE_PATH, 
    RANDOM_SEED, 
    RUL_BINS,
    ENGINE_COUNTS_ALL,  # Use engine counts instead of percentages
    RIDGE_ALPHA_GRID,
    LASSO_ALPHA_GRID,
    ELASTICNET_ALPHA_GRID,
    ELASTICNET_L1_RATIO_GRID
)
from Utilities.Plots_Metrics import (
    rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc,
    plot_actual_vs_predicted, plot_rmse_by_bins
)
import matplotlib.pyplot as plt

# Import models
from MLR.mlr_model import get_model as get_mlr_model, MODEL_INFO as MLR_INFO
from Ridge.ridge_model import get_model as get_ridge_model, MODEL_INFO as RIDGE_INFO
from Lasso.lasso_model import get_model as get_lasso_model, MODEL_INFO as LASSO_INFO
from ElasticNet.elasticnet_model import get_model as get_elasticnet_model, MODEL_INFO as ELASTICNET_INFO
from Polynomial_Deg2.poly2_model import get_model as get_poly2_model, MODEL_INFO as POLY2_INFO
from Polynomial_Deg3.poly3_model import get_model as get_poly3_model, MODEL_INFO as POLY3_INFO

# ==========
# CONFIGURATION
# ==========

# Models to run (add more as they're implemented)
MODELS = {
    'MLR': {
        'info': MLR_INFO,
        'get_model': get_mlr_model,
        'param_grid': None  # No hyperparameters to tune
    },
    'Ridge': {
        'info': RIDGE_INFO,
        'get_model': get_ridge_model,
        'param_grid': {'alpha': RIDGE_ALPHA_GRID}  # Will tune on 100% data
    },
    'Lasso': {
        'info': LASSO_INFO,
        'get_model': get_lasso_model,
        'param_grid': {'alpha': LASSO_ALPHA_GRID}  # Will tune on 100% data
    },
    'ElasticNet': {
        'info': ELASTICNET_INFO,
        'get_model': get_elasticnet_model,
        'param_grid': {
            'alpha': ELASTICNET_ALPHA_GRID,
            'l1_ratio': ELASTICNET_L1_RATIO_GRID
        }  # Will tune on 100% data (2D grid search)
    },
    'Poly2': {
        'info': POLY2_INFO,
        'get_model': get_poly2_model,
        'param_grid': None  # No hyperparameters to tune (pipeline handles feature generation)
    },
    'Poly3': {
        'info': POLY3_INFO,
        'get_model': get_poly3_model,
        'param_grid': None  # No hyperparameters to tune (pipeline handles feature generation)
    }
}

# Which engine counts to generate plots for (to avoid 100+ plots)
PLOT_ENGINE_COUNTS = [80, 40, 10, 5, 1]

# ==========
# HELPER FUNCTIONS
# ==========

def load_data():
    print("\n[1/5] Loading FD001 data...")
    
    train_features = pd.read_csv(os.path.join(NON_WINDOWED_DATA_PATH, "FD001_train_features.csv"))
    train_ids = pd.read_csv(os.path.join(NON_WINDOWED_DATA_PATH, "FD001_train_ids.csv"))
    
    val_features = pd.read_csv(os.path.join(NON_WINDOWED_DATA_PATH, "FD001_val_features.csv"))
    val_ids = pd.read_csv(os.path.join(NON_WINDOWED_DATA_PATH, "FD001_val_ids.csv"))
    
    test_features = pd.read_csv(os.path.join(NON_WINDOWED_DATA_PATH, "FD001_test_features.csv"))
    test_ids = pd.read_csv(os.path.join(NON_WINDOWED_DATA_PATH, "FD001_test_ids.csv"))
    
    data = {
        'X_train_full': train_features.values,
        'y_train_full': train_ids['RUL'].values,
        'train_ids': train_ids,
        'X_val': val_features.values,
        'y_val': val_ids['RUL'].values,
        'val_ids': val_ids,
        'X_test': test_features.values,
        'y_test': test_ids['RUL'].values,
        'test_ids': test_ids,
        'n_features': train_features.shape[1]
    }
    
    print(f"Train: {data['X_train_full'].shape[0]} samples, {data['n_features']} features")
    print(f"Val:   {len(data['y_val'])} samples")
    print(f"Test:  {len(data['y_test'])} samples")
    
    return data

def subset_training_data(data, n_engines_target):
    unique_engines = data['train_ids']['engine'].unique()
    total_engines = len(unique_engines)
    
    # Handle case where target equals total
    if n_engines_target >= total_engines:
        return data['X_train_full'], data['y_train_full'], total_engines
    
    # Sample exact number of engines
    np.random.seed(RANDOM_SEED)
    selected_engines = np.random.choice(unique_engines, size=n_engines_target, replace=False)
    selected_engines = np.sort(selected_engines)  # Sort for reproducibility
    
    mask = data['train_ids']['engine'].isin(selected_engines)
    X_train = data['X_train_full'][mask]
    y_train = data['y_train_full'][mask]
    
    n_samples = len(y_train)
    print(f"Selected {n_engines_target} engines: {list(selected_engines)}  {n_samples} samples")
    
    return X_train, y_train, n_engines_target

def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    
    metrics = {
        'dataset': dataset_name,
        'rmse': rmse(y, y_pred),
        'mae': mae(y, y_pred),
        'r2': r2(y, y_pred),
        'cmapss_score': cmapss_score(y, y_pred),
    }
    
    # Calculate RMSE by bins and AUC-RMSE
    rmse_bins, auc_rmse = rmse_by_bins_with_auc(y, y_pred, RUL_BINS)
    metrics['auc_rmse'] = auc_rmse
    metrics['rmse_bins'] = rmse_bins
    
    return metrics, y_pred

def save_plots(y_true, y_pred, model_key, model_name, n_engines, plots_dir):
    # Plot 1: Actual vs Predicted
    plot_path = os.path.join(plots_dir, f"actual_vs_pred_test_{n_engines}engines.png")
    plot_actual_vs_predicted(
        y_true, y_pred,
        dataset_name="FD001",
        model_name=f"{model_name} ({n_engines} engines)",
        save_path=plot_path,
        show=False
    )
    
    # Plot 2: RMSE by bins
    rmse_bins, auc_rmse = rmse_by_bins_with_auc(y_true, y_pred, RUL_BINS)
    plot_path_bins = os.path.join(plots_dir, f"rmse_by_bins_test_{n_engines}engines.png")
    plot_rmse_by_bins(
        edges=RUL_BINS,
        rmse_bins=rmse_bins,
        auc_rmse_norm=auc_rmse,
        model_name=f"{model_name} ({n_engines} engines)",
        save_path=plot_path_bins,
        show=False
    )

# ==========
# MAIN EXPERIMENT LOOP
# ==========

def main():
    
    print("="*40)
    print("REGRESSION MODELS - BASELINE EXPERIMENTS (ENGINE-BASED)")
    print("="*40)
    print(f"Models to run: {', '.join([MODELS[k]['info']['name'] for k in MODELS.keys()])}")
    print(f"Engine counts: {ENGINE_COUNTS_ALL}")
    print(f"Total experiments: {len(MODELS)} models × {len(ENGINE_COUNTS_ALL)} engine counts = {len(MODELS) * len(ENGINE_COUNTS_ALL)}")
    print("="*40)
    
    # Load data once
    data = load_data()
    
    # Store all results
    all_results = []
    
    # Loop through each model
    for model_key, model_config in MODELS.items():
        model_info = model_config['info']
        get_model_func = model_config['get_model']
        param_grid = model_config.get('param_grid', None)
        
        print(f"\n{'='*40}")
        print(f"MODEL: {model_info['name']} ({model_key})")
        print(f"{'='*40}")
        
        # Step 1: Tune hyperparameters on 100% data (if needed)
        best_params = {}
        if param_grid is not None:
            print(f"\n[TUNING] Tuning hyperparameters using validation set...")
            
            # Get all combinations of hyperparameters
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # Track results
            tuning_results = []
            best_val_rmse = float('inf')
            best_params = None
            
            # Try each hyperparameter combination
            import itertools
            for param_combo in itertools.product(*param_values):
                params = dict(zip(param_names, param_combo))
                
                # Train on 100% training data
                model = get_model_func(**params)
                model.fit(data['X_train_full'], data['y_train_full'])
                
                # Evaluate on train and val
                train_rmse = rmse(data['y_train_full'], model.predict(data['X_train_full']))
                val_rmse = rmse(data['y_val'], model.predict(data['X_val']))
                
                tuning_results.append({
                    **params,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse
                })
                
                # Track best
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_params = params
            
            print(f"Best params: {best_params}")
            print(f"Best val RMSE: {best_val_rmse:.4f}")
            
            # Save tuning plots
            if len(param_names) == 1:  # Single hyperparameter (like alpha)
                param_name = param_names[0]
                param_vals = [r[param_name] for r in tuning_results]
                train_rmses = [r['train_rmse'] for r in tuning_results]
                val_rmses = [r['val_rmse'] for r in tuning_results]
                
                plt.figure(figsize=(10, 6))
                plt.plot(param_vals, train_rmses, 'o-', label='Train RMSE', linewidth=2)
                plt.plot(param_vals, val_rmses, 's-', label='Val RMSE', linewidth=2)
                plt.axvline(best_params[param_name], color='red', linestyle='--', 
                           label=f'Best {param_name}={best_params[param_name]}')
                plt.xlabel(f'{param_name}', fontsize=15)
                plt.ylabel('RMSE', fontsize=15)
                # Title intentionally omitted for thesis captioning.
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.legend(fontsize=17)
                plt.grid(True, alpha=0.3)
                plt.xscale('log')  # Log scale for alpha values
                if model_key in ['Ridge', 'Lasso']:
                    plt.ylim(20, 45)
                plt.tight_layout()
                
                # Save plot
                tuning_plot_path = os.path.join(RESULTS_BASE_PATH, "Phase1_Baseline", "FD001", model_key, 
                                                f"{model_key}_hp_tuning_curve.png")
                os.makedirs(os.path.dirname(tuning_plot_path), exist_ok=True)
                plt.savefig(tuning_plot_path, dpi=300)
                plt.close()
                print(f"Saved tuning curve: {tuning_plot_path}")
            
            elif len(param_names) == 2:  # Two hyperparameters (like ElasticNet)
                param_x, param_y = param_names[0], param_names[1]
                tuning_df = pd.DataFrame(tuning_results)
                
                x_vals = tuning_df[param_x].values
                y_vals = tuning_df[param_y].values
                z_vals = tuning_df['val_rmse'].values
                
                # Use log10 transform for alpha-like parameters so spacing is readable in 3D
                x_vals_plot = np.log10(x_vals) if param_x.lower() == 'alpha' else x_vals
                best_x_plot = np.log10(best_params[param_x]) if param_x.lower() == 'alpha' else best_params[param_x]
                
                fig = plt.figure(figsize=(11, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Surface-like triangulation across all grid points
                surf = ax.plot_trisurf(
                    x_vals_plot,
                    y_vals,
                    z_vals,
                    cmap='viridis',
                    alpha=0.9,
                    linewidth=0.2,
                    antialiased=True
                )
                
                # Overlay sampled points
                ax.scatter(x_vals_plot, y_vals, z_vals, color='black', s=18, alpha=0.8)
                
                # Highlight best point
                ax.scatter(
                    best_x_plot,
                    best_params[param_y],
                    best_val_rmse,
                    color='red',
                    s=90,
                    marker='*',
                    label='Best'
                )
                
                # Axis labeling
                if param_x.lower() == 'alpha':
                    unique_x = np.array(sorted(np.unique(x_vals)))
                    ax.set_xticks(np.log10(unique_x))
                    ax.set_xticklabels([f'{v:g}' for v in unique_x], rotation=20, ha='right', fontsize=15)
                    ax.set_xlabel(f'{param_x} (log10 scale)', labelpad=12, fontsize=15)
                else:
                    ax.set_xlabel(param_x, labelpad=12, fontsize=15)
                ax.set_ylabel(param_y, labelpad=10, fontsize=15)
                ax.set_zlabel('Validation RMSE', labelpad=10, fontsize=15)
                ax.tick_params(axis='x', labelsize=15)
                ax.tick_params(axis='y', labelsize=15)
                ax.tick_params(axis='z', labelsize=15)
                
                # Title intentionally omitted for thesis captioning.
                cbar = fig.colorbar(surf, ax=ax, shrink=0.65, aspect=15, pad=0.08)
                cbar.set_label('Validation RMSE', fontsize=15)
                cbar.ax.tick_params(labelsize=15)
                ax.view_init(elev=28, azim=-135)
                ax.legend(loc='upper left', fontsize=17)
                plt.tight_layout()
                
                # Save plot
                surface_path = os.path.join(
                    RESULTS_BASE_PATH,
                    "Phase1_Baseline",
                    "FD001",
                    model_key,
                    f"{model_key}_hp_tuning_3d.png"
                )
                os.makedirs(os.path.dirname(surface_path), exist_ok=True)
                plt.savefig(surface_path, dpi=300)
                plt.close()
                print(f"Saved tuning 3D plot: {surface_path}")
        
        # Create results directory for this model
        model_results_dir = os.path.join(RESULTS_BASE_PATH, "Phase1_Baseline", "FD001", model_key)
        plots_dir = os.path.join(model_results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        print(f"\n[2/5] Running experiments for {model_key}...")
        
        # Loop through each engine count
        for n_engines in ENGINE_COUNTS_ALL:
            print(f"\n    Testing with {n_engines} engines...")
            
            # Get subset of training data
            X_train, y_train, n_engines_used = subset_training_data(data, n_engines)
            print(f"Using {n_engines_used} engines, {len(y_train)} samples")
            
            # Train model with best params (or default if no tuning)
            start_time = time.time()
            if best_params:
                model = get_model_func(**best_params)  # Use tuned params
            else:
                model = get_model_func()  # Use defaults
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate on all datasets
            for dataset_name, X_eval, y_eval in [
                ('train', X_train, y_train),
                ('val', data['X_val'], data['y_val']),
                ('test', data['X_test'], data['y_test'])
            ]:
                metrics, y_pred = evaluate_model(model, X_eval, y_eval, dataset_name)
                
                # Store results
                result = {
                    'phase': 'Phase1_Baseline',
                    'model_name': model_key,
                    'model_full_name': model_info['name'],
                    'n_engines': n_engines_used,
                    'n_train_samples': len(y_train) if dataset_name == 'train' else 'N/A',
                    'n_samples': len(y_eval),
                    'training_time_sec': training_time if dataset_name == 'train' else 0.0,
                    **best_params,  # Include best hyperparameters in results
                    **metrics
                }
                all_results.append(result)
                
                print(f"{dataset_name.upper():5s}  RMSE: {metrics['rmse']:.2f}, "
                      f"R²: {metrics['r2']:.3f}, CMAPSS: {metrics['cmapss_score']:.1f}")
            
            # Generate plots for test set (only for selected engine counts)
            if n_engines in PLOT_ENGINE_COUNTS:
                print(f"Generating plots for {n_engines} engines...")
                y_test_pred = model.predict(data['X_test'])
                save_plots(data['y_test'], y_test_pred, model_key, model_info['name'], n_engines, plots_dir)
                print(f"Plots saved")
        
        # Save model-specific results
        print(f"\n[3/5] Saving results for {model_key}...")
        model_results = [r for r in all_results if r['model_name'] == model_key]
        results_df = pd.DataFrame(model_results)
        
        # Remove rmse_bins from CSV (too large, just for plotting)
        results_csv = results_df.drop(columns=['rmse_bins'], errors='ignore')
        results_path = os.path.join(model_results_dir, f"{model_key}_metrics_summary.csv")
        results_csv.to_csv(results_path, index=False)
        print(f"Saved: {results_path}")
    
    # Save combined results for all models
    print(f"\n[4/5] Saving combined results...")
    all_results_df = pd.DataFrame(all_results)
    all_results_csv = all_results_df.drop(columns=['rmse_bins'], errors='ignore')
    combined_path = os.path.join(RESULTS_BASE_PATH, "Phase1_Baseline", "FD001", "regression_all_results.csv")
    all_results_csv.to_csv(combined_path, index=False)
    print(f"Saved: {combined_path}")
    
    # Update experiment log
    print(f"\n[5/5] Updating experiment log...")
    update_log(all_results_df)
    
    # Print summary
    print("\n" + "="*40)
    print("REGRESSION EXPERIMENTS COMPLETE!")
    print("="*40)
    print(f"\nTotal experiments run: {len(all_results) // 3}")  # Divide by 3 (train/val/test)
    print(f"Results saved to: {RESULTS_BASE_PATH}/Phase1_Baseline/FD001/")
    print("\nSummary (Test Set Performance):")
    test_results = all_results_df[all_results_df['dataset'] == 'test']
    summary = test_results.groupby(['model_name', 'n_engines'])[['rmse', 'r2', 'cmapss_score']].mean()
    print(summary.to_string())
    print("="*40)

def update_log(results_df):
    log_entry = f"""
===============================================
PHASE 1 BASELINE: Regression Models Complete (ENGINE-BASED)
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
===============================================
Models: {', '.join(MODELS.keys())}
Dataset: FD001
Engine Counts: {', '.join(map(str, ENGINE_COUNTS_ALL))}
Total Experiments: {len(results_df) // 3} (train/val/test per engine count)

Test Set Performance Summary:
{results_df[results_df['dataset'] == 'test'].groupby('model_name')[['rmse', 'r2', 'cmapss_score']].mean().to_string()}

Status: SUCCESS
===============================================
"""
    
    log_path = os.path.join(project_root, "EXPERIMENT_LOG.txt")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)

# ==========
# ENTRY POINT
# ==========

if __name__ == "__main__":
    main()

