import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Import config
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from Utilities import config

def suggest_hyperparameter(trial, param_name, param_config):
    param_type = param_config[0]
    
    if param_type == 'int':
        low, high = param_config[1], param_config[2]
        return trial.suggest_int(param_name, low, high)
    
    elif param_type == 'float':
        low, high = param_config[1], param_config[2]
        return trial.suggest_float(param_name, low, high)
    
    elif param_type == 'float_log':
        low, high = param_config[1], param_config[2]
        return trial.suggest_float(param_name, low, high, log=True)
    
    elif param_type == 'categorical':
        choices = param_config[1]
        return trial.suggest_categorical(param_name, choices)
    
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")

def create_study(study_name, direction='minimize', sampler=None, pruner=None, storage=None):
    if sampler is None:
        sampler = TPESampler(seed=config.RANDOM_SEED)
    
    if storage is None:
        storage = config.OPTUNA_STORAGE
    
    # Create or load study
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True
        )
    except:
        # If storage fails, create in-memory study
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
    
    return study

def build_tree_objective(model_class, search_space, X_train, y_train, X_val, y_val, 
                         fixed_params=None, fit_params=None):
    def objective(trial):
        # Suggest hyperparameters
        params = {}
        for param_name, param_config in search_space.items():
            params[param_name] = suggest_hyperparameter(trial, param_name, param_config)
        
        # Add fixed parameters
        if fixed_params is not None:
            params.update(fixed_params)
        
        # Create and train model
        model = model_class(**params)
        
        if fit_params is not None:
            model.fit(X_train, y_train, **fit_params)
        else:
            model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        return val_rmse
    
    return objective

def save_best_params(study, model_name, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_params = study.best_params
    best_value = study.best_value
    
    result = {
        'model_name': model_name,
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': len(study.trials)
    }
    
    output_file = save_dir / f'best_params_{model_name}.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Best parameters saved to: {output_file}")
    print(f"Best validation RMSE: {best_value:.4f}")
    print(f"Best params: {best_params}")
    
    return output_file

def save_study_results(study, model_name, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get trials dataframe
    df = study.trials_dataframe()
    
    # Save to CSV
    output_file = save_dir / f'optuna_trials_{model_name}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Study results saved to: {output_file}")
    print(f"Total trials: {len(df)}")
    
    return output_file

def plot_optimization_history(study, model_name, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract trial data
    trials = study.trials
    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials if t.value is not None]
    
    # Calculate best value so far
    best_values = []
    best_so_far = float('inf')
    for v in values:
        if v < best_so_far:
            best_so_far = v
        best_values.append(best_so_far)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all trials
    ax.scatter(trial_numbers[:len(values)], values, alpha=0.5, s=30, 
               label='Trial Value', color='lightblue')
    
    # Plot best value curve
    ax.plot(trial_numbers[:len(best_values)], best_values, 
            color='red', linewidth=2, label='Best Value')
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Validation RMSE', fontsize=12)
    ax.set_title(f'{model_name} - Optimization History', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_file = save_dir / f'optimization_history_{model_name}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Optimization history plot saved to: {output_file}")
    
    return output_file

def plot_param_importances(study, model_name, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get parameter importances
        importances = optuna.importance.get_param_importances(study)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = list(importances.keys())
        values = list(importances.values())
        
        # Sort by importance
        sorted_idx = np.argsort(values)
        params = [params[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        ax.barh(params, values, color='steelblue')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Hyperparameter', fontsize=12)
        ax.set_title(f'{model_name} - Hyperparameter Importances', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Save plot
        output_file = save_dir / f'param_importances_{model_name}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter importances plot saved to: {output_file}")
        
        return output_file
    
    except Exception as e:
        print(f"Could not generate parameter importances plot: {e}")
        return None

def plot_param_relationships(study, model_name, save_dir, max_params=4):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get trials dataframe
        df = study.trials_dataframe()
        
        # Get parameter columns (exclude system columns)
        param_cols = [col for col in df.columns if col.startswith('params_')]
        param_cols = param_cols[:max_params]  # Limit to max_params
        
        if len(param_cols) == 0:
            print("No parameters found to plot")
            return None
        
        # Create subplots
        n_params = len(param_cols)
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
        
        if n_params == 1:
            axes = [axes]
        
        for ax, param_col in zip(axes, param_cols):
            param_name = param_col.replace('params_', '')
            
            # Handle categorical parameters (convert None to string)
            x_data = df[param_col].copy()
            if x_data.dtype == 'object' or pd.api.types.is_categorical_dtype(x_data):
                # Convert None to 'None' string for plotting
                x_data = x_data.fillna('None').astype(str)
            
            # Plot scatter
            ax.scatter(x_data, df['value'], alpha=0.5, s=30)
            ax.set_xlabel(param_name, fontsize=10)
            ax.set_ylabel('Validation RMSE', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if categorical
            if x_data.dtype == 'object':
                ax.tick_params(axis='x', rotation=45)
        
        fig.suptitle(f'{model_name} - Hyperparameter vs. Objective', 
                     fontsize=14, fontweight='bold')
        
        # Save plot
        output_file = save_dir / f'param_relationships_{model_name}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter relationships plot saved to: {output_file}")
        
        return output_file
    
    except Exception as e:
        print(f"Could not generate parameter relationships plot: {e}")
        return None

def run_tuning(model_class, model_name, search_space, X_train, y_train, X_val, y_val,
               n_trials=None, n_jobs=None, fixed_params=None, fit_params=None,
               save_dir=None):
    print(f"\n{'='*40}")
    print(f"HYPERPARAMETER TUNING: {model_name}")
    print(f"{'='*40}")
    
    # Set defaults
    if n_trials is None:
        n_trials = config.OPTUNA_N_TRIALS
    if n_jobs is None:
        n_jobs = config.OPTUNA_N_JOBS
    if save_dir is None:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / model_name
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"Trials: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Search space: {len(search_space)} hyperparameters")
    print(f"Save directory: {save_dir}")
    
    # Create study
    study = create_study(
        study_name=f'{model_name}_tuning',
        direction='minimize',
        sampler=TPESampler(seed=config.RANDOM_SEED),
        pruner=None  # No pruning for tree models
    )
    
    # Build objective
    objective = build_tree_objective(
        model_class=model_class,
        search_space=search_space,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fixed_params=fixed_params,
        fit_params=fit_params
    )
    
    # Run optimization
    print(f"\n[TUNING] Starting optimization...")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    
    print(f"\n[COMPLETE] Optimization finished!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Save results
    print(f"\n[SAVING] Saving results...")
    best_params_file = save_best_params(study, model_name, save_dir)
    trials_file = save_study_results(study, model_name, save_dir)
    
    # Generate plots
    print(f"\n[PLOTTING] Generating plots...")
    history_plot = plot_optimization_history(study, model_name, save_dir)
    importance_plot = plot_param_importances(study, model_name, save_dir)
    relationships_plot = plot_param_relationships(study, model_name, save_dir)
    
    print(f"\n{'='*40}")
    print(f"TUNING COMPLETE: {model_name}")
    print(f"{'='*40}\n")
    
    return {
        'study': study,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'files': {
            'best_params': best_params_file,
            'trials': trials_file,
            'plots': {
                'history': history_plot,
                'importances': importance_plot,
                'relationships': relationships_plot
            }
        }
    }

def load_best_params(model_name, load_dir=None):
    if load_dir is None:
        load_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / model_name
    
    load_dir = Path(load_dir)
    param_file = load_dir / f'best_params_{model_name}.json'
    
    if not param_file.exists():
        raise FileNotFoundError(f"Best params file not found: {param_file}")
    
    with open(param_file, 'r') as f:
        result = json.load(f)
    
    return result['best_params']

