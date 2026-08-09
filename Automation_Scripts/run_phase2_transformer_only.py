import pandas as pd
import numpy as np
import os
import sys
import json
import time
import pickle
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import utilities
from Utilities.config import RANDOM_SEED, ENGINE_COUNTS_ALL, RUL_BINS
from Utilities.Plots_Metrics import (
    rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc,
    plot_actual_vs_predicted, plot_rmse_by_bins
)

# Import Transformer model
sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "3_Deep_Learning_Models", "Transformer"))
from transformer_model import build_model as build_transformer_model, train_model as train_transformer_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========
# CONFIGURATION
# ==========

# Paths
DATA_BASE = Path(project_root) / "CodeBase_Experiments" / "0_Data_Processing" / "Data_CMAPSS" / "2_Cleaned_Data"
WINDOWED_DATA = DATA_BASE / "Windowed"

FEATURE_ANALYSIS = Path(project_root) / "Results" / "Phase2_Feature_Selection" / "FD001" / "Feature_Analysis"
OUTPUT_BASE = Path(project_root) / "Results" / "Phase2_Feature_Selection" / "FD001"

HYPERPARAM_DIR = Path(project_root) / "Results" / "Hyperparameter_Tuning"

# Create output directories
for fs_method in ['Correlation_FS', 'Tree_FS']:
    (OUTPUT_BASE / fs_method / "Transformer").mkdir(parents=True, exist_ok=True)

# Random seed
np.random.seed(RANDOM_SEED)

# Window size for Transformer
WINDOW_SIZE = 30

# Engine counts to run
# All engine counts from 80 down to 1 for complete FS analysis
ENGINE_COUNTS = [80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# ==========
# LOAD SELECTED FEATURES
# ==========

def load_selected_features():
    print("\nLoading selected features...")
    
    # Load from feature_selection_summary.json
    summary_file = FEATURE_ANALYSIS / "feature_selection_summary.json"
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    corr_features = data['correlation_based']['features']
    tree_features = data['tree_based']['features']
    
    print(f"Correlation-based: {len(corr_features)} features")
    print(f"Tree-based: {len(tree_features)} features")
    
    return corr_features, tree_features

# ==========
# LOAD HYPERPARAMETERS
# ==========

def load_hyperparameters():
    print("\nLoading hyperparameters from Phase 1...")
    
    hp_file = HYPERPARAM_DIR / "Transformer" / "Transformer_best_hyperparameters.json"
    
    if hp_file.exists():
        with open(hp_file, 'r') as f:
            hyperparams = json.load(f)
        print(f"Loaded Transformer hyperparameters")
        print(f"d_model={hyperparams.get('d_model')}, num_heads={hyperparams.get('num_heads')}, "
              f"ff_dim={hyperparams.get('ff_dim')}, blocks={hyperparams.get('num_transformer_blocks')}")
        return hyperparams
    else:
        print(f"Warning: Hyperparameters not found at {hp_file}, using defaults")
        return {}

# ==========
# DATA LOADING
# ==========

def load_windowed_data(n_engines_to_use, selected_features):
    
    # Load full training data
    X_train_full = np.load(WINDOWED_DATA / "FD001_X_train_windowed.npy")
    y_train_full = np.load(WINDOWED_DATA / "FD001_y_train_windowed.npy")
    
    # Load engine IDs to sample by engine
    train_ids = pd.read_csv(WINDOWED_DATA / "FD001_train_ids_windowed.csv")
    
    # Get unique engines
    unique_engines = train_ids['engine'].unique()
    n_total_engines = len(unique_engines)
    
    if n_engines_to_use == n_total_engines:
        # Use all data
        X_train = X_train_full
        y_train = y_train_full
    else:
        # Sample BY ENGINES (not random sequences)
        n_selected = min(n_engines_to_use, n_total_engines)
        
        np.random.seed(RANDOM_SEED)
        sampled_engines = np.random.choice(unique_engines, size=n_selected, replace=False)
        sampled_engines = sorted(sampled_engines)  # For reproducibility
        
        # Filter data for sampled engines
        mask = train_ids['engine'].isin(sampled_engines).values
        X_train = X_train_full[mask]
        y_train = y_train_full[mask]
    
    # Load validation and test (always full)
    X_val = np.load(WINDOWED_DATA / "FD001_X_val_windowed.npy")
    y_val = np.load(WINDOWED_DATA / "FD001_y_val_windowed.npy")
    X_test = np.load(WINDOWED_DATA / "FD001_X_test_windowed.npy")
    y_test = np.load(WINDOWED_DATA / "FD001_y_test_windowed.npy")
    
    # Filter features if specified
    if selected_features is not None:
        # Windowed data shape: (samples, timesteps, features)
        # Need to get feature indices
        all_features = ['os1', 'os2', 'os3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 
                       's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        
        # Get indices of selected features
        feature_indices = [all_features.index(f) for f in selected_features if f in all_features]
        
        # Filter features
        X_train = X_train[:, :, feature_indices]
        X_val = X_val[:, :, feature_indices]
        X_test = X_test[:, :, feature_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==========
# TRAINING AND EVALUATION
# ==========

def train_and_evaluate_transformer(n_engines, fs_method, selected_features, hyperparams):
    
    # Output directory
    output_dir = OUTPUT_BASE / fs_method / "Transformer"
    model_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    preds_dir = output_dir / "predictions"
    
    model_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    preds_dir.mkdir(exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_windowed_data(n_engines, selected_features)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Build model with input shape
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    
    # Use hyperparameters or defaults
    model = build_transformer_model(
        d_model=hyperparams.get('d_model', 64),
        num_heads=hyperparams.get('num_heads', 4),
        ff_dim=hyperparams.get('ff_dim', 128),
        num_transformer_blocks=hyperparams.get('num_transformer_blocks', 2),
        dropout_rate=hyperparams.get('dropout_rate', 0.1),
        learning_rate=hyperparams.get('learning_rate', 0.001),
        input_shape=input_shape
    )
    
    # Train model
    start_time = time.time()
    history = train_transformer_model(
        model, X_train, y_train, X_val, y_val,
        epochs=100,  # Use same as Phase 1
        batch_size=hyperparams.get('batch_size', 64)
    )
    training_time = time.time() - start_time
    epochs_trained = len(history.history['loss'])  # Get actual epochs from history
    
    # Save model
    model.save(model_dir / f"model_{n_engines}engines.keras")
    
    # Make predictions
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate metrics
    metrics = {}
    for split, y_true, y_pred in [('train', y_train, y_train_pred),
                                    ('val', y_val, y_val_pred),
                                    ('test', y_test, y_test_pred)]:
        metrics[f'{split}_rmse'] = rmse(y_true, y_pred)
        metrics[f'{split}_mae'] = mae(y_true, y_pred)
        metrics[f'{split}_r2'] = r2(y_true, y_pred)
        metrics[f'{split}_cmapss'] = cmapss_score(y_true, y_pred)
        
        _, auc = rmse_by_bins_with_auc(y_true, y_pred, RUL_BINS)
        metrics[f'{split}_auc_rmse'] = auc
    
    # Save predictions
    for split, y_true, y_pred in [('train', y_train, y_train_pred),
                                    ('val', y_val, y_val_pred),
                                    ('test', y_test, y_test_pred)]:
        pred_df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred
        })
        pred_df.to_csv(preds_dir / f"{split}_{n_engines}engines.csv", index=False)
    
    # Generate plots for selected engine counts
    if n_engines in [80, 40, 10, 5, 1]:
        # Actual vs Predicted
        plot_actual_vs_predicted(
            y_test, y_test_pred,
            dataset_name=f"FD001 ({fs_method})",
            model_name=f"Transformer ({n_engines} engines)",
            save_path=plots_dir / f"actual_vs_pred_test_{n_engines}engines.png"
        )
        
        # RMSE by bins
        bin_rmse, auc_rmse = rmse_by_bins_with_auc(y_test, y_test_pred, RUL_BINS)
        plot_rmse_by_bins(
            edges=RUL_BINS,
            rmse_bins=bin_rmse,
            auc_rmse_norm=auc_rmse,
            model_name=f"Transformer ({n_engines} engines)",
            save_path=plots_dir / f"rmse_by_bins_test_{n_engines}engines.png",
            show=False
        )
        
        # Training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Transformer Training History ({fs_method}) - {n_engines} Engines')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / f"training_history_{n_engines}engines.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Print results
    print(f"RMSE - Train: {metrics['train_rmse']:.2f}, Val: {metrics['val_rmse']:.2f}, Test: {metrics['test_rmse']:.2f}")
    print(f"Training time: {training_time:.1f}s ({epochs_trained} epochs)")
    
    # Return metrics summary in Phase 2 format (one row per engine count)
    return {
        'model_name': 'Transformer',
        'fs_method': fs_method,
        'n_engines': n_engines,
        'training_time_sec': training_time,
        **metrics
    }

# ==========
# MAIN EXECUTION
# ==========

def main():
    
    print("="*40)
    print("PHASE 2: TRANSFORMER FEATURE SELECTION EXPERIMENTS")
    print("="*40)
    print(f"\nTotal experiments: 1 model × 2 FS methods × {len(ENGINE_COUNTS)} engine counts = {2*len(ENGINE_COUNTS)}")
    print(f"Output directory: {OUTPUT_BASE}")
    
    # Load selected features
    corr_features, tree_features = load_selected_features()
    
    # Load hyperparameters
    hyperparams = load_hyperparameters()
    
    fs_methods = {
        'Correlation_FS': corr_features,
        'Tree_FS': tree_features
    }
    
    # Run experiments
    total = 2 * len(ENGINE_COUNTS)
    current = 0
    
    for fs_method, selected_features in fs_methods.items():
        print(f"\n{'='*40}")
        print(f"Feature Selection: {fs_method} ({len(selected_features)} features)")
        print(f"{'='*40}")
        
        fs_results = []
        
        for n_engines in ENGINE_COUNTS:
            current += 1
            print(f"\n  [{current}/{total}] Training Transformer with {fs_method} using {n_engines} engines...")
            
            try:
                result = train_and_evaluate_transformer(n_engines, fs_method, selected_features, hyperparams)
                fs_results.append(result)
                
            except Exception as e:
                print(f"ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save metrics summary for this FS method
        if fs_results:
            output_dir = OUTPUT_BASE / fs_method / "Transformer"
            
            # Convert to DataFrame
            df = pd.DataFrame(fs_results)
            
            # Save
            metrics_file = output_dir / "Transformer_metrics_summary.csv"
            df.to_csv(metrics_file, index=False)
            print(f"\n Saved metrics summary: {metrics_file}")
    
    print("\n" + "="*40)
    print(" TRANSFORMER PHASE 2 EXPERIMENTS COMPLETE!")
    print("="*40)
    print(f"\nResults saved to:")
    print(f"- {OUTPUT_BASE / 'Correlation_FS' / 'Transformer'}")
    print(f"- {OUTPUT_BASE / 'Tree_FS' / 'Transformer'}")

if __name__ == "__main__":
    main()

