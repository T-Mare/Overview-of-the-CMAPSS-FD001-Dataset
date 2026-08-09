import pandas as pd
import numpy as np
import os
import sys
import json
import time
import pickle
import argparse
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

# Import model builders
# Regression
sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "1_Regression_Models", "Polynomial_Deg2"))
from poly2_model import get_model as get_poly2_model

sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "1_Regression_Models", "Lasso"))
from lasso_model import get_model as get_lasso_model

# Tree-based
sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "2_Tree_Based_Models", "RandomForest"))
from rf_model import get_model as get_rf_model

sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "2_Tree_Based_Models", "XGBoost"))
from xgb_model import get_model as get_xgb_model

# Deep Learning
sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "3_Deep_Learning_Models", "GRU"))
from gru_model import build_model as build_gru_model, train_model as train_gru_model

sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "3_Deep_Learning_Models", "BiLSTM"))
from bilstm_model import build_model as build_bilstm_model, train_model as train_bilstm_model

sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "3_Deep_Learning_Models", "LSTM"))
from lstm_model import build_model as build_lstm_model, train_model as train_lstm_model

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
NON_WINDOWED_DATA = DATA_BASE / "Non_Windowed"
WINDOWED_DATA = DATA_BASE / "Windowed"

FEATURE_ANALYSIS = Path(project_root) / "Results" / "Phase2_Feature_Selection" / "FD001" / "Feature_Analysis"
OUTPUT_BASE = Path(project_root) / "Results" / "Phase2_Feature_Selection" / "FD001"

HYPERPARAM_DIR = Path(project_root) / "Results" / "Hyperparameter_Tuning"

# Create output directories
for fs_method in ['Correlation_FS', 'Tree_FS']:
    (OUTPUT_BASE / fs_method).mkdir(parents=True, exist_ok=True)

# Random seed
np.random.seed(RANDOM_SEED)

# Window size for DL models
WINDOW_SIZE = 30

# Engine counts for data scarcity experiments (from config.py)
ENGINE_COUNTS = ENGINE_COUNTS_ALL  # [80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
N_TOTAL_ENGINES = 80  # FD001 has 80 training engines

# ==========
# LOAD SELECTED FEATURES
# ==========

def load_selected_features():
    print("\n" + "="*40)
    print("LOADING SELECTED FEATURES")
    print("="*40)
    
    # Load correlation-based features
    corr_file = FEATURE_ANALYSIS / "correlation_based" / "selected_features.txt"
    with open(corr_file, 'r') as f:
        corr_features = [line.strip() for line in f.readlines()]
    print(f"\n  Correlation-based FS: {len(corr_features)} features")
    print(f"{corr_features}")
    
    # Load tree-based features
    tree_file = FEATURE_ANALYSIS / "tree_based" / "selected_features.txt"
    with open(tree_file, 'r') as f:
        tree_features = [line.strip() for line in f.readlines()]
    print(f"\n  Tree-based FS: {len(tree_features)} features")
    print(f"{tree_features}")
    
    return corr_features, tree_features

# ==========
# LOAD PHASE 1 HYPERPARAMETERS
# ==========

# Hardcoded Phase 1 best hyperparameters (from Optuna tuning)
PHASE1_HYPERPARAMS = {
    'Poly2': {},  # No hyperparameters (uses default Pipeline)
    'Lasso': {'alpha': 0.5},  # From Phase 1
    'RF': None,  # Will load from JSON
    'XGB': None,  # Will load from JSON
    'LGBM': None,  # Will load from JSON
    'GRU': {
        'n_recurrent_layers': 1,
        'units_layer1': 128,
        'units_layer2': 64,
        'dropout_rate': 0.2,
        'recurrent_dropout': 0.0,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100
    },
    'BiLSTM': {
        'n_recurrent_layers': 1,
        'units_layer1': 64,
        'units_layer2': 32,
        'dropout_rate': 0.3,
        'recurrent_dropout': 0.0,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100
    },
    'LSTM': {
        'n_recurrent_layers': 1,
        'units_layer1': 128,
        'units_layer2': 64,
        'dropout_rate': 0.2,
        'recurrent_dropout': 0.0,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100
    }
}

def load_hyperparameters():
    print("\n" + "="*40)
    print("LOADING PHASE 1 HYPERPARAMETERS")
    print("="*40)
    
    hyperparams = PHASE1_HYPERPARAMS.copy()
    
    # Load tree-based hyperparameters from JSON files
    for model_name in ['RF', 'XGB', 'LGBM']:
        json_file = HYPERPARAM_DIR / model_name / f"best_params_{model_name}.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                hyperparams[model_name] = data['best_params']
            print(f"Loaded {model_name} hyperparameters from JSON")
        else:
            print(f"WARNING: {json_file} not found. Using defaults.")
            hyperparams[model_name] = {}
    
    return hyperparams

# ==========
# DATA LOADING
# ==========

def load_non_windowed_data(n_engines_to_use, selected_features=None):
    
    # Load full training data
    full_features = pd.read_csv(NON_WINDOWED_DATA / "FD001_train_features.csv")
    full_ids = pd.read_csv(NON_WINDOWED_DATA / "FD001_train_ids.csv")
    full_data = pd.concat([full_ids, full_features], axis=1)
    
    # Get unique engines
    unique_engines = full_data['engine'].unique()
    n_total_engines = len(unique_engines)
    
    if n_engines_to_use == n_total_engines:
        # Use all data
        train_ids = full_ids
        train_features = full_features
    else:
        # Sample by engines
        n_selected = min(n_engines_to_use, n_total_engines)
        np.random.seed(RANDOM_SEED)
        sampled_engines = np.random.choice(unique_engines, size=n_selected, replace=False)
        sampled_engines = sorted(sampled_engines)  # For reproducibility
        sampled_data = full_data[full_data['engine'].isin(sampled_engines)]
        
        train_ids = sampled_data[['engine', 'cycle', 'RUL']]
        train_features = sampled_data.drop(['engine', 'cycle', 'RUL'], axis=1)
    
    # Load validation and test (always full)
    val_features = pd.read_csv(NON_WINDOWED_DATA / "FD001_val_features.csv")
    val_ids = pd.read_csv(NON_WINDOWED_DATA / "FD001_val_ids.csv")
    test_features = pd.read_csv(NON_WINDOWED_DATA / "FD001_test_features.csv")
    test_ids = pd.read_csv(NON_WINDOWED_DATA / "FD001_test_ids.csv")
    
    # Filter features if specified
    if selected_features is not None:
        train_features = train_features[selected_features]
        val_features = val_features[selected_features]
        test_features = test_features[selected_features]
    
    # Extract targets
    y_train = train_ids['RUL'].values
    y_val = val_ids['RUL'].values
    y_test = test_ids['RUL'].values
    
    X_train = train_features.values
    X_val = val_features.values
    X_test = test_features.values
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_windowed_data(n_engines_to_use, selected_features=None):
    
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
        feature_indices = [all_features.index(f) for f in selected_features if f in all_features]
        
        X_train = X_train[:, :, feature_indices]
        X_val = X_val[:, :, feature_indices]
        X_test = X_test[:, :, feature_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==========
# MODEL TRAINING & EVALUATION
# ==========

def train_regression_model(model_name, model_builder, X_train, y_train, X_val, y_val, X_test, y_test, hyperparams):
    
    # Build model - handle different param formats
    if model_name == 'Lasso' and hyperparams and 'alpha' in hyperparams:
        # Lasso expects alpha as direct parameter, not dict
        model = model_builder(alpha=hyperparams['alpha'])
    elif hyperparams is None or len(hyperparams) == 0:
        model = model_builder(None)
    else:
        model = model_builder(hyperparams)
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train': {
            'rmse': rmse(y_train, train_pred),
            'mae': mae(y_train, train_pred),
            'r2': r2(y_train, train_pred),
            'cmapss': cmapss_score(y_train, train_pred),
            'auc_rmse': rmse_by_bins_with_auc(y_train, train_pred, RUL_BINS)[1]
        },
        'val': {
            'rmse': rmse(y_val, val_pred),
            'mae': mae(y_val, val_pred),
            'r2': r2(y_val, val_pred),
            'cmapss': cmapss_score(y_val, val_pred),
            'auc_rmse': rmse_by_bins_with_auc(y_val, val_pred, RUL_BINS)[1]
        },
        'test': {
            'rmse': rmse(y_test, test_pred),
            'mae': mae(y_test, test_pred),
            'r2': r2(y_test, test_pred),
            'cmapss': cmapss_score(y_test, test_pred),
            'auc_rmse': rmse_by_bins_with_auc(y_test, test_pred, RUL_BINS)[1]
        },
        'training_time_sec': training_time
    }
    
    predictions = {
        'train': train_pred,
        'val': val_pred,
        'test': test_pred
    }
    
    return model, metrics, predictions

def train_tree_model(model_name, model_builder, X_train, y_train, X_val, y_val, X_test, y_test, hyperparams):
    
    # Build model - pass hyperparams as kwargs to override defaults
    if hyperparams and len(hyperparams) > 0:
        model = model_builder(use_tuned_params=False, **hyperparams)
    else:
        model = model_builder(use_tuned_params=True)
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train': {
            'rmse': rmse(y_train, train_pred),
            'mae': mae(y_train, train_pred),
            'r2': r2(y_train, train_pred),
            'cmapss': cmapss_score(y_train, train_pred),
            'auc_rmse': rmse_by_bins_with_auc(y_train, train_pred, RUL_BINS)[1]
        },
        'val': {
            'rmse': rmse(y_val, val_pred),
            'mae': mae(y_val, val_pred),
            'r2': r2(y_val, val_pred),
            'cmapss': cmapss_score(y_val, val_pred),
            'auc_rmse': rmse_by_bins_with_auc(y_val, val_pred, RUL_BINS)[1]
        },
        'test': {
            'rmse': rmse(y_test, test_pred),
            'mae': mae(y_test, test_pred),
            'r2': r2(y_test, test_pred),
            'cmapss': cmapss_score(y_test, test_pred),
            'auc_rmse': rmse_by_bins_with_auc(y_test, test_pred, RUL_BINS)[1]
        },
        'training_time_sec': training_time
    }
    
    predictions = {
        'train': train_pred,
        'val': val_pred,
        'test': test_pred
    }
    
    return model, metrics, predictions

def train_dl_model(model_name, model_builder, train_fn, X_train, y_train, X_val, y_val, X_test, y_test, hyperparams):
    
    # Extract hyperparameters
    n_features = X_train.shape[2]
    input_shape = (WINDOW_SIZE, n_features)
    
    # Build model
    build_params = {
        'n_recurrent_layers': hyperparams.get('n_recurrent_layers', 1),
        'units_layer1': hyperparams.get('units_layer1', 128),
        'units_layer2': hyperparams.get('units_layer2', 64),
        'dropout_rate': hyperparams.get('dropout_rate', 0.2),
        'recurrent_dropout': hyperparams.get('recurrent_dropout', 0.0),
        'learning_rate': hyperparams.get('learning_rate', 0.001),
        'input_shape': input_shape,
        'random_seed': RANDOM_SEED
    }
    
    # BiLSTM has an extra parameter
    if model_name == 'BiLSTM':
        build_params['merge_mode'] = 'concat'
    
    model = model_builder(**build_params)
    
    # Train
    start_time = time.time()
    history = train_fn(
        model, X_train, y_train, X_val, y_val,
        epochs=hyperparams.get('epochs', 100),
        batch_size=hyperparams.get('batch_size', 64),
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Predictions
    train_pred = model.predict(X_train, verbose=0).flatten()
    val_pred = model.predict(X_val, verbose=0).flatten()
    test_pred = model.predict(X_test, verbose=0).flatten()
    
    # Metrics
    metrics = {
        'train': {
            'rmse': rmse(y_train, train_pred),
            'mae': mae(y_train, train_pred),
            'r2': r2(y_train, train_pred),
            'cmapss': cmapss_score(y_train, train_pred),
            'auc_rmse': rmse_by_bins_with_auc(y_train, train_pred, RUL_BINS)[1]
        },
        'val': {
            'rmse': rmse(y_val, val_pred),
            'mae': mae(y_val, val_pred),
            'r2': r2(y_val, val_pred),
            'cmapss': cmapss_score(y_val, val_pred),
            'auc_rmse': rmse_by_bins_with_auc(y_val, val_pred, RUL_BINS)[1]
        },
        'test': {
            'rmse': rmse(y_test, test_pred),
            'mae': mae(y_test, test_pred),
            'r2': r2(y_test, test_pred),
            'cmapss': cmapss_score(y_test, test_pred),
            'auc_rmse': rmse_by_bins_with_auc(y_test, test_pred, RUL_BINS)[1]
        },
        'training_time_sec': training_time
    }
    
    predictions = {
        'train': train_pred,
        'val': val_pred,
        'test': test_pred
    }
    
    return model, metrics, predictions, history

# ==========
# SAVE RESULTS
# ==========

def save_results(model_name, fs_method, n_engines, metrics, predictions, model=None):
    
    # Create output directory
    output_dir = OUTPUT_BASE / fs_method / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    
    # Save predictions
    for split in ['train', 'val', 'test']:
        pred_file = output_dir / "predictions" / f"{split}_{n_engines}engines.csv"
        pd.DataFrame({
            'y_true': predictions[f'{split}_true'] if f'{split}_true' in predictions else [],
            'y_pred': predictions[split]
        }).to_csv(pred_file, index=False)
    
    # Save model
    if model is not None:
        model_file = output_dir / "models" / f"model_{n_engines}engines"
        if hasattr(model, 'save'):  # Keras model
            model.save(str(model_file) + ".keras")
        else:  # Sklearn model
            with open(str(model_file) + ".pkl", 'wb') as f:
                pickle.dump(model, f)
    
    # Append metrics to summary
    summary_file = output_dir / f"{model_name}_metrics_summary.csv"
    
    # Create metrics row
    metrics_row = {
        'model_name': model_name,
        'fs_method': fs_method,
        'n_engines': n_engines,
        'training_time_sec': metrics['training_time_sec']
    }
    
    for split in ['train', 'val', 'test']:
        for metric_name in ['rmse', 'mae', 'r2', 'cmapss', 'auc_rmse']:
            metrics_row[f'{split}_{metric_name}'] = metrics[split][metric_name]
    
    # Append to CSV
    df_row = pd.DataFrame([metrics_row])
    if summary_file.exists():
        df_existing = pd.read_csv(summary_file)
        df_combined = pd.concat([df_existing, df_row], ignore_index=True)
        df_combined.to_csv(summary_file, index=False)
    else:
        df_row.to_csv(summary_file, index=False)

# ==========
# MAIN EXPERIMENT LOOP
# ==========

def run_experiments(test_mode=False):
    
    print("\n" + "="*40)
    print("PHASE 2: FEATURE SELECTION EXPERIMENTS")
    if test_mode:
        print("*** TEST MODE: 2 models × 1 FS × 2 engine counts × 1 epoch ***")
    print("="*40)
    
    n_engine_counts = len(ENGINE_COUNTS)  # 17 engine counts
    if not test_mode:
        print(f"\nTotal experiments: 8 models × 2 FS methods × {n_engine_counts} engine counts = {8*2*n_engine_counts}")
    else:
        print(f"\nTotal experiments: 2 models × 1 FS method × 2 engine counts = 4 (TESTING)")
    print(f"Output directory: {OUTPUT_BASE}")
    
    # Load selected features
    corr_features, tree_features = load_selected_features()
    
    # Load hyperparameters
    hyperparams = load_hyperparameters()
    
    # Define experiments
    experiments = [
        # Regression models
        {'name': 'Poly2', 'type': 'regression', 'builder': get_poly2_model, 'windowed': False},
        {'name': 'Lasso', 'type': 'regression', 'builder': get_lasso_model, 'windowed': False},
        # Tree-based models
        {'name': 'RF', 'type': 'tree', 'builder': get_rf_model, 'windowed': False},
        {'name': 'XGB', 'type': 'tree', 'builder': get_xgb_model, 'windowed': False},
        # Deep learning models
        {'name': 'GRU', 'type': 'dl', 'builder': build_gru_model, 'train_fn': train_gru_model, 'windowed': True},
        {'name': 'BiLSTM', 'type': 'dl', 'builder': build_bilstm_model, 'train_fn': train_bilstm_model, 'windowed': True},
        {'name': 'LSTM', 'type': 'dl', 'builder': build_lstm_model, 'train_fn': train_lstm_model, 'windowed': True},
        {'name': 'Transformer', 'type': 'dl', 'builder': build_transformer_model, 'train_fn': train_transformer_model, 'windowed': True},
    ]
    
    # Test mode: only run 2 models
    if test_mode:
        experiments = [
            {'name': 'Poly2', 'type': 'regression', 'builder': get_poly2_model, 'windowed': False},
            {'name': 'GRU', 'type': 'dl', 'builder': build_gru_model, 'train_fn': train_gru_model, 'windowed': True},
        ]
    
    fs_methods = {
        'Correlation_FS': corr_features,
        'Tree_FS': tree_features
    }
    
    # Test mode: only run 1 FS method
    if test_mode:
        fs_methods = {'Correlation_FS': corr_features}
    
    engine_counts_to_run = ENGINE_COUNTS if not test_mode else [80, 40]  # Test mode: only 80 and 40 engines
    
    # Run experiments
    total = len(experiments) * len(fs_methods) * len(engine_counts_to_run)
    current = 0
    
    for exp in experiments:
        model_name = exp['name']
        model_type = exp['type']
        
        print("\n" + "="*40)
        print(f"MODEL: {model_name} ({model_type})")
        print("="*40)
        
        for fs_method, selected_features in fs_methods.items():
            print(f"\n  Feature Selection: {fs_method} ({len(selected_features)} features)")
            
            for n_engines in engine_counts_to_run:
                current += 1
                print(f"\n    [{current}/{total}] Training {model_name} with {fs_method} using {n_engines} engines...")
                
                try:
                    # Load data
                    if exp['windowed']:
                        X_train, y_train, X_val, y_val, X_test, y_test = load_windowed_data(n_engines, selected_features)
                    else:
                        X_train, y_train, X_val, y_val, X_test, y_test = load_non_windowed_data(n_engines, selected_features)
                    
                    # Train model
                    model_hyperparams = hyperparams.get(model_name, {})
                    
                    if not model_hyperparams and model_type != 'regression':
                        print(f"WARNING: No hyperparameters found for {model_name}. Using defaults.")
                    
                    # Make a copy to avoid modifying original
                    model_hyperparams = model_hyperparams.copy() if model_hyperparams else {}
                    
                    # Test mode: override epochs to 1
                    if test_mode and model_type == 'dl':
                        model_hyperparams['epochs'] = 1
                        print(f"(Test mode: 1 epoch only)")
                    
                    if model_type == 'dl':
                        model, metrics, predictions, history = train_dl_model(
                            model_name, exp['builder'], exp['train_fn'],
                            X_train, y_train, X_val, y_val, X_test, y_test,
                            model_hyperparams
                        )
                    else:
                        model, metrics, predictions = train_regression_model(
                            model_name, exp['builder'],
                            X_train, y_train, X_val, y_val, X_test, y_test,
                            model_hyperparams
                        ) if model_type == 'regression' else train_tree_model(
                            model_name, exp['builder'],
                            X_train, y_train, X_val, y_val, X_test, y_test,
                            model_hyperparams
                        )
                    
                    # Add true values to predictions dict
                    predictions['train_true'] = y_train
                    predictions['val_true'] = y_val
                    predictions['test_true'] = y_test
                    
                    # Save results
                    save_results(model_name, fs_method, n_engines, metrics, predictions, model)
                    
                    # Print metrics
                    print(f"Val RMSE: {metrics['val']['rmse']:.2f} | Test RMSE: {metrics['test']['rmse']:.2f} | Time: {metrics['training_time_sec']:.1f}s")
                    
                except Exception as e:
                    print(f"ERROR: {str(e)}")
                    continue
    
    print("\n" + "="*40)
    print(" PHASE 2 EXPERIMENTS COMPLETE!")
    print("="*40)
    print(f"\nResults saved to: {OUTPUT_BASE}")
 
# ==========
# MAIN EXECUTION
# ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 2 Feature Selection Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (full production run):
  # 7 models × 2 FS methods × 17 engine counts = 238 experiments
  python run_phase2_fs_experiments.py

  # Test mode (4 experiments only - quick test):
  python run_phase2_fs_experiments.py --test
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: Run only 2 models × 1 FS × 2 engine counts × 1 epoch (4 experiments)'
    )
    
    args = parser.parse_args()
    
    run_experiments(test_mode=args.test)

