import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from Utilities.Plots_Metrics import cmapss_score, rmse_by_bins_with_auc
from Utilities.config import (
    TCN_LSTM_SEARCH_SPACE,
    TCN_LSTM_FIXED_PARAMS,
    OPTUNA_N_TRIALS_PHASE3,
    RUL_BINS,
    ENGINE_COUNTS_ALL,
    WINDOW_SIZE,
    RANDOM_SEED
)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========
# CONFIGURATION
# ==========

DATASET = 'FD001'

# Directories
DATA_DIR = PROJECT_ROOT / 'CodeBase_Experiments' / '0_Data_Processing' / 'Data_CMAPSS' / '2_Cleaned_Data' / 'Windowed'
PHASE1_DIR = PROJECT_ROOT / 'Results' / 'Phase1_Baseline' / DATASET
PHASE2_DIR = PROJECT_ROOT / 'Results' / 'Phase2_Feature_Selection' / DATASET
OUTPUT_DIR = PROJECT_ROOT / 'Results' / 'Phase3_Feature_Extraction' / DATASET / 'TCN_GRU'

# Feature selection (from Phase 2)
FS_METHOD = 'Correlation_FS'
SELECTED_FEATURES_FILE = PHASE2_DIR / 'Feature_Analysis' / 'correlation_based' / 'selected_features.txt'

# GRU hyperparameters (fixed from Phase 1)
GRU_FIXED_HPS = None  # Will be loaded from Phase 1 best config

# Import settings from config
N_TRIALS = OPTUNA_N_TRIALS_PHASE3
BATCH_SIZE = TCN_LSTM_FIXED_PARAMS['batch_size']
MAX_EPOCHS = TCN_LSTM_FIXED_PARAMS['max_epochs']
PATIENCE = TCN_LSTM_FIXED_PARAMS['early_stopping_patience']

# ==========
# LOAD PHASE 1 GRU HYPERPARAMETERS
# ==========

def load_phase1_gru_hps():
    print("\nLoading Phase 1 GRU hyperparameters...")
    
    # Try the correct path first
    gru_hp_file = PROJECT_ROOT / 'Results' / 'Hyperparameter_Tuning' / 'GRU' / 'GRU_best_hyperparameters.json'
    
    if gru_hp_file.exists():
        with open(gru_hp_file, 'r') as f:
            raw_hps = json.load(f)
        
        # Convert from Optuna format to our format
        gru_hps = {
            'units': raw_hps.get('units_layer1', 128),
            'num_layers': raw_hps.get('n_gru_layers', 1),
            'dropout': raw_hps.get('dropout_rate', 0.1),
            'learning_rate': raw_hps.get('learning_rate', 0.001),
            'batch_size': raw_hps.get('batch_size', 32)
        }
        print(f"Loaded GRU HPs from Hyperparameter_Tuning: {gru_hps}")
        return gru_hps
    else:
        print("Phase 1 GRU HPs not found, using defaults")
        return {
            'units': 128,
            'num_layers': 1,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32
        }

def load_selected_features():
    print("\nLoading selected features from Phase 2...")
    
    if SELECTED_FEATURES_FILE.exists():
        with open(SELECTED_FEATURES_FILE, 'r') as f:
            features = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(features)} features: {features}")
        return features
    else:
        print("Selected features file not found!")
        return None

# ==========
# DATA LOADING
# ==========

def load_windowed_data(n_engines, selected_features):
    print(f"\nLoading data with {n_engines} engines...")
    
    # Load full numpy arrays
    X_train_full = np.load(DATA_DIR / f'{DATASET}_X_train_windowed.npy')
    y_train_full = np.load(DATA_DIR / f'{DATASET}_y_train_windowed.npy')
    X_val = np.load(DATA_DIR / f'{DATASET}_X_val_windowed.npy')
    y_val = np.load(DATA_DIR / f'{DATASET}_y_val_windowed.npy')
    X_test = np.load(DATA_DIR / f'{DATASET}_X_test_windowed.npy')
    y_test = np.load(DATA_DIR / f'{DATASET}_y_test_windowed.npy')
    
    # Load engine IDs to sample by engine
    train_ids = pd.read_csv(DATA_DIR / f'{DATASET}_train_ids_windowed.csv')
    
    print(f"Full train: {X_train_full.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Get unique engines
    unique_engines = train_ids['engine'].unique()
    n_total_engines = len(unique_engines)
    
    # Apply engine-based sampling to training set only
    if n_engines < n_total_engines:
        # Sample BY ENGINES (not random sequences)
        n_selected = min(n_engines, n_total_engines)
        
        np.random.seed(RANDOM_SEED)
        sampled_engines = np.random.choice(unique_engines, size=n_selected, replace=False)
        sampled_engines = sorted(sampled_engines)  # For reproducibility
        
        # Filter data for sampled engines
        mask = train_ids['engine'].isin(sampled_engines).values
        train_X = X_train_full[mask]
        train_y = y_train_full[mask]
        print(f"Using {n_engines} engines: {train_X.shape}")
    else:
        train_X = X_train_full
        train_y = y_train_full
        print(f"Using all {n_total_engines} engines: {train_X.shape}")
    
    # Data shape is (samples, timesteps, all_features)
    # Feature order in numpy arrays (from non-windowed data)
    all_feature_names = ['os1', 'os2', 'os3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 
                         's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 
                         's19', 's20', 's21']
    
    # Find indices of selected features
    feature_indices = [i for i, feat in enumerate(all_feature_names) if feat in selected_features]
    
    if len(feature_indices) != len(selected_features):
        print(f"Warning: Only found {len(feature_indices)}/{len(selected_features)} features")
        print(f"Available features: {all_feature_names}")
        print(f"Requested features: {selected_features}")
    else:
        print(f"Selected {len(feature_indices)} features: {[all_feature_names[i] for i in feature_indices]}")
    
    # Select features
    train_X = train_X[:, :, feature_indices]
    val_X = X_val[:, :, feature_indices]
    test_X = X_test[:, :, feature_indices]
    
    # Data is already in (samples, timesteps, features) format
    print(f"Final shape: {train_X.shape} (samples, timesteps={WINDOW_SIZE}, features={len(feature_indices)})")
    
    return (train_X, train_y), (val_X, y_val), (test_X, y_test)

# ==========
# MODEL BUILDING
# ==========

def build_tcn_block(x, filters, kernel_size, dilation_rate, dropout, block_id):
    # Store input for residual connection
    residual = x
    
    # First Conv1D layer
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu',
        name=f'tcn_block{block_id}_conv1_dil{dilation_rate}'
    )(x)
    x = layers.SpatialDropout1D(dropout, name=f'tcn_block{block_id}_dropout1')(x)
    
    # Second Conv1D layer
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu',
        name=f'tcn_block{block_id}_conv2_dil{dilation_rate}'
    )(x)
    x = layers.SpatialDropout1D(dropout, name=f'tcn_block{block_id}_dropout2')(x)
    
    # Residual connection (match dimensions if needed)
    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters, 1, padding='same', name=f'tcn_block{block_id}_residual')(residual)
    
    # Add residual
    x = layers.Add(name=f'tcn_block{block_id}_add')([x, residual])
    
    return x

def build_tcn_gru_model(input_shape, tcn_config, gru_config):
    inputs = keras.Input(shape=input_shape, name='input')
    x = inputs
    
    # TCN blocks for feature extraction
    dilation_rates = TCN_LSTM_FIXED_PARAMS['dilation_rates']
    
    for block_idx in range(tcn_config['num_blocks']):
        for dilation_rate in dilation_rates:
            x = build_tcn_block(
                x,
                filters=tcn_config['filters'],
                kernel_size=tcn_config['kernel_size'],
                dilation_rate=dilation_rate,
                dropout=tcn_config['dropout'],
                block_id=f"{block_idx+1}_{dilation_rate}"
            )
    
    # GRU layers (fixed from Phase 1)
    for i in range(gru_config['num_layers']):
        return_sequences = (i < gru_config['num_layers'] - 1)
        x = layers.GRU(
            units=gru_config['units'],
            return_sequences=return_sequences,
            name=f'gru_{i+1}'
        )(x)
        
        if gru_config['dropout'] > 0:
            x = layers.Dropout(gru_config['dropout'], name=f'gru_dropout_{i+1}')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='linear', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TCN_GRU')
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=gru_config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# ==========
# TRAINING & EVALUATION
# ==========

def train_model(model, train_data, val_data):
    train_X, train_y = train_data
    val_X, val_y = val_data
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )
    
    epochs_trained = len(history.history['loss'])
    return history, epochs_trained

def evaluate_model(model, train_data, val_data, test_data):
    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data
    
    # Predictions
    train_pred = model.predict(train_X, verbose=0).flatten()
    val_pred = model.predict(val_X, verbose=0).flatten()
    test_pred = model.predict(test_X, verbose=0).flatten()
    
    # RMSE
    train_rmse = np.sqrt(np.mean((train_y - train_pred) ** 2))
    val_rmse = np.sqrt(np.mean((val_y - val_pred) ** 2))
    test_rmse = np.sqrt(np.mean((test_y - test_pred) ** 2))
    
    # MAE
    train_mae = np.mean(np.abs(train_y - train_pred))
    val_mae = np.mean(np.abs(val_y - val_pred))
    test_mae = np.mean(np.abs(test_y - test_pred))
    
    # R²
    train_r2 = 1 - (np.sum((train_y - train_pred) ** 2) / np.sum((train_y - np.mean(train_y)) ** 2))
    val_r2 = 1 - (np.sum((val_y - val_pred) ** 2) / np.sum((val_y - np.mean(val_y)) ** 2))
    test_r2 = 1 - (np.sum((test_y - test_pred) ** 2) / np.sum((test_y - np.mean(test_y)) ** 2))
    
    # CMAPSS Score
    train_cmapss = cmapss_score(train_y, train_pred)
    val_cmapss = cmapss_score(val_y, val_pred)
    test_cmapss = cmapss_score(test_y, test_pred)
    
    # AUC-RMSE (returns tuple: rmse_bins, auc_rmse_norm)
    _, train_auc = rmse_by_bins_with_auc(train_y, train_pred, RUL_BINS)
    _, val_auc = rmse_by_bins_with_auc(val_y, val_pred, RUL_BINS)
    _, test_auc = rmse_by_bins_with_auc(test_y, test_pred, RUL_BINS)
    
    metrics = {
        'train_rmse': train_rmse, 'val_rmse': val_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'val_mae': val_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'val_r2': val_r2, 'test_r2': test_r2,
        'train_cmapss': train_cmapss, 'val_cmapss': val_cmapss, 'test_cmapss': test_cmapss,
        'train_auc_rmse': train_auc, 'val_auc_rmse': val_auc, 'test_auc_rmse': test_auc
    }
    
    return metrics

# ==========
# HYPERPARAMETER SEARCH WITH OPTUNA
# ==========

def optuna_objective(trial, train_data, val_data, gru_config, selected_features):
    # Suggest TCN hyperparameters from config.py search space
    tcn_config = {}
    for param_name, (param_type, param_values) in TCN_LSTM_SEARCH_SPACE.items():
        if param_type == 'categorical':
            tcn_config[param_name] = trial.suggest_categorical(param_name, param_values)
        elif param_type == 'int':
            tcn_config[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
        elif param_type == 'float':
            tcn_config[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
        elif param_type == 'float_log':
            tcn_config[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1], log=True)
    
    input_shape = (WINDOW_SIZE, len(selected_features))
    
    try:
        # Build model
        model = build_tcn_gru_model(input_shape, tcn_config, gru_config)
        
        # Train
        history, epochs = train_model(model, train_data, val_data)
        
        # Get validation RMSE (objective to minimize)
        val_predictions = model.predict(val_data[0], verbose=0).flatten()
        val_rmse = np.sqrt(np.mean((val_data[1] - val_predictions) ** 2))
        
        # Clear memory
        del model
        keras.backend.clear_session()
        
        return val_rmse
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

def optimize_tcn_hps(train_data, val_data, gru_config, selected_features):
    print("\n" + "="*40)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA (80 Engines)")
    print("="*40)
    print(f"\nNumber of trials: {N_TRIALS}")
    print(f"Search space (from config.py):")
    for param_name, (param_type, param_values) in TCN_LSTM_SEARCH_SPACE.items():
        print(f"{param_name}: {param_values}")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name='TCN_GRU_HP_Optimization'
    )
    
    # Run optimization
    study.optimize(
        lambda trial: optuna_objective(trial, train_data, val_data, gru_config, selected_features),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    # Get best trial
    best_trial = study.best_trial
    best_config = best_trial.params
    best_val_rmse = best_trial.value
    
    print(f"\n{'='*40}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*40}")
    print(f"Best Val RMSE: {best_val_rmse:.4f}")
    print(f"Best configuration:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Best trial: #{best_trial.number}")
    
    # Save all trials (for thesis plots)
    trials_df = study.trials_dataframe()
    trials_file = OUTPUT_DIR / 'optuna_trials.csv'
    trials_file.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(trials_file, index=False)
    print(f"\n All trials saved: {trials_file}")
    
    # Save best config
    best_config_file = OUTPUT_DIR / 'best_tcn_config.json'
    with open(best_config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved: {best_config_file}")
    
    # Save study object for later analysis
    study_file = OUTPUT_DIR / 'optuna_study.pkl'
    import pickle
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    print(f"Study object saved: {study_file}")
    
    # Save optimization history (for convergence plots)
    optimization_history = []
    for trial in study.trials:
        optimization_history.append({
            'trial_number': trial.number,
            'value': trial.value,
            'best_value_so_far': min([t.value for t in study.trials[:trial.number+1] if t.value != float('inf')]),
            'duration_sec': trial.duration.total_seconds() if trial.duration else None,
            'state': trial.state.name,
            **trial.params
        })
    opt_history_df = pd.DataFrame(optimization_history)
    opt_history_file = OUTPUT_DIR / 'optimization_history.csv'
    opt_history_df.to_csv(opt_history_file, index=False)
    print(f"Optimization history saved: {opt_history_file}")
    
    # Save parameter importance (if enough trials)
    if len(study.trials) >= 10:
        try:
            from optuna.importance import get_param_importances
            importances = get_param_importances(study)
            importance_df = pd.DataFrame([
                {'parameter': param, 'importance': imp} 
                for param, imp in importances.items()
            ]).sort_values('importance', ascending=False)
            importance_file = OUTPUT_DIR / 'parameter_importance.csv'
            importance_df.to_csv(importance_file, index=False)
            print(f"Parameter importance saved: {importance_file}")
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")
    
    return best_config, study

# ==========
# EVALUATE ACROSS DATA PERCENTAGES
# ==========

def save_predictions(model, train_data, val_data, test_data, n_engines):
    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data
    
    # Get predictions
    train_pred = model.predict(train_X, verbose=0).flatten()
    val_pred = model.predict(val_X, verbose=0).flatten()
    test_pred = model.predict(test_X, verbose=0).flatten()
    
    # Create predictions directory
    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV files (same format as baseline models)
    pd.DataFrame({'y_true': train_y, 'y_pred': train_pred}).to_csv(
        pred_dir / f'train_{n_engines}engines.csv', index=False)
    pd.DataFrame({'y_true': val_y, 'y_pred': val_pred}).to_csv(
        pred_dir / f'val_{n_engines}engines.csv', index=False)
    pd.DataFrame({'y_true': test_y, 'y_pred': test_pred}).to_csv(
        pred_dir / f'test_{n_engines}engines.csv', index=False)
    
    print(f"Predictions saved: {pred_dir}")

def evaluate_all_percentages(best_tcn_config, gru_config, selected_features, engine_counts):
    print("\n" + "="*40)
    print("EVALUATING BEST CONFIG ACROSS ALL ENGINE COUNTS")
    print("="*40)
    
    all_results = []
    input_shape = (WINDOW_SIZE, len(selected_features))
    
    for n_engines in engine_counts:
        print(f"\n{'='*40}")
        print(f"ENGINE COUNT: {n_engines} engines")
        print(f"{'='*40}")
        
        try:
            # Load data
            train_data, val_data, test_data = load_windowed_data(n_engines, selected_features)
            
            # Build model
            model = build_tcn_gru_model(input_shape, best_tcn_config, gru_config)
            
            # Train
            start_time = time.time()
            history, epochs = train_model(model, train_data, val_data)
            training_time = time.time() - start_time
            
            print(f"Training complete: {epochs} epochs, {training_time:.1f}s")
            
            # Save training history (for convergence plots)
            history_df = pd.DataFrame({
                'epoch': range(1, epochs + 1),
                'train_loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'train_mae': history.history.get('mae', [None] * epochs),
                'val_mae': history.history.get('val_mae', [None] * epochs),
            })
            history_dir = OUTPUT_DIR / 'training_history'
            history_dir.mkdir(parents=True, exist_ok=True)
            history_df.to_csv(history_dir / f'history_{n_engines}engines.csv', index=False)
            
            # Evaluate on all splits
            metrics = evaluate_model(model, train_data, val_data, test_data)
            
            # Save predictions for AUC-RMSE curve plotting
            save_predictions(model, train_data, val_data, test_data, n_engines)
            
            # Store results
            result = {
                'model_name': 'TCN_GRU',
                'n_engines': n_engines,
                **best_tcn_config,
                **metrics,
                'epochs_trained': epochs,
                'training_time_sec': training_time
            }
            all_results.append(result)
            
            #  CHECKPOINT: Save after each engine count (in case of crash)
            checkpoint_df = pd.DataFrame(all_results)
            checkpoint_file = OUTPUT_DIR / 'tcn_gru_all_results_checkpoint.csv'
            checkpoint_df.to_csv(checkpoint_file, index=False)
            
            print(f"\nResults for {n_engines} engines:")
            print(f"Val RMSE:  {metrics['val_rmse']:.4f}")
            print(f"Test RMSE: {metrics['test_rmse']:.4f}")
            print(f"Val R²:    {metrics['val_r2']:.4f}")
            print(f"Test R²:   {metrics['test_r2']:.4f}")
            print(f"Checkpoint saved ({len(all_results)}/{len(engine_counts)} engine counts complete)")
            
            # Save model
            model_dir = OUTPUT_DIR / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / f'tcn_gru_{n_engines}engines.keras')
            
            # Clear memory
            del model
            keras.backend.clear_session()
            
        except Exception as e:
            print(f"Error at {n_engines} engines: {e}")
            continue
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_file = OUTPUT_DIR / 'tcn_gru_all_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n All results saved: {results_file}")
    
    # Save summary statistics (for quick analysis)
    summary_stats = results_df.groupby('n_engines').agg({
        'val_rmse': ['mean', 'std', 'min', 'max'],
        'test_rmse': ['mean', 'std', 'min', 'max'],
        'val_r2': ['mean', 'std', 'min', 'max'],
        'training_time_sec': ['mean', 'std', 'min', 'max']
    }).round(4)
    summary_file = OUTPUT_DIR / 'summary_statistics.csv'
    summary_stats.to_csv(summary_file)
    print(f"Summary statistics saved: {summary_file}")
    
    # Generate metrics comparison CSV (for Phase 3 analysis plots)
    metrics_comparison = results_df[['n_engines', 'val_rmse', 'test_rmse', 'val_mae', 
                                      'test_mae', 'val_r2', 'test_r2', 'val_cmapss', 
                                      'test_cmapss', 'val_auc_rmse', 'test_auc_rmse',
                                      'training_time_sec', 'epochs_trained']].copy()
    metrics_file = OUTPUT_DIR / 'metrics_comparison.csv'
    metrics_comparison.to_csv(metrics_file, index=False)
    print(f"Metrics comparison saved: {metrics_file}")
    
    return results_df

# ==========
# MAIN EXECUTION
# ==========

def main(test_mode=False):
    print("\n" + "="*40)
    print("TCN-GRU FEATURE EXTRACTION EXPERIMENT")
    if test_mode:
        print("(TEST MODE: 2 trials, 80 engines, 1 epoch)")
    print("="*40)
    print(f"\nDataset: {DATASET}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Feature selection: {FS_METHOD}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Override settings for test mode
    global N_TRIALS, MAX_EPOCHS
    if test_mode:
        N_TRIALS = 2
        MAX_EPOCHS = 1
        engine_counts_to_use = [80]  # Only test on full data
        print("\n TEST MODE ACTIVE:")
        print(f"N_TRIALS: {N_TRIALS}")
        print(f"MAX_EPOCHS: {MAX_EPOCHS}")
        print(f"ENGINE_COUNTS: {engine_counts_to_use}")
    else:
        engine_counts_to_use = ENGINE_COUNTS_ALL
    
    # Load configurations
    global GRU_FIXED_HPS
    GRU_FIXED_HPS = load_phase1_gru_hps()
    selected_features = load_selected_features()
    
    if selected_features is None:
        print("Cannot proceed without selected features!")
        return
    
    print(f"\nFixed GRU config: {GRU_FIXED_HPS}")
    print(f"Selected features: {len(selected_features)} features")
    
    # Check if we already have best config (skip Optuna if we do)
    best_config_file = OUTPUT_DIR / "best_tcn_config.json"
    if best_config_file.exists():
        print("\n" + "="*40)
        print("FOUND EXISTING BEST CONFIG - SKIPPING OPTUNA")
        print("="*40)
        with open(best_config_file, 'r') as f:
            best_tcn_config = json.load(f)
        print(f"\nLoaded best config: {best_tcn_config}")
        study = None  # No study object when loading from file
    else:
        # Step 1: Hyperparameter optimization on 80 engines
        print("\n" + "="*40)
        print("STEP 1: HYPERPARAMETER OPTIMIZATION (100% DATA)")
        print("="*40)
        
        train_data_80, val_data_80, _ = load_windowed_data(80, selected_features)
        best_tcn_config, study = optimize_tcn_hps(
            train_data_80, val_data_80, GRU_FIXED_HPS, selected_features
        )
    
    # Step 2: Evaluate best config on all engine counts
    print("\n" + "="*40)
    print("STEP 2: EVALUATE ON ALL ENGINE COUNTS")
    print("="*40)
    
    final_results = evaluate_all_percentages(best_tcn_config, GRU_FIXED_HPS, selected_features, engine_counts_to_use)
    
    # Save experiment metadata (for thesis documentation)
    metadata = {
        'experiment_name': 'TCN_GRU_Feature_Extraction',
        'dataset': DATASET,
        'window_size': WINDOW_SIZE,
        'feature_selection_method': FS_METHOD,
        'num_selected_features': len(selected_features),
        'selected_features': selected_features,
        'engine_counts': engine_counts_to_use,
        'optuna_trials': N_TRIALS,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'early_stopping_patience': PATIENCE,
        'gru_fixed_config': GRU_FIXED_HPS,
        'best_tcn_config': best_tcn_config,
        'best_val_rmse_80engines': study.best_value if study else "loaded_from_file",
        'total_optuna_trials_completed': len(study.trials) if study else "loaded_from_file",
        'experiment_timestamp': pd.Timestamp.now().isoformat()
    }
    metadata_file = OUTPUT_DIR / 'experiment_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n Experiment metadata saved: {metadata_file}")
    
    # Final summary
    print("\n" + "="*40)
    print("TCN-GRU EXPERIMENT COMPLETE!")
    print("="*40)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"experiment_metadata.json - Experiment configuration & metadata")
    print(f"optuna_trials.csv - All Optuna trials (for HP search analysis)")
    print(f"optimization_history.csv - Trial-by-trial convergence data")
    print(f"parameter_importance.csv - Which HPs mattered most")
    print(f"optuna_study.pkl - Full Optuna study object")
    print(f"best_tcn_config.json - Best TCN configuration found")
    print(f"tcn_gru_all_results.csv - Full results across all engine counts")
    print(f"metrics_comparison.csv - Clean metrics table for plotting")
    print(f"summary_statistics.csv - Aggregated statistics")
    print(f"training_history/ - Per-epoch loss curves for each engine count")
    print(f"models/ - Trained .keras models for each engine count")
    print()

if __name__ == '__main__':
    import sys
    # Check for test mode flag
    test_mode = '--test' in sys.argv or '-t' in sys.argv
    main(test_mode=test_mode)

