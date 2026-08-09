import pandas as pd
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Import utilities
from Utilities.config import (
    NON_WINDOWED_DATA_PATH,
    WINDOWED_DATA_PATH,
    RESULTS_BASE_PATH, 
    RANDOM_SEED, 
    RUL_BINS,
    ENGINE_COUNTS_ALL,
)
from Utilities.Plots_Metrics import (
    rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc,
    plot_actual_vs_predicted, plot_rmse_by_bins
)

# Import model modules
sys.path.insert(0, str(Path(__file__).parent))
from ANN import ann_model
from RNN import rnn_model
from LSTM import lstm_model
from GRU import gru_model
from BiLSTM import bilstm_model
from CNN import cnn_model
from TCN import tcn_model
from Transformer import transformer_model

# ==========
# GPU CONFIGURATION
# ==========

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) detected: {len(gpus)}")
            print(f"GPU device(s): {[gpu.name for gpu in gpus]}")
            print(f"Memory growth enabled (dynamic allocation)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected. Running on CPU (this will be slow).")

# ==========
# MODEL CONFIGURATIONS
# ==========

# Models dictionary
MODELS = {
    'ANN': {
        'module': ann_model,
        'uses_windowed_data': False,
    },
    'RNN': {
        'module': rnn_model,
        'uses_windowed_data': True,
    },
    'LSTM': {
        'module': lstm_model,
        'uses_windowed_data': True,
    },
    'GRU': {
        'module': gru_model,
        'uses_windowed_data': True,
    },
    'BiLSTM': {
        'module': bilstm_model,
        'uses_windowed_data': True,
    },
    'CNN': {
        'module': cnn_model,
        'uses_windowed_data': True,
    },
    'TCN': {
        'module': tcn_model,
        'uses_windowed_data': True,
    },
    'Transformer': {
        'module': transformer_model,
        'uses_windowed_data': True,
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
    
    # Load non-windowed data (for ANN)
    data_path_non_windowed = Path(NON_WINDOWED_DATA_PATH)
    
    train_features = pd.read_csv(data_path_non_windowed / 'FD001_train_features.csv')
    train_ids = pd.read_csv(data_path_non_windowed / 'FD001_train_ids.csv')
    
    val_features = pd.read_csv(data_path_non_windowed / 'FD001_val_features.csv')
    val_ids = pd.read_csv(data_path_non_windowed / 'FD001_val_ids.csv')
    
    test_features = pd.read_csv(data_path_non_windowed / 'FD001_test_features.csv')
    test_ids = pd.read_csv(data_path_non_windowed / 'FD001_test_ids.csv')
    
    X_train_full = train_features.values
    y_train_full = train_ids['RUL'].values
    
    X_val = val_features.values
    y_val = val_ids['RUL'].values
    
    X_test = test_features.values
    y_test = test_ids['RUL'].values
    
    # Load windowed data (for sequential models)
    data_path_windowed = Path(WINDOWED_DATA_PATH)
    
    X_train_full_windowed = np.load(data_path_windowed / 'FD001_X_train_windowed.npy')
    y_train_full_windowed = np.load(data_path_windowed / 'FD001_y_train_windowed.npy')
    
    X_val_windowed = np.load(data_path_windowed / 'FD001_X_val_windowed.npy')
    y_val_windowed = np.load(data_path_windowed / 'FD001_y_val_windowed.npy')
    
    X_test_windowed = np.load(data_path_windowed / 'FD001_X_test_windowed.npy')
    y_test_windowed = np.load(data_path_windowed / 'FD001_y_test_windowed.npy')
    
    print(f"\n Non-windowed data loaded:")
    print(f"Full train: {len(X_train_full):,} samples, {len(train_ids['engine'].unique())} engines")
    print(f"Validation: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")
    print(f"Features: {X_train_full.shape[1]}")
    
    print(f"\n Windowed data loaded:")
    print(f"Full train: {len(X_train_full_windowed):,} sequences")
    print(f"Validation: {len(X_val_windowed):,} sequences")
    print(f"Test: {len(X_test_windowed):,} sequences")
    print(f"Sequence shape: {X_train_full_windowed.shape[1:]}")
    
    # Also load windowed train IDs for engine-based sampling
    train_ids_windowed = pd.read_csv(data_path_windowed / 'FD001_train_ids_windowed.csv')
    
    return {
        # Non-windowed (for ANN)
        'X_train_full': X_train_full,
        'y_train_full': y_train_full,
        'train_ids': train_ids,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        
        # Windowed (for sequential models)
        'X_train_full_windowed': X_train_full_windowed,
        'y_train_full_windowed': y_train_full_windowed,
        'train_ids_windowed': train_ids_windowed,
        'X_val_windowed': X_val_windowed,
        'y_val_windowed': y_val_windowed,
        'X_test_windowed': X_test_windowed,
        'y_test_windowed': y_test_windowed,
    }

def create_engine_subset(data, n_engines_target, use_windowed=False):
    # Select which dataset to use
    if use_windowed:
        X_full = data['X_train_full_windowed']
        y_full = data['y_train_full_windowed']
        train_ids = data['train_ids_windowed']
    else:
        X_full = data['X_train_full']
        y_full = data['y_train_full']
        train_ids = data['train_ids']
    
    unique_engines = train_ids['engine'].unique()
    total_engines = len(unique_engines)
    
    # Handle case where target equals or exceeds total
    if n_engines_target >= total_engines:
        return X_full, y_full, total_engines
    
    # Sample exact number of engines
    np.random.seed(RANDOM_SEED)
    selected_engines = np.random.choice(unique_engines, size=n_engines_target, replace=False)
    selected_engines = np.sort(selected_engines)  # Sort for reproducibility
    
    # Get indices for selected engines
    mask = train_ids['engine'].isin(selected_engines)
    indices = np.where(mask)[0]
    
    X_subset = X_full[indices]
    y_subset = y_full[indices]
    
    n_samples = len(y_subset)
    data_type = "windowed" if use_windowed else "non-windowed"
    print(f"Selected {n_engines_target} engines ({data_type}): {list(selected_engines)}  {n_samples:,} samples")
    
    return X_subset, y_subset, n_engines_target

def load_best_hyperparameters(model_name):
    hp_path = Path(RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / model_name / f'{model_name}_best_hyperparameters.json'
    
    if not hp_path.exists():
        raise FileNotFoundError(f"Hyperparameters not found: {hp_path}")
    
    with open(hp_path, 'r') as f:
        best_params = json.load(f)
    
    return best_params

def build_and_train_model(model_name, model_module, X_train, y_train, X_val, y_val, hyperparams, epochs, verbose=0):
    # Determine input shape
    input_shape = X_train.shape[1:] if len(X_train.shape) > 2 else X_train.shape[1]
    
    # Build model based on model type (each has different signature)
    if model_name == 'ANN':
        model = model_module.build_model(
            n_hidden_layers=hyperparams['n_hidden_layers'],
            units_layer1=hyperparams['units_layer1'],
            units_layer2=hyperparams.get('units_layer2', 0),
            dropout_rate=hyperparams['dropout_rate'],
            learning_rate=hyperparams['learning_rate'],
            input_shape=input_shape
        )
    
    elif model_name in ['RNN', 'LSTM', 'GRU']:
        layer_key = f"n_{model_name.lower()}_layers" if model_name in ['LSTM', 'GRU'] else 'n_recurrent_layers'
        model = model_module.build_model(
            n_recurrent_layers=hyperparams[layer_key],
            units_layer1=hyperparams['units_layer1'],
            units_layer2=hyperparams.get('units_layer2', 0),
            dropout_rate=hyperparams['dropout_rate'],
            recurrent_dropout=0.1,  # Fixed parameter
            learning_rate=hyperparams['learning_rate'],
            input_shape=input_shape
        )
    
    elif model_name == 'BiLSTM':
        model = model_module.build_model(
            n_recurrent_layers=hyperparams['n_bilstm_layers'],
            units_layer1=hyperparams['units_layer1'],
            units_layer2=hyperparams.get('units_layer2', 0),
            dropout_rate=hyperparams['dropout_rate'],
            recurrent_dropout=0.1,  # Fixed parameter
            learning_rate=hyperparams['learning_rate'],
            merge_mode='concat',
            input_shape=input_shape
        )
    
    elif model_name == 'CNN':
        model = model_module.build_model(
            n_conv_layers=hyperparams['n_conv_layers'],
            n_filters=hyperparams['n_filters'],
            kernel_size=3,  # Fixed parameter
            pool_size=2,  # Fixed parameter
            dropout_rate=hyperparams['dropout_rate'],
            learning_rate=hyperparams['learning_rate'],
            input_shape=input_shape
        )
    
    elif model_name == 'TCN':
        model = model_module.build_model(
            n_blocks=hyperparams['n_blocks'],
            n_filters=hyperparams['n_filters'],
            kernel_size=3,  # Fixed parameter
            dilation_rates=[1, 2, 4],  # Fixed parameter
            dropout_rate=hyperparams['dropout_rate'],
            learning_rate=hyperparams['learning_rate'],
            input_shape=input_shape
        )
    
    elif model_name == 'Transformer':
        model = model_module.build_model(
            d_model=hyperparams['d_model'],
            num_heads=hyperparams['num_heads'],
            ff_dim=hyperparams['ff_dim'],
            num_transformer_blocks=hyperparams['num_transformer_blocks'],
            dropout_rate=hyperparams['dropout_rate'],
            learning_rate=hyperparams['learning_rate'],
            use_positional_encoding=True,  # Fixed parameter
            input_shape=input_shape
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=hyperparams['batch_size'],
        callbacks=[early_stop],
        verbose=verbose
    )
    
    return model, history

def evaluate_model(model, X, y):
    y_pred = model.predict(X, verbose=0).flatten()
    
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
    
    print(f"Metrics saved: {output_file.name}")
    
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

def save_training_history_plot(history, n_engines, model_name, save_dir):
    plots_dir = Path(save_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Training Loss ({n_engines} engines)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss (MSE)', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: MAE
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_title(f'{model_name} - Training MAE ({n_engines} engines)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('MAE', fontsize=11)
    else:
        # Fallback if MAE not available
        axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title(f'{model_name} - Training Loss ({n_engines} engines)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Loss (MSE)', fontsize=11)
    
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = plots_dir / f'training_history_{n_engines}engines.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_plots(y_true, y_pred, n_engines, split_name, model_name, save_dir):
    plots_dir = Path(save_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Actual vs Predicted
    plot_actual_vs_predicted(
        y_true, y_pred,
        dataset_name=f'{split_name.upper()}, {n_engines} engines',
        model_name=model_name,
        save_path=plots_dir / f'actual_vs_pred_{split_name}_{n_engines}engines.png',
        show=False
    )
    
    # 2. RMSE by Bins
    bin_rmse_list, auc_rmse = rmse_by_bins_with_auc(y_true, y_pred, RUL_BINS)
    
    plot_rmse_by_bins(
        edges=RUL_BINS,
        rmse_bins=bin_rmse_list,
        auc_rmse_norm=auc_rmse,
        model_name=model_name,
        save_path=plots_dir / f'rmse_by_bins_{split_name}_{n_engines}engines.png',
        show=False
    )

def run_single_experiment(model_name, model_config, data, n_engines, best_params, epochs=100):
    print(f"\n  [ENGINES: {n_engines}]")
    
    # Determine which data to use (windowed vs non-windowed)
    uses_windowed = model_config['uses_windowed_data']
    
    # Create subset by engine
    X_train, y_train, n_engines_used = create_engine_subset(data, n_engines, use_windowed=uses_windowed)
    
    # Get validation and test sets
    if uses_windowed:
        X_val = data['X_val_windowed']
        y_val = data['y_val_windowed']
        X_test = data['X_test_windowed']
        y_test = data['y_test_windowed']
    else:
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
    
    print(f"Training samples: {len(X_train):,}")
    
    # Build and train model
    model_module = model_config['module']
    
    start_time = time.time()
    model, history = build_and_train_model(
        model_name, model_module, X_train, y_train, X_val, y_val,
        best_params, epochs, verbose=0
    )
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} seconds ({len(history.history['loss'])} epochs)")
    
    # Evaluate on all splits
    metrics_train, preds_train = evaluate_model(model, X_train, y_train)
    print(f"Train RMSE: {metrics_train['rmse']:.4f}, R²: {metrics_train['r2']:.4f}")
    
    metrics_val, preds_val = evaluate_model(model, X_val, y_val)
    print(f"Val   RMSE: {metrics_val['rmse']:.4f}, R²: {metrics_val['r2']:.4f}")
    
    metrics_test, preds_test = evaluate_model(model, X_test, y_test)
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
            'epochs_trained': len(history.history['loss']) if split_name == 'train' else None,
            **metrics
        })
    
    # Save predictions
    save_predictions(y_train, preds_train, n_engines, 'train', model_name, results_dir)
    save_predictions(y_val, preds_val, n_engines, 'val', model_name, results_dir)
    save_predictions(y_test, preds_test, n_engines, 'test', model_name, results_dir)
    
    # Save training history plot for all engine counts
    print(f"Saving training history plot...")
    save_training_history_plot(history, n_engines, model_name, results_dir)
    
    # Generate evaluation plots for selected engine counts
    if n_engines in PLOT_ENGINE_COUNTS:
        print(f"Generating evaluation plots...")
        generate_plots(y_test, preds_test, n_engines, 'test', model_name, results_dir)
    
    # Save model
    model_save_path = results_dir / 'models' / f'model_{n_engines}engines.keras'
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved: {model_save_path.name}")
    
    return metrics_list

def main(models_to_run=None, engine_counts_to_run=None, epochs=100):
    print("\n" + "="*40)
    print("DEEP LEARNING MODELS - BASELINE EXPERIMENTS (ENGINE-BASED)")
    print("="*40)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup GPU
    print("\n" + "="*40)
    print("GPU CONFIGURATION")
    print("="*40)
    setup_gpu()
    
    # Determine which models and engine counts to run
    if models_to_run is None:
        models_to_run = list(MODELS.keys())
    if engine_counts_to_run is None:
        engine_counts_to_run = ENGINE_COUNTS_ALL
    
    print("\n" + "="*40)
    print("EXPERIMENT CONFIGURATION")
    print("="*40)
    print(f"Models to run: {', '.join(models_to_run)}")
    print(f"Engine counts: {engine_counts_to_run}")
    print(f"Epochs per experiment: {epochs} (with early stopping patience=10)")
    
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
        model_module = model_config['module']
        
        print(f"\n{'='*40}")
        print(f"MODEL: {model_module.MODEL_INFO['name']} ({model_key})")
        print(f"{'='*40}")
        
        # Load best hyperparameters
        try:
            best_params = load_best_hyperparameters(model_key)
            print(f"Loaded best hyperparameters:")
            for key, value in best_params.items():
                print(f"{key}: {value}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print(f"Please run hyperparameter tuning first: python tune_dl_models.py --models {model_key}")
            continue
        
        model_results = []
        
        # Run experiments for each engine count
        for n_engines in engine_counts_to_run:
            try:
                metrics_list = run_single_experiment(model_key, model_config, data, n_engines, best_params, epochs)
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
        output_file = combined_dir / 'dl_all_results.csv'
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
            'cmapss_score': 'mean',
            'auc_rmse': 'mean'
        }).round(4)
        print(summary)
    
    print("\n" + "="*40)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*40)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {RESULTS_BASE_PATH}/Phase1_Baseline/FD001/")
    print()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run deep learning model baseline experiments')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()), 
                       default=None, help='Models to run (default: all)')
    parser.add_argument('--engines', nargs='+', type=int,
                       default=None, help='Engine counts to run (default: all)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train (default: 100)')
    
    args = parser.parse_args()
    
    main(models_to_run=args.models, engine_counts_to_run=args.engines, epochs=args.epochs)
