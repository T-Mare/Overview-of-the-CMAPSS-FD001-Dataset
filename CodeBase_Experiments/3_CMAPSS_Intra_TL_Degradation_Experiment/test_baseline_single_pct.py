import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import time
import pandas as pd
import sys
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Utilities.Plots_Metrics import rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc
from Utilities import config

# ==========
# CONFIGURATION
# ==========

DATASET = 'FD001'

# 14 Correlation_FS features (same as Phase 2)
SELECTED_FEATURES = ['s11', 's4', 's12', 's7', 's15', 's21', 's20', 's17', 
                     's2', 's3', 's8', 's13', 's9', 's14']

FEATURE_MAP = {
    's2': 0, 's3': 1, 's4': 2, 's7': 3, 's8': 4, 's9': 5,
    's11': 6, 's12': 7, 's13': 8, 's14': 9, 's15': 10,
    's17': 11, 's20': 12, 's21': 13
}

# Phase 1 best LSTM hyperparameters
LSTM_PARAMS = {
    'units_layer1': 128,
    'num_layers': 1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'patience': 5
}

# Paths
WINDOWED_DATA_PATH = Path(config.WINDOWED_DATA_PATH)
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / 'Baseline_Engine_Based'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = config.RANDOM_SEED
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ==========
# DATA LOADING
# ==========

def extract_selected_features(X):
    indices = [FEATURE_MAP[feat] for feat in SELECTED_FEATURES]
    return X[:, :, indices]

def load_data(data_percentage):
    print(f"\nLoading {DATASET} data ({data_percentage}%)...")
    
    # Load full data
    X_train_full = np.load(WINDOWED_DATA_PATH / f'{DATASET}_X_train_windowed.npy')
    y_train_full = np.load(WINDOWED_DATA_PATH / f'{DATASET}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{DATASET}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{DATASET}_y_val_windowed.npy')
    X_test = np.load(WINDOWED_DATA_PATH / f'{DATASET}_X_test_windowed.npy')
    y_test = np.load(WINDOWED_DATA_PATH / f'{DATASET}_y_test_windowed.npy')
    
    # Load engine IDs to sample by engine
    train_ids_path = WINDOWED_DATA_PATH / f'{DATASET}_train_ids_windowed.csv'
    train_ids = pd.read_csv(train_ids_path)
    
    # Get unique engines
    unique_engines = train_ids['engine'].unique()
    n_engines = len(unique_engines)
    
    print(f"Full training set: {len(X_train_full)} samples from {n_engines} engines")
    
    # Sample engines based on percentage
    n_engines_to_sample = max(int(n_engines * (data_percentage / 100)), 1)
    n_engines_to_sample = min(n_engines_to_sample, n_engines)
    
    print(f"Sampling {data_percentage}%: {n_engines_to_sample} engines (out of {n_engines})")
    
    # Randomly select engines
    selected_engines = np.random.choice(unique_engines, size=n_engines_to_sample, replace=False)
    selected_engines = sorted(selected_engines)
    
    print(f"Selected engines: {selected_engines}")
    
    # Get indices of samples from selected engines
    sample_mask = train_ids['engine'].isin(selected_engines).values
    sample_indices = np.where(sample_mask)[0]
    
    # Extract samples
    X_train = X_train_full[sample_indices]
    y_train = y_train_full[sample_indices]
    
    print(f"Train: {X_train.shape[0]} samples from {n_engines_to_sample} engines")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    
    # Extract 14 Correlation_FS features
    X_train = extract_selected_features(X_train)
    X_val = extract_selected_features(X_val)
    X_test = extract_selected_features(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, n_engines_to_sample

# ==========
# MODEL
# ==========

def build_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(LSTM_PARAMS['units_layer1'], 
                         return_sequences=False, 
                         input_shape=input_shape),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_PARAMS['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ==========
# TRAINING
# ==========

def train_model(model, X_train, y_train, X_val, y_val):
    print("\nTraining LSTM...")
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=LSTM_PARAMS['patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_PARAMS['epochs'],
        batch_size=LSTM_PARAMS['batch_size'],
        callbacks=[early_stop],
        verbose=1
    )
    training_time = time.time() - start_time
    
    epochs_trained = len(history.history['loss'])
    best_epoch = int(np.argmin(history.history['val_loss']) + 1)
    
    print(f"\n Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Epochs trained: {epochs_trained}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
    
    return history, training_time

# ==========
# EVALUATION
# ==========

def evaluate_model(model, X_val, y_val, X_test, y_test):
    print("\nEvaluating model...")
    
    metrics = {}
    
    # Validation predictions
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    metrics['val_rmse'] = float(rmse(y_val, y_val_pred))
    metrics['val_mae'] = float(mae(y_val, y_val_pred))
    metrics['val_r2'] = float(r2(y_val, y_val_pred))
    metrics['val_cmapss'] = float(cmapss_score(y_val, y_val_pred, reduction='sum'))
    
    rmse_bins_val, auc_rmse_val = rmse_by_bins_with_auc(y_val, y_val_pred, config.RUL_BINS)
    metrics['val_auc_rmse'] = float(auc_rmse_val) if auc_rmse_val is not None else None
    
    # Test predictions
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    metrics['test_rmse'] = float(rmse(y_test, y_test_pred))
    metrics['test_mae'] = float(mae(y_test, y_test_pred))
    metrics['test_r2'] = float(r2(y_test, y_test_pred))
    metrics['test_cmapss'] = float(cmapss_score(y_test, y_test_pred, reduction='sum'))
    
    rmse_bins_test, auc_rmse_test = rmse_by_bins_with_auc(y_test, y_test_pred, config.RUL_BINS)
    metrics['test_auc_rmse'] = float(auc_rmse_test) if auc_rmse_test is not None else None
    
    print(f"Validation - RMSE: {metrics['val_rmse']:.4f}, MAE: {metrics['val_mae']:.4f}, R²: {metrics['val_r2']:.4f}")
    print(f"Test       - RMSE: {metrics['test_rmse']:.4f}, MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.4f}")
    
    return metrics

# ==========
# MAIN
# ==========

def main(data_percentage):
    print("\n" + "="*40)
    print(f"BASELINE LSTM @ {data_percentage}% (ENGINE-BASED SAMPLING)")
    print("="*40)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, n_engines = load_data(data_percentage)
    
    # Build model
    print(f"\nBuilding LSTM model...")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    print(f"Total params: {model.count_params():,}")
    
    # Train
    history, training_time = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
    
    # Add metadata
    metrics['data_pct'] = data_percentage
    metrics['n_engines'] = n_engines
    metrics['n_train_samples'] = len(X_train)
    metrics['training_time_sec'] = training_time
    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['best_epoch'] = int(np.argmin(history.history['val_loss']) + 1)
    
    # Save results
    result_file = OUTPUT_PATH / f'baseline_lstm_{data_percentage}pct.json'
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Results saved: {result_file}")
    print("="*40 + "\n")
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test baseline LSTM at single data percentage')
    parser.add_argument('--data_pct', type=int, required=True, help='Data percentage')
    
    args = parser.parse_args()
    
    main(args.data_pct)

