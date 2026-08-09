import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import time
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Utilities.Plots_Metrics import rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc
from Utilities import config

print("\n" + "="*40)
print("BASELINE LSTM AT VERY LOW DATA (1-5%)")
print("="*40)
print("Goal: Understand baseline performance at extreme data scarcity")
print("These are the regimes where Transfer Learning should help most!")
print("="*40 + "\n")

# ==========
# CONFIGURATION
# ==========

DATASET = 'FD001'
DATA_PERCENTAGES = [1, 2, 3, 4, 5]  # Very low data regimes

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
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / 'Baseline_Very_Low_Data'
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
    n_engines_to_sample = max(int(n_engines * (data_percentage / 100)), 1)  # At least 1 engine
    n_engines_to_sample = min(n_engines_to_sample, n_engines)  # Don't exceed total
    
    print(f"Sampling {data_percentage}%: {n_engines_to_sample} engines (out of {n_engines})")
    
    # Randomly select engines
    selected_engines = np.random.choice(unique_engines, size=n_engines_to_sample, replace=False)
    selected_engines = sorted(selected_engines)  # Sort for reproducibility
    
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
# MAIN WORKFLOW
# ==========

def run_experiment(data_percentage):
    print("\n" + "="*40)
    print(f"EXPERIMENT: {data_percentage}% DATA (BY ENGINE)")
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
    
    print(f"\n Results saved: {result_file.name}")
    print("="*40)
    
    return metrics

# ==========
# RUN ALL EXPERIMENTS
# ==========

def main():
    print("\nStarting baseline LSTM experiments at very low data (1-5%)...")
    print(f"Total experiments: {len(DATA_PERCENTAGES)}")
    print(f"Estimated time: ~{len(DATA_PERCENTAGES) * 3} minutes\n")
    
    all_results = []
    
    for data_pct in DATA_PERCENTAGES:
        try:
            metrics = run_experiment(data_pct)
            all_results.append(metrics)
        except Exception as e:
            print(f"\n Error at {data_pct}%: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('data_pct')
        
        results_path = OUTPUT_PATH / 'baseline_very_low_data_summary.csv'
        results_df.to_csv(results_path, index=False)
        
        print("\n" + "="*40)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*40)
        print(f"\n Combined results saved: {results_path}")
        
        # Display summary
        print("\n" + "="*40)
        print("SUMMARY: BASELINE LSTM AT VERY LOW DATA")
        print("="*40)
        
        summary = results_df[['data_pct', 'n_engines', 'n_train_samples', 'val_rmse', 'test_rmse', 
                              'test_mae', 'test_r2', 'test_auc_rmse']].copy()
        summary.columns = ['Data %', 'Engines', 'Train Samples', 'Val RMSE', 'Test RMSE', 
                          'Test MAE', 'Test R²', 'Test AUC']
        
        for col in ['Val RMSE', 'Test RMSE', 'Test MAE', 'Test R²', 'Test AUC']:
            summary[col] = summary[col].round(3)
        
        print("\n" + summary.to_string(index=False))
        
        print("\n" + "-"*80)
        print("KEY INSIGHTS:")
        print("-"*80)
        print(f"Worst performance (highest RMSE): {results_df['test_rmse'].max():.3f} @ {results_df.loc[results_df['test_rmse'].idxmax(), 'data_pct']:.0f}%")
        print(f"Best performance (lowest RMSE):  {results_df['test_rmse'].min():.3f} @ {results_df.loc[results_df['test_rmse'].idxmin(), 'data_pct']:.0f}%")
        print(f"Average test RMSE: {results_df['test_rmse'].mean():.3f}")
        
        print("\n" + "-"*80)
        print("IMPLICATIONS FOR TRANSFER LEARNING:")
        print("-"*80)
        print("If baseline performs very poorly at 1-5%, these are prime candidates")
        print("for Transfer Learning to show significant improvement!")
        print("Target: Reduce RMSE by 20-30% at these low data regimes.")
        print("="*40)

if __name__ == '__main__':
    main()

