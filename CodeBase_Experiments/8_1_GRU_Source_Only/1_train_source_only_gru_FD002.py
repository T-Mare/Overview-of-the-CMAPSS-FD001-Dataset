import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import sys

# Add parent directories to path (the root of the project)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

#import the utilities plots and metrics functions
from Utilities.Plots_Metrics import rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc
from Utilities import config

# ==========
# CONFIGURATION
# ==========

# Default datasets (can use fd003 as source CLI argument aggument --source fd003)
SOURCE_DATASET = 'FD002'
TARGET_DATASET = 'FD001'

# The 14 features selected in Phase2 (Correlation_FS)
SELECTED_FEATURES = ['s11', 's4', 's12', 's7', 's15', 's21', 's20', 's17', 
                     's2', 's3', 's8', 's13', 's9', 's14']

# Mapping: to tell the script which column index in the Numpy Array is associated to whjioch sensor
FEATURE_MAP = {
    's2': 0, 's3': 1, 's4': 2, 's7': 3, 's8': 4, 's9': 5,
    's11': 6, 's12': 7, 's13': 8, 's14': 9, 's15': 10,
    's17': 11, 's20': 12, 's21': 13
}

# Phase 1 GRU best hyperparameters
GRU_HYPERPARAMS = {
    'units': 256,
    'dropout': 0.3,
    'learning_rate': 0.0005,
    'batch_size': 32,
    'epochs': 50,
    'patience': 5
}

# Paths
WINDOWED_DATA_PATH = Path(config.WINDOWED_DATA_PATH)
OUTPUT_DIR = SCRIPT_DIR / f"outputs_source_only_{SOURCE_DATASET}_to_{TARGET_DATASET}"

# Random seed
RANDOM_SEED = config.RANDOM_SEED #this is from the config.py (42)

def set_reproducibility(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def extract_selected_features(X, selected_features=SELECTED_FEATURES):
    indices = [FEATURE_MAP[feat] for feat in selected_features]
    return X[:, :, indices]

# ==========
# DATA LOADING
# ==========

def load_dataset(dataset_name):
    print(f"\n {dataset_name} is loading...")
    
    X_train = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_X_train_windowed.npy')
    y_train = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_y_val_windowed.npy')
    X_test = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_X_test_windowed.npy')
    y_test = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_y_test_windowed.npy')
    
    # Extract 14 Correlation_FS features
    X_train = extract_selected_features(X_train)
    X_val = extract_selected_features(X_val)
    X_test = extract_selected_features(X_test)
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    print(f"Shape: {X_train.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

# ==========
#  BUILDING the GRU Source only Model
# ==========

def build_gru_model(input_shape=(30, 14), hyperparams=None):
  
    hyperparams = GRU_HYPERPARAMS ## can use a an if statement to include the arguments
    
    model = keras.Sequential(name='GRU_Source_Only') #created the name of the model
    # Add the GRU layer to the model
    model.add(keras.layers.GRU(units=hyperparams['units'], dropout=hyperparams['dropout'], return_sequences=False, input_shape=input_shape, name='gru_layer'))
    
    # Add the output layer to the model
    model.add(keras.layers.Dense(1, activation='linear', name='rul_output'))

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model
# ==========
# TRAINING
# ==========

def train_source_model(source_data, hyperparams):
    print(f"\nTraining the GRU Source only Model on {SOURCE_DATASET}...")
    
    # Build model
    model = build_gru_model(input_shape=(30, 14), hyperparams=hyperparams)
    print("\nModel architecture:")
    model.summary()
    
    # Add the Early Stopping callback to the model
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=hyperparams['patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    # Train
    print(f"\nTraining the GRU Source only Model on {SOURCE_DATASET} (full dataset)...")
    start_time = time.time() #capture the start time of the training
    
    history = model.fit(
        source_data['X_train'], source_data['y_train'],
        validation_data=(source_data['X_val'], source_data['y_val']),
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        callbacks=[early_stop],
        verbose=1
    )
    
    training_time = time.time() - start_time
    epochs_trained = len(history.history['loss'])
    
    #print the training time, the number of epochs trained and the best validation loss
    print(f"\nTraining time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Epochs trained: {epochs_trained}")
    print(f"Best val_loss: {min(history.history['val_loss']):.6f}")

    return model, history, training_time #return the model, the training history and the training time

# ==========
# EVALUATING the GRU Source only Model
# ==========

def evaluate_split(model, X, y, split_name=''):
    y_pred = model.predict(X, verbose=0).flatten()
    
    metrics = { #calculate the metrics
        'rmse': float(rmse(y, y_pred)),
        'mae': float(mae(y, y_pred)),
        'r2': float(r2(y, y_pred)),
        'cmapss': float(cmapss_score(y, y_pred, reduction='sum'))
    }
    
    rmse_bins, auc_rmse = rmse_by_bins_with_auc(y, y_pred, config.RUL_BINS) #calculate the AUC-RMSE
    metrics['auc_rmse'] = float(auc_rmse) if auc_rmse is not None else None
    
    if split_name:
        print(f"{split_name:8s} - RMSE: {metrics['rmse']:.4f}, "
              f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return metrics, y_pred

def evaluate_model(model, source_data, target_data, perform_test=False):
    print(f"\nEvaluating the GRU Source only Model on {SOURCE_DATASET} and {TARGET_DATASET}...")
    
    metrics = {}
    
    # Source domain (FD002) - for reference
    print(f"\n{SOURCE_DATASET} (source domain):")
    source_val_metrics, _ = evaluate_split(
        model, source_data['X_val'], source_data['y_val'], 'Val'
    )
    
    if perform_test:
        source_test_metrics, _ = evaluate_split(
            model, source_data['X_test'], source_data['y_test'], 'Test'
        )
    else:
        source_test_metrics = None
    
    metrics['source_val_rmse'] = source_val_metrics['rmse']
    metrics['source_val_mae'] = source_val_metrics['mae']
    metrics['source_val_r2'] = source_val_metrics['r2']
    metrics['source_val_cmapss'] = source_val_metrics['cmapss']
    metrics['source_val_auc_rmse'] = source_val_metrics['auc_rmse']
    
    if perform_test:
        metrics['source_test_rmse'] = source_test_metrics['rmse']
        metrics['source_test_mae'] = source_test_metrics['mae']
        metrics['source_test_r2'] = source_test_metrics['r2']
        metrics['source_test_cmapss'] = source_test_metrics['cmapss']
        metrics['source_test_auc_rmse'] = source_test_metrics['auc_rmse']
    
    # Target domain (FD001) - main evaluation (The target lables are used only after the training for evaluation purely)
    print(f"\n{TARGET_DATASET} (target domain, zero-shot):")
    target_val_metrics, target_val_pred = evaluate_split(
        model, target_data['X_val'], target_data['y_val'], 'Val'
    )
    
    if perform_test: # only evaluates test set if explicitly called to prevent the evaluation of the test set during the training
        target_test_metrics, target_test_pred = evaluate_split(
            model, target_data['X_test'], target_data['y_test'], 'Test'
        )
    else:
        target_test_metrics = None
        target_test_pred = None
    
    metrics['target_val_rmse'] = target_val_metrics['rmse']
    metrics['target_val_mae'] = target_val_metrics['mae']
    metrics['target_val_r2'] = target_val_metrics['r2']
    metrics['target_val_cmapss'] = target_val_metrics['cmapss']
    metrics['target_val_auc_rmse'] = target_val_metrics['auc_rmse']
    
    if perform_test:
        metrics['target_test_rmse'] = target_test_metrics['rmse']
        metrics['target_test_mae'] = target_test_metrics['mae']
        metrics['target_test_r2'] = target_test_metrics['r2']
        metrics['target_test_cmapss'] = target_test_metrics['cmapss']
        metrics['target_test_auc_rmse'] = target_test_metrics['auc_rmse']
    
    print(f"\nKey metric (Source-Only Baseline):")
    print(f"{TARGET_DATASET} Val RMSE:  {metrics['target_val_rmse']:.4f}")
    if perform_test:
        print(f"{TARGET_DATASET} Test RMSE: {metrics['target_test_rmse']:.4f}") #print the test RMSE if the test set is evaluated
    else:
        print("(Test set not evaluated - use --perform-test flag to evaluate the test set)") #print a message if the test set is not evaluated
    
    return metrics, target_val_pred, target_test_pred

# ==========
# SAVING
# ==========

def save_results(model, history, metrics, training_time, hyperparams,
                 target_val_pred, target_test_pred, target_data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = OUTPUT_DIR / f'gru_source_only_{SOURCE_DATASET}.keras'
    model.save(model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save hyperparameters
    params_path = OUTPUT_DIR / 'hyperparameters.json'
    with open(params_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Save training history
    history_path = OUTPUT_DIR / 'training_history.json'
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'epochs_trained': len(history.history['loss']),
        'training_time_sec': float(training_time)
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save metrics
    metrics_path = OUTPUT_DIR / 'metrics.json'
    metrics_full = {
        'method': 'Source_Only_GRU',
        'architecture': 'GRU (256 units, dropout 0.3)',
        'source_dataset': SOURCE_DATASET,
        'target_dataset': TARGET_DATASET,
        'training_time_sec': float(training_time),
        'n_features': len(SELECTED_FEATURES),
        'feature_selection': 'Correlation_FS',
        **metrics,
        **hyperparams
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_full, f, indent=2)
    
    # Save target domain predictions
    val_pred_df = pd.DataFrame({
        'y_true': target_data['y_val'],
        'y_pred': target_val_pred,
        'error': target_val_pred - target_data['y_val']
    })
    val_pred_df.to_csv(OUTPUT_DIR / 'target_val_predictions.csv', index=False)
    
    if target_test_pred is not None:
        test_pred_df = pd.DataFrame({
            'y_true': target_data['y_test'],
            'y_pred': target_test_pred,
            'error': target_test_pred - target_data['y_test']
        })
        test_pred_df.to_csv(OUTPUT_DIR / 'target_test_predictions.csv', index=False)
    
    print(f"All results saved to: {OUTPUT_DIR}")
    
    return OUTPUT_DIR

# ==========
# MAIN

# ==========

def parse_args(): 
    parser = argparse.ArgumentParser(
        description="Train the GRU Source only Model on the source dataset and evaluate it on the target dataset"
    )
    parser.add_argument(
        "--source", type=str, default=SOURCE_DATASET,
        help=f"Source dataset to train the model on (default: {SOURCE_DATASET})"
    )
    parser.add_argument(
        "--target", type=str, default=TARGET_DATASET,
        help=f"Target dataset to evaluate the model on (default: {TARGET_DATASET})"
    )
    parser.add_argument(
        "--epochs", type=int, default=GRU_HYPERPARAMS['epochs'],
        help=f"Maximum number of training epochs (default: {GRU_HYPERPARAMS['epochs']})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=GRU_HYPERPARAMS['batch_size'],
        help=f"Batch size (default: {GRU_HYPERPARAMS['batch_size']})"
    )
    parser.add_argument(
        "--lr", type=float, default=GRU_HYPERPARAMS['learning_rate'],
        help=f"Learning rate (default: {GRU_HYPERPARAMS['learning_rate']})"
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help="Random seed"
    )
    parser.add_argument(
        "--perform-test", action="store_true",
        help="Evaluate on test set during training run"
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Load saved model and evaluate test set only (no retraining)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    set_reproducibility(args.seed)
    keras.backend.clear_session()
    
    # set source/target datasets from CLI
    global SOURCE_DATASET, TARGET_DATASET, OUTPUT_DIR
    SOURCE_DATASET = args.source
    TARGET_DATASET = args.target
    OUTPUT_DIR = SCRIPT_DIR / f"outputs_source_only_{SOURCE_DATASET}_to_{TARGET_DATASET}"
    
    # update hyperparameters from CLI arguments
    hyperparams = dict(GRU_HYPERPARAMS)
    hyperparams['epochs'] = args.epochs
    hyperparams['batch_size'] = args.batch_size
    hyperparams['learning_rate'] = args.lr
    
    # ========================================================
    # TEST-ONLY MODE: Load saved model and evaluate test set
    # ========================================================
    if args.test_only:
        print("\n" + "="*40)
        print(f"TEST-ONLY MODE: {SOURCE_DATASET} -> {TARGET_DATASET}")
        print("="*40)
        
        # load the saved model
        model_path = OUTPUT_DIR / f'gru_source_only_{SOURCE_DATASET}.keras'
        if not model_path.exists():
            raise FileNotFoundError(
                f"No saved model found at {model_path}\n"
                f"Train the model first without --test-only flag."
            )
        
        model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
        
        # load data for test evaluation
        source_data = load_dataset(SOURCE_DATASET)
        target_data = load_dataset(TARGET_DATASET)
        
        # evaluate test sets
        print(f"\nEvaluating test sets...")
        source_test_metrics, _ = evaluate_split(model, source_data['X_test'], source_data['y_test'], 'Test')
        target_test_metrics, target_test_pred = evaluate_split(model, target_data['X_test'], target_data['y_test'], 'Test')
        
        # load existing metrics and update
        metrics_path = OUTPUT_DIR / 'metrics.json'
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # add test metrics
        metrics.update({
            'source_test_rmse': source_test_metrics['rmse'],
            'source_test_mae': source_test_metrics['mae'],
            'source_test_r2': source_test_metrics['r2'],
            'source_test_cmapss': source_test_metrics['cmapss'],
            'source_test_auc_rmse': source_test_metrics['auc_rmse'],
            'target_test_rmse': target_test_metrics['rmse'],
            'target_test_mae': target_test_metrics['mae'],
            'target_test_r2': target_test_metrics['r2'],
            'target_test_cmapss': target_test_metrics['cmapss'],
            'target_test_auc_rmse': target_test_metrics['auc_rmse']
        })
        
        # save updated metrics and predictions
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        test_pred_df = pd.DataFrame({
            'y_true': target_data['y_test'],
            'y_pred': target_test_pred,
            'error': target_test_pred - target_data['y_test']
        })
        test_pred_df.to_csv(OUTPUT_DIR / 'target_test_predictions.csv', index=False)
        
        print(f"\n Test evaluation complete")
        print(f"{TARGET_DATASET} Test RMSE: {metrics['target_test_rmse']:.4f}")
        print("\nthe test evaluation is completed")
        return
    
    # ========================================================
    # TRAINING MODE: Train source-only model
    # ========================================================
    print("\nthe training is starting")
    print(f"SOURCE-ONLY GRU BASELINE: {SOURCE_DATASET} -> {TARGET_DATASET}")
    print("="*40)
    print(f"Architecture: GRU ({hyperparams['units']} units, dropout {hyperparams['dropout']})")
    print(f"Features: {len(SELECTED_FEATURES)} (Correlation_FS)")
    print(f"Test evaluation: {'Yes' if args.perform_test else 'No (use --test-only later)'}")
    print("\n######################")
    
    # load datasets
    source_data = load_dataset(SOURCE_DATASET)
    target_data = load_dataset(TARGET_DATASET)
    
    # train on source domain
    model, history, training_time = train_source_model(source_data, hyperparams)
    
    # evaluate on both domains
    metrics, target_val_pred, target_test_pred = evaluate_model(
        model, source_data, target_data, perform_test=args.perform_test
    )
    
    # save all results
    save_results(
        model, history, metrics, training_time, hyperparams,
        target_val_pred, target_test_pred, target_data
    )
    
    print("\nthe training is completed######################")
    print("")
    if args.perform_test:
        print(f"{TARGET_DATASET} Test RMSE: {metrics['target_test_rmse']:.4f}")
    else:
        print(f"Use --test-only to evaluate test set later") #print a message if the test set is not evaluated
    print("\nthe training is completed######################")

if __name__ == '__main__':
    main()
