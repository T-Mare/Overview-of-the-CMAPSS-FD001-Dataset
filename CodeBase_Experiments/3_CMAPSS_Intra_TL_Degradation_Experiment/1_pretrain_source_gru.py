import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import time
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from Utilities.Plots_Metrics import rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc
from Utilities import config

# ==========
# CONFIGURATION
# ==========

# Source datasets
# Testing FD003 first (best input distribution match with FD001)
SOURCE_DATASETS = ['FD003']  # Add 'FD002' later if FD003 works well

# 14 Correlation_FS features (from Phase 2)
SELECTED_FEATURES = ['s11', 's4', 's12', 's7', 's15', 's21', 's20', 's17', 
                     's2', 's3', 's8', 's13', 's9', 's14']

# Mapping: feature name -> index in original 24-feature data
# Original features: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
FEATURE_MAP = {
    's2': 0, 's3': 1, 's4': 2, 's7': 3, 's8': 4, 's9': 5,
    's11': 6, 's12': 7, 's13': 8, 's14': 9, 's15': 10,
    's17': 11, 's20': 12, 's21': 13
}

# Phase 1 GRU best hyperparameters
GRU_HYPERPARAMS = {
    'n_gru_layers': 1,
    'units_layer1': 256,
    'units_layer2': 128,  # Not used (only 1 layer)
    'dropout_rate': 0.3,
    'recurrent_dropout': 0.0,  # GRU typically doesn't use recurrent dropout
    'learning_rate': 0.0005,
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 5
}

# Paths
WINDOWED_DATA_PATH = Path(config.WINDOWED_DATA_PATH)
OUTPUT_BASE = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / 'Pretrained_Models'

# Random seed
RANDOM_SEED = config.RANDOM_SEED

# ==========
# FEATURE EXTRACTION
# ==========

def extract_selected_features(X, selected_features=SELECTED_FEATURES):
    # Get indices of selected features
    indices = [FEATURE_MAP[feat] for feat in selected_features]
    
    # Extract features
    X_selected = X[:, :, indices]
    
    print(f"Extracted {len(selected_features)} features")
    print(f"Original shape: {X.shape}  Selected shape: {X_selected.shape}")
    
    return X_selected

# ==========
# DATA LOADING
# ==========

def load_source_data(source_dataset):
    print(f"\nLoading {source_dataset} data...")
    
    # Load windowed data (24 features)
    X_train = np.load(WINDOWED_DATA_PATH / f'{source_dataset}_X_train_windowed.npy')
    y_train = np.load(WINDOWED_DATA_PATH / f'{source_dataset}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{source_dataset}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{source_dataset}_y_val_windowed.npy')
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    
    # Extract 14 Correlation_FS features
    X_train_selected = extract_selected_features(X_train)
    X_val_selected = extract_selected_features(X_val)
    
    return X_train_selected, y_train, X_val_selected, y_val

# ==========
# MODEL BUILDING
# ==========

def build_gru_model(input_shape=(30, 14)):
    # Set random seeds
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    model = keras.Sequential(name='GRU_Pretrain')
    
    # Input layer
    model.add(keras.layers.Input(shape=input_shape, name='input'))
    
    # GRU layer (1 layer, 256 units)
    model.add(keras.layers.GRU(
        units=GRU_HYPERPARAMS['units_layer1'],
        dropout=GRU_HYPERPARAMS['dropout_rate'],
        recurrent_dropout=GRU_HYPERPARAMS['recurrent_dropout'],
        return_sequences=False,
        name='gru_1'
    ))
    
    # Output layer
    model.add(keras.layers.Dense(1, activation='linear', name='output'))
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=GRU_HYPERPARAMS['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ==========
# TRAINING
# ==========

def train_pretrained_model(model, X_train, y_train, X_val, y_val):
    print("\nTraining GRU model...")
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=GRU_HYPERPARAMS['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    # Train
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=GRU_HYPERPARAMS['epochs'],
        batch_size=GRU_HYPERPARAMS['batch_size'],
        callbacks=[early_stop],
        verbose=1
    )
    training_time = time.time() - start_time
    
    print(f"\n Training complete in {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Epochs trained: {len(history.history['loss'])}")
    print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
    
    return history, training_time

# ==========
# EVALUATION
# ==========

def evaluate_pretrained_model(model, X_val, y_val):
    print("\nEvaluating GRU model...")
    
    # Predictions
    y_pred = model.predict(X_val, verbose=0).flatten()
    
    # Calculate metrics
    metrics = {
        'val_rmse': float(rmse(y_val, y_pred)),
        'val_mae': float(mae(y_val, y_pred)),
        'val_r2': float(r2(y_val, y_pred)),
        'val_cmapss': float(cmapss_score(y_val, y_pred, reduction='sum')),
    }
    
    # AUC-RMSE
    rmse_bins, auc_rmse = rmse_by_bins_with_auc(y_val, y_pred, config.RUL_BINS)
    metrics['val_auc_rmse'] = float(auc_rmse) if auc_rmse is not None else None
    
    print(f"RMSE: {metrics['val_rmse']:.4f}")
    print(f"MAE:  {metrics['val_mae']:.4f}")
    print(f"R²:   {metrics['val_r2']:.4f}")
    
    return metrics

# ==========
# SAVING
# ==========

def save_pretrained_model(model, history, metrics, training_time, source_dataset):
    save_dir = OUTPUT_BASE / source_dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_dir / f'gru_pretrained_{source_dataset}.keras'
    model.save(model_path)
    print(f"\n Model saved: {model_path}")
    
    # Save hyperparameters
    params_path = save_dir / f'gru_hyperparameters_{source_dataset}.json'
    with open(params_path, 'w') as f:
        json.dump(GRU_HYPERPARAMS, f, indent=2)
    
    # Save training history
    history_path = save_dir / f'gru_training_history_{source_dataset}.json'
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'epochs_trained': len(history.history['loss']),
        'training_time_sec': training_time
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save evaluation metrics
    metrics_path = save_dir / f'gru_pretrain_metrics_{source_dataset}.json'
    metrics['training_time_sec'] = training_time
    metrics['source_dataset'] = source_dataset
    metrics['n_features'] = 14
    metrics['feature_selection'] = 'Correlation_FS'
    metrics['architecture'] = 'GRU'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metadata saved: {save_dir}")
    
    return save_dir

# ==========
# MAIN FUNCTION
# ==========

def pretrain_source_model(source_dataset):
    print("\n" + "="*40)
    print(f"PRE-TRAINING GRU ON {source_dataset}")
    print("="*40)
    
    # Load data
    X_train, y_train, X_val, y_val = load_source_data(source_dataset)
    
    # Build model
    print("\nBuilding GRU model...")
    model = build_gru_model(input_shape=(30, 14))
    model.summary()
    
    # Train model
    history, training_time = train_pretrained_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics = evaluate_pretrained_model(model, X_val, y_val)
    
    # Save model
    save_dir = save_pretrained_model(model, history, metrics, training_time, source_dataset)
    
    print("\n" + "="*40)
    print(f"PRE-TRAINING COMPLETE: {source_dataset}")
    print(f"Model saved: {save_dir}")
    print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
    print("="*40)
    
    return model, metrics

# ==========
# SCRIPT EXECUTION
# ==========

if __name__ == '__main__':
    print("\n" + "="*40)
    print("PHASE 4: TRANSFER LEARNING - GRU PRE-TRAINING SCRIPT")
    print("="*40)
    print(f"Source datasets: {SOURCE_DATASETS}")
    print(f"Features: 14 Correlation_FS features")
    print(f"Architecture: {GRU_HYPERPARAMS['n_gru_layers']} GRU layer, {GRU_HYPERPARAMS['units_layer1']} units")
    print(f"Output: {OUTPUT_BASE}")
    
    # Pre-train on all source datasets
    results = {}
    
    for source_dataset in SOURCE_DATASETS:
        try:
            model, metrics = pretrain_source_model(source_dataset)
            results[source_dataset] = metrics
        except Exception as e:
            print(f"\n Error pre-training {source_dataset}: {e}")
            continue
    
    # Summary
    print("\n" + "="*40)
    print("GRU PRE-TRAINING SUMMARY")
    print("="*40)
    
    for dataset, metrics in results.items():
        print(f"{dataset}: RMSE={metrics['val_rmse']:.4f}, MAE={metrics['val_mae']:.4f}, R²={metrics['val_r2']:.4f}")
    
    print("\n All pre-training complete!")
    print(f"Models saved in: {OUTPUT_BASE}")

