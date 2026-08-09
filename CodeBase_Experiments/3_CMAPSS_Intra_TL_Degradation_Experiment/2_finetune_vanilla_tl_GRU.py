import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import time
import pandas as pd
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from Utilities.Plots_Metrics import rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc
from Utilities import config

# ==========
# CONFIGURATION
# ==========

# Source and target datasets
SOURCE_DATASETS = ['FD002', 'FD003']  # Multiple source options for comparison
TARGET_DATASET = 'FD001'

# Engine counts to test (engine-based sampling)
ENGINE_COUNTS_ALL = config.ENGINE_COUNTS_ALL  # [80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# Phase 2 Baseline path (GRU + Correlation_FS, no transfer learning)
PHASE2_BASELINE_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase2_Feature_Selection' / 'FD001' / 'Correlation_FS' / 'GRU' / 'GRU_metrics_summary.csv'

# 14 Correlation_FS features (same as pre-training)
SELECTED_FEATURES = ['s11', 's4', 's12', 's7', 's15', 's21', 's20', 's17', 
                     's2', 's3', 's8', 's13', 's9', 's14']

FEATURE_MAP = {
    's2': 0, 's3': 1, 's4': 2, 's7': 3, 's8': 4, 's9': 5,
    's11': 6, 's12': 7, 's13': 8, 's14': 9, 's15': 10,
    's17': 11, 's20': 12, 's21': 13
}

# Fine-tuning hyperparameters
FINETUNE_PARAMS = {
    'learning_rate': 0.0001,  # 1/10 of pre-training LR (standard TL practice)
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 5
}

# Paths
WINDOWED_DATA_PATH = Path(config.WINDOWED_DATA_PATH)
PRETRAINED_MODELS_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / 'Pretrained_Models'
OUTPUT_BASE = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / TARGET_DATASET

# Random seed
RANDOM_SEED = config.RANDOM_SEED

# ==========
# FEATURE EXTRACTION
# ==========

def extract_selected_features(X, selected_features=SELECTED_FEATURES):
    indices = [FEATURE_MAP[feat] for feat in selected_features]
    return X[:, :, indices]

# ==========
# BASELINE LOADING
# ==========

def load_phase2_baseline():
    if not PHASE2_BASELINE_PATH.exists():
        print(f"Phase 2 baseline not found: {PHASE2_BASELINE_PATH}")
        print(f"Will skip TL gain comparison")
        return None
    
    baseline_df = pd.read_csv(PHASE2_BASELINE_PATH)
    # Filter for GRU + Correlation_FS
    baseline_df = baseline_df[baseline_df['fs_method'] == 'Correlation_FS']
    
    print(f"Loaded Phase 2 baseline (GRU + Correlation_FS)")
    return baseline_df

# ==========
# DATA LOADING
# ==========

def load_target_data(n_engines):
    print(f"\nLoading {TARGET_DATASET} data ({n_engines} engines)...")
    
    # Load full windowed data
    X_train_full = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_train_windowed.npy')
    y_train_full = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_val_windowed.npy')
    X_test = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_test_windowed.npy')
    y_test = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_test_windowed.npy')
    
    # Sample training data by engine (not random sampling!)
    if n_engines < 80:
        # Load engine IDs to enable engine-based sampling
        train_ids_path = WINDOWED_DATA_PATH / f'{TARGET_DATASET}_train_ids_windowed.csv'
        train_ids = pd.read_csv(train_ids_path)
        
        # Get unique engines and sample n_engines
        unique_engines = train_ids['engine'].unique()
        total_engines = len(unique_engines)
        
        np.random.seed(RANDOM_SEED)
        n_engines_to_sample = min(n_engines, total_engines)
        selected_engines = np.random.choice(unique_engines, size=n_engines_to_sample, replace=False)
        selected_engines = sorted(selected_engines)
        
        # Get indices of all samples from selected engines
        sample_mask = train_ids['engine'].isin(selected_engines).values
        indices = np.where(sample_mask)[0]
        
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]
        
        print(f"Sampled {n_engines_to_sample} engines (out of {total_engines})")
    else:
        X_train = X_train_full
        y_train = y_train_full
    
    print(f"Train: {X_train.shape[0]} samples ({n_engines} engines)")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    
    # Extract 14 Correlation_FS features
    X_train = extract_selected_features(X_train)
    X_val = extract_selected_features(X_val)
    X_test = extract_selected_features(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==========
# MODEL LOADING
# ==========

def load_pretrained_model(source_dataset):
    model_path = PRETRAINED_MODELS_PATH / source_dataset / f'gru_pretrained_{source_dataset}.keras'
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Pre-trained GRU model not found: {model_path}\n"
            f"Run 1_pretrain_source_gru.py first!"
        )
    
    print(f"\nLoading pre-trained GRU model from {source_dataset}...")
    model = keras.models.load_model(model_path)
    print(f"Model loaded: {model_path.name}")
    print(f"Parameters: {model.count_params():,}")
    
    return model

def load_pretrained_metrics(source_dataset):
    metrics_path = PRETRAINED_MODELS_PATH / source_dataset / f'pretrain_metrics_{source_dataset}.json'
    
    if not metrics_path.exists():
        print(f"Pre-training metrics not found: {metrics_path.name}")
        return {}
    
    with open(metrics_path, 'r') as f:
        pretrain_metrics = json.load(f)
    
    # Prefix keys with 'pretrain_' to avoid confusion with fine-tuning metrics
    return {f'pretrain_{k}': v for k, v in pretrain_metrics.items()}

# ==========
# FINE-TUNING
# ==========

def compile_for_finetuning(model):
    print(f"\nCompiling for fine-tuning (LR={FINETUNE_PARAMS['learning_rate']})...")
    
    # All layers trainable (vanilla fine-tuning)
    for layer in model.layers:
        layer.trainable = True
    
    # Compile with lower learning rate
    optimizer = keras.optimizers.Adam(learning_rate=FINETUNE_PARAMS['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print(f"All layers trainable")
    print(f"Learning rate: {FINETUNE_PARAMS['learning_rate']}")
    
    return model

def finetune_model(model, X_train, y_train, X_val, y_val):
    print("\nFine-tuning GRU model on target data...")
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=FINETUNE_PARAMS['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    # Fine-tune
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=FINETUNE_PARAMS['epochs'],
        batch_size=FINETUNE_PARAMS['batch_size'],
        callbacks=[early_stop],
        verbose=1
    )
    finetune_time = time.time() - start_time
    
    print(f"\n Fine-tuning complete in {finetune_time:.1f}s ({finetune_time/60:.1f} min)")
    print(f"Epochs trained: {len(history.history['loss'])}")
    print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
    
    return history, finetune_time

# ==========
# EVALUATION
# ==========

def evaluate_model(model, X_val, y_val, X_test, y_test, n_engines, baseline_df=None):
    print("\nEvaluating fine-tuned GRU model...")
    
    metrics = {}
    predictions = {}
    
    # Validation predictions
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    predictions['y_val_true'] = y_val
    predictions['y_val_pred'] = y_val_pred
    
    metrics['val_rmse'] = float(rmse(y_val, y_val_pred))
    metrics['val_mae'] = float(mae(y_val, y_val_pred))
    metrics['val_r2'] = float(r2(y_val, y_val_pred))
    metrics['val_cmapss'] = float(cmapss_score(y_val, y_val_pred, reduction='sum'))
    
    rmse_bins_val, auc_rmse_val = rmse_by_bins_with_auc(y_val, y_val_pred, config.RUL_BINS)
    metrics['val_auc_rmse'] = float(auc_rmse_val) if auc_rmse_val is not None else None
    
    # Test predictions
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    predictions['y_test_true'] = y_test
    predictions['y_test_pred'] = y_test_pred
    
    metrics['test_rmse'] = float(rmse(y_test, y_test_pred))
    metrics['test_mae'] = float(mae(y_test, y_test_pred))
    metrics['test_r2'] = float(r2(y_test, y_test_pred))
    metrics['test_cmapss'] = float(cmapss_score(y_test, y_test_pred, reduction='sum'))
    
    rmse_bins_test, auc_rmse_test = rmse_by_bins_with_auc(y_test, y_test_pred, config.RUL_BINS)
    metrics['test_auc_rmse'] = float(auc_rmse_test) if auc_rmse_test is not None else None
    
    print(f"Validation - RMSE: {metrics['val_rmse']:.4f}, MAE: {metrics['val_mae']:.4f}, R²: {metrics['val_r2']:.4f}, AUC-RMSE: {metrics['val_auc_rmse']:.4f}")
    print(f"Test       - RMSE: {metrics['test_rmse']:.4f}, MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.4f}, AUC-RMSE: {metrics['test_auc_rmse']:.4f}")
    
    # Calculate TL gain vs Phase 2 baseline (if available)
    if baseline_df is not None:
        baseline_row = baseline_df[baseline_df['n_engines'] == n_engines]
        
        if not baseline_row.empty:
            baseline_val_rmse = float(baseline_row['val_rmse'].values[0])
            baseline_test_rmse = float(baseline_row['test_rmse'].values[0])
            baseline_val_auc = float(baseline_row['val_auc_rmse'].values[0])
            baseline_test_auc = float(baseline_row['test_auc_rmse'].values[0])
            
            # Calculate gains
            val_rmse_gain = ((baseline_val_rmse - metrics['val_rmse']) / baseline_val_rmse) * 100
            test_rmse_gain = ((baseline_test_rmse - metrics['test_rmse']) / baseline_test_rmse) * 100
            val_auc_gain = ((baseline_val_auc - metrics['val_auc_rmse']) / baseline_val_auc) * 100
            test_auc_gain = ((baseline_test_auc - metrics['test_auc_rmse']) / baseline_test_auc) * 100
            
            print(f"\n   Transfer Learning Gain vs Phase 2 Baseline (GRU + Correlation_FS, no TL):")
            print(f"Val RMSE:     {metrics['val_rmse']:.4f} vs {baseline_val_rmse:.4f}  {val_rmse_gain:+.2f}% {'' if val_rmse_gain > 0 else ''}")
            print(f"Test RMSE:    {metrics['test_rmse']:.4f} vs {baseline_test_rmse:.4f}  {test_rmse_gain:+.2f}% {'' if test_rmse_gain > 0 else ''}")
            print(f"Val AUC-RMSE: {metrics['val_auc_rmse']:.4f} vs {baseline_val_auc:.4f}  {val_auc_gain:+.2f}% {'' if val_auc_gain > 0 else ''}")
            print(f"Test AUC-RMSE:{metrics['test_auc_rmse']:.4f} vs {baseline_test_auc:.4f}  {test_auc_gain:+.2f}% {'' if test_auc_gain > 0 else ''}")
            
            # Store in metrics
            metrics['baseline_val_rmse'] = baseline_val_rmse
            metrics['baseline_test_rmse'] = baseline_test_rmse
            metrics['baseline_val_auc_rmse'] = baseline_val_auc
            metrics['baseline_test_auc_rmse'] = baseline_test_auc
            metrics['val_rmse_gain_pct'] = float(val_rmse_gain)
            metrics['test_rmse_gain_pct'] = float(test_rmse_gain)
            metrics['val_auc_gain_pct'] = float(val_auc_gain)
            metrics['test_auc_gain_pct'] = float(test_auc_gain)
        else:
            print(f"No baseline found for {n_engines} engines")
    
    return metrics, predictions

# ==========
# SAVING
# ==========

def save_finetuned_model(model, history, metrics, finetune_time, 
                         source_dataset, n_engines, 
                         y_val_true=None, y_val_pred=None, 
                         y_test_true=None, y_test_pred=None):
    save_dir = OUTPUT_BASE / f'{source_dataset}_to_{TARGET_DATASET}' / 'TL1_Vanilla_FT_GRU'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_dir / f'tl1_gru_model_{n_engines}engines.keras'
    model.save(model_path)
    
    # Save history (for convergence plots)
    history_path = save_dir / f'tl1_gru_history_{n_engines}engines.json'
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'epochs_trained': len(history.history['loss']),
        'finetune_time_sec': finetune_time,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_epoch': int(np.argmin(history.history['val_loss']) + 1)
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save metrics
    metrics_path = save_dir / f'tl1_gru_metrics_{n_engines}engines.json'
    metrics['source_dataset'] = source_dataset
    metrics['target_dataset'] = TARGET_DATASET
    metrics['n_engines'] = n_engines
    metrics['finetune_time_sec'] = finetune_time
    metrics['tl_strategy'] = 'TL1_Vanilla_FT_GRU'
    metrics['architecture'] = 'GRU'
    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['best_epoch'] = int(np.argmin(history.history['val_loss']) + 1)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions (for actual vs predicted plots)
    if y_val_pred is not None and y_val_true is not None:
        pred_dir = save_dir / 'predictions'
        pred_dir.mkdir(exist_ok=True)
        
        val_pred_df = pd.DataFrame({
            'y_true': y_val_true.flatten(),
            'y_pred': y_val_pred.flatten(),
            'error': y_val_true.flatten() - y_val_pred.flatten(),
            'abs_error': np.abs(y_val_true.flatten() - y_val_pred.flatten())
        })
        val_pred_df.to_csv(pred_dir / f'tl1_gru_val_predictions_{n_engines}engines.csv', index=False)
    
    if y_test_pred is not None and y_test_true is not None:
        pred_dir = save_dir / 'predictions'
        pred_dir.mkdir(exist_ok=True)
        
        test_pred_df = pd.DataFrame({
            'y_true': y_test_true.flatten(),
            'y_pred': y_test_pred.flatten(),
            'error': y_test_true.flatten() - y_test_pred.flatten(),
            'abs_error': np.abs(y_test_true.flatten() - y_test_pred.flatten())
        })
        test_pred_df.to_csv(pred_dir / f'tl1_gru_test_predictions_{n_engines}engines.csv', index=False)
    
    return save_dir

# ==========
# MAIN WORKFLOW
# ==========

def run_vanilla_tl_gru(source_dataset, n_engines, baseline_df=None):
    print("\n" + "="*40)
    print(f"TL1-GRU: VANILLA FINE-TUNING (GRU Architecture)")
    print(f"Source: {source_dataset}  Target: {TARGET_DATASET} ({n_engines} engines)")
    print("="*40)
    
    # Load target data
    X_train, y_train, X_val, y_val, X_test, y_test = load_target_data(n_engines)
    
    # Load pre-trained GRU model and metrics
    model = load_pretrained_model(source_dataset)
    pretrain_metrics = load_pretrained_metrics(source_dataset)
    
    # Compile for fine-tuning
    model = compile_for_finetuning(model)
    
    # Fine-tune
    history, finetune_time = finetune_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics, predictions = evaluate_model(model, X_val, y_val, X_test, y_test, 
                                         n_engines, baseline_df)
    
    # Add pre-training metrics to results (for combined CSV)
    metrics.update(pretrain_metrics)
    
    # Save
    save_dir = save_finetuned_model(
        model, history, metrics, finetune_time, 
        source_dataset, n_engines,
        y_val_true=predictions['y_val_true'],
        y_val_pred=predictions['y_val_pred'],
        y_test_true=predictions['y_test_true'],
        y_test_pred=predictions['y_test_pred']
    )
    
    print(f"\n Saved: {save_dir}")
    print("="*40)
    
    return metrics

# ==========
# BATCH PROCESSING
# ==========

def run_all_experiments():
    
    print("\n" + "="*40)
    print("PHASE 4: TL1-GRU - VANILLA FINE-TUNING (ALL EXPERIMENTS)")
    print("="*40)
    print(f"Sources: {SOURCE_DATASETS}")
    print(f"Target: {TARGET_DATASET}")
    print(f"Engine counts: {ENGINE_COUNTS_ALL}")
    print(f"Architecture: GRU")
    
    # Load Phase 2 baseline for comparison
    baseline_df = load_phase2_baseline()
    
    all_results = []
    
    for source_dataset in SOURCE_DATASETS:
        for n_engines in ENGINE_COUNTS_ALL:
            try:
                metrics = run_vanilla_tl_gru(source_dataset, n_engines, baseline_df)
                metrics['source'] = source_dataset
                metrics['n_engines'] = n_engines
                all_results.append(metrics)
            except Exception as e:
                print(f"\n Error: {source_dataset}  {TARGET_DATASET} ({n_engines} engines): {e}")
                continue
    
    # Save combined results (separate file per source dataset)
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Group by source dataset and save separately to avoid overwrites
        for source in results_df['source'].unique():
            source_results = results_df[results_df['source'] == source]
            results_path = OUTPUT_BASE / f'TL1_GRU_{source}_all_results.csv'
            results_path.parent.mkdir(parents=True, exist_ok=True)
            source_results.to_csv(results_path, index=False)
            print(f"\n {source} results saved: {results_path}")
        
        # Print summary
        print("\n" + "="*40)
        print("SUMMARY")
        print("="*40)
        for source in SOURCE_DATASETS:
            source_results = results_df[results_df['source'] == source]
            avg_val_rmse = source_results['val_rmse'].mean()
            avg_test_rmse = source_results['test_rmse'].mean()
            print(f"{source}  {TARGET_DATASET}: Val RMSE={avg_val_rmse:.4f}, Test RMSE={avg_test_rmse:.4f}")

# ==========
# SCRIPT EXECUTION
# ==========

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TL1-GRU: Vanilla Fine-Tuning with GRU')
    parser.add_argument('--source', type=str, choices=SOURCE_DATASETS + ['all'], 
                       default='all', help='Source dataset')
    parser.add_argument('--n_engines', type=int, choices=ENGINE_COUNTS_ALL + [0], 
                       default=0, help='Number of engines for training (0=all)')
    
    args = parser.parse_args()
    
    if args.source == 'all' or args.n_engines == 0:
        # Run all experiments
        run_all_experiments()
    else:
        # Run single experiment
        baseline_df = load_phase2_baseline()
        run_vanilla_tl_gru(args.source, args.n_engines, baseline_df)

