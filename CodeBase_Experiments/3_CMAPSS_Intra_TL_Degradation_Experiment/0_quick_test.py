import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import time
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from Utilities.Plots_Metrics import rmse, mae, r2, cmapss_score
from Utilities import config

# ==========
# QUICK TEST CONFIGURATION
# ==========

# Use FD003  FD001 as test case (best input distribution match)
SOURCE_DATASET = 'FD003'
TARGET_DATASET = 'FD001'
TARGET_DATA_PCT = 30  # Test with 30% data

# Phase 2 Baseline path
# __file__ = .../3_CMAPSS_Intra_TL_Degradation_Experiment/0_quick_test.py
# .parent = .../3_CMAPSS_Intra_TL_Degradation_Experiment/
# .parent.parent = .../CodeBase_Experiments/
# .parent.parent.parent = .../Submission_Tiaan_Mare_34584757/ (project root)
PHASE2_BASELINE_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase2_Feature_Selection' / 'FD001' / 'Correlation_FS' / 'LSTM' / 'LSTM_metrics_summary.csv'

# 14 Correlation_FS features
SELECTED_FEATURES = ['s11', 's4', 's12', 's7', 's15', 's21', 's20', 's17', 
                     's2', 's3', 's8', 's13', 's9', 's14']

FEATURE_MAP = {
    's2': 0, 's3': 1, 's4': 2, 's7': 3, 's8': 4, 's9': 5,
    's11': 6, 's12': 7, 's13': 8, 's14': 9, 's15': 10,
    's17': 11, 's20': 12, 's21': 13
}

# QUICK TEST HYPERPARAMETERS (1 epoch only!)
QUICK_PARAMS = {
    'n_lstm_layers': 1,
    'units_layer1': 128,
    'dropout_rate': 0.1,
    'recurrent_dropout': 0.1,
    'pretrain_learning_rate': 0.001,
    'finetune_learning_rate': 0.0001,
    'batch_size': 32,
    'pretrain_epochs': 1,      #  QUICK TEST: 1 epoch only
    'finetune_epochs': 1,      #  QUICK TEST: 1 epoch only
}

# Paths
WINDOWED_DATA_PATH = Path(config.WINDOWED_DATA_PATH)
TEST_OUTPUT_DIR = Path(__file__).parent / 'TEST_OUTPUT'
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = config.RANDOM_SEED

# ==========
# HELPER FUNCTIONS
# ==========

def extract_selected_features(X):
    indices = [FEATURE_MAP[feat] for feat in SELECTED_FEATURES]
    return X[:, :, indices]

def build_lstm_model(input_shape=(30, 14), learning_rate=0.001):
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    model = keras.Sequential(name='LSTM_QuickTest')
    model.add(keras.layers.Input(shape=input_shape, name='input'))
    model.add(keras.layers.LSTM(
        units=QUICK_PARAMS['units_layer1'],
        dropout=QUICK_PARAMS['dropout_rate'],
        recurrent_dropout=QUICK_PARAMS['recurrent_dropout'],
        return_sequences=False,
        name='lstm_1'
    ))
    model.add(keras.layers.Dense(1, activation='linear', name='output'))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# ==========
# TEST 1: PRE-TRAINING
# ==========

def test_pretraining():
    print("\n" + "="*40)
    print("TEST 1: PRE-TRAINING ON SOURCE")
    print("="*40)
    print(f"Source: {SOURCE_DATASET}")
    print(f"Epochs: {QUICK_PARAMS['pretrain_epochs']} (quick test)")
    
    # Load source data
    print(f"\nLoading {SOURCE_DATASET} data...")
    X_train = np.load(WINDOWED_DATA_PATH / f'{SOURCE_DATASET}_X_train_windowed.npy')
    y_train = np.load(WINDOWED_DATA_PATH / f'{SOURCE_DATASET}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{SOURCE_DATASET}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{SOURCE_DATASET}_y_val_windowed.npy')
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    
    # Extract features
    print(f"\nExtracting 14 Correlation_FS features...")
    X_train = extract_selected_features(X_train)
    X_val = extract_selected_features(X_val)
    print(f"Shape: {X_train.shape}")
    
    # Build model
    print(f"\nBuilding LSTM model...")
    model = build_lstm_model(input_shape=(30, 14), 
                            learning_rate=QUICK_PARAMS['pretrain_learning_rate'])
    print(f"Parameters: {model.count_params():,}")
    
    # Train (1 epoch only)
    print(f"\nPre-training (1 epoch)...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=QUICK_PARAMS['pretrain_epochs'],
        batch_size=QUICK_PARAMS['batch_size'],
        verbose=1
    )
    pretrain_time = time.time() - start_time
    
    # Evaluate
    print(f"\nEvaluating...")
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    val_rmse = rmse(y_val, y_val_pred)
    val_mae = mae(y_val, y_val_pred)
    val_r2 = r2(y_val, y_val_pred)
    
    print(f"Time: {pretrain_time:.1f}s")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Val MAE:  {val_mae:.4f}")
    print(f"Val R²:   {val_r2:.4f}")
    
    # Save model
    model_path = TEST_OUTPUT_DIR / 'test_pretrained_model.keras'
    model.save(model_path)
    print(f"\n Pre-trained model saved: {model_path.name}")
    
    print("\n" + "="*40)
    print("TEST 1 PASSED: Pre-training works!")
    print("="*40)
    
    return model, val_rmse

# ==========
# TEST 2: FINE-TUNING
# ==========

def test_finetuning(pretrained_model):
    print("\n" + "="*40)
    print("TEST 2: FINE-TUNING ON TARGET")
    print("="*40)
    print(f"Target: {TARGET_DATASET} ({TARGET_DATA_PCT}% data)")
    print(f"Epochs: {QUICK_PARAMS['finetune_epochs']} (quick test)")
    
    # Load target data
    print(f"\nLoading {TARGET_DATASET} data...")
    X_train_full = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_train_windowed.npy')
    y_train_full = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_val_windowed.npy')
    X_test = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_test_windowed.npy')
    y_test = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_test_windowed.npy')
    
    # Sample to target percentage
    np.random.seed(RANDOM_SEED)
    n_samples = int(len(X_train_full) * (TARGET_DATA_PCT / 100))
    indices = np.random.choice(len(X_train_full), size=n_samples, replace=False)
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]
    
    print(f"Train: {X_train.shape[0]} samples ({TARGET_DATA_PCT}% of {len(X_train_full)})")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    
    # Extract features
    print(f"\nExtracting 14 Correlation_FS features...")
    X_train = extract_selected_features(X_train)
    X_val = extract_selected_features(X_val)
    X_test = extract_selected_features(X_test)
    print(f"Shape: {X_train.shape}")
    
    # Re-compile for fine-tuning
    print(f"\nCompiling for fine-tuning (LR={QUICK_PARAMS['finetune_learning_rate']})...")
    optimizer = keras.optimizers.Adam(learning_rate=QUICK_PARAMS['finetune_learning_rate'])
    pretrained_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    print(f"All layers trainable")
    
    # Fine-tune (1 epoch only)
    print(f"\nFine-tuning (1 epoch)...")
    start_time = time.time()
    history = pretrained_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=QUICK_PARAMS['finetune_epochs'],
        batch_size=QUICK_PARAMS['batch_size'],
        verbose=1
    )
    finetune_time = time.time() - start_time
    
    # Evaluate
    print(f"\nEvaluating...")
    y_val_pred = pretrained_model.predict(X_val, verbose=0).flatten()
    y_test_pred = pretrained_model.predict(X_test, verbose=0).flatten()
    
    val_rmse = rmse(y_val, y_val_pred)
    val_mae = mae(y_val, y_val_pred)
    test_rmse = rmse(y_test, y_test_pred)
    test_mae = mae(y_test, y_test_pred)
    
    print(f"Time: {finetune_time:.1f}s")
    print(f"Val RMSE:  {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Load Phase 2 baseline for comparison
    if PHASE2_BASELINE_PATH.exists():
        baseline_df = pd.read_csv(PHASE2_BASELINE_PATH)
        baseline_df = baseline_df[baseline_df['fs_method'] == 'Correlation_FS']
        baseline_row = baseline_df[baseline_df['data_pct'] == TARGET_DATA_PCT]
        
        if not baseline_row.empty:
            baseline_val_rmse = float(baseline_row['val_rmse'].values[0])
            baseline_test_rmse = float(baseline_row['test_rmse'].values[0])
            
            val_tl_gain = ((baseline_val_rmse - val_rmse) / baseline_val_rmse) * 100
            test_tl_gain = ((baseline_test_rmse - test_rmse) / baseline_test_rmse) * 100
            
            print(f"\n   TL Gain vs Phase 2 Baseline (LSTM + Correlation_FS, no TL):")
            print(f"Val:  {val_rmse:.4f} vs {baseline_val_rmse:.4f}  {val_tl_gain:+.2f}% {'' if val_tl_gain > 0 else ''}")
            print(f"Test: {test_rmse:.4f} vs {baseline_test_rmse:.4f}  {test_tl_gain:+.2f}% {'' if test_tl_gain > 0 else ''}")
            print(f"(Note: 1 epoch only - expect negative, full training will improve)")
    else:
        print(f"\n   Phase 2 baseline not found, skipping comparison")
    
    # Save fine-tuned model
    model_path = TEST_OUTPUT_DIR / 'test_finetuned_model.keras'
    pretrained_model.save(model_path)
    print(f"\n Fine-tuned model saved: {model_path.name}")
    
    print("\n" + "="*40)
    print("TEST 2 PASSED: Fine-tuning works!")
    print("="*40)
    
    return val_rmse, test_rmse

# ==========
# MAIN TEST
# ==========

def run_quick_test():
    print("\n" + "="*40)
    print("PHASE 4: QUICK TEST - TRANSFER LEARNING PIPELINE")
    print("="*40)
    print(f"Test case: {SOURCE_DATASET}  {TARGET_DATASET} ({TARGET_DATA_PCT}%)")
    print(f"Duration: ~3-5 minutes (1 epoch pre-train + 1 epoch fine-tune)")
    print(f"Output: {TEST_OUTPUT_DIR}")
    
    overall_start = time.time()
    
    try:
        # Test 1: Pre-training
        pretrained_model, pretrain_rmse = test_pretraining()
        
        # Test 2: Fine-tuning
        val_rmse, test_rmse = test_finetuning(pretrained_model)
        
        # Summary
        total_time = time.time() - overall_start
        
        print("\n" + "="*40)
        print("Sucessful run flag")
        print("="*40)
        print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"\nResults (1 epoch only - NOT final performance!):")
        print(f"Pre-training:  {SOURCE_DATASET} RMSE = {pretrain_rmse:.4f}")
        print(f"Fine-tuning:   {TARGET_DATASET} Val RMSE = {val_rmse:.4f}")
        print(f"Fine-tuning:   {TARGET_DATASET} Test RMSE = {test_rmse:.4f}")
        
        # Load baseline for final comparison
        if PHASE2_BASELINE_PATH.exists():
            baseline_df = pd.read_csv(PHASE2_BASELINE_PATH)
            baseline_df = baseline_df[baseline_df['fs_method'] == 'Correlation_FS']
            baseline_row = baseline_df[baseline_df['data_pct'] == TARGET_DATA_PCT]
            
            if not baseline_row.empty:
                baseline_val_rmse = float(baseline_row['val_rmse'].values[0])
                baseline_test_rmse = float(baseline_row['test_rmse'].values[0])
                val_tl_gain = ((baseline_val_rmse - val_rmse) / baseline_val_rmse) * 100
                test_tl_gain = ((baseline_test_rmse - test_rmse) / baseline_test_rmse) * 100
                
                print(f"\nComparison to Phase 2 Baseline (30% data):")
                print(f"Val:  {val_tl_gain:+.2f}% (baseline: {baseline_val_rmse:.4f})")
                print(f"Test: {test_tl_gain:+.2f}% (baseline: {baseline_test_rmse:.4f})")
                print(f"(Note: Negative expected - only 1 epoch training)")
        
        print(f"\n Pipeline verified! Ready for full training.")
        print(f"\nNext steps:")
        print(f"1. Run: python 1_pretrain_source_lstm.py")
        print(f"2. Run: python 2_finetune_vanilla_tl.py --source FD002 --data_pct 30")
        print(f"3. If step 2 works, run full batch: python 2_finetune_vanilla_tl.py")
        
        print("\n" + "="*40)
        
        return True
        
    except Exception as e:
        print("\n" + "="*40)
        print("TEST FAILED!")
        print("="*40)
        print(f"\nError: {e}")
        print(f"\nDebugging tips:")
        print(f"- Check data paths are correct")
        print(f"- Verify windowed data exists: {WINDOWED_DATA_PATH}")
        print(f"- Check Utilities module is accessible")
        print(f"- Try: pip install tensorflow numpy pandas scikit-learn")
        
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()
        
        return False

# ==========
# SCRIPT EXECUTION
# ==========

if __name__ == '__main__':
    success = run_quick_test()
    
    if success:
        print("\n SUCCESS! TL pipeline is working correctly.")
    else:
        print("\n FAILURE! Fix errors before running full training.")
    
    # Clean up test files (optional)
    cleanup = input("\nDelete test files? (y/n): ").strip().lower()
    if cleanup == 'y':
        import shutil
        if TEST_OUTPUT_DIR.exists():
            shutil.rmtree(TEST_OUTPUT_DIR)
            print(f"Deleted {TEST_OUTPUT_DIR}")

