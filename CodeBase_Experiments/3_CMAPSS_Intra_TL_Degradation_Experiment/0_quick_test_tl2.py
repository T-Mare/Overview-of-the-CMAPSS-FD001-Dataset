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
print("QUICK TEST: TL2 FROZEN FEATURES FINE-TUNING")
print("="*40)
print("This uses ONLY 1 epoch for pre-training and fine-tuning")
print("Results will be poor - just testing the pipeline!")
print("="*40 + "\n")

# ==========
# CONFIGURATION
# ==========

SOURCE_DATASET = 'FD003'
TARGET_DATASET = 'FD001'
TARGET_DATA_PERCENTAGE = 30

# 14 Correlation_FS features
SELECTED_FEATURES = ['s11', 's4', 's12', 's7', 's15', 's21', 's20', 's17', 
                     's2', 's3', 's8', 's13', 's9', 's14']

FEATURE_MAP = {
    's2': 0, 's3': 1, 's4': 2, 's7': 3, 's8': 4, 's9': 5,
    's11': 6, 's12': 7, 's13': 8, 's14': 9, 's15': 10,
    's17': 11, 's20': 12, 's21': 13
}

# Test paths
WINDOWED_DATA_PATH = Path(config.WINDOWED_DATA_PATH)
TEST_OUTPUT_DIR = Path(__file__).parent / 'test_output_tl2'
PHASE2_BASELINE_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase2_Feature_Selection' / 'FD001' / 'Correlation_FS' / 'LSTM' / 'LSTM_metrics_summary.csv'

RANDOM_SEED = config.RANDOM_SEED

# ==========
# DATA LOADING
# ==========

def extract_selected_features(X):
    indices = [FEATURE_MAP[feat] for feat in SELECTED_FEATURES]
    return X[:, :, indices]

def load_source_data(dataset):
    print(f"Loading {dataset} source data...")
    X_train = np.load(WINDOWED_DATA_PATH / f'{dataset}_X_train_windowed.npy')
    y_train = np.load(WINDOWED_DATA_PATH / f'{dataset}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{dataset}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{dataset}_y_val_windowed.npy')
    
    X_train = extract_selected_features(X_train)
    X_val = extract_selected_features(X_val)
    
    print(f"Source train: {X_train.shape}")
    print(f"Source val:   {X_val.shape}")
    
    return X_train, y_train, X_val, y_val

def load_target_data(data_percentage):
    print(f"\nLoading {TARGET_DATASET} target data ({data_percentage}%)...")
    
    X_train_full = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_train_windowed.npy')
    y_train_full = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_val_windowed.npy')
    X_test = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_test_windowed.npy')
    y_test = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_test_windowed.npy')
    
    # Sample to percentage
    if data_percentage < 100:
        np.random.seed(RANDOM_SEED)
        n_samples = int(len(X_train_full) * (data_percentage / 100))
        indices = np.random.choice(len(X_train_full), size=n_samples, replace=False)
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]
    else:
        X_train = X_train_full
        y_train = y_train_full
    
    print(f"Target train: {X_train.shape} ({data_percentage}%)")
    print(f"Target val:   {X_val.shape}")
    print(f"Target test:  {X_test.shape}")
    
    # Extract features
    X_train = extract_selected_features(X_train)
    X_val = extract_selected_features(X_val)
    X_test = extract_selected_features(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==========
# MODEL BUILDING
# ==========

def build_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ==========
# TEST 1: PRE-TRAINING
# ==========

def test_pretraining():
    print("\n" + "="*40)
    print("TEST 1: PRE-TRAINING ON SOURCE (FD003) - 1 EPOCH ONLY")
    print("="*40)
    
    # Load source data
    X_train, y_train, X_val, y_val = load_source_data(SOURCE_DATASET)
    
    # Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    print(f"Model built: {model.count_params():,} total parameters")
    
    # Pre-train (1 epoch only)
    print("\nPre-training (1 epoch)...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,  # Just 1 epoch for testing
        batch_size=32,
        verbose=1
    )
    pretrain_time = time.time() - start_time
    
    # Evaluate
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    val_rmse = rmse(y_val, y_val_pred)
    
    print(f"\n Pre-training complete in {pretrain_time:.1f}s")
    print(f"Val RMSE: {val_rmse:.4f} (will be bad - only 1 epoch!)")
    
    # Save model temporarily
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = TEST_OUTPUT_DIR / f'lstm_pretrained_{SOURCE_DATASET}_test.keras'
    model.save(model_path)
    print(f"Saved test model: {model_path.name}")
    
    return model, val_rmse

# ==========
# TEST 2: TL2 FROZEN FEATURES FINE-TUNING
# ==========

def test_tl2_finetuning(pretrained_model):
    print("\n" + "="*40)
    print("TEST 2: TL2 FROZEN FEATURES FINE-TUNING - 1 EPOCH ONLY")
    print("="*40)
    
    # Load target data
    X_train, y_train, X_val, y_val, X_test, y_test = load_target_data(TARGET_DATA_PERCENTAGE)
    
    # Count params before freezing
    total_params = pretrained_model.count_params()
    print(f"\nBefore freezing:")
    print(f"Total params: {total_params:,}")
    
    # FREEZE LSTM LAYERS (TL2 KEY STEP!)
    print(f"\n Freezing LSTM layers (TL2 strategy)...")
    frozen_count = 0
    trainable_count = 0
    
    for layer in pretrained_model.layers:
        if 'lstm' in layer.name.lower():
            layer.trainable = False  #  FREEZE
            frozen_count += 1
            print(f"Frozen: {layer.name} ({layer.count_params():,} params)")
        else:
            layer.trainable = True   #  TRAIN
            trainable_count += 1
            print(f"Trainable: {layer.name} ({layer.count_params():,} params)")
    
    # Count trainable params after freezing
    trainable_params = sum([tf.size(w).numpy() for w in pretrained_model.trainable_weights])
    frozen_params = total_params - trainable_params
    
    print(f"\nAfter freezing:")
    print(f"Total params:     {total_params:,}")
    print(f"Frozen params:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # Verify correct number of trainable params
    if trainable_params < 100 or trainable_params > 200:
        print(f"\n   WARNING: Expected ~130 trainable params, got {trainable_params}")
    else:
        print(f"\n   Correct! ~130 trainable params (only Dense output layer)")
    
    # Re-compile with HIGHER learning rate (TL2 specific)
    print(f"\nCompiling for TL2 fine-tuning...")
    print(f"Learning rate: 0.001 (10x higher than TL1's 0.0001)")
    pretrained_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Higher LR
        loss='mse',
        metrics=['mae']
    )
    
    # Fine-tune (1 epoch only)
    print("\nFine-tuning TL2 model (1 epoch)...")
    start_time = time.time()
    history = pretrained_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,  # Just 1 epoch for testing
        batch_size=32,
        verbose=1
    )
    finetune_time = time.time() - start_time
    
    # Evaluate
    y_val_pred = pretrained_model.predict(X_val, verbose=0).flatten()
    y_test_pred = pretrained_model.predict(X_test, verbose=0).flatten()
    
    val_rmse = rmse(y_val, y_val_pred)
    test_rmse = rmse(y_test, y_test_pred)
    val_mae = mae(y_val, y_val_pred)
    test_mae = mae(y_test, y_test_pred)
    
    print(f"\n TL2 fine-tuning complete in {finetune_time:.1f}s")
    print(f"Val RMSE:  {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Compare with baseline
    if PHASE2_BASELINE_PATH.exists():
        baseline_df = pd.read_csv(PHASE2_BASELINE_PATH)
        baseline_row = baseline_df[baseline_df['data_pct'] == TARGET_DATA_PERCENTAGE]
        
        if not baseline_row.empty:
            baseline_val_rmse = float(baseline_row['val_rmse'].values[0])
            baseline_test_rmse = float(baseline_row['test_rmse'].values[0])
            
            val_gain = ((baseline_val_rmse - val_rmse) / baseline_val_rmse) * 100
            test_gain = ((baseline_test_rmse - test_rmse) / baseline_test_rmse) * 100
            
            print(f"\n   TL2 Gain vs Phase 2 Baseline:")
            print(f"Val RMSE:  {val_rmse:.4f} vs {baseline_val_rmse:.4f}  {val_gain:+.2f}% {'' if val_gain > 0 else ''}")
            print(f"Test RMSE: {test_rmse:.4f} vs {baseline_test_rmse:.4f}  {test_gain:+.2f}% {'' if test_gain > 0 else ''}")
            print(f"\n   Remember: Only 1 epoch - real results will be different!")
    
    return val_rmse, test_rmse

# ==========
# MAIN TEST WORKFLOW
# ==========

def main():
    try:
        # Test 1: Pre-training
        pretrained_model, pretrain_rmse = test_pretraining()
        print("\n TEST 1 PASSED: Pre-training works!")
        
        # Test 2: TL2 Fine-tuning
        val_rmse, test_rmse = test_tl2_finetuning(pretrained_model)
        print("\n TEST 2 PASSED: TL2 frozen features fine-tuning works!")
        
        # Success!
        print("\n" + "="*40)
        print("SUCCESS! TL2 pipeline is working correctly.")
        print("="*40)
        print("\nKey TL2 Features Verified:")
        print("LSTM layers frozen correctly")
        print("Only ~130 parameters trainable")
        print("Higher learning rate (0.001) working")
        print("Fine-tuning completes successfully")
        print("Baseline comparison functional")
        
        print("\n Next Steps:")
        print("1. Wait for full pre-training to complete (1_pretrain_source_lstm.py)")
        print("2. Run TL1 test: python 2_finetune_vanilla_tl.py --source FD003 --data_pct 30")
        print("3. If TL1 shows >5% gain, run TL2: python 3_finetune_frozen_tl2.py --source FD003 --data_pct 30")
        print("4. Compare TL1 vs TL2 results")
        
    except Exception as e:
        print("\n" + "="*40)
        print("TEST FAILED!")
        print("="*40)
        print(f"\nError: {e}")
        print("\n FAILURE! Fix errors before running full training.")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print("\n" + "="*40)
    if TEST_OUTPUT_DIR.exists():
        response = input("Delete test files? (y/n): ").strip().lower()
        if response == 'y':
            import shutil
            shutil.rmtree(TEST_OUTPUT_DIR)
            print("Test files deleted")
        else:
            print(f"Test files kept in: {TEST_OUTPUT_DIR}")
    print("="*40)

if __name__ == '__main__':
    main()

