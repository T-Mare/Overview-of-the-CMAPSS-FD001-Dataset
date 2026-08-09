import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import DL model builders
sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "3_Deep_Learning_Models", "GRU"))
from gru_model import build_model as build_gru

sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "3_Deep_Learning_Models", "BiLSTM"))
from bilstm_model import build_model as build_bilstm

sys.path.insert(0, os.path.join(project_root, "CodeBase_Experiments", "1_CMAPSS_ML_Degradation_Experiment", "3_Deep_Learning_Models", "LSTM"))
from lstm_model import build_model as build_lstm

# Test parameters (14 features for Correlation-FS)
test_params = {
    'n_recurrent_layers': 1,
    'units_layer1': 128,
    'units_layer2': 64,
    'dropout_rate': 0.2,
    'recurrent_dropout': 0.0,
    'learning_rate': 0.001,
    'input_shape': (30, 14),  # 14 features
    'random_seed': 42
}

print("Testing DL models with 14 features (Correlation-FS)...")
print("="*40)

# Test GRU
try:
    print("\n1. Testing GRU...")
    model_gru = build_gru(**test_params)
    print(f"GRU built successfully")
    print(f"Input shape: {model_gru.input_shape}")
    print(f"Output shape: {model_gru.output_shape}")
except Exception as e:
    print(f"GRU failed: {e}")

# Test BiLSTM (with merge_mode)
try:
    print("\n2. Testing BiLSTM...")
    bilstm_params = test_params.copy()
    bilstm_params['merge_mode'] = 'concat'
    model_bilstm = build_bilstm(**bilstm_params)
    print(f"BiLSTM built successfully")
    print(f"Input shape: {model_bilstm.input_shape}")
    print(f"Output shape: {model_bilstm.output_shape}")
except Exception as e:
    print(f"BiLSTM failed: {e}")

# Test LSTM
try:
    print("\n3. Testing LSTM...")
    model_lstm = build_lstm(**test_params)
    print(f"LSTM built successfully")
    print(f"Input shape: {model_lstm.input_shape}")
    print(f"Output shape: {model_lstm.output_shape}")
except Exception as e:
    print(f"LSTM failed: {e}")

# Test with 12 features (Tree-FS)
print("\n" + "="*40)
print("Testing DL models with 12 features (Tree-FS)...")
print("="*40)

test_params_12 = test_params.copy()
test_params_12['input_shape'] = (30, 12)

try:
    print("\n1. Testing GRU with 12 features...")
    model_gru_12 = build_gru(**test_params_12)
    print(f"GRU (12 feat) built successfully")
except Exception as e:
    print(f"GRU (12 feat) failed: {e}")

print("\n" + "="*40)
print(" Sucessful run flag" if all else " Some tests failed")
print("="*40)

