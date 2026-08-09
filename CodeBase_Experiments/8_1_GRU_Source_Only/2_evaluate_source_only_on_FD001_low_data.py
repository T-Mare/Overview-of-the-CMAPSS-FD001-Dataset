import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utilities.Plots_Metrics import rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc
from Utilities import config

# ==========
# CONFIG
# ==========

SOURCE_DATASET = 'FD002'
TARGET_DATASET = 'FD001'
DEFAULT_ENGINE_COUNTS = [80, 20, 10, 5, 3]
SELECTED_FEATURES = ['s11', 's4', 's12', 's7', 's15', 's21', 's20', 's17', 
                     's2', 's3', 's8', 's13', 's9', 's14']

FEATURE_MAP = {
    's2': 0, 's3': 1, 's4': 2, 's7': 3, 's8': 4, 's9': 5,
    's11': 6, 's12': 7, 's13': 8, 's14': 9, 's15': 10,
    's17': 11, 's20': 12, 's21': 13
}

WINDOWED_DATA_PATH = Path(config.WINDOWED_DATA_PATH)
MODEL_PATH = SCRIPT_DIR / "outputs_source_only_FD002_to_FD001" / "gru_source_only_FD002.keras"
OUTPUT_DIR = SCRIPT_DIR / "outputs_source_only_evaluation"

# ==========
# DATA LOADING
# ==========

def extract_selected_features(X, selected_features=SELECTED_FEATURES):
    indices = [FEATURE_MAP[feat] for feat in selected_features]
    return X[:, :, indices]

def load_target_data():
    print(f"\n{TARGET_DATASET} is loading...")
    
    X_train = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_train_windowed.npy')
    y_train = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_val_windowed.npy')
    X_test = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_X_test_windowed.npy')
    y_test = np.load(WINDOWED_DATA_PATH / f'{TARGET_DATASET}_y_test_windowed.npy')
    
    # extract features
    X_train = extract_selected_features(X_train)
    X_val = extract_selected_features(X_val)
    X_test = extract_selected_features(X_test)
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_pretrained_model():
    if not MODEL_PATH.exists(): ##debug problem
        raise FileNotFoundError(
            f"Pretrained model not found: {MODEL_PATH}\n"
            
            f"Run 1_train_source_only_gru_FD002.py first."
        )
    
    print(f"\nModel is loading from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
    model.summary()
    
    return model

# ==========
# EVALUATION
# ==========

def evaluate_split(model, X, y, split_name=''):
    y_pred = model.predict(X, verbose=0).flatten()
    
    metrics = { #calculate metrics
        'rmse': float(rmse(y, y_pred)),
        'mae': float(mae(y, y_pred)),
        'r2': float(r2(y, y_pred)),
        'cmapss': float(cmapss_score(y, y_pred, reduction='sum'))
    }
    
    rmse_bins, auc_rmse = rmse_by_bins_with_auc(y, y_pred, config.RUL_BINS)
    metrics['auc_rmse'] = float(auc_rmse) if auc_rmse is not None else None
    
    if split_name:
        print(f"{split_name:8s} - RMSE: {metrics['rmse']:.4f}, "
              f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return metrics, y_pred

def evaluate_for_engine_count(model, X_val, y_val, X_test, y_test, n_engines):
    print(f"\nSource-Only on {TARGET_DATASET} ({n_engines} engines scenario)")
    print("="*40)
    
    val_metrics, val_pred = evaluate_split(model, X_val, y_val, 'Val')
    test_metrics, test_pred = evaluate_split(model, X_test, y_test, 'Test')
    
    results = {
        'n_engines': n_engines,
        'method': 'Source_Only_GRU',
        'source_dataset': SOURCE_DATASET,
        'target_dataset': TARGET_DATASET,
        'architecture': 'GRU (256 units, dropout 0.3)',
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'val_r2': val_metrics['r2'],
        'val_cmapss': val_metrics['cmapss'],
        'val_auc_rmse': val_metrics['auc_rmse'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'test_cmapss': test_metrics['cmapss'],
        'test_auc_rmse': test_metrics['auc_rmse']
    }
    
    return results, val_pred, test_pred

def save_evaluation_results(all_results, model):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # save combined CSV
    results_df = pd.DataFrame(all_results)
    results_path = OUTPUT_DIR / 'source_only_all_engine_counts.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nCombined results saved: {results_path}")
    
    # save individual JSON files
    for result in all_results:
        n_engines = result['n_engines']
        json_path = OUTPUT_DIR / f'source_only_{n_engines}_engines.json'
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"All results saved to: {OUTPUT_DIR}")
    
    return results_df

# ==========
# MAIN
# ==========

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate source-only GRU on FD001 (zero-shot transfer baseline)"
    )
    parser.add_argument(
        "--n_engines", type=int, nargs="*", default=None,
        help=f"Engine counts to report (for comparison). Default: {DEFAULT_ENGINE_COUNTS}"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    engine_counts = args.n_engines if args.n_engines else DEFAULT_ENGINE_COUNTS
    
    print(f"\nSOURCE-ONLY GRU EVALUATION: {SOURCE_DATASET} -> {TARGET_DATASET}")
    print("\n######################")
    print(f"Engine count scenarios: {engine_counts}")
    print(f"Note: Same performance for all (no target labels used)")
    
    # load model and data
    model = load_pretrained_model()
    X_train, y_train, X_val, y_val, X_test, y_test = load_target_data()
    
    # evaluate for each engine count scenario
    all_results = []
    for n_engines in engine_counts:
        results, val_pred, test_pred = evaluate_for_engine_count(
            model, X_val, y_val, X_test, y_test, n_engines
        )
        all_results.append(results)
    
    # save results
    results_df = save_evaluation_results(all_results, model)
    
    # print summary
    print("\n_________________________Summary Source-Only Baseline_________________________")
    print("\n######################")
    print(f"\n{'Engines':<10} {'Val RMSE':<12} {'Test RMSE':<12}")
    print("\n######################")
    
    for result in all_results:
        print(f"{result['n_engines']:<10} {result['val_rmse']:<12.4f} {result['test_rmse']:<12.4f}")
    
    print("\n######################")

if __name__ == '__main__':
    main()
