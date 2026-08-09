import argparse
import json
import math
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import sys

# Add parent directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from costa_dann_model import build_costa_dann_fd002_to_fd001, compile_costa_dann
from costa_data_utils import load_cmapss_dataset, create_domain_batches

# Import config and utilities from Utilities folder
from Utilities import config
from Utilities.Plots_Metrics import rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc

OUTPUT_DIR = SCRIPT_DIR / "outputs_FD002_to_FD001"
BASELINE_PATH = (
    PROJECT_ROOT
    / "Results"/ "Phase2_Feature_Selection"/ "FD001"/ "Correlation_FS"/ "GRU"/ "GRU_metrics_summary.csv"
)

# Configuration (can change source and target datasets with cli argumemts)
SOURCE_DATASET = 'FD002'
TARGET_DATASET = 'FD001'
DEFAULT_ENGINE_COUNTS = [20, 10, 5, 3]

# Costa et al. hyperparameters (Table 3) - Source-Target specific configurations
HYPERPARAMS_BY_SOURCE_TARGET = {
    ('FD002', 'FD001'): {
        'lstm_layers': 1,
        'lstm_units': (64,),
        'lstm_dropout': 0.1,
        'feature_units': 64,
        'rul_layers': 1,
        'rul_units': (32,),
        'rul_dropout': 0.0,
        'domain_layers': 2,
        'domain_units': (16, 16),
        'domain_dropout': 0.1,
        'l2_reg': 0.01,
        'lambda_adversarial': 1.0,
        'batch_size': 512,
        'lr_rul': 0.01,
        'lr_domain': 0.01,
        'clipnorm': 1.0,  # "We clip the norm values of the gradients to 1 in the SGD algorithm to avoid exploding gradients."
        'max_epochs': 200,
        'patience': 15,   # this hp is not explicitly stated in study and is set to 15 (which is similar to other experiments in this workspace)
        'lr_decay_epoch': 100,
        'lr_decay_factor': 0.1
    },
    ('FD003', 'FD001'): {
        'lstm_layers': 2,
        'lstm_units': (64, 32),
        'lstm_dropout': 0.3,
        'feature_units': 128,
        'rul_layers': 2,
        'rul_units': (32, 32),
        'rul_dropout': 0.1,
        'domain_layers': 2,
        'domain_units': (32, 32),
        'domain_dropout': 0.1,
        'l2_reg': 0.01,
        'lambda_adversarial': 2.0,
        'batch_size': 256,
        'lr_rul': 0.01,
        'lr_domain': 0.01,
        'clipnorm': 1.0,  
        'max_epochs': 200,
        'patience': 15,  #assumed hp
        'lr_decay_epoch': 100,
        'lr_decay_factor': 0.1
    }
}

def get_hyperparams(source, target):

    key = (source, target)
    if key not in HYPERPARAMS_BY_SOURCE_TARGET:

        raise ValueError(
            f"No hyperparameters defined for {source}{target}. "
            f"Available configurations: {list(HYPERPARAMS_BY_SOURCE_TARGET.keys())}"
        )

    return HYPERPARAMS_BY_SOURCE_TARGET[key]

def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_baseline():

    if not BASELINE_PATH.exists():
        print(f"Baseline not found at {BASELINE_PATH}")
        return None
    
    baseline_df = pd.read_csv(BASELINE_PATH)
    print("Loaded Phase 2 baseline (GRU + Correlation_FS)")
    return baseline_df

class LearningRateDecayCallback(keras.callbacks.Callback):
    
    def __init__(self, decay_epoch=100, decay_factor=0.1): #At epoch 100, multiply the learning rate by 0.1.
        super().__init__() 
        self.decay_epoch = decay_epoch
        self.decay_factor = decay_factor
        self.initial_lr = None
    def on_epoch_begin(self, epoch, logs=None):
        if self.initial_lr is None:
            self.initial_lr = float(keras.backend.get_value(self.model.optimizer.lr))
        
        if epoch == self.decay_epoch:
            "multiply the initial learning rate by the decay factor"
            new_lr = self.initial_lr * self.decay_factor
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\nLR decay at epoch {epoch}: {self.initial_lr:.6f} -> {new_lr:.6f}")

class RestoreBestWeightsAtEnd(keras.callbacks.Callback):
    def __init__(self, monitor="val_rul_output_loss", mode="min"):
        super().__init__()
        self.monitor = monitor #monitor the validation loss
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = current < self.best if self.mode == "min" else current > self.best
        if improved:
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(
                f"\nRestored best weights from epoch {self.best_epoch + 1} "
                f"({self.monitor}={self.best:.4f})"
            )

def evaluate_rul_split(model, X, y, rul_max):
    pred_rul, _ = model.predict(X, verbose=0)
    y_pred = pred_rul.flatten() * rul_max
    y_true = np.asarray(y) * rul_max

    rmse_bins, auc_rmse = rmse_by_bins_with_auc(y_true, y_pred, config.RUL_BINS)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mae(y_true, y_pred)),
        "r2": float(r2(y_true, y_pred)),
        "cmapss": float(cmapss_score(y_true, y_pred)),
        "auc_rmse": float(auc_rmse) if auc_rmse is not None else None,
        "rmse_bins": rmse_bins,  # Bin-wise RMSE for plotting EOL curve
    }

def evaluate_domain_accuracy(model, source_X, target_X):
    X_domain = np.concatenate([source_X, target_X], axis=0)
    y_domain = np.concatenate([
        np.zeros(len(source_X), dtype=np.float32),
        np.ones(len(target_X), dtype=np.float32),
    ])
    _, pred_domain = model.predict(X_domain, verbose=0)
    pred_labels = (pred_domain.flatten() > 0.5).astype(np.float32)
    return float((pred_labels == y_domain).mean())

def save_predictions(path, y_true, y_pred):
    predictions_df = pd.DataFrame({
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "error": np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float),
    })
    predictions_df.to_csv(path, index=False)

###################
#Build DANN model
##################

def train_costa_dann(n_engines, hyperparams, baseline_df=None):
    
    print(f"\nCOSTA DANN: {SOURCE_DATASET}->{TARGET_DATASET} ({n_engines} engines)")
    print("==============================================")
    
    # =====================================================
    # LOAD DATA
    # =====================================================
    
    print("\nData is loading...")
    
    # load source domain (all training engines with labels)
    source_data = load_cmapss_dataset(SOURCE_DATASET, use_all_engines=True)
    
    # load target domain (N engines for training, saved val/test for eval)
    #it shoul be noted The lables are loaded for evaluation only, not for training.
    target_data = load_cmapss_dataset(
        TARGET_DATASET, use_all_engines=False, n_engines=n_engines
    )
    
    # =====================================================
    # BUILD MODEL
    # ===============================================
    
    print("\nBuilding DANN model...")
    model = build_costa_dann_fd002_to_fd001(
        input_shape=(30, 14),
        lstm_layers=hyperparams['lstm_layers'],
        lstm_units=hyperparams['lstm_units'],
        lstm_dropout=hyperparams['lstm_dropout'],
        feature_units=hyperparams['feature_units'],
        rul_layers=hyperparams['rul_layers'],
        rul_units=hyperparams['rul_units'],
        rul_dropout=hyperparams['rul_dropout'],
        domain_layers=hyperparams['domain_layers'],
        domain_units=hyperparams['domain_units'],
        domain_dropout=hyperparams['domain_dropout'],
        l2_reg=hyperparams['l2_reg'],
        lambda_adversarial=hyperparams['lambda_adversarial']
    )
    
    model = compile_costa_dann(
        model,
        lr_rul=hyperparams['lr_rul'],
        lr_domain=hyperparams['lr_domain'],
        clipnorm=hyperparams['clipnorm']
    )
    
    model.summary()
    
    # ==========================
    # Training configuration
    # ============================
    
    n_batches_source = int(math.ceil(source_data['n_train'] / hyperparams['batch_size']))
    n_batches_target = int(math.ceil(target_data['n_train'] / hyperparams['batch_size']))
    steps_per_epoch = max(n_batches_source, n_batches_target)
    
    print(f"\nTraining configuration:")
    print(f"Learning rate: {hyperparams['lr_rul']:.3f}")
    print(f"Batch size: {hyperparams['batch_size']}")
    print(f"Max epochs: {hyperparams['max_epochs']}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Early stopping patience: {hyperparams['patience']} epochs")
    print(f"LR decay: {hyperparams['lr_decay_factor']}x at epoch {hyperparams['lr_decay_epoch']}")
    print(f"Lambda (adversarial): {hyperparams['lambda_adversarial']}")
    
    # create batch generator this feeds in the data to DANN
    batch_generator = create_domain_batches(
        source_data,
        target_data,
        batch_size=hyperparams['batch_size'],
        oversample_smaller=True
    )
    
    # Validate on TARGET domain ONLY 
    X_val = target_data['X_val']
    y_val_rul = target_data['y_val']
    y_val_domain = np.zeros(len(X_val), dtype=np.float32)
    val_weights_rul = np.ones(len(X_val), dtype=np.float32)
    val_weights_domain = np.zeros(len(X_val), dtype=np.float32)
    
    validation_data = (
        X_val,
        {'rul_output': y_val_rul, 'domain_output': y_val_domain},
        {'rul_output': val_weights_rul, 'domain_output': val_weights_domain},
    )
    
    # Callbacks (self objects to call back for memory between epochs)
    callbacks = [
        LearningRateDecayCallback(
            decay_epoch=hyperparams['lr_decay_epoch'],
            decay_factor=hyperparams['lr_decay_factor']
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_rul_output_loss',
            patience=hyperparams['patience'],
            restore_best_weights=False,
            verbose=1
        ),
        RestoreBestWeightsAtEnd(monitor="val_rul_output_loss", mode="min"),
    ]
    
    # ============================================
    # TRAINING
    # ============================================
    
    print("\nTraining...")
    start_time = time.time()
    
    history = model.fit(
        batch_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=hyperparams['max_epochs'],
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    epochs_trained = len(history.history['loss'])
    print(f"\nTraining complete in {training_time:.1f}s ({epochs_trained} epochs)")
    
    # ============================================
    # EVALUATION ON TARGET DOMAIN
    # ============================================
    
    print("\nEvaluating target domain...")
    
    train_metrics = evaluate_rul_split(
        model, target_data['X_train'], target_data['y_train'], target_data['rul_max']
    )
    val_metrics = evaluate_rul_split(
        model, target_data['X_val'], target_data['y_val'], target_data['rul_max']
    )
    
    # test set not evaluated during training (use --test-only later)
    test_metrics = None
    
    val_domain_acc = evaluate_domain_accuracy(
        model, source_data['X_val'], target_data['X_val']
    )
    
    print(f"\nTarget-domain RMSE summary:")
    print(f"Domain Acc: {val_domain_acc:.4f} (closer to 0.5 is better)")
    print(f"Train RMSE: {train_metrics['rmse']:.4f}")
    print(f"Val RMSE:   {val_metrics['rmse']:.4f}")
    
    
    # compare to baseline
    baseline_info = None
    if baseline_df is not None:
        baseline_row = baseline_df[baseline_df['n_engines'] == n_engines]
        if not baseline_row.empty:
            baseline_val_rmse = float(baseline_row['val_rmse'].iloc[0])
            val_gain = (baseline_val_rmse - val_metrics['rmse']) / baseline_val_rmse * 100.0
            
            baseline_info = {
                'baseline_val_rmse': baseline_val_rmse,
                'val_gain': val_gain,
            }
            
            print(f"\nAgainst Phase 2 baseline:")
            print(
                f"  Val RMSE:  {val_metrics['rmse']:.4f} vs {baseline_val_rmse:.4f} "
                f"({val_gain:+.2f}%)"
            )
    
    # ============================================
    # SAVE RESULTS
    # ============================================
    
    run_dir = OUTPUT_DIR / f"{n_engines}_engines"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(run_dir / "costa_dann_model.keras")
    
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2, default=float)
    
    save_predictions(
        run_dir / "train_predictions.csv",
        train_metrics['y_true'], train_metrics['y_pred']
    )
    save_predictions(
        run_dir / "val_predictions.csv",
        val_metrics['y_true'], val_metrics['y_pred']
    )
    # test predictions saved separately with --test-only
    
    metrics_dict = {
        'n_engines': n_engines,
        'source_dataset': SOURCE_DATASET,
        'target_dataset': TARGET_DATASET,
        'method': 'Costa_DANN_Fixed',
        'architecture': 'LSTM + GRL (Costa)',
        'train_rmse': train_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'train_r2': train_metrics['r2'],
        'train_cmapss': train_metrics['cmapss'],
        'train_auc_rmse': train_metrics['auc_rmse'],
        'train_rmse_bins': train_metrics['rmse_bins'],
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'val_r2': val_metrics['r2'],
        'val_cmapss': val_metrics['cmapss'],
        'val_auc_rmse': val_metrics['auc_rmse'],
        'val_rmse_bins': val_metrics['rmse_bins'],
        'val_domain_accuracy': val_domain_acc,
        'training_time_sec': float(training_time),
        'epochs_trained': int(epochs_trained),
        **hyperparams,
    }
    # test metrics added separately with --test-only
    
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, default=float)
    
    # print paste-friendly summary
    print("=================")
    print("SUMMARY")
    print("=================")
    print(
        f"Costa DANN: epochs={epochs_trained}/{hyperparams['max_epochs']}, "
        f"n_engines={n_engines}, "
        f"lambda={hyperparams['lambda_adversarial']}, "
        f"batch_size={hyperparams['batch_size']}, "
        f"lr={hyperparams['lr_rul']}, "
        f"optimizer=SGD"
    )
    print(
        f"Architecture: LSTM{hyperparams['lstm_units']}, "
        f"feature({hyperparams['feature_units']}), "
        f"rul{hyperparams['rul_units']}, "
        f"domain{hyperparams['domain_units']}"
    )
    print(f"Train RMSE: {train_metrics['rmse']:.4f}")
    print(f"Val RMSE:   {val_metrics['rmse']:.4f}")
    print(f"Domain Acc: {val_domain_acc:.4f}")
    
    if baseline_info:
        print(f"Baseline comparison: Val {baseline_info['val_gain']:+.2f}%")
    
    print(f"\nSaved outputs to: {run_dir}")
    return metrics_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Costa et al. DANN (LSTM + GRL) with project preprocessed data."
    )
    parser.add_argument(
        "--source", type=str, default=SOURCE_DATASET,
        help=f"Source dataset (default: {SOURCE_DATASET}). Supported: FD002, FD003",
    )
    parser.add_argument(
        "--target", type=str, default=TARGET_DATASET,
        help=f"Target dataset (default: {TARGET_DATASET}). Currently only FD001 supported",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Maximum training epochs (default: from Table 3 config)",
    )
    parser.add_argument(
        "--n_engines", type=int, nargs="*", default=None,
        help=f"Target engine counts to run. Default: {DEFAULT_ENGINE_COUNTS}",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help=f"Batch size (default: from Table 3 config)",
    )
    parser.add_argument(
        "--lambda_adversarial", type=float, default=None,
        help=f"GRL adversarial strength (default: from Table 3 config)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help=f"SGD learning rate (default: from Table 3 config)",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help=f"Early stopping patience (default: from Table 3 config)",
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help="Random seed",
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
    
    # Override source/target datasets from CLI
    global SOURCE_DATASET, TARGET_DATASET, OUTPUT_DIR
    SOURCE_DATASET = args.source
    TARGET_DATASET = args.target
    OUTPUT_DIR = SCRIPT_DIR / f"outputs_{SOURCE_DATASET}_to_{TARGET_DATASET}"
    
    # Get hyperparameters for this source-target pair
    try:
        hyperparams = dict(get_hyperparams(SOURCE_DATASET, TARGET_DATASET))
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Please use one of the supported configurations:")
        print("- FD002  FD001")
        print("- FD003  FD001")
        return
    
    # Override hyperparameters from CLI if provided
    if args.epochs:
        hyperparams['max_epochs'] = args.epochs
    if args.batch_size:
        hyperparams['batch_size'] = args.batch_size
    if args.lambda_adversarial:
        hyperparams['lambda_adversarial'] = args.lambda_adversarial
    if args.lr:
        hyperparams['lr_rul'] = args.lr
        hyperparams['lr_domain'] = args.lr
    if args.patience:
        hyperparams['patience'] = args.patience
    
    engine_counts = args.n_engines if args.n_engines else list(DEFAULT_ENGINE_COUNTS)
    
    # ========================================================
    # TEST-ONLY MODE: Load saved models and evaluate test sets
    # ========================================================
    if args.test_only:
        print(f"\nTEST-ONLY MODE: {SOURCE_DATASET} -> {TARGET_DATASET}")
        
        print(f"Engine counts: {engine_counts}")
        print("========")
        
        for n_engines in engine_counts:
            run_dir = OUTPUT_DIR / f"{n_engines}_engines"
            model_path = run_dir / "costa_dann_model.keras"
            
            if not model_path.exists():
                print(f"\nNo saved model for {n_engines} engines at {model_path}")
                print("Train the model first without --test-only flag.")
                continue
            
            print(f"\n{n_engines} engines: Loading model...")
            model = keras.models.load_model(model_path)
            
            # load target data
            target_data = load_cmapss_dataset(
                TARGET_DATASET, use_all_engines=False, n_engines=n_engines
            )
            
            # evaluate test set
            test_metrics = evaluate_rul_split(
                model, target_data['X_test'], target_data['y_test'], target_data['rul_max']
            )
            
            print(f"Test RMSE: {test_metrics['rmse']:.4f}")
            
            # load existing metrics and update
            metrics_path = run_dir / "metrics.json"
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # add test metrics
            metrics.update({
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2'],
                'test_cmapss': test_metrics['cmapss'],
                'test_auc_rmse': test_metrics['auc_rmse'],
                'test_rmse_bins': test_metrics['rmse_bins'],
            })
            
            # save updated metrics and predictions
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=float)
            
            save_predictions(
                run_dir / "test_predictions.csv",
                test_metrics['y_true'], test_metrics['y_pred']
            )
            
            print(f"Test metrics updated: {metrics_path}")
        
        print("\n=================")
        print("Test evaluation complete")
        print("=================")
        return
    
    # ========================================================
    # TRAINING MODE: Train DANN models
    # ========================================================
    print("=====================")
    print("COSTA ET AL. DANN")
    print("============")
    print(f"Source dataset: {SOURCE_DATASET}")
    print(f"Target dataset: {TARGET_DATASET}")
    print(f"Engine counts: {engine_counts}")
    print(f"\nConfiguration (Table 3):")
    print(f"LSTM: {hyperparams['lstm_layers']} layer(s), {hyperparams['lstm_units']}, dropout={hyperparams['lstm_dropout']}")
    print(f"Feature: {hyperparams['feature_units']} units")
    print(f"RUL: {hyperparams['rul_layers']} layer(s), {hyperparams['rul_units']}, dropout={hyperparams['rul_dropout']}")
    print(f"Domain: {hyperparams['domain_layers']} layers, {hyperparams['domain_units']}, dropout={hyperparams['domain_dropout']}")
    print(f"Lambda: {hyperparams['lambda_adversarial']}")
    print(f"Batch size: {hyperparams['batch_size']}")
    print(f"LR: {hyperparams['lr_rul']}")
    print(f"Epochs: {hyperparams['max_epochs']}")
    print(f"Optimizer: SGD (Costa et al.)")
    print(f"Output directory: {OUTPUT_DIR}")
    
    baseline_df = load_baseline()
    
    all_results = []
    for n_engines in engine_counts:
        try:
            metrics = train_costa_dann(
                n_engines=n_engines,
                hyperparams=hyperparams,
                baseline_df=baseline_df,
            )
            all_results.append(metrics)
        except Exception as exc:
            print(f"\nError training {n_engines} engines: {exc}")
            import traceback
            traceback.print_exc()
    
    if all_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(all_results)
        results_path = OUTPUT_DIR / "all_results.csv"
        results_df.to_csv(results_path, index=False)
        
        print("\n" + "="*40)
        print("FINAL RMSE SUMMARY")
        print("="*40)
        for row in all_results:
            summary = (
                f"{row['n_engines']:>2} engines | "
                f"Train: {row['train_rmse']:.3f} | "
                f"Val: {row['val_rmse']:.3f} | "
            )
            if 'test_rmse' in row:
                summary += f"Test: {row['test_rmse']:.3f}"
            else:
                summary += "Test: N/A"
            
            if baseline_df is not None:
                baseline_row = baseline_df[baseline_df['n_engines'] == row['n_engines']]
                if not baseline_row.empty:
                    baseline_val = float(baseline_row['val_rmse'].iloc[0])
                    baseline_test = float(baseline_row['test_rmse'].iloc[0])
                    val_gain = (baseline_val - row['val_rmse']) / baseline_val * 100.0
                    summary += f" | Base Val: {baseline_val:.3f} ({val_gain:+.2f}%)"
                    
                    if 'test_rmse' in row:
                        test_gain = (baseline_test - row['test_rmse']) / baseline_test * 100.0
                        summary += f" | Base Test: {baseline_test:.3f} ({test_gain:+.2f}%)"
            print(summary)
        print("=================")
        print(f"Saved combined results to: {results_path}")
    else:
        print("\nNo runs completed successfully.")

if __name__ == '__main__':
    main()
