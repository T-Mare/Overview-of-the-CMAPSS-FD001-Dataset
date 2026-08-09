import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import time
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from Utilities.Plots_Metrics import rmse, mae, r2, cmapss_score, rmse_by_bins_with_auc
from Utilities import config

# Import DANN modules
from dann_model import build_dann_model
from dann_utils import load_domain_data, create_mixed_dataset, load_phase2_baseline, print_data_summary

# ==========
# CONFIGURATION
# ==========

# Source and target datasets
SOURCE_DATASET = 'FD002'  # Large domain shift (6 operational conditions)
TARGET_DATASET = 'FD001'  # Target (1 operational condition)

# Engine counts to test (testing lambda=0.5 on higher data levels)
ENGINE_COUNTS = [10, 20, 40]  # Focus on problematic engine counts

# DANN hyperparameters
DANN_PARAMS = {
    'gru_units': 256,
    'dropout': 0.3,
    'learning_rate_feature': 0.0001,  # Feature extractor
    'learning_rate_rul': 0.0001,       # RUL predictor
    'learning_rate_domain': 0.001,     # Domain discriminator (higher LR)
    'lambda_adversarial': 0.5,         # Weight for adversarial loss (testing stronger adversarial signal)
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 5
}

# Paths
OUTPUT_BASE = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / TARGET_DATASET / f'DANN_{SOURCE_DATASET}_to_{TARGET_DATASET}'
PHASE2_BASELINE_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase2_Feature_Selection' / 'FD001' / 'Correlation_FS' / 'GRU' / 'GRU_metrics_summary.csv'

# Random seed
RANDOM_SEED = config.RANDOM_SEED
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ==========
# ADVERSARIAL TRAINING
# ==========

class DANNTrainer:
    
    def __init__(self, models, params):
        self.models = models
        self.params = params
        
        # Create optimizers
        self.optimizer_feature = keras.optimizers.Adam(learning_rate=params['learning_rate_feature'])
        self.optimizer_rul = keras.optimizers.Adam(learning_rate=params['learning_rate_rul'])
        self.optimizer_domain = keras.optimizers.Adam(learning_rate=params['learning_rate_domain'])
        
        # Loss functions
        self.rul_loss_fn = keras.losses.MeanSquaredError()
        self.domain_loss_fn = keras.losses.BinaryCrossentropy()
        
        # Metrics
        self.train_rul_loss = keras.metrics.Mean(name='train_rul_loss')
        self.train_domain_loss = keras.metrics.Mean(name='train_domain_loss')
        self.train_domain_acc = keras.metrics.BinaryAccuracy(name='train_domain_acc')
        
        self.val_rul_loss = keras.metrics.Mean(name='val_rul_loss')
        
    @tf.function
    def train_step(self, X_batch, y_batch, domain_batch):
        # =====================================================================
        # Step 1: Update RUL Predictor + Feature Extractor (minimize RUL loss)
        # =====================================================================
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            features = self.models['feature_extractor'](X_batch, training=True)
            rul_pred = self.models['rul_predictor'](features, training=True)
            
            # RUL loss
            rul_loss = self.rul_loss_fn(y_batch, rul_pred)
        
        # Compute gradients
        rul_grad = tape.gradient(rul_loss, self.models['rul_predictor'].trainable_variables)
        feature_grad_rul = tape.gradient(rul_loss, self.models['feature_extractor'].trainable_variables)
        
        # Apply gradients
        self.optimizer_rul.apply_gradients(zip(rul_grad, self.models['rul_predictor'].trainable_variables))
        self.optimizer_feature.apply_gradients(zip(feature_grad_rul, self.models['feature_extractor'].trainable_variables))
        
        del tape
        
        # =====================================================================
        # Step 2: Update Domain Discriminator (maximize domain classification)
        # =====================================================================
        with tf.GradientTape() as tape:
            # Forward pass (features should be extracted again to get updated features)
            features = self.models['feature_extractor'](X_batch, training=True)
            domain_pred = self.models['domain_discriminator'](features, training=True)
            
            # Domain loss
            domain_loss = self.domain_loss_fn(domain_batch, domain_pred)
        
        # Compute and apply gradients
        domain_grad = tape.gradient(domain_loss, self.models['domain_discriminator'].trainable_variables)
        self.optimizer_domain.apply_gradients(zip(domain_grad, self.models['domain_discriminator'].trainable_variables))
        
        # =====================================================================
        # Step 3: Update Feature Extractor (fool discriminator - adversarial)
        # =====================================================================
        with tf.GradientTape() as tape:
            # Forward pass
            features = self.models['feature_extractor'](X_batch, training=True)
            domain_pred = self.models['domain_discriminator'](features, training=True)
            
            # Adversarial loss (reverse labels to fool discriminator)
            # We want features that make the discriminator confused
            reversed_domain = 1.0 - domain_batch  # Flip labels
            adversarial_loss = self.domain_loss_fn(reversed_domain, domain_pred)
            adversarial_loss = self.params['lambda_adversarial'] * adversarial_loss
        
        # Compute and apply gradients (only to feature extractor)
        feature_grad_adv = tape.gradient(adversarial_loss, self.models['feature_extractor'].trainable_variables)
        self.optimizer_feature.apply_gradients(zip(feature_grad_adv, self.models['feature_extractor'].trainable_variables))
        
        # Update metrics
        self.train_rul_loss.update_state(rul_loss)
        self.train_domain_loss.update_state(domain_loss)
        self.train_domain_acc.update_state(domain_batch, domain_pred)
    
    @tf.function
    def val_step(self, X_batch, y_batch):
        # Forward pass
        features = self.models['feature_extractor'](X_batch, training=False)
        rul_pred = self.models['rul_predictor'](features, training=False)
        
        # Loss
        rul_loss = self.rul_loss_fn(y_batch, rul_pred)
        
        # Update metrics
        self.val_rul_loss.update_state(rul_loss)
    
    def reset_metrics(self):
        self.train_rul_loss.reset_states()
        self.train_domain_loss.reset_states()
        self.train_domain_acc.reset_states()
        self.val_rul_loss.reset_states()

def train_dann(source_dataset, target_dataset, n_engines, baseline_df=None):
    print("\n" + "="*40)
    print(f"DANN: DOMAIN-ADVERSARIAL TRAINING")
    print(f"Source: {source_dataset}  Target: {target_dataset} ({n_engines} engines)")
    print("="*40)
    
    # Create output directory
    save_dir = OUTPUT_BASE
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'predictions').mkdir(exist_ok=True)
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    print("\nLoading data...")
    domain_data = load_domain_data(source_dataset, target_dataset, n_engines)
    mixed_data = create_mixed_dataset(domain_data, shuffle=True)
    print_data_summary(domain_data)
    
    # =========================================================================
    # MODEL BUILDING
    # =========================================================================
    
    print("\nBuilding DANN model...")
    models = build_dann_model(
        input_shape=(30, 14),
        gru_units=DANN_PARAMS['gru_units'],
        dropout=DANN_PARAMS['dropout']
    )
    print(f"Model built: {models['feature_extractor'].count_params():,} parameters in feature extractor")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    print("\nTraining DANN...")
    print(f"Batch size: {DANN_PARAMS['batch_size']}")
    print(f"Max epochs: {DANN_PARAMS['epochs']}")
    print(f"Early stopping patience: {DANN_PARAMS['early_stopping_patience']}")
    print(f"Lambda (adversarial weight): {DANN_PARAMS['lambda_adversarial']}")
    
    trainer = DANNTrainer(models, DANN_PARAMS)
    
    # Create datasets
    batch_size = DANN_PARAMS['batch_size']
    train_dataset = tf.data.Dataset.from_tensor_slices((
        mixed_data['X_train_mixed'],
        mixed_data['y_train_mixed'],
        mixed_data['domain_train_mixed']
    )).shuffle(10000).batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        domain_data['X_val'],
        domain_data['y_val']
    )).batch(batch_size)
    
    # Training loop
    history = {
        'train_rul_loss': [],
        'train_domain_loss': [],
        'train_domain_acc': [],
        'val_rul_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(DANN_PARAMS['epochs']):
        # Train
        trainer.reset_metrics()
        
        for X_batch, y_batch, domain_batch in train_dataset:
            trainer.train_step(X_batch, y_batch, domain_batch)
        
        # Validate
        for X_batch, y_batch in val_dataset:
            trainer.val_step(X_batch, y_batch)
        
        # Get metrics
        train_rul_loss = trainer.train_rul_loss.result().numpy()
        train_domain_loss = trainer.train_domain_loss.result().numpy()
        train_domain_acc = trainer.train_domain_acc.result().numpy()
        val_rul_loss = trainer.val_rul_loss.result().numpy()
        
        # Save history
        history['train_rul_loss'].append(float(train_rul_loss))
        history['train_domain_loss'].append(float(train_domain_loss))
        history['train_domain_acc'].append(float(train_domain_acc))
        history['val_rul_loss'].append(float(val_rul_loss))
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{DANN_PARAMS['epochs']}: "
              f"RUL Loss: {train_rul_loss:.4f} -> {val_rul_loss:.4f} | "
              f"Domain Loss: {train_domain_loss:.4f} | "
              f"Domain Acc: {train_domain_acc:.3f}")
        
        # Early stopping
        if val_rul_loss < best_val_loss:
            best_val_loss = val_rul_loss
            patience_counter = 0
            # Save best model weights (feature extractor + RUL predictor)
            best_weights = {
                'feature_extractor': models['feature_extractor'].get_weights(),
                'rul_predictor': models['rul_predictor'].get_weights()
            }
        else:
            patience_counter += 1
            if patience_counter >= DANN_PARAMS['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    
    # Restore best weights
    models['feature_extractor'].set_weights(best_weights['feature_extractor'])
    models['rul_predictor'].set_weights(best_weights['rul_predictor'])
    
    print(f"\n Training complete in {training_time:.1f}s ({len(history['train_rul_loss'])} epochs)")
    print(f"Best val RUL loss: {best_val_loss:.4f}")
    print(f"Final domain accuracy: {history['train_domain_acc'][-1]:.3f} (target: 0.5-0.6 for confusion)")
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
    print("\nEvaluating DANN model...")
    
    # Validation predictions
    val_features = models['feature_extractor'].predict(domain_data['X_val'], verbose=0)
    val_predictions = models['rul_predictor'].predict(val_features, verbose=0).flatten()
    
    # Test predictions
    test_features = models['feature_extractor'].predict(domain_data['X_test'], verbose=0)
    test_predictions = models['rul_predictor'].predict(test_features, verbose=0).flatten()
    
    # Calculate metrics
    val_rmse = rmse(domain_data['y_val'], val_predictions)
    val_mae_score = mae(domain_data['y_val'], val_predictions)
    val_r2 = r2(domain_data['y_val'], val_predictions)
    val_cmapss = cmapss_score(domain_data['y_val'], val_predictions, reduction='sum')
    _, val_auc_rmse = rmse_by_bins_with_auc(domain_data['y_val'], val_predictions, config.RUL_BINS)
    
    test_rmse = rmse(domain_data['y_test'], test_predictions)
    test_mae_score = mae(domain_data['y_test'], test_predictions)
    test_r2 = r2(domain_data['y_test'], test_predictions)
    test_cmapss = cmapss_score(domain_data['y_test'], test_predictions, reduction='sum')
    _, test_auc_rmse = rmse_by_bins_with_auc(domain_data['y_test'], test_predictions, config.RUL_BINS)
    
    print(f"Validation - RMSE: {val_rmse:.4f}, MAE: {val_mae_score:.4f}, R²: {val_r2:.4f}, AUC-RMSE: {val_auc_rmse:.4f}")
    print(f"Test       - RMSE: {test_rmse:.4f}, MAE: {test_mae_score:.4f}, R²: {test_r2:.4f}, AUC-RMSE: {test_auc_rmse:.4f}")
    
    # =========================================================================
    # BASELINE COMPARISON
    # =========================================================================
    
    if baseline_df is not None:
        baseline_row = baseline_df[baseline_df['n_engines'] == n_engines]
        if not baseline_row.empty:
            baseline_val_rmse = baseline_row['val_rmse'].values[0]
            baseline_test_rmse = baseline_row['test_rmse'].values[0]
            
            val_improvement = ((baseline_val_rmse - val_rmse) / baseline_val_rmse * 100)
            test_improvement = ((baseline_test_rmse - test_rmse) / baseline_test_rmse * 100)
            
            print(f"\n DANN Gain vs Phase 2 Baseline (GRU + Correlation_FS, no TL):")
            print(f"Val RMSE:  {val_rmse:.4f} vs {baseline_val_rmse:.4f}  {val_improvement:+.2f}% {'' if val_improvement > 0 else ''}")
            print(f"Test RMSE: {test_rmse:.4f} vs {baseline_test_rmse:.4f}  {test_improvement:+.2f}% {'' if test_improvement > 0 else ''}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    # Save model
    model_path = save_dir / f'dann_gru_model_{n_engines}engines.keras'
    models['combined_rul'].save(model_path)
    
    # Save history
    history_path = save_dir / f'dann_gru_history_{n_engines}engines.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save metrics
    metrics = {
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'n_engines': n_engines,
        'method': 'DANN',
        'architecture': 'GRU',
        'val_rmse': float(val_rmse),
        'val_mae': float(val_mae_score),
        'val_r2': float(val_r2),
        'val_cmapss': float(val_cmapss),
        'val_auc_rmse': float(val_auc_rmse) if val_auc_rmse is not None else None,
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae_score),
        'test_r2': float(test_r2),
        'test_cmapss': float(test_cmapss),
        'test_auc_rmse': float(test_auc_rmse) if test_auc_rmse is not None else None,
        'training_time_sec': float(training_time),
        'total_epochs': len(history['train_rul_loss']),
        'final_domain_acc': float(history['train_domain_acc'][-1]),
        'hyperparameters': DANN_PARAMS
    }
    
    metrics_path = save_dir / f'dann_gru_metrics_{n_engines}engines.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    pd_val = pd.DataFrame({
        'actual': domain_data['y_val'],
        'predicted': val_predictions
    })
    pd_val.to_csv(save_dir / 'predictions' / f'dann_val_predictions_{n_engines}engines.csv', index=False)
    
    pd_test = pd.DataFrame({
        'actual': domain_data['y_test'],
        'predicted': test_predictions
    })
    pd_test.to_csv(save_dir / 'predictions' / f'dann_test_predictions_{n_engines}engines.csv', index=False)
    
    print(f"\n DANN results saved: {save_dir}")
    
    return metrics

# ==========
# BATCH PROCESSING
# ==========

def run_all_experiments():
    
    print("\n" + "="*40)
    print("PHASE 4: DANN - DOMAIN-ADVERSARIAL NEURAL NETWORK")
    print("="*40)
    print(f"Source: {SOURCE_DATASET}")
    print(f"Target: {TARGET_DATASET}")
    print(f"Engine counts: {ENGINE_COUNTS}")
    
    # Load Phase 2 baseline for comparison
    baseline_df = load_phase2_baseline()
    
    all_results = []
    
    for n_engines in ENGINE_COUNTS:
        try:
            metrics = train_dann(SOURCE_DATASET, TARGET_DATASET, n_engines, baseline_df)
            all_results.append(metrics)
        except Exception as e:
            print(f"\n Error: {SOURCE_DATASET}  {TARGET_DATASET} ({n_engines} engines): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = OUTPUT_BASE.parent / f'DANN_GRU_{SOURCE_DATASET}_all_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\n All DANN results saved: {results_path}")
        
        # Print summary
        print("\n" + "="*40)
        print("DANN EXPERIMENT SUMMARY")
        print("="*40)
        print(f"Total experiments: {len(all_results)}")
        print(f"Average val RMSE: {results_df['val_rmse'].mean():.4f}")
        print(f"Average test RMSE: {results_df['test_rmse'].mean():.4f}")
        print(f"Average domain accuracy: {results_df['final_domain_acc'].mean():.3f} (target: 0.5-0.6)")
        print("="*40)

# Import pandas for saving predictions
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN: Domain-Adversarial Training')
    parser.add_argument('--n_engines', type=int, choices=ENGINE_COUNTS + [0], 
                       default=0, help='Number of engines for training (0=all)')
    
    args = parser.parse_args()
    
    if args.n_engines == 0:
        # Run all experiments
        run_all_experiments()
    else:
        # Run single experiment
        baseline_df = load_phase2_baseline()
        train_dann(SOURCE_DATASET, TARGET_DATASET, args.n_engines, baseline_df)

