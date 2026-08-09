"""
Configuration file for all hyperparameters and settings
Single source of truth for reproducibility
"""

# =============================================================================
# DATA PREPROCESSING HYPERPARAMETERS
# =============================================================================

# RUL capping threshold - maximum RUL value to prevent long flat regions
RUL_CAP = 125

# Window size for sequential models (LSTM, GRU, CNN)
WINDOW_SIZE = 30


# =============================================================================
# DATA PATHS
# =============================================================================

# Base paths for data (relative to project root for cross-platform compatibility)
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent  # Submission_Tiaan_Mare_34584757/
DATA_BASE_PATH = _PROJECT_ROOT / "CodeBase_Experiments" / "0_Data_Processing" / "Data_CMAPSS" / "2_Cleaned_Data"

NON_WINDOWED_DATA_PATH = DATA_BASE_PATH / "Non_Windowed"
WINDOWED_DATA_PATH = DATA_BASE_PATH / "Windowed"


# =============================================================================
# RESULTS PATHS
# =============================================================================

RESULTS_BASE_PATH = _PROJECT_ROOT / "Results"


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# Training data percentages to test (DEPRECATED - use ENGINE_COUNTS instead)
DATA_PERCENTAGES = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

# Engine-based data subsets for Phase 1 experiments
# FD001 has 80 training engines - we test with different numbers of complete engines
ENGINE_COUNTS_HIGH = [80, 70, 60, 50, 40, 30, 20, 10]  # Medium to full data
ENGINE_COUNTS_LOW = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]    # Extreme scarcity
ENGINE_COUNTS_ALL = [80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # Complete range

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# MODEL HYPERPARAMETERS - SEARCH GRIDS
# =============================================================================

# Ridge Regression - alpha values to search
RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Lasso Regression - alpha values to search
LASSO_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# ElasticNet - alpha and l1_ratio values to search
ELASTICNET_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ELASTICNET_L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]  # 0=Ridge, 1=Lasso, 0.5=equal mix


# =============================================================================
# OPTUNA HYPERPARAMETER TUNING CONFIGURATION
# =============================================================================

# Optuna settings
OPTUNA_N_TRIALS = 50  # Number of trials per model (reduced from 100 for faster tuning)
OPTUNA_N_JOBS = 1     # Sequential jobs (parallel causes crashes on Windows with XGBoost/LightGBM)
OPTUNA_TIMEOUT = None  # No timeout (let it complete all trials)
OPTUNA_EARLY_STOPPING_PATIENCE = 15  # Stop study if no improvement after 15 consecutive trials (for fair comparison across all models)

# Optuna storage (SQLite database for persistence and thesis appendix)
OPTUNA_STORAGE = "sqlite:///optuna_studies.db"


# =============================================================================
# TREE-BASED MODELS - OPTUNA SEARCH SPACES
# =============================================================================
# Design Philosophy: Library defaults as starting points, with directional
# exploration toward ranges known to improve performance based on empirical
# evidence. See: Thesis/Notes/Baseline_Experiments/Tree_Based_Models_Notes.md

# Random Forest Hyperparameter Search Space
# sklearn defaults: n_estimators=100, max_depth=None, min_samples_split=2,
#                   min_samples_leaf=1, max_features=1.0 (for regression)
RANDOM_FOREST_SEARCH_SPACE = {
    'n_estimators': ('int', 100, 500),              # Explore upward from default
    'max_depth': ('int', 5, 20),                    # Default=None (unlimited); we limit for regularization
    'min_samples_split': ('int', 2, 20),            # From default upward (regularization)
    'min_samples_leaf': ('int', 1, 10),             # From default upward (regularization)
    'max_features': ('categorical', ['sqrt', 'log2', None]),  # Feature sampling strategies
    # Fixed parameters (not tuned):
    # - bootstrap: True (standard practice)
    # - criterion: 'squared_error' (regression default)
    # - random_state: RANDOM_SEED
    # - n_jobs: -1 (parallel tree building)
}

# XGBoost Hyperparameter Search Space
# XGBoost defaults: learning_rate=0.3, max_depth=6, n_estimators=100,
#                   subsample=1.0, colsample_bytree=1.0, min_child_weight=1, gamma=0
XGBOOST_SEARCH_SPACE = {
    'n_estimators': ('int', 100, 1000),             # Extended for lower learning rates
    'max_depth': ('int', 3, 8),                     # Shallow trees (weak learners)
    'learning_rate': ('float_log', 0.01, 0.3),      # Low to default (log scale)
    'subsample': ('float', 0.6, 1.0),               # Stochastic boosting
    'colsample_bytree': ('float', 0.6, 1.0),        # Feature sampling per tree
    'min_child_weight': ('int', 1, 7),              # Regularization (from default upward)
    'gamma': ('float', 0, 5),                       # Min loss reduction (from default upward)
    # Fixed parameters (not tuned):
    # - objective: 'reg:squarederror' (regression)
    # - random_state: RANDOM_SEED
    # - n_jobs: -1
    # Excluded: reg_alpha, reg_lambda (no regularization benefit on FD001)
}

# LightGBM Hyperparameter Search Space
# LightGBM defaults: num_leaves=31, learning_rate=0.1, n_estimators=100,
#                    subsample=1.0, colsample_bytree=1.0, min_child_samples=20
LIGHTGBM_SEARCH_SPACE = {
    'n_estimators': ('int', 100, 1000),             # Extended for lower learning rates
    'num_leaves': ('int', 20, 100),                 # Key parameter (leaf-wise growth)
    'learning_rate': ('float_log', 0.01, 0.3),      # Low to high (log scale)
    'subsample': ('float', 0.6, 1.0),               # Bagging fraction
    'colsample_bytree': ('float', 0.6, 1.0),        # Feature fraction
    'min_child_samples': ('int', 5, 50),            # Regularization (RF analogue)
    # Fixed parameters (not tuned):
    # - objective: 'regression'
    # - metric: 'rmse'
    # - random_state: RANDOM_SEED
    # - n_jobs: -1
    # - verbose: -1
    # Excluded: reg_alpha, reg_lambda (no regularization benefit on FD001)
    # Excluded: max_depth (controlled by num_leaves in leaf-wise growth)
}


# =============================================================================
# TREE-BASED MODELS - FIXED PARAMETERS (NOT TUNED)
# =============================================================================
# These parameters are fixed for all experiments to ensure fair comparison
# and reproducibility

# Random Forest Fixed Parameters
RANDOM_FOREST_FIXED_PARAMS = {
    'random_state': RANDOM_SEED,
    'n_jobs': 1,               # Sequential to avoid Windows threading issues
    'bootstrap': True,         # Standard practice for RF
    'criterion': 'squared_error'  # Regression default
}

# XGBoost Fixed Parameters
XGBOOST_FIXED_PARAMS = {
    'random_state': RANDOM_SEED,
    'n_jobs': 1,               # XGBoost handles parallelism internally
    'objective': 'reg:squarederror',  # Regression objective
    'verbosity': 0             # Suppress output
}

# LightGBM Fixed Parameters
LIGHTGBM_FIXED_PARAMS = {
    'random_state': RANDOM_SEED,
    'n_jobs': 1,               # Sequential to avoid Windows threading issues
    'objective': 'regression', # Regression objective
    'metric': 'rmse',          # Primary metric
    'verbosity': -1,           # Suppress output
    'force_col_wise': True     # Recommended for small datasets (<10k rows)
}


# =============================================================================
# DEEP LEARNING MODELS - OPTUNA SEARCH SPACES
# =============================================================================
# Design Philosophy: Keras/TensorFlow defaults as starting points, scaled to 
# dataset characteristics (720 flattened input features, 16,561 samples).
# Pragmatic approach: Start simple, add complexity only if needed.
# See: Thesis/Notes/Baseline_Experiments/Deep_Learning_Models_Notes.md

# ANN (Artificial Neural Network) Hyperparameter Search Space
# Input: Flattened (30 timesteps × 24 features = 720 features)
# Output: Single RUL value (scalar regression)
# Note: Unit ranges use powers of 2 (64, 128, 256) for computational efficiency
#       and alignment with GPU/memory architectures
ANN_SEARCH_SPACE = {
    # Architecture
    'n_hidden_layers': ('categorical', [1, 2]),     # 1-2 layers to avoid overfitting on small data
    'units_layer1': ('categorical', [64, 128, 256]), # First hidden layer (powers of 2)
    'units_layer2': ('categorical', [32, 64, 128]), # Second hidden layer (if used)
    
    # Regularization
    'dropout_rate': ('categorical', [0.1, 0.2, 0.3, 0.4, 0.5]), # Dropout after each hidden layer
    
    # Optimization
    'learning_rate': ('categorical', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]),     # Adam learning rate (default=1e-3)
    'batch_size': ('categorical', [32, 64, 128, 256]),
    
    # Fixed parameters (not tuned):
    # - activation: 'relu' (standard, fast, works well)
    # - optimizer: Adam (with tuned learning_rate)
    # - loss: 'mse' (mean squared error)
    # - output_activation: 'linear' (regression)
    # - epochs: 100 (with early stopping)
    # - early_stopping: patience=10, monitor='val_loss', restore_best_weights=True
    # Excluded: L2 regularization, batch normalization (keep simple, add later if needed)
}

# ANN Fixed Parameters
ANN_FIXED_PARAMS = {
    'activation': 'relu',                   # Standard activation for hidden layers
    'optimizer': 'adam',                    # Adaptive learning rate optimizer
    'loss': 'mse',                          # Mean squared error for regression
    'output_activation': 'linear',          # Regression output (no activation)
    'epochs': 50,                           # Max epochs for tuning (reduced from 100 for faster hyperparameter search)
    'early_stopping_patience': 10,          # Stop if val_loss doesn't improve for 10 epochs
    'monitor': 'val_loss',                  # Metric to monitor for early stopping
    'restore_best_weights': True,           # Load best model weights after training
    'random_seed': RANDOM_SEED,             # Reproducibility
}

# RNN (Recurrent Neural Network) Hyperparameter Search Space
# Input: Sequential (30 timesteps × 24 features)
# Output: Single RUL value (scalar regression)
# Note: Unit ranges use powers of 2 (64, 128, 256) - same as ANN for fair comparison
RNN_SEARCH_SPACE = {
    # Architecture
    'n_recurrent_layers': ('categorical', [1, 2]),  # 1-2 layers to avoid overfitting
    'units_layer1': ('categorical', [64, 128, 256]), # First RNN layer (same as ANN)
    'units_layer2': ('categorical', [32, 64, 128]), # Second RNN layer (if used)
    
    # Regularization
    'dropout_rate': ('categorical', [0.1, 0.2, 0.3, 0.4, 0.5]), # Standard dropout (applied to outputs)
    
    # Optimization
    'learning_rate': ('categorical', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]),     # Adam learning rate (default=1e-3)
    'batch_size': ('categorical', [32, 64, 128, 256]),
    
    # Fixed parameters (not tuned):
    # - activation: 'tanh' (RNN standard, better than ReLU for recurrent connections)
    # - optimizer: Adam (with tuned learning_rate)
    # - loss: 'mse' (mean squared error)
    # - output_activation: 'linear' (regression)
    # - epochs: 100 (with early stopping)
    # - early_stopping: patience=10, monitor='val_loss', restore_best_weights=True
}

# RNN Fixed Parameters
RNN_FIXED_PARAMS = {
    'activation': 'tanh',                   # RNN standard (bounds outputs to [-1,1], prevents exploding activations)
    'recurrent_dropout': 0.1,               # Fixed recurrent dropout (some regularization on recurrent connections)
    'optimizer': 'adam',                    # Adaptive learning rate optimizer
    'loss': 'mse',                          # Mean squared error for regression
    'output_activation': 'linear',          # Regression output (no activation)
    'epochs': 50,                           # Max epochs for tuning (reduced from 100 for faster hyperparameter search)
    'early_stopping_patience': 10,          # Stop if val_loss doesn't improve for 10 epochs
    'monitor': 'val_loss',                  # Metric to monitor for early stopping
    'restore_best_weights': True,           # Load best model weights after training
    'random_seed': RANDOM_SEED,             # Reproducibility
}

# TCN (Temporal Convolutional Network) Hyperparameter Search Space
# Input: Sequential (30 timesteps × 24 features)
# Output: Single RUL value (scalar regression)
# Note: Filters (32-128) differ from ANN/RNN units - filters detect spatial patterns
TCN_SEARCH_SPACE = {
    # Architecture
    'n_filters': ('categorical', [32, 64, 128]),    # Number of convolutional filters (powers of 2)
    'n_blocks': ('categorical', [2, 3]),            # Number of TCN blocks (2=8 layers, 3=12 layers)
    
    # Regularization
    'dropout_rate': ('categorical', [0.1, 0.2, 0.3, 0.4, 0.5]), # Spatial dropout (applied after each block)
    
    # Optimization
    'learning_rate': ('categorical', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]),     # Adam learning rate (default=1e-3)
    'batch_size': ('categorical', [32, 64, 128, 256]),
    
    # Fixed parameters (not tuned):
    # - kernel_size: 3 (standard TCN, not tuned)
    # - dilation_rates: [1, 2, 4, 8] per block (exponential growth, standard TCN)
    # - activation: 'relu' (CNN standard)
    # - use_residual: True (residual connections, TCN standard)
    # - padding: 'causal' (prevents future information leakage)
}

# TCN Fixed Parameters
TCN_FIXED_PARAMS = {
    'kernel_size': 3,                       # Convolution window size (standard, not tuned)
    'dilation_rates': [1, 2, 4, 8],         # Exponential dilation per block (receptive field = 31 timesteps)
    'activation': 'relu',                   # CNN standard activation
    'use_residual': True,                   # Residual connections (helps gradient flow)
    'padding': 'causal',                    # Causal convolutions (no future leakage)
    'optimizer': 'adam',                    # Adaptive learning rate optimizer
    'loss': 'mse',                          # Mean squared error for regression
    'output_activation': 'linear',          # Regression output (no activation)
    'epochs': 50,                           # Max epochs for tuning (reduced from 100 for faster hyperparameter search)
    'early_stopping_patience': 10,          # Stop if val_loss doesn't improve for 10 epochs
    'monitor': 'val_loss',                  # Metric to monitor for early stopping
    'restore_best_weights': True,           # Load best model weights after training
    'random_seed': RANDOM_SEED,             # Reproducibility
}

# LSTM (Long Short-Term Memory) Hyperparameter Search Space
# Input: Sequential (30 timesteps × 24 features)
# Output: Single RUL value (scalar regression)
# Note: Same search space as RNN for fair comparison (isolates effect of gating mechanism)
LSTM_SEARCH_SPACE = {
    # Architecture
    'n_lstm_layers': ('categorical', [1, 2]),       # 1-2 LSTM layers
    'units_layer1': ('categorical', [64, 128, 256]), # First LSTM layer (same as RNN)
    'units_layer2': ('categorical', [32, 64, 128]), # Second LSTM layer (if used)
    
    # Regularization
    'dropout_rate': ('categorical', [0.1, 0.2, 0.3, 0.4, 0.5]), # Standard dropout (outputs)
    
    # Optimization
    'learning_rate': ('categorical', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]),     # Adam learning rate
    'batch_size': ('categorical', [32, 64, 128, 256]),
    
    # Fixed parameters (not tuned):
    # - activation: 'tanh' (cell state and output, LSTM standard)
    # - recurrent_activation: 'sigmoid' (gate activations, LSTM standard)
    # - optimizer: Adam (with tuned learning_rate)
    # - loss: 'mse' (mean squared error)
    # - output_activation: 'linear' (regression)
    # - epochs: 100 (with early stopping)
}

# LSTM Fixed Parameters
LSTM_FIXED_PARAMS = {
    'activation': 'tanh',                   # LSTM standard (cell state and output)
    'recurrent_activation': 'sigmoid',      # Gate activation (forget, input, output gates)
    'recurrent_dropout': 0.1,               # Fixed recurrent dropout (some regularization on recurrent connections)
    'optimizer': 'adam',                    # Adaptive learning rate optimizer
    'loss': 'mse',                          # Mean squared error for regression
    'output_activation': 'linear',          # Regression output (no activation)
    'epochs': 50,                           # Max epochs for tuning (reduced from 100 for faster hyperparameter search)
    'early_stopping_patience': 10,          # Stop if val_loss doesn't improve for 10 epochs
    'monitor': 'val_loss',                  # Metric to monitor for early stopping
    'restore_best_weights': True,           # Load best model weights after training
    'random_seed': RANDOM_SEED,             # Reproducibility
}

# BiLSTM (Bidirectional LSTM) Hyperparameter Search Space
# Input: Sequential (30 timesteps × 24 features)
# Output: Single RUL value (scalar regression)
# Note: Same search space as LSTM - bidirectional processing doubles parameters internally
BILSTM_SEARCH_SPACE = {
    # Architecture
    'n_bilstm_layers': ('categorical', [1, 2]),     # 1-2 BiLSTM layers
    'units_layer1': ('categorical', [64, 128, 256]), # First BiLSTM layer (same as LSTM)
    'units_layer2': ('categorical', [32, 64, 128]), # Second BiLSTM layer (if used)
    
    # Regularization
    'dropout_rate': ('categorical', [0.1, 0.2, 0.3, 0.4, 0.5]), # Standard dropout (outputs)
    
    # Optimization
    'learning_rate': ('categorical', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]),     # Adam learning rate
    'batch_size': ('categorical', [32, 64, 128, 256]),
    
    # Fixed parameters (not tuned):
    # - merge_mode: 'concat' (concatenate forward + backward outputs, standard practice)
    # - activation: 'tanh' (same as LSTM)
    # - recurrent_activation: 'sigmoid' (same as LSTM)
}

# BiLSTM Fixed Parameters
BILSTM_FIXED_PARAMS = {
    'merge_mode': 'concat',                 # Concatenate forward + backward (standard BiLSTM)
    'activation': 'tanh',                   # Same as LSTM
    'recurrent_activation': 'sigmoid',      # Same as LSTM
    'recurrent_dropout': 0.1,               # Fixed recurrent dropout (some regularization on recurrent connections)
    'optimizer': 'adam',                    # Adaptive learning rate optimizer
    'loss': 'mse',                          # Mean squared error for regression
    'output_activation': 'linear',          # Regression output (no activation)
    'epochs': 50,                           # Max epochs for tuning (reduced from 100 for faster hyperparameter search)
    'early_stopping_patience': 10,          # Stop if val_loss doesn't improve for 10 epochs
    'monitor': 'val_loss',                  # Metric to monitor for early stopping
    'restore_best_weights': True,           # Load best model weights after training
    'random_seed': RANDOM_SEED,             # Reproducibility
}

# GRU (Gated Recurrent Unit) Hyperparameter Search Space
# Input: Sequential (30 timesteps × 24 features)
# Output: Single RUL value (scalar regression)
# Note: Same search space as LSTM/RNN - GRU is simplified LSTM (2 gates vs 3, fewer parameters)
GRU_SEARCH_SPACE = {
    # Architecture
    'n_gru_layers': ('categorical', [1, 2]),        # 1-2 GRU layers
    'units_layer1': ('categorical', [64, 128, 256]), # First GRU layer (same as LSTM/RNN)
    'units_layer2': ('categorical', [32, 64, 128]), # Second GRU layer (if used)
    
    # Regularization
    'dropout_rate': ('categorical', [0.1, 0.2, 0.3, 0.4, 0.5]), # Standard dropout (outputs)
    
    # Optimization
    'learning_rate': ('categorical', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]),     # Adam learning rate
    'batch_size': ('categorical', [32, 64, 128, 256]),
    
    # Fixed parameters (not tuned):
    # - activation: 'tanh' (hidden state, GRU standard)
    # - recurrent_activation: 'sigmoid' (gate activations, GRU standard)
    # - GRU has 2 gates (reset, update) vs LSTM's 3 gates (forget, input, output)
}

# GRU Fixed Parameters
GRU_FIXED_PARAMS = {
    'activation': 'tanh',                   # Hidden state activation (GRU standard)
    'recurrent_activation': 'sigmoid',      # Gate activation (reset, update gates)
    'recurrent_dropout': 0.1,               # Fixed recurrent dropout (some regularization on recurrent connections)
    'optimizer': 'adam',                    # Adaptive learning rate optimizer
    'loss': 'mse',                          # Mean squared error for regression
    'output_activation': 'linear',          # Regression output (no activation)
    'epochs': 50,                           # Max epochs for tuning (reduced from 100 for faster hyperparameter search)
    'early_stopping_patience': 10,          # Stop if val_loss doesn't improve for 10 epochs
    'monitor': 'val_loss',                  # Metric to monitor for early stopping
    'restore_best_weights': True,           # Load best model weights after training
    'random_seed': RANDOM_SEED,             # Reproducibility
}

# CNN (1D Convolutional Neural Network) Hyperparameter Search Space
# Input: Sequential (30 timesteps × 24 features)
# Output: Single RUL value (scalar regression)
# Note: Standard 1D CNN with MaxPooling (simpler alternative to TCN)
CNN_SEARCH_SPACE = {
    # Architecture
    'n_conv_layers': ('categorical', [2, 3]),       # 2-3 convolutional layers
    'n_filters': ('categorical', [32, 64, 128]),    # Number of filters (powers of 2)
    'kernel_size': ('categorical', [3, 5]),         # Convolution window size
    
    # Regularization
    'dropout_rate': ('categorical', [0.1, 0.2, 0.3, 0.4, 0.5]), # Dropout after each conv block
    
    # Optimization
    'learning_rate': ('categorical', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]),     # Adam learning rate
    'batch_size': ('categorical', [32, 64, 128, 256]),
    
    # Fixed parameters (not tuned):
    # - pool_size: 2 (MaxPooling reduces sequence length by half, standard practice)
    # - activation: 'relu' (CNN standard)
    # - padding: 'same' (maintains sequence length before pooling)
    # - pool_type: 'max' (MaxPooling standard for CNNs)
}

# CNN Fixed Parameters
CNN_FIXED_PARAMS = {
    'pool_size': 2,                         # MaxPooling size (reduces sequence by half: 30→15→7)
    'activation': 'relu',                   # CNN standard activation
    'padding': 'same',                      # Maintain sequence length before pooling
    'pool_type': 'max',                     # MaxPooling (standard for CNNs)
    'optimizer': 'adam',                    # Adaptive learning rate optimizer
    'loss': 'mse',                          # Mean squared error for regression
    'output_activation': 'linear',          # Regression output (no activation)
    'epochs': 50,                           # Max epochs for tuning (reduced from 100 for faster hyperparameter search)
    'early_stopping_patience': 10,          # Stop if val_loss doesn't improve for 10 epochs
    'monitor': 'val_loss',                  # Metric to monitor for early stopping
    'restore_best_weights': True,           # Load best model weights after training
    'random_seed': RANDOM_SEED,             # Reproducibility
}

# Transformer Hyperparameter Search Space
# Input: Sequential (30 timesteps × 24 features)
# Output: Single RUL value (scalar regression)
# Note: Self-attention mechanism (alternative to recurrence and convolution)
TRANSFORMER_SEARCH_SPACE = {
    # Architecture
    'num_transformer_blocks': ('categorical', [1, 2]),  # 1-2 transformer encoder blocks
    'd_model': ('categorical', [64, 128]),              # Embedding/model dimension (internal representation)
    'num_heads': ('categorical', [2, 4, 8]),            # Number of attention heads (must divide d_model)
    'ff_dim': ('categorical', [128, 256]),              # Feed-forward network dimension (2-4× d_model typical)
    
    # Regularization
    'dropout_rate': ('categorical', [0.1, 0.2, 0.3, 0.4, 0.5]), # Dropout (applied in attention and FF layers)
    
    # Optimization
    'learning_rate': ('categorical', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]),         # Adam learning rate
    'batch_size': ('categorical', [32, 64, 128, 256]),
    
    # Fixed parameters (not tuned):
    # - activation: 'relu' (feed-forward network activation)
    # - use_positional_encoding: True (adds position information to inputs)
    # - Constraint: d_model must be divisible by num_heads (64%2=0, 64%4=0, 64%8=0 ; 128%2=0, 128%4=0, 128%8=0 )
}

# Transformer Fixed Parameters
TRANSFORMER_FIXED_PARAMS = {
    'activation': 'relu',                   # Feed-forward network activation
    'use_positional_encoding': True,        # Add position info (critical for transformers)
    'optimizer': 'adam',                    # Adaptive learning rate optimizer
    'loss': 'mse',                          # Mean squared error for regression
    'output_activation': 'linear',          # Regression output (no activation)
    'epochs': 50,                           # Max epochs for tuning (reduced from 100 for faster hyperparameter search)
    'early_stopping_patience': 10,          # Stop if val_loss doesn't improve for 10 epochs
    'monitor': 'val_loss',                  # Metric to monitor for early stopping
    'restore_best_weights': True,           # Load best model weights after training
    'random_seed': RANDOM_SEED,             # Reproducibility
}


# =============================================================================
# PHASE 3: FEATURE EXTRACTION MODELS
# =============================================================================
# CNN-LSTM, TCN-LSTM, AE-LSTM architectures
# Strategy: Fixed LSTM backend (Phase 1 best HPs), tune feature extraction component only

# CNN-LSTM Hyperparameter Search Space
# Architecture: CNN (feature extraction) → LSTM (temporal modeling) → Dense (RUL)
# Input: Sequential (30 timesteps × 14 features from Correlation_FS)
# Key difference from Phase 1: Only CNN part is tuned, LSTM is fixed
CNN_LSTM_SEARCH_SPACE = {
    # CNN Architecture (Feature Extraction Component - TUNED)
    'conv_layers': ('categorical', [1, 2]),             # Number of Conv1D layers
    'filters': ('categorical', [32, 64, 128]),          # Number of convolutional filters
    'kernel_size': ('categorical', [3, 5]),             # Temporal window size for convolution
    'pooling': ('categorical', [True, False]),          # Whether to use MaxPooling1D
    
    # LSTM Component: Fixed from Phase 1 best hyperparameters
    # - units: From Phase 1 LSTM best config
    # - num_layers: From Phase 1 LSTM best config
    # - dropout: From Phase 1 LSTM best config
    # - learning_rate: From Phase 1 LSTM best config
    
    # Note: This gives 2 × 3 × 2 × 2 = 24 possible configurations
    # Optuna will intelligently sample from this space (typically 30 trials sufficient)
}

# CNN-LSTM Fixed Parameters
CNN_LSTM_FIXED_PARAMS = {
    'cnn_activation': 'relu',               # CNN layers activation
    'cnn_padding': 'same',                  # Padding for Conv1D (maintains sequence length)
    'pool_size': 2,                         # MaxPooling size (if pooling=True)
    'batch_size': 256,                      # Fixed batch size (from Phase 1 best)
    'max_epochs': 100,                      # Maximum training epochs
    'early_stopping_patience': 15,          # Early stopping patience
    'monitor': 'val_loss',                  # Metric to monitor
    'restore_best_weights': True,           # Restore best model after training
}

# Optuna settings for Phase 3 Feature Extraction
# Reduced from Phase 1 since search space is smaller (only feature extractor tuned)
OPTUNA_N_TRIALS_PHASE3 = 30                 # Sufficient for search spaces

# TCN-LSTM Hyperparameter Search Space
# Architecture: TCN (feature extraction via dilated convolutions) → LSTM → Dense → RUL
# Input: Sequential (30 timesteps × 14 features from Correlation_FS)
# Key difference: TCN uses dilated convolutions for long-range dependencies
TCN_LSTM_SEARCH_SPACE = {
    # TCN Architecture (Feature Extraction Component - TUNED)
    'num_blocks': ('categorical', [1, 2, 3]),           # Number of TCN stack repeats
    'filters': ('categorical', [32, 64, 128]),          # Number of filters per conv layer
    'kernel_size': ('categorical', [2, 3, 4]),             # Convolution kernel size
    'dropout': ('categorical', [0.1, 0.2, 0.3]),       # Spatial dropout rate
    
    # LSTM Component: Fixed from Phase 1 best hyperparameters
    # Note: This gives 3 × 3 × 2 × 3 = 54 possible configurations
}

# TCN-LSTM Fixed Parameters
TCN_LSTM_FIXED_PARAMS = {
    'dilation_rates': [1, 2, 4, 8],         # Exponential dilation per block (standard TCN)
    'activation': 'relu',                   # Activation function
    'use_skip_connections': True,           # Residual connections (TCN standard)
    'padding': 'causal',                    # Causal padding (no future leakage)
    'batch_size': 32,                       # Fixed batch size (from Phase 1 best)
    'max_epochs': 100,                      # Maximum training epochs
    'early_stopping_patience': 15,          # Early stopping patience
    'monitor': 'val_loss',                  # Metric to monitor
    'restore_best_weights': True,           # Restore best model after training
}

# AE-LSTM Hyperparameter Search Space
# Architecture: Autoencoder (compresses features) → LSTM (temporal modeling) → Dense → RUL
# Input: Sequential (30 timesteps × 14 features from Correlation_FS)
# Key difference: AE learns compressed representation, LSTM models temporal patterns on compressed features
AE_LSTM_SEARCH_SPACE = {
    # Autoencoder Architecture (Feature Compression Component - TUNED)
    'encoder_layers': ('categorical', [1, 2]),          # Number of encoder dense layers
    'encoder_units': ('categorical', [32, 64, 128]),    # Units in encoder hidden layers
    'bottleneck_dim': ('categorical', [4, 8, 12]),      # Compressed feature dimension (14→bottleneck)
    'activation': ('categorical', ['relu', 'tanh']),    # Encoder activation function
    
    # LSTM Component: Fixed from Phase 1 best hyperparameters
    # Note: This gives 2 × 3 × 3 × 2 = 36 possible configurations
}

# AE-LSTM Fixed Parameters
AE_LSTM_FIXED_PARAMS = {
    'dropout': 0.2,                         # Dropout between encoder layers
    'batch_size': 32,                       # Fixed batch size (from Phase 1 best)
    'max_epochs': 100,                      # Maximum training epochs
    'early_stopping_patience': 15,          # Early stopping patience
    'monitor': 'val_loss',                  # Metric to monitor
    'restore_best_weights': True,           # Restore best model after training
}


# =============================================================================
# EVALUATION METRICS
# =============================================================================

# RUL bins for RMSE-by-bins analysis
RUL_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125]

