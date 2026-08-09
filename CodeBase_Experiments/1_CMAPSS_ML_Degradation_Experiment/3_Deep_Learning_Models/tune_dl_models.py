import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Import utilities
from Utilities import config
from sklearn.metrics import mean_squared_error
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import ANN model
from ANN.ann_model import build_model as build_ann, MODEL_INFO as ANN_INFO

# ==========
# DATA LOADING
# ==========

def load_data():
    print("\n" + "="*40)
    print("LOADING DATA")
    print("="*40)
    
    # -------------------------------------------------------------------------
    # 1. Load non-windowed data (for ANN)
    # -------------------------------------------------------------------------
    data_path_nonwindowed = Path(config.NON_WINDOWED_DATA_PATH)
    
    train_features = pd.read_csv(data_path_nonwindowed / 'FD001_train_features.csv')
    train_ids = pd.read_csv(data_path_nonwindowed / 'FD001_train_ids.csv')
    
    val_features = pd.read_csv(data_path_nonwindowed / 'FD001_val_features.csv')
    val_ids = pd.read_csv(data_path_nonwindowed / 'FD001_val_ids.csv')
    
    X_train_full = train_features.values.astype(np.float32)
    y_train_full = train_ids['RUL'].values.astype(np.float32)
    
    X_val = val_features.values.astype(np.float32)
    y_val = val_ids['RUL'].values.astype(np.float32)
    
    feature_names = list(train_features.columns)
    
    # -------------------------------------------------------------------------
    # 2. Load windowed data (for RNN/LSTM/GRU/BiLSTM/TCN/CNN/Transformer)
    # -------------------------------------------------------------------------
    data_path_windowed = Path(config.WINDOWED_DATA_PATH)
    
    # Load training sequences and labels
    X_train_full_windowed = np.load(data_path_windowed / 'FD001_X_train_windowed.npy').astype(np.float32)
    y_train_full_windowed = np.load(data_path_windowed / 'FD001_y_train_windowed.npy').astype(np.float32)
    
    # Load validation sequences and labels
    X_val_windowed = np.load(data_path_windowed / 'FD001_X_val_windowed.npy').astype(np.float32)
    y_val_windowed = np.load(data_path_windowed / 'FD001_y_val_windowed.npy').astype(np.float32)
    
    print(f"\n Non-windowed data loaded:")
    print(f"Training samples: {len(X_train_full):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Features: {len(feature_names)}")
    print(f"Input shape: {X_train_full.shape}")
    
    print(f"\n Windowed data loaded:")
    print(f"Training sequences: {len(X_train_full_windowed):,}")
    print(f"Validation sequences: {len(X_val_windowed):,}")
    print(f"Sequence shape: {X_train_full_windowed.shape}")
    print(f"(Note: Windowed data has fewer samples due to 30-timestep window requirement)")
    
    return {
        # Non-windowed (for ANN)
        'X_train_full': X_train_full,
        'y_train_full': y_train_full,
        'X_val': X_val,
        'y_val': y_val,
        'feature_names': feature_names,
        
        # Windowed (for RNN/LSTM/GRU/BiLSTM/TCN/CNN/Transformer)
        'X_train_full_windowed': X_train_full_windowed,
        'y_train_full_windowed': y_train_full_windowed,
        'X_val_windowed': X_val_windowed,
        'y_val_windowed': y_val_windowed
    }

# ==========
# EARLY STOPPING CALLBACK
# ==========

class EarlyStoppingCallback:
    def __init__(self, patience=15):
        self.patience = patience
        self.no_improvement_count = 0
        self.best_value = float('inf')
    
    def __call__(self, study, trial):
        # Only count completed trials
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        
        # Check if this trial improved
        if trial.value < self.best_value:
            self.best_value = trial.value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Stop study if no improvement for patience trials
        if self.no_improvement_count >= self.patience:
            print(f"\n Early stopping triggered! No improvement for {self.patience} consecutive trials.")
            print(f"Best RMSE: {self.best_value:.4f}")
            study.stop()

# ==========
# ANN TUNING
# ==========

def tune_ann(data, n_trials=None, n_jobs=None, save_dir=None):
    print("\n" + "="*40)
    print("TUNING: ANN (Artificial Neural Network)")
    print("="*40)
    
    # Get config settings
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    n_jobs = n_jobs or config.OPTUNA_N_JOBS
    
    # Set default save directory to match tree-based models structure
    if save_dir is None:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / 'ANN'
    else:
        save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters from search space
        hyperparams = suggest_hyperparameters_ann(trial)
        
        # Build model
        model = build_ann(
            n_hidden_layers=hyperparams['n_hidden_layers'],
            units_layer1=hyperparams['units_layer1'],
            units_layer2=hyperparams['units_layer2'],
            dropout_rate=hyperparams['dropout_rate'],
            learning_rate=hyperparams['learning_rate'],
            input_shape=data['X_train_full'].shape[1]
        )
        
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Optuna pruning callback (stops bad trials early)
        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial, 'val_loss'
        )
        
        # Train model
        history = model.fit(
            data['X_train_full'], data['y_train_full'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=100,
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stop, pruning_callback],
            verbose=0
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(data['X_val'], verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(data['y_val'], y_val_pred))
        
        return val_rmse
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config.RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'ANN_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Early stopping callback (stop if no improvement after patience trials)
    early_stopping = EarlyStoppingCallback(patience=config.OPTUNA_EARLY_STOPPING_PATIENCE)
    
    # Run optimization
    print(f"\nStarting Optuna optimization:")
    print(f"Trials: {n_trials}")
    print(f"Jobs: {n_jobs}")
    print(f"Objective: Minimize validation RMSE")
    print(f"Early stopping: Patience = {config.OPTUNA_EARLY_STOPPING_PATIENCE} trials")
    print()
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, 
                   callbacks=[early_stopping], show_progress_bar=True)
    
    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*40)
    print("TUNING COMPLETE!")
    print("="*40)
    print(f"\nBest Validation RMSE: {best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Save results
    results = {
        'model': 'ANN',
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': n_trials,
        'study': study
    }
    
    # Save best hyperparameters as JSON
    hyperparams_path = save_dir / 'ANN_best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\n Best hyperparameters saved: {hyperparams_path}")
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_path = save_dir / 'ANN_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved: {trials_path}")
    
    return results

def suggest_hyperparameters_ann(trial):
    search_space = config.ANN_SEARCH_SPACE
    
    hyperparams = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config[0]
        
        if param_type == 'categorical':
            choices = param_config[1]
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)
        
        elif param_type == 'int':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)
        
        elif param_type == 'int_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high, log=True)
        
        elif param_type == 'float':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)
        
        elif param_type == 'float_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high, log=True)
    
    return hyperparams

# ==========
# RNN TUNING
# ==========

def tune_rnn(data, n_trials=None, n_jobs=None, save_dir=None):
    print("\n" + "="*40)
    print("TUNING: RNN (Recurrent Neural Network)")
    print("="*40)
    
    # Get config settings
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    n_jobs = n_jobs or config.OPTUNA_N_JOBS
    
    # Set default save directory
    if save_dir is None:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / 'RNN'
    else:
        save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Import RNN model
    sys.path.insert(0, str(Path(__file__).parent / 'RNN'))
    import rnn_model
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters from search space
        hyperparams = suggest_hyperparameters_rnn(trial)
        
        # Build model
        model = rnn_model.build_model(
            n_recurrent_layers=hyperparams['n_recurrent_layers'],
            units_layer1=hyperparams['units_layer1'],
            units_layer2=hyperparams['units_layer2'],
            dropout_rate=hyperparams['dropout_rate'],
            recurrent_dropout=config.RNN_FIXED_PARAMS['recurrent_dropout'],
            learning_rate=hyperparams['learning_rate'],
            input_shape=(30, 24)  # Windowed data
        )
        
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Optuna pruning callback (stops bad trials early)
        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial, 'val_loss'
        )
        
        # Train model
        history = model.fit(
            data['X_train_full_windowed'], data['y_train_full_windowed'],
            validation_data=(data['X_val_windowed'], data['y_val_windowed']),
            epochs=100,
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stop, pruning_callback],
            verbose=0
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(data['X_val_windowed'], verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(data['y_val_windowed'], y_val_pred))
        
        return val_rmse
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config.RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'RNN_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Early stopping callback (stop if no improvement after patience trials)
    early_stopping = EarlyStoppingCallback(patience=config.OPTUNA_EARLY_STOPPING_PATIENCE)
    
    # Run optimization
    print(f"\nStarting Optuna optimization:")
    print(f"Trials: {n_trials}")
    print(f"Jobs: {n_jobs}")
    print(f"Objective: Minimize validation RMSE")
    print(f"Early stopping: Patience = {config.OPTUNA_EARLY_STOPPING_PATIENCE} trials")
    print()
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, 
                   callbacks=[early_stopping], show_progress_bar=True)
    
    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*40)
    print("TUNING COMPLETE!")
    print("="*40)
    print(f"\nBest Validation RMSE: {best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Save results
    results = {
        'model': 'RNN',
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': n_trials,
        'study': study
    }
    
    # Save best hyperparameters as JSON
    hyperparams_path = save_dir / 'RNN_best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\n Best hyperparameters saved: {hyperparams_path}")
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_path = save_dir / 'RNN_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved: {trials_path}")
    
    return results

def suggest_hyperparameters_rnn(trial):
    search_space = config.RNN_SEARCH_SPACE
    
    hyperparams = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config[0]
        
        if param_type == 'categorical':
            choices = param_config[1]
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)
        
        elif param_type == 'int':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)
        
        elif param_type == 'int_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high, log=True)
        
        elif param_type == 'float':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)
        
        elif param_type == 'float_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high, log=True)
    
    return hyperparams

# ==========
# LSTM TUNING
# ==========

def tune_lstm(data, n_trials=None, n_jobs=None, save_dir=None):
    print("\n" + "="*40)
    print("TUNING: LSTM (Long Short-Term Memory)")
    print("="*40)
    
    # Get config settings
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    n_jobs = n_jobs or config.OPTUNA_N_JOBS
    
    # Set default save directory
    if save_dir is None:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / 'LSTM'
    else:
        save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Import LSTM model
    sys.path.insert(0, str(Path(__file__).parent / 'LSTM'))
    import lstm_model
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters from search space
        hyperparams = suggest_hyperparameters_lstm(trial)
        
        # Build model
        model = lstm_model.build_model(
            n_recurrent_layers=hyperparams['n_lstm_layers'],  # LSTM search space uses 'n_lstm_layers'
            units_layer1=hyperparams['units_layer1'],
            units_layer2=hyperparams['units_layer2'],
            dropout_rate=hyperparams['dropout_rate'],
            recurrent_dropout=config.LSTM_FIXED_PARAMS['recurrent_dropout'],
            learning_rate=hyperparams['learning_rate'],
            input_shape=(30, 24)  # Windowed data
        )
        
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Optuna pruning callback (stops bad trials early)
        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial, 'val_loss'
        )
        
        # Train model
        history = model.fit(
            data['X_train_full_windowed'], data['y_train_full_windowed'],
            validation_data=(data['X_val_windowed'], data['y_val_windowed']),
            epochs=100,
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stop, pruning_callback],
            verbose=0
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(data['X_val_windowed'], verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(data['y_val_windowed'], y_val_pred))
        
        return val_rmse
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config.RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'LSTM_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Early stopping callback (stop if no improvement after patience trials)
    early_stopping = EarlyStoppingCallback(patience=config.OPTUNA_EARLY_STOPPING_PATIENCE)
    
    # Run optimization
    print(f"\nStarting Optuna optimization:")
    print(f"Trials: {n_trials}")
    print(f"Jobs: {n_jobs}")
    print(f"Objective: Minimize validation RMSE")
    print(f"Early stopping: Patience = {config.OPTUNA_EARLY_STOPPING_PATIENCE} trials")
    print()
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, 
                   callbacks=[early_stopping], show_progress_bar=True)
    
    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*40)
    print("TUNING COMPLETE!")
    print("="*40)
    print(f"\nBest Validation RMSE: {best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Save results
    results = {
        'model': 'LSTM',
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': n_trials,
        'study': study
    }
    
    # Save best hyperparameters as JSON
    hyperparams_path = save_dir / 'LSTM_best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\n Best hyperparameters saved: {hyperparams_path}")
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_path = save_dir / 'LSTM_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved: {trials_path}")
    
    return results

def suggest_hyperparameters_lstm(trial):
    search_space = config.LSTM_SEARCH_SPACE
    
    hyperparams = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config[0]
        
        if param_type == 'categorical':
            choices = param_config[1]
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)
        
        elif param_type == 'int':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)
        
        elif param_type == 'int_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high, log=True)
        
        elif param_type == 'float':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)
        
        elif param_type == 'float_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high, log=True)
    
    return hyperparams

# ==========
# GRU TUNING
# ==========

def tune_gru(data, n_trials=None, n_jobs=None, save_dir=None):
    print("\n" + "="*40)
    print("TUNING: GRU (Gated Recurrent Unit)")
    print("="*40)
    
    # Get config settings
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    n_jobs = n_jobs or config.OPTUNA_N_JOBS
    
    # Set default save directory
    if save_dir is None:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / 'GRU'
    else:
        save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Import GRU model
    sys.path.insert(0, str(Path(__file__).parent / 'GRU'))
    import gru_model
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters from search space
        hyperparams = suggest_hyperparameters_gru(trial)
        
        # Build model
        model = gru_model.build_model(
            n_recurrent_layers=hyperparams['n_gru_layers'],  # GRU search space uses 'n_gru_layers'
            units_layer1=hyperparams['units_layer1'],
            units_layer2=hyperparams['units_layer2'],
            dropout_rate=hyperparams['dropout_rate'],
            recurrent_dropout=config.GRU_FIXED_PARAMS['recurrent_dropout'],
            learning_rate=hyperparams['learning_rate'],
            input_shape=(30, 24)  # Windowed data
        )
        
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Optuna pruning callback (stops bad trials early)
        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial, 'val_loss'
        )
        
        # Train model
        history = model.fit(
            data['X_train_full_windowed'], data['y_train_full_windowed'],
            validation_data=(data['X_val_windowed'], data['y_val_windowed']),
            epochs=100,
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stop, pruning_callback],
            verbose=0
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(data['X_val_windowed'], verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(data['y_val_windowed'], y_val_pred))
        
        return val_rmse
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config.RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'GRU_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Early stopping callback (stop if no improvement after patience trials)
    early_stopping = EarlyStoppingCallback(patience=config.OPTUNA_EARLY_STOPPING_PATIENCE)
    
    # Run optimization
    print(f"\nStarting Optuna optimization:")
    print(f"Trials: {n_trials}")
    print(f"Jobs: {n_jobs}")
    print(f"Objective: Minimize validation RMSE")
    print(f"Early stopping: Patience = {config.OPTUNA_EARLY_STOPPING_PATIENCE} trials")
    print()
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, 
                   callbacks=[early_stopping], show_progress_bar=True)
    
    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*40)
    print("TUNING COMPLETE!")
    print("="*40)
    print(f"\nBest Validation RMSE: {best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Save results
    results = {
        'model': 'GRU',
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': n_trials,
        'study': study
    }
    
    # Save best hyperparameters as JSON
    hyperparams_path = save_dir / 'GRU_best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\n Best hyperparameters saved: {hyperparams_path}")
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_path = save_dir / 'GRU_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved: {trials_path}")
    
    return results

def suggest_hyperparameters_gru(trial):
    search_space = config.GRU_SEARCH_SPACE
    
    hyperparams = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config[0]
        
        if param_type == 'categorical':
            choices = param_config[1]
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)
        
        elif param_type == 'int':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)
        
        elif param_type == 'int_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high, log=True)
        
        elif param_type == 'float':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)
        
        elif param_type == 'float_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high, log=True)
    
    return hyperparams

# ==========
# BiLSTM TUNING
# ==========

def tune_bilstm(data, n_trials=None, n_jobs=None, save_dir=None):
    print("\n" + "="*40)
    print("TUNING: BiLSTM (Bidirectional LSTM)")
    print("="*40)
    
    # Get config settings
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    n_jobs = n_jobs or config.OPTUNA_N_JOBS
    
    # Set default save directory
    if save_dir is None:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / 'BiLSTM'
    else:
        save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Import BiLSTM model
    sys.path.insert(0, str(Path(__file__).parent / 'BiLSTM'))
    import bilstm_model
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters from search space
        hyperparams = suggest_hyperparameters_bilstm(trial)
        
        # Build model
        model = bilstm_model.build_model(
            n_recurrent_layers=hyperparams['n_bilstm_layers'],  # BiLSTM search space uses 'n_bilstm_layers'
            units_layer1=hyperparams['units_layer1'],
            units_layer2=hyperparams['units_layer2'],
            dropout_rate=hyperparams['dropout_rate'],
            recurrent_dropout=config.BILSTM_FIXED_PARAMS['recurrent_dropout'],
            learning_rate=hyperparams['learning_rate'],
            merge_mode=config.BILSTM_FIXED_PARAMS['merge_mode'],
            input_shape=(30, 24)  # Windowed data
        )
        
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Optuna pruning callback (stops bad trials early)
        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial, 'val_loss'
        )
        
        # Train model
        history = model.fit(
            data['X_train_full_windowed'], data['y_train_full_windowed'],
            validation_data=(data['X_val_windowed'], data['y_val_windowed']),
            epochs=100,
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stop, pruning_callback],
            verbose=0
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(data['X_val_windowed'], verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(data['y_val_windowed'], y_val_pred))
        
        return val_rmse
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config.RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'BiLSTM_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Early stopping callback (stop if no improvement after patience trials)
    early_stopping = EarlyStoppingCallback(patience=config.OPTUNA_EARLY_STOPPING_PATIENCE)
    
    # Run optimization
    print(f"\nStarting Optuna optimization:")
    print(f"Trials: {n_trials}")
    print(f"Jobs: {n_jobs}")
    print(f"Objective: Minimize validation RMSE")
    print(f"Early stopping: Patience = {config.OPTUNA_EARLY_STOPPING_PATIENCE} trials")
    print()
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, 
                   callbacks=[early_stopping], show_progress_bar=True)
    
    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*40)
    print("TUNING COMPLETE!")
    print("="*40)
    print(f"\nBest Validation RMSE: {best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Save results
    results = {
        'model': 'BiLSTM',
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': n_trials,
        'study': study
    }
    
    # Save best hyperparameters as JSON
    hyperparams_path = save_dir / 'BiLSTM_best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\n Best hyperparameters saved: {hyperparams_path}")
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_path = save_dir / 'BiLSTM_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved: {trials_path}")
    
    return results

def suggest_hyperparameters_bilstm(trial):
    search_space = config.BILSTM_SEARCH_SPACE
    
    hyperparams = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config[0]
        
        if param_type == 'categorical':
            choices = param_config[1]
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)
        
        elif param_type == 'int':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)
        
        elif param_type == 'int_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high, log=True)
        
        elif param_type == 'float':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)
        
        elif param_type == 'float_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high, log=True)
    
    return hyperparams

# ==========
# CNN TUNING
# ==========

def tune_cnn(data, n_trials=None, n_jobs=None, save_dir=None):
    print("\n" + "="*40)
    print("TUNING: CNN (Convolutional Neural Network)")
    print("="*40)
    
    # Get config settings
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    n_jobs = n_jobs or config.OPTUNA_N_JOBS
    
    # Set default save directory
    if save_dir is None:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / 'CNN'
    else:
        save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Import CNN model
    sys.path.insert(0, str(Path(__file__).parent / 'CNN'))
    import cnn_model
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters from search space
        hyperparams = suggest_hyperparameters_cnn(trial)
        
        # Build model
        model = cnn_model.build_model(
            n_conv_layers=hyperparams['n_conv_layers'],
            n_filters=hyperparams['n_filters'],
            kernel_size=hyperparams['kernel_size'],
            pool_size=config.CNN_FIXED_PARAMS['pool_size'],
            dropout_rate=hyperparams['dropout_rate'],
            learning_rate=hyperparams['learning_rate'],
            input_shape=(30, 24)  # Windowed data
        )
        
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Optuna pruning callback (stops bad trials early)
        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial, 'val_loss'
        )
        
        # Train model
        history = model.fit(
            data['X_train_full_windowed'], data['y_train_full_windowed'],
            validation_data=(data['X_val_windowed'], data['y_val_windowed']),
            epochs=100,
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stop, pruning_callback],
            verbose=0
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(data['X_val_windowed'], verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(data['y_val_windowed'], y_val_pred))
        
        return val_rmse
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config.RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'CNN_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Early stopping callback (stop if no improvement after patience trials)
    early_stopping = EarlyStoppingCallback(patience=config.OPTUNA_EARLY_STOPPING_PATIENCE)
    
    # Run optimization
    print(f"\nStarting Optuna optimization:")
    print(f"Trials: {n_trials}")
    print(f"Jobs: {n_jobs}")
    print(f"Objective: Minimize validation RMSE")
    print(f"Early stopping: Patience = {config.OPTUNA_EARLY_STOPPING_PATIENCE} trials")
    print()
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, 
                   callbacks=[early_stopping], show_progress_bar=True)
    
    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*40)
    print("TUNING COMPLETE!")
    print("="*40)
    print(f"\nBest Validation RMSE: {best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Save results
    results = {
        'model': 'CNN',
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': n_trials,
        'study': study
    }
    
    # Save best hyperparameters as JSON
    hyperparams_path = save_dir / 'CNN_best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\n Best hyperparameters saved: {hyperparams_path}")
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_path = save_dir / 'CNN_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved: {trials_path}")
    
    return results

def suggest_hyperparameters_cnn(trial):
    search_space = config.CNN_SEARCH_SPACE
    
    hyperparams = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config[0]
        
        if param_type == 'categorical':
            choices = param_config[1]
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)
        
        elif param_type == 'int':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)
        
        elif param_type == 'int_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high, log=True)
        
        elif param_type == 'float':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)
        
        elif param_type == 'float_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high, log=True)
    
    return hyperparams

# ==========
# TCN TUNING
# ==========

def tune_tcn(data, n_trials=None, n_jobs=None, save_dir=None):
    print("\n" + "="*40)
    print("TUNING: TCN (Temporal Convolutional Network)")
    print("="*40)
    
    # Get config settings
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    n_jobs = n_jobs or config.OPTUNA_N_JOBS
    
    # Set default save directory
    if save_dir is None:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / 'TCN'
    else:
        save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Import TCN model
    sys.path.insert(0, str(Path(__file__).parent / 'TCN'))
    import tcn_model
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters from search space
        hyperparams = suggest_hyperparameters_tcn(trial)
        
        # Build model
        model = tcn_model.build_model(
            n_blocks=hyperparams['n_blocks'],
            n_filters=hyperparams['n_filters'],
            kernel_size=config.TCN_FIXED_PARAMS['kernel_size'],
            dilation_rates=config.TCN_FIXED_PARAMS['dilation_rates'],
            dropout_rate=hyperparams['dropout_rate'],
            learning_rate=hyperparams['learning_rate'],
            input_shape=(30, 24)  # Windowed data
        )
        
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Optuna pruning callback (stops bad trials early)
        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial, 'val_loss'
        )
        
        # Train model
        history = model.fit(
            data['X_train_full_windowed'], data['y_train_full_windowed'],
            validation_data=(data['X_val_windowed'], data['y_val_windowed']),
            epochs=100,
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stop, pruning_callback],
            verbose=0
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(data['X_val_windowed'], verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(data['y_val_windowed'], y_val_pred))
        
        return val_rmse
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config.RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'TCN_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Early stopping callback (stop if no improvement after patience trials)
    early_stopping = EarlyStoppingCallback(patience=config.OPTUNA_EARLY_STOPPING_PATIENCE)
    
    # Run optimization
    print(f"\nStarting Optuna optimization:")
    print(f"Trials: {n_trials}")
    print(f"Jobs: {n_jobs}")
    print(f"Objective: Minimize validation RMSE")
    print(f"Early stopping: Patience = {config.OPTUNA_EARLY_STOPPING_PATIENCE} trials")
    print()
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, 
                   callbacks=[early_stopping], show_progress_bar=True)
    
    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*40)
    print("TUNING COMPLETE!")
    print("="*40)
    print(f"\nBest Validation RMSE: {best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Save results
    results = {
        'model': 'TCN',
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': n_trials,
        'study': study
    }
    
    # Save best hyperparameters as JSON
    hyperparams_path = save_dir / 'TCN_best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\n Best hyperparameters saved: {hyperparams_path}")
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_path = save_dir / 'TCN_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved: {trials_path}")
    
    return results

def suggest_hyperparameters_tcn(trial):
    search_space = config.TCN_SEARCH_SPACE
    
    hyperparams = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config[0]
        
        if param_type == 'categorical':
            choices = param_config[1]
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)
        
        elif param_type == 'int':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)
        
        elif param_type == 'int_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high, log=True)
        
        elif param_type == 'float':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)
        
        elif param_type == 'float_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high, log=True)
    
    return hyperparams

# ==========
# TRANSFORMER TUNING
# ==========

def tune_transformer(data, n_trials=None, n_jobs=None, save_dir=None):
    print("\n" + "="*40)
    print("TUNING: Transformer")
    print("="*40)
    
    # Get config settings
    n_trials = n_trials or config.OPTUNA_N_TRIALS
    n_jobs = n_jobs or config.OPTUNA_N_JOBS
    
    # Set default save directory
    if save_dir is None:
        save_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / 'Transformer'
    else:
        save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Import Transformer model
    sys.path.insert(0, str(Path(__file__).parent / 'Transformer'))
    import transformer_model
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters from search space
        hyperparams = suggest_hyperparameters_transformer(trial)
        
        # Build model
        model = transformer_model.build_model(
            d_model=hyperparams['d_model'],
            num_heads=hyperparams['num_heads'],
            ff_dim=hyperparams['ff_dim'],
            num_transformer_blocks=hyperparams['num_transformer_blocks'],
            dropout_rate=hyperparams['dropout_rate'],
            learning_rate=hyperparams['learning_rate'],
            use_positional_encoding=config.TRANSFORMER_FIXED_PARAMS['use_positional_encoding'],
            input_shape=(30, 24)  # Windowed data
        )
        
        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Optuna pruning callback (stops bad trials early)
        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial, 'val_loss'
        )
        
        # Train model
        history = model.fit(
            data['X_train_full_windowed'], data['y_train_full_windowed'],
            validation_data=(data['X_val_windowed'], data['y_val_windowed']),
            epochs=100,
            batch_size=hyperparams['batch_size'],
            callbacks=[early_stop, pruning_callback],
            verbose=0
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(data['X_val_windowed'], verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(data['y_val_windowed'], y_val_pred))
        
        return val_rmse
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=config.RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f'Transformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Early stopping callback (stop if no improvement after patience trials)
    early_stopping = EarlyStoppingCallback(patience=config.OPTUNA_EARLY_STOPPING_PATIENCE)
    
    # Run optimization
    print(f"\nStarting Optuna optimization:")
    print(f"Trials: {n_trials}")
    print(f"Jobs: {n_jobs}")
    print(f"Objective: Minimize validation RMSE")
    print(f"Early stopping: Patience = {config.OPTUNA_EARLY_STOPPING_PATIENCE} trials")
    print()
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, 
                   callbacks=[early_stopping], show_progress_bar=True)
    
    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*40)
    print("TUNING COMPLETE!")
    print("="*40)
    print(f"\nBest Validation RMSE: {best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Save results
    results = {
        'model': 'Transformer',
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': n_trials,
        'study': study
    }
    
    # Save best hyperparameters as JSON
    hyperparams_path = save_dir / 'Transformer_best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\n Best hyperparameters saved: {hyperparams_path}")
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_path = save_dir / 'Transformer_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    print(f"All trials saved: {trials_path}")
    
    return results

def suggest_hyperparameters_transformer(trial):
    search_space = config.TRANSFORMER_SEARCH_SPACE
    
    hyperparams = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config[0]
        
        if param_type == 'categorical':
            choices = param_config[1]
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)
        
        elif param_type == 'int':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)
        
        elif param_type == 'int_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_int(param_name, low, high, log=True)
        
        elif param_type == 'float':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)
        
        elif param_type == 'float_log':
            low, high = param_config[1], param_config[2]
            hyperparams[param_name] = trial.suggest_float(param_name, low, high, log=True)
    
    return hyperparams

# ==========
# MAIN EXECUTION
# ==========

def main():
    parser = argparse.ArgumentParser(description='Tune Deep Learning Models with Optuna')
    parser.add_argument('--models', nargs='+', default=['ANN'],
                       help='Models to tune: ANN RNN TCN LSTM GRU BiLSTM CNN Transformer, or "all"')
    parser.add_argument('--n_trials', type=int, default=None,
                       help='Number of Optuna trials (default: from config)')
    parser.add_argument('--n_jobs', type=int, default=None,
                       help='Number of parallel jobs (default: from config)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results (default: Results/Hyperparameter_Tuning/[MODEL])')
    
    args = parser.parse_args()
    
    # Handle "all" models
    if 'all' in args.models:
        args.models = ['ANN', 'RNN', 'LSTM', 'GRU', 'BiLSTM', 'CNN', 'TCN', 'Transformer']  # All 8 DL models implemented!
    
    print("="*40)
    print("DEEP LEARNING HYPERPARAMETER TUNING")
    print("="*40)
    print(f"\nModels to tune: {', '.join(args.models)}")
    print(f"Trials per model: {args.n_trials or config.OPTUNA_N_TRIALS}")
    print(f"Parallel jobs: {args.n_jobs or config.OPTUNA_N_JOBS}")
    
    # Show save directory
    if args.save_dir:
        print(f"Save directory: {args.save_dir}")
    else:
        print(f"Save directory: Results/Hyperparameter_Tuning/[MODEL]/")
    
    # Load data
    data = load_data()
    
    # Tune each model
    results = {}
    
    if 'ANN' in args.models:
        results['ANN'] = tune_ann(data, args.n_trials, args.n_jobs, args.save_dir)
    
    if 'RNN' in args.models:
        results['RNN'] = tune_rnn(data, args.n_trials, args.n_jobs, args.save_dir)
    
    if 'LSTM' in args.models:
        results['LSTM'] = tune_lstm(data, args.n_trials, args.n_jobs, args.save_dir)
    
    if 'GRU' in args.models:
        results['GRU'] = tune_gru(data, args.n_trials, args.n_jobs, args.save_dir)
    
    if 'BiLSTM' in args.models:
        results['BiLSTM'] = tune_bilstm(data, args.n_trials, args.n_jobs, args.save_dir)
    
    if 'CNN' in args.models:
        results['CNN'] = tune_cnn(data, args.n_trials, args.n_jobs, args.save_dir)
    
    if 'TCN' in args.models:
        results['TCN'] = tune_tcn(data, args.n_trials, args.n_jobs, args.save_dir)
    
    if 'Transformer' in args.models:
        results['Transformer'] = tune_transformer(data, args.n_trials, args.n_jobs, args.save_dir)
    
    print("\n" + "="*40)
    print("ALL TUNING COMPLETE!")
    print("="*40)
    print(f"\nResults saved to:")
    for model_name in results.keys():
        if args.save_dir:
            model_dir = Path(args.save_dir) / model_name
        else:
            model_dir = Path(config.RESULTS_BASE_PATH) / 'Hyperparameter_Tuning' / model_name
        print(f"{model_name}: {model_dir}")
    
    print("\nBest Validation RMSE per model:")
    for model_name, result in results.items():
        print(f"{model_name}: {result['best_value']:.4f}")

if __name__ == "__main__":
    # Import TensorFlow here (after argparse, so --help works fast)
    import tensorflow as tf
    
    # Configure TensorFlow
    # Suppress info messages
    tf.get_logger().setLevel('ERROR')
    
    # GPU Configuration
    print("\n" + "="*40)
    print("GPU CONFIGURATION")
    print("="*40)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) detected: {len(gpus)}")
            print(f"GPU device(s): {[gpu.name for gpu in gpus]}")
            print(f"TensorFlow will use GPU for training")
            print(f"Memory growth enabled (dynamic allocation)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            print(f"Falling back to CPU")
    else:
        print("No GPU detected!")
        print("Training will use CPU (will be slower)")
        print("Consider using Google Colab or a GPU machine for faster training")
    
    print("="*40 + "\n")
    
    # Run main
    main()

