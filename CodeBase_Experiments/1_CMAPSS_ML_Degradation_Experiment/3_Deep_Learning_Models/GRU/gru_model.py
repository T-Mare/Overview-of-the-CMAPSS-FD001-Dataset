import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from pathlib import Path
import json

# ==========
# MODEL METADATA
# ==========

MODEL_INFO = {
    'name': 'GRU',
    'full_name': 'Gated Recurrent Unit',
    'type': 'deep_learning',
    'input_type': 'windowed',  # Uses sequential data
    'input_shape': (30, 24),  # (timesteps, features)
    'description': 'GRU network with 1-2 GRU layers (simpler than LSTM, often faster)'
}

# ==========
# MODEL BUILDING
# ==========

def build_model(n_recurrent_layers, units_layer1, units_layer2,
                dropout_rate, recurrent_dropout, learning_rate,
                input_shape=(30, 24), random_seed=42):
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Build model
    model = models.Sequential(name='GRU')
    
    # Input layer
    model.add(layers.Input(shape=input_shape, name='input'))
    
    # First GRU layer
    if n_recurrent_layers == 1:
        # Single layer: return sequences = False (output only final timestep)
        model.add(layers.GRU(
            units_layer1,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
            name='gru_1'
        ))
    else:
        # First of multiple layers: return sequences = True
        model.add(layers.GRU(
            units_layer1,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            name='gru_1'
        ))
        
        # Second GRU layer (final layer: return sequences = False)
        model.add(layers.GRU(
            units_layer2,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
            name='gru_2'
        ))
    
    # Output layer (regression)
    model.add(layers.Dense(1, activation='linear', name='output'))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'RootMeanSquaredError']
    )
    
    return model

# ==========
# TRAINING
# ==========

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=64, verbose=1):
    # Early stopping callback
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=verbose
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )
    
    return history

# ==========
# SAVING
# ==========

def save_model(model, history, params, save_dir, data_percentage):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model in .keras format (Keras 3)
    model_path = save_dir / f'GRU_{data_percentage}pct.keras'
    model.save(model_path)
    
    # Save hyperparameters
    params_path = save_dir / f'GRU_{data_percentage}pct_params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Save training history
    history_path = save_dir / f'GRU_{data_percentage}pct_history.json'
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'root_mean_squared_error': [float(x) for x in history.history['root_mean_squared_error']],
        'val_root_mean_squared_error': [float(x) for x in history.history['val_root_mean_squared_error']],
        'epochs_trained': len(history.history['loss'])
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    return {
        'model_path': str(model_path),
        'params_path': str(params_path),
        'history_path': str(history_path)
    }

# ==========
# TESTING
# ==========

if __name__ == "__main__":
    """
    Test GRU model building and training with dummy data
    """
    print("="*40)
    print("TESTING GRU MODEL")
    print("="*40)
    
    # Create dummy data (windowed sequences)
    np.random.seed(42)
    X_train_dummy = np.random.randn(1000, 30, 24)  # 1000 sequences
    y_train_dummy = np.random.randn(1000)
    X_val_dummy = np.random.randn(200, 30, 24)
    y_val_dummy = np.random.randn(200)
    
    # Test hyperparameters
    test_params = {
        'n_recurrent_layers': 2,
        'units_layer1': 128,
        'units_layer2': 64,
        'dropout_rate': 0.2,
        'recurrent_dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 64
    }
    
    print("\nTest hyperparameters:")
    for key, value in test_params.items():
        print(f"{key}: {value}")
    
    # Build model
    print("\n" + "-"*70)
    print("Building GRU model...")
    print("-"*70)
    model = build_model(
        n_recurrent_layers=test_params['n_recurrent_layers'],
        units_layer1=test_params['units_layer1'],
        units_layer2=test_params['units_layer2'],
        dropout_rate=test_params['dropout_rate'],
        recurrent_dropout=test_params['recurrent_dropout'],
        learning_rate=test_params['learning_rate']
    )
    
    print("\n Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    model.summary()
    
    # Test prediction
    print("\n" + "-"*70)
    print("Testing prediction...")
    print("-"*70)
    predictions = model.predict(X_val_dummy[:3], verbose=0)
    print(f"Predictions work! Sample: {predictions.flatten()}")
    
    # Test training (1 epoch only)
    print("\n" + "-"*70)
    print("Testing training (1 epoch)...")
    print("-"*70)
    history = train_model(
        model, 
        X_train_dummy, y_train_dummy,
        X_val_dummy, y_val_dummy,
        epochs=1,
        batch_size=test_params['batch_size'],
        verbose=1
    )
    print(f"Training works! Final loss: {history.history['loss'][-1]:.4f}")
    
    print("\n" + "="*40)
    print("Sucessful run flag")
    print("="*40)

