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
    'name': 'TCN',
    'full_name': 'Temporal Convolutional Network',
    'type': 'deep_learning',
    'input_type': 'windowed',  # Uses sequential data
    'input_shape': (30, 24),  # (timesteps, features)
    'description': 'TCN with dilated causal convolutions and residual connections for long-term dependencies'
}

# ==========
# TCN RESIDUAL BLOCK
# ==========

def residual_block(x, dilation_rate, n_filters, kernel_size, dropout_rate, block_idx):
    # First dilated causal conv
    conv1 = layers.Conv1D(
        filters=n_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu',
        name=f'tcn_conv1_block{block_idx}_dilation{dilation_rate}'
    )(x)
    conv1 = layers.Dropout(dropout_rate, name=f'tcn_dropout1_block{block_idx}_dilation{dilation_rate}')(conv1)
    
    # Second dilated causal conv
    conv2 = layers.Conv1D(
        filters=n_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu',
        name=f'tcn_conv2_block{block_idx}_dilation{dilation_rate}'
    )(conv1)
    conv2 = layers.Dropout(dropout_rate, name=f'tcn_dropout2_block{block_idx}_dilation{dilation_rate}')(conv2)
    
    # Residual connection (adjust dimensions if needed)
    if x.shape[-1] != n_filters:
        # 1x1 conv to match dimensions
        residual = layers.Conv1D(
            filters=n_filters,
            kernel_size=1,
            padding='same',
            name=f'tcn_residual_block{block_idx}_dilation{dilation_rate}'
        )(x)
    else:
        residual = x
    
    # Add residual connection
    output = layers.Add(name=f'tcn_add_block{block_idx}_dilation{dilation_rate}')([conv2, residual])
    output = layers.Activation('relu', name=f'tcn_activation_block{block_idx}_dilation{dilation_rate}')(output)
    
    return output

# ==========
# MODEL BUILDING
# ==========

def build_model(n_blocks, n_filters, kernel_size, dilation_rates,
                dropout_rate, learning_rate,
                input_shape=(30, 24), random_seed=42):
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Initial convolution to match filter size
    x = layers.Conv1D(
        filters=n_filters,
        kernel_size=1,
        padding='same',
        name='initial_conv'
    )(inputs)
    
    # TCN blocks with increasing dilation
    for block_idx in range(n_blocks):
        for dilation_rate in dilation_rates:
            x = residual_block(
                x,
                dilation_rate=dilation_rate,
                n_filters=n_filters,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                block_idx=block_idx
            )
    
    # Global pooling to aggregate temporal info
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Dense layer before output
    x = layers.Dense(64, activation='relu', name='dense')(x)
    x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
    
    # Output layer (regression)
    outputs = layers.Dense(1, activation='linear', name='output')(x)
    
    # Build model
    model = models.Model(inputs=inputs, outputs=outputs, name='TCN')
    
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
    model_path = save_dir / f'TCN_{data_percentage}pct.keras'
    model.save(model_path)
    
    # Save hyperparameters
    params_path = save_dir / f'TCN_{data_percentage}pct_params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Save training history
    history_path = save_dir / f'TCN_{data_percentage}pct_history.json'
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
    Test TCN model building and training with dummy data
    """
    print("="*40)
    print("TESTING TCN MODEL")
    print("="*40)
    
    # Create dummy data (windowed sequences)
    np.random.seed(42)
    X_train_dummy = np.random.randn(1000, 30, 24)  # 1000 sequences
    y_train_dummy = np.random.randn(1000)
    X_val_dummy = np.random.randn(200, 30, 24)
    y_val_dummy = np.random.randn(200)
    
    # Test hyperparameters
    test_params = {
        'n_blocks': 2,
        'n_filters': 64,
        'kernel_size': 3,
        'dilation_rates': [1, 2, 4, 8],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64
    }
    
    print("\nTest hyperparameters:")
    for key, value in test_params.items():
        print(f"{key}: {value}")
    
    # Build model
    print("\n" + "-"*70)
    print("Building TCN model...")
    print("-"*70)
    model = build_model(
        n_blocks=test_params['n_blocks'],
        n_filters=test_params['n_filters'],
        kernel_size=test_params['kernel_size'],
        dilation_rates=test_params['dilation_rates'],
        dropout_rate=test_params['dropout_rate'],
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

