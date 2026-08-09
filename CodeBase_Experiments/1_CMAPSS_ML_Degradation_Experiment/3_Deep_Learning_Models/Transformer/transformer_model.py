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
    'name': 'Transformer',
    'full_name': 'Transformer',
    'type': 'deep_learning',
    'input_type': 'windowed',  # Uses sequential data
    'input_shape': (30, 24),  # (timesteps, features)
    'description': 'Transformer with multi-head self-attention for temporal dependencies'
}

# ==========
# POSITIONAL ENCODING
# ==========

class PositionalEncoding(layers.Layer):
    def __init__(self, max_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        
        # Create positional encoding matrix
        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        seq_length = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_length, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "d_model": self.d_model,
        })
        return config

# ==========
# TRANSFORMER BLOCK
# ==========

def transformer_block(x, d_model, num_heads, ff_dim, dropout_rate, block_idx):
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        name=f'attention_block{block_idx}'
    )(x, x)
    attention_output = layers.Dropout(dropout_rate, name=f'attention_dropout_block{block_idx}')(attention_output)
    
    # Add & Norm (residual connection + layer normalization)
    x1 = layers.Add(name=f'attention_add_block{block_idx}')([x, attention_output])
    x1 = layers.LayerNormalization(epsilon=1e-6, name=f'attention_norm_block{block_idx}')(x1)
    
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='relu', name=f'ff_dense1_block{block_idx}')(x1)
    ff_output = layers.Dropout(dropout_rate, name=f'ff_dropout1_block{block_idx}')(ff_output)
    ff_output = layers.Dense(d_model, name=f'ff_dense2_block{block_idx}')(ff_output)
    ff_output = layers.Dropout(dropout_rate, name=f'ff_dropout2_block{block_idx}')(ff_output)
    
    # Add & Norm (residual connection + layer normalization)
    x2 = layers.Add(name=f'ff_add_block{block_idx}')([x1, ff_output])
    x2 = layers.LayerNormalization(epsilon=1e-6, name=f'ff_norm_block{block_idx}')(x2)
    
    return x2

# ==========
# MODEL BUILDING
# ==========

def build_model(d_model, num_heads, ff_dim, num_transformer_blocks,
                dropout_rate, learning_rate, use_positional_encoding=True,
                input_shape=(30, 24), random_seed=42):
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Project input features to d_model dimensions
    x = layers.Dense(d_model, name='input_projection')(inputs)
    
    # Add positional encoding
    if use_positional_encoding:
        x = PositionalEncoding(max_length=input_shape[0], d_model=d_model)(x)
    
    # Transformer blocks
    for i in range(num_transformer_blocks):
        x = transformer_block(
            x,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            block_idx=i
        )
    
    # Global average pooling to aggregate temporal info
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Dense layer before output
    x = layers.Dense(64, activation='relu', name='dense')(x)
    x = layers.Dropout(dropout_rate, name='dropout_dense')(x)
    
    # Output layer (regression)
    outputs = layers.Dense(1, activation='linear', name='output')(x)
    
    # Build model
    model = models.Model(inputs=inputs, outputs=outputs, name='Transformer')
    
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
    model_path = save_dir / f'Transformer_{data_percentage}pct.keras'
    model.save(model_path)
    
    # Save hyperparameters
    params_path = save_dir / f'Transformer_{data_percentage}pct_params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Save training history
    history_path = save_dir / f'Transformer_{data_percentage}pct_history.json'
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
    Test Transformer model building and training with dummy data
    """
    print("="*40)
    print("TESTING TRANSFORMER MODEL")
    print("="*40)
    
    # Create dummy data (windowed sequences)
    np.random.seed(42)
    X_train_dummy = np.random.randn(1000, 30, 24)  # 1000 sequences
    y_train_dummy = np.random.randn(1000)
    X_val_dummy = np.random.randn(200, 30, 24)
    y_val_dummy = np.random.randn(200)
    
    # Test hyperparameters
    test_params = {
        'd_model': 64,
        'num_heads': 4,
        'ff_dim': 128,
        'num_transformer_blocks': 2,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'use_positional_encoding': True,
        'batch_size': 64
    }
    
    print("\nTest hyperparameters:")
    for key, value in test_params.items():
        print(f"{key}: {value}")
    
    # Build model
    print("\n" + "-"*70)
    print("Building Transformer model...")
    print("-"*70)
    model = build_model(
        d_model=test_params['d_model'],
        num_heads=test_params['num_heads'],
        ff_dim=test_params['ff_dim'],
        num_transformer_blocks=test_params['num_transformer_blocks'],
        dropout_rate=test_params['dropout_rate'],
        learning_rate=test_params['learning_rate'],
        use_positional_encoding=test_params['use_positional_encoding']
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

