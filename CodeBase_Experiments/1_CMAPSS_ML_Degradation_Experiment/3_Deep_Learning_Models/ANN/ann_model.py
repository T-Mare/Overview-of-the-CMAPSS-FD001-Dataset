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
    'name': 'ANN',
    'full_name': 'Artificial Neural Network',
    'type': 'deep_learning',
    'input_type': 'non_windowed',  # Uses flattened features
    'input_shape': 720,  # 30 timesteps × 24 features = 720
    'description': 'Feedforward neural network with 1-2 hidden layers'
}

# ==========
# MODEL BUILDING
# ==========

def build_model(n_hidden_layers, units_layer1, units_layer2, 
                dropout_rate, learning_rate, 
                input_shape=720, random_seed=42):
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Build model
    model = models.Sequential(name='ANN')
    
    # Input layer
    model.add(layers.Input(shape=(input_shape,), name='input'))
    
    # First hidden layer
    model.add(layers.Dense(
        units_layer1, 
        activation='relu', 
        name='dense_1'
    ))
    model.add(layers.Dropout(dropout_rate, name='dropout_1'))
    
    # Second hidden layer (if specified)
    if n_hidden_layers == 2:
        model.add(layers.Dense(
            units_layer2, 
            activation='relu', 
            name='dense_2'
        ))
        model.add(layers.Dropout(dropout_rate, name='dropout_2'))
    
    # Output layer (regression)
    model.add(layers.Dense(1, activation='linear', name='output'))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean squared error
        metrics=['mae']  # Also track MAE during training
    )
    
    return model

# ==========
# MODEL TRAINING
# ==========

def train_model(X_train, y_train, X_val, y_val, hyperparams, verbose=1):
    # Build model
    model = build_model(
        n_hidden_layers=hyperparams['n_hidden_layers'],
        units_layer1=hyperparams['units_layer1'],
        units_layer2=hyperparams['units_layer2'],
        dropout_rate=hyperparams['dropout_rate'],
        learning_rate=hyperparams['learning_rate'],
        input_shape=X_train.shape[1]
    )
    
    # Print model summary
    if verbose > 0:
        print("\nModel Architecture:")
        model.summary()
    
    # Define callbacks
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
        epochs=100,
        batch_size=hyperparams['batch_size'],
        callbacks=[early_stop],
        verbose=verbose
    )
    
    return model, history

# ==========
# MODEL SAVING
# ==========

def save_model(model, save_dir, hyperparams=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_dir / 'model.keras'
    model.save(model_path)
    
    saved_files = {'model': str(model_path)}
    
    # Save hyperparameters
    if hyperparams is not None:
        hyperparams_path = save_dir / 'hyperparameters.json'
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        saved_files['hyperparameters'] = str(hyperparams_path)
    
    return saved_files

# ==========
# UTILITY FUNCTIONS
# ==========

def load_model(model_path):
    return keras.models.load_model(model_path)

def predict(model, X):
    predictions = model.predict(X, verbose=0)
    return predictions.flatten()  # Return 1D array

if __name__ == "__main__":
    # Test model building with dummy data
    print("="*40)
    print("TESTING ANN MODEL")
    print("="*40)
    
    # Create dummy data
    n_samples = 100
    n_features = 720
    X_dummy = np.random.randn(n_samples, n_features).astype(np.float32)
    y_dummy = np.random.uniform(0, 125, n_samples).astype(np.float32)
    
    # Test hyperparameters
    test_hyperparams = {
        'n_hidden_layers': 2,
        'units_layer1': 128,
        'units_layer2': 64,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32
    }
    
    print("\nTest Hyperparameters:")
    for key, value in test_hyperparams.items():
        print(f"{key}: {value}")
    
    # Build and test model
    model = build_model(
        n_hidden_layers=test_hyperparams['n_hidden_layers'],
        units_layer1=test_hyperparams['units_layer1'],
        units_layer2=test_hyperparams['units_layer2'],
        dropout_rate=test_hyperparams['dropout_rate'],
        learning_rate=test_hyperparams['learning_rate']
    )
    
    print("\n Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Test prediction
    predictions = predict(model, X_dummy[:5])
    print(f"\n Predictions work! Sample: {predictions[:3]}")
    
    print("\n" + "="*40)
    print("Sucessful run flag")
    print("="*40)

