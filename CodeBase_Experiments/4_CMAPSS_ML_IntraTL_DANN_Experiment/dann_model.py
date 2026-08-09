import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_feature_extractor(input_shape=(30, 14), gru_units=256, dropout=0.3, name='feature_extractor'):
    inputs = layers.Input(shape=input_shape, name='input_sequences')
    
    # GRU layer (returns sequences=False for single output vector)
    x = layers.GRU(
        units=gru_units,
        dropout=dropout,
        recurrent_dropout=0.0,  # GRU typically doesn't use recurrent dropout
        return_sequences=False,
        name='gru_layer'
    )(inputs)
    
    # Additional dropout after GRU
    features = layers.Dropout(dropout, name='feature_dropout')(x)
    
    model = keras.Model(inputs=inputs, outputs=features, name=name)
    return model

def build_rul_predictor(feature_dim=256, name='rul_predictor'):
    inputs = layers.Input(shape=(feature_dim,), name='features')
    
    # Dense layers for RUL prediction
    x = layers.Dense(128, activation='relu', name='rul_dense1')(inputs)
    x = layers.Dropout(0.3, name='rul_dropout1')(x)
    x = layers.Dense(64, activation='relu', name='rul_dense2')(x)
    x = layers.Dropout(0.3, name='rul_dropout2')(x)
    
    # Output layer (linear activation for regression)
    rul_output = layers.Dense(1, activation='linear', name='rul_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=rul_output, name=name)
    return model

def build_domain_discriminator(feature_dim=256, name='domain_discriminator'):
    inputs = layers.Input(shape=(feature_dim,), name='features')
    
    # Dense layers for domain classification
    x = layers.Dense(128, activation='relu', name='domain_dense1')(inputs)
    x = layers.Dropout(0.5, name='domain_dropout')(x)  # Higher dropout for discriminator
    
    # Output layer (sigmoid for binary classification: source=0, target=1)
    domain_output = layers.Dense(1, activation='sigmoid', name='domain_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=domain_output, name=name)
    return model

def build_dann_model(input_shape=(30, 14), gru_units=256, dropout=0.3):
    # Build individual components
    feature_extractor = build_feature_extractor(input_shape, gru_units, dropout)
    rul_predictor = build_rul_predictor(feature_dim=gru_units)
    domain_discriminator = build_domain_discriminator(feature_dim=gru_units)
    
    # Create combined models for end-to-end training
    
    # Combined model for RUL prediction
    input_sequences = layers.Input(shape=input_shape, name='input_sequences')
    features = feature_extractor(input_sequences)
    rul_output = rul_predictor(features)
    combined_rul = keras.Model(
        inputs=input_sequences,
        outputs=rul_output,
        name='combined_rul_model'
    )
    
    # Combined model for domain classification
    features_domain = feature_extractor(input_sequences)
    domain_output = domain_discriminator(features_domain)
    combined_domain = keras.Model(
        inputs=input_sequences,
        outputs=domain_output,
        name='combined_domain_model'
    )
    
    return {
        'feature_extractor': feature_extractor,
        'rul_predictor': rul_predictor,
        'domain_discriminator': domain_discriminator,
        'combined_rul': combined_rul,
        'combined_domain': combined_domain
    }

def get_model_summary(models):
    print("\n" + "="*40)
    print("DANN MODEL ARCHITECTURE SUMMARY")
    print("="*40)
    
    print("\n1. FEATURE EXTRACTOR (GRU):")
    print("-" * 70)
    models['feature_extractor'].summary()
    
    print("\n2. RUL PREDICTOR:")
    print("-" * 70)
    models['rul_predictor'].summary()
    
    print("\n3. DOMAIN DISCRIMINATOR:")
    print("-" * 70)
    models['domain_discriminator'].summary()
    
    # Count total parameters
    total_params = (
        models['feature_extractor'].count_params() +
        models['rul_predictor'].count_params() +
        models['domain_discriminator'].count_params()
    )
    
    print("\n" + "="*40)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("="*40)

if __name__ == '__main__':
    """Test DANN model architecture."""
    
    print("Building DANN model components...")
    models = build_dann_model(input_shape=(30, 14), gru_units=256, dropout=0.3)
    
    # Print summaries
    get_model_summary(models)
    
    # Test forward pass
    import numpy as np
    print("\n" + "="*40)
    print("TESTING FORWARD PASS")
    print("="*40)
    
    batch_size = 4
    timesteps = 30
    features = 14
    
    # Create dummy input
    dummy_input = np.random.randn(batch_size, timesteps, features).astype(np.float32)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Test feature extractor
    extracted_features = models['feature_extractor'].predict(dummy_input, verbose=0)
    print(f"Extracted features shape: {extracted_features.shape}")
    
    # Test RUL predictor
    rul_predictions = models['combined_rul'].predict(dummy_input, verbose=0)
    print(f"RUL predictions shape: {rul_predictions.shape}")
    print(f"RUL predictions sample: {rul_predictions.flatten()}")
    
    # Test domain discriminator
    domain_predictions = models['combined_domain'].predict(dummy_input, verbose=0)
    print(f"Domain predictions shape: {domain_predictions.shape}")
    print(f"Domain predictions sample: {domain_predictions.flatten()}")
    
    print("\n DANN model architecture test complete!")

