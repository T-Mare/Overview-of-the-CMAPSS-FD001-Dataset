import tensorflow as tf
from tensorflow import keras
from gradient_reversal import GradientReversalLayer

###############################################################################
# BUILD LSTM-DANN 
###############################################################################
def build_costa_dann_fd002_to_fd001(
    input_shape=(30, 14),
    lstm_layers=1,
    lstm_units=(64,),
    lstm_dropout=0.1,
    feature_units=64,
    rul_layers=1,
    rul_units=(32,),
    rul_dropout=0.0,
    domain_layers=2,
    domain_units=(16, 16),
    domain_dropout=0.1,
    l2_reg=0.01,
    lambda_adversarial=1.0
):
    
    regularizer = keras.regularizers.l2(l2_reg)
    
    ###########################################################################
    # SHARED LSTM FEATURE EXTRACTOR (Costa et al. Figure 5)
    ###########################################################################
    # This extracts temporal features from time-series sensor data.
    # Features are shared between RUL predictor and domain classifier.
    # Reference: Costa et al. (2019), Section 3.4 "LSTM Deep Adversarial NN"
    ###########################################################################
    
    input_layer = keras.layers.Input(shape=input_shape, name='sensor_input')
    
    # LSTM layers (1 or 2 layers, variable units, dropout)
    # Table 3: FD002FD001 uses 1(64), FD003FD001 uses 2(64,32)
    x = input_layer
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)  # All but last layer return sequences
        x = keras.layers.LSTM(
            units=units,
            dropout=lstm_dropout,
            return_sequences=return_sequences,
            kernel_regularizer=regularizer,
            name=f'lstm_layer_{i+1}'
        )(x)
    lstm_out = x
    
    # Feature extractor layer (f in Costa et al. paper)
    # This creates domain-invariant features through adversarial training
    features = keras.layers.Dense(
        units=feature_units,
        activation='relu',
        kernel_regularizer=regularizer,
        name='feature_extractor'
    )(lstm_out)
    
    ###########################################################################
    # RUL PREDICTOR BRANCH (Costa et al. Figure 5)
    ###########################################################################
    # Feeds into "Compute regression loss in a source mini batch" (Figure 5)
    # This branch predicts Remaining Useful Life using SOURCE domain labels.
    # Loss: MAE (Mean Absolute Error) - Equation (8) with p=1 in Costa et al.
    # Reference: Costa et al. (2019), Section 3.4, Equations (7-8)
    ###########################################################################
    
    # Hidden layers (1 or 2 layers with optional dropout)
    # Table 3: FD002FD001 uses 1(32), FD003FD001 uses 2(32,32)
    rul_hidden = features
    for i, units in enumerate(rul_units):
        rul_hidden = keras.layers.Dense(
            units=units,
            activation='relu',
            kernel_regularizer=regularizer,
            name=f'rul_hidden_{i+1}'
        )(rul_hidden)
        if rul_dropout > 0.0:
            rul_hidden = keras.layers.Dropout(rul_dropout, name=f'rul_dropout_{i+1}')(rul_hidden)
    
    # RUL output (linear activation for regression)
    # Outputs normalized RUL value [0,1] (multiply by RUL_max to get actual cycles)
    rul_output = keras.layers.Dense(
        units=1,
        activation='linear',
        kernel_regularizer=regularizer,
        name='rul_output'
    )(rul_hidden)
    
    ###########################################################################
    # DOMAIN CLASSIFIER BRANCH WITH GRL 
    ###########################################################################
    #The Gradient Reversal Layer (GRL) is the key layer that enables adversarial Domain Adaptation.
    #When the network does a backward pass through GRL during training, the gradients are reversed and scaled by some factor (lambda).
    #This makes the feature extractor learn domain-invariant representations using two conflicting objectives:
    #* Minimize the RUL loss (same as normal back propagation) = minimizes the error between predicted labels and true labels.
    #* Maximize the domain classification loss (because of reversing gradients via GRL)= maximizes the error between predicted domains and true domains.
    #
    # Reference: Costa et al. (2019), Section 3.4, Equation (7)
    #           Ganin et al. (2016), "Domain-Adversarial Training of NNs"
    ###########################################################################
    
    # Gradient Reversal Layer 
    # Lambda (α) controls adversarial strength: FD002FD001=1.0, FD003FD001=2.0
    grl_features = GradientReversalLayer(
        lambda_=lambda_adversarial,
        name='gradient_reversal'
    )(features)
    
    # Hidden layers ( 2 layers with dropout)
    # Table 3 in Costa et al. study: FD002FD001 uses 2(16,16), FD003FD001 uses 2(32,32)
    domain_hidden = grl_features
    for i, units in enumerate(domain_units):
        domain_hidden = keras.layers.Dense(
            units=units,
            activation='relu',
            kernel_regularizer=regularizer,
            name=f'domain_hidden_{i+1}'
        )(domain_hidden)
        domain_hidden = keras.layers.Dropout(domain_dropout, name=f'domain_dropout_{i+1}')(domain_hidden)
    
    # Domain output (sigmoid for binary classification: source=0, target=1)
    # Loss: Binary crossentropy - Equation (9) in Costa et al.
    domain_output = keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_regularizer=regularizer,
        name='domain_output'
    )(domain_hidden)
    
    ###########################################################################
    # ASSEMBLE MULTI-OUTPUT MODEL
    ###########################################################################
    # Creates single model with two outputs:
    # 1. rul_output: For regression loss (source domain only)
    # 2. domain_output: For classification loss (source + target domains)
    #
    # Training loop in 1_train_costa_dann.py implements:
    # - "Compute regression loss in a source mini batch"  trains rul_output
    # - "Compute classification loss in source and target mini batches"  trains domain_output
    # - "Updates weights via backpropagation"  normal gradients for RUL
    # - "Updates weights via backpropagation and GRL"  reversed gradients for domain
    ###########################################################################
    
    model = keras.Model(
        inputs=input_layer,
        outputs=[rul_output, domain_output],
        name='Costa_DANN'
    )
    
    return model

###############################################################################
# COMPILE MODEL WITH LOSSES AND OPTIMIZER 
###############################################################################
# Implements the loss computation boxes from Figure 5:
# - "Compute regression loss in a source mini batch"  MAE loss
# - "Compute classification loss in source and target mini batches"  Binary crossentropy
#
# Optimizer: SGD with gradient clipping (Table 3, Section 5.1)
###############################################################################
def compile_costa_dann(
    model,
    lr_rul=0.01,
    lr_domain=0.01,
    clipnorm=1.0
):
    # SGD optimizer with gradient clipping
    # Costa et al.: "We clip the norm values of the gradients to 1 in the SGD algorithm"
    optimizer = keras.optimizers.SGD(
        learning_rate=lr_rul,
        clipnorm=clipnorm,  # explicitly stated: gradient norm clipping = 1.0
        momentum=0.0,  # this hp is not explicitly stated in study and is assumed (vanilla SGD)
        nesterov=False
    )
    
    # Compile with two losses
    model.compile(
        optimizer=optimizer,
        loss={
            'rul_output': 'mae',  # MAE for RUL (p=1 in paper)
            'domain_output': 'binary_crossentropy'
        },
        loss_weights={
            'rul_output': 1.0,
            'domain_output': 1.0  # Equal weight (lambda is in GRL)
        },
        metrics={
            'rul_output': ['mae'],
            'domain_output': ['accuracy']
        },
        weighted_metrics=[]  # Suppress sample_weight warning
    )
    
    return model

if __name__ == '__main__':
    """Test model building for both configurations."""
    print("="*40)
    print("COSTA ET AL. DANN MODEL - CONFIGURATION TESTS")
    print("="*40)
    
    # Test FD002  FD001 configuration
    print("\n" + "="*40)
    print("Configuration 1: FD002  FD001")
    print("="*40)
    model_fd002 = build_costa_dann_fd002_to_fd001(
        lstm_layers=1,
        lstm_units=(64,),
        lstm_dropout=0.1,
        feature_units=64,
        rul_layers=1,
        rul_units=(32,),
        rul_dropout=0.0,
        domain_layers=2,
        domain_units=(16, 16),
        domain_dropout=0.1,
        lambda_adversarial=1.0
    )
    model_fd002 = compile_costa_dann(model_fd002)
    model_fd002.summary()
    
    print("\nFD002FD001 Specifications:")
    print("LSTM: 1 layer (64 units), dropout 0.1")
    print("Feature: 64 units")
    print("RUL: 1 layer (32 units), dropout 0.0")
    print("Domain: 2 layers (16, 16), dropout 0.1")
    print("Lambda: 1.0, Batch: 512")
    
    # Test FD003  FD001 configuration
    print("\n" + "="*40)
    print("Configuration 2: FD003  FD001")
    print("="*40)
    model_fd003 = build_costa_dann_fd002_to_fd001(
        lstm_layers=2,
        lstm_units=(64, 32),
        lstm_dropout=0.3,
        feature_units=128,
        rul_layers=2,
        rul_units=(32, 32),
        rul_dropout=0.1,
        domain_layers=2,
        domain_units=(32, 32),
        domain_dropout=0.1,
        lambda_adversarial=2.0
    )
    model_fd003 = compile_costa_dann(model_fd003)
    model_fd003.summary()
    
    print("\nFD003FD001 Specifications:")
    print("LSTM: 2 layers (64, 32), dropout 0.3")
    print("Feature: 128 units")
    print("RUL: 2 layers (32, 32), dropout 0.1")
    print("Domain: 2 layers (32, 32), dropout 0.1")
    print("Lambda: 2.0, Batch: 256")
    print("="*40)

