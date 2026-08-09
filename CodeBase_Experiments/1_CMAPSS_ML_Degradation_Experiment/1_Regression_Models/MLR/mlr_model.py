from sklearn.linear_model import LinearRegression

def get_model(params=None):
    if params is None:
        params = get_default_params()
    
    return LinearRegression(**params)

def get_default_params():
    return {
        'fit_intercept': True,
        'copy_X': True
    }

# Model metadata for automation scripts
MODEL_INFO = {
    'name': 'Multiple Linear Regression',
    'short_name': 'MLR',
    'family': 'regression',
    'requires_scaling': True,  # Data should be scaled (already done in preprocessing)
    'supports_windowed': False,  # Uses non-windowed data
    'needs_tuning': False,  # No hyperparameters to tune
    'description': 'Baseline linear regression model with no regularization'
}

# For testing this model independently
if __name__ == "__main__":
    import numpy as np
    
    print("Testing MLR model...")
    print(f"Model: {MODEL_INFO['name']}")
    print(f"Default params: {get_default_params()}")
    
    # Quick test
    model = get_model()
    X_test = np.random.randn(100, 5)
    y_test = np.random.randn(100)
    model.fit(X_test, y_test)
    
    print("Model initialized and tested successfully!")

