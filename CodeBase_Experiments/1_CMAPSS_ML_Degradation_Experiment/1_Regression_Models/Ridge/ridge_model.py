from sklearn.linear_model import Ridge

def get_model(alpha=1.0):
    return Ridge(alpha=alpha, random_state=42)

def get_default_params():
    return {'alpha': 1.0}

# Model metadata
MODEL_INFO = {
    'name': 'Ridge Regression',
    'short_name': 'Ridge',
    'requires_scaling': True,  # Already done in preprocessing
    'supports_windowed': False  # Uses non-windowed data
}

