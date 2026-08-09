from sklearn.linear_model import Lasso

def get_model(alpha=1.0):
    return Lasso(alpha=alpha, random_state=42, max_iter=10000)

def get_default_params():
    return {'alpha': 1.0}

# Model metadata
MODEL_INFO = {
    'name': 'Lasso Regression',
    'short_name': 'Lasso',
    'requires_scaling': True,  # Already done in preprocessing
    'supports_windowed': False  # Uses non-windowed data
}

