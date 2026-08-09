from sklearn.linear_model import ElasticNet

def get_model(alpha=1.0, l1_ratio=0.5):
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)

def get_default_params():
    return {'alpha': 1.0, 'l1_ratio': 0.5}

# Model metadata
MODEL_INFO = {
    'name': 'ElasticNet Regression',
    'short_name': 'ElasticNet',
    'requires_scaling': True,  # Already done in preprocessing
    'supports_windowed': False  # Uses non-windowed data
}

