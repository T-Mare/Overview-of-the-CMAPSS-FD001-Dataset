from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def get_model(params=None):
    if params is None:
        params = {}
    
    return Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('linear', LinearRegression())
    ])

def get_default_params():
    return {}

# Model metadata
MODEL_INFO = {
    'name': 'Polynomial Regression (Degree 3)',
    'short_name': 'Poly3',
    'requires_scaling': True,  # Already done in preprocessing
    'supports_windowed': False  # Uses non-windowed data
}

