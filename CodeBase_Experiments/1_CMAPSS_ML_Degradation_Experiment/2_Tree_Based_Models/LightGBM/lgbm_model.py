import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from Utilities import config
from Utilities import optuna_tuner

def get_model(use_tuned_params=True, **kwargs):
    # Start with fixed parameters
    params = config.LIGHTGBM_FIXED_PARAMS.copy()
    
    # Load tuned hyperparameters if requested
    if use_tuned_params:
        try:
            tuned_params = optuna_tuner.load_best_params('LGBM')
            params.update(tuned_params)
            print(f"Loaded tuned hyperparameters: {tuned_params}")
        except FileNotFoundError:
            print(f"No tuned params found. Using defaults.")
            # Use reasonable defaults if tuning hasn't been run
            params.update({
                'n_estimators': 300,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20
            })
    
    # Override with any provided kwargs
    params.update(kwargs)
    
    return lgb.LGBMRegressor(**params)

def train_model(X_train, y_train, use_tuned_params=True, **kwargs):
    model = get_model(use_tuned_params=use_tuned_params, **kwargs)
    model.fit(X_train, y_train)
    return model

def save_model(model, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(load_path):
    with open(load_path, 'rb') as f:
        model = pickle.load(f)
    return model

def get_feature_importances(model, feature_names=None, importance_type='gain'):
    if importance_type == 'split':
        importances = model.booster_.feature_importance(importance_type='split')
    else:  # gain
        importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    df = df.sort_values('importance', ascending=False)
    
    return df

# Model information
MODEL_INFO = {
    'name': 'LightGBM',
    'abbrev': 'LGBM',
    'family': 'tree_based',
    'type': 'gradient_boosting',
    'requires_scaling': False,
    'requires_windowing': False,
    'description': 'Gradient boosting with leaf-wise growth and histogram-based learning'
}

