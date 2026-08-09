import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("\n" + "="*40)
print("QUICK TEST SUITE - TREE-BASED MODELS")
print("="*40)

def test_imports():
    print("\n[TEST 1] Testing imports...")
    try:
        from Utilities import config
        from Utilities import optuna_tuner
        from RandomForest.rf_model import train_model as train_rf, MODEL_INFO as RF_INFO
        from XGBoost.xgb_model import train_model as train_xgb, MODEL_INFO as XGB_INFO
        from LightGBM.lgbm_model import train_model as train_lgbm, MODEL_INFO as LGBM_INFO
        print("All imports successful")
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    print("\n[TEST 2] Testing data loading...")
    try:
        import pandas as pd
        from Utilities.config import NON_WINDOWED_DATA_PATH
        
        data_path = Path(NON_WINDOWED_DATA_PATH)
        
        # Load features and IDs separately (actual file structure)
        train_features = pd.read_csv(data_path / 'FD001_train_features.csv')
        train_ids = pd.read_csv(data_path / 'FD001_train_ids.csv')
        val_features = pd.read_csv(data_path / 'FD001_val_features.csv')
        val_ids = pd.read_csv(data_path / 'FD001_val_ids.csv')
        test_features = pd.read_csv(data_path / 'FD001_test_features.csv')
        test_ids = pd.read_csv(data_path / 'FD001_test_ids.csv')
        
        print(f"Data loaded successfully")
        print(f"Train: {len(train_ids):,} samples")
        print(f"Val: {len(val_ids):,} samples")
        print(f"Test: {len(test_ids):,} samples")
        return True
    except Exception as e:
        print(f"Data loading failed: {e}")
        return False

def test_single_model_training(model_name='RF'):
    print(f"\n[TEST 3] Testing {model_name} training (no tuning)...")
    try:
        import pandas as pd
        import numpy as np
        from Utilities.config import NON_WINDOWED_DATA_PATH, RANDOM_SEED
        
        # Load small subset of data
        data_path = Path(NON_WINDOWED_DATA_PATH)
        train_features = pd.read_csv(data_path / 'FD001_train_features.csv')
        train_ids = pd.read_csv(data_path / 'FD001_train_ids.csv')
        
        X_train = train_features.values
        y_train = train_ids['RUL'].values
        
        # Use small subset for quick test
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(X_train), size=1000, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Train model
        if model_name == 'RF':
            from RandomForest.rf_model import train_model
        elif model_name == 'XGB':
            from XGBoost.xgb_model import train_model
        elif model_name == 'LGBM':
            from LightGBM.lgbm_model import train_model
        
        print(f"Training on {len(X_train)} samples...")
        model = train_model(X_train, y_train, use_tuned_params=False, n_estimators=10)
        
        # Make prediction
        y_pred = model.predict(X_train[:10])
        print(f"Sample predictions: {y_pred[:3]}")
        
        print(f"{model_name} training successful")
        return True
        
    except Exception as e:
        print(f"{model_name} training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optuna_quick(model_name='RF', n_trials=3):
    print(f"\n[TEST 4] Testing Optuna tuning for {model_name} ({n_trials} trials)...")
    try:
        import pandas as pd
        from Utilities import config
        from Utilities import optuna_tuner
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        import lightgbm as lgb
        
        # Load data
        data_path = Path(config.NON_WINDOWED_DATA_PATH)
        train_features = pd.read_csv(data_path / 'FD001_train_features.csv')
        train_ids = pd.read_csv(data_path / 'FD001_train_ids.csv')
        val_features = pd.read_csv(data_path / 'FD001_val_features.csv')
        val_ids = pd.read_csv(data_path / 'FD001_val_ids.csv')
        
        X_train = train_features.values[:5000]  # Use subset
        y_train = train_ids['RUL'].values[:5000]
        X_val = val_features.values
        y_val = val_ids['RUL'].values
        
        # Select model and config
        if model_name == 'RF':
            model_class = RandomForestRegressor
            search_space = config.RANDOM_FOREST_SEARCH_SPACE
            fixed_params = config.RANDOM_FOREST_FIXED_PARAMS
        elif model_name == 'XGB':
            model_class = xgb.XGBRegressor
            search_space = config.XGBOOST_SEARCH_SPACE
            fixed_params = config.XGBOOST_FIXED_PARAMS
        elif model_name == 'LGBM':
            model_class = lgb.LGBMRegressor
            search_space = config.LIGHTGBM_SEARCH_SPACE
            fixed_params = config.LIGHTGBM_FIXED_PARAMS
        
        print(f"Running {n_trials} trials...")
        results = optuna_tuner.run_tuning(
            model_class=model_class,
            model_name=f'{model_name}_test',
            search_space=search_space,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
            n_jobs=1,  # Single job for testing
            fixed_params=fixed_params
        )
        
        print(f"Best val RMSE: {results['best_value']:.4f}")
        print(f"Best params: {results['best_params']}")
        print(f"Optuna tuning successful")
        return True
        
    except Exception as e:
        print(f"Optuna tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baseline_single_experiment(model_name='RF'):
    print(f"\n[TEST 5] Testing {model_name} baseline experiment (100% data)...")
    try:
        import pandas as pd
        import numpy as np
        from Utilities import config
        from Utilities.Plots_Metrics import rmse, r2
        
        # Load data
        data_path = Path(config.NON_WINDOWED_DATA_PATH)
        train_features = pd.read_csv(data_path / 'FD001_train_features.csv')
        train_ids = pd.read_csv(data_path / 'FD001_train_ids.csv')
        test_features = pd.read_csv(data_path / 'FD001_test_features.csv')
        test_ids = pd.read_csv(data_path / 'FD001_test_ids.csv')
        
        X_train = train_features.values[:5000]  # Use subset
        y_train = train_ids['RUL'].values[:5000]
        X_test = test_features.values
        y_test = test_ids['RUL'].values
        
        # Train model
        if model_name == 'RF':
            from RandomForest.rf_model import train_model
        elif model_name == 'XGB':
            from XGBoost.xgb_model import train_model
        elif model_name == 'LGBM':
            from LightGBM.lgbm_model import train_model
        
        print(f"Training...")
        model = train_model(X_train, y_train, use_tuned_params=False, n_estimators=50)
        
        # Evaluate
        y_pred_test = model.predict(X_test)
        test_rmse = rmse(y_test, y_pred_test)
        test_r2 = r2(y_test, y_pred_test)
        
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Baseline experiment successful")
        return True
        
    except Exception as e:
        print(f"Baseline experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    
    print("\nThis script runs quick tests to verify all components work.")
    print("Each test uses small subsets of data for speed.\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Data loading
    results.append(("Data Loading", test_data_loading()))
    
    # Test 3: Model training (RF only for quick test)
    results.append(("RF Training", test_single_model_training('RF')))
    
    # Test 4: Optuna tuning (RF only, 3 trials)
    results.append(("Optuna (3 trials)", test_optuna_quick('RF', n_trials=3)))
    
    # Test 5: Baseline experiment
    results.append(("Baseline Experiment", test_baseline_single_experiment('RF')))
    
    # Summary
    print("\n" + "="*40)
    print("TEST SUMMARY")
    print("="*40)
    
    for test_name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n Sucessful run flag Ready to run full experiments.")
        print("\nNext steps:")
        print("1. Run full tuning: python tune_tree_models.py --models all --n_trials 50")
        print("2. Run baseline: python run_all_tree.py")
    else:
        print("\n SOME TESTS FAILED! Fix issues before running full experiments.")
    
    print()

if __name__ == "__main__":
    main()

