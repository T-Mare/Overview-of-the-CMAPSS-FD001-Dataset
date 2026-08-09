import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from Utilities import config

# ==========
# CONFIGURATION
# ==========

# 14 Correlation_FS features (from Phase 2)
SELECTED_FEATURES = ['s11', 's4', 's12', 's7', 's15', 's21', 's20', 's17', 
                     's2', 's3', 's8', 's13', 's9', 's14']

# Mapping: feature name -> index in original 24-feature data
FEATURE_MAP = {
    's2': 0, 's3': 1, 's4': 2, 's7': 3, 's8': 4, 's9': 5,
    's11': 6, 's12': 7, 's13': 8, 's14': 9, 's15': 10,
    's17': 11, 's20': 12, 's21': 13
}

# Paths
WINDOWED_DATA_PATH = Path(config.WINDOWED_DATA_PATH)
PHASE2_BASELINE_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase2_Feature_Selection' / 'FD001' / 'Correlation_FS' / 'GRU' / 'GRU_metrics_summary.csv'

# Random seed
RANDOM_SEED = config.RANDOM_SEED

# ==========
# FEATURE EXTRACTION
# ==========

def extract_selected_features(X, selected_features=SELECTED_FEATURES):
    indices = [FEATURE_MAP[feat] for feat in selected_features]
    return X[:, :, indices]

# ==========
# DATA LOADING
# ==========

def load_dataset(dataset_name, n_engines=None):
    # Load windowed data
    X_train = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_X_train_windowed.npy')
    y_train = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_y_val_windowed.npy')
    X_test = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_X_test_windowed.npy')
    y_test = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_y_test_windowed.npy')
    
    # Extract features
    X_train = extract_selected_features(X_train)
    X_val = extract_selected_features(X_val)
    X_test = extract_selected_features(X_test)
    
    # Engine-based sampling for training set
    sampled_engines = None
    if n_engines is not None:
        # Load engine IDs
        engine_ids_path = WINDOWED_DATA_PATH / f'{dataset_name}_train_ids_windowed.csv'
        if not engine_ids_path.exists():
            raise FileNotFoundError(f"Engine IDs file not found: {engine_ids_path}")
        
        engine_ids_df = pd.read_csv(engine_ids_path)
        engine_ids = engine_ids_df['engine'].values
        
        # Get unique engine IDs
        unique_engines = np.unique(engine_ids)
        total_engines = len(unique_engines)
        
        if n_engines > total_engines:
            raise ValueError(f"Requested {n_engines} engines, but only {total_engines} available in {dataset_name}")
        
        # Sample engines
        np.random.seed(RANDOM_SEED)
        sampled_engines = np.random.choice(unique_engines, size=n_engines, replace=False)
        
        # Filter training data to include only sampled engines
        mask = np.isin(engine_ids, sampled_engines)
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        print(f"Sampled {n_engines} engines from {dataset_name} (out of {total_engines})")
        print(f"Train: {len(X_train)} samples ({n_engines} engines)")
    else:
        print(f"Using all engines from {dataset_name}")
        print(f"Train: {len(X_train)} samples")
    
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'train_engines': sampled_engines
    }

def load_domain_data(source_dataset, target_dataset, n_engines):
    print(f"\nLoading data:")
    print(f"Source: {source_dataset} ({n_engines} engines)")
    print(f"Target: {target_dataset} ({n_engines} engines)")
    
    # Load source dataset
    print(f"\nLoading {source_dataset} (source)...")
    source_data = load_dataset(source_dataset, n_engines)
    
    # Load target dataset
    print(f"\nLoading {target_dataset} (target)...")
    target_data = load_dataset(target_dataset, n_engines)
    
    # Create domain labels (0 = source, 1 = target)
    domain_source = np.zeros(len(source_data['X_train']), dtype=np.float32)
    domain_target = np.ones(len(target_data['X_train']), dtype=np.float32)
    
    return {
        'X_train_source': source_data['X_train'],
        'y_train_source': source_data['y_train'],
        'X_train_target': target_data['X_train'],
        'y_train_target': target_data['y_train'],
        'X_val': target_data['X_val'],
        'y_val': target_data['y_val'],
        'X_test': target_data['X_test'],
        'y_test': target_data['y_test'],
        'domain_train_source': domain_source,
        'domain_train_target': domain_target
    }

def create_mixed_dataset(domain_data, shuffle=True):
    # Concatenate source and target data
    X_train_mixed = np.concatenate([
        domain_data['X_train_source'],
        domain_data['X_train_target']
    ], axis=0)
    
    y_train_mixed = np.concatenate([
        domain_data['y_train_source'],
        domain_data['y_train_target']
    ], axis=0)
    
    domain_train_mixed = np.concatenate([
        domain_data['domain_train_source'],
        domain_data['domain_train_target']
    ], axis=0)
    
    # Shuffle if requested
    if shuffle:
        np.random.seed(RANDOM_SEED)
        indices = np.random.permutation(len(X_train_mixed))
        X_train_mixed = X_train_mixed[indices]
        y_train_mixed = y_train_mixed[indices]
        domain_train_mixed = domain_train_mixed[indices]
    
    print(f"\nCreated mixed dataset:")
    print(f"Total samples: {len(X_train_mixed)}")
    print(f"Source samples (label=0): {np.sum(domain_train_mixed == 0)}")
    print(f"Target samples (label=1): {np.sum(domain_train_mixed == 1)}")
    
    return {
        'X_train_mixed': X_train_mixed,
        'y_train_mixed': y_train_mixed,
        'domain_train_mixed': domain_train_mixed
    }

# ==========
# BASELINE LOADING
# ==========

def load_phase2_baseline():
    if not PHASE2_BASELINE_PATH.exists():
        print(f"Phase 2 baseline not found: {PHASE2_BASELINE_PATH}")
        print(f"Will skip baseline comparison")
        return None
    
    baseline_df = pd.read_csv(PHASE2_BASELINE_PATH)
    
    # Remove duplicates - keep best (lowest RMSE)
    baseline_df = baseline_df.sort_values('val_rmse').drop_duplicates('n_engines', keep='first')
    
    # Filter for GRU + Correlation_FS
    baseline_df = baseline_df[baseline_df['fs_method'] == 'Correlation_FS']
    
    print(f"Loaded Phase 2 baseline (GRU + Correlation_FS)")
    return baseline_df

# ==========
# DATA STATISTICS
# ==========

def print_data_summary(domain_data):
    print("\n" + "="*40)
    print("DATA SUMMARY")
    print("="*40)
    
    print("\nSource Domain:")
    print(f"Training samples: {len(domain_data['X_train_source'])}")
    print(f"RUL range: [{domain_data['y_train_source'].min():.1f}, {domain_data['y_train_source'].max():.1f}]")
    print(f"RUL mean: {domain_data['y_train_source'].mean():.1f}")
    
    print("\nTarget Domain:")
    print(f"Training samples: {len(domain_data['X_train_target'])}")
    print(f"RUL range: [{domain_data['y_train_target'].min():.1f}, {domain_data['y_train_target'].max():.1f}]")
    print(f"RUL mean: {domain_data['y_train_target'].mean():.1f}")
    print(f"Validation samples: {len(domain_data['X_val'])}")
    print(f"Test samples: {len(domain_data['X_test'])}")
    
    print("\nFeature Shape:")
    print(f"Input shape: {domain_data['X_train_source'].shape[1:]}")
    print("="*40)

if __name__ == '__main__':
    """Test data loading functions."""
    
    print("="*40)
    print("TESTING DANN DATA UTILITIES")
    print("="*40)
    
    # Test loading single dataset
    print("\n1. Testing single dataset loading (FD001, 10 engines)...")
    data = load_dataset('FD001', n_engines=10)
    print(f"Loaded FD001: {data['X_train'].shape[0]} training samples")
    
    # Test loading domain data
    print("\n2. Testing domain data loading (FD002->FD001, 5 engines each)...")
    domain_data = load_domain_data('FD002', 'FD001', n_engines=5)
    print_data_summary(domain_data)
    
    # Test creating mixed dataset
    print("\n3. Testing mixed dataset creation...")
    mixed_data = create_mixed_dataset(domain_data, shuffle=True)
    print(f"Created mixed dataset: {mixed_data['X_train_mixed'].shape[0]} samples")
    
    # Test baseline loading
    print("\n4. Testing baseline loading...")
    baseline = load_phase2_baseline()
    if baseline is not None:
        print(f"Loaded baseline: {len(baseline)} engine counts")
        print(f"Engine counts: {sorted(baseline['n_engines'].unique())}")
    
    print("\n" + "="*40)
    print("ALL DATA UTILITY TESTS PASSED")
    print("="*40)

