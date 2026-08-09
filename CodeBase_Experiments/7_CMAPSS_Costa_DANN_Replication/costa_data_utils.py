import math
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Utilities import config

# Paths - use the SAME preprocessed data as all other experiments
WINDOWED_DATA_PATH = Path(config.WINDOWED_DATA_PATH)

# ==========
# FEATURE SELECTION (14 Correlation_FS features)
# ==========

# 14 Correlation_FS features (from Phase 2)
SELECTED_FEATURES = ['s11', 's4', 's12', 's7', 's15', 's21', 's20', 's17', 
                     's2', 's3', 's8', 's13', 's9', 's14']

# Mapping: feature name -> index in original 14-feature windowed data
FEATURE_MAP = {
    's2': 0, 's3': 1, 's4': 2, 's7': 3, 's8': 4, 's9': 5,
    's11': 6, 's12': 7, 's13': 8, 's14': 9, 's15': 10,
    's17': 11, 's20': 12, 's21': 13
}

def extract_selected_features(X, selected_features=SELECTED_FEATURES):
    indices = [FEATURE_MAP[feat] for feat in selected_features]
    return X[:, :, indices]

def load_cmapss_dataset(dataset_name='FD002', use_all_engines=True, n_engines=None):
    
    print(f"\nLoading {dataset_name} saved split...")
    
    # Load from the SAME preprocessed windowed data as all other experiments
    X_train_full = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_X_train_windowed.npy')
    y_train_full = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_y_train_windowed.npy')
    X_val = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_X_val_windowed.npy')
    y_val = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_y_val_windowed.npy')
    X_test = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_X_test_windowed.npy')
    y_test = np.load(WINDOWED_DATA_PATH / f'{dataset_name}_y_test_windowed.npy')
    
    # Sample engines if needed
    sampled_engines = None
    if not use_all_engines and n_engines is not None:
        train_ids_path = WINDOWED_DATA_PATH / f'{dataset_name}_train_ids_windowed.csv'
        train_ids = pd.read_csv(train_ids_path)
        unique_engines = train_ids['engine'].unique()
        total_engines = len(unique_engines)
        
        if n_engines > total_engines:
            raise ValueError(
                f"Requested {n_engines} engines, but {dataset_name} only has {total_engines}."
            )
        
        np.random.seed(config.RANDOM_SEED)
        sampled_engines = sorted(
            np.random.choice(unique_engines, size=n_engines, replace=False)
        )
        mask = train_ids['engine'].isin(sampled_engines).values
        X_train_full = X_train_full[mask]
        y_train_full = y_train_full[mask]
        print(f"Sampled {n_engines} train engines from saved train split.")
    else:
        print("Using all saved train engines.")
    
    # Extract 14 Correlation_FS features
    X_train = extract_selected_features(X_train_full)
    X_val = extract_selected_features(X_val)
    X_test = extract_selected_features(X_test)
    
    # Use consistent RUL_CAP for normalization (same as all other experiments)
    rul_max = float(config.RUL_CAP)
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples (saved split)")
    print(f"Test:  {len(X_test)} samples (saved split)")
    print(f"RUL max for normalization: {rul_max:.2f}")
    
    dataset = {
        'X_train': X_train.astype(np.float32),
        'y_train': (y_train_full / rul_max).astype(np.float32),
        'X_val': X_val.astype(np.float32),
        'y_val': (y_val / rul_max).astype(np.float32),
        'X_test': X_test.astype(np.float32),
        'y_test': (y_test / rul_max).astype(np.float32),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'rul_max': rul_max,
        'sampled_engines': sampled_engines,
    }
    
    return dataset

def create_domain_batches(
    source_data,
    target_data,
    batch_size=512,
    oversample_smaller=True
):
    
    X_source = source_data['X_train']
    y_source = source_data['y_train']
    X_target = target_data['X_train']
    
    n_source = len(X_source)
    n_target = len(X_target)
    
    # Calculate number of batches
    n_batches_source = int(math.ceil(n_source / batch_size))
    n_batches_target = int(math.ceil(n_target / batch_size))
    n_batches = max(n_batches_source, n_batches_target)
    
    print(f"\nBatch configuration:")
    print(f"Source: {n_source} samples -> {n_batches_source} batches")
    print(f"Target: {n_target} samples -> {n_batches_target} batches")
    print(f"Total batches per epoch: {n_batches}")
    print(f"Batch size: {batch_size}")
    
    while True:  # Infinite generator
        # Shuffle indices
        source_indices = np.random.permutation(n_source)
        target_indices = np.random.permutation(n_target)
        
        for batch_idx in range(n_batches):
            # Get source batch
            start_source = (batch_idx * batch_size) % n_source
            end_source = min(start_source + batch_size, n_source)
            if end_source - start_source < batch_size and oversample_smaller:
                # Wrap around (oversample)
                indices_source = np.concatenate([
                    source_indices[start_source:],
                    source_indices[:batch_size - (n_source - start_source)]
                ])
            else:
                indices_source = source_indices[start_source:end_source]
            
            X_source_batch = X_source[indices_source]
            y_source_batch = y_source[indices_source]
            
            # Get target batch
            start_target = (batch_idx * batch_size) % n_target
            end_target = min(start_target + batch_size, n_target)
            if end_target - start_target < batch_size and oversample_smaller:
                # Wrap around (oversample)
                indices_target = np.concatenate([
                    target_indices[start_target:],
                    target_indices[:batch_size - (n_target - start_target)]
                ])
            else:
                indices_target = target_indices[start_target:end_target]
            
            X_target_batch = X_target[indices_target]
            
            # Combine source and target
            X_batch = np.concatenate([X_source_batch, X_target_batch], axis=0)
            
            # RUL labels (only for source, target gets dummy zeros)
            y_rul_batch = np.concatenate([
                y_source_batch,
                np.zeros(len(X_target_batch), dtype=np.float32)
            ])
            
            # Domain labels (0 = source, 1 = target)
            y_domain_batch = np.concatenate([
                np.zeros(len(X_source_batch), dtype=np.float32),
                np.ones(len(X_target_batch), dtype=np.float32)
            ])
            
            # Sample weights (for RUL loss: only source samples count)
            sample_weights_rul = np.concatenate([
                np.ones(len(X_source_batch), dtype=np.float32),
                np.zeros(len(X_target_batch), dtype=np.float32)
            ])
            
            # Sample weights (for domain loss: all samples count)
            sample_weights_domain = np.ones(len(X_batch), dtype=np.float32)
            
            yield (
                X_batch,
                {'rul_output': y_rul_batch, 'domain_output': y_domain_batch},
                {'rul_output': sample_weights_rul, 'domain_output': sample_weights_domain}
            )

if __name__ == '__main__':
    """Test data loading."""
    print("=================")
    print("Test data loading")
    print("=================")
    
    # Load FD002 (source) and FD001 (target)
    source_data = load_cmapss_dataset('FD002', use_all_engines=True)
    target_data = load_cmapss_dataset('FD001', use_all_engines=False, n_engines=10)
    
    print(f"\nSource rul_max: {source_data['rul_max']}")
    print(f"Target rul_max: {target_data['rul_max']}")
    print(f"Both should be {config.RUL_CAP} (consistent!)")
    
    
    # Create batch generator
    batch_gen = create_domain_batches(source_data, target_data, batch_size=512)
    
    # Get one batch
    X_batch, y_batch, weights = next(batch_gen)
    
    print(f"\nBatch shapes:")
    print(f"X_batch: {X_batch.shape}")
    print(f"y_rul_batch: {y_batch['rul_output'].shape}")
    print(f"y_domain_batch: {y_batch['domain_output'].shape}")
    print(f"weights_rul: {weights['rul_output'].shape}")
    print(f"weights_domain: {weights['domain_output'].shape}")
    
    print(f"\nDomain distribution in batch:")
    print(f"Source samples (domain=0): {(y_batch['domain_output'] == 0).sum()}")
    print(f"Target samples (domain=1): {(y_batch['domain_output'] == 1).sum()}")
    
    print("=================")
