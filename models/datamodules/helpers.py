# data/features.py
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from feast import FeatureStore

def fetch_and_prepare_features(repo_path="features/feast_repo", n_samples=60000):
    """Fetch from Feast and prepare image_flat, stats, y tensors for splits."""
    store = FeatureStore(repo_path=repo_path)
    
    # Create entity_df
    entity_df = pd.DataFrame({
        'image_id': range(n_samples),
        'event_timestamp': [datetime.now()] * n_samples
    })
    
    # Fetch historical features
    feature_refs = [
        'mnist_stats:flat', 'mnist_stats:pix_mean', 'mnist_stats:pix_var', 'mnist_stats:label'
    ] + [f'mnist_stats:hist_{i}' for i in range(16)]
    training_df = store.get_historical_features(
        entity_df=entity_df, features=feature_refs
    ).to_df()
    
    assert len(training_df) == n_samples, "Missing joins—check image_id range"
    
    # Prepare arrays
    assert np.frombuffer(training_df['flat'].iloc[0], dtype=np.float32).shape == (784,), "Flat image shape mismatch, must be (784,)"
    training_df['flat'] = training_df['flat'].apply(lambda x: np.array(np.frombuffer(x, dtype=np.float32).reshape(784,), dtype=np.float64))
    image_flat = np.stack(training_df['flat'].values)  # (N, 784)
    
    stats_cols = ['pix_mean', 'pix_var'] + [f'hist_{i}' for i in range(16)]
    stats_data = training_df[stats_cols].values.astype(np.float32)  # (N, 18)
    
    y_data = training_df['label'].values.astype(np.int64)  # (N,)
    
    # Split (80/20, stratified)
    train_idx, val_idx = train_test_split(
        range(len(y_data)), test_size=0.2, stratify=y_data, random_state=42
    )
    
    train_image = image_flat[train_idx]
    train_stats = stats_data[train_idx]
    train_y = y_data[train_idx]
    
    val_image = image_flat[val_idx]
    val_stats = stats_data[val_idx]
    val_y = y_data[val_idx]
    
    print(f"Train sizes: image {train_image.shape}, stats {train_stats.shape}, y {train_y.shape}")
    print(f"Val sizes: image {val_image.shape}, stats {val_stats.shape}, y {val_y.shape}")

    return (train_image, train_stats, train_y), (val_image, val_stats, val_y)
