
import os
import numpy as np
from feast import FeatureStore

REPO = os.environ.get('FEAST_REPO_PATH', 'features/feast_repo')

def get_online_features(image_id: int) -> np.ndarray:
    store = FeatureStore(repo_path=REPO)
    resp = store.get_online_features(
        features=[
            'mnist_stats:pix_mean',
            'mnist_stats:pix_var',
            *[f'mnist_stats:hist_{i}' for i in range(16)],
        ],
        entity_rows=[{'image_id': image_id}],
    ).to_dict()
    # Order: hist_0..15, pix_mean, pix_var to match training
    hist = [resp[f'mnist_stats__hist_{i}'][0] for i in range(16)]
    mean = resp['mnist_stats__pix_mean'][0]
    var = resp['mnist_stats__pix_var'][0]
    return np.array(hist + [mean, var], dtype=np.float32)
