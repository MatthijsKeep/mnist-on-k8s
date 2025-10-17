import os
import numpy as np
from feast import FeatureStore

REPO = os.environ.get('FEAST_REPO_PATH', 'features/feast_repo')

def get_online_features(image_id: int) -> np.ndarray:
    store = FeatureStore(repo_path=REPO)
    resp = store.get_online_features(
        features=[
            'mnist_stats:flat',
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
    image = np.array(resp['mnist_stats__flat'][0], dtype=np.float32).reshape(1, 28, 28)  # (1, 28, 28)
    return image, np.array(hist + [mean, var], dtype=np.float32)

def transform_online_features(image: np.ndarray, stats: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    # feats is np.array(flat hist + mean, var), shape (19,), image is the first, stats is the rest
    # Model expects image, stats, image is shape (N, 1, 28, 28) stats is (N, 18)
    stats = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)  # (1, 18)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 1, 28, 28)
    image_norm = (image - 0.1307) / 0.3081
    return image_norm, stats
