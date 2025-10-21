import os
import numpy as np
from feast import FeatureStore
from feast.repo_config import load_repo_config
import torch

REPO = os.environ.get("FEAST_REPO_PATH", "features/feast_repo")
REPO_YAML = os.environ.get(
    "FEAST_REPO_CONFIG", "features/feast_repo/feature_store.yaml"
)

def get_online_features(image_id: int) -> np.ndarray:
    # Load base config from yaml
    config = load_repo_config(REPO, REPO_YAML)

    # Override connection_string with actual env
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_password = os.environ.get(
        "REDIS_PASSWORD", ""
    )  # Empty fallback (no hardcode; use env only)

    config.online_store.type = "redis"

    # Build connection_string with password if provided
    if redis_password:
        config.online_store.connection_string = (
            f"{redis_host}:6379,password={redis_password}"
        )
    else:
        config.online_store.connection_string = f"{redis_host}:6379"

    store = FeatureStore(config=config, repo_path=REPO)
    resp = store.get_online_features(
        features=[
            "mnist_stats:flat",
            "mnist_stats:pix_mean",
            "mnist_stats:pix_var",
            *[f"mnist_stats:hist_{i}" for i in range(16)],
        ],
        entity_rows=[
            {
                "image_id": image_id,
            }
        ],
    ).to_dict()
    # Order: hist_0..15, pix_mean, pix_var to match training
    hist = [resp[f"hist_{i}"][0] for i in range(16)]
    mean = resp["pix_mean"][0]
    var = resp["pix_var"][0]
    # Deserialize flat
    flat_bytes = resp["flat"][0]
    if flat_bytes is None:
        raise ValueError("Flat image data missing")
    image = np.frombuffer(flat_bytes, dtype=np.float32).reshape(1, 28, 28)
    return image, np.array(hist + [mean, var], dtype=np.float32)


def transform_online_features(
    image: np.ndarray, stats: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    # feats is np.array(flat hist + mean, var), shape (19,), image is the first, stats is the rest
    # Model expects image, stats, image is shape (N, 1, 28, 28) stats is (N, 18)
    stats = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)  # (1, 18)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 1, 28, 28)
    image_norm = (image - 0.1307) / 0.3081
    return image_norm, stats
