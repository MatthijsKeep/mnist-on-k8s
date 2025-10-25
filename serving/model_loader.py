import torch
from pathlib import Path
import mlflow.pytorch
import os

from models.simple_cnn import SimpleCNN

ARTIFACT = Path("artifacts/model.pt")

# MinIO/S3 config
MINIO_ENDPOINT_URL = "http://minio.ml.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MLFLOW_TRACKING_URI = "http://mlflow-server.ml.svc.cluster.local:5000"

def load_model():
    ckpt = torch.load(ARTIFACT, map_location="cpu")
    model = SimpleCNN(
        in_dim_stats=ckpt["in_dim_stats"], n_classes=ckpt["n_classes"], lr=ckpt["lr"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def load_model_from_mlflow(model_uri: str):
    # Use environment variable for tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set MinIO/S3 credentials from environment
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000")
    
    print(f"Loading model from: {model_uri}")
    print(f"Tracking URI: {tracking_uri}")
    
    model = mlflow.pytorch.load_model(model_uri=model_uri, map_location="cpu")
    model.eval()
    return model

def predict_from_features(
    model, image: torch.Tensor, stats: torch.Tensor
) -> tuple[int, dict[int, float]]:
    with torch.no_grad():
        logits = model(image, stats)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_class = logits.argmax(dim=1).item()
        confs_ret = {
            i: round(float(probs[i].item() * 100), 2) for i in range(len(probs))
        }
        return pred_class, confs_ret
