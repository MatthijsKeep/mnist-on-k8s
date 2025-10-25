from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from serving.model_loader import load_model, predict_from_features, load_model_from_mlflow
from serving.feast_client import get_online_features, transform_online_features
from serving.metrics import REQUEST_TIME
from PIL import Image

import os
import numpy as np
import io
import torch

# Get config from environment variables (set in Kubernetes deployment)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server.ml.svc.cluster.local:5000")
MINIO_ENDPOINT_URL = os.getenv("MINIO_ENDPOINT_URL", "http://minio.ml.svc.cluster.local:9000")
MODEL_URI = os.getenv("MODEL_URI", "models:/mnist_cnn_model/1")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model AFTER FastAPI server starts
    global model
    print("=" * 50)
    print("Starting application...")
    print(f"Loading model from: {MODEL_URI}")
    try:
        model = load_model_from_mlflow(MODEL_URI)
        print("Model loaded successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield  # Application runs here
    
    # Shutdown: cleanup if needed
    print("Shutting down application...")

app = FastAPI(lifespan=lifespan)

PRED_COUNT = Counter("predictions_total", "Total predictions", ["outcome"])

# For development, allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for model
model = None

@app.get("/healthz")
def healthz():
    # Return healthy immediately (even if model still loading)
    # For production, check if model is loaded:
    # if model is None:
    #     return {"ok": False, "error": "Model not loaded yet"}, 503
    return {"ok": True}

@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# Load model at startup, the model is either MNIST_CNN_Model or MNIST_CNN_Model_complex
# model uri is built in train.py like this: model_uri = f"runs:/{run_id}/model"
MODEL_URI = "models:/mnist_cnn_model/1"
model = load_model_from_mlflow(MODEL_URI)

@app.post("/predict_by_id")
def predict_by_id(image_id: int):
    if model is None:
        return {"error": "Model not ready"}, 503
    
    with REQUEST_TIME.time():
        image, stats = get_online_features(image_id)
        image, stats = transform_online_features(image, stats)
        prediction, confidences = predict_from_features(model, image, stats)
        PRED_COUNT.labels("ok").inc()
        return {
            "image_id": image_id,
            "pred": int(prediction),
            "confidences": confidences,
        }


def extract_features_np(
    img28x28: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Model expects image, stats, image is shape (N, 1, 28, 28) stats is (N, 18)
    # incoming data is (28, 28) grayscale
    flat = img28x28.reshape(-1).astype(np.float32) / 255.0
    image = (
        torch.tensor(img28x28, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, 28, 28)
    image_norm = (image - 0.1307) / 0.3081

    bins = np.linspace(0, 1, 17)
    hist = np.histogram(flat, bins=bins)[0] / len(flat)
    mean = flat.mean()
    var = flat.var()
    stats = torch.tensor(
        np.array(list(hist) + [mean, var], dtype=np.float32)
    ).unsqueeze(0)  # (1, 18)
    return image_norm, stats


@app.post("/predict_drawing")  # expects base64-encoded 28x28 grayscale image
async def predict_drawing(file: UploadFile):
    if model is None:
        return {"error": "Model not ready"}, 503
    img = Image.open(io.BytesIO(await file.read())).convert("L").resize((28, 28))
    arr = np.asarray(img)
    image, stats = extract_features_np(arr)
    pred, conf = predict_from_features(model, image, stats)
    PRED_COUNT.labels("ok").inc()
    return {"pred": int(pred), "confidences": conf}
