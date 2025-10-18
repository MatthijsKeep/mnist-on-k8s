
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from serving.model_loader import load_model, predict_from_features
from serving.feast_client import get_online_features, transform_online_features
from serving.metrics import REQUEST_TIME
from PIL import Image

import numpy as np
import io
import torch

app = FastAPI()
PRED_COUNT = Counter("predictions_total", "Total predictions", ["outcome"])  

# For development, allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

@app.get('/healthz')
def healthz():
    return {"ok": True}

@app.get('/metrics')
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.post('/predict_by_id')
def predict_by_id(image_id: int):
    with REQUEST_TIME.time():
        image, stats = get_online_features(image_id)
        image, stats = transform_online_features(image, stats)
        pred = int(predict_from_features(model, image, stats))
        PRED_COUNT.labels('ok').inc()
        return {"image_id": image_id, "pred": pred}

def extract_features_np(img28x28: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Model expects image, stats, image is shape (N, 1, 28, 28) stats is (N, 18)
    # incoming data is (28, 28) grayscale
    flat = img28x28.reshape(-1).astype(np.float32) / 255.0
    image = torch.tensor(img28x28, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    image_norm = (image - 0.1307) / 0.3081

    bins = np.linspace(0, 1, 17)
    hist = np.histogram(flat, bins=bins)[0]/len(flat)
    mean = flat.mean()
    var = flat.var()
    stats = torch.tensor(np.array(list(hist) + [mean, var], dtype=np.float32)).unsqueeze(0)  # (1, 18)
    return image_norm, stats

@app.post('/predict_drawing') # expects base64-encoded 28x28 grayscale image
async def predict_drawing(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert('L').resize((28, 28))
    arr = np.asarray(img)
    image, stats = extract_features_np(arr)
    pred = int(predict_from_features(model, image, stats))
    PRED_COUNT.labels('ok').inc()
    return {"pred": pred}