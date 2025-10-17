
import torch
from pathlib import Path
from models.mnist_logreg import MNISTLogReg
import numpy as np

ARTIFACT = Path('artifacts/model.pt')

def load_model():
    ckpt = torch.load(ARTIFACT, map_location='cpu')
    model = MNISTLogReg(in_dim=ckpt['in_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model

def predict_from_features(model, feats: np.ndarray):
    import torch
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        return logits.argmax(1).item()
