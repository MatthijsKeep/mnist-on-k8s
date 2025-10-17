
import torch
from pathlib import Path
from models.simple_cnn import SimpleCNN
import numpy as np

ARTIFACT = Path('artifacts/model.pt')

def load_model():
    ckpt = torch.load(ARTIFACT, map_location='cpu')
    model = SimpleCNN(in_dim_stats=ckpt['in_dim_stats'],
                      n_classes=ckpt['n_classes'],
                      lr=ckpt['lr'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model

def predict_from_features(model, image: torch.Tensor, stats: torch.Tensor) -> int:
    import torch
    with torch.no_grad():
        logits = model(image, stats)
        return logits.argmax(1).item()
