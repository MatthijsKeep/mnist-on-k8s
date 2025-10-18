
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

import torch

def predict_from_features(model, image: torch.Tensor, stats: torch.Tensor) -> tuple[int, dict[int, float]]:
    with torch.no_grad():
        logits = model(image, stats)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_class = logits.argmax(dim=1).item()
        confs_ret = {i: round(float(probs[i].item() * 100), 2) for i in range(len(probs))}
        return pred_class, confs_ret

