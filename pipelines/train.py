
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import polars as pl
from pathlib import Path
from models.mnist_logreg import MNISTLogReg

ARTIFACTS = Path('artifacts')
ARTIFACTS.mkdir(exist_ok=True, parents=True)

def main():
    df = pl.read_parquet(ARTIFACTS / 'features.parquet')
    # Use histogram + stats as features
    X = torch.tensor(df.select([c for c in df.columns if c.startswith('hist_')] + ['pix_mean','pix_var']).to_numpy(), dtype=torch.float32)
    y = torch.tensor(df['label'].to_numpy(), dtype=torch.long)
    ds = TensorDataset(X, y)
    n = len(ds)
    n_val = int(0.1*n)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n-n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = MNISTLogReg(in_dim=X.shape[1])
    trainer = pl.Trainer(max_epochs=5, accelerator='auto', devices=1, logger=False)
    trainer.fit(model, train_loader, val_loader)
    # Save
    torch.save({'state_dict': model.state_dict(), 'in_dim': X.shape[1]}, ARTIFACTS / 'model.pt')
    print('Saved model to artifacts/model.pt')

if __name__ == '__main__':
    main()
