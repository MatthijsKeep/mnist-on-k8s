
import torch
import torch.nn as nn
import pytorch_lightning as pl

class MNISTLogReg(pl.LightningModule):
    def __init__(self, in_dim: int, n_classes: int = 10, lr: float = 1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(in_dim, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        acc = (self(x).argmax(1) == y).float().mean()
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
