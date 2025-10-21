import torch
import torch.nn as nn
import lightning as L

from torch.optim.lr_scheduler import ReduceLROnPlateau


class SimpleCNN(L.LightningModule):
    """A simple CNN combined with an MLP for image and stats classification."""

    def __init__(self, in_dim_stats: int, n_classes: int = 10, lr: float = 1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # (B, 1, 28, 28) -> (B, 16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (B, 16, 28, 28) -> (B, 16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # (B, 16, 14, 14) -> (B, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (B, 32, 14, 14) -> (B, 32, 7, 7)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # (B, 32, 7, 7) -> (B, 64, 7, 7)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            # (B, 64, 7, 7) -> (B, 64, 4, 4)
            nn.Flatten(),
            # (B, 64, 4, 4) -> (B, 1024)
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            # (B, 1024) -> (B, 128)
        )

        # MLP branch for flat features (stats)
        self.mlp_stats = nn.Sequential(
            nn.Linear(in_dim_stats, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            # (B, in_dim_stats) -> (B, 64)
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
            # (B, 192) -> (B, n_classes)
        )
        # Initialize for stable gradients
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image, stats):
        # Image branch
        img_feat = self.cnn_block(image)
        # (B, 1024)
        img_feat = self.cnn_fc(img_feat)
        # (B, 128)

        # Stats branch
        stat_feat = self.mlp_stats(stats)
        # (B, 64)

        # Concat and classify
        combined = torch.cat([img_feat, stat_feat], dim=1)
        # (B, 192)
        logits = self.classifier(combined)
        # (B, n_classes)
        return logits

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def _shared_step(self, batch, step_name: str):
        image_flat, stats, y = batch
        logits = self(image_flat, stats)
        loss = self.criterion(logits, y)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(
            f"{step_name}_acc",
            (logits.argmax(1) == y).float().mean(),
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=3e-4, weight_decay=1e-5
        )  # Lower lr, L2 reg for stability
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },  # Decays if val_loss stalls
        }
