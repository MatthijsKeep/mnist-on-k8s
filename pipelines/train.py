import torch
import lightning as L

from datetime import datetime
from pathlib import Path
from typing import Tuple
from lightning.pytorch.loggers.mlflow import MLFlowLogger

from models.simple_cnn import SimpleCNN
from models.datamodules.mnist import MNISTDataModule


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

MODEL_SAVE_PATH = ARTIFACTS_DIR / "model.pt"


def get_sample_batch(
    dm: MNISTDataModule,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieve and return a sample batch from the training dataloader."""
    sample_batch = next(iter(dm.train_dataloader()))
    print(f"Sample batch shapes: {[x.shape for x in sample_batch]}")
    return sample_batch


def create_trainer(epochs: int = 10, device: str = "cpu") -> L.Trainer:
    """Create and configure the PyTorch Lightning Trainer."""
    # Log to our MLFlow server, running at localhost:5000, have to create an experiment first

    mlflow_logger = MLFlowLogger(
        experiment_name="mnist_cnn",
        tracking_uri="http://localhost:5000",
        log_model=True,
    )

    early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
        monitor="val_acc", patience=1, mode="max"
    )
    return L.Trainer(
        max_epochs=epochs,
        accelerator=device,
        devices=1,
        callbacks=[early_stop_callback],
        logger=mlflow_logger,
    )


def save_model(
    model: SimpleCNN,
    stats_dim: int,
    feature_refs: list[str],
) -> None:
    """Save the model state dict with metadata for reproducibility."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim_stats": stats_dim,  # Corrected to reference stats dimension
            "n_classes": model.classifier[-1].out_features,
            "lr": model.hparams.lr,
            "feature_refs": feature_refs,
            "saved_at": datetime.now().isoformat(),
        },
        MODEL_SAVE_PATH,
    )
    print(f"Saved model to {MODEL_SAVE_PATH} with Feast features")


def main() -> None:
    """Main training pipeline for the MNIST model."""
    L.seed_everything(42, workers=True)  # For reproducibility

    batch_size = 256
    num_workers = 4
    lr = 1e-2
    n_classes = 10

    dm = MNISTDataModule(batch_size=batch_size, num_workers=num_workers)
    dm.setup()

    X, stats, y = get_sample_batch(dm)
    stats_dim = stats.shape[1]

    model = SimpleCNN(in_dim_stats=stats_dim, n_classes=n_classes, lr=lr)
    trainer = create_trainer()
    trainer.fit(model, datamodule=dm)

    # Define feature references for reproducibility (Feast-specific)
    feature_refs = (
        [
            "mnist_stats:flat",
            "mnist_stats:pix_mean",
            "mnist_stats:pix_var",
        ]
        + [f"mnist_stats:hist_{i}" for i in range(16)]
        + ["mnist_stats:label"]
    )

    save_model(model, stats_dim, feature_refs)


if __name__ == "__main__":
    main()
