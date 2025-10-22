import json
import lightning as L
import mlflow
import mlflow.pytorch
import os
import torch

from datetime import datetime
from pathlib import Path
from typing import Tuple
from lightning.pytorch.loggers.mlflow import MLFlowLogger

from models.complex_cnn import ComplexCNN
from models.datamodules.mnist import MNISTDataModule

# Constants for better maintainability (group related configs)
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

MODEL_SAVE_PATH = ARTIFACTS_DIR / "model.pt"

# MLflow experiment and registry names as constants
EXPERIMENT_NAME = "mnist_cnn"
REGISTERED_MODEL_NAME = "MNIST_CNN_Model_complex"

# Training hyperparameters as constants (avoids magic numbers)
BATCH_SIZE = 256
NUM_WORKERS = 4
LEARNING_RATE = 1e-2
NUM_CLASSES = 10
EPOCHS = 10
SEED = 42

# MinIO/S3 config (moved to env setup in main; constants for clarity, but use os.environ for security)
MINIO_ENDPOINT_URL = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MLFLOW_TRACKING_URI = "http://localhost:5000"


def get_sample_batch(
    dm: MNISTDataModule,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieve and return a sample batch from the training dataloader."""
    sample_batch = next(iter(dm.train_dataloader()))
    print(f"Sample batch shapes: {[x.shape for x in sample_batch]}")
    return sample_batch


def create_trainer(epochs: int = EPOCHS, device: str = "cpu") -> L.Trainer:
    """Create and configure the PyTorch Lightning Trainer."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    mlflow_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=MLFLOW_TRACKING_URI,
        log_model=False,
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
    model: ComplexCNN,
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
    print(f"Saved model locally to {MODEL_SAVE_PATH} with Feast features")

    # Create sample inputs (use actual batch shapes from your datamodule)
    sample_image = torch.randn(1, 1, 28, 28)  # Batch of 1 image
    sample_stats = torch.randn(1, stats_dim)  # Batch of 1 stats
    sample_output = model(
        sample_image, sample_stats
    )  # Forward pass to get output shape

    signature = mlflow.models.infer_signature(
        {
            "image": sample_image.numpy(),
            "stats": sample_stats.numpy(),
        },  # Inputs as dict of numpy
        sample_output.detach().numpy(),  # Output as numpy
    )

    mlflow.pytorch.log_model(
        model,
        "model",
        signature=signature,
    )

    metadata = {
        "in_dim_stats": stats_dim,
        "n_classes": model.classifier[-1].out_features,
        "lr": model.hparams.lr,
        "feature_refs": feature_refs,
        "saved_at": datetime.now().isoformat(),
    }

    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Logged model to MLflow.")


def main() -> None:
    """Main training pipeline for the MNIST model."""
    L.seed_everything(SEED, workers=True)  # For reproducibility

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    dm = MNISTDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    dm.setup()

    X, stats, y = get_sample_batch(dm)
    stats_dim = stats.shape[1]

    model = ComplexCNN(in_dim_stats=stats_dim, n_classes=NUM_CLASSES, lr=LEARNING_RATE)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = create_trainer(epochs=EPOCHS, device=device)
    trainer.fit(model, datamodule=dm)

    # Get the tracked run_id from the logger (MLFlowLogger creates/uses one)
    run_id = trainer.logger.run_id
    print(f"Training run ID: {run_id}")
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

    # Resume the existing run and log the model to it
    with mlflow.start_run(run_id=run_id):
        save_model(model, stats_dim, feature_refs)  # No run_id needed
        print(f"Model logged to run {run_id}")

    # Register the model (outside the run context, as registration doesn't need active run)
    model_uri = f"runs:/{run_id}/model"
    try:
        mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        print(
            f"Model registered successfully as '{REGISTERED_MODEL_NAME}' "
            f"(view at {MLFLOW_TRACKING_URI}/#/models/{REGISTERED_MODEL_NAME})"
        )
    except Exception as e:
        print(
            f"Registration failed: {e} (check MinIO permissions or if model artifact exists)"
        )


if __name__ == "__main__":
    main()
