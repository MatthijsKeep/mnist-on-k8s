import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_data: np.ndarray,
        stats_data: np.ndarray,
        labels_data: np.ndarray,
        is_train: bool = False,
        stats_scaler=None,
    ):
        # Convert to tensors once for efficiency + normalization
        images = (
            torch.tensor(image_data, dtype=torch.float32)
            .unsqueeze(1)
            .view(-1, 1, 28, 28)
        )  # (N, 1, 28, 28)
        image_normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])

        # Stats: Standard scale (zero-mean, unit-var)
        if is_train:
            self.stats_scaler = StandardScaler()
            stats_scaled = self.stats_scaler.fit_transform(stats_data)
        else:
            stats_scaled = stats_scaler.transform(stats_data)  # Use train's scaler

        self.image = image_normalize(images)  # (N, 1, 28, 28)
        self.stats = torch.tensor(stats_scaled, dtype=torch.float32)  # (N, 18)
        self.labels = torch.tensor(labels_data, dtype=torch.long)  # (N,)

        assert (
            self.image.shape[1] == 1
            and self.image.shape[2] == self.image.shape[3] == 28
        ), f"Image shape mismatch: {self.image.shape}"
        assert self.stats.shape[1] == 18, f"Stats shape mismatch: {self.stats.shape}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.image[idx]  # (1, 28, 28)
        stats = self.stats[idx]  # (18,)
        y = self.labels[idx]  # scalar
        return image, stats, y


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=256, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        from models.datamodules.helpers import fetch_and_prepare_features

        (train_image, train_stats, train_y), (val_image, val_stats, val_y) = (
            fetch_and_prepare_features()
        )
        print(
            f"Train data shapes: image {train_image.shape}, stats {train_stats.shape}, y {train_y.shape}"
        )
        self.train_dataset = MNISTDataset(
            train_image, train_stats, train_y, is_train=True
        )
        self.val_dataset = MNISTDataset(
            val_image,
            val_stats,
            val_y,
            is_train=False,
            stats_scaler=self.train_dataset.stats_scaler,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
