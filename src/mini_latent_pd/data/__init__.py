"""Data modules for diffusion playground."""

from mini_latent_pd.data.MNIST_datamodule import MNISTDataModule
from mini_latent_pd.data.moons_datamodule import MoonsDataModule
from mini_latent_pd.data.consecutive_digit_datamodule import (
    ConsecutiveDigitDataModule,
    ConsecutiveDigitDataset,
)

__all__ = [
    "MNISTDataModule",
    "MoonsDataModule",
    "ConsecutiveDigitDataModule",
    "ConsecutiveDigitDataset",
]
