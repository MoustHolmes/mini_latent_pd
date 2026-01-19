"""Data modules for diffusion playground."""

from mini_latent_pd.data.MNIST_datamodule import MNISTDataModule
from mini_latent_pd.data.moons_datamodule import MoonsDataModule

__all__ = [
    "MNISTDataModule",
    "MoonsDataModule",
]
