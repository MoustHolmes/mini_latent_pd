from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from mini_latent_pd.data.MNIST_datamodule import MNISTDataModule
from mini_latent_pd.data.moons_datamodule import MoonsDataModule


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path / "data")


# --- MoonsDataModule Tests ---
@pytest.fixture
def moons_datamodule():
    """Fixture for MoonsDataModule with a smaller dataset for testing."""
    return MoonsDataModule(n_samples=1000, batch_size=32, num_workers=0)


def test_moons_datamodule_attributes(moons_datamodule):
    """Test if the MoonsDataModule has the correct attributes."""
    assert moons_datamodule.hparams.batch_size == 32
    assert moons_datamodule.hparams.num_workers == 0


def test_moons_setup(moons_datamodule):
    """Test setup creates the correct splits for MoonsDataModule."""
    moons_datamodule.setup()
    assert moons_datamodule.train_split is not None
    assert moons_datamodule.val_split is not None
    assert len(moons_datamodule.train_split) == 800
    assert len(moons_datamodule.val_split) == 100


def test_moons_train_dataloader(moons_datamodule):
    """Test if Moons train_dataloader returns the correct type and batch size."""
    moons_datamodule.setup()
    loader = moons_datamodule.train_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == moons_datamodule.hparams.batch_size
    batch = next(iter(loader))
    assert len(batch) == 2  # (x, y) tuple
    assert batch[0].shape == (moons_datamodule.hparams.batch_size, 2)
    assert batch[1].shape == (moons_datamodule.hparams.batch_size,)


# --- MNISTDataModule Tests ---
@pytest.fixture
def mnist_datamodule(data_dir):
    return MNISTDataModule(data_dir=data_dir, batch_size=32, num_workers=0)


def test_mnist_prepare_data(mnist_datamodule):
    """Test prepare_data creates the required files for MNIST."""
    mnist_datamodule.prepare_data()
    data_path = Path(mnist_datamodule.data_dir)
    assert (data_path / "MNIST").exists()


def test_mnist_setup(mnist_datamodule):
    """Test setup creates the correct splits for MNISTDataModule."""
    mnist_datamodule.prepare_data()
    mnist_datamodule.setup(stage="fit")
    assert mnist_datamodule.train_split is not None
    assert mnist_datamodule.val_split is not None
    assert len(mnist_datamodule.train_split) == 55000
    assert len(mnist_datamodule.val_split) == 5000

    mnist_datamodule.setup(stage="test")
    assert mnist_datamodule.test_split is not None
    assert len(mnist_datamodule.test_split) == 10000


def test_mnist_train_dataloader(mnist_datamodule):
    """Test if MNIST train_dataloader returns the correct type and batch size."""
    mnist_datamodule.prepare_data()
    mnist_datamodule.setup(stage="fit")
    loader = mnist_datamodule.train_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == mnist_datamodule.batch_size
    batch = next(iter(loader))
    assert len(batch) == 2  # (x, y) tuple
    assert batch[0].shape[1:] == (1, 28, 28)
    assert batch[1].shape[0] == mnist_datamodule.batch_size


def test_mnist_test_dataloader(mnist_datamodule):
    """Test if MNIST test_dataloader returns the correct type and batch size."""
    mnist_datamodule.prepare_data()
    mnist_datamodule.setup(stage="test")

    loader = mnist_datamodule.test_dataloader()

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == mnist_datamodule.batch_size

    batch = next(iter(loader))
    assert len(batch) == 2
    assert batch[0].shape[1:] == (1, 28, 28)  # Image dimensions (channels, height, width)
    assert batch[1].shape[0] == mnist_datamodule.batch_size
