"""Data module that provides pairs of consecutive MNIST digits.

Each sample is a triplet (source_image, target_image, source_label) where
target_label == (source_label + 1) % 10. For example, a source image of a '3'
is paired with a randomly chosen image of a '4'.
"""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class ConsecutiveDigitDataset(Dataset):
    """A dataset that yields (source_image, target_image, source_label) pairs.

    Images are grouped by digit. Each __getitem__ call picks a random source
    image of some digit N and a random target image of digit (N+1) % 10.

    Args:
        mnist_dataset: An underlying MNIST dataset (with transforms applied).
        length: The virtual length of this dataset per epoch. Since pairs are
            sampled randomly, this controls how many samples per epoch.
    """

    def __init__(self, mnist_dataset: MNIST, length: int | None = None) -> None:
        super().__init__()

        # Group image indices by label
        self.digit_indices: dict[int, list[int]] = {i: [] for i in range(10)}
        for idx in range(len(mnist_dataset)):
            _, label = mnist_dataset[idx]
            self.digit_indices[label].append(idx)

        self.mnist_dataset = mnist_dataset
        self._length = length or len(mnist_dataset)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Pick a random source digit class
        source_label = torch.randint(0, 10, (1,)).item()
        target_label = (source_label + 1) % 10

        # Pick random images from each class
        src_indices = self.digit_indices[source_label]
        tgt_indices = self.digit_indices[target_label]
        src_idx = src_indices[torch.randint(0, len(src_indices), (1,)).item()]
        tgt_idx = tgt_indices[torch.randint(0, len(tgt_indices), (1,)).item()]

        source_image, _ = self.mnist_dataset[src_idx]
        target_image, _ = self.mnist_dataset[tgt_idx]

        return source_image, target_image, source_label


class ConsecutiveDigitDataModule(L.LightningDataModule):
    """Lightning DataModule for consecutive-digit MNIST pairs.

    Each batch contains ``(source_images, target_images, source_labels)`` where
    ``target_label == (source_label + 1) % 10``.

    Args:
        data_dir: Root directory for raw MNIST data.
        batch_size: Mini-batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin CUDA memory.
        train_length: Virtual epoch length for training. Defaults to 55000.
        val_length: Virtual epoch length for validation. Defaults to 5000.
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
        train_length: int = 55000,
        val_length: int = 5000,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_length = train_length
        self.val_length = val_length

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # Split underlying MNIST into train/val, then wrap each in ConsecutiveDigitDataset
            train_mnist, val_mnist = random_split(mnist_full, [55000, 5000])
            # We use the full training set as the pool for pairing (not just the split)
            # because we need all digits available for every class.
            self.train_dataset = ConsecutiveDigitDataset(mnist_full, length=self.train_length)
            self.val_dataset = ConsecutiveDigitDataset(mnist_full, length=self.val_length)

        if stage == "test" or stage is None:
            mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_dataset = ConsecutiveDigitDataset(mnist_test, length=len(mnist_test))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
