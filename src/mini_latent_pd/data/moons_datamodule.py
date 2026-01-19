import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_moons


class MoonsDataModule(L.LightningDataModule):
    def __init__(
        self,
        n_samples: int = 50000,
        noise: float = 0.05,
        batch_size: int = 2048,
        num_workers: int = 0,
    ):
        """
        Initializes the DataModule for the scikit-learn moons dataset.

        Args:
            n_samples (int): Total number of data points to generate.
            noise (float): Standard deviation of Gaussian noise added to the data.
            batch_size (int): The batch size for the dataloaders.
            num_workers (int): The number of worker processes for loading data.
        """
        super().__init__()
        self.save_hyperparameters()
        # This dataset doesn't require transforms like MNIST
        self.transform = None

    def prepare_data(self):
        # This method is used for downloading data, etc.
        # Since we generate the data on the fly, we can leave this empty.
        pass

    def setup(self, stage: str = None):
        """
        Generates and splits the dataset. Called on every GPU.
        """
        # Generate the full dataset, keeping both the data and the labels
        moons_data, moons_labels = make_moons(
            n_samples=self.hparams.n_samples, noise=self.hparams.noise, random_state=42
        )

        # Create a TensorDataset that includes both data and labels
        full_dataset = TensorDataset(
            torch.from_numpy((moons_data)).float(),
            torch.from_numpy(moons_labels).long(),  # Labels should be long integers
        )

        # Split into train, validation, and test sets (80%, 10%, 10%)
        train_size = int(0.8 * self.hparams.n_samples)
        val_size = int(0.1 * self.hparams.n_samples)
        test_size = self.hparams.n_samples - train_size - val_size

        self.train_split, self.val_split, self.test_split = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_split, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
