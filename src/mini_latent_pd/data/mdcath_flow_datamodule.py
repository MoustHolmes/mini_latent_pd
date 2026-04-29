"""Lightning DataModule wrapping MDCathToMinifoldDataset for distance flow training."""

import lightning as L
from torch.utils.data import DataLoader

from mini_latent_pd.data.mdcath_minifold_dataset import MDCathToMinifoldDataset
from mini_latent_pd.data.collator import mdcath_collate, BucketBatchSampler


class MDCathFlowDataModule(L.LightningDataModule):
    """DataModule for distance-flow training on mdCATH frame pairs.

    Wraps ``MDCathToMinifoldDataset`` in frame-pair mode with the
    ``mdcath_collate`` padder and optional ``BucketBatchSampler``.

    Args:
        data_dir: Directory containing mdCATH HDF5 files.
        embedding_dir: Directory with precomputed ESM embeddings.
        batch_size: Samples per batch.
        samples_per_epoch: Virtual epoch size (random frame-pair sampling).
        max_lag: Maximum frame lag between source and target.
        max_seq_len: Skip proteins longer than this.
        num_workers: DataLoader workers.
        pin_memory: Pin tensors to GPU memory.
        use_bucket_sampler: Group proteins by length to reduce padding.
        bucket_limits: Upper bounds for length buckets.
    """

    def __init__(
        self,
        data_dir: str = "data/mdcath/data",
        embedding_dir: str = "data/embeddings",
        batch_size: int = 4,
        samples_per_epoch: int = 1000,
        val_samples_per_epoch: int = 200,
        max_lag: int = 200,
        max_seq_len: int = 900,
        num_workers: int = 0,
        pin_memory: bool = True,
        use_bucket_sampler: bool = False,
        bucket_limits: list[int] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        common = dict(
            data_dir=self.hparams.data_dir,
            max_lag=self.hparams.max_lag,
            max_seq_len=self.hparams.max_seq_len,
            embedding_dir=self.hparams.embedding_dir,
        )
        if stage in ("fit", None):
            self.train_dataset = MDCathToMinifoldDataset(
                samples_per_epoch=self.hparams.samples_per_epoch,
                **common,
            )
            self.val_dataset = MDCathToMinifoldDataset(
                samples_per_epoch=self.hparams.val_samples_per_epoch,
                **common,
            )
        if stage in ("test", None):
            self.test_dataset = MDCathToMinifoldDataset(
                samples_per_epoch=self.hparams.val_samples_per_epoch,
                **common,
            )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)

    def _make_loader(self, dataset: MDCathToMinifoldDataset, shuffle: bool) -> DataLoader:
        sampler = None
        if self.hparams.use_bucket_sampler and hasattr(dataset, "proteins"):
            lengths = [p["n_residues"] for p in dataset.proteins]
            # Repeat lengths to cover virtual epoch
            lengths_full = (lengths * (len(dataset) // len(lengths) + 1))[: len(dataset)]
            sampler = BucketBatchSampler(
                lengths=lengths_full,
                batch_size=self.hparams.batch_size,
                bucket_limits=self.hparams.bucket_limits,
                shuffle=shuffle,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=mdcath_collate,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            collate_fn=mdcath_collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
