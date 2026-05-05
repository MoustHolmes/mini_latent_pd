"""PyTorch Lightning DataModule for OpenFold3 dynamics training.

Combines MDCATHOpenFold3Dataset (dynamics) with an optional
PDBOpenFold3Dataset (folding) and exposes them through a collated
DataLoader.
"""

from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from mini_latent_pd.data.of3_collator import of3_collate


class OF3DynamicsDataModule(pl.LightningDataModule):
    """DataModule for OpenFold3 dynamics (+ optional folding) training.

    Args:
        mdcath_data_dir: Directory of mdcath HDF5 files.
        pdb_paths: Optional list of PDB/CIF files for mixed folding training.
        use_template: Whether to include x0 as a template in dynamics samples.
        msa_dir: Directory of precomputed `.a3m` files (or None).
        batch_size: Samples per batch.
        num_workers: DataLoader workers.
        samples_per_epoch: Virtual epoch size for the mdCATH dataset.
        max_lag: Max frame lag for dynamics sampling.
        max_seq_len: Skip proteins longer than this.
        dynamics_weight: Relative weight of dynamics samples (for mixed training).
        folding_weight: Relative weight of PDB folding samples.
        temps: mdCATH temperature keys to include.
        replicas: mdCATH replica keys to include.
    """

    def __init__(
        self,
        mdcath_data_dir: str | Path,
        pdb_paths: list[str | Path] | None = None,
        use_template: bool = True,
        msa_dir: str | Path | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        samples_per_epoch: int = 1000,
        max_lag: int = 200,
        max_seq_len: int = 900,
        dynamics_weight: float = 1.0,
        folding_weight: float = 1.0,
        temps: list[str] | None = None,
        replicas: list[str] | None = None,
    ):
        super().__init__()
        self.mdcath_data_dir = Path(mdcath_data_dir)
        self.pdb_paths = [Path(p) for p in pdb_paths] if pdb_paths else []
        self.use_template = use_template
        self.msa_dir = Path(msa_dir) if msa_dir is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.samples_per_epoch = samples_per_epoch
        self.max_lag = max_lag
        self.max_seq_len = max_seq_len
        self.dynamics_weight = dynamics_weight
        self.folding_weight = folding_weight
        self.temps = temps
        self.replicas = replicas

        self._train_dataset = None

    def setup(self, stage: str | None = None) -> None:
        from mini_latent_pd.data.mdcath_of3_dataset import MDCATHOpenFold3Dataset

        dyn_dataset = MDCATHOpenFold3Dataset(
            data_dir=self.mdcath_data_dir,
            use_template=self.use_template,
            msa_dir=self.msa_dir,
            samples_per_epoch=self.samples_per_epoch,
            max_lag=self.max_lag,
            max_seq_len=self.max_seq_len,
            temps=self.temps,
            replicas=self.replicas,
        )

        if self.pdb_paths:
            from mini_latent_pd.data.pdb_of3_dataset import PDBOpenFold3Dataset

            fold_dataset = PDBOpenFold3Dataset(
                pdb_paths=self.pdb_paths,
                msa_dir=self.msa_dir,
                max_seq_len=self.max_seq_len,
            )

            # Weighted mixing: repeat folding dataset to match requested ratio
            all_datasets = [dyn_dataset, fold_dataset]
            weights = (
                [self.dynamics_weight] * len(dyn_dataset)
                + [self.folding_weight] * len(fold_dataset)
            )
            self._train_dataset = ConcatDataset(all_datasets)
            self._train_weights = weights
        else:
            self._train_dataset = dyn_dataset
            self._train_weights = None

    def train_dataloader(self) -> DataLoader:
        if self._train_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self._train_weights,
                num_samples=len(self._train_weights),
                replacement=True,
            )
            return DataLoader(
                self._train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=of3_collate,
                pin_memory=False,
            )

        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=of3_collate,
            pin_memory=False,
        )
