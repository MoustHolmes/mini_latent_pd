"""Abstract base dataset for mdCATH MD trajectory data.

Handles all HDF5 I/O and indexes the dataset at init time. Subclasses
implement `_process_frame_pair` to convert raw coordinate arrays to
their target format.
"""

import random
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mini_latent_pd.data.mdcath_utils import (
    MDCATH_REPLICAS,
    MDCATH_TEMPS,
    MD_TO_STD_RESNAMES,
    decode_bytes_array,
    parse_atom_names_from_pdb,
)


class ProteinRecord:
    """Pre-computed per-protein metadata loaded once at init time.

    Attributes set at construction:
        id, h5_path, n_residues, sequence,
        heavy_atom_names, heavy_elements, heavy_z, heavy_resids,
        heavy_resnames_std, resnames_std, resids, heavy_mask, traj_index.

    Subclass datasets may attach additional attributes (e.g. _of3_slot_mapping)
    after construction.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MDCATHBaseDataset(Dataset, ABC):
    """Abstract base for mdCATH datasets.

    Discovers all `mdcath_dataset_*.h5` files in `data_dir`, builds a
    per-protein metadata index, and samples random frame pairs on each
    `__getitem__` call.

    Args:
        data_dir: Directory containing mdcath HDF5 files.
        samples_per_epoch: Virtual epoch size (each call draws a random pair).
        max_lag: Maximum frame lag when sampling pairs.
        max_seq_len: Skip proteins longer than this.
        temps: Temperature keys to include; default all MDCATH_TEMPS.
        replicas: Replica keys to include; default all MDCATH_REPLICAS.
    """

    def __init__(
        self,
        data_dir: str | Path,
        samples_per_epoch: int = 1000,
        max_lag: int = 200,
        max_seq_len: int = 900,
        temps: list[str] | None = None,
        replicas: list[str] | None = None,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.max_lag = max_lag
        self.temps = temps or MDCATH_TEMPS
        self.replicas = replicas or MDCATH_REPLICAS

        self.proteins: list[ProteinRecord] = []
        self._index_proteins(Path(data_dir), max_seq_len)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_proteins(self, data_dir: Path, max_seq_len: int) -> None:
        h5_files = sorted(data_dir.glob("**/*.h5"))
        for h5_path in h5_files:
            record = self._load_protein_record(h5_path, max_seq_len)
            if record is not None:
                self.proteins.append(record)
                total_frames = sum(nf for _, _, nf in record.traj_index)
                print(
                    f"  Indexed {record.id}: L={record.n_residues}, "
                    f"{total_frames} frames"
                )

    def _load_protein_record(self, h5_path: Path, max_seq_len: int) -> ProteinRecord | None:
        with h5py.File(h5_path, "r") as f:
            domain_id = list(f.keys())[0]
            domain = f[domain_id]

            z = domain["z"][:]
            resids = domain["resid"][:]
            resnames = decode_bytes_array(domain["resname"][:])

            pdb_text = domain["pdbProteinAtoms"][()].decode("utf-8")
            atom_names = parse_atom_names_from_pdb(pdb_text)

            # Element symbols: prefer stored 'element' field, fall back to deriving from z
            if "element" in domain:
                elements = decode_bytes_array(domain["element"][:])
            else:
                from rdkit.Chem import GetPeriodicTable
                pt = GetPeriodicTable()
                elements = np.array([pt.GetElementSymbol(int(zi)) for zi in z])

            heavy_mask = z > 1
            heavy_z = z[heavy_mask]
            heavy_resids = resids[heavy_mask]
            heavy_resnames = resnames[heavy_mask]
            heavy_elements = elements[heavy_mask]
            heavy_atom_names = atom_names[heavy_mask]

            # Normalise MD residue names to standard 3-letter codes
            heavy_resnames_std = np.array(
                [MD_TO_STD_RESNAMES.get(r, r) for r in heavy_resnames]
            )

            # Per-residue arrays (unique resids in their first-occurrence order)
            unique_resids, first_occ = np.unique(heavy_resids, return_index=True)
            order = np.argsort(first_occ)
            unique_resids = unique_resids[order]

            n_res = len(unique_resids)
            if n_res > max_seq_len:
                return None

            resnames_std = np.array(
                [heavy_resnames_std[np.where(heavy_resids == rid)[0][0]] for rid in unique_resids]
            )

            # Sequence string
            from openfold3.core.data.resources.residues import STANDARD_PROTEIN_RESIDUES_3
            _3to1 = {r: c for r, c in zip(
                STANDARD_PROTEIN_RESIDUES_3,
                "ARNDCQEGHILKMFPSTWYVX",
            )}
            sequence = "".join(_3to1.get(r, "X") for r in resnames_std)

            # Available trajectories
            traj_index = []
            for temp in self.temps:
                if temp not in domain:
                    continue
                for repl in self.replicas:
                    if repl not in domain[temp]:
                        continue
                    if "coords" not in domain[temp][repl]:
                        continue
                    n_frames = domain[temp][repl]["coords"].shape[0]
                    traj_index.append((temp, repl, n_frames))

        if not traj_index:
            return None

        return ProteinRecord(
            id=domain_id,
            h5_path=str(h5_path),
            n_residues=n_res,
            sequence=sequence,
            heavy_atom_names=heavy_atom_names,
            heavy_elements=heavy_elements,
            heavy_z=heavy_z,
            heavy_resids=heavy_resids,
            heavy_resnames_std=heavy_resnames_std,
            resnames_std=resnames_std,
            resids=unique_resids,
            heavy_mask=heavy_mask,
            traj_index=traj_index,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        prot = random.choice(self.proteins)
        temp, repl, n_frames = random.choice(prot.traj_index)

        idx0 = random.randint(0, n_frames - 2)
        max_valid_lag = min(n_frames - idx0 - 1, self.max_lag)
        lag = random.randint(1, max_valid_lag)

        with h5py.File(prot.h5_path, "r") as f:
            coords_x0 = f[prot.id][temp][repl]["coords"][idx0].astype(np.float32)
            coords_xt = f[prot.id][temp][repl]["coords"][idx0 + lag].astype(np.float32)

        return self._process_frame_pair(
            prot=prot,
            coords_x0=coords_x0,
            coords_xt=coords_xt,
            lag=lag,
            temp=int(temp),
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _process_frame_pair(
        self,
        prot: ProteinRecord,
        coords_x0: np.ndarray,
        coords_xt: np.ndarray,
        lag: int,
        temp: int,
    ) -> dict:
        """Convert a raw frame pair to the target format.

        Args:
            prot: Protein record with topology metadata.
            coords_x0: (N_all_atoms, 3) float32 coordinates of source frame.
            coords_xt: (N_all_atoms, 3) float32 coordinates of target frame.
            lag: Frame lag used when sampling.
            temp: Temperature in Kelvin.

        Returns:
            Sample dict in the subclass-specific format.
        """
        ...
