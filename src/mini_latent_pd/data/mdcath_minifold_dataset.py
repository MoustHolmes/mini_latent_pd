"""
MDCathToMinifoldDataset: lazy-loading mdCATH dataset producing Minifold Atom37 tensors.

Reads coordinates from HDF5 on each __getitem__ call instead of loading
everything into memory, making it feasible for the full 3TB dataset.

Supports two modes:
  - Frame-pair mode (data_dir): yields (x0, xt) pairs for dynamics training.
  - Single-frame mode (hdf5_path + keys): yields individual frames.
"""

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mini_latent_pd.data.mdcath_utils import (
    MDCATH_REPLICAS,
    MDCATH_TEMPS,
    aatype_from_resnames,
    aatype_to_sequence,
    build_scatter_indices,
    compute_atom37_mask,
    decode_bytes_array,
    flat_coords_to_atom37,
    parse_atom_names_from_pdb,
)


class MDCathToMinifoldDataset(Dataset):

    def __init__(
        self,
        hdf5_path=None,
        keys=None,
        transform=None,
        data_dir=None,
        samples_per_epoch=1000,
        max_lag=200,
        max_seq_len=900,
        embedding_dir=None,
    ):
        """
        Args:
            hdf5_path: Path to a single mdCATH HDF5 file (single-frame mode).
            keys: 'temperature/replica/frame_index' strings (single-frame mode).
            transform: Optional Minifold data transforms.
            data_dir: Directory containing mdCATH HDF5 files (frame-pair mode).
            samples_per_epoch: Virtual epoch size for frame-pair mode.
            max_lag: Maximum frame lag for pair sampling.
            max_seq_len: Skip proteins longer than this.
            embedding_dir: Directory with pre-computed ESM embeddings ({domain_id}.pt).
                Each .pt file should contain {"s_s": (L,1024), "s_z": (L,L,128)}.
        """
        self.transform = transform
        self.frame_pair_mode = data_dir is not None
        self.embedding_dir = Path(embedding_dir) if embedding_dir is not None else None

        if self.frame_pair_mode:
            self._init_frame_pair(data_dir, samples_per_epoch, max_lag, max_seq_len)
        else:
            self._init_single_frame(hdf5_path, keys)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_frame_pair(self, data_dir, samples_per_epoch, max_lag, max_seq_len):
        self.samples_per_epoch = samples_per_epoch
        self.max_lag = max_lag
        self.proteins = []

        data_dir = Path(data_dir)
        h5_files = sorted(data_dir.glob("**/*.h5"))

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                domain_id = list(f.keys())[0]
                domain = f[domain_id]

                # Per-atom topology (shared across all frames)
                z = domain["z"][:]
                resids = domain["resid"][:]
                resnames = decode_bytes_array(domain["resname"][:])
                atom_names = parse_atom_names_from_pdb(
                    domain["pdbProteinAtoms"][()].decode("utf-8")
                )

                heavy_mask = z > 1
                heavy_resids = resids[heavy_mask]
                heavy_resnames = resnames[heavy_mask]
                heavy_atom_names = atom_names[heavy_mask]

                aatype, residue_index, _ = aatype_from_resnames(
                    heavy_resnames, heavy_resids
                )
                num_res = len(aatype)
                if num_res > max_seq_len:
                    continue

                atom37_mask = compute_atom37_mask(aatype)
                sequence = aatype_to_sequence(aatype)
                valid_heavy_idx, res_scatter, atom37_scatter = build_scatter_indices(
                    heavy_atom_names, heavy_resids
                )

                # Discover available (temp, replica, n_frames) combos
                traj_index = []
                for temp in MDCATH_TEMPS:
                    if temp not in domain:
                        continue
                    for repl in MDCATH_REPLICAS:
                        if repl not in domain[temp]:
                            continue
                        if "coords" not in domain[temp][repl]:
                            continue
                        n_frames = domain[temp][repl]["coords"].shape[0]
                        traj_index.append((temp, repl, n_frames))

            if not traj_index:
                continue

            # Check for pre-computed embeddings
            emb_path = None
            if self.embedding_dir is not None:
                candidate = self.embedding_dir / f"{domain_id}.pt"
                if candidate.exists():
                    emb_path = str(candidate)

            total_frames = sum(nf for _, _, nf in traj_index)
            self.proteins.append(
                {
                    "id": domain_id,
                    "h5_path": str(h5_path),
                    "sequence": sequence,
                    "n_residues": num_res,
                    "aatype": torch.from_numpy(aatype),
                    "atom37_mask": torch.from_numpy(atom37_mask),
                    # Scatter indices for fast atom37 conversion
                    "heavy_mask": heavy_mask,
                    "valid_heavy_idx": valid_heavy_idx,
                    "res_scatter": res_scatter,
                    "atom37_scatter": atom37_scatter,
                    # Available trajectories (temp, replica, n_frames)
                    "traj_index": traj_index,
                    # Pre-computed ESM embedding path (None if not available)
                    "embedding_path": emb_path,
                }
            )
            print(
                f"  Indexed {domain_id}: L={num_res}, {total_frames} frames, "
                f"seq={sequence[:20]}..."
                + (", emb=✓" if emb_path else "")
            )

    def _init_single_frame(self, hdf5_path, keys):
        self.hdf5_path = hdf5_path
        self.keys = keys or []

        with h5py.File(hdf5_path, "r") as f:
            self.domain_id = list(f.keys())[0]
            domain = f[self.domain_id]

            z = domain["z"][:]
            resids = domain["resid"][:]
            resnames = decode_bytes_array(domain["resname"][:])
            pdb_text = domain["pdbProteinAtoms"][()].decode("utf-8")
            atom_names = parse_atom_names_from_pdb(pdb_text)

            self.heavy_mask = z > 1
            heavy_resids = resids[self.heavy_mask]
            heavy_resnames = resnames[self.heavy_mask]
            heavy_atom_names = atom_names[self.heavy_mask]

            self.aatype, self.residue_index, _ = aatype_from_resnames(
                heavy_resnames, heavy_resids
            )
            self.num_res = len(self.aatype)
            self.atom37_mask = compute_atom37_mask(self.aatype)
            self.valid_heavy_idx, self.res_scatter, self.atom37_scatter = (
                build_scatter_indices(heavy_atom_names, heavy_resids)
            )

    # ------------------------------------------------------------------
    # Length / getitem
    # ------------------------------------------------------------------

    def __len__(self):
        if self.frame_pair_mode:
            return self.samples_per_epoch
        return len(self.keys)

    def __getitem__(self, idx):
        if self.frame_pair_mode:
            return self._getitem_pair()
        return self._getitem_single(idx)

    def _read_frame_atom37(self, h5_file, domain_id, temp, repl, frame_idx, prot):
        """Read one frame from HDF5 and convert to atom37 on the fly."""
        coords_flat = h5_file[domain_id][temp][repl]["coords"][frame_idx]  # (n_atoms, 3)
        return flat_coords_to_atom37(
            coords_flat,
            prot["heavy_mask"],
            prot["valid_heavy_idx"],
            prot["res_scatter"],
            prot["atom37_scatter"],
            prot["n_residues"],
        )

    def _getitem_pair(self):
        prot = random.choice(self.proteins)
        temp, repl, n_frames = random.choice(prot["traj_index"])

        idx0 = random.randint(0, n_frames - 2)
        max_valid_lag = min(n_frames - idx0 - 1, self.max_lag)
        lag = random.randint(1, max_valid_lag)

        with h5py.File(prot["h5_path"], "r") as f:
            x0 = self._read_frame_atom37(f, prot["id"], temp, repl, idx0, prot)
            xt = self._read_frame_atom37(f, prot["id"], temp, repl, idx0 + lag, prot)

        result = {
            "id": prot["id"],
            "x0": torch.from_numpy(x0),
            "xt": torch.from_numpy(xt),
            "lag": lag,
            "temp": int(temp),
            "sequence": prot["sequence"],
            "aatype": prot["aatype"],
            "atom37_mask": prot["atom37_mask"],
        }

        # Lazy-load pre-computed ESM embeddings from disk
        if prot["embedding_path"] is not None:
            emb = torch.load(prot["embedding_path"], map_location="cpu", weights_only=True)
            result["s_s"] = emb["s_s"]
            result["s_z"] = emb["s_z"]

        return result

    def _getitem_single(self, idx):
        key = self.keys[idx]
        temp, replica, frame_idx = key.split("/")
        frame_idx = int(frame_idx)

        with h5py.File(self.hdf5_path, "r") as f:
            coords_flat = f[self.domain_id][temp][replica]["coords"][frame_idx]

        atom37_pos = flat_coords_to_atom37(
            coords_flat,
            self.heavy_mask,
            self.valid_heavy_idx,
            self.res_scatter,
            self.atom37_scatter,
            self.num_res,
        )

        feature_dict = {
            "aatype": torch.tensor(self.aatype, dtype=torch.long),
            "all_atom_positions": torch.from_numpy(atom37_pos),
            "all_atom_mask": torch.from_numpy(self.atom37_mask),
            "residue_index": torch.tensor(self.residue_index, dtype=torch.long),
            "seq_length": torch.tensor([self.num_res], dtype=torch.long),
        }

        if self.transform:
            feature_dict = self.transform(feature_dict)

        return feature_dict
