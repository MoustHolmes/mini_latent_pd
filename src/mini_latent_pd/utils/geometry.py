"""Geometry utilities for structure encoding/decoding.

Functions for converting between atom37 coordinates and distogram
representations compatible with MiniFold's FoldingTrunk recycling format.
"""

import numpy as np
import torch
import torch.nn.functional as F

from mini_latent_pd.models.minifold.utils import residue_constants as rc


def pseudo_beta_from_atom37(
    atom37_pos: torch.Tensor,
    aatype: torch.Tensor,
) -> torch.Tensor:
    """Extract pseudo-Cb positions: Cb for all residues, Ca for Glycine.

    Args:
        atom37_pos: (*, L, 37, 3) all-atom positions in Angstroms.
        aatype: (*, L) residue type indices (0-20).

    Returns:
        Pseudo-beta positions (*, L, 3).
    """
    ca_idx = rc.atom_order["CA"]  # 1
    cb_idx = rc.atom_order["CB"]  # 3
    is_gly = aatype == rc.restype_order["G"]

    is_gly_3d = is_gly.unsqueeze(-1).expand_as(atom37_pos[..., ca_idx, :])
    return torch.where(
        is_gly_3d,
        atom37_pos[..., ca_idx, :],
        atom37_pos[..., cb_idx, :],
    )


def coords_to_distogram(
    atom37_pos: torch.Tensor,
    aatype: torch.Tensor,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    no_bins: int = 64,
) -> torch.Tensor:
    """Convert atom37 coordinates to a one-hot distogram.

    Produces the same binning used by MiniFold's FoldingTrunk recycling loop
    and ``distogram_loss``.

    Args:
        atom37_pos: (B, L, 37, 3) all-atom positions in Angstroms.
        aatype: (B, L) residue type indices.
        min_bin: Lower distance bin edge (Angstroms).
        max_bin: Upper distance bin edge (Angstroms).
        no_bins: Number of histogram bins (64 for MiniFold).

    Returns:
        One-hot distogram (B, L, L, no_bins).
    """
    pseudo_beta = pseudo_beta_from_atom37(atom37_pos, aatype)

    sq_dists = torch.sum(
        (pseudo_beta.unsqueeze(-2) - pseudo_beta.unsqueeze(-3)) ** 2,
        dim=-1,
    )

    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=atom37_pos.device)
    boundaries = boundaries ** 2

    bin_indices = torch.sum(sq_dists.unsqueeze(-1) > boundaries, dim=-1)
    return F.one_hot(bin_indices.long(), no_bins).to(atom37_pos.dtype)


def coords_to_bin_indices(
    atom37_pos: torch.Tensor,
    aatype: torch.Tensor,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    no_bins: int = 64,
) -> torch.Tensor:
    """Convert atom37 coordinates to distance bin indices (no one-hot).

    Same binning as :func:`coords_to_distogram` but returns integer indices
    directly, avoiding the memory overhead of one-hot encoding.

    Args:
        atom37_pos: (B, L, 37, 3) all-atom positions in Angstroms.
        aatype: (B, L) residue type indices.
        min_bin: Lower distance bin edge (Angstroms).
        max_bin: Upper distance bin edge (Angstroms).
        no_bins: Number of histogram bins (64 for MiniFold).

    Returns:
        Bin indices (B, L, L) long, values in [0, no_bins).
    """
    pseudo_beta = pseudo_beta_from_atom37(atom37_pos, aatype)

    sq_dists = torch.sum(
        (pseudo_beta.unsqueeze(-2) - pseudo_beta.unsqueeze(-3)) ** 2,
        dim=-1,
    )

    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=atom37_pos.device)
    boundaries = boundaries ** 2

    bin_indices = torch.sum(sq_dists.unsqueeze(-1) > boundaries, dim=-1)
    return bin_indices.long()


# ---------------------------------------------------------------------------
# Kabsch alignment utilities
# ---------------------------------------------------------------------------


def kabsch_align(
    P: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """Align point cloud *P* onto *Q* via Kabsch algorithm (NumPy).

    Args:
        P: (N, 3) mobile point cloud.
        Q: (N, 3) reference point cloud.

    Returns:
        Aligned copy of *P* with shape (N, 3).
    """
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign @ U.T
    return Pc @ R.T + Q.mean(0)


def kabsch_rmsd(
    P: torch.Tensor,
    Q: torch.Tensor,
) -> torch.Tensor:
    """Kabsch-aligned RMSD between two point clouds (PyTorch).

    Args:
        P: (N, 3) mobile points.
        Q: (N, 3) reference points.

    Returns:
        Scalar RMSD after optimal superposition.
    """
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    sign = torch.diag(torch.tensor([1.0, 1.0, d.sign()], device=P.device, dtype=P.dtype))
    R = Vt.T @ sign @ U.T
    P_aligned = Pc @ R.T
    return (P_aligned - Qc).pow(2).sum(-1).mean().sqrt()
