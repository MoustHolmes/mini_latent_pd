"""MDCATHAtom37Dataset: mdCATH frame pairs in atom37 format.

Used by the all-atom companion project. Inherits HDF5 loading from
MDCATHBaseDataset and converts coordinates to the (L, 37, 3) atom37
representation using utilities from mdcath_utils.
"""

import numpy as np
import torch

from mini_latent_pd.data.mdcath_base_dataset import MDCATHBaseDataset, ProteinRecord
from mini_latent_pd.data.mdcath_utils import (
    aatype_from_resnames,
    aatype_to_sequence,
    build_scatter_indices,
    compute_atom37_mask,
    flat_coords_to_atom37,
)


class MDCATHAtom37Dataset(MDCATHBaseDataset):
    """mdCATH dataset that outputs frame pairs in atom37 format.

    Each sample contains:
        x0:          (L, 37, 3) float32 — source frame in atom37
        xt:          (L, 37, 3) float32 — target frame in atom37
        aatype:      (L,)       int64   — residue type indices
        atom37_mask: (L, 37)   float32 — 1.0 for existing atoms
        lag:         int        — frame lag
        temp:        int        — temperature in K
        sequence:    str        — 1-letter amino acid sequence
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pre-compute per-protein scatter indices for atom37 conversion
        for prot in self.proteins:
            self._precompute_atom37(prot)

    def _precompute_atom37(self, prot: ProteinRecord) -> None:
        """Attach atom37 scatter indices and type arrays to a protein record."""
        aatype, _, _ = aatype_from_resnames(prot.heavy_resnames_std, prot.heavy_resids)
        valid_heavy_idx, res_scatter, atom37_scatter = build_scatter_indices(
            prot.heavy_atom_names, prot.heavy_resids
        )
        atom37_mask = compute_atom37_mask(aatype)

        # Store as attributes on the record object (ProteinRecord uses __slots__,
        # so we attach dynamically via object.__setattr__ trick by storing alongside)
        prot._atom37_aatype = aatype
        prot._atom37_mask = atom37_mask
        prot._valid_heavy_idx = valid_heavy_idx
        prot._res_scatter = res_scatter
        prot._atom37_scatter = atom37_scatter

    def _process_frame_pair(
        self,
        prot: ProteinRecord,
        coords_x0: np.ndarray,
        coords_xt: np.ndarray,
        lag: int,
        temp: int,
    ) -> dict:
        x0 = flat_coords_to_atom37(
            coords_x0,
            prot.heavy_mask,
            prot._valid_heavy_idx,
            prot._res_scatter,
            prot._atom37_scatter,
            prot.n_residues,
        )
        xt = flat_coords_to_atom37(
            coords_xt,
            prot.heavy_mask,
            prot._valid_heavy_idx,
            prot._res_scatter,
            prot._atom37_scatter,
            prot.n_residues,
        )
        return {
            "id": prot.id,
            "x0": torch.from_numpy(x0),
            "xt": torch.from_numpy(xt),
            "aatype": torch.from_numpy(prot._atom37_aatype),
            "atom37_mask": torch.from_numpy(prot._atom37_mask),
            "lag": lag,
            "temp": temp,
            "sequence": prot.sequence,
        }
