"""MDCATHOpenFold3Dataset: mdCATH frame pairs in OpenFold3 batch format.

Converts raw mdCATH MD trajectory frame pairs into the dict-of-tensors
format expected by the OpenFold3 model. Supports toggling template
conditioning (x0 as structural template) for ablation studies.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mini_latent_pd.data.mdcath_base_dataset import MDCATHBaseDataset, ProteinRecord

# OpenFold3 residue / atom constants
from openfold3.core.data.resources.residues import (
    STANDARD_PROTEIN_RESIDUES_3,
    STANDARD_RESIDUES_WITH_GAP_3,
    get_with_unknown_3_to_idx,
)
from openfold3.core.data.resources.token_atom_constants import (
    AA_NAME_TO_ATOM_NAMES,
    TOKEN_NAME_TO_ATOM_NAMES,
)

# Number of restype classes in OF3 (len(STANDARD_RESIDUES_WITH_GAP_3) = 32)
_N_RESTYPES = len(STANDARD_RESIDUES_WITH_GAP_3)

# Template distogram bins (matching OF3 defaults)
_TEMPL_MIN_BIN: float = 3.25
_TEMPL_MAX_BIN: float = 50.75
_TEMPL_N_BINS: int = 39
_TEMPL_BIN_LOWER_SQ = np.linspace(_TEMPL_MIN_BIN, _TEMPL_MAX_BIN, _TEMPL_N_BINS) ** 2
_TEMPL_BIN_UPPER_SQ = np.concatenate(
    [_TEMPL_BIN_LOWER_SQ[1:], np.array([1e8], dtype=np.float32)]
)

# Default loss weights (used for dynamics training)
_DEFAULT_LOSS_WEIGHTS = {
    "bond": 4.0,
    "smooth_lddt": 4.0,
    "mse": 4.0,
    "plddt": 1e-4,
    "pde": 1e-4,
    "experimentally_resolved": 1e-4,
    "pae": 1e-4,
    "distogram": 3e-2,
}

# Element symbol → atomic number (standard protein atoms)
_ELEMENT_TO_Z: dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "SE": 34, "P": 15,
    "FE": 26, "ZN": 30, "CA": 20, "MG": 12, "MN": 25, "CL": 17,
}


class MDCATHOpenFold3Dataset(MDCATHBaseDataset):
    """mdCATH dataset that outputs OpenFold3-compatible batch dicts.

    Each sample corresponds to a (x0, xt) frame pair from a single MD
    trajectory. The x0 frame can optionally be used as a structural
    template to condition the model on the initial conformation (for
    dynamics training). Setting `use_template=False` produces a
    template-free sample suitable for folding training.

    Args:
        data_dir: Directory containing mdcath HDF5 files.
        use_template: If True, encode x0 as a template feature.
        msa_dir: Directory of precomputed MSA `.a3m` files named
            `{domain_id}.a3m`. If None, falls back to single-sequence MSA.
        **kwargs: Forwarded to MDCATHBaseDataset.
    """

    def __init__(
        self,
        data_dir: str | Path,
        use_template: bool = True,
        msa_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)
        self.use_template = use_template
        self.msa_dir = Path(msa_dir) if msa_dir is not None else None

        # Pre-compute per-protein OF3 atom layout
        for prot in self.proteins:
            _precompute_of3_layout(prot)

    # ------------------------------------------------------------------
    # Core conversion
    # ------------------------------------------------------------------

    def _process_frame_pair(
        self,
        prot: ProteinRecord,
        coords_x0: np.ndarray,
        coords_xt: np.ndarray,
        lag: int,
        temp: int,
    ) -> dict:
        L = prot.n_residues
        heavy_x0 = coords_x0[prot.heavy_mask]  # (N_heavy, 3)
        heavy_xt = coords_xt[prot.heavy_mask]   # (N_heavy, 3)

        slot_mapping = prot._of3_slot_mapping   # list[list[int]], one per residue
        num_atoms_per_token = prot._of3_num_atoms_per_token  # (L,) int

        # Build flat atom arrays
        n_atom = int(sum(num_atoms_per_token))
        ref_pos = np.zeros((n_atom, 3), dtype=np.float32)
        gt_pos = np.zeros((n_atom, 3), dtype=np.float32)
        ref_mask = np.zeros(n_atom, dtype=np.int32)
        ref_element = np.zeros(n_atom, dtype=np.int64)
        ref_charge = np.zeros(n_atom, dtype=np.float32)
        ref_atom_name_chars = np.zeros((n_atom, 4), dtype=np.int64)
        atom_to_token = np.zeros(n_atom, dtype=np.int32)

        flat_idx = 0
        for res_idx, (resname, slots) in enumerate(
            zip(prot.resnames_std, slot_mapping)
        ):
            expected_atoms = TOKEN_NAME_TO_ATOM_NAMES.get(
                resname, TOKEN_NAME_TO_ATOM_NAMES.get("UNK", [])
            )
            for atom_idx, (aname, heavy_idx) in enumerate(zip(expected_atoms, slots)):
                atom_to_token[flat_idx] = res_idx
                if heavy_idx >= 0:
                    ref_pos[flat_idx] = heavy_x0[heavy_idx]
                    gt_pos[flat_idx] = heavy_xt[heavy_idx]
                    ref_mask[flat_idx] = 1
                    ref_element[flat_idx] = max(int(prot.heavy_z[heavy_idx]) - 1, 0)
                # Atom name: encode as 4 chars via ord(c) - 32
                padded = aname.ljust(4)[:4]
                ref_atom_name_chars[flat_idx] = [ord(c) - 32 for c in padded]
                flat_idx += 1

        start_atom_index = np.concatenate(
            [np.array([0]), np.cumsum(num_atoms_per_token)[:-1]]
        )

        # Restype one-hot (L, 32)
        restype_idx = np.array(
            [get_with_unknown_3_to_idx(r) for r in prot.resnames_std], dtype=np.int64
        )
        restype_oh = F.one_hot(
            torch.from_numpy(restype_idx), num_classes=_N_RESTYPES
        ).int()

        is_protein = torch.tensor(
            [1 if r in STANDARD_PROTEIN_RESIDUES_3 else 0 for r in prot.resnames_std],
            dtype=torch.int32,
        )

        features = {
            # Token-level indexing
            "residue_index": torch.arange(L, dtype=torch.int32),
            "token_index": torch.arange(L, dtype=torch.int32),
            "asym_id": torch.ones(L, dtype=torch.int32),
            "entity_id": torch.ones(L, dtype=torch.int32),
            "sym_id": torch.ones(L, dtype=torch.int32),
            # Residue features
            "restype": restype_oh,
            "is_protein": is_protein,
            "is_rna": torch.zeros(L, dtype=torch.int32),
            "is_dna": torch.zeros(L, dtype=torch.int32),
            "is_ligand": torch.zeros(L, dtype=torch.int32),
            "is_atomized": torch.zeros(L, dtype=torch.int32),
            # Reference conformer features (derived from x0)
            "ref_pos": torch.from_numpy(ref_pos),
            "ref_mask": torch.from_numpy(ref_mask),
            "ref_element": F.one_hot(
                torch.from_numpy(ref_element), num_classes=119
            ).int(),
            "ref_charge": torch.from_numpy(ref_charge),
            "ref_atom_name_chars": F.one_hot(
                torch.from_numpy(ref_atom_name_chars), num_classes=64
            ).int(),
            "ref_space_uid": torch.from_numpy(atom_to_token).int(),
            # Connectivity
            "token_bonds": torch.zeros((L, L), dtype=torch.int32),
            "token_mask": torch.ones(L, dtype=torch.float32),
            "atom_mask": torch.from_numpy(ref_mask).float(),
            "start_atom_index": torch.from_numpy(start_atom_index).int(),
            "num_atoms_per_token": torch.from_numpy(num_atoms_per_token).int(),
            "atom_to_token_index": torch.from_numpy(atom_to_token),
            # Permutation-alignment fields at top level (single-chain: trivial values)
            # permutation_alignment.py reads these from the top-level batch dict
            "mol_entity_id": torch.ones(L, dtype=torch.int32),
            "mol_sym_id": torch.ones(L, dtype=torch.int32),
            "mol_sym_component_id": torch.ones(L, dtype=torch.int32),
            "mol_sym_token_index": torch.arange(L, dtype=torch.int32),
            # Ground truth (xt is the diffusion target)
            "ground_truth": {
                "atom_positions": torch.from_numpy(gt_pos),
                "atom_resolved_mask": torch.from_numpy(ref_mask).float(),
                "token_mask": torch.ones(L, dtype=torch.float32),
                "token_index": torch.arange(L, dtype=torch.int32),
                "atom_mask": torch.from_numpy(ref_mask).float(),
                "num_atoms_per_token": torch.from_numpy(num_atoms_per_token).int(),
                "is_ligand": torch.zeros(L, dtype=torch.int32),
                # Permutation-alignment fields also kept in ground_truth for loss module
                "mol_entity_id": torch.ones(L, dtype=torch.int32),
                "mol_sym_id": torch.ones(L, dtype=torch.int32),
                "mol_sym_component_id": torch.ones(L, dtype=torch.int32),
                "mol_sym_token_index": torch.arange(L, dtype=torch.int32),
            },
            # Loss weights — scalar per sample (batch dim added by collator)
            "loss_weights": {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in _DEFAULT_LOSS_WEIGHTS.items()
            },
            # Metadata — used by training loop logging and RMSD callback
            "pdb_id": prot.id,
            "preferred_chain_or_interface": "A",
            "domain_id": prot.id,
            "temp": torch.tensor(temp, dtype=torch.int32),
            "lag": torch.tensor(lag, dtype=torch.int32),
        }

        # MSA features
        features.update(self._build_msa_features(prot, restype_oh, L))

        # Template features
        features.update(
            self._build_template_features(
                prot, heavy_x0, restype_oh, L
            )
        )

        return features

    # ------------------------------------------------------------------
    # MSA helpers
    # ------------------------------------------------------------------

    def _build_msa_features(
        self, prot: ProteinRecord, restype_oh: torch.Tensor, L: int
    ) -> dict:
        if self.msa_dir is not None:
            msa_path = self.msa_dir / f"{prot.id}.a3m"
            if msa_path.exists():
                return _load_msa_from_a3m(msa_path, L, _N_RESTYPES)

        # Single-sequence fallback: the protein sequence itself as the only MSA row
        msa = restype_oh.unsqueeze(0).int()  # (1, L, 32)
        return {
            "msa": msa,
            "has_deletion": torch.zeros((1, L), dtype=torch.float32),
            "deletion_value": torch.zeros((1, L), dtype=torch.float32),
            "profile": restype_oh.float(),
            "deletion_mean": torch.zeros(L, dtype=torch.float32),
            "msa_mask": torch.ones((1, L), dtype=torch.float32),
            "num_paired_seqs": torch.tensor(0, dtype=torch.int64),
        }

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------

    def _build_template_features(
        self,
        prot: ProteinRecord,
        heavy_x0: np.ndarray,
        restype_oh: torch.Tensor,
        L: int,
    ) -> dict:
        if not self.use_template:
            return _zero_template_features(L, n_templ=1)

        # Pseudo-beta positions: CB for non-GLY, CA for GLY
        pseudo_beta = np.zeros((L, 3), dtype=np.float32)
        pseudo_beta_mask = np.zeros(L, dtype=np.float32)
        backbone_frame_mask = np.zeros(L, dtype=np.float32)
        backbone_coords = np.zeros((L, 3, 3), dtype=np.float32)  # (L, 3=N/CA/C, 3)

        for res_idx, (resname, slots) in enumerate(
            zip(prot.resnames_std, prot._of3_slot_mapping)
        ):
            expected = TOKEN_NAME_TO_ATOM_NAMES.get(resname, [])
            name_to_slot = {aname: i for i, aname in enumerate(expected)}

            # Pseudo-beta (CB or CA for GLY)
            pb_atom = "CA" if resname == "GLY" else "CB"
            if pb_atom in name_to_slot:
                hidx = slots[name_to_slot[pb_atom]]
                if hidx >= 0:
                    pseudo_beta[res_idx] = heavy_x0[hidx]
                    pseudo_beta_mask[res_idx] = 1.0

            # Backbone frame atoms (N, CA, C)
            all_present = True
            for bb_slot, bb_name in enumerate(["N", "CA", "C"]):
                if bb_name in name_to_slot:
                    hidx = slots[name_to_slot[bb_name]]
                    if hidx >= 0:
                        backbone_coords[res_idx, bb_slot] = heavy_x0[hidx]
                    else:
                        all_present = False
                else:
                    all_present = False
            if all_present:
                backbone_frame_mask[res_idx] = 1.0

        # Distogram from pseudo-beta positions (1, L, L, 39)
        distogram = _compute_distogram(pseudo_beta, pseudo_beta_mask)

        # Unit vectors from backbone frames (1, L, L, 3)
        unit_vectors = _compute_unit_vectors(
            backbone_coords, backbone_frame_mask, L
        )

        return {
            "template_restype": restype_oh.unsqueeze(0).int(),  # (1, L, 32)
            "template_pseudo_beta_mask": torch.from_numpy(pseudo_beta_mask).unsqueeze(0),
            "template_backbone_frame_mask": torch.from_numpy(backbone_frame_mask).unsqueeze(0),
            "template_distogram": torch.from_numpy(distogram),  # (1, L, L, 39)
            "template_unit_vector": torch.from_numpy(unit_vectors),  # (1, L, L, 3)
        }


# ------------------------------------------------------------------
# Per-protein OF3 layout pre-computation
# ------------------------------------------------------------------

def _precompute_of3_layout(prot: ProteinRecord) -> None:
    """Attach OF3 atom-slot mapping to a protein record (in-place).

    For each residue, builds a list mapping each expected OF3 atom slot
    to an index into the protein's heavy-atom arrays (or -1 if absent).
    """
    slot_mapping: list[list[int]] = []
    num_atoms_per_token: list[int] = []

    for resname, resid in zip(prot.resnames_std, prot.resids):
        expected = TOKEN_NAME_TO_ATOM_NAMES.get(resname, TOKEN_NAME_TO_ATOM_NAMES.get("UNK", []))
        n_atoms = len(expected)
        num_atoms_per_token.append(n_atoms)

        # Map atom name → heavy-atom index within this residue
        mask = prot.heavy_resids == resid
        res_heavy_indices = np.where(mask)[0]
        res_atom_names = prot.heavy_atom_names[res_heavy_indices]
        name_to_hidx = dict(zip(res_atom_names, res_heavy_indices))

        slot_mapping.append(
            [int(name_to_hidx.get(aname, -1)) for aname in expected]
        )

    prot._of3_slot_mapping = slot_mapping
    prot._of3_num_atoms_per_token = np.array(num_atoms_per_token, dtype=np.int32)


# ------------------------------------------------------------------
# Template computation helpers
# ------------------------------------------------------------------

def _compute_distogram(
    pseudo_beta: np.ndarray,  # (L, 3)
    pseudo_beta_mask: np.ndarray,  # (L,)
) -> np.ndarray:
    """Compute OF3-style distogram from pseudo-beta positions. Returns (1, L, L, 39)."""
    diff = pseudo_beta[:, None, :] - pseudo_beta[None, :, :]  # (L, L, 3)
    sq_dist = np.sum(diff ** 2, axis=-1, keepdims=True)       # (L, L, 1)

    dgram = ((sq_dist > _TEMPL_BIN_LOWER_SQ) & (sq_dist < _TEMPL_BIN_UPPER_SQ)).astype(
        np.float32
    )  # (L, L, 39)

    pair_mask = pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]  # (L, L)
    dgram *= pair_mask[:, :, None]

    return dgram[None]  # (1, L, L, 39)


def _compute_unit_vectors(
    backbone_coords: np.ndarray,  # (L, 3, 3) — N, CA, C per residue
    backbone_frame_mask: np.ndarray,  # (L,)
    L: int,
) -> np.ndarray:
    """Compute unit vectors between CA atoms. Returns (1, L, L, 3)."""
    ca = backbone_coords[:, 1, :]   # (L, 3)
    diff = ca[:, None, :] - ca[None, :, :]   # (L, L, 3)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8
    unit_vec = diff / norm   # (L, L, 3)

    pair_mask = backbone_frame_mask[:, None] * backbone_frame_mask[None, :]
    unit_vec *= pair_mask[:, :, None]

    return unit_vec[None].astype(np.float32)  # (1, L, L, 3)


def _zero_template_features(L: int, n_templ: int = 1) -> dict:
    """Return all-zero template features for folding mode / ablation."""
    return {
        "template_restype": torch.zeros((n_templ, L, 32), dtype=torch.int32),
        "template_pseudo_beta_mask": torch.zeros((n_templ, L), dtype=torch.float32),
        "template_backbone_frame_mask": torch.zeros((n_templ, L), dtype=torch.float32),
        "template_distogram": torch.zeros((n_templ, L, L, 39), dtype=torch.float32),
        "template_unit_vector": torch.zeros((n_templ, L, L, 3), dtype=torch.float32),
    }


# ------------------------------------------------------------------
# MSA loader
# ------------------------------------------------------------------

def _load_msa_from_a3m(
    a3m_path: Path,
    L: int,
    n_restypes: int,
) -> dict:
    """Load a precomputed MSA from an .a3m file.

    Falls back to single-sequence MSA on any parse error.
    """
    try:
        from openfold3.core.data.tools.parse_msa_files import parse_a3m
        msa_seqs, deletion_matrix = parse_a3m(str(a3m_path))
    except Exception:
        # Graceful fallback to single-sequence
        return _single_seq_msa(L, n_restypes)

    # Encode sequences to one-hot (N_msa, L, n_restypes)
    from openfold3.core.data.resources.residues import STANDARD_RESIDUES_WITH_GAP_3
    res3to1 = {r: i for i, r in enumerate(STANDARD_RESIDUES_WITH_GAP_3)}
    unk_idx = res3to1.get("UNK", n_restypes - 1)

    N_msa = len(msa_seqs)
    msa_idx = torch.zeros((N_msa, L), dtype=torch.int64)
    for i, seq in enumerate(msa_seqs[:L]):
        for j, aa in enumerate(seq[:L]):
            msa_idx[i, j] = res3to1.get(aa, unk_idx)

    msa = F.one_hot(msa_idx, num_classes=n_restypes).int()
    deletion = torch.tensor(deletion_matrix, dtype=torch.float32)[:N_msa, :L]
    has_del = (deletion > 0).float()
    del_val = torch.log(deletion.clamp(min=0) + 1.0) / np.log(3.0)

    return {
        "msa": msa,
        "has_deletion": has_del,
        "deletion_value": del_val,
        "profile": msa.float().mean(0),
        "deletion_mean": del_val.mean(0),
        "msa_mask": torch.ones((N_msa, L), dtype=torch.float32),
        "num_paired_seqs": torch.tensor(0, dtype=torch.int64),
    }


def _single_seq_msa(L: int, n_restypes: int) -> dict:
    """Dummy single-sequence MSA (identity)."""
    return {
        "msa": torch.zeros((1, L, n_restypes), dtype=torch.int32),
        "has_deletion": torch.zeros((1, L), dtype=torch.float32),
        "deletion_value": torch.zeros((1, L), dtype=torch.float32),
        "profile": torch.zeros((L, n_restypes), dtype=torch.float32),
        "deletion_mean": torch.zeros(L, dtype=torch.float32),
        "msa_mask": torch.ones((1, L), dtype=torch.float32),
        "num_paired_seqs": torch.tensor(0, dtype=torch.int64),
    }
