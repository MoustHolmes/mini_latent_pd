"""PDBOpenFold3Dataset: PDB / mmCIF structures in OpenFold3 batch format.

Loads standard PDB or CIF files using biotite and converts them to the
dict-of-tensors format expected by OpenFold3. Used for folding training
(no template conditioning) and for smoke-testing the pipeline without
requiring full mdCATH trajectories.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openfold3.core.data.resources.residues import (
    STANDARD_PROTEIN_RESIDUES_3,
    STANDARD_RESIDUES_WITH_GAP_3,
    get_with_unknown_3_to_idx,
)
from openfold3.core.data.resources.token_atom_constants import TOKEN_NAME_TO_ATOM_NAMES
from mini_latent_pd.data.mdcath_of3_dataset import (
    _DEFAULT_LOSS_WEIGHTS,
    _N_RESTYPES,
    _single_seq_msa,
    _zero_template_features,
)

# Element symbol → atomic number (0-indexed for OF3 ref_element)
_ELEM_TO_Z: dict[str, int] = {
    "H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "SE": 34, "P": 15,
}

# Standard 3-letter → 1-letter for sanity checks
_3TO1 = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWVY",
))


class PDBOpenFold3Dataset(Dataset):
    """Dataset that loads PDB/CIF files and produces OpenFold3 batch dicts.

    Each file is treated as a single data point (one conformation).
    Template conditioning is off by default (folding mode).

    Args:
        pdb_paths: List of .pdb or .cif file paths.
        msa_dir: Directory of precomputed `.a3m` files (keyed by stem of
            the PDB file). If None, uses single-sequence MSA.
        max_seq_len: Skip structures longer than this.
    """

    def __init__(
        self,
        pdb_paths: list[str | Path],
        msa_dir: str | Path | None = None,
        max_seq_len: int = 900,
    ):
        self.msa_dir = Path(msa_dir) if msa_dir is not None else None
        self.samples: list[dict] = []

        for path in pdb_paths:
            record = _load_pdb_record(Path(path), max_seq_len)
            if record is not None:
                self.samples.append(record)
                print(f"  Indexed {record['id']}: L={record['n_residues']}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        record = self.samples[idx]
        return _pdb_record_to_of3_batch(record, self.msa_dir)


# ------------------------------------------------------------------
# PDB loading
# ------------------------------------------------------------------

def _load_pdb_record(path: Path, max_seq_len: int) -> dict | None:
    """Load a PDB/CIF file and return a protein record dict."""
    import biotite.structure as struc
    import biotite.structure.io as strucio

    try:
        atom_array = strucio.load_structure(str(path))
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None

    # If a stack is returned (multiple models), take the first model
    if isinstance(atom_array, struc.AtomArrayStack):
        atom_array = atom_array[0]

    # Keep only protein ATOM records for standard residues
    is_protein = struc.filter_amino_acids(atom_array)
    atom_array = atom_array[is_protein]
    if len(atom_array) == 0:
        return None

    # Remove hydrogens
    heavy_mask = atom_array.element != "H"
    atom_array = atom_array[heavy_mask]

    # Get per-atom data
    atom_names = atom_array.atom_name       # (N,)
    res_names = atom_array.res_name         # (N,)
    res_ids = atom_array.res_id             # (N,)
    elements = atom_array.element           # (N,)
    coords = atom_array.coord.astype(np.float32)  # (N, 3)

    # Normalise MD residue names (handles HIE, CYX, etc. if present)
    from mini_latent_pd.data.mdcath_utils import MD_TO_STD_RESNAMES
    res_names_std = np.array([MD_TO_STD_RESNAMES.get(r, r) for r in res_names])

    # Per-residue arrays
    unique_resids, first_occ = np.unique(res_ids, return_index=True)
    order = np.argsort(first_occ)
    unique_resids = unique_resids[order]
    n_res = len(unique_resids)

    if n_res > max_seq_len or n_res == 0:
        return None

    resnames_per_residue = np.array(
        [res_names_std[np.where(res_ids == rid)[0][0]] for rid in unique_resids]
    )

    # Build OF3 atom-slot mapping
    slot_mapping: list[list[int]] = []
    num_atoms_per_token: list[int] = []

    for resname, resid in zip(resnames_per_residue, unique_resids):
        expected = TOKEN_NAME_TO_ATOM_NAMES.get(resname, TOKEN_NAME_TO_ATOM_NAMES.get("UNK", []))
        num_atoms_per_token.append(len(expected))

        mask = res_ids == resid
        res_indices = np.where(mask)[0]
        res_anames = atom_names[res_indices]
        name_to_idx = dict(zip(res_anames, res_indices))

        slot_mapping.append([int(name_to_idx.get(a, -1)) for a in expected])

    # Element → atomic number (0-indexed)
    z_per_heavy = np.array([
        _ELEM_TO_Z.get(e.upper(), 6) - 1 for e in elements  # default to C if unknown
    ], dtype=np.int64)

    sequence = "".join(_3TO1.get(r, "X") for r in resnames_per_residue)

    return {
        "id": path.stem,
        "n_residues": n_res,
        "sequence": sequence,
        "resnames_std": resnames_per_residue,
        "resids": unique_resids,
        "coords": coords,              # (N_heavy, 3)
        "atom_names": atom_names,      # (N_heavy,)
        "z_per_heavy": z_per_heavy,    # (N_heavy,)
        "slot_mapping": slot_mapping,
        "num_atoms_per_token": np.array(num_atoms_per_token, dtype=np.int32),
    }


# ------------------------------------------------------------------
# Featurization
# ------------------------------------------------------------------

def _pdb_record_to_of3_batch(record: dict, msa_dir: Path | None) -> dict:
    """Convert a loaded PDB record to an OF3 batch dict."""
    L = record["n_residues"]
    coords = record["coords"]
    slot_mapping = record["slot_mapping"]
    num_atoms_per_token = record["num_atoms_per_token"]
    n_atom = int(sum(num_atoms_per_token))

    ref_pos = np.zeros((n_atom, 3), dtype=np.float32)
    ref_mask = np.zeros(n_atom, dtype=np.int32)
    ref_element = np.zeros(n_atom, dtype=np.int64)
    ref_charge = np.zeros(n_atom, dtype=np.float32)
    ref_atom_name_chars = np.zeros((n_atom, 4), dtype=np.int64)
    atom_to_token = np.zeros(n_atom, dtype=np.int32)

    flat_idx = 0
    for res_idx, (resname, slots) in enumerate(
        zip(record["resnames_std"], slot_mapping)
    ):
        expected = TOKEN_NAME_TO_ATOM_NAMES.get(resname, TOKEN_NAME_TO_ATOM_NAMES.get("UNK", []))
        for aname, heavy_idx in zip(expected, slots):
            atom_to_token[flat_idx] = res_idx
            if heavy_idx >= 0:
                ref_pos[flat_idx] = coords[heavy_idx]
                ref_mask[flat_idx] = 1
                ref_element[flat_idx] = max(int(record["z_per_heavy"][heavy_idx]), 0)
            padded = aname.ljust(4)[:4]
            ref_atom_name_chars[flat_idx] = [ord(c) - 32 for c in padded]
            flat_idx += 1

    start_atom_index = np.concatenate(
        [np.array([0]), np.cumsum(num_atoms_per_token)[:-1]]
    )

    restype_idx = np.array(
        [get_with_unknown_3_to_idx(r) for r in record["resnames_std"]], dtype=np.int64
    )
    restype_oh = F.one_hot(torch.from_numpy(restype_idx), num_classes=_N_RESTYPES).int()
    is_protein = torch.tensor(
        [1 if r in STANDARD_PROTEIN_RESIDUES_3 else 0 for r in record["resnames_std"]],
        dtype=torch.int32,
    )

    # MSA
    if msa_dir is not None:
        msa_path = msa_dir / f"{record['id']}.a3m"
        if msa_path.exists():
            from mini_latent_pd.data.mdcath_of3_dataset import _load_msa_from_a3m
            msa_feats = _load_msa_from_a3m(msa_path, L, _N_RESTYPES)
        else:
            msa_feats = _single_seq_msa(L, _N_RESTYPES)
    else:
        msa_feats = _single_seq_msa(L, _N_RESTYPES)

    features = {
        "residue_index": torch.arange(L, dtype=torch.int32),
        "token_index": torch.arange(L, dtype=torch.int32),
        "asym_id": torch.ones(L, dtype=torch.int32),
        "entity_id": torch.ones(L, dtype=torch.int32),
        "sym_id": torch.ones(L, dtype=torch.int32),
        "restype": restype_oh,
        "is_protein": is_protein,
        "is_rna": torch.zeros(L, dtype=torch.int32),
        "is_dna": torch.zeros(L, dtype=torch.int32),
        "is_ligand": torch.zeros(L, dtype=torch.int32),
        "is_atomized": torch.zeros(L, dtype=torch.int32),
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
        "token_bonds": torch.zeros((L, L), dtype=torch.int32),
        "token_mask": torch.ones(L, dtype=torch.float32),
        "atom_mask": torch.from_numpy(ref_mask).float(),
        "start_atom_index": torch.from_numpy(start_atom_index).int(),
        "num_atoms_per_token": torch.from_numpy(num_atoms_per_token).int(),
        "atom_to_token_index": torch.from_numpy(atom_to_token),
        # Permutation-alignment fields at top level (single-chain: trivial values)
        "mol_entity_id": torch.ones(L, dtype=torch.int32),
        "mol_sym_id": torch.ones(L, dtype=torch.int32),
        "mol_sym_component_id": torch.ones(L, dtype=torch.int32),
        "mol_sym_token_index": torch.arange(L, dtype=torch.int32),
        "ground_truth": {
            "atom_positions": torch.from_numpy(ref_pos.copy()),
            "atom_resolved_mask": torch.from_numpy(ref_mask).float(),
            "token_mask": torch.ones(L, dtype=torch.float32),
            "token_index": torch.arange(L, dtype=torch.int32),
            "atom_mask": torch.from_numpy(ref_mask).float(),
            "num_atoms_per_token": torch.from_numpy(record["num_atoms_per_token"]).int(),
            "is_ligand": torch.zeros(L, dtype=torch.int32),
            "mol_entity_id": torch.ones(L, dtype=torch.int32),
            "mol_sym_id": torch.ones(L, dtype=torch.int32),
            "mol_sym_component_id": torch.ones(L, dtype=torch.int32),
            "mol_sym_token_index": torch.arange(L, dtype=torch.int32),
        },
        "loss_weights": {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in _DEFAULT_LOSS_WEIGHTS.items()
        },
        **msa_feats,
        **_zero_template_features(L, n_templ=1),
    }

    return features
