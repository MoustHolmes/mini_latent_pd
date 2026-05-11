"""Utilities for converting mdCATH HDF5 data to Atom37 / OF3 format."""

import numpy as np


def _rc():
    """Lazy import of residue_constants (only needed for atom37 functions)."""
    from mini_latent_pd.models.minifold.utils import residue_constants
    return residue_constants


# MD force-field residue names that differ from standard PDB names
MD_TO_STD_RESNAMES = {
    "HIE": "HIS",
    "HID": "HIS",
    "HIP": "HIS",
    "CYX": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
}

MDCATH_TEMPS = ["320", "348", "379", "413", "450"]
MDCATH_REPLICAS = ["0", "1", "2", "3", "4"]


def decode_bytes_array(arr):
    """Decode a numpy array of byte strings to a numpy array of Python strings."""
    return np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in arr])


def parse_atom_names_from_pdb(pdb_text: str) -> np.ndarray:
    """Extract atom names from PDB ATOM/HETATM lines in order."""
    atom_names = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_names.append(line[12:16].strip())
    return np.array(atom_names)


def compute_atom37_indices(heavy_atom_names, heavy_resids):
    """
    Compute scatter indices for converting flat heavy-atom arrays to atom37 format.

    Returns:
        valid_heavy_idx: indices into heavy-atom sub-array for recognized atom37 types
        res_scatter: residue index for each valid heavy atom
        atom37_scatter: atom37 slot for each valid heavy atom
    """
    rc = _rc()
    unique_resids, unique_indices = np.unique(heavy_resids, return_index=True)
    sorted_idx = np.argsort(unique_indices)
    unique_resids = unique_resids[sorted_idx]
    resid_to_seq_idx = {resid: i for i, resid in enumerate(unique_resids)}

    valid_heavy_idx = []
    res_scatter = []
    atom37_scatter = []

    for i, (aname, resid) in enumerate(zip(heavy_atom_names, heavy_resids)):
        if aname in rc.atom_order:
            valid_heavy_idx.append(i)
            res_scatter.append(resid_to_seq_idx[resid])
            atom37_scatter.append(rc.atom_order[aname])

    return np.array(valid_heavy_idx), np.array(res_scatter), np.array(atom37_scatter)


build_scatter_indices = compute_atom37_indices  # backwards-compat alias


def aatype_from_resnames(resnames, resids):
    """
    Compute aatype integer array from per-atom resnames and resids.

    Returns:
        aatype: (num_residues,) int64 array of residue type indices
        residue_index: (num_residues,) int64 array of original residue IDs
        unique_resids: ordered unique residue IDs
    """
    rc = _rc()
    unique_resids, unique_indices = np.unique(resids, return_index=True)
    sorted_idx = np.argsort(unique_indices)
    unique_resids = unique_resids[sorted_idx]

    num_res = len(unique_resids)
    aatype = np.zeros(num_res, dtype=np.int64)
    residue_index = np.zeros(num_res, dtype=np.int64)

    for seq_idx, resid in enumerate(unique_resids):
        first_atom_idx = np.where(resids == resid)[0][0]
        md_resname = resnames[first_atom_idx]
        std_resname = MD_TO_STD_RESNAMES.get(md_resname, md_resname)
        res_1_letter = rc.restype_3to1.get(std_resname, "X")
        aatype[seq_idx] = rc.restype_order.get(res_1_letter, 20)
        residue_index[seq_idx] = resid

    return aatype, residue_index, unique_resids


def aatype_to_sequence(aatype) -> str:
    """Convert an aatype integer array to a 1-letter amino acid sequence string."""
    rc = _rc()
    restypes = rc.restypes
    return "".join(restypes[a] if a < len(restypes) else "X" for a in aatype)


def flat_coords_to_atom37(
    coords_flat, heavy_mask, valid_heavy_idx, res_scatter, atom37_scatter, num_res
):
    """
    Convert flat per-atom coordinates to atom37 format using pre-computed indices.

    Args:
        coords_flat: (num_atoms, 3) all-atom coordinates for one frame
        heavy_mask: (num_atoms,) bool mask for heavy atoms (Z > 1)
        valid_heavy_idx: indices into heavy-atom sub-array for recognized atom37 types
        res_scatter: residue index for each valid heavy atom
        atom37_scatter: atom37 slot for each valid heavy atom
        num_res: number of residues

    Returns:
        atom37_coords: (num_res, 37, 3) float32
    """
    heavy_coords = coords_flat[heavy_mask]
    atom37_coords = np.zeros((num_res, 37, 3), dtype=np.float32)
    atom37_coords[res_scatter, atom37_scatter] = heavy_coords[valid_heavy_idx]
    return atom37_coords


def compute_atom37_mask(aatype):
    """
    Compute which atom37 slots exist for each residue based on its type.

    Args:
        aatype: (num_res,) int64 array

    Returns:
        atom37_mask: (num_res, 37) float32
    """
    rc = _rc()
    num_res = len(aatype)
    mask = np.zeros((num_res, 37), dtype=np.float32)
    for i, aa in enumerate(aatype):
        if aa < len(rc.restypes):
            resname_3 = rc.restype_1to3[rc.restypes[aa]]
            for atom_name in rc.residue_atoms[resname_3]:
                if atom_name in rc.atom_order:
                    mask[i, rc.atom_order[atom_name]] = 1.0
    return mask
