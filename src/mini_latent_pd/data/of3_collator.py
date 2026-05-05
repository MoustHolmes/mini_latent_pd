"""Collator for OpenFold3 format batches.

Pads variable-length sequences and flat atom arrays to the maximum in the batch.
Handles the nested `ground_truth` and `loss_weights` sub-dicts.
"""

import torch


# Token-level keys that need padding along dim 0 to max sequence length.
_TOKEN_PAD_KEYS = {
    "residue_index", "token_index", "asym_id", "entity_id", "sym_id",
    "restype", "is_protein", "is_rna", "is_dna", "is_ligand", "is_atomized",
    "token_mask", "start_atom_index", "num_atoms_per_token",
    "profile", "deletion_mean",
    # Permutation-alignment fields (top-level)
    "mol_entity_id", "mol_sym_id", "mol_sym_component_id", "mol_sym_token_index",
}

# Token x Token keys (pad both dims 0 and 1).
_TOKEN_TOKEN_PAD_KEYS = {"token_bonds"}

# Atom-level keys (pad dim 0 to max n_atom).
_ATOM_PAD_KEYS = {
    "ref_pos", "ref_mask", "ref_element", "ref_charge",
    "ref_atom_name_chars", "ref_space_uid", "atom_mask", "atom_to_token_index",
}

# MSA keys (pad dim 1 = token dim).
_MSA_PAD_KEYS = {"msa", "has_deletion", "deletion_value", "msa_mask"}

# Template keys: (n_templ, L, ...) — pad dim 1 to max L; (n_templ, L, L, ...) pad both.
_TEMPL_TOKEN_KEYS = {
    "template_restype", "template_pseudo_beta_mask", "template_backbone_frame_mask"
}
_TEMPL_TOKEN_TOKEN_KEYS = {"template_distogram", "template_unit_vector"}

# Ground truth keys that are atom-level.
_GT_ATOM_KEYS = {"atom_positions", "atom_resolved_mask", "atom_mask"}
# Ground truth keys that are token-level.
_GT_TOKEN_KEYS = {
    "token_mask", "token_index", "num_atoms_per_token", "is_ligand",
    "mol_entity_id", "mol_sym_id", "mol_sym_component_id", "mol_sym_token_index",
}


def of3_collate(batch: list[dict]) -> dict:
    """Collate a list of OF3 samples into a padded batch dict.

    Also adds ``seq_mask`` (B, L_max) float32 and ``atom_mask`` is already
    included per sample as a flat atom mask.
    """
    L_max = max(_seq_len(s) for s in batch)
    A_max = max(_n_atoms(s) for s in batch)

    out: dict = {}
    keys = batch[0].keys()

    for key in keys:
        if key == "ground_truth":
            out["ground_truth"] = _collate_ground_truth(batch, A_max, L_max)
        elif key == "loss_weights":
            out["loss_weights"] = _collate_loss_weights(batch)
        elif key in _TOKEN_PAD_KEYS:
            out[key] = _pad_and_stack([s[key] for s in batch], L_max, dim=0)
        elif key in _TOKEN_TOKEN_PAD_KEYS:
            out[key] = _pad_and_stack_2d([s[key] for s in batch], L_max)
        elif key in _ATOM_PAD_KEYS:
            out[key] = _pad_and_stack([s[key] for s in batch], A_max, dim=0)
        elif key in _MSA_PAD_KEYS:
            # Shape: (N_msa, L, ...) — pad dim 1
            out[key] = _pad_and_stack([s[key] for s in batch], L_max, dim=1)
        elif key in _TEMPL_TOKEN_KEYS:
            # Shape: (n_templ, L, ...) — pad dim 1
            out[key] = _pad_and_stack([s[key] for s in batch], L_max, dim=1)
        elif key in _TEMPL_TOKEN_TOKEN_KEYS:
            # Shape: (n_templ, L, L, ...) — pad dims 1 and 2
            out[key] = _pad_and_stack_templ_ll([s[key] for s in batch], L_max)
        elif isinstance(batch[0].get(key), torch.Tensor):
            # Scalar tensors or anything else: try plain stack
            out[key] = torch.stack([s[key] for s in batch])
        else:
            out[key] = [s[key] for s in batch]

    # Sequence-level padding mask
    seq_lens = torch.tensor([_seq_len(s) for s in batch], dtype=torch.long)
    out["seq_mask"] = (
        torch.arange(L_max).unsqueeze(0) < seq_lens.unsqueeze(1)
    ).float()

    return out


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _seq_len(sample: dict) -> int:
    return sample["token_mask"].shape[0]


def _n_atoms(sample: dict) -> int:
    return int(sample["num_atoms_per_token"].sum().item())


def _pad_and_stack(tensors: list[torch.Tensor], target: int, dim: int) -> torch.Tensor:
    padded = []
    for t in tensors:
        pad_amount = target - t.shape[dim]
        if pad_amount > 0:
            pad_spec = [0] * (2 * t.ndim)
            pad_spec[2 * (t.ndim - 1 - dim) + 1] = pad_amount
            t = torch.nn.functional.pad(t, pad_spec, value=0)
        padded.append(t)
    return torch.stack(padded)


def _pad_and_stack_2d(tensors: list[torch.Tensor], target: int) -> torch.Tensor:
    """Pad both dim 0 and dim 1 (token × token tensors)."""
    padded = []
    for t in tensors:
        d0, d1 = t.shape[0], t.shape[1]
        t = torch.nn.functional.pad(
            t, (0, 0, 0, target - d1, 0, target - d0), value=0
        )
        padded.append(t)
    return torch.stack(padded)


def _pad_and_stack_templ_ll(tensors: list[torch.Tensor], target: int) -> torch.Tensor:
    """Pad dims 1 and 2 for (n_templ, L, L, ...) tensors."""
    padded = []
    for t in tensors:
        # t: (n_templ, L, L, ...) — pad dims 1 and 2
        # torch.nn.functional.pad processes from the last dimension
        n_extra = t.ndim - 3  # trailing dims after L×L
        l1, l2 = t.shape[1], t.shape[2]
        padding = (0,) * (2 * n_extra) + (0, target - l2, 0, target - l1)
        t = torch.nn.functional.pad(t, padding, value=0)
        padded.append(t)
    return torch.stack(padded)


def _collate_ground_truth(batch: list[dict], A_max: int, L_max: int) -> dict:
    gt_out: dict = {}
    gt_keys = batch[0]["ground_truth"].keys()
    for key in gt_keys:
        tensors = [s["ground_truth"][key] for s in batch]
        if key in _GT_ATOM_KEYS:
            gt_out[key] = _pad_and_stack(tensors, A_max, dim=0)
        elif key in _GT_TOKEN_KEYS:
            gt_out[key] = _pad_and_stack(tensors, L_max, dim=0)
        else:
            gt_out[key] = torch.stack(tensors)
    return gt_out


def _collate_loss_weights(batch: list[dict]) -> dict:
    keys = batch[0]["loss_weights"].keys()
    return {k: torch.stack([s["loss_weights"][k] for s in batch]) for k in keys}
