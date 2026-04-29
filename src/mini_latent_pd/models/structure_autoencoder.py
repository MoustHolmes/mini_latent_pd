"""Structure Autoencoder: encode atom37 coords → latent z, decode z → atom37.

Wraps MiniFold's FoldingTrunk encoder components and StructureModule decoder
into a single module that can be initialised from a pretrained checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_latent_pd.models.minifold.model.model import (
    SequenceToPair,
    RelativePosition,
    PairToSequence,
)
from mini_latent_pd.models.minifold.model.structure import StructureModule
from mini_latent_pd.models.minifold.data.data_transforms import make_atom14_masks
from mini_latent_pd.models.minifold.utils.feats import atom14_to_atom37
from mini_latent_pd.utils.geometry import coords_to_distogram


class StructureAutoencoder(nn.Module):
    """Encode all-atom coordinates into a pair representation and decode back.

    **Encode** ``(atom37, s_s, s_z, aatype, seq_mask) → z``
        1. ``s_z_static = proj(cat([s_z, seq_to_pair(s_s), rel_pos(residx, pair_mask)]))``
        2. ``disto = coords_to_distogram(atom37, aatype)``  — one-hot (B, L, L, 64)
        3. ``z = s_z_static + recycle(disto)``               — (B, L, L, c_z)

    **Decode** ``(z, s_s, aatype, seq_mask) → atom37_positions``
        1. ``s = pair_to_seq(z, s_s, pair_mask)``
        2. ``sm_out = structure_module(s, z, aatype, seq_mask)``  → atom14
        3. ``atom37 = atom14_to_atom37(atom14, feats)``

    All sub-modules match the architecture / hyperparameters of MiniFold so
    that weights can be loaded directly from a pretrained checkpoint.
    """

    def __init__(
        self,
        c_s: int = 1024,
        c_z: int = 128,
        rel_pos_bins: int = 32,
        disto_bins: int = 64,
        num_structure_blocks: int = 8,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.disto_bins = disto_bins

        # --- Encoder components (from FoldingTrunk) ---
        self.seq_to_pair = SequenceToPair(c_s, c_z // 2, c_z)
        self.rel_pos = RelativePosition(rel_pos_bins, c_z)
        self.proj = nn.Linear(c_z * 3, c_z)
        self.recycle = nn.Linear(disto_bins, c_z)

        # --- Decoder components ---
        self.pair_to_seq = PairToSequence(c_z=c_z, c_s=c_s)
        self.structure_module = StructureModule(
            c_s=c_s,
            c_z=c_z,
            c_resnet=128,
            head_dim=64,
            no_heads=16,
            no_blocks=num_structure_blocks,
            no_resnet_blocks=2,
            no_angles=7,
            trans_scale_factor=10,
            epsilon=1e-5,
            inf=1e5,
        )

        # Zero-init the seq_to_pair output projection (same as FoldingTrunk)
        nn.init.zeros_(self.seq_to_pair.o_proj.weight)
        nn.init.zeros_(self.seq_to_pair.o_proj.bias)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    # Mapping from AE parameter names → MiniFold checkpoint key prefixes.
    # Checkpoint keys have the form  "model.<prefix>.<param_name>".
    _CKPT_KEY_MAP = {
        "seq_to_pair.": "model.fold.seq_to_pair.",
        "rel_pos.":     "model.fold.positional_embedding.",
        "proj.":        "model.fold.projection.",
        "recycle.":     "model.fold.recycle.",
        "pair_to_seq.": "model.sz_project.",
        "structure_module.": "model.structure_module.",
    }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "StructureAutoencoder":
        """Load autoencoder weights directly from a MiniFold ``.ckpt`` file.

        This avoids instantiating the full MiniFold model (and ESM) — only
        the small encoder/decoder components are created and loaded.

        Args:
            checkpoint_path: Path to the MiniFold ``.ckpt`` file.
            device: Target device for the model.
            **kwargs: Forwarded to ``__init__`` (e.g. ``num_structure_blocks``).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        full_sd = ckpt["state_dict"]
        # Clean keys (same normalization as the training script)
        full_sd = {
            k.replace("_orig_mod.", ""): v
            for k, v in full_sd.items()
            if "boundaries" not in k and "mid_points" not in k
        }

        ae = cls(**kwargs)
        ae_sd = ae.state_dict()

        loaded, skipped = 0, 0
        for ae_key in ae_sd:
            matched = False
            for ae_prefix, ckpt_prefix in cls._CKPT_KEY_MAP.items():
                if ae_key.startswith(ae_prefix):
                    suffix = ae_key[len(ae_prefix):]
                    ckpt_key = ckpt_prefix + suffix
                    if ckpt_key in full_sd:
                        ae_sd[ae_key] = full_sd[ckpt_key]
                        loaded += 1
                        matched = True
                    break
            if not matched:
                skipped += 1

        ae.load_state_dict(ae_sd)
        ae.to(device)
        print(
            f"StructureAutoencoder: loaded {loaded} params from checkpoint"
            f" ({skipped} buffer/init-only)"
        )
        return ae

    @classmethod
    def from_minifold(cls, minifold_model: nn.Module, **kwargs) -> "StructureAutoencoder":
        """Construct and load weights from an already-instantiated MiniFoldModel."""
        ae = cls(**kwargs)
        fold = minifold_model.fold

        # Encoder
        ae.seq_to_pair.load_state_dict(fold.seq_to_pair.state_dict())
        ae.rel_pos.load_state_dict(fold.positional_embedding.state_dict())
        ae.proj.load_state_dict(fold.projection.state_dict())
        ae.recycle.load_state_dict(fold.recycle.state_dict())

        # Decoder
        ae.pair_to_seq.load_state_dict(minifold_model.sz_project.state_dict())
        ae.structure_module.load_state_dict(
            minifold_model.structure_module.state_dict()
        )
        return ae

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        atom37_pos: torch.Tensor,
        s_s: torch.Tensor,
        s_z: torch.Tensor,
        aatype: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode atom37 coordinates into the latent pair representation.

        Args:
            atom37_pos: (B, L, 37, 3) all-atom coordinates in Angstroms.
            s_s: (B, L, c_s) pre-computed single (sequence) representation.
            s_z: (B, L, L, c_z) pre-computed pair representation.
            aatype: (B, L) residue type indices (long).
            seq_mask: (B, L) binary mask for valid residues.

        Returns:
            z: (B, L, L, c_z) latent pair representation.
        """
        pair_mask = seq_mask[:, None, :] * seq_mask[:, :, None]
        B, L = s_s.shape[:2]
        residx = torch.arange(L, device=s_s.device).unsqueeze(0).expand(B, -1)

        # Static pair features
        s_z_cat = torch.cat(
            [
                s_z,
                self.seq_to_pair(s_s),
                self.rel_pos(residx, mask=pair_mask),
            ],
            dim=-1,
        )
        s_z_static = self.proj(s_z_cat)

        # Structure-informed distogram
        disto = coords_to_distogram(atom37_pos, aatype)  # (B, L, L, disto_bins)

        z = s_z_static + self.recycle(disto)
        return z

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        z: torch.Tensor,
        s_s: torch.Tensor,
        aatype: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode the latent pair representation into atom37 coordinates.

        Args:
            z: (B, L, L, c_z) latent pair representation.
            s_s: (B, L, c_s) single representation.
            aatype: (B, L) residue type indices (long).
            seq_mask: (B, L) binary mask.

        Returns:
            Dictionary with keys:
                ``atom37_pos``: (B, L, 37, 3) predicted coordinates.
                ``atom14_pos``: (B, L, 14, 3) atom14 coordinates.
                ``atom37_mask``: (B, L, 37) atom existence mask.
                ``sm_out``: raw StructureModule output dict.
        """
        pair_mask = seq_mask[:, None, :] * seq_mask[:, :, None]

        # Project pair → single repr
        single = self.pair_to_seq(z, s_s, pair_mask)

        # Run StructureModule
        sm_out = self.structure_module(
            s=single,
            z=z,
            aatype=aatype,
            mask=seq_mask,
        )

        # atom14 → atom37
        pred_atom14 = sm_out["positions"][-1]  # last block
        feats = _make_atom14_feats(aatype)
        pred_atom37 = atom14_to_atom37(pred_atom14, feats)
        atom37_mask = feats["atom37_atom_exists"]

        return {
            "atom37_pos": pred_atom37,
            "atom14_pos": pred_atom14,
            "atom37_mask": atom37_mask,
            "sm_out": sm_out,
        }

    # ------------------------------------------------------------------
    # Convenience: full round-trip
    # ------------------------------------------------------------------

    def forward(
        self,
        atom37_pos: torch.Tensor,
        s_s: torch.Tensor,
        s_z: torch.Tensor,
        aatype: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full encode → decode round-trip.

        Returns the decode dict plus the latent ``z``.
        """
        z = self.encode(atom37_pos, s_s, s_z, aatype, seq_mask)
        out = self.decode(z, s_s, aatype, seq_mask)
        out["z"] = z
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atom14_feats(aatype: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build atom14/atom37 index mappings from aatype.

    Works for batched input (B, L) — applies ``make_atom14_masks`` per-sample
    and re-stacks.

    Returns:
        dict with keys ``aatype``, ``atom14_atom_exists``,
        ``residx_atom14_to_atom37``, ``residx_atom37_to_atom14``,
        ``atom37_atom_exists``.
    """
    if aatype.dim() == 1:
        return make_atom14_masks({"aatype": aatype})

    feats_list = [make_atom14_masks({"aatype": aa}) for aa in aatype]
    return {
        k: torch.stack([f[k] for f in feats_list])
        for k in feats_list[0].keys()
    }
