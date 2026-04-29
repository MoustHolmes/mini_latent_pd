"""Frozen MiniFold decoder: distance bin indices → atom37 coordinates.

Wraps the pretrained MiniFold components needed to decode a distogram into
3D structure. All weights are loaded from a MiniFold checkpoint and frozen.

Pipeline:
    1. Compute s_z_static from precomputed ESM embeddings (s_s, s_z)
    2. bin_indices → one_hot(64) → recycle → + s_z_static → z_pre
    3. z_post = MiniFormer(z_pre, pair_mask)
    4. single = PairToSequence(z_post, s_s, pair_mask)
    5. sm_out = StructureModule(single, z_post, aatype, seq_mask) → atom14
    6. atom37 = atom14_to_atom37(atom14)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_latent_pd.models.minifold.model.model import (
    SequenceToPair,
    RelativePosition,
    PairToSequence,
)
from mini_latent_pd.models.minifold.model.miniformer import MiniFormer
from mini_latent_pd.models.minifold.model.structure import StructureModule
from mini_latent_pd.models.minifold.data.data_transforms import make_atom14_masks
from mini_latent_pd.models.minifold.utils.feats import atom14_to_atom37


class MiniFoldDecoder(nn.Module):
    """Frozen MiniFold decode pipeline: bin indices → atom37 coordinates.

    Args:
        c_s: Sequence representation dimension (1024).
        c_z: Pair representation dimension (128).
        rel_pos_bins: Number of relative position bins (32).
        disto_bins: Number of distance histogram bins (64).
        num_miniformer_layers: Number of MiniFormer transformer blocks.
        num_structure_blocks: Number of StructureModule IPA blocks.
        kernels: Whether to use fused kernels in MiniFormer (inference only).
    """

    def __init__(
        self,
        c_s: int = 1024,
        c_z: int = 128,
        rel_pos_bins: int = 32,
        disto_bins: int = 64,
        num_miniformer_layers: int = 12,
        num_structure_blocks: int = 8,
        kernels: bool = False,
    ):
        super().__init__()
        self.c_z = c_z
        self.disto_bins = disto_bins

        # Static pair feature computation
        self.seq_to_pair = SequenceToPair(c_s, c_z // 2, c_z)
        self.rel_pos = RelativePosition(rel_pos_bins, c_z)
        self.proj = nn.Linear(c_z * 3, c_z)

        # Recycle: one-hot distogram → pair repr
        self.recycle = nn.Linear(disto_bins, c_z)

        # MiniFormer (critical refinement step)
        self.miniformer = MiniFormer(c_z, blocks=num_miniformer_layers, kernels=kernels)

        # Decode: pair → single → structure
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

        # Match FoldingTrunk init
        nn.init.zeros_(self.seq_to_pair.o_proj.weight)
        nn.init.zeros_(self.seq_to_pair.o_proj.bias)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    _CKPT_KEY_MAP = {
        "seq_to_pair.": "model.fold.seq_to_pair.",
        "rel_pos.": "model.fold.positional_embedding.",
        "proj.": "model.fold.projection.",
        "recycle.": "model.fold.recycle.",
        "miniformer.": "model.fold.miniformer.",
        "pair_to_seq.": "model.sz_project.",
        "structure_module.": "model.structure_module.",
    }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "MiniFoldDecoder":
        """Load all decoder weights from a MiniFold checkpoint; freeze them.

        Args:
            checkpoint_path: Path to MiniFold ``.ckpt`` file.
            device: Target device.
            **kwargs: Forwarded to ``__init__``.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        full_sd = ckpt["state_dict"]
        full_sd = {
            k.replace("_orig_mod.", ""): v
            for k, v in full_sd.items()
            if "boundaries" not in k and "mid_points" not in k
        }

        decoder = cls(**kwargs)
        dec_sd = decoder.state_dict()

        loaded, skipped = 0, 0
        for dec_key in dec_sd:
            matched = False
            for dec_prefix, ckpt_prefix in cls._CKPT_KEY_MAP.items():
                if dec_key.startswith(dec_prefix):
                    suffix = dec_key[len(dec_prefix) :]
                    ckpt_key = ckpt_prefix + suffix
                    if ckpt_key in full_sd:
                        dec_sd[dec_key] = full_sd[ckpt_key]
                        loaded += 1
                        matched = True
                    break
            if not matched:
                skipped += 1

        decoder.load_state_dict(dec_sd)
        decoder.to(device)
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad = False

        print(
            f"MiniFoldDecoder: loaded {loaded} params, "
            f"{skipped} buffer/init-only. All weights frozen."
        )
        return decoder

    # ------------------------------------------------------------------
    # Static feature computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_static_features(
        self,
        s_s: torch.Tensor,
        s_z: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute static pair repr from precomputed ESM embeddings.

        Args:
            s_s: (B, L, c_s) sequence representation.
            s_z: (B, L, L, c_z) pair representation from ESM attentions.
            seq_mask: (B, L) mask.

        Returns:
            s_z_static: (B, L, L, c_z)
        """
        pair_mask = seq_mask[:, None, :] * seq_mask[:, :, None]
        B, L = s_s.shape[:2]
        residx = torch.arange(L, device=s_s.device).unsqueeze(0).expand(B, -1)

        s_z_cat = torch.cat(
            [
                s_z,
                self.seq_to_pair(s_s),
                self.rel_pos(residx, mask=pair_mask),
            ],
            dim=-1,
        )
        return self.proj(s_z_cat)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        bin_indices: torch.Tensor,
        s_s: torch.Tensor,
        s_z_static: torch.Tensor,
        aatype: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode distance bin indices into atom37 coordinates.

        Args:
            bin_indices: (B, L, L) long, values in [0, 63].
            s_s: (B, L, c_s) sequence representation.
            s_z_static: (B, L, L, c_z) precomputed static pair features.
            aatype: (B, L) residue type indices.
            seq_mask: (B, L) binary mask.

        Returns:
            dict with ``atom37_pos``, ``atom14_pos``, ``atom37_mask``.
        """
        pair_mask = seq_mask[:, None, :] * seq_mask[:, :, None]

        # bin indices → one-hot → recycle → + s_z_static
        disto = F.one_hot(bin_indices.long(), self.disto_bins).to(s_z_static.dtype)
        z = s_z_static + self.recycle(disto)

        # MiniFormer refinement
        z = self.miniformer(z, pair_mask)

        # PairToSequence → StructureModule
        single = self.pair_to_seq(z, s_s, pair_mask)
        sm_out = self.structure_module(
            s=single,
            z=z,
            aatype=aatype,
            mask=seq_mask,
        )

        # atom14 → atom37
        pred_atom14 = sm_out["positions"][-1]
        feats = _make_atom14_feats(aatype)
        pred_atom37 = atom14_to_atom37(pred_atom14, feats)

        return {
            "atom37_pos": pred_atom37,
            "atom14_pos": pred_atom14,
            "atom37_mask": feats["atom37_atom_exists"],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atom14_feats(aatype: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build atom14/atom37 index mappings from aatype (batched)."""
    if aatype.dim() == 1:
        return make_atom14_masks({"aatype": aatype})
    feats_list = [make_atom14_masks({"aatype": aa}) for aa in aatype]
    return {k: torch.stack([f[k] for f in feats_list]) for k in feats_list[0].keys()}
