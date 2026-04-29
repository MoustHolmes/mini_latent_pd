"""Velocity network for predicting distance-bin changes between MD frames.

Operates in bin-index space: given current distance bin indices d_0 ∈ {0,...,63}^{L×L}
and static pair features, predicts Δd ∈ R^{L×L} (real-valued bin velocity).
"""

import torch
import torch.nn as nn


class PairResBlock(nn.Module):
    """Residual block on pair features (B, L, L, D) with axial mixing."""

    def __init__(self, d_model: int, expand: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, expand * d_model),
            nn.GELU(),
            nn.Linear(expand * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.row_proj = nn.Linear(d_model, d_model, bias=False)
        self.col_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, pair_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, L, D) pair features.
            pair_mask: (B, L, L) float mask (1=valid, 0=pad).
        """
        # Channel-wise MLP
        x = x + self.mlp(self.norm1(x))

        # Axial mixing via masked row/col mean pooling
        h = self.norm2(x)
        if pair_mask is not None:
            h = h * pair_mask.unsqueeze(-1)

        # Row average: aggregate over columns (dim=2)
        if pair_mask is not None:
            row_count = pair_mask.sum(dim=2, keepdim=True).unsqueeze(-1).clamp(min=1)
            row_agg = h.sum(dim=2, keepdim=True) / row_count
        else:
            row_agg = h.mean(dim=2, keepdim=True)
        x = x + self.row_proj(row_agg)

        # Column average: aggregate over rows (dim=1)
        if pair_mask is not None:
            col_count = pair_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            col_agg = h.sum(dim=1, keepdim=True) / col_count
        else:
            col_agg = h.mean(dim=1, keepdim=True)
        x = x + self.col_proj(col_agg)

        return x


class DistanceVelocityNet(nn.Module):
    """Predicts bin-index velocity Δd from current bin indices and static pair features.

    Architecture:
        1. Embed bin indices (B, L, L) → (B, L, L, d_model)
        2. Add projected static pair features s_z_static
        3. Stack of PairResBlocks with axial mixing
        4. Project to scalar velocity per pair, symmetrize

    Args:
        num_bins: Number of distance bins (64 for MiniFold).
        d_model: Hidden dimension of the pair representation.
        n_layers: Number of PairResBlock layers.
        c_z_static: Dimension of precomputed static pair features.
        expand: MLP expansion factor in each block.
    """

    def __init__(
        self,
        num_bins: int = 64,
        d_model: int = 64,
        n_layers: int = 4,
        c_z_static: int = 128,
        expand: int = 4,
    ):
        super().__init__()
        self.bin_embed = nn.Embedding(num_bins, d_model)
        self.static_proj = nn.Linear(c_z_static, d_model)

        self.layers = nn.ModuleList(
            [PairResBlock(d_model, expand=expand) for _ in range(n_layers)]
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        d_indices: torch.Tensor,
        s_z_static: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            d_indices: (B, L, L) long — current distance bin indices in [0, num_bins).
            s_z_static: (B, L, L, c_z_static) — precomputed static pair features.
            pair_mask: (B, L, L) float — 1.0 for valid pairs, 0.0 for padding.

        Returns:
            delta_d: (B, L, L) — predicted bin velocity (real-valued, symmetrized).
        """
        x = self.bin_embed(d_indices) + self.static_proj(s_z_static)

        for layer in self.layers:
            x = layer(x, pair_mask)

        delta_d = self.out_proj(x).squeeze(-1)  # (B, L, L)

        # Enforce symmetry (d(i,j) == d(j,i))
        delta_d = 0.5 * (delta_d + delta_d.transpose(1, 2))

        if pair_mask is not None:
            delta_d = delta_d * pair_mask

        return delta_d
