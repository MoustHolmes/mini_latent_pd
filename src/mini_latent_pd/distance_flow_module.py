"""Distance-space flow matching for molecular dynamics.

Lightning module that trains a velocity network to predict distance-bin
changes between MD frames, using a frozen MiniFold decoder to optionally
decode predictions into 3D coordinates for validation / logging.

Latent space: distance bin indices d ∈ {0, ..., 63}^{L×L}.
Encoding:  atom37_coords → pseudo-beta distances → bin index.
Velocity:  Δd = d_target - d_source (integer target, real-valued prediction).
Decoding:  bin_indices → one_hot → recycle → MiniFormer → PairToSeq → SM → atom37.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from mini_latent_pd.models.velocity_net import DistanceVelocityNet
from mini_latent_pd.models.minifold_decoder import MiniFoldDecoder
from mini_latent_pd.utils.geometry import (
    coords_to_bin_indices,
    kabsch_rmsd,
    pseudo_beta_from_atom37,
)


class DistanceFlowModule(L.LightningModule):
    """Train a velocity field in distance-bin space for MD dynamics.

    The module owns:
        - ``velocity_net``: trainable network predicting Δd (bin velocity).
        - ``decoder``: frozen MiniFold decoder (bin_indices → atom37).

    Training loop:
        1. Receive frame pairs (x0, xt) from MDCathToMinifoldDataset.
        2. Encode both to bin indices d_0, d_t.
        3. Target velocity: v_target = (d_t - d_0).float()
        4. Predict: v_pred = velocity_net(d_0, s_z_static, pair_mask)
        5. Loss: MSE on upper-triangle elements (symmetric, exclude diagonal).

    Args:
        velocity_net: The trainable velocity network.
        decoder: Frozen MiniFold decoder (optional, for val-time structure decode).
        decoder_checkpoint: Path to MiniFold checkpoint for lazy decoder init.
        num_miniformer_layers: MiniFormer layers for the decoder.
        decode_every_n_val_steps: Decode and compute RMSD every N val batches.
        optimizer: Partial function for the optimizer.
        lr_scheduler: Partial function for the LR scheduler.
    """

    def __init__(
        self,
        velocity_net: nn.Module | None = None,
        decoder: MiniFoldDecoder | None = None,
        decoder_checkpoint: str | None = None,
        num_miniformer_layers: int = 12,
        decode_every_n_val_steps: int = 50,
        optimizer: Callable[..., Optimizer] | None = None,
        lr_scheduler: Callable[..., _LRScheduler] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["velocity_net", "decoder"],
        )

        self.velocity_net = velocity_net or DistanceVelocityNet()
        self.decoder = decoder
        self._decoder_checkpoint = decoder_checkpoint
        self._num_miniformer_layers = num_miniformer_layers
        self._decode_every_n_val_steps = decode_every_n_val_steps

        self.hparams.optimizer = optimizer or partial(Adam, lr=1e-4)
        self.hparams.lr_scheduler = lr_scheduler or partial(
            ReduceLROnPlateau, patience=5, factor=0.2
        )

    # ------------------------------------------------------------------
    # Lazy decoder init (heavy, only needed for decode steps)
    # ------------------------------------------------------------------

    def _ensure_decoder(self) -> MiniFoldDecoder | None:
        """Load the frozen decoder on first use (avoids loading at init)."""
        if self.decoder is not None:
            return self.decoder
        if self._decoder_checkpoint is not None:
            self.decoder = MiniFoldDecoder.from_checkpoint(
                self._decoder_checkpoint,
                device=str(self.device),
                num_miniformer_layers=self._num_miniformer_layers,
            )
            return self.decoder
        return None

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def encode_bin_indices(
        atom37_pos: torch.Tensor,
        aatype: torch.Tensor,
    ) -> torch.Tensor:
        """Encode atom37 coordinates to distance bin indices.

        Returns:
            bin_indices: (B, L, L) long, values in [0, 63].
        """
        return coords_to_bin_indices(atom37_pos, aatype)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute the velocity-prediction loss for one batch.

        Args:
            batch: dict with keys x0, xt, aatype, seq_mask, s_s, s_z.

        Returns:
            dict with 'loss' and per-metric scalars.
        """
        x0 = batch["x0"]  # (B, L, 37, 3)
        xt = batch["xt"]  # (B, L, 37, 3)
        aatype = batch["aatype"]  # (B, L)
        seq_mask = batch["seq_mask"]  # (B, L)
        s_s = batch["s_s"]  # (B, L, 1024)
        s_z = batch["s_z"]  # (B, L, L, 128)

        pair_mask = seq_mask[:, None, :] * seq_mask[:, :, None]  # (B, L, L)

        # Encode frames to bin indices
        d_0 = self.encode_bin_indices(x0, aatype)  # (B, L, L)
        d_t = self.encode_bin_indices(xt, aatype)  # (B, L, L)

        # Target velocity (real-valued bin differences)
        v_target = (d_t - d_0).float()

        # Compute static pair features (frozen weights, no grad)
        decoder = self._ensure_decoder()
        if decoder is not None:
            with torch.no_grad():
                s_z_static = decoder.compute_static_features(s_s, s_z, seq_mask)
        else:
            # Fallback: use raw s_z as static features (less accurate)
            s_z_static = s_z

        # Predict velocity
        v_pred = self.velocity_net(d_0, s_z_static, pair_mask)

        # MSE on upper-triangle (exclude diagonal, exclude padding)
        B, L, _ = pair_mask.shape
        triu_mask = torch.triu(torch.ones(L, L, device=pair_mask.device), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0) * pair_mask  # (B, L, L)

        n_valid = triu_mask.sum().clamp(min=1)
        loss = ((v_pred - v_target) ** 2 * triu_mask).sum() / n_valid

        # Metrics
        with torch.no_grad():
            mae = ((v_pred - v_target).abs() * triu_mask).sum() / n_valid
            # Fraction of bins predicted exactly right (rounded)
            bin_acc = (
                (v_pred.round() == v_target).float() * triu_mask
            ).sum() / n_valid

        return {"loss": loss, "mae": mae, "bin_accuracy": bin_acc}

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self.model_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["seq_mask"].shape[0],
        )
        return metrics["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        metrics = self.model_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["seq_mask"].shape[0],
        )

        # Optionally decode and compute structural RMSD
        if batch_idx % self._decode_every_n_val_steps == 0:
            self._decode_and_log_rmsd(batch)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        metrics = self.model_step(batch)
        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["seq_mask"].shape[0],
        )

    # ------------------------------------------------------------------
    # Decode + RMSD logging (validation only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_and_log_rmsd(self, batch: dict) -> None:
        """Decode a predicted next-frame and log Cα RMSD vs ground truth."""
        decoder = self._ensure_decoder()
        if decoder is None:
            return

        x0 = batch["x0"]
        xt = batch["xt"]
        aatype = batch["aatype"]
        seq_mask = batch["seq_mask"]
        s_s = batch["s_s"]
        s_z = batch["s_z"]

        d_0 = self.encode_bin_indices(x0, aatype)

        pair_mask = seq_mask[:, None, :] * seq_mask[:, :, None]
        s_z_static = decoder.compute_static_features(s_s, s_z, seq_mask)

        # Predict and apply velocity
        v_pred = self.velocity_net(d_0, s_z_static, pair_mask)
        d_pred = (d_0 + v_pred.round().long()).clamp(0, 63)

        # Decode predicted distogram → atom37
        decoded = decoder.decode(d_pred, s_s, s_z_static, aatype, seq_mask)
        pred_pos = decoded["atom37_pos"]  # (B, L, 37, 3)

        # Compute Cα RMSD for first sample in batch
        mask = seq_mask[0].bool()
        ca_idx = 1  # atom37 index for Cα
        ca_pred = pred_pos[0, mask, ca_idx]
        ca_gt = xt[0, mask, ca_idx]

        rmsd = kabsch_rmsd(ca_pred, ca_gt)
        self.log("val/ca_rmsd", rmsd, prog_bar=True)

    # ------------------------------------------------------------------
    # Inference: single-step prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_next_frame(
        self,
        atom37_pos: torch.Tensor,
        aatype: torch.Tensor,
        seq_mask: torch.Tensor,
        s_s: torch.Tensor,
        s_z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict the next MD frame from current coordinates.

        Args:
            atom37_pos: (B, L, 37, 3) current frame.
            aatype: (B, L) residue types.
            seq_mask: (B, L) mask.
            s_s: (B, L, 1024) sequence features.
            s_z: (B, L, L, 128) pair features.

        Returns:
            dict with ``atom37_pos``, ``bin_indices``, ``delta_d``.
        """
        decoder = self._ensure_decoder()
        assert decoder is not None, "Decoder required for inference"

        pair_mask = seq_mask[:, None, :] * seq_mask[:, :, None]
        s_z_static = decoder.compute_static_features(s_s, s_z, seq_mask)

        d_0 = self.encode_bin_indices(atom37_pos, aatype)
        delta_d = self.velocity_net(d_0, s_z_static, pair_mask)
        d_next = (d_0 + delta_d.round().long()).clamp(0, 63)

        decoded = decoder.decode(d_next, s_s, s_z_static, aatype, seq_mask)
        decoded["bin_indices"] = d_next
        decoded["delta_d"] = delta_d
        return decoded

    # ------------------------------------------------------------------
    # Inference: multi-step rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        atom37_pos: torch.Tensor,
        aatype: torch.Tensor,
        seq_mask: torch.Tensor,
        s_s: torch.Tensor,
        s_z: torch.Tensor,
        n_steps: int = 100,
        re_encode: bool = True,
    ) -> list[torch.Tensor]:
        """Generate a trajectory by iteratively predicting next frames.

        Args:
            atom37_pos: (1, L, 37, 3) starting frame (batch size 1).
            aatype: (1, L) residue types.
            seq_mask: (1, L) mask.
            s_s: (1, L, 1024) sequence features.
            s_z: (1, L, L, 128) pair features.
            n_steps: Number of prediction steps.
            re_encode: If True, re-encode decoded coordinates to prevent
                drift in bin-index space. If False, just accumulate Δd.

        Returns:
            List of atom37_pos tensors: [(1, L, 37, 3)] * n_steps.
        """
        decoder = self._ensure_decoder()
        assert decoder is not None, "Decoder required for rollout"

        pair_mask = seq_mask[:, None, :] * seq_mask[:, :, None]
        s_z_static = decoder.compute_static_features(s_s, s_z, seq_mask)

        d_current = self.encode_bin_indices(atom37_pos, aatype)
        trajectory = []

        for _ in range(n_steps):
            # Predict velocity
            delta_d = self.velocity_net(d_current, s_z_static, pair_mask)
            d_next = (d_current + delta_d.round().long()).clamp(0, 63)

            # Decode → atom37
            decoded = decoder.decode(d_next, s_s, s_z_static, aatype, seq_mask)
            trajectory.append(decoded["atom37_pos"])

            # Re-encode from decoded coords to prevent bin-space drift
            if re_encode:
                d_current = self.encode_bin_indices(decoded["atom37_pos"], aatype)
            else:
                d_current = d_next

        return trajectory

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(self.velocity_net.parameters())
        scheduler = self.hparams.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }
