"""OF3DiffusionModule: Lightning module for protein dynamics using OpenFold3 diffusion.

Imports OpenFold3 components as a plain library — no OF3 project infrastructure.

Direct imports:
  OpenFold3            — the all-atom diffusion nn.Module
  OpenFold3Loss        — aggregated loss (diffusion + distogram + confidence heads)
  ExponentialMovingAverage  — EMA weight averaging
  AlphaFoldLRScheduler — cosine-decay LR schedule with linear warmup

The OF3 model auto-selects train vs. inference mode via PyTorch's train()/eval():
  Training:   single-step denoising loss (cheap, used every step)
  Validation: full diffusion rollout → atom_positions_predicted (expensive;
              set limit_val_batches=8 in trainer to keep validation fast)
"""

import copy
import logging

import lightning.pytorch as pl
import torch

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.utils.exponential_moving_average import ExponentialMovingAverage
from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold3.projects.of3_all_atom.config.model_config import model_config as _base_config
from openfold3.projects.of3_all_atom.model import OpenFold3

logger = logging.getLogger(__name__)


def _build_model_config(overrides=None):
    """Deep-copy the base OF3 config and apply any nested overrides.

    Accepts a plain dict or an OmegaConf DictConfig (from Hydra instantiate).
    """
    cfg = copy.deepcopy(_base_config)
    if overrides:
        # OmegaConf DictConfig → plain Python dict so ml_collections can merge it
        try:
            from omegaconf import OmegaConf
            overrides = OmegaConf.to_container(overrides, resolve=True)
        except Exception:
            pass
        cfg.update(overrides)
    return cfg


class OF3DiffusionModule(pl.LightningModule):
    """OpenFold3 diffusion model for protein dynamics training.

    Args:
        of3_overrides: Nested dict of overrides on top of the base OF3 config,
            e.g. {"architecture": {"pairformer": {"no_blocks": 1}}}.
            Leave empty ({}) for the default full model.
        optimizer: Partial optimizer constructor (Hydra _partial_: true).
        lr_scheduler: Optional partial AlphaFoldLRScheduler constructor.
        disable_evo_attention: Disable DeepSpeed evo attention (required when
            not running under DeepSpeed).
    """

    def __init__(
        self,
        of3_overrides: dict,
        optimizer,
        lr_scheduler=None,
        disable_evo_attention: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "lr_scheduler"])

        cfg = _build_model_config(of3_overrides)

        if disable_evo_attention:
            cfg.settings.memory.train.use_deepspeed_evo_attention = False
            cfg.settings.memory.eval.use_deepspeed_evo_attention = False

        self.cfg = cfg
        self.model = OpenFold3(cfg)
        self.loss_fn = OpenFold3Loss(cfg.architecture.loss_module)
        self.ema = ExponentialMovingAverage(model=self.model, **cfg.settings.ema)

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._cached_weights = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch):
        return self.model(batch)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch, outputs = self.model(batch)
        loss, breakdown = self.loss_fn(batch, outputs, _return_breakdown=True)

        log_kw = dict(on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, prog_bar=True, **log_kw)
        for name, val in breakdown.items():
            if name != "loss":
                self.log(f"train/{name}", val, **log_kw)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update(self.model)

    # ------------------------------------------------------------------
    # Validation — runs with EMA weights and full diffusion rollout
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self):
        self._cached_weights = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def validation_step(self, batch, batch_idx):
        batch, outputs = self(batch)
        _, breakdown = self.loss_fn(batch, outputs, _return_breakdown=True)

        log_kw = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss", breakdown.get("loss", 0.0), prog_bar=True, **log_kw)
        for name, val in breakdown.items():
            if name != "loss":
                self.log(f"val/{name}", val, **log_kw)

        # The model applies unsqueeze(1) to the whole batch before the rollout,
        # so ground_truth tensors have an extra N_samples dim at position 1.
        # Squeeze it out so the callback gets (B, N_atom, ...) shaped tensors.
        gt_pos = batch["ground_truth"]["atom_positions"]
        atom_mask = batch["ground_truth"]["atom_resolved_mask"]
        # Remove the injected sample dimension if present
        if gt_pos.ndim == 4:
            gt_pos = gt_pos[:, 0]         # (B, N_atom, 3)
        if atom_mask.ndim == 3:
            atom_mask = atom_mask[:, 0]   # (B, N_atom)

        pred_pos = outputs.get("atom_positions_predicted")
        # pred_pos may be (B, N_samples, N_atom, 3) — take first sample
        if pred_pos is not None and pred_pos.ndim == 4:
            pred_pos = pred_pos[:, 0]     # (B, N_atom, 3)

        return {
            "atom_positions_predicted": pred_pos,
            "gt_atom_positions": gt_pos,
            "atom_mask": atom_mask,
            "domain_id": batch.get("domain_id"),
            "temp": batch.get("temp"),
            "lag": batch.get("lag"),
        }

    def on_validation_epoch_end(self):
        self.model.load_state_dict(self._cached_weights)
        self._cached_weights = None

    # ------------------------------------------------------------------
    # Optimizer + LR scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = self._optimizer(self.parameters())
        if self._lr_scheduler is None:
            return {"optimizer": optimizer}

        scheduler = self._lr_scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "lr"},
        }

    # ------------------------------------------------------------------
    # Checkpoint: persist EMA alongside model weights
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
