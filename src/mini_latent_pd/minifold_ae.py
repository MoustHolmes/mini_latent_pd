from functools import partial
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import lightning as L

class MiniFoldAE(L.LightningModule):
    def __init__(
            self, 

            optimizer: Callable[..., Optimizer] | None = None,
            lr_scheduler: Callable[..., _LRScheduler] | None = None,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.hparams.optimizer = optimizer or partial(Adam, lr=1e-4)
        self.hparams.lr_scheduler = lr_scheduler or partial(ReduceLROnPlateau, patience=5, factor=0.2)

    def forward(self, x):
        

        return 

    
    def model_step(self, batch):


        return 
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        losses = self.model_step(batch)
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, on_step=True, on_epoch=True, prog_bar=True)
        return losses["loss"]
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        losses = self.model_step(batch)
        self.log_dict({f"val/{k}": v for k, v in losses.items()}, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        losses = self.model_step(batch)
        self.log_dict({f"test/{k}": v for k, v in losses.items()}, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures the optimizers and learning rate schedulers."""
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = self.hparams.lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }