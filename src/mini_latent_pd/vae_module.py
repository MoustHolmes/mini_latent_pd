from __future__ import annotations
from functools import partial
from typing import Any, Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

class SpatialEncoder(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1, 
        latent_channels: int = 4,
        hidden_dims: list[int] = [32, 64], # Configurable via Hydra
    ):
        super().__init__()
        
        modules = []
        # Build Encoder Layers dynamically
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # Projection to Latent stats
        self.mean_var = nn.Conv2d(in_channels, latent_channels * 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        mu_logvar = self.mean_var(x)
        mu, log_var = torch.chunk(mu_logvar, 2, dim=1)
        return mu, log_var

class SpatialDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 1, 
        latent_channels: int = 4,
        hidden_dims: list[int] = [32, 64], # Same list as encoder
    ):
        super().__init__()
        
        # Reverse the dims for the decoder
        hidden_dims = hidden_dims[::-1] 
        
        # Initial projection
        self.decoder_input = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.SiLU()
        )

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'), # Or use ConvTranspose2d
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.SiLU()
                )
            )
        
        # Final Upsample and Output
        modules.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.SiLU(),
                nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x # Remember to add Sigmoid in model_step or here if data is [0,1]

class SpatialVAE(L.LightningModule):
    def __init__(
            self, 
            in_channels: int = 1, 
            latent_channels: int = 4, 
            kl_weight: float = 0.00025,
            encoder_hidden_dims: list[int] = [32, 64],
            decoder_hidden_dims: list[int] = [32, 64],
            optimizer: Callable[..., Optimizer] | None = None,
            lr_scheduler: Callable[..., _LRScheduler] | None = None,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = SpatialEncoder(in_channels, latent_channels, encoder_hidden_dims)
        self.decoder = SpatialDecoder(in_channels, latent_channels, decoder_hidden_dims)

        self.kl_weight = kl_weight
        self.hparams.optimizer = optimizer or partial(Adam, lr=1e-4)
        self.hparams.lr_scheduler = lr_scheduler or partial(ReduceLROnPlateau, patience=5, factor=0.2)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def model_step(self, batch):
        x, _ = batch
        recon, mu, log_var = self(x)
        # Reconstruction Loss
        recon_loss = F.mse_loss(recon, x)
        # KL Divergence
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=[1, 2, 3]))

        loss = recon_loss + self.kl_weight * kld_loss

        return {"loss": loss, "recon_loss": recon_loss, "kld_loss": kld_loss}
    
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

