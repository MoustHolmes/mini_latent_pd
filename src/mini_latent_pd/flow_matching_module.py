from __future__ import annotations

from functools import partial
from typing import Any, Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from mini_latent_pd.modules.schedulers import LinearScheduler
from mini_latent_pd.modules.samplers import GaussianSampler
from mini_latent_pd.modules.solvers import EulerSolver


class FlowMatching(L.LightningModule):
    """A PyTorch Lightning module for training a Flow Matching model.

    This module encapsulates the training, validation, and test logic for a
    conditional flow matching model. It is designed for modularity, allowing for
    easy swapping of models, samplers, ODE solvers, and schedulers via configuration.

    Attributes:
        model (torch.nn.Module): The neural network that predicts the vector field.
        optimizer (Callable[..., Optimizer]): A partial function to create the optimizer.
        lr_scheduler (Callable[..., _LRScheduler]): A partial function to create the LR scheduler.
        alpha_beta_scheduler (torch.nn.Module): The scheduler for interpolation coefficients.
        sampler (torch.nn.Module): The sampler for the initial distribution P_0.
        ode_solver (torch.nn.Module): The ODE solver for generating samples.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha_beta_scheduler: nn.Module = LinearScheduler(data_dim=4),
        sampler: nn.Module = GaussianSampler(target_shape=(1, 28, 28)),
        ode_solver: nn.Module = EulerSolver(),
        optimizer: Callable[..., Optimizer] | None = None,
        lr_scheduler: Callable[..., _LRScheduler] | None = None,
    ) -> None:
        """Initializes the FlowMatching module.

        Args:
            model: The neural network to be trained (predicts the vector field).
            alpha_beta_scheduler: The scheduler for interpolation coefficients (alpha_t, beta_t).
            sampler: An object to sample from the initial distribution P_0.
            ode_solver: An object to solve the ODE for generation.
            optimizer: A partial function for creating the optimizer. If None,
                Adam with lr=1e-4 is used as a default.
            lr_scheduler: A partial function for creating the learning rate scheduler.
                If None, ReduceLROnPlateau is used as a default.
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "alpha_beta_scheduler", "sampler", "ode_solver"])

        self.model = model
        self.alpha_beta_scheduler = alpha_beta_scheduler
        self.sampler = sampler
        self.ode_solver = ode_solver

        # Provide sensible defaults for notebook usage, which can be overridden by Hydra configs.
        self.hparams.optimizer = optimizer or partial(Adam, lr=1e-3)
        self.hparams.lr_scheduler = lr_scheduler or partial(ReduceLROnPlateau, patience=5, factor=0.2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        return self.model(x, t, y)

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Performs a single model step on a batch of data."""
        real_images, labels = batch
        batch_size = real_images.shape[0]

        x0 = self.sampler(num_samples=batch_size, device=self.device)
        t = torch.rand(batch_size, device=self.device)

        alpha_t, beta_t = self.alpha_beta_scheduler(t)

        xt = alpha_t * real_images + beta_t * x0
        u_target = real_images - x0
        u_pred = self(xt, t, labels)

        loss = F.mse_loss(u_pred, u_target)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        loss = self.model_step(batch)
        self.log('train/flow_matching_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Performs a single validation step."""
        loss = self.model_step(batch)
        self.log('val/flow_matching_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Performs a single test step."""
        loss = self.model_step(batch)
        self.log('test/flow_matching_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures the optimizers and learning rate schedulers."""
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = self.hparams.lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/flow_matching_loss",
                "interval": "epoch",
            },
        }

    @torch.no_grad()
    def generate_samples(self, labels: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Generates samples from the learned distribution."""
        self.model.eval()

        # Sample initial noise
        x0 = self.sampler(num_samples=len(labels), device=self.device)

        # Solve the ODE
        samples = self.ode_solver.solve(self.model, x0, labels, steps)
        return samples

class FlowMatchingCFG(L.LightningModule):
    """A PyTorch Lightning module for training a Flow Matching model with
    Classifier-Free Guidance (CFG).

    This module extends the standard Flow Matching setup by incorporating CFG,
    which involves training the model on both conditional and unconditional
    (label-dropped) data. This allows for improved sample quality at inference time
    by guiding the generation process.

    Attributes:
        model (torch.nn.Module): The neural network that predicts the vector field.
        optimizer (Callable[..., Optimizer]): A partial function to create the optimizer.
        lr_scheduler (Callable[..., _LRScheduler]): A partial function to create the LR scheduler.
        alpha_beta_scheduler (torch.nn.Module): The scheduler for interpolation coefficients.
        sampler (torch.nn.Module): The sampler for the initial distribution P_0.
        ode_solver (torch.nn.Module): The ODE solver for generating samples.
        cfg_prob (float): The probability of dropping a label during training.
        num_classes (int): The total number of classes, including the null token.
        guidance_scale (float): The default scale for classifier-free guidance at inference.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha_beta_scheduler: nn.Module = LinearScheduler(data_dim=4),
        sampler: nn.Module = GaussianSampler(target_shape=(1, 28, 28)),
        ode_solver: nn.Module= EulerSolver(),
        num_classes: int = 10,
        cfg_prob: float = 0.1,
        guidance_scale: float = 3.0,
        optimizer: Callable[..., Optimizer] | None = None,
        lr_scheduler: Callable[..., _LRScheduler] | None = None,
    ) -> None:
        """Initializes the FlowMatchingCFG module.

        Args:
            model: The neural network to be trained.
            alpha_beta_scheduler: The scheduler for interpolation coefficients.
            sampler: An object to sample from the initial distribution P_0.
            ode_solver: An object to solve the ODE for generation.
            num_classes: The number of conditional classes (e.g., 10 for MNIST).
            cfg_prob: The probability of dropping a class label during training.
            guidance_scale: The default guidance strength for generation.
            optimizer: A partial function for creating the optimizer. Defaults to Adam.
            lr_scheduler: A partial function for creating the LR scheduler. Defaults to ReduceLROnPlateau.
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "alpha_beta_scheduler", "sampler", "ode_solver"])

        self.model = model
        self.alpha_beta_scheduler = alpha_beta_scheduler
        self.sampler = sampler
        self.ode_solver = ode_solver

        # Add 1 to num_classes to account for the null token used for unconditional training
        self.hparams.num_classes = num_classes + 1

        # Provide sensible defaults for optimizer and scheduler
        self.hparams.optimizer = optimizer or partial(Adam, lr=1e-4)
        self.hparams.lr_scheduler = lr_scheduler or partial(ReduceLROnPlateau, patience=5, factor=0.2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        return self.model(x, t, y)

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Performs a single model step on a batch of data with CFG."""
        real_images, labels = batch
        batch_size = real_images.shape[0]

        # Randomly drop labels for classifier-free guidance training
        is_unconditional = torch.rand(batch_size, device=self.device) < self.hparams.cfg_prob
        # The null token is the highest class index
        labels[is_unconditional] = self.hparams.num_classes - 1

        x0 = self.sampler(num_samples=batch_size, device=self.device)
        t = torch.rand(batch_size, device=self.device)
        alpha_t, beta_t = self.alpha_beta_scheduler(t)

        xt = alpha_t * real_images + beta_t * x0
        u_target = real_images - x0
        u_pred = self(xt, t, labels)

        loss = F.mse_loss(u_pred, u_target)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        loss = self.model_step(batch)
        self.log('train/flow_matching_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Performs a single validation step."""
        loss = self.model_step(batch)
        self.log('val/flow_matching_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Performs a single test step."""
        loss = self.model_step(batch)
        self.log('test/flow_matching_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures the optimizers and learning rate schedulers."""
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = self.hparams.lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/flow_matching_loss",
                "interval": "epoch",
            },
        }

    def _guided_forward(
        self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, guidance_scale: float
    ) -> torch.Tensor:
        """Performs a guided forward pass for CFG."""
        # Create unconditional labels (null token)
        uncond_labels = torch.full_like(labels, self.hparams.num_classes - 1)

        # Get both conditional and unconditional predictions
        pred_cond = self.model(x, t, labels)
        pred_uncond = self.model(x, t, uncond_labels)

        # Apply classifier-free guidance formula
        return (1 + guidance_scale) * pred_cond - guidance_scale * pred_uncond

    @torch.no_grad()
    def generate_samples(
        self, labels: torch.Tensor, steps: int = 50, guidance_scale: float | None = None
    ) -> torch.Tensor:
        """Generates samples using Classifier-Free Guidance.

        Args:
            labels (torch.Tensor): The conditional labels for the samples to generate.
            steps (int): The number of steps for the ODE solver.
            guidance_scale (float, optional): The guidance strength. If None, uses the
                default scale from initialization.

        Returns:
            torch.Tensor: The generated samples.
        """
        self.model.eval()

        scale = guidance_scale if guidance_scale is not None else self.hparams.guidance_scale

        # The ODE solver needs a callable that matches the `model(x, t, labels)` signature.
        # We create a partial function that wraps our guided forward pass.
        guided_model_callable = partial(self._guided_forward, guidance_scale=scale)

        # Sample initial noise and solve the ODE with the guided model
        x0 = self.sampler(num_samples=len(labels), device=self.device)

        # We pass the guided callable to the solver. The solver will internally call it like:
        # guided_model_callable(x, t, labels=labels)
        return self.ode_solver.solve(guided_model_callable, x0=x0, labels=labels, steps=steps)
