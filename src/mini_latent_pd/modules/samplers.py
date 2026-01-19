import torch
import torch.nn as nn


class GaussianSampler(nn.Module):
    """Samples from a standard Gaussian distribution N(0, I)."""

    def __init__(self, target_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.target_shape = target_shape

    def forward(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generates samples.

        Args:
            num_samples (int): The number of samples to generate (batch size).
            device (torch.device): The device to place the samples on.

        Returns:
            torch.Tensor: A tensor of random samples.
        """
        shape = (num_samples, *self.target_shape)
        return torch.randn(shape, device=device)


class UniformSampler(nn.Module):
    """Samples from a uniform distribution U(-1, 1)."""

    def __init__(self, target_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.target_shape = target_shape

    def forward(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generates samples.

        Args:
            num_samples (int): The number of samples to generate (batch size).
            device (torch.device): The device to place the samples on.

        Returns:
            torch.Tensor: A tensor of random samples.
        """
        shape = (num_samples, *self.target_shape)
        return torch.rand(shape, device=device) * 2.0 - 1.0
