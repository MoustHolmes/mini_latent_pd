import torch
import torch.nn as nn
from math import pi


class LinearScheduler(nn.Module):
    """
    Linear scheduler where alpha_t = t.
    """

    def __init__(self, data_dim=4):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, t):
        shape = [-1] + [1] * (self.data_dim - 1)
        alpha_t = t.view(*shape)
        beta_t = (1 - t).view(*shape)
        return alpha_t, beta_t


class CosineScheduler(nn.Module):
    """
    Cosine scheduler often used in diffusion models.
    """

    def __init__(self, data_dim=4):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, t):
        shape = [-1] + [1] * (self.data_dim - 1)
        alpha_t = torch.cos(pi * 0.5 * t).view(*shape)
        beta_t = torch.sin(pi * 0.5 * t).view(*shape)
        return alpha_t, beta_t
