import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1) or (bs,)
        Returns:
        - embeddings: (bs, dim)
        """
        if len(t.shape) > 1:
            t = t.view(-1, 1)  # (bs, 1)
        else:
            t = t.unsqueeze(-1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)


class ResidualBlock(nn.Module):
    """Basic residual block with time and class conditioning."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.cond_mlp = nn.Linear(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, cond):
        identity = self.shortcut(x)

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        # Add time and class embedding
        cond_emb = F.silu(self.cond_mlp(cond))[:, :, None, None]
        x = x + cond_emb

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + identity


class UNet(nn.Module):
    """Simplified U-Net architecture for conditional vector field/score estimation."""

    def __init__(
        self, in_channels=1, model_channels=64, out_channels=1, time_dim=256, num_classes=11
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            FourierEncoder(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding
        self.class_emb = nn.Embedding(num_classes, time_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.down1 = ResidualBlock(model_channels, model_channels, time_dim)
        self.down2 = ResidualBlock(model_channels, model_channels * 2, time_dim)
        self.down3 = ResidualBlock(model_channels * 2, model_channels * 2, time_dim)

        # Bottleneck
        self.bottleneck1 = ResidualBlock(model_channels * 2, model_channels * 4, time_dim)
        self.bottleneck2 = ResidualBlock(model_channels * 4, model_channels * 4, time_dim)

        # Decoder
        self.up1 = ResidualBlock(model_channels * 6, model_channels * 2, time_dim)
        self.up2 = ResidualBlock(model_channels * 4, model_channels * 2, time_dim)
        self.up3 = ResidualBlock(model_channels * 3, model_channels, time_dim)

        # Output
        self.final_norm = nn.GroupNorm(8, model_channels)
        self.final_act = nn.SiLU()
        self.final = nn.Conv2d(model_channels, out_channels, 1)

    def forward(self, x, t, y):
        # Time and class embedding
        t_emb = self.time_mlp(t)
        y_emb = self.class_emb(y)
        cond_emb = t_emb + y_emb

        # Initial processing
        x = self.init_conv(x)

        # Encoder path with skip connections
        d1 = self.down1(x, cond_emb)
        d2 = self.down2(F.avg_pool2d(d1, 2), cond_emb)
        d3 = self.down3(F.avg_pool2d(d2, 2), cond_emb)

        # Bottleneck
        b = self.bottleneck1(F.avg_pool2d(d3, 2), cond_emb)
        b = self.bottleneck2(b, cond_emb)

        # Decoder with skip connections
        u1 = self.up1(
            torch.cat([F.interpolate(b, size=d3.shape[-2:], mode="bilinear"), d3], dim=1), cond_emb
        )
        u2 = self.up2(
            torch.cat([F.interpolate(u1, size=d2.shape[-2:], mode="bilinear"), d2], dim=1), cond_emb
        )
        u3 = self.up3(
            torch.cat([F.interpolate(u2, size=d1.shape[-2:], mode="bilinear"), d1], dim=1), cond_emb
        )

        # Final processing
        out = self.final_norm(u3)
        out = self.final_act(out)
        out = self.final(out)

        return out
