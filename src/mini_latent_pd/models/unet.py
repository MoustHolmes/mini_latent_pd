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
        self, 
        in_channels: int = 4,
        out_channels: int = 4, 
        model_channels: int = 64, 
        channel_mult: list[int] = [1, 2, 2],
        time_dim: int = 256, 
        num_classes: int = 11
    ):
        super().__init__()
        self.time_dim = time_dim
        
        # 1. Time & Class Embeddings
        self.time_mlp = nn.Sequential(
            FourierEncoder(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)

        # 2. Input Convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # 3. Dynamic Encoder (Down)
        self.downs = nn.ModuleList()
        ch = model_channels
        dims = [ch] # Keep track of channels for skip connections
        
        for mult in channel_mult:
            out_ch = model_channels * mult
            # For 7x7 input, be careful with pooling! 
            # We can use stride=1 (no downsample) for the last layers if needed
            self.downs.append(ResidualBlock(ch, out_ch, time_dim)) 
            ch = out_ch
            dims.append(ch)

        # 4. Bottleneck
        self.mid_block1 = ResidualBlock(ch, ch, time_dim)
        self.mid_block2 = ResidualBlock(ch, ch, time_dim)

        # 5. Dynamic Decoder (Up)
        self.ups = nn.ModuleList()
        # Remove the last dim from list as it is the bottleneck input
        dims.pop() 
        
        for mult in reversed(channel_mult):
            out_ch = model_channels * mult
            skip_ch = dims.pop()
            # Input to ResBlock is current_ch + skip_ch
            self.ups.append(ResidualBlock(ch + skip_ch, out_ch, time_dim))
            ch = out_ch

        # 6. Output
        self.final_norm = nn.GroupNorm(8, ch)
        self.final_act = nn.SiLU()
        self.final = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x, t, y):
        # Embeddings
        t_emb = self.time_mlp(t)
        y_emb = self.class_emb(y)
        cond = t_emb + y_emb

        h = self.init_conv(x)
        skips = [h]

        # Down
        for layer in self.downs:
            h = layer(h, cond)
            skips.append(h)
            # Standard UNet pools AFTER the block. 
            # Note: For 7x7 latents, consider removing pooling or using it sparingly.
            if h.shape[-1] > 1: 
                h = F.avg_pool2d(h, 2)

        # Bottleneck
        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        # Up
        for layer in self.ups:
            skip = skips.pop()
            # Upsample to match skip connection size
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = layer(h, cond)

        return self.final(self.final_act(self.final_norm(h)))
