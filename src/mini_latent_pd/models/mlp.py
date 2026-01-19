import torch
import torch.nn as nn

class MoonsNet(nn.Module):
    """
    Simple MLP for 2D moons dataset that matches the UNet interface.
    Takes (x, t, y) just like UNet but works with 2D data instead of images.
    """
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=32, time_dim=16, num_classes=11):
        super().__init__()

        # Time embedding - same style as UNet
        from mini_latent_pd.models.unet import FourierEncoder
        self.time_mlp = nn.Sequential(
            FourierEncoder(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding
        self.class_emb = nn.Embedding(num_classes, time_dim)

        # Main network
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)  # Output same dim as input
        )

    def forward(self, x, t, y):
        """
        x: (batch, 2) - 2D moons data points
        t: (batch,) - time values
        y: (batch,) - class labels (0, 1, or 2 for null class)
        """
        # Get embeddings
        t_emb = self.time_mlp(t)  # (batch, time_dim)
        y_emb = self.class_emb(y)  # (batch, time_dim)

        # Concatenate everything
        inputs = torch.cat([x, t_emb, y_emb], dim=-1)

        # Forward through network
        return self.net(inputs)
