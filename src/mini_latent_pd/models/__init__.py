"""PyTorch models and architectures."""

from mini_latent_pd.models.unet import UNet, FourierEncoder
from mini_latent_pd.models.mlp import MoonsNet
from mini_latent_pd.vae_module import SpatialEncoder, SpatialDecoder, SpatialVAE


__all__ = [
    "UNet",
    "FourierEncoder",
    "MoonsNet",
    "StructureAutoencoder",
    "SpatialEncoder",
    "SpatialDecoder",
    "SpatialVAE",
    "DistanceVelocityNet",
    "MiniFoldDecoder",
]


def __getattr__(name):
    if name == "StructureAutoencoder":
        from mini_latent_pd.models.structure_autoencoder import StructureAutoencoder
        return StructureAutoencoder
    if name == "DistanceVelocityNet":
        from mini_latent_pd.models.velocity_net import DistanceVelocityNet
        return DistanceVelocityNet
    if name == "MiniFoldDecoder":
        from mini_latent_pd.models.minifold_decoder import MiniFoldDecoder
        return MiniFoldDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
