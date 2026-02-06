import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class LatentProcessor(nn.Module, ABC):
    """
    Abstract base class for any method that compresses images to latents.
    """
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes x -> z"""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes z -> x"""
        pass
    
    @property
    @abstractmethod
    def latent_dim(self) -> tuple:
        """Returns the shape of the latent space (C, H, W) or (D,)"""
        pass

class AutoencoderWrapper(LatentProcessor):
    def __init__(
        self, 
        autoencoder: nn.Module, 
        checkpoint_path: str | None = None,
        scale_factor: float = 1.0,
        freeze: bool = True
    ):
        super().__init__()
        self.model = autoencoder
        self.scale_factor = scale_factor
        
        # 1. Load Weights if provided
        if checkpoint_path:
            # Simple loading logic (adjust key parsing if needed)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            # Strip "autoencoder." prefix if present
            state_dict = {k.replace("autoencoder.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)

        # 2. Check for internal scale factor (override manual if found)
        if hasattr(self.model, 'scale_factor'):
             self.scale_factor = self.model.scale_factor

        # 3. Freeze Weights
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Handle VAE tuple outputs automatically
        res = self.model.encoder(x)
        if isinstance(res, tuple):
            mu = res[0]
            return mu * self.scale_factor
        return res * self.scale_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Un-scale
        z = z / self.scale_factor
        return self.model.decoder(z)

    @property
    def latent_dim(self) -> tuple:
        # You might need to hardcode this or infer it via a dummy forward pass
        return (4, 7, 7)