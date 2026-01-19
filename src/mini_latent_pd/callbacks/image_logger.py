from typing import Any, Dict

import lightning as L
import torch
import wandb


class ImageLoggerCallback(L.Callback):
    def __init__(self, log_every_n_batches: int = 100, num_samples: int = 8):
        """
        Args:
            log_every_n_batches: Log images every N batches
            num_samples: Number of images to log each time
        """
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.num_samples = num_samples
        # MNIST normalization constants
        self.mean = torch.tensor([0.1307])
        self.std = torch.tensor([0.3081])

    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize images from [-mean/std, (1-mean)/std] back to [0, 1]."""
        images = images * self.std[None, :, None, None] + self.mean[None, :, None, None]
        return torch.clamp(images, 0, 1)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log images to WandB on train batch end."""
        if batch_idx % self.log_every_n_batches == 0:
            # Get images and predictions
            images, targets = batch
            images = images[: self.num_samples]

            # Get model predictions
            with torch.no_grad():
                logits = pl_module(images)
                preds = torch.argmax(logits, dim=1)

            # Convert images to format suitable for logging
            images = self.denormalize(images)
            images_np = [
                img.permute(1, 2, 0).cpu().numpy() for img in images
            ]  # Convert to (H, W, C) format

            # Create a list of wandb Image objects with predictions as captions
            image_list = [
                wandb.Image(img, caption=f"Pred: {pred.item()}, True: {target.item()}")
                for img, pred, target in zip(images_np, preds, targets[: self.num_samples])
            ]

            # Log to wandb
            trainer.logger.experiment.log(
                {
                    "train/example_images": image_list,
                },
                step=trainer.global_step,
            )
