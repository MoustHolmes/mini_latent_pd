import torch
import matplotlib.pyplot as plt
import torchvision.utils
import math
import numpy as np


# Helper function to display images
def show_imgs(imgs, title=None, row_size=4):
    # Denormalize: x = (x * std) + mean, then scale to [0,1]
    imgs = (imgs * 0.3081) + 0.1307
    imgs = torch.clamp(imgs, 0, 1)

    if not isinstance(imgs, torch.Tensor):
        imgs = torch.stack([img[0] if isinstance(img, tuple) else img for img in imgs])

    # Get number of images correctly from first dimension
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    is_int = imgs.dtype == torch.int32

    # Convert to float for displaying
    if is_int:
        imgs = imgs.float() / 255.0
        is_int = False

    nrow = min(num_imgs, row_size)

    # Ensure all images are 3 channels for torchvision.utils.make_grid if they are grayscale
    if imgs.shape[1] == 1:  # If grayscale (1 channel)
        imgs = imgs.repeat(1, 3, 1, 1)  # Convert to 3 channels for consistent display

    grid_imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128 if is_int else 0.5)
    np_imgs = grid_imgs.cpu().numpy()

    plt.figure(figsize=(1.5 * nrow, 1.5 * math.ceil(num_imgs / nrow)))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation="nearest")
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()
