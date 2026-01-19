import os
import random
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import omegaconf
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

config_path = str(Path(__file__).resolve().parent.parent.parent / "configs")


@hydra.main(config_path=config_path, config_name="train_config", version_base="1.2")
def train(cfg: DictConfig):
    # 1. Instantiate the data module
    data_module = instantiate(cfg.data)

    # 2. Instantiate the model
    model = instantiate(cfg.model)

    # 3. Set up logger if configured
    logger = instantiate(cfg.logger)

    # 4. Set up callbacks if configured
    callbacks = [instantiate(cb) for _, cb in cfg.callbacks.items()]

    # 5. Instantiate the trainer
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    # 6. Train the model
    trainer.fit(model, data_module)

    # 7. Test the model
    trainer.test(model, data_module)


if __name__ == "__main__":
    train()
