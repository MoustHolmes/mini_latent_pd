"""Pytest configuration file for test fixtures."""

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict
from pathlib import Path


@pytest.fixture(scope="function")
def cfg_train() -> DictConfig:
    """A pytest fixture for loading the training configuration.

    Returns:
        A DictConfig containing a valid training configuration.
    """
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Get the absolute path to the configs directory
    config_dir = str(Path(__file__).parent.parent / "configs")

    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train_config",
            return_hydra_config=True,
            overrides=[f"data_dir={str(Path(__file__).parent.parent / 'data')}"],
        )

    return cfg


@pytest.fixture(scope="function")
def cfg_train_debug() -> DictConfig:
    """A pytest fixture for loading the training configuration with debug overrides.

    Returns:
        A DictConfig containing a valid training configuration for debugging.
    """
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Get the absolute path to the configs directory
    config_dir = str(Path(__file__).parent.parent / "configs")

    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train_config",
            return_hydra_config=True,
            overrides=[
                "experiment=debug",
                f"data_dir={str(Path(__file__).parent.parent / 'data')}",
            ],
        )

    return cfg
