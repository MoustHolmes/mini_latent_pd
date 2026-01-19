import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from mini_latent_pd.train import train


@pytest.fixture(scope="function")
def hydra_initialize():
    """Fixture to initialize Hydra for testing."""
    # Clear Hydra's global state before each test
    GlobalHydra.instance().clear()
    # Initialize Hydra
    initialize(config_path="../configs", version_base=None)


def test_train_fast_dev_run(hydra_initialize):
    """
    Test if the training script runs with fast_dev_run=True.
    This checks for basic configuration and model instantiation errors.
    """
    # Compose the configuration
    cfg = compose(
        config_name="train_config",
        overrides=["experiment=debug"],
    )

    # Run the training function
    # The train function will raise an exception if something is wrong
    train(cfg)
