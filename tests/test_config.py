"""Tests for configuration setup and instantiation."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import pytest


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    # Test that we can instantiate the data module
    data_module = hydra.utils.instantiate(cfg_train.data)
    assert data_module is not None

    # Test that we can instantiate the model
    model = hydra.utils.instantiate(cfg_train.model)
    assert model is not None

    # Test that we can instantiate the trainer
    trainer = hydra.utils.instantiate(cfg_train.trainer)
    assert trainer is not None


def test_train_config_debug(cfg_train_debug: DictConfig) -> None:
    """Tests the training configuration with debug overrides.

    :param cfg_train_debug: A DictConfig containing a valid debug training configuration.
    """
    assert cfg_train_debug
    assert cfg_train_debug.data
    assert cfg_train_debug.model
    assert cfg_train_debug.trainer

    HydraConfig().set_config(cfg_train_debug)

    # Test that we can instantiate the data module
    data_module = hydra.utils.instantiate(cfg_train_debug.data)
    assert data_module is not None

    # Test that we can instantiate the model
    model = hydra.utils.instantiate(cfg_train_debug.model)
    assert model is not None

    # Test that we can instantiate the trainer
    trainer = hydra.utils.instantiate(cfg_train_debug.trainer)
    assert trainer is not None


def test_config_has_required_fields(cfg_train: DictConfig) -> None:
    """Tests that the configuration has all required fields.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    # Check top-level fields
    assert "data" in cfg_train
    assert "model" in cfg_train
    assert "trainer" in cfg_train
    assert "task_name" in cfg_train

    # Check data config has _target_
    assert "_target_" in cfg_train.data

    # Check model config has _target_
    assert "_target_" in cfg_train.model

    # Check trainer config has _target_
    assert "_target_" in cfg_train.trainer


def test_config_overrides(cfg_train: DictConfig) -> None:
    """Tests that configuration values are properly set.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    # Test that trainer has expected configuration
    assert cfg_train.trainer.max_epochs >= 1
    assert cfg_train.trainer.min_epochs >= 1

    # Test that data module has expected configuration
    assert cfg_train.data.batch_size > 0
    assert cfg_train.data.num_workers >= 0


def test_model_components(cfg_train: DictConfig) -> None:
    """Tests that model configuration has all required components.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    # Check model has required subcomponents
    assert "model" in cfg_train.model
    assert "alpha_beta_scheduler" in cfg_train.model
    assert "sampler" in cfg_train.model
    assert "ode_solver" in cfg_train.model
    assert "optimizer" in cfg_train.model

    # Check that each component has _target_
    assert "_target_" in cfg_train.model.model
    assert "_target_" in cfg_train.model.alpha_beta_scheduler
    assert "_target_" in cfg_train.model.sampler
    assert "_target_" in cfg_train.model.ode_solver
    assert "_target_" in cfg_train.model.optimizer


def test_instantiate_all_components(cfg_train: DictConfig) -> None:
    """Tests that all configuration components can be instantiated without errors.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)

    # Instantiate data module
    data_module = hydra.utils.instantiate(cfg_train.data)
    assert data_module is not None
    assert hasattr(data_module, "prepare_data")
    assert hasattr(data_module, "setup")
    assert hasattr(data_module, "train_dataloader")

    # Instantiate model
    model = hydra.utils.instantiate(cfg_train.model)
    assert model is not None
    assert hasattr(model, "training_step")
    assert hasattr(model, "validation_step")
    assert hasattr(model, "configure_optimizers")

    # Instantiate trainer
    trainer = hydra.utils.instantiate(cfg_train.trainer)
    assert trainer is not None
    assert hasattr(trainer, "fit")


def test_train_with_fast_dev_run(cfg_train_debug: DictConfig) -> None:
    """Tests that training runs successfully with fast_dev_run enabled.

    This test performs a complete training run with the debug configuration,
    which includes fast_dev_run=True to run only 1 batch through train/val/test.

    :param cfg_train_debug: A DictConfig containing the debug training configuration.
    """
    HydraConfig().set_config(cfg_train_debug)

    # Instantiate all components
    data_module = hydra.utils.instantiate(cfg_train_debug.data)
    model = hydra.utils.instantiate(cfg_train_debug.model)

    # Logger is None in debug config
    logger = hydra.utils.instantiate(cfg_train_debug.logger) if cfg_train_debug.logger else None

    # Callbacks is None in debug config
    callbacks = (
        [hydra.utils.instantiate(cb) for _, cb in cfg_train_debug.callbacks.items()]
        if cfg_train_debug.callbacks
        else None
    )

    trainer = hydra.utils.instantiate(cfg_train_debug.trainer, logger=logger, callbacks=callbacks)

    # Verify fast_dev_run is enabled
    assert trainer.fast_dev_run is True

    # Run training (will only run 1 batch)
    trainer.fit(model, data_module)

    # Run testing (will only run 1 batch)
    trainer.test(model, data_module)

    # If we get here without errors, the training pipeline works correctly
    assert True
