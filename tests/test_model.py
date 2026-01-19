import pytest
import torch
from functools import partial
from torch.optim import Adam

from mini_latent_pd.models.mlp import MoonsNet
from mini_latent_pd.flow_matching_module import FlowMatching, FlowMatchingCFG
from mini_latent_pd.modules.samplers import GaussianSampler
from mini_latent_pd.modules.schedulers import LinearScheduler
from mini_latent_pd.modules.solvers import EulerSolver


@pytest.fixture
def moons_model():
    return MoonsNet()


@pytest.fixture
def flow_matching_model(moons_model):
    return FlowMatching(
        model=moons_model,
        optimizer=partial(Adam, lr=1e-3),
        alpha_beta_scheduler=LinearScheduler(data_dim=2),
        sampler=GaussianSampler(target_shape=(2,)),
        ode_solver=EulerSolver(),
    )


@pytest.fixture
def flow_matching_cfg_model(moons_model):
    return FlowMatchingCFG(
        model=moons_model,
        optimizer=partial(Adam, lr=1e-3),
        alpha_beta_scheduler=LinearScheduler(data_dim=2),
        sampler=GaussianSampler(target_shape=(2,)),
        ode_solver=EulerSolver(),
        num_classes=2,  # For moons dataset (2 classes)
    )


def test_flow_matching_init(flow_matching_model):
    """Test if the FlowMatching model initializes correctly."""
    assert isinstance(flow_matching_model, FlowMatching)
    assert flow_matching_model.hparams.optimizer.keywords["lr"] == 1e-3


def test_flow_matching_training_step(flow_matching_model):
    """Test the training step of the FlowMatching model."""
    batch_size = 32
    x = torch.randn(batch_size, 2)
    y = torch.randint(0, 2, (batch_size,))
    loss = flow_matching_model.training_step((x, y), 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert not torch.isnan(loss).any()


def test_flow_matching_generate_samples(flow_matching_model):
    """Test sample generation for FlowMatching model."""
    num_samples = 10
    labels = torch.randint(0, 2, (num_samples,))
    samples = flow_matching_model.generate_samples(labels=labels, steps=10)
    assert samples.shape == (num_samples, 2)


def test_flow_matching_cfg_init(flow_matching_cfg_model):
    """Test if the FlowMatchingCFG model initializes correctly."""
    assert isinstance(flow_matching_cfg_model, FlowMatchingCFG)
    assert flow_matching_cfg_model.hparams.optimizer.keywords["lr"] == 1e-3


def test_flow_matching_cfg_training_step(flow_matching_cfg_model):
    """Test the training step of the FlowMatchingCFG model."""
    batch_size = 32
    x = torch.randn(batch_size, 2)
    y = torch.randint(0, 2, (batch_size,))
    loss = flow_matching_cfg_model.training_step((x, y), 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert not torch.isnan(loss).any()


def test_flow_matching_cfg_generate_samples(flow_matching_cfg_model):
    """Test sample generation for FlowMatchingCFG model."""
    num_samples = 10
    labels = torch.randint(0, 2, (num_samples,))
    samples = flow_matching_cfg_model.generate_samples(
        labels=labels, steps=10, guidance_scale=2.0
    )
    assert samples.shape == (num_samples, 2)


def test_configure_optimizers(flow_matching_model):
    """Test if the model configures optimizer correctly."""
    config = flow_matching_model.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.Adam)
    assert config["optimizer"].defaults["lr"] == 1e-3
