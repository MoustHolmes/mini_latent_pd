# Diffusion Playground

A minimal, well-structured PyTorch Lightning template for training Flow Matching models.

## Overview

This project provides a clean, modular framework for experimenting with Flow Matching, a powerful generative modeling technique. It includes:

- **Flow Matching**: Standard conditional flow matching
- **Flow Matching with CFG**: Classifier-Free Guidance for improved sample quality
- **Modular Components**: Easy-to-swap models, schedulers, samplers, and ODE solvers
- **Configuration Management**: Hydra-based config system for reproducible experiments
- **Experiment Tracking**: Weights & Biases integration
- **Production Ready**: Type hints, tests, and documentation

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/username/diffusion_playground.git
cd diffusion_playground
```

2. Create a virtual environment:
```bash
conda create -n diffusion_playground python=3.11
conda activate diffusion_playground
```

3. Install dependencies:
```bash
pip install -e .
```

### Training

Train a Flow Matching model on MNIST:
```bash
python src/diffusion_playground/train.py
```

Train with the debug configuration (fast_dev_run):
```bash
python src/diffusion_playground/train.py experiment=debug
```

Train on the Moons 2D dataset:
```bash
python src/diffusion_playground/train.py experiment=moons
```

### Configuration

Override any configuration from the command line:
```bash
# Change batch size and learning rate
python src/diffusion_playground/train.py data.batch_size=64 model.optimizer.lr=0.0001

# Use a different model configuration
python src/diffusion_playground/train.py model=moons_model data=moons_data_module
```

## Project Structure

```
├── configs/              # Hydra configuration files
│   ├── train_config.yaml     # Main config
│   ├── data/                 # Data module configs
│   ├── model/                # Model configs
│   ├── trainer/              # Trainer configs
│   ├── callbacks/            # Callback configs
│   └── experiments/          # Experiment configs
├── src/
│   └── diffusion_playground/
│       ├── train.py           # Main training script
│       ├── models/            # LightningModules (training logic)
│       │   └── flow_matching.py        # FlowMatching & FlowMatchingCFG
│       ├── networks/          # Neural network architectures
│       │   ├── unet.py                 # U-Net for images
│       │   └── mlp.py                  # MLP for 2D data
│       ├── modules/           # Reusable building blocks
│       │   ├── schedulers.py           # Alpha/beta schedulers
│       │   ├── samplers.py             # Noise samplers
│       │   └── solvers.py              # ODE solvers
│       ├── data/              # Data modules
│       │   ├── MNIST_datamodule.py
│       │   └── moons_datamodule.py
│       ├── callbacks/         # Custom callbacks
│       └── util/              # Utility functions
├── tests/                # Unit tests
├── data/                 # Data directory
└── outputs/              # Training outputs
```

## Key Features

### Flow Matching

Standard conditional flow matching that learns to transform noise into data:

```python
from diffusion_playground.models import FlowMatching

model = FlowMatching(
    model=unet,
    alpha_beta_scheduler=scheduler,
    sampler=sampler,
    ode_solver=solver,
)
```

### Classifier-Free Guidance

Improved generation quality through guidance:

```python
from diffusion_playground.models import FlowMatchingCFG

model = FlowMatchingCFG(
    model=unet,
    num_classes=10,
    cfg_prob=0.1,
    guidance_scale=3.0,
)

# Generate with custom guidance
samples = model.generate_samples(labels, guidance_scale=5.0)
```

## Examples

See the `configs/experiments/` directory for example configurations:

- `debug.yaml` - Fast development run
- `moons.yaml` - 2D toy dataset example

## Testing

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_config.py -v
```

## License

MIT License - see LICENSE file for details.
