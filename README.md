# mini_latent_pd

A pytorch package for doing protein dynamics in latent spaces defined by pretrained protein folding models

## Overview

A minimal, production-ready PyTorch Lightning template for training **Flow Matching** models. Features include:

- 🎯 **Flow Matching & Classifier-Free Guidance**: State-of-the-art generative modeling
- 🔧 **Modular Design**: Easy-to-swap components (models, networks, schedulers, samplers, solvers)
- ⚙️ **Hydra Configuration**: Clean, composable configs for reproducible experiments
- 📊 **Experiment Tracking**: Built-in Weights & Biases integration
- ✅ **Production Ready**: Type hints, comprehensive tests, and documentation
- 🚀 **Quick Start**: Train on MNIST or 2D toy datasets out of the box

## What is Flow Matching?

Flow Matching is a simulation-free approach to training continuous normalizing flows. Key advantages:

- **Simple Training**: No need for ODE solvers during training
- **Fast Sampling**: Direct path from noise to data
- **Flexible**: Works with any neural network architecture
- **Stable**: More stable training than diffusion models

## Getting Started

### Installation

1. Create and activate a conda environment:
```bash
conda create -n mini_latent_pd python=3.14
conda activate mini_latent_pd
```

2. Install the package:
```bash
pip install -e .  # Basic installation
pip install -e ".[dev]"  # With development dependencies
```

3. Initialize pre-commit hooks (optional):
```bash
pre-commit install
```

### Quick Training Examples

Train on MNIST (default):
```bash
python src/mini_latent_pd/train.py
```

Quick test with debug mode (1 batch only):
```bash
python src/mini_latent_pd/train.py experiment=debug
```

Train on 2D Moons dataset:
```bash
python src/mini_latent_pd/train.py experiment=moons
```

Override any config parameter:
```bash
python src/mini_latent_pd/train.py \
    data.batch_size=64 \
    model.optimizer.lr=0.0001 \
    trainer.max_epochs=20
```

## Project Structure

```
├── configs/                    # Hydra configuration files
│   ├── train_config.yaml      # Main training config
│   ├── paths_config.yaml      # Path configurations
│   ├── data/                  # Data module configs
│   ├── model/                 # Model configs
│   ├── trainer/               # PyTorch Lightning trainer configs
│   ├── callbacks/             # Callback configs
│   ├── logger/                # Logger configs
│   └── experiment/            # Full experiment configs
├── src/mini_latent_pd/
│   ├── train.py               # Main training script
│   ├── models/                # LightningModules (training logic)
│   │   └── flow_matching.py  # FlowMatching & FlowMatchingCFG
│   ├── models/                # Neural network architectures
│   │   ├── unet.py           # U-Net for images
│   │   └── mlp.py            # MLP for 2D data
│   ├── modules/               # Reusable building blocks
│   │   ├── schedulers.py     # Alpha/beta schedulers
│   │   ├── samplers.py       # Noise samplers
│   │   └── solvers.py        # ODE solvers
│   ├── data/                  # Data modules
│   ├── callbacks/             # Custom callbacks
│   └── util/                  # Utilities
├── tests/                     # Unit tests
├── data/                      # Data directory
└── outputs/                   # Training outputs (logs, checkpoints)
```

## Features

### Flow Matching Models

**Standard Flow Matching**:
```python
from mini_latent_pd.models import FlowMatching

model = FlowMatching(
    model=unet,
    alpha_beta_scheduler=scheduler,
    sampler=sampler,
    ode_solver=solver,
)
```

**Classifier-Free Guidance**:
```python
from mini_latent_pd.models import FlowMatchingCFG

model = FlowMatchingCFG(
    model=unet,
    num_classes=10,
    cfg_prob=0.1,        # 10% unconditional training
    guidance_scale=3.0,  # Guidance strength
)

# Generate with stronger guidance
samples = model.generate_samples(labels, guidance_scale=5.0)
```

### Modular Components

- **Models**: LightningModules for training (FlowMatching, FlowMatchingCFG)
- **Networks**: U-Net for images, MLP for low-dimensional data
- **Modules**: Schedulers (Linear, Cosine, Stable), Samplers (Gaussian), Solvers (Euler, RK4)

## Configuration System

All hyperparameters are managed through Hydra configs:

```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - override /model: default_model
  - override /data: default_data_module

task_name: "my_experiment"

model:
  optimizer:
    lr: 0.001

data:
  batch_size: 128

trainer:
  max_epochs: 10
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/mini_latent_pd

# Run specific test file
pytest tests/test_config.py -v
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy src/
```

## Extending the Template

### Adding a Custom Dataset

1. Create `src/mini_latent_pd/data/my_data.py`
2. Inherit from `L.LightningDataModule`
3. Create config in `configs/data/my_data.yaml`
4. Use with: `python train.py data=my_data`

See `data/README.md` for detailed instructions.

### Adding a Custom Model

1. Create your model in `src/mini_latent_pd/models/`
2. Create config in `configs/model/my_model.yaml`
3. Use with: `python train.py model=my_model`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ mini_latent_pd,
  author = { Moust Holmes },
  title = { mini_latent_pd: A PyTorch Lightning Template for Flow Matching},
  year = {2025},
}
```

## License

MIT License - see LICENSE file for details.

## Author

Moust Holmes
