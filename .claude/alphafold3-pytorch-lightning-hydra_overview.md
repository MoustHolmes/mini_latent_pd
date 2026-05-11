# alphafold3-pytorch-lightning-hydra Overview

**Location:** `alphafold3-pytorch-lightning-hydra/`  
**What it is:** PyTorch Lightning + Hydra implementation of AlphaFold3. A fork of an open-source AF3 implementation (now evolved into MegaFold). Uses the same Hydra/Lightning template as Plattito.

**Relevance to mini_latent_pd:** Reference for how to structure a Hydra-based training setup for an AlphaFold-style model. If we later add a Hydra entry point (Option B in the training plan), this is the template to follow.

---

## Architecture

```
alphafold3-pytorch-lightning-hydra/
├── alphafold3_pytorch/
│   ├── models/
│   │   ├── alphafold3_module.py    # Alphafold3LitModule (pl.LightningModule)
│   │   └── components/
│   │       └── alphafold3.py       # Alphafold3 nn.Module
│   ├── data/
│   │   ├── pdb_datamodule.py       # PDBDataModule (main)
│   │   ├── atom_datamodule.py      # AtomDataModule
│   │   └── components/             # BatchedAtomInput, data_pipeline.py
│   ├── train.py                    # @hydra.main entry point
│   └── eval.py                     # Evaluation entry point
└── configs/
    ├── train.yaml                  # Default config
    ├── model/                      # Model architecture configs
    ├── data/                       # DataModule configs
    ├── trainer/                    # gpu.yaml, cpu.yaml, ddp.yaml
    ├── callbacks/                  # ModelCheckpoint, EMA, LRMonitor
    ├── logger/                     # wandb.yaml, tensorboard.yaml, etc.
    ├── experiment/                 # Full experiment configs
    ├── strategy/                   # DDP, DeepSpeed strategies
    └── paths/default.yaml          # Data paths
```

---

## Lightning Module Pattern

**File:** `alphafold3_pytorch/models/alphafold3_module.py:Alphafold3LitModule`

```python
class Alphafold3LitModule(pl.LightningModule):
    def __init__(self, net, optimizer, scheduler, compile=False):
        self.net = net               # Alphafold3 nn.Module
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def training_step(self, batch, batch_idx):
        loss = self.net(batch)
        self.train_loss(loss)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.net(batch)
        self.val_loss(loss)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}
```

---

## Hydra Config Pattern

**Entry:** `alphafold3_pytorch/train.py`

```python
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def train(cfg: DictConfig):
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
```

**Command line overrides:**
```bash
python alphafold3_pytorch/train.py trainer=gpu
python alphafold3_pytorch/train.py experiment=alphafold3_initial_training.yaml
python alphafold3_pytorch/train.py trainer.max_steps=1e6 data.batch_size=2
```

---

## Hydra Config Structure (relevant if we add Hydra)

`configs/train.yaml`:
```yaml
defaults:
  - _self_
  - data: pdb
  - model: alphafold3
  - callbacks: default
  - logger: wandb
  - trainer: gpu
  - paths: default

task_name: train
tags: [dev]
seed: 42
train: true
test: false
ckpt_path: null
```

`configs/experiment/alphafold3_initial_training.yaml`:
```yaml
# @package _global_
defaults:
  - override /trainer: gpu

trainer:
  max_steps: 1_000_000
  precision: bf16-mixed

data:
  batch_size: 1
  num_workers: 8

model:
  optimizer:
    lr: 1.8e-3
```

---

## Available Callbacks

```yaml
# configs/callbacks/default.yaml
model_checkpoint:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  monitor: val/loss
  save_top_k: 3
  save_last: true

ema:
  _target_: ...
  decay: 0.999

learning_rate_monitor:
  _target_: lightning.pytorch.callbacks.LearningRateMonitor
  logging_interval: step

model_summary:
  _target_: lightning.pytorch.callbacks.ModelSummary
  max_depth: 1
```

---

## Available Distributed Strategies

```yaml
# configs/strategy/ddp.yaml
_target_: lightning.pytorch.strategies.DDPStrategy
find_unused_parameters: false

# configs/strategy/deepspeed.yaml
_target_: lightning.pytorch.strategies.DeepSpeedStrategy
stage: 2
```

---

## Key Takeaways for mini_latent_pd

If we add a Hydra entry point later:
1. Use `hydra.utils.instantiate` for all components (model, data, callbacks, loggers)
2. Separate configs into `model/`, `data/`, `callbacks/`, `trainer/`, `logger/`, `paths/`
3. Use `experiment/` configs to compose full training runs
4. Keep `paths/default.yaml` for all data directory paths
5. The `instantiate_callbacks()` and `instantiate_loggers()` utility functions (from `alphafold3_pytorch/utils/`) are worth copying

**Current approach (Option A):** Use OF3's own `run_openfold` CLI with a custom runner YAML. Simpler, no Hydra dependency, but less flexible config composition.
