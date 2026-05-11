# OpenFold3 Codebase Overview

**Location:** `openfold-3/`  
**What it is:** Bitwise reproduction of DeepMind's AlphaFold3 by the AlQuraishi Lab. Supports proteins, RNA, DNA, and small molecules. Full training + inference pipeline with PyTorch Lightning.

---

## Directory Structure

```
openfold-3/
├── openfold3/
│   ├── core/                          # Model, data, loss, runners, utils
│   │   ├── data/                      # Data pipeline (~90 files)
│   │   │   ├── framework/             # DataModule, SingleDataset, samplers
│   │   │   ├── pipelines/             # Featurization, preprocessing
│   │   │   ├── primitives/            # Caching, clustering, tokenization
│   │   │   ├── resources/             # Residue/atom constants
│   │   │   └── tools/                 # MSA parsing, ColabFold tools
│   │   ├── model/                     # Layers, heads, embedders (~30 files)
│   │   │   ├── feature_embedders/     # Input, template, MSA embedders
│   │   │   ├── heads/                 # Confidence heads (pLDDT, PAE, PDE)
│   │   │   ├── primitives/            # Attention, layer norms, transitions
│   │   │   └── structure/             # DiffusionModule, SampleDiffusion
│   │   ├── loss/                      # Loss functions (4 files)
│   │   ├── runners/                   # ModelRunner (Lightning base class)
│   │   ├── utils/                     # LR schedulers, grad manager, geometry
│   │   └── metrics/                   # Quality metrics (LDDT, PAE, RMSD)
│   ├── entry_points/                  # Training/inference CLI + config validation
│   ├── projects/of3_all_atom/         # All-atom project (main model)
│   │   ├── config/                    # model_config.py, dataset_configs.py
│   │   ├── model.py                   # OpenFold3 nn.Module
│   │   ├── runner.py                  # OpenFold3AllAtom (Lightning module)
│   │   └── project_entry.py           # OF3ProjectEntry (factory)
│   └── run_openfold.py                # CLI entry point
├── examples/
│   ├── training_yamls/                # Training config examples
│   └── reference_full_config/         # Full reference YAML
└── scripts/                           # Data preprocessing utilities
```

---

## Training Entry Point

```bash
run_openfold train --runner-yaml configs/training.yaml --seed 42
```

**Files involved:**
- `run_openfold.py` → `entry_points/experiment_runner.py:TrainingExperimentRunner`
- Config loaded via `entry_points/validator.py:TrainingExperimentConfig` (Pydantic)
- Runner instantiated via `projects/of3_all_atom/project_entry.py:OF3ProjectEntry`

---

## Config System

**Format:** YAML → Pydantic validation → ml_collections ConfigDict

### Training YAML Structure (`examples/training_yamls/initial_training.yml`)
```yaml
experiment_settings:
  mode: train
  output_dir: ./output
  restart_checkpoint_path: last   # resume from last checkpoint
  seed: 42

data_module_args:
  batch_size: 1
  num_workers: 16
  epoch_len: 128000               # steps per epoch (triggers checkpoint)

logging_config:
  log_lr: true
  wandb_config:
    project: <name>
    entity: <entity>

pl_trainer_args:
  devices: 8                      # GPUs per node
  num_nodes: 32
  precision: bf16-mixed
  max_epochs: -1
  log_every_n_steps: 50
  deepspeed_config_path: null

checkpoint_config:
  every_n_epochs: 1
  save_last: true
  save_top_k: -1

model_update:
  presets: [train]               # applies train preset from model_setting_presets.yml
  custom:                        # overrides on top of preset
    architecture:
      shared:
        use_confidence_emb_prob: 0.8

dataset_configs:
  train:
    weighted-pdb:
      dataset_class: WeightedPDBDataset
      weight: 0.5
      config:
        template: {n_templates: 4}
        crop:
          token_crop: {enabled: true, token_budget: 384}

dataset_paths:
  weighted-pdb:
    alignment_array_directory: /path/to/alignment_arrays
    dataset_cache_file: /path/to/cache.json
    target_structures_directory: /path/to/structures
    reference_molecule_directory: /path/to/ref_mols
    template_cache_directory: /path/to/templates
```

### Model Config (`projects/of3_all_atom/config/model_config.py`)

Key hidden dimensions:
- `c_s = 384` — single (token) representation
- `c_z = 128` — pair representation
- `c_m = 64` — MSA representation
- `c_atom = 128` — atom features
- `sigma_data = 16` — diffusion data variance

Config sections:
```
settings:
  memory: chunking, deepspeed_evo_attention toggle
  gradient_clipping: per_sample_clipping, clip_val
  ema: decay rate
  train_confidence_only: bool (freeze all but confidence heads)

architecture:
  input_embedder, template, msa, pairformer
  diffusion_module, sample_diffusion
  loss_module:
    loss_weights: {bond, smooth_lddt, mse, plddt, pde, pae, experimentally_resolved, distogram}
  heads: plddt, pae, pde, experimentally_resolved, distogram
```

**For dynamics training, set in `model_update.custom`:**
```yaml
architecture:
  loss_module:
    loss_weights:
      plddt: 0.0
      pae: 0.0
      pde: 0.0
      experimentally_resolved: 0.0
```

---

## Lightning Module: `OpenFold3AllAtom`

**File:** `openfold-3/openfold3/projects/of3_all_atom/runner.py`  
**Inherits:** `ModelRunner(pl.LightningModule)` in `core/runners/model_runner.py`

### Key Methods
```python
class OpenFold3AllAtom(ModelRunner):
    def __init__(self, model_config, log_dir=None):
        # Sets up: OpenFold3Loss, EMA, per-sample grad clipping

    def setup(self, stage):
        # Initializes train/val metrics, optionally freezes non-confidence params

    def training_step(self, batch, batch_idx):
        # Calls self(batch) → loss, logs metrics

    def validation_step(self, batch, batch_idx):
        # Forward pass (no diffusion sampling), computes LDDT/PAE metrics

    def configure_optimizers(self):
        # AdamW + AlphaFoldLRScheduler (cosine + warmup)
        # from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler

    def on_validation_epoch_end(self):
        # Aggregates val metrics, computes model selection score

    def _setup_train_metrics(self):
        # MeanMetric for each loss in TRAIN_LOSSES + TRAIN_LOGGED_METRICS

    def _setup_val_metrics(self):
        # MeanMetric for each loss in VAL_LOSSES + VAL_LOGGED_METRICS
```

**`ModelRunner.__call__`** (base class, `core/runners/model_runner.py`):
- Calls `self.model.forward(batch)` → `(batch, output_dict)`
- Handles EMA weight swapping during validation

### Tracked Metrics
Training: `mse_loss`, `smooth_lddt_loss`, `bond_loss`, `distogram_loss`, `plddt_loss`, `pde_loss`, `pae_loss`, `experimentally_resolved_loss`  
Validation: same + quality metrics (LDDT, PAE, clashes, dRMSD)  
Model selection: `compute_final_model_selection_metric()` from `core/metrics/model_selection.py`

---

## Model Architecture: `OpenFold3`

**File:** `openfold-3/openfold3/projects/of3_all_atom/model.py`

```
OpenFold3(nn.Module)
├── InputEmbedderAllAtom          — token + atom feature embedding
├── TemplateEmbedderAllAtom       — structural template processing
├── MSAModuleEmbedder             — MSA embedding
├── MSAModuleStack                — Evoformer MSA stack
├── PairFormerStack               — pair representation updates (O(L²))
├── DiffusionModule               — structure diffusion sampling
└── AuxiliaryHeadsAllAtom         — pLDDT, PAE, PDE, distogram, experimentally_resolved
```

**Forward pass:** `run_trunk()` → embeddings → N pairformer iterations → `_rollout()` (diffusion) → heads

---

## Diffusion Module

**File:** `openfold-3/openfold3/core/model/structure/diffusion_module.py`

Implements AF3 Algorithm 3.7. Key components:
- `DiffusionModule(nn.Module)` — denoising transformer for atom positions
- `SampleDiffusion` — inference-time rollout (not used in training loss)
- `centre_random_augmentation()` — random rotation applied to `ref_pos`
- `create_noise_schedule()` — AF3 noise schedule (σ_data=16)

**During training:** loss computed via `OpenFold3Loss.diffusion_loss()` — adds noise to ground truth positions and computes denoising loss. Does NOT run full rollout.  
**During inference/validation:** `_rollout()` runs full diffusion to generate coordinates.

---

## Loss Module

**File:** `openfold-3/openfold3/core/loss/loss_module.py`

```python
class OpenFold3Loss(nn.Module):
    def forward(self, batch, output, _return_breakdown=False):
        # Returns scalar loss (+ breakdown dict if requested)
        # Aggregates:
        #   diffusion_loss() → mse, smooth_lddt, bond
        #   confidence_loss() → plddt, pae, pde, experimentally_resolved
        #   all_atom_distogram_loss() → distogram
```

Loss weights come from `model_config.architecture.loss_module.loss_weights`.  
**For dynamics: set plddt/pae/pde/experimentally_resolved weights to 0.**

---

## Data Pipeline (OF3's own)

**Not used directly in our project** — we supply our own `OF3DynamicsDataModule`. Documented here for reference.

**File:** `openfold-3/openfold3/core/data/framework/data_module.py:DataModule`

Dataset classes:
- `WeightedPDBDataset` — PDB structures (main training data)
- `ProteinMonomerDataset` — protein distillation data
- `RNAMonomerDataset` — RNA distillation
- `DisorderedPDBDataset` — disordered region data
- `ValidationPDBDataset` — validation set

Data format: structures stored as `.npz` files (preprocessed), MSAs as alignment arrays, templates as separate cache.

**Our replacement:** `src/mini_latent_pd/data/of3_dynamics_datamodule.py` wraps `MDCATHOpenFold3Dataset` and `PDBOpenFold3Dataset`.

---

## Relevant Utilities

| Utility | Path | Purpose |
|---------|------|---------|
| `AlphaFoldLRScheduler` | `core/utils/lr_schedulers.py` | Cosine + warmup LR schedule |
| `PerSampleGradManager` | `core/utils/grad_manager.py` | Per-sample gradient clipping |
| `get_metrics` | `core/metrics/quality.py` | LDDT, dRMSD, clash metrics |
| `get_confidence_scores` | `core/metrics/aggregate_confidence_ranking.py` | Model selection scores |
| `TOKEN_NAME_TO_ATOM_NAMES` | `core/data/resources/token_atom_constants.py` | Canonical atom order per residue |
| `STANDARD_RESIDUES_WITH_GAP_3` | `core/data/resources/residues.py` | 32-class residue vocabulary |
| `parse_a3m` | `core/data/tools/parse_msa_files.py` | MSA file parsing |

---

## OF3 Batch Format (relevant keys for dynamics)

```python
{
  # Token-level (L,)
  "residue_index", "token_index", "asym_id", "entity_id", "sym_id",
  "restype",           # (L, 32) one-hot
  "is_protein",        # (L,) int
  "token_mask",        # (L,) float
  "mol_entity_id", "mol_sym_id", "mol_sym_component_id", "mol_sym_token_index",  # permutation alignment

  # Atom-level (N_atom,)
  "ref_pos",           # (N_atom, 3) x0 coords as reference conformer
  "ref_mask",          # (N_atom,) int
  "ref_element",       # (N_atom, 119) one-hot atomic number - 1
  "ref_charge",        # (N_atom,) float
  "ref_atom_name_chars", # (N_atom, 4, 64) one-hot atom name chars
  "ref_space_uid",     # (N_atom,) atom→token index
  "atom_mask",         # (N_atom,) float
  "start_atom_index",  # (L,) int
  "num_atoms_per_token", # (L,) int
  "atom_to_token_index", # (N_atom,) int

  # MSA (N_msa, L, *)
  "msa", "has_deletion", "deletion_value", "profile", "deletion_mean",
  "msa_mask", "num_paired_seqs",

  # Template (1, L, *) or (1, L, L, *)
  "template_restype", "template_pseudo_beta_mask", "template_backbone_frame_mask",
  "template_distogram",    # (1, L, L, 39)
  "template_unit_vector",  # (1, L, L, 3)

  # Ground truth
  "ground_truth": {
    "atom_positions",        # (N_atom, 3) ← xt frame for dynamics
    "atom_resolved_mask",    # (N_atom,)
    "token_mask", "atom_mask", "num_atoms_per_token",
    "is_ligand", "mol_entity_id", "mol_sym_id", "mol_sym_component_id", "mol_sym_token_index",
  },

  # Loss weights (scalar per sample)
  "loss_weights": {bond, smooth_lddt, mse, plddt, pae, pde, experimentally_resolved, distogram},
}
```

---

## Mac Debug Config (CPU testing)

```python
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
config = OF3ProjectEntry().get_model_config_with_presets()
config.architecture.pairformer.no_blocks = 1
config.architecture.diffusion_module.diffusion_transformer.no_blocks = 1
config.settings.memory.train.use_deepspeed_evo_attention = False
config.settings.memory.eval.use_deepspeed_evo_attention = False
runner = OF3ProjectEntry().runner(config)
```
