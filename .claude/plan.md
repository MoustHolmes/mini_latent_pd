# mini_latent_pd: Project Plan

## Goal

Train a single model that does both **protein folding** and **protein dynamics** (MD trajectory prediction). Backbone: **OpenFold3**'s all-atom diffusion architecture. Training data: PDB structures (folding) + mdCATH MD trajectories (dynamics). Key mechanism: conditioning on x0 as a structural template for dynamics mode; no template for folding mode. Same model, same weights — toggleable via `use_template`.

---

## Status

### Data Pipeline ✓ Complete

| File | Status |
|------|--------|
| `src/mini_latent_pd/data/mdcath_base_dataset.py` | ✓ Done |
| `src/mini_latent_pd/data/mdcath_of3_dataset.py` | ✓ Done |
| `src/mini_latent_pd/data/mdcath_atom37_dataset.py` | ✓ Done |
| `src/mini_latent_pd/data/pdb_of3_dataset.py` | ✓ Done |
| `src/mini_latent_pd/data/of3_collator.py` | ✓ Done |
| `src/mini_latent_pd/data/of3_dynamics_datamodule.py` | ✓ Done (+ val_dataloader) |
| `test_of3_smoke.py` | ✓ Passing (forward pass + loss) |
| `scripts/extract_sequences.py` | ✓ Done |
| `scripts/generate_msa.py` | ✓ Done (batch ColabFold API) |
| `notebooks/data_pipeline_walkthrough.ipynb` | ✓ Done |

MSA generation running on HPC.

### Training Module ← Current Phase

| File | Status |
|------|--------|
| `src/mini_latent_pd/of3_diffusion_module.py` | ✓ Done |
| `src/mini_latent_pd/callbacks/rmsd_validation.py` | ✓ Done |
| `configs/train_dynamics.yaml` | ✓ Done |
| `configs/model/of3_dynamics.yaml` | ✓ Done |
| `configs/data/of3_dynamics_datamodule.yaml` | ✓ Done |
| `configs/trainer/of3_gpu_trainer.yaml` | ✓ Done |
| `configs/callbacks/of3_dynamics_callbacks.yaml` | ✓ Done |
| `src/mini_latent_pd/train.py` | ✓ Done (existing, unchanged) |
| Single-GPU smoke test on HPC | TODO |
| Multi-GPU scale-up | TODO |

---

## Architecture

**Training setup:** Hydra + PyTorch Lightning, same pattern as `scalability_playground`.

**Entry point:**
```bash
python src/mini_latent_pd/train.py --config-name=train_dynamics
```

**What we import from OF3 (libraries only, no project infrastructure):**
- `OpenFold3` — the nn.Module (all-atom diffusion architecture)
- `OpenFold3Loss` — loss module (diffusion + distogram + confidence heads)
- `ExponentialMovingAverage` — EMA for model weights
- `AlphaFoldLRScheduler` — cosine-decay LR schedule with warmup
- `model_config` — base `ml_collections.ConfigDict`; deep-copied and overrides applied directly

**What we write ourselves:**
- `OF3DiffusionModule(pl.LightningModule)` — wraps OF3 model + loss + EMA + optimizer
- `OF3DynamicsDataModule` — our mdCATH + PDB data pipeline
- `DynamicsRMSDCallback` — logs Cα RMSD(xt_pred, xt_real) during validation

**Key design choices:**
- No dependency on OF3's training runner (`OpenFold3AllAtom`) or config system
- OF3 model auto-selects train vs. inference mode via `model.train()` / `model.eval()`
  - Train: single-step denoising loss (fast)
  - Validation: full diffusion rollout → `atom_positions_predicted` (use `limit_val_batches=8`)
- EMA weights used during validation only; restored after `on_validation_epoch_end`
- Confidence head losses near-zero by construction: `plddt=1e-4, pae=1e-4, ...` set in `_DEFAULT_LOSS_WEIGHTS` inside each dataset sample

---

## Runtime Commands

### HPC training (single GPU):
```bash
python src/mini_latent_pd/train.py --config-name=train_dynamics \
  data.mdcath_data_dir=/hpc/path/mdcath/data \
  data.msa_dir=/hpc/path/mdcath/msa \
  trainer.devices=1
```

### HPC training (multi-GPU DDP):
```bash
python src/mini_latent_pd/train.py --config-name=train_dynamics \
  data.mdcath_data_dir=/hpc/path/mdcath/data \
  data.msa_dir=/hpc/path/mdcath/msa \
  trainer.devices=4 \
  trainer.strategy=ddp
```

### Mac smoke test (CPU, tiny model):
```bash
python src/mini_latent_pd/train.py --config-name=train_dynamics \
  data.mdcath_data_dir=./data/mdcath/data \
  data.msa_dir=null \
  data.samples_per_epoch=4 \
  data.max_seq_len=100 \
  trainer=default_trainer \
  trainer.accelerator=cpu \
  model.presets=[] \
  "model.of3_overrides={architecture: {pairformer: {no_blocks: 1}, diffusion_module: {diffusion_transformer: {no_blocks: 1}}}}" \
  trainer.max_steps=5 \
  trainer.limit_val_batches=2
```

---

## Next Steps

1. **Smoke test locally** — run the Mac CPU command above to verify the Hydra config resolves and model instantiates
2. **Single-GPU smoke test on HPC** — verify end-to-end training step with real mdCATH data
3. **Multi-GPU scale-up** — DDP with full mdCATH + MSAs
4. **Eventually:** port to an OF3-style project once training is stable

---

## Key Reference Paths

| What | Path |
|------|------|
| Our Lightning module | `src/mini_latent_pd/of3_diffusion_module.py` |
| RMSD callback | `src/mini_latent_pd/callbacks/rmsd_validation.py` |
| Data module | `src/mini_latent_pd/data/of3_dynamics_datamodule.py` |
| Top-level Hydra config | `configs/train_dynamics.yaml` |
| Model config | `configs/model/of3_dynamics.yaml` |
| OF3 runner (reference) | `openfold-3/openfold3/projects/of3_all_atom/runner.py` |
| OF3 loss module | `openfold-3/openfold3/core/loss/loss_module.py` |
| OF3 LR scheduler | `openfold-3/openfold3/core/utils/lr_schedulers.py` |
| Plattito RMSD callback | `scalability_playground/src/callbacks/rmsd_validation.py` |
