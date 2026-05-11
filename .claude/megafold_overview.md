# MegaFold Overview

**Location:** `MegaFold/`  
**What it is:** Cross-platform system to accelerate AlphaFold3 training and inference via Triton-based kernel optimizations. Achieves 3.36× longer sequence support, 1.73× training speedup, 1.23× memory reduction. Based on AlphaFold3 architecture with custom CUDA/Triton kernels.

**Relevance to mini_latent_pd:** Reference only — kernel optimizations to consider when scaling training on the HPC. The architecture mirrors OpenFold3 (same AF3 model), so MegaFold's optimizations could in principle be applied to OF3.

---

## Key Optimizations

### 1. EvoFlash-3D (`megafold/model/FusedEvoAttention/`)
Memory-efficient 3D attention kernel for the Evoformer stack. Supports:
- Single attention with pair bias
- Triangle attention around starting/ending nodes
- MSA row-wise and column-wise attention

Implemented with Triton (works on both NVIDIA and AMD). Replaces the standard flash-attention for pairwise operations.

### 2. EvoSP-3D (`megafold/distributed/`)
Communication-efficient sharding for 2D pairwise representations (`(L×L)` pair tensors get split across GPUs). Applied to MSAModule, PairformerStack, DiffusionModule. Key ops: `scatter`, `gather`, `all_to_all`.

### 3. EvoFusion (`megafold/model/FusedLayernormLinear/`, `FusedTransition/`)
Fused operator kernels:
- `LayernormLinear` — fuses LayerNorm + Linear into one kernel
- `FusedTransition` — fuses AF3's Transition layer (reduces memory bandwidth)

### 4. EvoPipe (`megafold/inputs.py`)
Ahead-of-time caching for data pipeline. Deterministic input feature generation cached to disk (avoids repeated preprocessing).

---

## Training Setup (different from OF3)

MegaFold uses a **custom trainer** (not PyTorch Lightning) backed by **Lightning Fabric** + **DeepSpeed**:

```bash
deepspeed --num_gpus=2 train.py --config configs/megafold_1x2.yaml --trainer_name initial_training
```

Config format (`configs/megafold_1x2.yaml`):
```yaml
training_order: [initial_training]
initial_training:
  num_train_steps: 1000
  batch_size: 1
  # model dims, kernel toggles, etc.
```

**Not compatible** with OF3's training pipeline — different config system, different trainer.

---

## What to Take from MegaFold

1. **EvoFlash-3D** — if pairformer attention becomes the bottleneck on long sequences (>512 tokens), these Triton kernels could replace OF3's attention ops. OF3 already has hooks for DeepSpeed EvoAttention; MegaFold's Triton kernels are an alternative.

2. **EvoFusion** — fused LayerNorm+Linear could reduce memory overhead in the pairformer stack.

3. **EvoSP-3D** — relevant if training on very long sequences (>1024 tokens) requiring tensor parallelism across GPUs.

**When to care:** Only after basic multi-GPU training is working and you've identified specific memory/compute bottlenecks. For mdCATH proteins (L ≈ 50–500), standard OF3 with bf16 should be sufficient initially.
