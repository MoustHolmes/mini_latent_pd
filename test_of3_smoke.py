"""Smoke test for the OpenFold3 data pipeline.

Test order:
  1. Extract a PDB structure from a local mdCATH HDF5 file.
  2. Load it via PDBOpenFold3Dataset and verify batch keys/shapes.
  3. Load an mdCATH frame pair via MDCATHOpenFold3Dataset (with and
     without template conditioning) and verify shapes.
  4. Run a forward pass through a reduced OpenFold3 model.
  5. Compute the diffusion loss to verify end-to-end correctness.

Run with:
    python test_of3_smoke.py
"""

import sys
import tempfile
from pathlib import Path

import h5py
import torch

MDCATH_DIR = Path("data/mdcath/data")
FIRST_H5 = sorted(MDCATH_DIR.glob("*.h5"))[0]

sys.path.insert(0, "src")
sys.path.insert(0, "openfold-3")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_pdb_from_hdf5(h5_path: Path) -> tuple[str, Path]:
    """Extract the PDB text from an mdCATH HDF5 and save to a temp file."""
    with h5py.File(h5_path, "r") as f:
        domain_id = list(f.keys())[0]
        pdb_text = f[domain_id]["pdbProteinAtoms"][()].decode("utf-8")

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp.write(pdb_text.encode())
    tmp.flush()
    return domain_id, Path(tmp.name)


def build_debug_model():
    """Build a reduced OpenFold3 model suitable for CPU smoke testing."""
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    project_entry = OF3ProjectEntry()
    config = project_entry.get_model_config_with_presets()

    # Reduce to minimal size for CPU testing
    config.architecture.pairformer.no_blocks = 1
    config.architecture.diffusion_module.diffusion_transformer.no_blocks = 1

    # Disable DeepSpeed kernels (not available on CPU / without DeepSpeed install)
    config.settings.memory.train.use_deepspeed_evo_attention = False
    config.settings.memory.eval.use_deepspeed_evo_attention = False

    runner = project_entry.runner(config)
    runner.eval()
    return runner, config


def add_batch_dim(sample: dict) -> dict:
    """Add a leading batch dimension to all tensors in a sample dict."""
    out = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            out[k] = add_batch_dim(v)
        elif isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pdb_dataset(pdb_path: Path) -> dict:
    print("\n--- Test 1: PDB → OF3 batch ---")
    from mini_latent_pd.data.pdb_of3_dataset import PDBOpenFold3Dataset

    ds = PDBOpenFold3Dataset(pdb_paths=[pdb_path])
    assert len(ds) == 1, "Expected 1 sample"
    sample = ds[0]

    required_keys = {
        "restype", "ref_pos", "ref_mask", "ref_element", "ref_atom_name_chars",
        "msa", "token_mask", "atom_mask", "num_atoms_per_token",
        "start_atom_index", "atom_to_token_index", "template_distogram",
        "ground_truth", "loss_weights",
    }
    missing = required_keys - set(sample.keys())
    assert not missing, f"Missing keys: {missing}"

    L = sample["token_mask"].shape[0]
    n_atom = int(sample["num_atoms_per_token"].sum().item())
    print(f"  L={L}, N_atom={n_atom}")
    print(f"  ref_pos shape: {sample['ref_pos'].shape}")
    print(f"  restype shape: {sample['restype'].shape}")
    print(f"  msa shape: {sample['msa'].shape}")
    assert sample["ref_pos"].shape == (n_atom, 3)
    assert sample["restype"].shape == (L, 32)
    print("  ✓ PDB dataset OK")
    return sample


def test_mdcath_dataset(with_template: bool) -> dict:
    label = "with template" if with_template else "no template"
    print(f"\n--- Test 2: mdCATH OF3 dataset ({label}) ---")
    from mini_latent_pd.data.mdcath_of3_dataset import MDCATHOpenFold3Dataset

    ds = MDCATHOpenFold3Dataset(
        data_dir=MDCATH_DIR,
        use_template=with_template,
        samples_per_epoch=2,
        max_lag=50,
        max_seq_len=100,  # keep small for CPU smoke test
    )
    assert len(ds) > 0, "No proteins indexed"
    sample = ds[0]

    L = sample["token_mask"].shape[0]
    n_atom = int(sample["num_atoms_per_token"].sum().item())
    print(f"  L={L}, N_atom={n_atom}")
    print(f"  template_distogram shape: {sample['template_distogram'].shape}")
    assert sample["template_distogram"].shape == (1, L, L, 39)
    assert sample["ground_truth"]["atom_positions"].shape == (n_atom, 3)
    print(f"  ✓ mdCATH dataset ({label}) OK")
    return sample


def test_forward_pass(sample: dict) -> None:
    print("\n--- Test 3: Forward pass through reduced OF3 model ---")
    runner, config = build_debug_model()

    batch = add_batch_dim(sample)

    with torch.no_grad():
        batch, outputs = runner(batch=batch)

    atom_pos = outputs["atom_positions_predicted"]
    print(f"  atom_positions_predicted shape: {atom_pos.shape}")
    assert atom_pos.ndim == 4, "Expected (B, samples, N_atom, 3)"
    print("  ✓ Forward pass OK")


def test_loss(sample: dict) -> None:
    print("\n--- Test 4: Loss computation ---")
    from openfold3.core.loss.loss_module import OpenFold3Loss

    runner, config = build_debug_model()
    of3_loss = OpenFold3Loss(config=config.architecture.loss_module)

    batch = add_batch_dim(sample)

    runner.train()
    batch, outputs = runner(batch=batch)
    loss, _ = of3_loss(batch=batch, output=outputs, _return_breakdown=True)

    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    print(f"  Loss = {loss.item():.4f}")
    print("  ✓ Loss computation OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Using HDF5 file: {FIRST_H5}")

    # Step 1: Extract PDB from HDF5
    domain_id, pdb_path = extract_pdb_from_hdf5(FIRST_H5)
    print(f"Extracted PDB for {domain_id} → {pdb_path}")

    # Step 2: PDB dataset test
    pdb_sample = test_pdb_dataset(pdb_path)

    # Step 3: mdCATH dataset tests
    mdcath_sample_templ = test_mdcath_dataset(with_template=True)
    mdcath_sample_no_templ = test_mdcath_dataset(with_template=False)

    # Step 4: Forward pass (use mdCATH sample with template)
    test_forward_pass(mdcath_sample_templ)

    # Step 5: Loss computation
    test_loss(mdcath_sample_templ)

    print("\n✓ All smoke tests passed!")
    pdb_path.unlink(missing_ok=True)
