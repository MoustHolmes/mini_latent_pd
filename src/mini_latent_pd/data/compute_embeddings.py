"""
Pre-compute ESM embeddings (s_s, s_z) for mdCATH proteins.

Runs the frozen ESM language model + fc_s/fc_z projections from a trained
MiniFold checkpoint, saving one .pt file per protein:
    {domain_id}.pt  →  {"s_s": (L, 1024), "s_z": (L, L, 128)}

Usage:
    python -m mini_latent_pd.data.compute_embeddings \
        --checkpoint path/to/minifold.ckpt \
        --data_dir path/to/mdcath/h5s \
        --output_dir path/to/embeddings \
        --max_seq_len 900
"""

import argparse
from pathlib import Path

import h5py
import torch

from mini_latent_pd.data.mdcath_utils import (
    aatype_from_resnames,
    aatype_to_sequence,
    decode_bytes_array,
    parse_atom_names_from_pdb,
)


def compute_esm_embeddings(
    model,
    data_dir: str | Path,
    output_dir: str | Path,
    max_seq_len: int = 900,
    device: str = "cpu",
):
    """
    Pre-compute and save s_s / s_z embeddings for all proteins in data_dir.

    Args:
        model: A loaded MiniFold model (with .lm, .fc_s, .fc_z attributes).
        data_dir: Directory containing mdCATH .h5 files.
        output_dir: Directory to save {domain_id}.pt files.
        max_seq_len: Skip proteins longer than this.
        device: Device to run inference on.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alphabet = model.lm.alphabet
    model = model.to(device)
    model.eval()

    h5_files = sorted(data_dir.glob("**/*.h5"))
    print(f"Found {len(h5_files)} HDF5 files in {data_dir}")

    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            domain_id = list(f.keys())[0]
            domain = f[domain_id]

            # Skip if already computed
            out_path = output_dir / f"{domain_id}.pt"
            if out_path.exists():
                print(f"  Skipping {domain_id} (already exists)")
                continue

            # Extract sequence
            z = domain["z"][:]
            resids = domain["resid"][:]
            resnames = decode_bytes_array(domain["resname"][:])
            atom_names = parse_atom_names_from_pdb(
                domain["pdbProteinAtoms"][()].decode("utf-8")
            )

            heavy_mask = z > 1
            heavy_resnames = resnames[heavy_mask]
            heavy_resids = resids[heavy_mask]

            aatype, _, _ = aatype_from_resnames(heavy_resnames, heavy_resids)
            num_res = len(aatype)

            if num_res > max_seq_len:
                print(f"  Skipping {domain_id} (L={num_res} > {max_seq_len})")
                continue

            sequence = aatype_to_sequence(aatype)

        # Tokenize
        tokens = torch.tensor(
            [alphabet.encode(sequence)], dtype=torch.long, device=device
        )

        # Run ESM + projections (no grad needed)
        with torch.no_grad():
            idx = model.lm.num_layers
            lm_out = model.lm(tokens, repr_layers=[idx], need_head_weights=True)

            s_s = lm_out["representations"][idx]
            s_s = model.fc_s(s_s)  # (1, L, 1024)

            B, N = s_s.shape[:2]
            s_z = lm_out["attentions"].view(B, N, N, -1)
            s_z = model.fc_z(s_z)  # (1, L, L, 128)

        # Save without batch dimension, on CPU
        torch.save(
            {"s_s": s_s[0].cpu(), "s_z": s_z[0].cpu()},
            out_path,
        )
        print(f"  Saved {domain_id}: s_s={tuple(s_s.shape[1:])}, s_z={tuple(s_z.shape[1:])}")

    print(f"Done. Embeddings saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute ESM embeddings for mdCATH proteins")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MiniFold checkpoint (.ckpt)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with mdCATH .h5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pt embeddings")
    parser.add_argument("--max_seq_len", type=int, default=900, help="Skip proteins longer than this")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    args = parser.parse_args()

    # Load MiniFold model from checkpoint
    from mini_latent_pd.models.minifold.model.model import MiniFoldModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})

    model = MiniFoldModel(
        esm_model_name=hparams.get("esm_model_name", "esm2_t36_3B_UR50D"),
        num_blocks=hparams.get("num_blocks", 4),
        no_bins=hparams.get("no_bins", 64),
    )

    # Load weights (handle Lightning state_dict prefix)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k: v for k, v in state_dict.items() if "boundaries" not in k}
    state_dict = {k: v for k, v in state_dict.items() if "mid_points" not in k}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    compute_esm_embeddings(
        model=model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
