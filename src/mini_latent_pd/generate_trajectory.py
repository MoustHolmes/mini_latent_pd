"""Generate MD trajectories using a trained distance-flow model.

Usage:
    python -m mini_latent_pd.generate_trajectory \
        --checkpoint path/to/distance_flow.ckpt \
        --data_dir data/mdcath/data \
        --embedding_dir data/embeddings \
        --n_steps 100 \
        --output_dir outputs/trajectories
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from mini_latent_pd.distance_flow_module import DistanceFlowModule
from mini_latent_pd.data.mdcath_minifold_dataset import MDCathToMinifoldDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MD trajectories")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained DistanceFlowModule checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/mdcath/data", help="mdCATH HDF5 directory")
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings", help="ESM embeddings directory")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of rollout steps")
    parser.add_argument("--protein_idx", type=int, default=0, help="Index of protein to simulate")
    parser.add_argument("--output_dir", type=str, default="outputs/trajectories", help="Output directory")
    parser.add_argument("--re_encode", action="store_true", default=True, help="Re-encode coords each step")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load model
    module = DistanceFlowModule.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    module.eval()
    module.to(device)

    # Load dataset (just for indexing — we only need one protein's starting frame)
    dataset = MDCathToMinifoldDataset(
        data_dir=args.data_dir,
        embedding_dir=args.embedding_dir,
        samples_per_epoch=1,
        max_lag=1,
    )

    prot = dataset.proteins[args.protein_idx]
    print(f"Protein: {prot['id']}, L={prot['n_residues']}")

    # Get starting frame
    temp, repl, n_frames = prot["traj_index"][0]
    with h5py.File(prot["h5_path"], "r") as f:
        from mini_latent_pd.data.mdcath_utils import flat_coords_to_atom37
        coords = f[prot["id"]][temp][repl]["coords"][0]
        atom37 = flat_coords_to_atom37(
            coords,
            prot["heavy_mask"],
            prot["valid_heavy_idx"],
            prot["res_scatter"],
            prot["atom37_scatter"],
            prot["n_residues"],
        )

    # Prepare tensors (batch dim = 1)
    x0 = torch.from_numpy(atom37).unsqueeze(0).to(device)
    aatype = prot["aatype"].unsqueeze(0).to(device)
    seq_mask = torch.ones(1, prot["n_residues"], device=device)

    # Load embeddings
    emb = torch.load(prot["embedding_path"], map_location=device, weights_only=True)
    s_s = emb["s_s"].unsqueeze(0).to(device)
    s_z = emb["s_z"].unsqueeze(0).to(device)

    # Run rollout
    print(f"Generating {args.n_steps}-step trajectory...")
    trajectory = module.rollout(
        atom37_pos=x0,
        aatype=aatype,
        seq_mask=seq_mask,
        s_s=s_s,
        s_z=s_z,
        n_steps=args.n_steps,
        re_encode=args.re_encode,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract Cα coordinates for easy analysis
    ca_idx = 1  # atom37 Cα index
    mask = seq_mask[0].bool().cpu()

    ca_traj = []
    for frame in trajectory:
        ca = frame[0, mask, ca_idx].cpu().numpy()
        ca_traj.append(ca)
    ca_traj = np.stack(ca_traj)  # (n_steps, L, 3)

    # Save full atom37 trajectory
    atom37_traj = torch.cat(trajectory, dim=0).cpu()  # (n_steps, L, 37, 3)

    out_path = output_dir / f"{prot['id']}_trajectory.pt"
    torch.save(
        {
            "protein_id": prot["id"],
            "n_residues": prot["n_residues"],
            "aatype": prot["aatype"],
            "atom37_trajectory": atom37_traj,
            "ca_trajectory": ca_traj,
            "starting_frame": x0.cpu(),
            "n_steps": args.n_steps,
        },
        out_path,
    )
    print(f"Saved trajectory to {out_path}")
    print(f"  Shape: {atom37_traj.shape} (steps, L, 37, 3)")
    print(f"  Cα trajectory: {ca_traj.shape} (steps, L, 3)")


if __name__ == "__main__":
    main()
