"""Quick smoke test for new distance-flow modules."""
import sys
sys.path.insert(0, "src")

from mini_latent_pd.models.velocity_net import DistanceVelocityNet, PairResBlock
from mini_latent_pd.utils.geometry import coords_to_bin_indices, coords_to_distogram
print("velocity_net + geometry: OK")

from mini_latent_pd.models.minifold_decoder import MiniFoldDecoder
print("minifold_decoder: OK")

from mini_latent_pd.distance_flow_module import DistanceFlowModule
print("distance_flow_module: OK")

from mini_latent_pd.data.mdcath_flow_datamodule import MDCathFlowDataModule
print("mdcath_flow_datamodule: OK")

# Quick shape test for velocity net
import torch
net = DistanceVelocityNet(num_bins=64, d_model=64, n_layers=4, c_z_static=128)
B, L = 2, 50
d_idx = torch.randint(0, 64, (B, L, L))
s_z_st = torch.randn(B, L, L, 128)
pair_mask = torch.ones(B, L, L)
out = net(d_idx, s_z_st, pair_mask)
print(f"velocity_net output: {out.shape}")
assert out.shape == (B, L, L), f"Expected (2, 50, 50), got {out.shape}"
# Check symmetry
assert torch.allclose(out, out.transpose(1, 2), atol=1e-6), "Not symmetric!"
print("symmetry check: OK")

# Count params
n_params = sum(p.numel() for p in net.parameters())
print(f"velocity_net params: {n_params:,}")

# Test DistanceFlowModule instantiation
module = DistanceFlowModule(velocity_net=net)
print(f"DistanceFlowModule created: OK")

print("\nAll imports and shape checks passed!")
