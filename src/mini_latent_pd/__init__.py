"""mini_latent_pd - A PyTorch Lightning project for Flow Matching."""

from mini_latent_pd.flow_matching_module import (
    FlowMatching,
    FlowMatchingCFG,
    LatentFlowMatching,
    LatentFlowMatchingCFG,
    DigitTransportFlowMatching,
)
from mini_latent_pd.distance_flow_module import DistanceFlowModule

__version__ = "0.1.0"

__all__ = [
    "FlowMatching",
    "FlowMatchingCFG",
    "LatentFlowMatching",
    "LatentFlowMatchingCFG",
    "DigitTransportFlowMatching",
    "DistanceFlowModule",
]
