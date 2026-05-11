"""DynamicsRMSDCallback: logs RMSD between predicted and ground-truth xt.

Collects per-batch predicted and real atom positions during validation,
computes Kabsch-aligned RMSD over all resolved heavy atoms, and logs
mean values (overall and per temperature) at epoch end.
"""

import logging
from collections import defaultdict

import torch
import lightning.pytorch as pl

logger = logging.getLogger(__name__)


def _kabsch_align(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return A rotated to minimise RMSD with B (both (N, 3), already centered)."""
    H = A.T @ B
    U, _, Vt = torch.linalg.svd(H)
    d = torch.linalg.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=A.device, dtype=A.dtype))
    R = Vt.T @ D @ U.T
    return A @ R.T


def kabsch_rmsd(pred: torch.Tensor, real: torch.Tensor, mask: torch.Tensor) -> float:
    """Kabsch-aligned RMSD between pred and real, restricted to masked atoms.

    Args:
        pred: (N, 3) predicted atom positions.
        real: (N, 3) ground-truth atom positions.
        mask: (N,) float/bool mask of resolved atoms.

    Returns:
        RMSD as a Python float, or NaN if fewer than 3 atoms are available.
    """
    sel = mask.bool()
    p = pred[sel].float()
    r = real[sel].float()
    if p.shape[0] < 3:
        return float("nan")

    p_c = p - p.mean(0)
    r_c = r - r.mean(0)
    p_rot = _kabsch_align(p_c, r_c)
    rmsd = torch.sqrt(((p_rot - r_c) ** 2).sum(-1).mean())
    return rmsd.item()


class DynamicsRMSDCallback(pl.Callback):
    """Computes and logs Kabsch-aligned RMSD(xt_pred, xt_real) per validation epoch.

    Reads outputs returned by DynamicsModule.validation_step:
        outputs["atom_positions_predicted"] — (B, [N_samples,] N_atom, 3)
        outputs["gt_atom_positions"]        — (B, N_atom, 3)
        outputs["atom_mask"]               — (B, N_atom) resolved-atom mask
        outputs["temp"]                    — (B,) int temperature (optional)

    At epoch end logs:
        val/rmsd_xt_pred_vs_real           — mean RMSD across all val batches
        val/rmsd_xt_pred_vs_real_{T}K      — mean RMSD per temperature group
    """

    def __init__(self):
        super().__init__()
        self._rmsds: list[float] = []
        self._temp_rmsds: dict[int, list[float]] = defaultdict(list)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if outputs is None:
            return

        pred_pos = outputs.get("atom_positions_predicted")
        gt_pos = outputs.get("gt_atom_positions")
        atom_mask = outputs.get("atom_mask")

        if pred_pos is None or gt_pos is None or atom_mask is None:
            return

        # Handle multiple diffusion samples: (B, N_samples, N_atom, 3) → take first
        if pred_pos.ndim == 4:
            pred_pos = pred_pos[:, 0]

        temps = outputs.get("temp")  # (B,) int tensor or None

        B = gt_pos.shape[0]
        for i in range(B):
            rmsd = kabsch_rmsd(
                pred=pred_pos[i].detach().cpu(),
                real=gt_pos[i].detach().cpu(),
                mask=atom_mask[i].detach().cpu(),
            )
            self._rmsds.append(rmsd)
            if temps is not None:
                t = int(temps[i].item())
                self._temp_rmsds[t].append(rmsd)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._rmsds:
            return

        valid = [r for r in self._rmsds if not (r != r)]  # filter NaN
        if valid:
            mean_rmsd = sum(valid) / len(valid)
            pl_module.log(
                "val/rmsd_xt_pred_vs_real",
                mean_rmsd,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            logger.info(f"val/rmsd_xt_pred_vs_real = {mean_rmsd:.4f} Å")

        for temp, rmsds in self._temp_rmsds.items():
            valid_t = [r for r in rmsds if not (r != r)]
            if valid_t:
                mean_t = sum(valid_t) / len(valid_t)
                pl_module.log(
                    f"val/rmsd_xt_pred_vs_real_{temp}K",
                    mean_t,
                    on_step=False,
                    on_epoch=True,
                )

        self._rmsds.clear()
        self._temp_rmsds.clear()
