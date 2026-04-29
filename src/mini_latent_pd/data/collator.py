"""Collator and sampler for batching variable-length mdCATH protein samples."""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

# Keys whose tensors have a residue-length dimension that needs padding.
# Maps key -> list of dims that are residue-length (0-indexed within the tensor).
_PAD_DIMS = {
    "x0": [0],            # (L, 37, 3)
    "xt": [0],            # (L, 37, 3)
    "aatype": [0],        # (L,)
    "atom37_mask": [0],   # (L, 37)
    "s_s": [0],           # (L, D)
    "s_z": [0, 1],        # (L, L, D)
    # single-frame keys
    "all_atom_positions": [0],  # (L, 37, 3)
    "all_atom_mask": [0],       # (L, 37)
    "residue_index": [0],       # (L,)
}


def mdcath_collate(batch: list[dict]) -> dict:
    """
    Collate variable-length protein samples into a padded batch.

    Produces:
        - All tensor values padded to the max length in the batch (zero-padded).
        - ``seq_mask``: (B, L_max) float32 — 1.0 for real residues, 0.0 for padding.
        - Scalar / string fields are collected into lists.
    """
    max_len = max(_get_seq_len(s) for s in batch)

    out: dict = {}
    keys = batch[0].keys()

    for key in keys:
        vals = [s[key] for s in batch]

        if not isinstance(vals[0], torch.Tensor):
            out[key] = vals
            continue

        if key in _PAD_DIMS:
            out[key] = _pad_tensors(vals, max_len, _PAD_DIMS[key])
        else:
            out[key] = torch.stack(vals)

    # Build the padding mask: 1.0 for real residues, 0.0 for padding
    seq_lengths = torch.tensor([_get_seq_len(s) for s in batch], dtype=torch.long)
    seq_mask = torch.arange(max_len).unsqueeze(0) < seq_lengths.unsqueeze(1)
    out["seq_mask"] = seq_mask.float()

    return out


def _get_seq_len(sample: dict) -> int:
    """Infer the sequence length from the first residue-dimensioned tensor."""
    for key in ("aatype", "x0", "all_atom_positions"):
        if key in sample and isinstance(sample[key], torch.Tensor):
            return sample[key].shape[0]
    raise ValueError("Cannot infer sequence length from sample keys")


def _pad_tensors(tensors: list[torch.Tensor], target_len: int, pad_dims: list[int]) -> torch.Tensor:
    """Zero-pad a list of tensors along the specified dimensions to target_len."""
    padded = []
    for t in tensors:
        pad_widths = [0] * (2 * t.ndim)  # torch.nn.functional.pad uses reversed pairs
        for d in pad_dims:
            amount = target_len - t.shape[d]
            # pad format: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
            # index into reversed pairs: dim d from the end
            idx = 2 * (t.ndim - 1 - d)
            pad_widths[idx + 1] = amount  # pad on the right side
        padded.append(torch.nn.functional.pad(t, pad_widths, value=0))
    return torch.stack(padded)


class BucketBatchSampler(Sampler):
    """
    Groups proteins by sequence length into buckets, then yields batches
    from within each bucket. This minimises padding waste.

    Args:
        lengths: Sequence length for each dataset index.
        batch_size: Target number of samples per batch.
        bucket_limits: Sorted list of upper bounds for each bucket.
            E.g. [100, 200, 400, 900] creates buckets [0-100], (100-200], (200-400], (400-900].
            If None, buckets are created automatically from the data.
        shuffle: Whether to shuffle within buckets each epoch.
        drop_last: Whether to drop the last incomplete batch in each bucket.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_limits: list[int] | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if bucket_limits is None:
            bucket_limits = _auto_bucket_limits(lengths)
        self.bucket_limits = sorted(bucket_limits)

        # Assign each index to a bucket
        self.buckets: list[list[int]] = [[] for _ in range(len(self.bucket_limits))]
        for idx, length in enumerate(lengths):
            for b, limit in enumerate(self.bucket_limits):
                if length <= limit:
                    self.buckets[b].append(idx)
                    break

    def __iter__(self):
        g = torch.Generator()
        if self.shuffle:
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        all_batches = []
        for bucket in self.buckets:
            if len(bucket) == 0:
                continue
            if self.shuffle:
                perm = torch.randperm(len(bucket), generator=g).tolist()
                indices = [bucket[i] for i in perm]
            else:
                indices = list(bucket)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        # Shuffle batch order so training doesn't always see short proteins first
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        yield from all_batches

    def __len__(self):
        total = 0
        for bucket in self.buckets:
            n = len(bucket) // self.batch_size
            if not self.drop_last and len(bucket) % self.batch_size > 0:
                n += 1
            total += n
        return total


def _auto_bucket_limits(lengths: list[int], n_buckets: int = 5) -> list[int]:
    """Create bucket boundaries from quantiles of the length distribution."""
    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    limits = []
    for i in range(1, n_buckets + 1):
        idx = min(int(i / n_buckets * n) - 1, n - 1)
        limits.append(sorted_lens[idx])
    # Deduplicate while keeping order
    seen = set()
    unique = []
    for lim in limits:
        if lim not in seen:
            seen.add(lim)
            unique.append(lim)
    return unique
