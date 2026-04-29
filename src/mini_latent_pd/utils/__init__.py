"""Utility functions for mini_latent_pd."""

__all__ = [
    "pseudo_beta_from_atom37",
    "coords_to_distogram",
    "kabsch_align",
    "kabsch_rmsd",
]


def __getattr__(name):
    if name in __all__:
        from mini_latent_pd.utils import geometry
        return getattr(geometry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
