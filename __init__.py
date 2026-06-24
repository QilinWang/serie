"""SERIE's clean time-series intervention and titration data layer.

Public API, also re-exported from `serie`:
    build_dgp_bank(source, seed)        → DGPBank
    build_empirical_dgp_bank(values)   → DGPBank
    inject_noise(bank, sigma=...)            → SplitTimeSeriesData
    export_dgp_bank_files(bank, path)   → dict
    load_data(source)            → TimeSeriesData

Typical usage:
    from serie import PartitionSpec, TrajectorySpec, build_dgp_bank, inject_noise

    bank = build_dgp_bank("lorenz", seed=0)
    split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())

Systems: "lorenz", "rossler", "chua", "lorenz96", "ou", "doublewell",
         "slds", "seasonal_ar", "garch", "ks", "diagonal"

Known clean recordings can be loaded via KnownMeta / KnownDataID when staged
locally. The wheel does not bundle large known datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .data import KnownCleanConfig, KnownLoadMode
from ._version import PROTOCOL_VERSION, __version__


def load_data(source, **kw) -> "TimeSeriesData":
    """Load one source into a TimeSeriesData array.

    Args:
        source: TrajectorySpec, KnownMeta, or KnownDataID.
        seed: Random seed (for synthetic sources).

    Returns:
        TimeSeriesData with .values [T, D], .feature_names, .source_id.

    Example:
        >>> ts = load_data(TrajectorySpec.lorenz(steps=1000), seed=0)
        >>> ts.values.shape    # (1001, 3)
    """
    from .data import load_data as _load_data
    return _load_data(source, **kw)


def build_dgp_bank(source, **kw) -> "DGPBank":
    """Generate clean trajectories + reusable noise draws once.

    Args:
        source: str ("lorenz", "ou", ...), TrajectorySpec, KnownMeta, or KnownDataID.
        seed: Random seed for trajectory generation.
        n_trajectory_draws: Number of independent trajectories.
        n_noise_draws: Number of reusable noise draws.

    Returns:
        DGPBank with .clean [N_trajectory, T, D] and .noise_draws [N_noise, T, D].

    Example:
        >>> bank = build_dgp_bank("lorenz", seed=42)
        >>> bank.T, bank.D    # (36000, 3)
        >>> bank = build_dgp_bank(TrajectorySpec.ou(steps=5000), seed=0)
    """
    from .api import build_dgp_bank as _build_dgp_bank
    return _build_dgp_bank(source, **kw)


def build_empirical_dgp_bank(values, **kw) -> "DGPBank":
    """Wrap a clean recorded `[T, D]` array as a one-trajectory-draw DGP bank."""
    from .api import build_empirical_dgp_bank as _build_empirical_dgp_bank
    return _build_empirical_dgp_bank(values, **kw)


def inject_noise(bank, **kw) -> "SplitTimeSeriesData":
    """Inject observation noise and return one train/val/test split.

    Args:
        bank: DGPBank from build_dgp_bank().
        sigma: Noise amplitude (0.0 = clean).
        sigma_scale: "unit" or "train_std".
        partition: PartitionSpec — use presets like .p70_20_10().

    Returns:
        SplitTimeSeriesData with .train_values, .val_values, .test_values,
        .train_stats.mean, .train_stats.std, etc.

    Example:
        >>> split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
        >>> split.train_values.shape    # (25200, 3)
        >>> split.train_stats.std       # array([12.3, 8.7, 9.1])
    """
    from .api import inject_noise as _inject_noise
    return _inject_noise(bank, **kw)


def export_dgp_bank_files(bank, out_dir, **kw) -> dict[str, object]:
    """Write one DGP bank's clean/noise arrays and manifest to files."""
    from .api import export_dgp_bank_files as _export_dgp_bank_files
    return _export_dgp_bank_files(bank, out_dir, **kw)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lazy exports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if TYPE_CHECKING:
    from .data import SplitTimeSeriesData, SourceKind, TimeSeriesData
    from .pipeline import DataBundle, KnownDataID, KnownMeta, PartitionSpec, DrawDesign, TitrationPlan
    from .specs import ShockEvent, ShockSchedule, ShockType, TitrationCell, TrajectorySpec, DGPBank
    from .api import build_clean_split_bundle, build_noisy_bundle_for_cell, export_dgp_bank_files, iter_cells


def __getattr__(name: str):
    if name in {"TrajectorySpec", "DGPBank", "TitrationCell", "ShockEvent", "ShockSchedule", "ShockType"}:
        from . import specs as _specs
        return getattr(_specs, name)
    if name in {"TimeSeriesData", "SplitTimeSeriesData", "KnownLoadMode", "SourceKind", "KnownCleanConfig"}:
        from . import data as _data
        return getattr(_data, name)
    if name in {"DataBundle", "KnownDataID", "KnownMeta", "PartitionSpec", "DrawDesign", "TitrationPlan"}:
        from . import pipeline as _pipeline
        return getattr(_pipeline, name)
    if name in {"iter_cells", "build_clean_split_bundle", "build_noisy_bundle_for_cell", "export_dgp_bank_files"}:
        from . import api as _api
        return getattr(_api, name)
    raise AttributeError(f"module 'serie' has no attribute {name!r}")


__all__ = [
    "__version__",
    "PROTOCOL_VERSION",
    # Types
    "TrajectorySpec",
    "KnownMeta",
    "KnownDataID",
    "PartitionSpec",
    "DrawDesign",
    "TitrationPlan",
    "KnownCleanConfig",
    "KnownLoadMode",
    "SourceKind",
    "DGPBank",
    "TitrationCell",
    "ShockEvent",
    "ShockSchedule",
    "ShockType",
    "DataBundle",
    "TimeSeriesData",
    "SplitTimeSeriesData",
    # Public functions
    "build_dgp_bank",
    "build_empirical_dgp_bank",
    "inject_noise",
    "export_dgp_bank_files",
    "load_data",
    "iter_cells",
    "build_clean_split_bundle",
    "build_noisy_bundle_for_cell",
]
