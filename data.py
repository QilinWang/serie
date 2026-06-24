"""Public handoff types and loaders.

TimeSeriesData      — one clean series array + provenance, returned by load_data()
SplitTimeSeriesData — one series after partition + stats, returned by inject_noise()
load_data(source)   — dispatch: TrajectorySpec -> synthetic, KnownMeta -> recorded

These are the objects that cross the boundary between SERIE internals and
user/trainer code.  Everything a user touches after build_dgp_bank/inject_noise
is defined here.
"""
from __future__ import annotations

from enum import StrEnum, auto
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING

import numpy as np
from attrs import define, field

if TYPE_CHECKING:
    from .pipeline import KnownMeta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# public enums and handoff
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SourceKind(StrEnum):
    SYNTHETIC = auto()
    KNOWN = auto()


class KnownLoadMode(StrEnum):
    CLEAN = auto()
    RAW = auto()


@define(frozen=True, slots=True)
class KnownCleanConfig:
    """Public cleaning config for known CSV datasets."""

    mode: KnownLoadMode = KnownLoadMode.CLEAN
    save_ipc: bool = True
    keep_cols: tuple[str, ...] | None = None
    keep_col_idx: tuple[int, ...] | None = None
    exempt_cols: tuple[str, ...] = ()
    drop_bad_columns: bool = True
    impute_short_runs: bool = True
    drop_col_threshold: float = 0.2
    drop_col_longrun_hours: int = 24 * 7

@define(frozen=True, slots=True, eq=False)
class TimeSeriesData:
    """Public handoff object: one clean time series array plus provenance."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    source_id: str
    source_kind: SourceKind
    metadata: Mapping[str, object] = field(factory=dict)

    def __attrs_post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "source_kind", SourceKind(self.source_kind))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def channels(self) -> int:
        return self.values.shape[1]

    def to_bundle(self) -> "DataBundle":
        """Internal bridge back to the trainer-side DataBundle."""
        from .pipeline import DataBundle

        db = DataBundle()
        db.series = self.values
        db.channels = self.channels
        db.feature_names = list(self.feature_names)
        db.dataset_id = self.source_id
        return db


@define(frozen=True, slots=True, eq=False)
class SplitTimeSeriesData:
    """Public split handoff: one series plus split boundaries and split stats."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    source_id: str
    partition: "PartitionSpec"
    indices: tuple[int, int, int, int, int, int]
    train_stats: "PartitionStats"
    val_stats: "PartitionStats"
    test_stats: "PartitionStats"
    cell: "TitrationCell | None" = None
    metadata: Mapping[str, object] = field(factory=dict)

    def __attrs_post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def channels(self) -> int:
        return self.values.shape[1]

    @property
    def train_values(self) -> np.ndarray:
        t0, t1, _, _, _, _ = self.indices
        return self.values[t0:t1]

    @property
    def val_values(self) -> np.ndarray:
        _, _, v0, v1, _, _ = self.indices
        return self.values[v0:v1]

    @property
    def test_values(self) -> np.ndarray:
        _, _, _, _, s0, s1 = self.indices
        return self.values[s0:s1]


def load_synthetic_data(
    trajectory_spec: "TrajectorySpec",
    *,
    seed: int = 0,
) -> TimeSeriesData:
    """Load one synthetic TrajectorySpec into public TimeSeriesData."""
    from .integrators import generate_from_trajectory_spec

    data = generate_from_trajectory_spec(
        trajectory_spec,
        seed=seed,
        dtype=np.float32,
    ).astype(np.float32, copy=False)
    values = np.ascontiguousarray(data, dtype=np.float32)
    feature_names = tuple(f"f{i}" for i in range(data.shape[1]))
    return TimeSeriesData(
        values=values,
        feature_names=feature_names,
        source_id=trajectory_spec.tag,
        source_kind=SourceKind.SYNTHETIC,
        metadata={
            "system": trajectory_spec.system.value,
            "tag": trajectory_spec.tag,
            "seed": int(seed),
            "steps": int(trajectory_spec.steps),
            "dt": float(trajectory_spec.dt),
            "dim": int(trajectory_spec.dim),
            "burn_in": int(trajectory_spec.burn_in),
        },
    )


def load_known_data(
    known_meta: "KnownMeta",
    *,
    mode: "KnownLoadMode | str" = KnownLoadMode.CLEAN,
    exempt_cols: tuple[str, ...] = (),
    save_ipc: bool = True,
    keep_cols: tuple[str, ...] | None = None,
    keep_col_idx: tuple[int, ...] | None = None,
    cleaning: KnownCleanConfig | None = None,
) -> TimeSeriesData:
    """Load one known dataset into public TimeSeriesData."""
    mode = KnownLoadMode(mode)
    if cleaning is not None:
        conflicting = []
        if mode != KnownLoadMode.CLEAN:
            conflicting.append("mode")
        if exempt_cols:
            conflicting.append("exempt_cols")
        if save_ipc is not True:
            conflicting.append("save_ipc")
        if keep_cols is not None:
            conflicting.append("keep_cols")
        if keep_col_idx is not None:
            conflicting.append("keep_col_idx")
        if conflicting:
            names = ", ".join(conflicting)
            raise ValueError(
                f"Pass either cleaning=... or individual known-data cleaning kwargs, not both. "
                f"Conflicting kwargs: {names}."
            )
        cfg = cleaning
    else:
        cfg = KnownCleanConfig(
            mode=mode,
            save_ipc=save_ipc,
            keep_cols=keep_cols,
            keep_col_idx=keep_col_idx,
            exempt_cols=exempt_cols,
        )
    mode = KnownLoadMode(cfg.mode)
    from .pipeline import (
        DataBundle,
        StageConvertToNumpy,
        StageDropBadColumns,
        StageImputeShortRuns,
        StageLoadKnownCSV,
        StageSelectColumns,
    )

    if mode == KnownLoadMode.CLEAN:
        db = (
            DataBundle()
            | StageLoadKnownCSV(
                dataset_id=known_meta.dataset_id,
                known_meta=known_meta,
                save_ipc=cfg.save_ipc,
            )
        )
        if cfg.keep_cols is not None or cfg.keep_col_idx is not None:
            db = db | StageSelectColumns(
                keep_cols=cfg.keep_cols,
                keep_col_idx=cfg.keep_col_idx,
            )
        if cfg.drop_bad_columns:
            db = db | StageDropBadColumns(
                exempt_cols=list(cfg.exempt_cols),
                drop_col_threshold=cfg.drop_col_threshold,
                drop_col_longrun_hours=cfg.drop_col_longrun_hours,
            )
        if cfg.impute_short_runs:
            db = db | StageImputeShortRuns(
                known_meta=known_meta,
                exempt_cols=list(cfg.exempt_cols),
            )
        db = db | StageConvertToNumpy()
    elif mode == KnownLoadMode.RAW:
        db = (
            DataBundle()
            | StageLoadKnownCSV(
                dataset_id=known_meta.dataset_id,
                known_meta=known_meta,
                save_ipc=cfg.save_ipc,
            )
        )
        if cfg.keep_cols is not None or cfg.keep_col_idx is not None:
            db = db | StageSelectColumns(
                keep_cols=cfg.keep_cols,
                keep_col_idx=cfg.keep_col_idx,
            )
        db = db | StageConvertToNumpy()
    else:
        raise ValueError(f"Unsupported known-data load mode: {mode!r}")

    if db.series is None:
        raise RuntimeError("known-data loading did not populate series")
    if db.feature_names is None:
        raise RuntimeError("known-data loading did not populate feature_names")
    return TimeSeriesData(
        values=db.series,
        feature_names=tuple(db.feature_names),
        source_id=known_meta.source_id,
        source_kind=SourceKind.KNOWN,
        metadata={
            "dataset_id": known_meta.source_id,
            "path": str(known_meta.path),
            "sampling_seconds": int(known_meta.sampling_seconds),
            "date_column": known_meta.date_column,
            "rows": int(db.series.shape[0]),
            "channels": int(db.series.shape[1]),
            "load_mode": mode.value,
            "drop_bad_columns": bool(cfg.drop_bad_columns),
            "impute_short_runs": bool(cfg.impute_short_runs),
            "drop_col_threshold": float(cfg.drop_col_threshold),
            "drop_col_longrun_hours": int(cfg.drop_col_longrun_hours),
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified loader (dispatches by source type)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_data(
    source,
    *,
    seed: int = 0,
    mode: KnownLoadMode = KnownLoadMode.CLEAN,
    exempt_cols: tuple[str, ...] = (),
    save_ipc: bool = True,
    keep_cols: tuple[str, ...] | None = None,
    keep_col_idx: tuple[int, ...] | None = None,
    cleaning: KnownCleanConfig | None = None,
) -> TimeSeriesData:
    """Load one source into the public `TimeSeriesData` handoff."""
    from .pipeline import KnownDataID, KnownMeta
    from .specs import TrajectorySpec

    if isinstance(source, TrajectorySpec):
        return load_synthetic_data(source, seed=seed)

    if isinstance(source, KnownMeta):
        return load_known_data(
            source,
            mode=mode,
            exempt_cols=exempt_cols,
            save_ipc=save_ipc,
            keep_cols=keep_cols,
            keep_col_idx=keep_col_idx,
            cleaning=cleaning,
        )

    if isinstance(source, KnownDataID):
        return load_known_data(
            KnownMeta.from_id(source),
            mode=mode,
            exempt_cols=exempt_cols,
            save_ipc=save_ipc,
            keep_cols=keep_cols,
            keep_col_idx=keep_col_idx,
            cleaning=cleaning,
        )

    raise TypeError(f"Unsupported load_data source type: {type(source)!r}")
