"""Public return types and loaders — what users get back from SERIE.
公共返回类型和加载器——用户从 SERIE 得到的对象。

TimeSeriesData      — one clean series [T, D] + provenance, from load_data()
                      一条干净序列 [T, D] + 来源信息，来自 load_data()
SplitTimeSeriesData — partitioned series + split stats, from inject_noise()
                      分割后的序列 + 分割统计，来自 inject_noise()
load_data(source)   — dispatch: TrajectorySpec → synthetic, KnownMeta → recorded
                      分发：TrajectorySpec → 合成，KnownMeta → 录制
"""

from __future__ import annotations

from enum import StrEnum, auto
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING

import numpy as np
from attrs import define, field

if TYPE_CHECKING:
    from .pipeline import DataBundle, KnownMeta, PartitionSpec, PartitionStats
    from .specs import TitrationCell, TrajectorySpec


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
    """One clean time series array with provenance metadata.
    一条带有来源元数据的干净时间序列数组。

    Returned by ``load_data()``. Contains the raw values, feature names,
    and source identity. Not yet split or noise-injected.
    由 ``load_data()`` 返回。包含原始值、特征名和数据源标识。
    尚未分割或注入噪声。
    """

    values: np.ndarray
    feature_names: tuple[str, ...]
    source_id: str
    source_kind: SourceKind
    metadata: Mapping[str, object] = field(factory=dict)

    def __attrs_post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float32, copy=True, order="C")
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
    """One time series after train/val/test partitioning with per-split statistics.
    经过训练/验证/测试分割后的一条时间序列，附带各分割的统计信息。

    Returned by ``inject_noise()``. Access splits via:
    由 ``inject_noise()`` 返回。通过以下方式访问分割：

        split.train_values   # [T_train, D] NumPy array
        split.val_values     # [T_val, D]
        split.test_values    # [T_test, D]
        split.train_stats.mean  # [D] per-channel mean (train only, no leakage)
        split.train_stats.std   # [D] per-channel std (train only)
    """

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
        values = np.array(self.values, dtype=np.float32, copy=True, order="C")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def channels(self) -> int:
        return self.values.shape[1]

    @property
    def train_values(self) -> np.ndarray:
        train_start, train_end, _, _, _, _ = self.indices
        return self.values[train_start:train_end]

    @property
    def val_values(self) -> np.ndarray:
        _, _, val_start, val_end, _, _ = self.indices
        return self.values[val_start:val_end]

    @property
    def test_values(self) -> np.ndarray:
        _, _, _, _, test_start, test_end = self.indices
        return self.values[test_start:test_end]


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
        db = DataBundle() | StageLoadKnownCSV(
            dataset_id=known_meta.dataset_id,
            known_meta=known_meta,
            save_ipc=cfg.save_ipc,
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
        db = DataBundle() | StageLoadKnownCSV(
            dataset_id=known_meta.dataset_id,
            known_meta=known_meta,
            save_ipc=cfg.save_ipc,
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
