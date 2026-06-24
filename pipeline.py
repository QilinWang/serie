"""Data pipeline — core types and stages.

Reading guide:
    The central object is DataBundle — a mutable accumulator that stages
    write into as they run.  Stages are BundleStep subclasses composed
    with the pipe operator:

        db = DataBundle() | StageLoadTensor(...) | StageSplitByFraction(...) | StageComputeSplitStats()

    Each ``|`` calls the next stage's .apply(db) and returns the same db
    (mutated in place).  This is the only pipeline pattern in the file.

Layout (top to bottom):
    1. Core types     — PartitionSpec, PartitionStats, DataBundle, BundleStep, BundlePlan
    2. Experiment     — DrawDesign, TitrationPlan (user-facing, placed early)
    3. Known data     — KnownDataID, KnownMeta
    4. Load stages    — StageLoadSynthetic, StageLoadTensor, StageLoadKnownCSV
    5. Clean stages   — StageSelectColumns, StageDropBadColumns, StageImputeShortRuns, StageConvertToNumpy
    6. Split stages   — StageSplitByFraction, StageComputeSplitStats
    7. Noise stages   — StageInjectNoise, StageInjectTrainScaledNoise
    8. Diagnostics    — StagePrintDiagnostics (reports + renderers in diagnostics.py)
"""
from __future__ import annotations

import abc
import logging
import os
from collections.abc import Iterator, Sequence
from enum import StrEnum, auto
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import polars as pl
from attrs import define, field, validators

from .integrators import generate_from_trajectory_spec
from .specs import TitrationCell, ODESystem, DGPBank, TrajectorySpec

logger = logging.getLogger(__name__)


def _configure_display():
    """Call once from a notebook or script entry point, not at import time."""
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_rows(20)
    pl.Config.set_fmt_float("mixed")
    np.set_printoptions(precision=3, suppress=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Core types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _in_unit_interval(_inst, attr, value: float) -> None:
    """REQUIRE: value in [0, 1]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{attr.name} must be in [0, 1]; got {value}")


@define(frozen=True, slots=True, kw_only=True)
class PartitionSpec:
    """Train/validation/test split fractions for a time series.
    时间序列的训练/验证/测试分割比例。

    Each pair defines a half-open fraction interval over the full series:
    ``data[int(T * train_start) : int(T * train_end)]`` selects the training
    segment, and likewise for validation and test.
    每对值定义序列上的半开分数区间：
    ``data[int(T * train_start) : int(T * train_end)]`` 选取训练段，验证和测试段同理。

    Use presets for common splits:
    使用预设获取常见分割：

        PartitionSpec.p70_20_10()   # 70% train, 20% val, 10% test
        PartitionSpec.p60_20_20()   # 60% train, 20% val, 20% test

    Or construct a custom split:
    或构建自定义分割：

        PartitionSpec(train_start=0.0, train_end=0.8,
                      val_start=0.8, val_end=0.9,
                      test_start=0.9, test_end=1.0)
    """

    train_start: float = field(validator=_in_unit_interval)
    train_end: float = field(validator=_in_unit_interval)
    val_start: float = field(validator=_in_unit_interval)
    val_end: float = field(validator=_in_unit_interval)
    test_start: float = field(validator=_in_unit_interval)
    test_end: float = field(validator=_in_unit_interval)

    def __attrs_post_init__(self) -> None:
        if not (0.0 <= self.train_start < self.train_end <= 1.0):
            raise ValueError(f"Train bounds invalid: {self.train_start}-{self.train_end}")
        if not (self.train_end <= self.val_start < self.val_end <= 1.0):
            raise ValueError(f"Val bounds invalid: {self.val_start}-{self.val_end}")
        if not (self.val_end <= self.test_start < self.test_end <= 1.0):
            raise ValueError(f"Test bounds invalid: {self.test_start}-{self.test_end}")

    @classmethod
    def p70_10_20(cls):
        """70% train, 10% val, 20% test. 70%训练，10%验证，20%测试。"""
        return cls(train_start=0.0, train_end=0.7, val_start=0.7, val_end=0.8, test_start=0.8, test_end=1.0)

    @classmethod
    def p70_20_10(cls):
        """70% train, 20% val, 10% test. 70%训练，20%验证，10%测试。"""
        return cls(train_start=0.0, train_end=0.7, val_start=0.7, val_end=0.9, test_start=0.9, test_end=1.0)

    @classmethod
    def p60_20_20(cls):
        """60% train, 20% val, 20% test. 60%训练，20%验证，20%测试。"""
        return cls(train_start=0.0, train_end=0.6, val_start=0.6, val_end=0.8, test_start=0.8, test_end=1.0)

    @classmethod
    def p50_25_25(cls):
        """50% train, 25% val, 25% test. 50%训练，25%验证，25%测试。"""
        return cls(train_start=0.0, train_end=0.5, val_start=0.5, val_end=0.75, test_start=0.75, test_end=1.0)

    @classmethod
    def p60_15_25(cls):
        """60% train, 15% val, 25% test. 60%训练，15%验证，25%测试。"""
        return cls(train_start=0.0, train_end=0.6, val_start=0.6, val_end=0.75, test_start=0.75, test_end=1.0)

    @classmethod
    def p60_10_30(cls):
        """60% train, 10% val, 30% test. 60%训练，10%验证，30%测试。"""
        return cls(train_start=0.0, train_end=0.6, val_start=0.6, val_end=0.7, test_start=0.7, test_end=1.0)

    @classmethod
    def p50_10_40(cls):
        """50% train, 10% val, 40% test. 50%训练，10%验证，40%测试。"""
        return cls(train_start=0.0, train_end=0.5, val_start=0.5, val_end=0.6, test_start=0.6, test_end=1.0)

    @classmethod
    def p70_gap_10_20(cls):
        """70% train, 10% val, 20% test with small gaps between splits.
        70%训练，10%验证，20%测试，分割之间有小间隔。"""
        return cls(train_start=0.0, train_end=0.7, val_start=0.71, val_end=0.81, test_start=0.82, test_end=1.0)


@define(slots=True, frozen=True)
class PartitionStats:
    """intent: statistical summary of a data split."""

    mean: np.ndarray  # [D]
    std: np.ndarray

    median: np.ndarray
    mad: np.ndarray  # Median Absolute Deviation
    q01: np.ndarray
    q05: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    q95: np.ndarray
    q99: np.ndarray

    max_: np.ndarray
    min_: np.ndarray
    counts: float

    def __repr__(self) -> str:  # Compact repr for logs
        return f"<PartitionStats: N={int(self.counts)} | Mean={self.mean[:3]}... | Std={self.std[:3]}...>"

    def center_and_spread(
        self,
        method: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (center, scale) for the given normalization method."""
        if method == "standard":
            return self.mean, self.std
        elif method == "robust_iqr":
            return self.median, (self.q75 - self.q25)
        elif method == "robust_mad":
            return self.median, self.mad * 1.4826
        elif method == "center_mean":
            return self.mean, np.ones_like(self.mean, dtype=np.float32)
        elif method == "center_median":
            return self.median, np.ones_like(self.median, dtype=np.float32)
        elif method == "none":
            return np.zeros_like(self.mean, dtype=np.float32), np.ones_like(self.std, dtype=np.float32)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


@define(slots=True, kw_only=True, eq=False)
class DataBundle:
    """Mutable pipeline accumulator for the Python data path."""

    # === Provenance (set by load stage) ===
    dataset_id: str | ODESystem | KnownDataID | None = None
    trajectory_draw_ordinal: int | None = None
    params: object | None = None  # TrajectorySpec for synthetic
    known_meta: KnownMeta | None = None  # For known datasets
    df: pl.DataFrame | None = None  # temp container for known data

    # === Data (set by load/clean stages) ===
    # The current full [T, D] series array. It starts as the clean source and
    # noise stages replace it with the noisy observed series.
    series: np.ndarray | None = None  # [T, D] contiguous float32
    channels: int | None = None
    feature_names: list[str] | None = None

    # === Partition (set by split stages) ===
    indices: tuple[int, int, int, int, int, int] | None = None
    partition_spec: PartitionSpec | None = None
    train_partition_stats: PartitionStats | None = None
    val_partition_stats: PartitionStats | None = None
    test_partition_stats: PartitionStats | None = None

    injected_sigma: float | None = None

    # === Pipeline operator ===
    # Pipe operator: ``db | stage`` calls stage.apply(db) and returns the result.
    # This is the only way stages compose.  See BundleStep below.
    def __or__(self, func: Callable[["DataBundle"], "DataBundle"]) -> "DataBundle":
        return func(self)

    # === Derived properties ===
    @property
    def train_std(self) -> np.ndarray:
        assert self.train_partition_stats is not None, "train_partition_stats not populated"
        return self.train_partition_stats.std


@define(frozen=True, slots=True)
class BundleStep:
    """Pipeable `DataBundle -> DataBundle` transform."""

    @abc.abstractmethod
    def apply(self, db: "DataBundle") -> "DataBundle":
        raise NotImplementedError

    def __call__(self, db: "DataBundle") -> "DataBundle":
        return self.apply(db)

    def __or__(self, right):  # REQUIRE: right is BundleStep or DataBundle
        if isinstance(right, DataBundle):  # op | db
            return self.apply(right)
        if isinstance(right, BundleStep):  # op1 | op2
            return BundlePlan(parts=(self, right))
        raise TypeError(f"Unsupported pipe: {type(self)} | {type(right)}")

    @property
    def name(self) -> str:
        return self.__class__.__name__


@define(frozen=True, slots=True)
class BundlePlan(BundleStep):
    """Deterministic sequence of `BundleStep`s."""

    parts: tuple[BundleStep, ...]  # [K]

    def apply(self, db: "DataBundle") -> "DataBundle":
        for op in self.parts:
            db = op(db)
        return db

    def __or__(self, right: BundleStep) -> "BundlePlan":
        return BundlePlan(parts=(*self.parts, right))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Experiment design (user-facing — placed early for discoverability)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DrawDesign(StrEnum):
    """How trajectory draws and noise draws are paired in a noise sweep.
    噪声扫描中轨迹抽取和噪声抽取的配对方式。

    - AUTO: let SERIE choose based on the source (default).
      自动：让 SERIE 根据数据源选择（默认）。
      Deterministic sources → NOISE_ONLY; stochastic → JOINT.
    - NOISE_ONLY: one clean trajectory, crossed with multiple noise draws.
      仅噪声：一条干净轨迹，与多个噪声抽取交叉。
    - JOINT: pair trajectory draw r with noise draw r (for stochastic sources).
      联合：将轨迹抽取 r 与噪声抽取 r 配对（用于随机数据源）。
    - TRAJECTORY_FIXED: one stochastic trajectory with many noise draws
      (measurement-noise-only diagnostic).
      固定轨迹：一条随机轨迹配多个噪声抽取（仅测量噪声诊断）。
    - CROSS_PRODUCT: full trajectory × noise grid (rarely needed).
      叉积：完整的轨迹 × 噪声网格（很少需要）。
    """

    AUTO = auto()
    NOISE_ONLY = auto()
    JOINT = auto()
    TRAJECTORY_FIXED = auto()
    CROSS_PRODUCT = auto()


@define(frozen=True, slots=True, kw_only=True)
class TitrationPlan:
    """Experiment grid for a noise-dose sweep: which noise levels to test and
    how many independent draws to use.
    噪声剂量扫描的实验网格：测试哪些噪声水平，使用多少独立抽取。

    A plan defines the combinations of (noise_level, trajectory_draw, noise_draw)
    that form the sweep. Use ``iter_cells(bank, plan)`` to iterate over them.
    计划定义了构成扫描的 (噪声水平, 轨迹抽取, 噪声抽取) 组合。
    使用 ``iter_cells(bank, plan)`` 来遍历它们。

    Example:
        >>> plan = TitrationPlan(noise_levels=(0.1, 0.5, 1.0))
        >>> for cell in plan.iter_cells(bank):
        ...     split = inject_noise(
        ...         bank,
        ...         sigma=cell.noise_level,
        ...         trajectory_draw_ordinal=cell.trajectory_draw_ordinal,
        ...         noise_draw_ordinal=cell.noise_draw_ordinal,
        ...     )
    """

    noise_levels: tuple[float, ...] = field(
        default=(1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        converter=tuple,
    )
    n_trajectory_draws: int = field(default=3, validator=validators.ge(1))
    n_noise_draws: int = field(default=3, validator=validators.ge(1))
    noise_seed_base: int = 1000
    draw_design: DrawDesign = field(
        default=DrawDesign.AUTO,
        converter=DrawDesign,
    )

    def __attrs_post_init__(self) -> None:
        if len(self.noise_levels) == 0:
            raise ValueError("TitrationPlan.noise_levels cannot be empty")

    @classmethod
    def quick(cls) -> "TitrationPlan":
        return cls(noise_levels=(0.1, 1.0), n_trajectory_draws=1, n_noise_draws=1)

    @classmethod
    def standard(cls) -> "TitrationPlan":
        return cls()

    def _resolved_draw_design(self, bank: "DGPBank") -> DrawDesign:
        return self._resolved_draw_design_from_flag(bank.is_stochastic)

    def _resolved_draw_design_from_flag(self, is_stochastic: bool) -> DrawDesign:
        if self.draw_design is not DrawDesign.AUTO:
            return self.draw_design
        return (
            DrawDesign.JOINT
            if is_stochastic
            else DrawDesign.NOISE_ONLY
        )

    def _require_available(self, bank: "DGPBank", *, n_trajectory: int, n_noise: int) -> None:
        if bank.n_trajectory_draws < n_trajectory:
            raise ValueError(
                f"bank has {bank.n_trajectory_draws} trajectory draws, but plan requires {n_trajectory}"
            )
        if bank.n_noise_draws < n_noise:
            raise ValueError(
                "bank has "
                f"{bank.n_noise_draws} noise draws, but plan requires {n_noise}"
            )

    def cell_count_per_sigma(self, *, is_stochastic: bool) -> int:
        """Return the number of draw cells per sigma without a DGP bank."""
        design = self._resolved_draw_design_from_flag(is_stochastic)
        if design is DrawDesign.NOISE_ONLY:
            if self.n_trajectory_draws != 1:
                raise ValueError("noise_only requires n_trajectory_draws=1")
            return self.n_noise_draws
        if design is DrawDesign.JOINT:
            if self.n_trajectory_draws != self.n_noise_draws:
                raise ValueError("joint requires n_trajectory_draws == n_noise_draws")
            return self.n_trajectory_draws
        if design is DrawDesign.TRAJECTORY_FIXED:
            if self.n_trajectory_draws != 1:
                raise ValueError("trajectory_fixed requires n_trajectory_draws=1")
            return self.n_noise_draws
        if design is DrawDesign.CROSS_PRODUCT:
            return self.n_trajectory_draws * self.n_noise_draws
        raise ValueError(f"unknown draw_design={self.draw_design!r}")

    def iter_cells(self, bank: "DGPBank") -> Iterator[TitrationCell]:
        """Yield titration cells under the plan's draw design.

        AUTO resolves to:
        - deterministic/recorded sources: one clean trajectory path crossed with
          noise draws (`noise_only`)
        - stochastic sources: paired trajectory + noise draws by spawn
          ordinal (`joint`)

        Use `cross_product` only when the full trajectory × noise grid is an
        explicit design choice, and `trajectory_fixed` only for the conditional
        diagnostic "one stochastic path, many noises".
        """
        design = self._resolved_draw_design(bank)

        if design is DrawDesign.NOISE_ONLY:
            if self.n_trajectory_draws != 1:
                raise ValueError(
                    "noise_only uses exactly one clean/trajectory draw. "
                    "Set n_trajectory_draws=1, or use draw_design='cross_product' "
                    "if you intentionally want the full grid."
                )
            self._require_available(bank, n_trajectory=1, n_noise=self.n_noise_draws)
            for noise_level in self.noise_levels:
                for noise_draw_ordinal in range(self.n_noise_draws):
                    yield bank.cell(
                        noise_level=noise_level,
                        trajectory_draw_ordinal=0,
                        noise_draw_ordinal=noise_draw_ordinal,
                    )
            return

        if design is DrawDesign.JOINT:
            if not bank.is_stochastic:
                raise ValueError("joint requires a stochastic DGP bank")
            if self.n_trajectory_draws != self.n_noise_draws:
                raise ValueError(
                    "joint pairs trajectory and noise draws by spawn ordinal, "
                    f"so n_trajectory_draws must equal n_noise_draws; got "
                    f"{self.n_trajectory_draws} and {self.n_noise_draws}."
                )
            self._require_available(
                bank,
                n_trajectory=self.n_trajectory_draws,
                n_noise=self.n_noise_draws,
            )
            for noise_level in self.noise_levels:
                for draw_ordinal in range(self.n_trajectory_draws):
                    yield bank.cell(
                        noise_level=noise_level,
                        trajectory_draw_ordinal=draw_ordinal,
                        noise_draw_ordinal=draw_ordinal,
                    )
            return

        if design is DrawDesign.TRAJECTORY_FIXED:
            if not bank.is_stochastic:
                raise ValueError("trajectory_fixed requires a stochastic DGP bank")
            if self.n_trajectory_draws != 1:
                raise ValueError(
                    "trajectory_fixed is the conditional diagnostic with one trajectory path "
                    "and many noise draws; set n_trajectory_draws=1."
                )
            self._require_available(bank, n_trajectory=1, n_noise=self.n_noise_draws)
            for noise_level in self.noise_levels:
                for noise_draw_ordinal in range(self.n_noise_draws):
                    yield bank.cell(
                        noise_level=noise_level,
                        trajectory_draw_ordinal=0,
                        noise_draw_ordinal=noise_draw_ordinal,
                    )
            return

        if design is DrawDesign.CROSS_PRODUCT:
            self._require_available(
                bank,
                n_trajectory=self.n_trajectory_draws,
                n_noise=self.n_noise_draws,
            )
            for trajectory_draw_ordinal in range(self.n_trajectory_draws):
                for noise_level in self.noise_levels:
                    for noise_draw_ordinal in range(self.n_noise_draws):
                        yield bank.cell(
                            noise_level=noise_level,
                            trajectory_draw_ordinal=trajectory_draw_ordinal,
                            noise_draw_ordinal=noise_draw_ordinal,
                        )
            return

        raise ValueError(f"unknown draw_design={self.draw_design!r}")

    def cells(self, bank: "DGPBank") -> tuple[TitrationCell, ...]:
        return tuple(self.iter_cells(bank))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Known data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class KnownDataID(StrEnum):
    ETTH1 = auto()
    ETTH2 = auto()
    ETTM1 = auto()
    ETTM2 = auto()
    WEATHER = auto()
    SOLAR_ENERGY = auto()
    SOLAR_ENERGY_SUBSET = auto()
    HUGADB_ACCEL_EMG = auto()
    HUGADB_WALK_BICYCLING_SWITCH = auto()
    HUGADB_WALK_BICYCLING_SWITCH_OOD80 = auto()

def known_data_root() -> Path:
    """Root for user-prepared known datasets.

    Default wheels do not bundle known datasets. Set SERIE_KNOWN_ROOT to point
    at a prepared dataset directory, or use the default user cache path.
    """

    root = os.environ.get("SERIE_KNOWN_ROOT")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "serie" / "known"


def _known_asset_path(*parts: str) -> Path:
    return known_data_root().joinpath(*parts)


_KNOWN_SOURCE_URLS: dict[KnownDataID, str] = {
    KnownDataID.ETTH1: "https://github.com/zhouhaoyi/ETDataset",
    KnownDataID.ETTH2: "https://github.com/zhouhaoyi/ETDataset",
    KnownDataID.ETTM1: "https://github.com/zhouhaoyi/ETDataset",
    KnownDataID.ETTM2: "https://github.com/zhouhaoyi/ETDataset",
    KnownDataID.HUGADB_ACCEL_EMG: "https://github.com/romanchereshnev/HuGaDB",
    KnownDataID.HUGADB_WALK_BICYCLING_SWITCH: "https://github.com/romanchereshnev/HuGaDB",
    KnownDataID.HUGADB_WALK_BICYCLING_SWITCH_OOD80: "https://github.com/romanchereshnev/HuGaDB",
}


def _known_data_missing_message(meta: "KnownMeta", path: Path) -> str:
    source_url = _KNOWN_SOURCE_URLS.get(meta.dataset_id) if isinstance(meta.dataset_id, KnownDataID) else None
    details = [
        f"Known dataset {meta.source_id!r} is not bundled with the SERIE wheel.",
        f"Expected prepared IPC or CSV at: {path}",
        "Set SERIE_KNOWN_ROOT to a directory containing prepared known datasets, "
        "or pass KnownMeta.custom(path=...) for an explicit local file.",
    ]
    if source_url:
        details.append(f"Upstream source: {source_url}")
    return "\n".join(details)


@define(frozen=True, slots=True, kw_only=True)
class KnownMeta:
    """Metadata and cleaning rules for one known dataset.

    This is the dataset-side contract for known tabular data:
    - where to load it from
    - which datetime column to drop before modeling
    - which sentinel values mean "bad reading"
    - how to impute short bad runs once they are masked

    It does not hold model or training choices. It only describes how to turn
    one raw dataset into usable clean columns.
    """

    dataset_id: str | KnownDataID
    path: Path
    sampling_seconds: int  # REQUIRE: > 0
    date_column: str | None = None

    # Fields with defaults
    universal_sentinels: list[float] = field(converter=list)
    sentinels: dict[str, list[float]] = field(factory=dict)  # { column_name: [value1, value2, ...] }
    impute_method: Literal["nearest", "linear"] = "linear"

    @property
    def steps_per_hour(self) -> int:
        return int(round(3600 / self.sampling_seconds))

    @property
    def source_id(self) -> str:
        if isinstance(self.dataset_id, KnownDataID):
            return self.dataset_id.value
        return str(self.dataset_id)

    @classmethod
    def from_id(cls, dataset_id: KnownDataID | str) -> "KnownMeta":
        known_id = KnownDataID(dataset_id)
        factory = {
            KnownDataID.ETTH1: cls.etth1,
            KnownDataID.ETTH2: cls.etth2,
            KnownDataID.ETTM1: cls.ettm1,
            KnownDataID.ETTM2: cls.ettm2,
            KnownDataID.WEATHER: cls.weather,
            KnownDataID.SOLAR_ENERGY: cls.solar_energy,
            KnownDataID.SOLAR_ENERGY_SUBSET: cls.solar_energy_subset,
            KnownDataID.HUGADB_ACCEL_EMG: cls.hugadb_accel_emg,
            KnownDataID.HUGADB_WALK_BICYCLING_SWITCH: cls.hugadb_walk_bicycling_switch,
            KnownDataID.HUGADB_WALK_BICYCLING_SWITCH_OOD80: cls.hugadb_walk_bicycling_switch_ood80,
        }
        return factory[known_id]()

    @classmethod
    def custom(
        cls,
        *,
        dataset_id: str,
        path: str | Path,
        sampling_seconds: int,
        date_column: str | None = None,
        universal_sentinels: tuple[float, ...] = (0.0,),
        sentinels: dict[str, list[float]] | None = None,
        impute_method: Literal["nearest", "linear"] = "linear",
    ) -> "KnownMeta":
        return cls(
            dataset_id=dataset_id,
            path=Path(path),
            sampling_seconds=sampling_seconds,
            date_column=date_column,
            universal_sentinels=list(universal_sentinels),
            sentinels={} if sentinels is None else sentinels,
            impute_method=impute_method,
        )

    @classmethod
    def etth1(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.ETTH1,
            path=_known_asset_path("ett", "ETTh1.ipc"),
            sampling_seconds=3600,
            universal_sentinels=[0.0],
            date_column="date",
        )

    @classmethod
    def etth2(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.ETTH2,
            path=_known_asset_path("ett", "ETTh2.ipc"),
            sampling_seconds=3600,
            universal_sentinels=[0.0],
            sentinels={"LULL": [-31.46199989, -28.15699959], "MUFL": [88.29799652]},
            date_column="date",
        )

    @classmethod
    def ettm1(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.ETTM1,
            path=_known_asset_path("ett", "ETTm1.ipc"),
            sampling_seconds=900,
            universal_sentinels=[0.0],
            date_column="date",
        )

    @classmethod
    def ettm2(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.ETTM2,
            path=_known_asset_path("ett", "ETTm2.ipc"),
            sampling_seconds=900,
            universal_sentinels=[0.0],
            sentinels={"LULL": [-31.46199989, -28.15699959], "MUFL": [88.29799652]},
            date_column="date",
        )

    @classmethod
    def weather(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.WEATHER,
            path=_known_asset_path("weather", "weather.ipc"),
            sampling_seconds=600,
            universal_sentinels=[0.0, -9999.0],
            date_column="date",
        )

    @classmethod
    def solar_energy(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.SOLAR_ENERGY,
            path=_known_asset_path("solar", "solar_AL.ipc"),
            sampling_seconds=600,
            universal_sentinels=[],
            date_column=None,
        )

    @classmethod
    def solar_energy_subset(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.SOLAR_ENERGY_SUBSET,
            path=_known_asset_path("solar", "solar_AL_subset16.ipc"),
            sampling_seconds=600,
            universal_sentinels=[],
            date_column=None,
        )

    @classmethod
    def hugadb_accel_emg(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.HUGADB_ACCEL_EMG,
            path=_known_asset_path("hugadb", "hugadb_accel_emg.ipc"),
            sampling_seconds=1,
            universal_sentinels=[],
            date_column=None,
        )

    @classmethod
    def hugadb_walk_bicycling_switch(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.HUGADB_WALK_BICYCLING_SWITCH,
            path=_known_asset_path("hugadb", "hugadb_walk_bicycling_switch.ipc"),
            sampling_seconds=1,
            universal_sentinels=[],
            date_column=None,
        )

    @classmethod
    def hugadb_walk_bicycling_switch_ood80(cls) -> "KnownMeta":
        return cls(
            dataset_id=KnownDataID.HUGADB_WALK_BICYCLING_SWITCH_OOD80,
            path=_known_asset_path("hugadb", "hugadb_walk_bicycling_switch_ood80.ipc"),
            sampling_seconds=1,
            universal_sentinels=[],
            date_column=None,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Load stages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, slots=True, kw_only=True)
class StageLoadSynthetic(BundleStep):
    """Load one synthetic trajectory from a `TrajectorySpec` into a `DataBundle`."""

    trajectory_spec: TrajectorySpec
    seed: int = 0

    def apply(self, db: "DataBundle") -> "DataBundle":
        logger.debug("loading synthetic source: %s", self.trajectory_spec.system)
        data = generate_from_trajectory_spec(
            self.trajectory_spec,
            seed=self.seed,
            dtype=np.float32,
        ).astype(np.float32, copy=False)
        feature_names = [f"f{i}" for i in range(data.shape[1])]
        db.dataset_id = self.trajectory_spec.system
        db.params = self.trajectory_spec
        db.feature_names = feature_names
        db.channels = data.shape[1]
        db.series = np.ascontiguousarray(data, dtype=np.float32)

        return db


@define(frozen=True, slots=True, kw_only=True)
class StageLoadTensor(BundleStep):
    """Load a precomputed trajectory array into a `DataBundle`."""

    data: np.ndarray  # [T, D] contiguous float32
    dataset_id: str  # e.g. "lorenz_ode42_s1.0e-01"
    feature_names: tuple[str, ...]

    def apply(self, db: DataBundle) -> DataBundle:
        db.series = np.ascontiguousarray(self.data, dtype=np.float32)
        db.channels = self.data.shape[1]
        db.dataset_id = self.dataset_id
        db.feature_names = list(self.feature_names)
        return db


@define(frozen=True, slots=True, kw_only=True)
class StageLoadKnownCSV(BundleStep):
    """Load a known CSV or IPC dataset into a `DataBundle`."""

    dataset_id: str | KnownDataID
    known_meta: KnownMeta
    save_ipc: bool = False

    def apply(self, db: "DataBundle") -> "DataBundle":
        logger.debug("loading known source: %s", self.dataset_id)
        source_path = Path(self.known_meta.path).expanduser()
        date_col = self.known_meta.date_column

        if not source_path.exists():
            csv_fallback = source_path.with_suffix(".csv")
            if csv_fallback.exists():
                source_path = csv_fallback
            else:
                raise FileNotFoundError(_known_data_missing_message(self.known_meta, source_path))

        if source_path.suffix.lower() == ".ipc":
            logger.debug("loading IPC: %s", source_path)
            df = pl.read_ipc(source_path)
            if date_col and date_col in df.columns:
                df = df.drop(date_col)
            feature_names = df.columns
            logger.debug("features: %s", feature_names)
            db.dataset_id = self.dataset_id
            db.known_meta = self.known_meta
            db.df = df
            db.feature_names = feature_names
            db.channels = len(feature_names)
            return db

        csv_path = source_path
        ipc_path = csv_path.with_suffix(".ipc")  # Feather/IPC format

        # === Try IPC cache first ===
        if ipc_path.exists() and (not csv_path.exists() or ipc_path.stat().st_mtime >= csv_path.stat().st_mtime):
            # st_mtime is the file's modification time — the timestamp of when the file's contents were last changed.
            logger.debug("loading IPC cache: %s", ipc_path)
            df = pl.read_ipc(ipc_path)
            feature_names = df.columns
            logger.debug("features: %s", feature_names)
            db.dataset_id = self.dataset_id
            db.known_meta = self.known_meta
            db.df = df
            db.feature_names = feature_names
            db.channels = len(feature_names)
            return db

        # === Fall back to CSV, then cache ===
        logger.debug("scanning CSV: %s", csv_path)
        lf = pl.scan_csv(csv_path)

        if date_col:
            if date_col not in lf.collect_schema().names():
                raise ValueError(f"date_column='{date_col}' not found in CSV header")
            logger.debug("excluding date column: %s", date_col)
            lf = lf.select(pl.exclude(date_col))

        df = lf.drop_nulls().collect()
        feature_names = df.columns
        logger.debug("final features: %s", feature_names)

        # === Write IPC cache ===
        if self.save_ipc:
            df.write_ipc(ipc_path)
            logger.info("cached IPC: %s", ipc_path)

        db.dataset_id = self.dataset_id
        db.known_meta = self.known_meta
        db.df = df
        db.feature_names = feature_names
        db.channels = len(feature_names)

        return db


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Clean stages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, slots=True, kw_only=True)
class StageSelectColumns(BundleStep):
    """Filter DataFrame to selected columns by name or index."""

    keep_cols: Sequence[str] | None = None
    keep_col_idx: Sequence[int] | None = None

    def apply(self, db: "DataBundle") -> "DataBundle":
        df = db.df

        if (self.keep_cols is not None) and (self.keep_col_idx is not None):
            raise ValueError("Provide only one of keep_cols or keep_col_idx")  # avoid ambiguous policy

        if self.keep_cols is not None:
            cols = list(self.keep_cols)
        elif self.keep_col_idx is not None:
            cols = [df.columns[i] for i in self.keep_col_idx]  # type: ignore
        else:
            return db  # no-op when nothing requested

        missing = [c for c in cols if c not in df.columns]  # type: ignore
        if missing:
            raise ValueError(f"Missing columns: {missing}\nAvailable: {df.columns}")  # type: ignore

        df2 = df.select(cols)  # type: ignore
        db.df = df2
        db.feature_names = list(df2.columns)
        db.channels = len(df2.columns)
        logger.debug("selected %s columns: %s", db.channels, db.feature_names)
        return db


@define(frozen=True, slots=True, kw_only=True)
class StageDropBadColumns(BundleStep):
    """Drop columns with excessive sentinel fraction or long sentinel runs."""

    exempt_cols: list[str] = field(factory=list)
    drop_col_threshold: float = 0.2
    drop_col_longrun_hours: int = 24 * 7
    value_tol: float = 1e-6

    def apply(self, db: "DataBundle") -> "DataBundle":
        df = db.df  # type: ignore
        N = df.height  # type: ignore
        cols = [c for c in df.columns if c not in self.exempt_cols]  # type: ignore

        # 1. Parallel Analysis Step
        stats_exprs = []
        for c in cols:
            cond, runlen = _detect_sentinel_runs(c, db.known_meta, self.value_tol)  # type: ignore
            stats_exprs.extend(
                [
                    cond.sum().alias(f"{c}_count"),
                    (cond.sum() / pl.lit(N)).alias(f"{c}_frac"),
                    runlen.max().alias(f"{c}_max_run"),
                ]
            )

        stats = df.select(stats_exprs)  # type: ignore

        # 2. Decision Step
        drop_cols = []
        threshold_longrun_steps = int(
            round(self.drop_col_longrun_hours * db.known_meta.steps_per_hour)  # type: ignore
        )

        for c in cols:
            frac = stats[f"{c}_frac"].item()
            max_run = stats[f"{c}_max_run"].item()

            if (frac >= self.drop_col_threshold) or (max_run >= threshold_longrun_steps):
                drop_cols.append(c)
                logger.debug(
                    "dropping column %s because frac=%s max_run=%s",
                    c,
                    frac,
                    max_run,
                )

        # 3. Action Step
        logger.debug("dropping %s columns: %s", len(drop_cols), drop_cols)
        dropped_df = df.drop(drop_cols)  # type: ignore
        db.df = dropped_df
        db.feature_names = list(dropped_df.columns)
        db.channels = len(dropped_df.columns)
        return db


@define(frozen=True, slots=True, kw_only=True)
class StageImputeShortRuns(BundleStep):
    """Detect sentinel runs and interpolate short bad segments."""

    known_meta: KnownMeta
    exempt_cols: list[str] = field(factory=list)
    value_tol: float = 1e-6

    def apply(self, db: "DataBundle") -> "DataBundle":
        df = db.df
        assert isinstance(df, pl.DataFrame)
        cols = [c for c in df.columns if c not in self.exempt_cols]

        # 1. Build Masking Expressions
        mask_exprs = []
        for c in cols:
            cond, runlen = _detect_sentinel_runs(c, db.known_meta, self.value_tol)  # type: ignore
            is_bad = runlen > 0  # impute any remaining short run zeros
            mask_exprs.append(pl.when(is_bad).then(None).otherwise(pl.col(c)).alias(c))
        # 2. Apply Masks
        masked_df = df.with_columns(mask_exprs)

        # 3. Impute # Note: interpolate() is cleaner than the list comp loop
        imputed_df = masked_df.with_columns([pl.col(c).interpolate(method=self.known_meta.impute_method).fill_null(strategy="forward").fill_null(strategy="backward") for c in cols])
        logger.debug("imputed short runs on %s columns", len(cols))
        db.df = imputed_df
        db.feature_names = list(imputed_df.columns)
        db.channels = len(imputed_df.columns)

        return db


def _detect_sentinel_runs(col_name: str, meta: KnownMeta, value_tol: float = 1e-6) -> tuple[pl.Expr, pl.Expr]:
    """
    === Run-length via change cumsum ===
    Example walkthrough:
    col:    [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0]
    cond:   [T,   T,   T,   F,   T,   T,   F  ]
    shift:  [N,   T,   T,   T,   F,   T,   T  ]
    !=:     [T,   F,   F,   T,   T,   F,   T  ]  (change points)
    cumsum: [1,   1,   1,   2,   3,   3,   4  ]  (group IDs)
    len:    [3,   3,   3,   1,   2,   2,   1  ]  (group sizes: pl.len().over(grp))
    runlen: [3,   3,   3,   0,   2,   2,   0  ]  (0 for non-zero groups: pl.when(cond).then(...).otherwise(0))
    df.select(...).item(): Execute the expression and get scalar result
    In our example: max([3,3,3,0,2,2,0]) = 3
    """
    # 1. Build Sentinel Condition
    cond_list = [(pl.col(col_name) - pl.lit(v)).abs() <= value_tol for v in meta.universal_sentinels]
    if meta.sentinels and (vals := meta.sentinels.get(col_name)):
        cond_list.extend([(pl.col(col_name) - pl.lit(v)).abs() <= value_tol for v in vals])

    cond: pl.Expr = pl.lit(False) if not cond_list else pl.fold(acc=pl.lit(False), function=lambda acc, e: acc | e, exprs=cond_list)
    # 2. Build Run-Length
    change: pl.Expr = (cond != cond.shift(1)).fill_null(True)
    grp: pl.Expr = change.cast(pl.Int32).cum_sum()
    runlen: pl.Expr = pl.when(cond).then(pl.len().over(grp)).otherwise(0)

    return cond, runlen


@define(frozen=True, slots=True, kw_only=True)
class StageConvertToNumpy(BundleStep):
    """Finalize cleaning by converting a Polars frame to a contiguous Float32 array."""

    def apply(self, db: "DataBundle") -> "DataBundle":
        logger.debug("converting Polars frame to NumPy")
        np_view = np.ascontiguousarray(
            db.df.select(pl.all().cast(pl.Float32)).to_numpy()  # type: ignore
        )
        db.series = np_view
        db.channels = db.series.shape[1]
        logger.debug("converted to %s shape=%s", db.series.dtype, tuple(db.series.shape))

        db.df = None  # Free the Polars DataFrame to save RAM
        return db


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Split stages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, slots=True, kw_only=True)
class StageSplitByFraction(BundleStep):
    """Partition data into train/val/test by fraction boundaries."""

    partition_spec: PartitionSpec

    def idx(self, a: float, b: float, n: int) -> tuple[int, int]:
        return int(n * a), int(n * b)

    def apply(self, db: "DataBundle") -> "DataBundle":
        logger.debug("partitioning data")
        n = db.series.shape[0]  # type: ignore
        t0_idx, t1_idx = self.idx(self.partition_spec.train_start, self.partition_spec.train_end, n)
        v0_idx, v1_idx = self.idx(self.partition_spec.val_start, self.partition_spec.val_end, n)
        s0_idx, s1_idx = self.idx(self.partition_spec.test_start, self.partition_spec.test_end, n)

        db.partition_spec = self.partition_spec
        db.indices = (t0_idx, t1_idx, v0_idx, v1_idx, s0_idx, s1_idx)

        return db


@define(frozen=True, slots=True, kw_only=True)
class StageComputeSplitStats(BundleStep):
    """Compute per-split statistics (mean, std, quantiles) for normalization."""

    ddof: int = 1

    def _get_stats(self, X: np.ndarray, begin_idx: int, end_idx: int) -> PartitionStats:
        view = X[begin_idx:end_idx]  # Create View (Zero Memory Cost)
        count = float(view.shape[0])

        mean = np.mean(view, axis=0, dtype=np.float32)
        std = np.std(view, axis=0, ddof=self.ddof, dtype=np.float32)

        q01, q05, q25, median, q75, q95, q99 = np.quantile(
            view,
            [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
            axis=0,
            method="linear",
        ).astype(np.float32, copy=False)
        mad = np.quantile(
            np.abs(view - median),
            0.5,
            axis=0,
            method="linear",
        ).astype(np.float32, copy=False)

        return PartitionStats(
            mean=mean,
            std=std,
            median=median,
            mad=mad,
            q01=q01,
            q05=q05,
            q25=q25,
            q75=q75,
            q95=q95,
            q99=q99,
            max_=np.max(view, axis=0),
            min_=np.min(view, axis=0),
            counts=count,
        )

    def apply(self, db: "DataBundle") -> "DataBundle":
        logger.debug("computing split statistics")
        X = db.series
        db.train_partition_stats = self._get_stats(X, db.indices[0], db.indices[1])  # type: ignore
        db.val_partition_stats = self._get_stats(X, db.indices[2], db.indices[3])  # type: ignore
        db.test_partition_stats = self._get_stats(X, db.indices[4], db.indices[5])  # type: ignore
        return db


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Noise stages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, slots=True, kw_only=True)
class StageInjectNoise(BundleStep):
    """Inject noise: `series += noise_level * noise_draw`."""

    noise_level: float
    noise_draw: np.ndarray

    def apply(self, db: DataBundle) -> DataBundle:
        if self.noise_level == 0.0:
            db.injected_sigma = 0.0
            return db

        T, D = db.series.shape
        assert self.noise_draw.shape == (T, D), (
            f"noise_draw {self.noise_draw.shape} != data ({T}, {D})"
        )

        noise = self.noise_level * self.noise_draw
        db.series = np.ascontiguousarray(db.series + noise, dtype=np.float32)
        db.injected_sigma = self.noise_level
        logger.debug(
            "injected noise: level=%.2e norm=%.2f",
            self.noise_level,
            np.linalg.norm(noise),
        )
        return db


@define(frozen=True, slots=True, kw_only=True)
class StageInjectTrainScaledNoise(BundleStep):
    """Inject noise scaled by train-set channel stds."""

    noise_level: float
    noise_draw: np.ndarray

    def apply(self, db: DataBundle) -> DataBundle:
        if self.noise_level == 0.0:
            db.injected_sigma = 0.0
            return db

        T, D = db.series.shape
        assert self.noise_draw.shape == (T, D), (
            f"noise_draw {self.noise_draw.shape} != data ({T}, {D})"
        )

        channel_scale = np.maximum(db.train_std.astype(np.float32, copy=False), 1e-8)
        noise = self.noise_level * (channel_scale[None, :] * self.noise_draw)
        db.series = np.ascontiguousarray(db.series + noise, dtype=np.float32)
        db.injected_sigma = self.noise_level
        logger.debug(
            "injected train-std-scaled noise: level=%.2e norm=%.2f",
            self.noise_level,
            np.linalg.norm(noise),
        )
        return db


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Diagnostics (see diagnostics.py for reports + renderers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from .diagnostics import print_diagnostics  # noqa: E402


@define(frozen=True, slots=True, kw_only=True)
class StagePrintDiagnostics(BundleStep):
    """Render split diagnostics after partitioning and stats are ready."""

    col: int = 10
    row: int = 5
    sparse_thr: float = 0.3
    heavy_tail_thr_ratio: float = 1.5
    drift_thr_ratio: float = 0.3
    print_verbose: bool = False

    def apply(self, db: "DataBundle") -> "DataBundle":
        print_diagnostics(
            db,
            col=self.col,
            row=self.row,
            sparse_thr=self.sparse_thr,
            heavy_tail_thr_ratio=self.heavy_tail_thr_ratio,
            drift_thr_ratio=self.drift_thr_ratio,
            print_verbose=self.print_verbose,
        )
        logger.info("data partitioned successfully")
        return db
