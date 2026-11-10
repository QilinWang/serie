"""Data pipeline — core types and stages.

Reading guide:
    The central object is DataBundle — a mutable accumulator that stages
    write into as they run.  Stages are BundleStep subclasses composed
    with the pipe operator:

        db = DataBundle() | StageLoadTensor(...) | StageSplitByFraction(...) | StageComputeSplitStats()

    Each ``|`` calls the next stage's .apply(db) and returns the same db
    (mutated in place).  This is the only pipeline pattern in the file.

Layout (top to bottom):
    1. Core types     — PartitionSpec, PartitionStats, DataBundle, Step, Chain, BundleStep,an
    2. Experiment     — DrawDesign, TitrationPlan, SplitTreatmentScheme,
                       SigmaScale (user-facing, placed early)
    3. Known data     — KnownDataID, KnownMeta
    4. Load stages    — StageLoadSynthetic, StageLoadTensor, StageLoadKnownCSV
    5. Clean stages   — StageSelectColumns, StageDropBadColumns, StageImputeShortRuns, StageConvertToNumpy
    6. Split stages   — StageSplitByFraction, StageComputeSplitStats
    7. Noise stages   — StageInjectNoise, StageInjectTrainScaledNoise
    8. Diagnostics    — StagePrintDiagnostics (reports + renderers in diagnostics.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from msgspec import Struct, field as struct_field
from msgspec.structs import force_setattr
from serie.records import DataclassRecord, StructRecord, record_hash

import abc
import copy
import copyreg
import logging
import math
import os
from collections.abc import Iterator, Sequence
from enum import StrEnum, auto
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import polars as pl

from .integrators import StepShockSchedule, generate_from_trajectory_spec
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


def _in_unit_interval(name: str, value: float) -> None:
    """REQUIRE: value in [0, 1]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]; got {value}")


class PartitionSpec(StructRecord, Struct, frozen=True, kw_only=True, eq=False, weakref=True, forbid_unknown_fields=True):
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

    train_start: float
    train_end: float
    val_start: float
    val_end: float
    test_start: float
    test_end: float

    def __post_init__(self) -> None:
        # CHG: converters and field checks ran before the attrs post-init.
        _in_unit_interval('train_start', self.train_start)
        _in_unit_interval('train_end', self.train_end)
        _in_unit_interval('val_start', self.val_start)
        _in_unit_interval('val_end', self.val_end)
        _in_unit_interval('test_start', self.test_start)
        _in_unit_interval('test_end', self.test_end)
        if not (0.0 <= self.train_start < self.train_end <= 1.0):
            raise ValueError(f"Train bounds invalid: {self.train_start}-{self.train_end}")
        if not (self.train_end <= self.val_start < self.val_end <= 1.0):
            raise ValueError(f"Val bounds invalid: {self.val_start}-{self.val_end}")
        if not (self.val_end <= self.test_start < self.test_end <= 1.0):
            raise ValueError(f"Test bounds invalid: {self.test_start}-{self.test_end}")

    @classmethod
    def p70_10_20(cls):
        """70% train, 10% val, 20% test. 70%训练，10%验证，20%测试。"""
        return cls(
            train_start=0.0, train_end=0.7, val_start=0.7, val_end=0.8, test_start=0.8, test_end=1.0
        )

    @classmethod
    def p70_20_10(cls):
        """70% train, 20% val, 10% test. 70%训练，20%验证，10%测试。"""
        return cls(
            train_start=0.0, train_end=0.7, val_start=0.7, val_end=0.9, test_start=0.9, test_end=1.0
        )

    @classmethod
    def p60_20_20(cls):
        """60% train, 20% val, 20% test. 60%训练，20%验证，20%测试。"""
        return cls(
            train_start=0.0, train_end=0.6, val_start=0.6, val_end=0.8, test_start=0.8, test_end=1.0
        )

    @classmethod
    def p50_25_25(cls):
        """50% train, 25% val, 25% test. 50%训练，25%验证，25%测试。"""
        return cls(
            train_start=0.0,
            train_end=0.5,
            val_start=0.5,
            val_end=0.75,
            test_start=0.75,
            test_end=1.0,
        )

    @classmethod
    def p60_15_25(cls):
        """60% train, 15% val, 25% test. 60%训练，15%验证，25%测试。"""
        return cls(
            train_start=0.0,
            train_end=0.6,
            val_start=0.6,
            val_end=0.75,
            test_start=0.75,
            test_end=1.0,
        )

    @classmethod
    def p60_10_30(cls):
        """60% train, 10% val, 30% test. 60%训练，10%验证，30%测试。"""
        return cls(
            train_start=0.0, train_end=0.6, val_start=0.6, val_end=0.7, test_start=0.7, test_end=1.0
        )

    @classmethod
    def p50_10_40(cls):
        """50% train, 10% val, 40% test. 50%训练，10%验证，40%测试。"""
        return cls(
            train_start=0.0, train_end=0.5, val_start=0.5, val_end=0.6, test_start=0.6, test_end=1.0
        )

    @classmethod
    def p70_gap_10_20(cls):
        """70% train, 10% val, 20% test with small gaps between splits.
        70%训练，10%验证，20%测试，分割之间有小间隔。"""
        return cls(
            train_start=0.0,
            train_end=0.7,
            val_start=0.71,
            val_end=0.81,
            test_start=0.82,
            test_end=1.0,
        )

    __hash__ = record_hash


class PartitionStats(StructRecord, Struct, frozen=True, kw_only=False, eq=False, weakref=True, forbid_unknown_fields=True):
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
            return np.zeros_like(self.mean, dtype=np.float32), np.ones_like(
                self.std, dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    __hash__ = record_hash


class _DataBundleState:
    """Copy and restore mutable bundle graphs without rerunning construction."""

    # CHG: a plain base supplies Python's allocator for the standard pickle
    # new-object/state protocol. A pure Struct has no __new__ for NEWOBJ.
    __slots__ = ()

    def __getstate__(self):
        return {name: getattr(self, name) for name in self.__struct_fields__}

    def __setstate__(self, state):
        # CHG: as with attrs slots, restore declared fields only. Inherited
        # Struct fields include the trainer extension without importing Fern.
        for name in self.__struct_fields__:
            if name in state:
                setattr(self, name, state[name])

    def __deepcopy__(self, memo):
        # CHG: register the new container BEFORE traversing its fields. Unlike
        # Struct's default reconstruction, this preserves aliases and cycles
        # (e.g. params referring back to this bundle) instead of recursing.
        copied = Struct.__copy__(self)
        memo[id(self)] = copied
        copied.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return copied

    def __reduce_ex__(self, protocol):
        # CHG: pickle must allocate/memoize the container before its state;
        # putting a self-reference in Struct constructor args cannot do that.
        return copyreg.__newobj__, (type(self),), self.__getstate__()


class DataBundle(_DataBundleState, Struct, kw_only=True, eq=False, weakref=True):
    """Mutable pipeline accumulator for the Python data path."""

    # === Provenance (set by load stage) ===
    dataset_id: str | ODESystem | KnownDataID | None = None
    trajectory_draw_ordinal: int | None = None
    params: object | None = None  # TrajectorySpec for synthetic
    known_meta: KnownMeta | None = None  # For known datasets
    calendar_features: np.ndarray | None = None  # [T, 2] observed time-of-day and weekday fractions
    # CHG: canonical precomputed calendar provenance; consumers may verify
    # encoding from the original datetimes without fitting another clock.
    calendar_source_datetimes: np.ndarray | None = None
    df: pl.DataFrame | None = None  # temp container for known data

    # === Data (set by load/clean stages) ===
    # The current full [T, D] series array. It starts as the clean source and
    # noise stages replace it with the noisy observed series.
    series: np.ndarray | None = None  # [T, D] contiguous float32
    # Optional transition controls. `actions[t]` drives `series[t]` to
    # `series[t+1]`; it remains separate from the observed state manifold.
    actions: np.ndarray | None = None  # [T-1, A] contiguous float32
    # Complete future-only episodes on the batch axis. `episode_states` is
    # [E, T, D] and `episode_actions` is [E, T-1, A]. Present only for banks
    # that declare episode_axis=trajectory_draw; the long-series window path
    # must not see these.
    # CHG 2026-09-02: a bank may also be read as sliding windows cut inside
    # each episode, so the episodes need not be future-only. `episode_lengths`
    # is how many of the T states each episode really recorded when the bank
    # pads shorter episodes out to a common T; None means every episode fills
    # T. The window cutter is the trainer's, not SERIE's.
    episode_states: np.ndarray | None = None
    episode_actions: np.ndarray | None = None
    episode_lengths: np.ndarray | None = None  # [E] int64
    channels: int | None = None
    feature_names: list[str] | None = None

    # === Partition (set by split stages) ===
    indices: tuple[int, int, int, int, int, int] | None = None
    partition_spec: PartitionSpec | None = None
    train_partition_stats: PartitionStats | None = None
    val_partition_stats: PartitionStats | None = None
    test_partition_stats: PartitionStats | None = None

    injected_sigma: float | None = None
    split_treatment_scheme: SplitTreatmentScheme | None = None

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


class Step(abc.ABC):
    """One transform of one subject, pipeable: `apply(subject) -> subject`.

    `a | b` is a `Chain`; `step | subject` applies the step to the subject.
    The bundle stages, the cell stages and the trainer's steps are all this
    one shape (CHG 2026-09-03: it was written four times, once per chain).
    """

    @abc.abstractmethod
    def apply(self, subject):
        raise NotImplementedError

    def __call__(self, subject):
        return self.apply(subject)

    def __or__(self, right):
        if isinstance(right, Step):  # step | step
            return Chain(parts=(self, right))
        return self.apply(right)  # step | subject

    @property
    def name(self) -> str:
        return self.__class__.__name__


@dataclass(frozen=True, slots=True, kw_only=False, eq=False, weakref_slot=True)
class Chain(DataclassRecord, Step):
    """A flat sequence of steps, applied in order; nested chains flatten on construction."""

    parts: tuple[Step, ...]

    def __post_init__(self) -> None:
        # CHG: converters and field checks ran before the attrs post-init.
        object.__setattr__(self, 'parts', tuple(self.parts))
        flat: list[Step] = []
        for part in self.parts:
            flat.extend(part.parts if isinstance(part, Chain) else (part,))
        object.__setattr__(self, "parts", tuple(flat))

    def apply(self, subject):
        for step in self.parts:
            subject = step(subject)
        return subject

    def __or__(self, right):
        if isinstance(right, Step):
            return Chain(parts=(*self.parts, right))
        return self.apply(right)

    def summary(self, depth: int = 0) -> str:
        """The chain as an indented tree, one step per line."""
        lines = [f"{'  ' * depth}* {self.name}:"]
        for part in self.parts:
            lines.append(
                part.summary(depth + 1)
                if isinstance(part, Chain)
                else f"{'  ' * (depth + 1)}- {part.name}"
            )
        return "\n".join(lines) + "\n"

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


class BundleStep(Step):
    """A `Step` whose subject is a `DataBundle`."""

    @abc.abstractmethod
    def apply(self, db: "DataBundle") -> "DataBundle":
        raise NotImplementedError


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Experiment design (user-facing — placed early for discoverability)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SplitTreatmentScheme(StrEnum):
    """Which chronological splits receive the titrated observation noise.

    ``NOISE_MATCHED_RETRAINING`` is Fern's historical titration: the same
    frozen noise draw and dose are applied to train, validation, and test.
    Every dose therefore defines a separately noise-matched training problem.

    ``CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION`` is the clean-selection-regret
    (CSR) data contract: train is clean, validation/model selection is clean,
    and only the test interval receives the dose.  Exact CSR also freezes the
    fitted model and any calibrator across doses.  Recipes that declare
    ``checkpoint_reuse_policy=clean_validation_best`` implement that second
    half explicitly: dose zero trains and saves the validation-best model;
    every positive dose names that clean source run in hashed protocol
    identity, loads its checkpoint, and evaluates only the noisy test split.
    The regret itself is still an analysis across candidate models: select one
    candidate once using dose-zero validation performance, then compare that
    fixed choice with the best candidate at each noisy test dose.  This enum
    controls split treatment; the separate checkpoint policy controls the
    fit-once/evaluate-many execution dependency.

    ``NOISE_TRAIN_NOISE_SELECT_CLEAN_TEST`` is the complementary return-to-clean
    view: train and validation receive the registered dose, while test remains
    clean.  It reuses the validation-best checkpoint fitted by the matching
    noise-matched cell at the same dose and bank.  Comparing this score with
    the noise-matched score separates adaptation to the noisy measurement
    world from retention of the underlying clean dynamics.

    Neither scheme needs gradient descent merely to inject noise or compute a
    score.  Gradient-based fitting happens only because a chosen forecasting
    model's trainer uses it.
    """

    NOISE_MATCHED_RETRAINING = auto()
    CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION = auto()
    NOISE_TRAIN_NOISE_SELECT_CLEAN_TEST = auto()


class SigmaScale(StrEnum):
    """How one shared f32 dose scales the frozen unit-noise draw.

    ``UNIT`` applies the dose directly. ``TRAIN_STD`` multiplies it by the
    clean training split's per-channel standard deviation before injection.
    This belongs to Serie's data semantics, not to a Fern worker request.
    """

    UNIT = "unit"
    TRAIN_STD = "train_std"


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


class TitrationPlan(StructRecord, Struct, frozen=True, kw_only=True, eq=False, weakref=True, forbid_unknown_fields=True):
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

    noise_levels: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    n_trajectory_draws: int = 3
    n_noise_draws: int = 3
    noise_seed_base: int = 1000
    draw_design: DrawDesign = DrawDesign.AUTO

    def __post_init__(self) -> None:
        # CHG: converters and field checks ran before the attrs post-init.
        force_setattr(self, 'noise_levels', tuple(self.noise_levels))
        force_setattr(self, 'draw_design', DrawDesign(self.draw_design))
        if not self.n_trajectory_draws >= 1:
            raise ValueError(f"'n_trajectory_draws' must be >= 1: {self.n_trajectory_draws}")
        if not self.n_noise_draws >= 1:
            raise ValueError(f"'n_noise_draws' must be >= 1: {self.n_noise_draws}")
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
        return DrawDesign.JOINT if is_stochastic else DrawDesign.NOISE_ONLY

    def _require_available(self, bank: "DGPBank", *, n_trajectory: int, n_noise: int) -> None:
        if bank.n_trajectory_draws < n_trajectory:
            raise ValueError(
                f"bank has {bank.n_trajectory_draws} trajectory draws, but plan requires {n_trajectory}"
            )
        if bank.n_noise_draws < n_noise:
            raise ValueError(
                f"bank has {bank.n_noise_draws} noise draws, but plan requires {n_noise}"
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

    __hash__ = record_hash


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
    HUGADB_WALK_RUN_NATURAL = auto()
    HUGADB_WALK_ONLY_NATURAL = auto()


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
    KnownDataID.HUGADB_WALK_RUN_NATURAL: "https://github.com/romanchereshnev/HuGaDB",
    KnownDataID.HUGADB_WALK_ONLY_NATURAL: "https://github.com/romanchereshnev/HuGaDB",
}


def _known_data_missing_message(meta: "KnownMeta", path: Path) -> str:
    source_url = (
        _KNOWN_SOURCE_URLS.get(meta.dataset_id)
        if isinstance(meta.dataset_id, KnownDataID)
        else None
    )
    details = [
        f"Known dataset {meta.source_id!r} is not bundled with the SERIE wheel.",
        f"Expected prepared IPC or CSV at: {path}",
        "Set SERIE_KNOWN_ROOT to a directory containing prepared known datasets, "
        "or pass KnownMeta.custom(path=...) for an explicit local file.",
    ]
    if source_url:
        details.append(f"Upstream source: {source_url}")
    return "\n".join(details)


class KnownMeta(StructRecord, Struct, frozen=True, kw_only=True, eq=False, weakref=True, forbid_unknown_fields=True):
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
    universal_sentinels: list[float]
    sentinels: dict[str, list[float]] = struct_field(default_factory=dict)  # { column_name: [value1, value2, ...] }
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
            KnownDataID.HUGADB_WALK_RUN_NATURAL: cls.hugadb_walk_run_natural,
            KnownDataID.HUGADB_WALK_ONLY_NATURAL: cls.hugadb_walk_only_natural,
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
    def hugadb_walk_run_natural(cls) -> "KnownMeta":
        """One continuous single-subject recording with a performed
        walk -> run -> walk activity change (transition at 36.4% of the
        record). Unlike the bicycling switch asset this is not a splice, so
        the acceleration and deceleration around each change are in the
        signal; see build_hugadb_subject14_worlds.py."""
        return cls(
            dataset_id=KnownDataID.HUGADB_WALK_RUN_NATURAL,
            path=_known_asset_path("hugadb", "hugadb_walk_run_natural.ipc"),
            sampling_seconds=1,
            universal_sentinels=[],
            date_column=None,
        )

    @classmethod
    def hugadb_walk_only_natural(cls) -> "KnownMeta":
        """The stationary control for hugadb_walk_run_natural: the SAME
        subject, sensors, and session, walking on every row. Comparing the
        changed world against this one isolates the activity change;
        comparing it against hugadb_accel_emg does not, because that asset is
        a different subject."""
        return cls(
            dataset_id=KnownDataID.HUGADB_WALK_ONLY_NATURAL,
            path=_known_asset_path("hugadb", "hugadb_walk_only_natural.ipc"),
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

    def __post_init__(self):
        # CHG: preserve converters/checks on direct Python construction.
        force_setattr(self, 'universal_sentinels', list(self.universal_sentinels))

    __hash__ = record_hash


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Load stages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageLoadSynthetic(DataclassRecord, BundleStep):
    """Load one synthetic trajectory from a `TrajectorySpec` into a `DataBundle`."""

    # CHG: an explicit step schedule travels with this stage into the generator
    # and the bundle's source description; the historical no-schedule path stays intact.

    trajectory_spec: TrajectorySpec
    seed: int = 0
    step_schedule: StepShockSchedule | None = None

    def apply(self, db: "DataBundle") -> "DataBundle":
        logger.debug("loading synthetic source: %s", self.trajectory_spec.system)
        data = generate_from_trajectory_spec(
            self.trajectory_spec,
            seed=self.seed,
            dtype=np.float32,
            step_schedule=self.step_schedule,
        ).astype(np.float32, copy=False)
        feature_names = [f"f{i}" for i in range(data.shape[1])]
        db.dataset_id = self.trajectory_spec.system
        db.params = self.trajectory_spec if self.step_schedule is None else self
        db.feature_names = feature_names
        db.channels = data.shape[1]
        db.series = np.ascontiguousarray(data, dtype=np.float32)

        return db

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageLoadTensor(DataclassRecord, BundleStep):
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

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


def calendar_features_from_datetimes(dates: np.ndarray) -> np.ndarray:
    """Convert real timestamps [T] to [T,2] day and Monday-based week fractions."""
    times = np.asarray(dates).astype("datetime64[s]")
    if times.ndim != 1 or np.isnat(times).any():
        raise ValueError("calendar input requires a one-dimensional array of real timestamps")
    days = times.astype("datetime64[D]").astype(np.int64)
    seconds = (times - times.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.int64)
    return np.ascontiguousarray(np.stack((seconds / 86400, ((days + 3) % 7) / 7), axis=1), dtype=np.float32)


def known_calendar_features(db: DataBundle) -> np.ndarray:
    """Read observed calendar time as [T, 2] fractions: day and week.

    Columns are [seconds since midnight / 86400, (Monday-based weekday) / 7].
    The same numeric-null row filter as StageLoadKnownCSV keeps dates aligned
    with the cleaned series. A missing real date column is refused.
    """
    meta = db.known_meta
    if meta is None or meta.date_column is None or db.series is None:
        raise ValueError("calendar features require a loaded known series with a declared date column")
    path = Path(meta.path).expanduser()
    if not path.exists() and path.with_suffix(".csv").exists():
        path = path.with_suffix(".csv")
    if path.suffix.lower() == ".ipc":
        frame = pl.read_ipc(path)
    else:
        if not path.exists():
            raise FileNotFoundError(f"calendar features require the original dated CSV: {path}")
        frame = pl.read_csv(path)
        if meta.date_column in frame.columns:
            numeric_columns = [name for name in frame.columns if name != meta.date_column]
            if not numeric_columns:
                raise ValueError("calendar source has no numeric observation columns")
            frame = frame.filter(pl.all_horizontal(pl.col(numeric_columns).is_not_null()))
    if meta.date_column not in frame.columns:
        raise ValueError(f"calendar source {path} has no declared date column {meta.date_column!r}")
    dates = frame.select(pl.col(meta.date_column).cast(pl.String).str.to_datetime(strict=True).alias("observed_at"))
    if dates["observed_at"].null_count() != 0:
        raise ValueError(f"calendar source {path} has null dates in retained observations")
    if len(dates) != len(db.series):
        raise ValueError(
            f"calendar and cleaned series have different lengths: {len(dates)} dates, {len(db.series)} observations"
        )
    return calendar_features_from_datetimes(dates["observed_at"].to_numpy())


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageLoadKnownCSV(DataclassRecord, BundleStep):
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
        if ipc_path.exists() and (
            not csv_path.exists() or ipc_path.stat().st_mtime >= csv_path.stat().st_mtime
        ):
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

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Clean stages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageSelectColumns(DataclassRecord, BundleStep):
    """Filter DataFrame to selected columns by name or index."""

    keep_cols: Sequence[str] | None = None
    keep_col_idx: Sequence[int] | None = None

    def apply(self, db: "DataBundle") -> "DataBundle":
        df = db.df

        if (self.keep_cols is not None) and (self.keep_col_idx is not None):
            raise ValueError(
                "Provide only one of keep_cols or keep_col_idx"
            )  # avoid ambiguous policy

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

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageDropBadColumns(DataclassRecord, BundleStep):
    """Drop columns with excessive sentinel fraction or long sentinel runs."""

    exempt_cols: list[str] = dataclass_field(default_factory=list)
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

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageImputeShortRuns(DataclassRecord, BundleStep):
    """Detect sentinel runs and interpolate short bad segments."""

    known_meta: KnownMeta
    exempt_cols: list[str] = dataclass_field(default_factory=list)
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
        imputed_df = masked_df.with_columns(
            [
                pl.col(c)
                .interpolate(method=self.known_meta.impute_method)
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                for c in cols
            ]
        )
        logger.debug("imputed short runs on %s columns", len(cols))
        db.df = imputed_df
        db.feature_names = list(imputed_df.columns)
        db.channels = len(imputed_df.columns)

        return db

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


def _detect_sentinel_runs(
    col_name: str, meta: KnownMeta, value_tol: float = 1e-6
) -> tuple[pl.Expr, pl.Expr]:
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
    cond_list = [
        (pl.col(col_name) - pl.lit(v)).abs() <= value_tol for v in meta.universal_sentinels
    ]
    if meta.sentinels and (vals := meta.sentinels.get(col_name)):
        cond_list.extend([(pl.col(col_name) - pl.lit(v)).abs() <= value_tol for v in vals])

    cond: pl.Expr = (
        pl.lit(False)
        if not cond_list
        else pl.fold(acc=pl.lit(False), function=lambda acc, e: acc | e, exprs=cond_list)
    )
    # 2. Build Run-Length
    change: pl.Expr = (cond != cond.shift(1)).fill_null(True)
    grp: pl.Expr = change.cast(pl.Int32).cum_sum()
    runlen: pl.Expr = pl.when(cond).then(pl.len().over(grp)).otherwise(0)

    return cond, runlen


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageConvertToNumpy(DataclassRecord, BundleStep):
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

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Split stages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageSplitByFraction(DataclassRecord, BundleStep):
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

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageComputeSplitStats(DataclassRecord, BundleStep):
    """Compute per-split statistics (mean, std, quantiles) for normalization."""

    ddof: int = 1

    def _get_stats(self, X: np.ndarray, begin_idx: int, end_idx: int) -> PartitionStats:
        view = X[begin_idx:end_idx]  # Create View (Zero Memory Cost)
        count = float(view.shape[0])

        mean = np.asarray(np.mean(view, axis=0, dtype=np.float32), dtype=np.float32)
        std = np.asarray(np.std(view, axis=0, ddof=self.ddof, dtype=np.float32), dtype=np.float32)

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

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Noise stages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _noise_row_slices(
    db: DataBundle, scheme: SplitTreatmentScheme, noise_on: NoisedSplits | None = None
) -> tuple[slice, ...]:
    """Resolve the only rows a noise stage is allowed to mutate: the named splits, else the scheme's."""
    if noise_on is not None:
        return noise_on.row_slices(db)
    if scheme is SplitTreatmentScheme.NOISE_MATCHED_RETRAINING:
        return (slice(None),)
    if scheme is SplitTreatmentScheme.CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION:
        if db.indices is None:
            raise RuntimeError(
                "clean_train_clean_select_test_titration requires split indices; "
                "run StageSplitByFraction before the noise stage"
            )
        test_start, test_end = db.indices[4], db.indices[5]
        return (slice(test_start, test_end),)
    if scheme is SplitTreatmentScheme.NOISE_TRAIN_NOISE_SELECT_CLEAN_TEST:
        if db.indices is None:
            raise RuntimeError(
                "noise_train_noise_select_clean_test requires split indices; "
                "run StageSplitByFraction before the noise stage"
            )
        train_start, train_end, val_start, val_end = db.indices[:4]
        return (slice(train_start, train_end), slice(val_start, val_end))
    raise ValueError(f"unsupported split treatment scheme: {scheme!r}")


class NoisedSplits(StructRecord, Struct, frozen=True, kw_only=False, eq=False, weakref=True, forbid_unknown_fields=True):
    """Which chronological splits receive the observation noise, named outright.

    The three scheme names above are three of the seven non-empty subsets of
    {train, val, test}; a card names the subset (owner 2026-09-04: "the card
    always says who has noise"), and each scheme is one of them.
    """

    train: bool
    val: bool
    test: bool

    NAMES = ("train", "val", "test")

    @classmethod
    def of(cls, names) -> "NoisedSplits":
        chosen = tuple(str(n) for n in names)
        unknown = [n for n in chosen if n not in cls.NAMES]
        if unknown or not chosen:
            raise ValueError(
                f"noise_on names a non-empty subset of {cls.NAMES}; got {list(chosen)!r}"
            )
        return cls(train="train" in chosen, val="val" in chosen, test="test" in chosen)

    @classmethod
    def of_scheme(cls, scheme: "SplitTreatmentScheme") -> "NoisedSplits":
        scheme = SplitTreatmentScheme(scheme)
        if scheme is SplitTreatmentScheme.NOISE_MATCHED_RETRAINING:
            return cls(train=True, val=True, test=True)
        if scheme is SplitTreatmentScheme.CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION:
            return cls(train=False, val=False, test=True)
        if scheme is SplitTreatmentScheme.NOISE_TRAIN_NOISE_SELECT_CLEAN_TEST:
            return cls(train=True, val=True, test=False)
        raise ValueError(f"unsupported split treatment scheme: {scheme!r}")

    def names(self) -> tuple[str, ...]:
        return tuple(n for n, on in zip(self.NAMES, (self.train, self.val, self.test)) if on)

    def row_slices(self, db: "DataBundle") -> tuple[slice, ...]:
        if self.train and self.val and self.test:
            return (slice(None),)
        if db.indices is None:
            raise RuntimeError(
                f"noise_on={list(self.names())} requires split indices; run StageSplitByFraction before the noise stage"
            )
        bounds = {"train": db.indices[0:2], "val": db.indices[2:4], "test": db.indices[4:6]}
        return tuple(slice(*bounds[name]) for name in self.names())

    __hash__ = record_hash


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageInjectNoise(DataclassRecord, BundleStep):
    """Inject unit-scaled observation noise into the scheme-selected rows.

    The historical scheme modifies the full ``[T, D]`` series.  The CSR
    scheme modifies only ``indices[4]:indices[5]`` (the test interval), so the
    training tensor, validation tensor, early stopping, and clean model
    selection remain uncontaminated by the test dose.
    """

    noise_level: float
    noise_draw: np.ndarray
    split_treatment_scheme: SplitTreatmentScheme = SplitTreatmentScheme.NOISE_MATCHED_RETRAINING
    # CHG 2026-09-04: the splits may be named outright; when given they decide the rows, and the
    # scheme is only recorded on the bundle for readers of the old word.
    noise_on: NoisedSplits | None = None

    def apply(self, db: DataBundle) -> DataBundle:
        db.split_treatment_scheme = self.split_treatment_scheme
        if self.noise_level == 0.0:
            db.injected_sigma = 0.0
            return db

        if db.series is None:
            raise RuntimeError("StageInjectNoise requires db.series")
        T, D = db.series.shape
        assert self.noise_draw.shape == (T, D), (
            f"noise_draw {self.noise_draw.shape} != data ({T}, {D})"
        )

        observed = db.series.copy()
        noise_norm_squared = 0.0
        for row_slice in _noise_row_slices(db, self.split_treatment_scheme, self.noise_on):
            noise = self.noise_level * self.noise_draw[row_slice]
            observed[row_slice] += noise
            noise_norm_squared += float(np.square(noise).sum())
        db.series = np.ascontiguousarray(observed, dtype=np.float32)
        db.injected_sigma = self.noise_level
        logger.debug(
            "injected noise: level=%.2e norm=%.2f",
            self.noise_level,
            math.sqrt(noise_norm_squared),
        )
        return db

    def __post_init__(self):
        # CHG: preserve converters/checks on direct Python construction.
        object.__setattr__(self, 'split_treatment_scheme', SplitTreatmentScheme(self.split_treatment_scheme))

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StageInjectTrainScaledNoise(DataclassRecord, BundleStep):
    """Inject noise scaled by clean-train channel standard deviations.

    Dose ``d`` means channelwise standard deviation ``d * train_std[j]``.
    Under the CSR scheme the scale is still estimated from the untouched clean
    training interval, and only test rows receive the resulting noise.  This
    is a deterministic data transformation, not a fitted loss, so it requires
    no gradient descent.
    """

    noise_level: float
    noise_draw: np.ndarray
    split_treatment_scheme: SplitTreatmentScheme = SplitTreatmentScheme.NOISE_MATCHED_RETRAINING
    # CHG 2026-09-04: the splits may be named outright; when given they decide the rows, and the
    # scheme is only recorded on the bundle for readers of the old word.
    noise_on: NoisedSplits | None = None

    def apply(self, db: DataBundle) -> DataBundle:
        db.split_treatment_scheme = self.split_treatment_scheme
        if self.noise_level == 0.0:
            db.injected_sigma = 0.0
            return db

        if db.series is None:
            raise RuntimeError("StageInjectTrainScaledNoise requires db.series")
        T, D = db.series.shape
        assert self.noise_draw.shape == (T, D), (
            f"noise_draw {self.noise_draw.shape} != data ({T}, {D})"
        )

        channel_scale = np.maximum(db.train_std.astype(np.float32, copy=False), 1e-8)
        observed = db.series.copy()
        noise_norm_squared = 0.0
        for row_slice in _noise_row_slices(db, self.split_treatment_scheme, self.noise_on):
            noise = self.noise_level * (channel_scale[None, :] * self.noise_draw[row_slice])
            observed[row_slice] += noise
            noise_norm_squared += float(np.square(noise).sum())
        db.series = np.ascontiguousarray(observed, dtype=np.float32)
        db.injected_sigma = self.noise_level
        logger.debug(
            "injected train-std-scaled noise: level=%.2e norm=%.2f",
            self.noise_level,
            math.sqrt(noise_norm_squared),
        )
        return db

    def __post_init__(self):
        # CHG: preserve converters/checks on direct Python construction.
        object.__setattr__(self, 'split_treatment_scheme', SplitTreatmentScheme(self.split_treatment_scheme))

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Diagnostics (see diagnostics.py for reports + renderers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from .diagnostics import print_diagnostics  # noqa: E402


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, weakref_slot=True)
class StagePrintDiagnostics(DataclassRecord, BundleStep):
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

    __hash__ = record_hash

    __setstate__ = DataclassRecord.__setstate__
