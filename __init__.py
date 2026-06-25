"""SERIE — stress-test time-series forecasters with controlled noise and interventions.
SERIE——用受控噪声和干预对时间序列预测器进行压力测试。

Public API (also re-exported from ``serie``):
公共 API（也从 ``serie`` 重新导出）：

    build_dgp_bank(source, seed)      → DGPBank   (build once / 构建一次)
    build_empirical_dgp_bank(values)  → DGPBank   (wrap real data / 包装真实数据)
    inject_noise(bank, sigma=...)     → SplitTimeSeriesData  (inject many times / 多次注入)
    export_dgp_bank_files(bank, path) → dict       (write arrays to disk / 写数组到磁盘)
    load_data(source)                 → TimeSeriesData  (load one source / 加载一个数据源)

DGP = Data Generating Process (数据生成过程).

Typical usage / 典型用法:

    from serie import PartitionSpec, TrajectorySpec, build_dgp_bank, inject_noise

    bank = build_dgp_bank("lorenz", seed=0)
    split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())

Systems / 系统: "lorenz", "rossler", "chua", "lorenz96", "ou", "doublewell",
                "slds", "seasonal_ar", "garch", "ks", "diagonal"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .data import KnownCleanConfig, KnownLoadMode
from ._version import PROTOCOL_VERSION, __version__


def load_data(source, **kw) -> "TimeSeriesData":
    """Load one source into a TimeSeriesData array.
    将一个数据源加载为 TimeSeriesData 数组。

    Args:
        source: TrajectorySpec, KnownMeta, or KnownDataID.
                TrajectorySpec、KnownMeta 或 KnownDataID。
        seed: Random seed (for synthetic sources). 随机种子（用于合成数据源）。

    Returns / 返回:
        TimeSeriesData with .values [T, D], .feature_names, .source_id.
        包含 .values [T, D]、.feature_names、.source_id 的 TimeSeriesData。

    Example / 示例:
        >>> ts = load_data(TrajectorySpec.lorenz(steps=1000), seed=0)
        >>> ts.values.shape    # (1001, 3)
    """
    from .data import load_data as _load_data

    return _load_data(source, **kw)


def build_dgp_bank(source, **kw) -> "DGPBank":
    """Build a DGP (Data Generating Process) bank: clean trajectories + frozen noise.
    构建一个 DGP（数据生成过程）bank：干净轨迹 + 冻结噪声。

    This is the expensive call — do it once, then call inject_noise() many times.
    这是昂贵的调用——只做一次，然后多次调用 inject_noise()。

    Args:
        source: System to generate. 要生成的系统。
            str ("lorenz", "ou", ...), TrajectorySpec, KnownMeta, or KnownDataID.
        seed: Random seed for trajectory generation. 轨迹生成的随机种子。
        n_trajectory_draws: Number of independent trajectories. 独立轨迹数量。
        n_noise_draws: Number of reusable noise draws. 可重用噪声抽取数量。

    Returns / 返回:
        DGPBank with .clean [N_trajectory, T, D] and .noise_draws [N_noise, T, D].
        包含 .clean [N_trajectory, T, D] 和 .noise_draws [N_noise, T, D] 的 DGPBank。

    Example / 示例:
        >>> bank = build_dgp_bank("lorenz", seed=42)
        >>> bank.T, bank.D    # (36000, 3)
        >>> bank = build_dgp_bank(TrajectorySpec.ou(steps=5000), seed=0)
    """
    from .api import build_dgp_bank as _build_dgp_bank

    return _build_dgp_bank(source, **kw)


def build_empirical_dgp_bank(values, **kw) -> "DGPBank":
    """Wrap a clean recorded [T, D] array as a DGP (Data Generating Process) bank.
    将一个干净的录制 [T, D] 数组包装为 DGP（数据生成过程）bank。

    Use this when you have your own real data (wearable sensors, production series,
    gait recordings) and want to stress-test a model against controlled noise.
    当你有自己的真实数据（可穿戴传感器、生产序列、步态录制）并想用受控噪声
    对模型进行压力测试时使用。

    Args:
        values: Clean time series, shape [T, D] (NumPy array).
                干净时间序列，形状 [T, D]（NumPy 数组）。
        name: Human-readable label for this recording. 此录制的可读标签。
        n_noise_draws: Number of frozen noise draws to pre-generate. 预生成的冻结噪声抽取数量。
        noise_seed_base: Base seed for noise generation. 噪声生成的基础种子。
        noise_kind: "gaussian", "laplace", or callable. 噪声类型。

    Returns / 返回:
        DGPBank with one trajectory draw (your data) and frozen noise draws.
        包含一个轨迹抽取（你的数据）和冻结噪声抽取的 DGPBank。

    Example / 示例:
        >>> bank = build_empirical_dgp_bank(my_array, name="wearable_gait")
        >>> split = inject_noise(bank, sigma=0.5, sigma_scale="train_std")
    """
    from .api import build_empirical_dgp_bank as _build_empirical_dgp_bank

    return _build_empirical_dgp_bank(values, **kw)


def inject_noise(bank, **kw) -> "SplitTimeSeriesData":
    """Inject observation noise and return one train/val/test split.
    注入观测噪声并返回一个训练/验证/测试分割。

    This is the cheap, repeatable call — use the same bank for every sigma.
    这是便宜且可重复的调用——对每个 sigma 使用相同的 bank。

    Args:
        bank: DGPBank from build_dgp_bank(). 来自 build_dgp_bank() 的 DGPBank。
        sigma: Noise amplitude (0.0 = clean). 噪声幅度（0.0 = 干净）。
        sigma_scale: "unit" (raw sigma) or "train_std" (scale per channel by
            training-set std). "unit"（原始sigma）或 "train_std"（按训练集标准差
            逐通道缩放）。
        partition: PartitionSpec — use presets like .p70_20_10().
                   PartitionSpec——使用预设如 .p70_20_10()。

    Returns / 返回:
        SplitTimeSeriesData with .train_values, .val_values, .test_values,
        .train_stats.mean, .train_stats.std, etc.
        包含 .train_values、.val_values、.test_values、
        .train_stats.mean、.train_stats.std 等的 SplitTimeSeriesData。

    Example / 示例:
        >>> split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
        >>> split.train_values.shape    # (25200, 3)
        >>> split.train_stats.std       # array([12.3, 8.7, 9.1])
    """
    from .api import inject_noise as _inject_noise

    return _inject_noise(bank, **kw)


def export_dgp_bank_files(bank, out_dir, **kw) -> dict[str, object]:
    """Write one DGP (Data Generating Process) bank's arrays and manifest to disk.
    将一个 DGP（数据生成过程）bank 的数组和清单写入磁盘。

    Writes clean.npy/csv, noise_draws.npy/csv, optional noisy cells, and
    manifest.json for inspection and reproducibility review.
    写入 clean.npy/csv、noise_draws.npy/csv、可选的含噪单元和
    manifest.json 以供检查和可复现性审查。
    """
    from .api import export_dgp_bank_files as _export_dgp_bank_files

    return _export_dgp_bank_files(bank, out_dir, **kw)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lazy exports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if TYPE_CHECKING:
    from .data import SplitTimeSeriesData, SourceKind, TimeSeriesData
    from .pipeline import (
        DataBundle,
        KnownDataID,
        KnownMeta,
        PartitionSpec,
        DrawDesign,
        TitrationPlan,
    )
    from .specs import ShockEvent, ShockSchedule, ShockType, TitrationCell, TrajectorySpec, DGPBank
    from .api import build_clean_split_bundle, build_noisy_bundle_for_cell, iter_cells


def __getattr__(name: str):
    if name in {
        "TrajectorySpec",
        "DGPBank",
        "TitrationCell",
        "ShockEvent",
        "ShockSchedule",
        "ShockType",
    }:
        from . import specs as _specs

        return getattr(_specs, name)
    if name in {
        "TimeSeriesData",
        "SplitTimeSeriesData",
        "KnownLoadMode",
        "SourceKind",
        "KnownCleanConfig",
    }:
        from . import data as _data

        return getattr(_data, name)
    if name in {
        "DataBundle",
        "KnownDataID",
        "KnownMeta",
        "PartitionSpec",
        "DrawDesign",
        "TitrationPlan",
    }:
        from . import pipeline as _pipeline

        return getattr(_pipeline, name)
    if name in {
        "iter_cells",
        "build_clean_split_bundle",
        "build_noisy_bundle_for_cell",
        "export_dgp_bank_files",
    }:
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
