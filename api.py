"""Public API entry points for data-generation banks, intervention, and titration.

Functions:
    build_dgp_bank(source, seed) → DGPBank
    build_empirical_dgp_bank(values, name) → DGPBank
    inject_noise(bank, sigma, partition) → SplitTimeSeriesData
    iter_cells(bank, plan) → Iterator[TitrationCell]
    build_clean_split_bundle(bank, trajectory_draw_ordinal, partition_spec) → DataBundle
    build_noisy_bundle_for_cell(bank, db_clean, cell) → DataBundle
    export_dgp_bank_files(bank, out_dir) → dict
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
import copy
from decimal import Decimal
import json
import logging
from pathlib import Path
from typing import Literal
import numpy as np
import polars as pl

from .data import KnownCleanConfig, KnownLoadMode, SplitTimeSeriesData, load_data
from .pipeline import (
    DataBundle,
    KnownDataID,
    KnownMeta,
    PartitionSpec,
    DrawDesign,
    StageComputeSplitStats,
    StageInjectNoise,
    StageInjectTrainScaledNoise,
    StageLoadTensor,
    StagePrintDiagnostics,
    StageSplitByFraction,
    TitrationPlan,
)
from ._version import PROTOCOL_VERSION, __version__
from .specs import TitrationCell, TrajectorySpec, DGPBank, spawn_draw_seeds

logger = logging.getLogger(__name__)


def _sigma_to_name(value: float) -> str:
    decimal = Decimal(str(float(value))).normalize()
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text.replace("-", "neg").replace(".", "p")


def _write_npy(path: Path, array: np.ndarray, *, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return str(path)


def _feature_columns(array: np.ndarray, feature_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    if len(feature_names) != array.shape[-1]:
        feature_names = tuple(f"f{i}" for i in range(array.shape[-1]))
    return {name: array[..., i].reshape(-1) for i, name in enumerate(feature_names)}


def _write_rep_csv(
    path: Path,
    array: np.ndarray,
    *,
    rep_col: str,
    feature_names: tuple[str, ...],
    overwrite: bool,
) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    if array.ndim != 3:
        raise ValueError(f"rep CSV export expects [N, T, D], got {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rep, T, _ = array.shape
    cols: dict[str, np.ndarray] = {
        rep_col: np.repeat(np.arange(n_rep, dtype=np.int64), T),
        "t": np.tile(np.arange(T, dtype=np.int64), n_rep),
    }
    cols.update(_feature_columns(array, feature_names))
    pl.DataFrame(cols).write_csv(path)
    return str(path)


def _write_cell_csv(
    path: Path,
    array: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    overwrite: bool,
) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    if array.ndim != 2:
        raise ValueError(f"cell CSV export expects [T, D], got {array.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, np.ndarray] = {"t": np.arange(array.shape[0], dtype=np.int64)}
    cols.update(_feature_columns(array, feature_names))
    pl.DataFrame(cols).write_csv(path)
    return str(path)


def _write_json(path: Path, payload: dict[str, object], *, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _wants_format(array_format: ArrayFormat, candidate: Literal["npy", "csv"]) -> bool:
    return array_format == "both" or array_format == candidate


def _plan_counts_for_export(
    bank: DGPBank,
    design: DrawDesign,
) -> tuple[int, int]:
    if design is DrawDesign.AUTO:
        resolved = (
            DrawDesign.JOINT
            if bank.is_stochastic
            else DrawDesign.NOISE_ONLY
        )
    else:
        resolved = design
    if resolved in {DrawDesign.NOISE_ONLY, DrawDesign.TRAJECTORY_FIXED}:
        return 1, bank.n_noise_draws
    return bank.n_trajectory_draws, bank.n_noise_draws


def _resolve_source(source) -> TrajectorySpec | KnownMeta | KnownDataID:
    """Convert a string system name to TrajectorySpec via catalog preset, or pass through."""
    if isinstance(source, str):
        from .catalog import list_sources, source_preset
        try:
            return source_preset(source)()
        except ValueError as exc:
            valid = ", ".join(list_sources())
            raise ValueError(
                f"Unknown system name {source!r}. "
                f"Valid names: {valid}. "
                f"Or pass a TrajectorySpec, KnownMeta, or KnownDataID directly."
            ) from exc
    if isinstance(source, (TrajectorySpec, KnownMeta, KnownDataID)):
        return source
    raise TypeError(
        f"source must be str, TrajectorySpec, KnownMeta, or KnownDataID, got {type(source).__name__}. "
        f"Examples: build_dgp_bank('lorenz'), build_dgp_bank(TrajectorySpec.lorenz()), "
        f"build_dgp_bank(KnownMeta.etth1())"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# A custom noise sampler: (rng, shape) -> float32 array of that shape.
# Contract: the returned draw must come from the law the caller declares.
# Built-in laws are unit-variance by convention so `sigma` reads as an injected
# standard deviation. Custom laws may use any known scale; then `sigma` is a raw
# multiplier and the draw distribution owns the physical scale.
NoiseSampler = Callable[[np.random.Generator, tuple[int, ...]], np.ndarray]
SigmaScale = Literal["unit", "train_std"]
ArrayFormat = Literal["npy", "csv", "both"]

_NOISE_KINDS: dict[str, NoiseSampler] = {
    # N(0, 1) — the default of both papers' main protocol.
    "gaussian": lambda rng, shape: rng.standard_normal(shape, dtype=np.float32),
    # Laplace(0, 1/sqrt(2)) has variance one, like the Gaussian path, so
    # noise_level remains a std-scale intervention for heavy-tailed checks.
    "laplace": lambda rng, shape: rng.laplace(
        loc=0.0, scale=1.0 / np.sqrt(2.0), size=shape
    ).astype(np.float32),
}


def _draw_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    noise_kind: str | NoiseSampler,
) -> np.ndarray:
    """Draw one noise realization under the requested law."""
    if callable(noise_kind):
        draw = np.asarray(noise_kind(rng, shape), dtype=np.float32)
        if draw.shape != shape:
            raise ValueError(
                f"custom noise_kind returned shape {draw.shape}, expected {shape}. "
                f"The callable must produce one float32 draw per requested shape."
            )
        return draw
    if noise_kind not in _NOISE_KINDS:
        valid = ", ".join(sorted(_NOISE_KINDS))
        raise ValueError(
            f"unknown noise_kind {noise_kind!r}. Valid: {valid}. "
            f"Or pass a callable (rng, shape) -> float32 array."
        )
    return _NOISE_KINDS[noise_kind](rng, shape)


def _draw_noise_draws(
    *,
    noise_draw_seeds: tuple[int, ...],
    shape: tuple[int, int],
    noise_kind: str | NoiseSampler,
) -> np.ndarray:
    """Draw one independent noise realization per spawned child seed."""
    T, D = shape
    return np.stack(
        [
            _draw_noise(np.random.default_rng(seed), (T, D), noise_kind)
            for seed in noise_draw_seeds
        ]
    ).astype(np.float32, copy=False)


def _require_split_bundle(db: DataBundle, *, context: str) -> None:
    missing = []
    if db.series is None:
        missing.append("series")
    if db.feature_names is None:
        missing.append("feature_names")
    if db.indices is None:
        missing.append("indices")
    if db.train_partition_stats is None:
        missing.append("train_partition_stats")
    if db.val_partition_stats is None:
        missing.append("val_partition_stats")
    if db.test_partition_stats is None:
        missing.append("test_partition_stats")
    if missing:
        raise RuntimeError(f"{context} did not populate: {', '.join(missing)}")


def build_dgp_bank(
    source: str | TrajectorySpec | KnownMeta | KnownDataID,
    *,
    seed: int = 0,
    mode: KnownLoadMode = KnownLoadMode.CLEAN,
    cleaning: KnownCleanConfig | None = None,
    n_trajectory_draws: int = 1,
    n_noise_draws: int = 1,
    noise_seed_base: int = 1000,
    noise_kind: str | NoiseSampler = "gaussian",
    verbose: bool = False,
) -> DGPBank:
    """Generate clean trajectories and reusable noise draws once.

    Args:
        source: System to generate. Accepts:
            - str: system name like "lorenz", "ou", "slds" (uses catalog preset)
            - TrajectorySpec: explicit spec with custom parameters
            - KnownMeta / KnownDataID: a staged recorded dataset
        seed: Random seed for trajectory generation.
        n_trajectory_draws: Number of independent trajectories to generate.
        n_noise_draws: Number of independent noise draws to pre-generate.
        noise_seed_base: Base seed for noise draw generation.
        noise_kind: Noise law for the draws. The protocol
            works with ANY known noise distribution you can run statistical
            tests against — Gaussian is just the default. Accepts:
            - "gaussian": N(0, 1) (default)
            - "laplace": Laplace(0, 1/sqrt(2)) — unit variance, heavy tails
            - a callable (rng, shape) -> float32 array, for custom laws.
            Built-ins are unit-variance by convention so `sigma` reads as an
            injected standard deviation. Custom draws may have any known scale;
            then `sigma` is a raw multiplier and the draw distribution owns the
            scale. The hard requirement for NLL/PIT/coverage-style tests is
            that the injected law is exactly known and reported.

    Returns:
        DGPBank with .clean [N_trajectory, T, D] and .noise_draws [N_noise, T, D].

    Example:
        >>> bank = build_dgp_bank("lorenz", seed=42)
        >>> bank.T, bank.D    # (36000, 3)
        >>> bank = build_dgp_bank(TrajectorySpec.ou(steps=5000), seed=0)
        >>> bank.T, bank.D    # (5001, 1)
        >>> bank = build_dgp_bank("ou", noise_kind="laplace")    # heavy-tailed
    """
    source = _resolve_source(source)

    if isinstance(source, TrajectorySpec):
        trajectory_draw_seeds = spawn_draw_seeds(seed, n_trajectory_draws)
        series = [load_data(source, seed=s) for s in trajectory_draw_seeds]
        trajectory_seed_base = seed
    elif isinstance(source, (KnownMeta, KnownDataID)):
        if n_trajectory_draws != 1:
            raise ValueError(
                f"Known data supports exactly n_trajectory_draws=1, got {n_trajectory_draws}. "
                f"Known datasets are fixed recordings, not generative."
            )
        series = [load_data(source, mode=mode, cleaning=cleaning)]
        trajectory_seed_base = 0
        trajectory_draw_seeds = (0,)
    else:
        raise TypeError(f"Unsupported source type after coercion: {type(source)!r}")

    clean = np.stack(
        [np.ascontiguousarray(ds.values, dtype=np.float32) for ds in series]
    )
    _, T, D = clean.shape
    noise_draw_seeds = spawn_draw_seeds(noise_seed_base, n_noise_draws)
    noise_draws = _draw_noise_draws(
        noise_draw_seeds=noise_draw_seeds,
        shape=(T, D),
        noise_kind=noise_kind,
    )

    if verbose:
        logger.info(
            "built dgp bank %s: clean=%s noise=%s",
            series[0].source_id,
            list(clean.shape),
            list(noise_draws.shape),
        )

    return DGPBank(
        name=series[0].source_id,
        trajectory_seed_base=trajectory_seed_base,
        noise_seed_base=noise_seed_base,
        trajectory_draw_seeds=trajectory_draw_seeds,
        noise_draw_seeds=noise_draw_seeds,
        clean=clean,
        noise_draws=noise_draws,
        feature_names=tuple(series[0].feature_names),
        is_stochastic=source.system.is_stochastic if isinstance(source, TrajectorySpec) else False,
        noise_kind=noise_kind if isinstance(noise_kind, str) else "custom",
    )


def build_empirical_dgp_bank(
    values: np.ndarray,
    *,
    name: str,
    n_noise_draws: int = 1,
    noise_seed_base: int = 1000,
    noise_kind: str | NoiseSampler = "gaussian",
    feature_names: tuple[str, ...] | None = None,
) -> DGPBank:
    """Wrap a clean recorded series as a one-trajectory-draw DGP bank.

    The recording is treated as the clean reference system. Use
    `inject_noise(..., sigma_scale="train_std")` when sigma should be relative to
    each channel's train-set scale.
    """
    data = np.asarray(values, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"values must have shape [T, D], got {data.shape}")
    if data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError(f"values must have at least 2 rows and 1 channel, got {data.shape}")

    clean = np.ascontiguousarray(data, dtype=np.float32)[None, :, :]
    _, T, D = clean.shape
    trajectory_draw_seeds = spawn_draw_seeds(0, 1)
    noise_draw_seeds = spawn_draw_seeds(noise_seed_base, n_noise_draws)
    noise_draws = _draw_noise_draws(
        noise_draw_seeds=noise_draw_seeds,
        shape=(T, D),
        noise_kind=noise_kind,
    )
    names = feature_names or tuple(f"f{i}" for i in range(D))
    if len(names) != D:
        raise ValueError(f"feature_names has length {len(names)}, expected D={D}")

    return DGPBank(
        name=name,
        trajectory_seed_base=0,
        noise_seed_base=noise_seed_base,
        trajectory_draw_seeds=trajectory_draw_seeds,
        noise_draw_seeds=noise_draw_seeds,
        clean=clean,
        noise_draws=noise_draws,
        feature_names=tuple(names),
        is_stochastic=False,
        noise_kind=noise_kind if isinstance(noise_kind, str) else "custom",
    )


def export_dgp_bank_files(
    bank: DGPBank,
    out_dir: str | Path,
    *,
    sigmas: tuple[float, ...] = (),
    array_format: ArrayFormat = "both",
    draw_design: DrawDesign | str = DrawDesign.AUTO,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write DGP-bank clean/noise arrays, optional noisy cells, and a manifest.

    This is an explicit audit/export path. Normal `build_dgp_bank(...)` stays
    memory-only; call this when you want generated trajectories on disk for
    inspection, cross-tool checks, or reproducibility review.

    Files:
    - `clean.npy` / `clean.csv`: clean trajectories
    - `noise_draws.npy` / `noise_draws.csv`: frozen noise draws
    - `noisy/*`: optional cells selected by `draw_design`
    - `manifest.json`: provenance and relative file names
    """
    if array_format not in {"npy", "csv", "both"}:
        raise ValueError("array_format must be 'npy', 'csv', or 'both'")
    design = DrawDesign(draw_design)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sigma_values = tuple(float(sigma) for sigma in sigmas)

    files: dict[str, object] = {}
    if _wants_format(array_format, "npy"):
        files["clean_npy"] = Path(
            _write_npy(out / "clean.npy", bank.clean, overwrite=overwrite)
        ).name
        files["noise_draws_npy"] = Path(
            _write_npy(out / "noise_draws.npy", bank.noise_draws, overwrite=overwrite)
        ).name
    if _wants_format(array_format, "csv"):
        files["clean_csv"] = Path(
            _write_rep_csv(
                out / "clean.csv",
                bank.clean,
                rep_col="trajectory_draw_ordinal",
                feature_names=bank.feature_names,
                overwrite=overwrite,
            )
        ).name
        files["noise_draws_csv"] = Path(
            _write_rep_csv(
                out / "noise_draws.csv",
                bank.noise_draws,
                rep_col="noise_draw_ordinal",
                feature_names=bank.feature_names,
                overwrite=overwrite,
            )
        ).name

    noisy_entries: list[dict[str, object]] = []
    n_trajectory_export, n_noise_export = _plan_counts_for_export(bank, design)
    export_cells = TitrationPlan(
        noise_levels=sigma_values or (0.0,),
        n_trajectory_draws=n_trajectory_export,
        n_noise_draws=n_noise_export,
        noise_seed_base=bank.noise_seed_base,
        draw_design=design,
    )
    for sigma in sigma_values:
        sigma_name = _sigma_to_name(sigma)
        for cell in export_cells.iter_cells(bank):
            if cell.noise_level != sigma:
                continue
            noisy = bank.build_observed_series(
                noise_level=sigma,
                trajectory_draw_ordinal=cell.trajectory_draw_ordinal,
                noise_draw_ordinal=cell.noise_draw_ordinal,
            )
            entry_files: dict[str, str] = {}
            stem = (
                f"{cell.source_name}_traj{cell.trajectory_draw_ordinal}"
                f"_sig{sigma_name}_noise{cell.noise_draw_ordinal}"
            )
            if _wants_format(array_format, "npy"):
                rel_path = Path("noisy") / f"{stem}.npy"
                _write_npy(out / rel_path, noisy, overwrite=overwrite)
                entry_files["npy"] = str(rel_path)
            if _wants_format(array_format, "csv"):
                rel_path = Path("noisy") / f"{stem}.csv"
                _write_cell_csv(
                    out / rel_path,
                    noisy,
                    feature_names=bank.feature_names,
                    overwrite=overwrite,
                )
                entry_files["csv"] = str(rel_path)
            noisy_entries.append(
                {
                    "files": entry_files,
                    "cell_tag": cell.tag,
                    "sigma": sigma,
                    "trajectory_draw_ordinal": cell.trajectory_draw_ordinal,
                    "noise_draw_ordinal": cell.noise_draw_ordinal,
                    "trajectory_draw_seed": int(cell.trajectory_draw_seed),
                    "noise_draw_seed": int(cell.noise_draw_seed),
                }
            )

    manifest: dict[str, object] = {
        "schema": "serie.dgp_bank_file_export.v1",
        "serie_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "source_name": bank.name,
        "noise_kind": bank.noise_kind,
        "is_stochastic": bool(bank.is_stochastic),
        "draw_design": design.value,
        "array_format": array_format,
        "shape": {
            "clean": list(bank.clean.shape),
            "noise_draws": list(bank.noise_draws.shape),
        },
        "feature_names": list(bank.feature_names),
        "trajectory_seed_base": int(bank.trajectory_seed_base),
        "noise_seed_base": int(bank.noise_seed_base),
        "trajectory_draw_seeds": [int(seed) for seed in bank.trajectory_draw_seeds],
        "noise_draw_seeds": [int(seed) for seed in bank.noise_draw_seeds],
        "files": files,
        "noisy": noisy_entries,
    }
    _write_json(out / "manifest.json", manifest, overwrite=overwrite)
    return manifest


def _inject_noise_unit(
    bank: DGPBank,
    *,
    noise_level: float = 0.0,
    partition: PartitionSpec,
    trajectory_draw_ordinal: int = 0,
    noise_draw_ordinal: int = 0,
    print_diagnostics: bool = False,
) -> SplitTimeSeriesData:
    """Unit-scaled observation-noise injection implementation."""
    cell = bank.cell(
        noise_level=noise_level,
        trajectory_draw_ordinal=trajectory_draw_ordinal,
        noise_draw_ordinal=noise_draw_ordinal,
    )
    noisy = np.ascontiguousarray(
        bank.build_observed_series(
            noise_level=noise_level,
            trajectory_draw_ordinal=trajectory_draw_ordinal,
            noise_draw_ordinal=noise_draw_ordinal,
        ),
        dtype=np.float32,
    )
    db = (
        DataBundle()
        | StageLoadTensor(
            data=noisy,
            dataset_id=cell.tag,
            feature_names=bank.feature_names,
        )
        | StageSplitByFraction(partition_spec=partition)
        | StageComputeSplitStats()
    )
    if print_diagnostics:
        db = db | StagePrintDiagnostics()

    _require_split_bundle(db, context="unit noise injection")

    return SplitTimeSeriesData(
        values=db.series,
        feature_names=tuple(db.feature_names),
        source_id=bank.name,
        partition=partition,
        indices=db.indices,
        train_stats=db.train_partition_stats,
        val_stats=db.val_partition_stats,
        test_stats=db.test_partition_stats,
        cell=cell,
        metadata={
            "source_name": bank.name,
            "serie_version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "cell_tag": cell.tag,
            "sigma": float(cell.noise_level),
            "sigma_scale": "unit",
            "trajectory_draw_ordinal": int(cell.trajectory_draw_ordinal),
            "noise_draw_ordinal": int(cell.noise_draw_ordinal),
            "trajectory_seed_base": int(cell.trajectory_seed_base),
            "noise_seed_base": int(cell.noise_seed_base),
            "trajectory_draw_seed": int(cell.trajectory_draw_seed),
            "noise_draw_seed": int(cell.noise_draw_seed),
        },
    )


def inject_noise(
    bank: DGPBank,
    *,
    sigma: float = 0.0,
    sigma_scale: SigmaScale = "unit",
    partition: PartitionSpec | None = None,
    trajectory_draw_ordinal: int = 0,
    noise_draw_ordinal: int = 0,
    print_diagnostics: bool = False,
) -> SplitTimeSeriesData:
    """Inject one observation-noise sigma and return train/val/test arrays."""
    partition = partition or PartitionSpec.p70_20_10()
    if sigma_scale == "unit":
        return _inject_noise_unit(
            bank,
            noise_level=sigma,
            partition=partition,
            trajectory_draw_ordinal=trajectory_draw_ordinal,
            noise_draw_ordinal=noise_draw_ordinal,
            print_diagnostics=print_diagnostics,
        )
    if sigma_scale != "train_std":
        raise ValueError("sigma_scale must be 'unit' or 'train_std'")

    # train_std scaling requires TWO passes:
    #   1. Build a clean split to get per-channel train std
    #   2. Inject noise scaled by that std
    # This is why the sigma_scale="unit" path above is simpler (one pass).
    cell = bank.cell(
        noise_level=sigma,
        trajectory_draw_ordinal=trajectory_draw_ordinal,
        noise_draw_ordinal=noise_draw_ordinal,
    )
    db_clean = build_clean_split_bundle(
        bank,
        trajectory_draw_ordinal=trajectory_draw_ordinal,
        partition_spec=partition,
        print_diagnostics=False,
    )
    db_noisy = build_noisy_bundle_for_cell(
        bank,
        db_clean,
        cell,
        scale_noise_by_std=True,
    )
    db_noisy = db_noisy | StageComputeSplitStats()
    if print_diagnostics:
        db_noisy = db_noisy | StagePrintDiagnostics()

    _require_split_bundle(db_noisy, context="train-std noise injection")

    return SplitTimeSeriesData(
        values=db_noisy.series,
        feature_names=tuple(db_noisy.feature_names),
        source_id=bank.name,
        partition=partition,
        indices=db_noisy.indices,
        train_stats=db_noisy.train_partition_stats,
        val_stats=db_noisy.val_partition_stats,
        test_stats=db_noisy.test_partition_stats,
        cell=cell,
        metadata={
            "source_name": bank.name,
            "serie_version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "cell_tag": cell.tag,
            "sigma": float(cell.noise_level),
            "sigma_scale": "train_std",
            "trajectory_draw_ordinal": int(cell.trajectory_draw_ordinal),
            "noise_draw_ordinal": int(cell.noise_draw_ordinal),
            "trajectory_seed_base": int(cell.trajectory_seed_base),
            "noise_seed_base": int(cell.noise_seed_base),
            "trajectory_draw_seed": int(cell.trajectory_draw_seed),
            "noise_draw_seed": int(cell.noise_draw_seed),
        },
    )


def iter_cells(bank: DGPBank, plan: TitrationPlan) -> Iterator[TitrationCell]:
    """Yield the `(trajectory_draw_ordinal, noise_level, noise_draw_ordinal)` cells selected by a plan.

    Example:
        >>> plan = TitrationPlan(noise_levels=(0.1, 1.0), n_trajectory_draws=1, n_noise_draws=1)
        >>> for cell in iter_cells(bank, plan):
        ...     print(cell.tag)
    """
    yield from plan.iter_cells(bank)


def build_clean_split_bundle(
    bank: DGPBank,
    trajectory_draw_ordinal: int,
    partition_spec: PartitionSpec,
    *,
    print_diagnostics: bool = True,
) -> DataBundle:
    """Return a clean NumPy `DataBundle` after split and split-stat stages.

    Stage:
    - input: one clean trajectory `bank.clean[trajectory_draw_ordinal]` as NumPy `[T, D]`
    - output: `DataBundle` with `series`, split indices, and split stats

    Not included:
    - no noise injection
    - no windowing
    - no dataloaders

    Set `print_diagnostics=False` to skip the render side effect.
    """
    clean = bank.clean[trajectory_draw_ordinal]
    db = (
        DataBundle()
        | StageLoadTensor(
            data=clean,
            dataset_id=f"{bank.name}_traj{trajectory_draw_ordinal}",
            feature_names=bank.feature_names,
        )
        | StageSplitByFraction(partition_spec=partition_spec)
        | StageComputeSplitStats()
    )
    db.trajectory_draw_ordinal = trajectory_draw_ordinal
    if print_diagnostics:
        db = db | StagePrintDiagnostics()
    return db


def build_noisy_bundle_for_cell(
    bank: DGPBank,
    db_clean: DataBundle,
    cell: TitrationCell,
    *,
    scale_noise_by_std: bool = False,
) -> DataBundle:
    """Return a noisy NumPy `DataBundle` before any downstream windowing.

    Stage:
    - input: clean split/stats `DataBundle`
    - output: same bundle shape, but `series` now holds the noisy `[T, D]`

    Not included:
    - no windowing
    - no dataloaders
    - no trainer-specific masks
    """
    if db_clean.trajectory_draw_ordinal is not None and db_clean.trajectory_draw_ordinal != cell.trajectory_draw_ordinal:
        raise ValueError(
            "clean bundle/cell draw mismatch: "
            f"db_clean.trajectory_draw_ordinal={db_clean.trajectory_draw_ordinal}, cell.trajectory_draw_ordinal={cell.trajectory_draw_ordinal}. "
            "Build or fetch the clean split for the same cell.trajectory_draw_ordinal before injecting noise."
        )
    db = copy.copy(db_clean)
    noise_draw = np.ascontiguousarray(bank.noise_draws[cell.noise_draw_ordinal], dtype=np.float32)

    if scale_noise_by_std:
        db = db | StageInjectTrainScaledNoise(
            noise_level=float(cell.noise_level),
            noise_draw=noise_draw,
        )
    else:
        db = db | StageInjectNoise(
            noise_level=float(cell.noise_level),
            noise_draw=noise_draw,
        )
    return db
