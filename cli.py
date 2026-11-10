"""SERIE command-line surface."""

from __future__ import annotations

from msgspec.structs import replace

from dataclasses import dataclass
from serie.records import DataclassRecord

import contextlib
import io
import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
from typing import Annotated, Any, Literal, cast

import cappa
import msgspec
import numpy as np

from .controller_fork_bank import RecordedReferenceLayout

from . import (
    PartitionSpec,
    build_dgp_bank,
    build_empirical_dgp_bank,
    export_dgp_bank_files,
    inject_noise,
)
from .canonical_json import canonical_hash_16, canonical_value
from .catalog import list_sources, source_preset
from .pipeline import KnownDataID, DrawDesign, known_data_root
from .specs import ShockSchedule, TrajectorySpec


OutputFormat = Literal["text", "json"]
PlotKind = Literal["timeseries", "attractor", "both"]
PlotRequest = Literal["none", "timeseries", "attractor", "both"]
PlotFormat = Literal["png", "pdf", "svg"]
ArrayFormat = Literal["npy", "csv", "ipc", "parquet", "table", "safetensors", "all"]
ArrayRequest = Literal["none", "npy", "csv", "ipc", "parquet", "table", "safetensors", "all"]
ShockRequest = Literal["none", "param", "state", "switch"]
_DEMO_SHAPE_SOURCES = ("lorenz", "rossler", "chua")
_DEMO_SOURCES = ("lorenz", "rossler", "chua", "vanderpol", "henon", "logistic")
_DEMO_CANDIDATE_SOURCES: tuple[str, ...] = ("garch",)
_DEMO_SOURCE_GROUPS = {
    "shapes": _DEMO_SHAPE_SOURCES,
    "all": _DEMO_SOURCES,
}


_PARTITIONS = {
    "p70_10_20": PartitionSpec.p70_10_20,
    "p70_20_10": PartitionSpec.p70_20_10,
    "p60_20_20": PartitionSpec.p60_20_20,
    "p50_25_25": PartitionSpec.p50_25_25,
    "p60_15_25": PartitionSpec.p60_15_25,
    "p60_10_30": PartitionSpec.p60_10_30,
    "p50_10_40": PartitionSpec.p50_10_40,
    "p70_gap_10_20": PartitionSpec.p70_gap_10_20,
}

_CLI_COMMANDS = (
    {
        "group": "01 discovery",
        "command": "list-cli-commands",
        "summary": "List SERIE CLI commands with examples.",
        "example": "serie list-cli-commands --format json",
    },
    {
        "group": "01 discovery",
        "command": "list-data-ids",
        "summary": "List built-in synthetic sources and known external dataset names.",
        "example": "serie list-data-ids --format json",
    },
    {
        "group": "01 discovery",
        "command": "show-data-dir",
        "summary": "Print the external dataset directory used for known datasets.",
        "example": "serie show-data-dir --format json",
    },
    {
        "group": "02 generate",
        "command": "generate",
        "summary": "Generate one SERIE cell: optional intervention, noise dose, split, and optional files.",
        "example": "serie generate lorenz --shock param --sigma 0.1 --out tmp/lorenz --plot both --array-format csv",
    },
    {
        "group": "02 generate",
        "command": "freeze-controller-candidate-bank",
        "summary": "Freeze all legal Reacher gear forks from matched controller recordings.",
        "example": "serie freeze-controller-candidate-bank --source-root results/corpus/controller_recordings/reacher-47s-sine-match --factors 0.5,1,1.5 --history 336 --horizon 192 --out results/corpus/controller_recordings/reacher-47s-sine-match/candidate-origins-l336-h192.npz",
    },
    {
        "group": "02 generate",
        "command": "freeze-recorded-controller-references",
        "summary": "Extract actual TRAIN state/action transitions from existing gear recordings without simulation.",
        "example": "serie freeze-recorded-controller-references --recordings GEAR05/manifest.json,GEAR1/manifest.json,GEAR15/manifest.json --factors 0.5,1,1.5 --out REFERENCES.npz",
    },
    {
        "group": "02 generate",
        "command": "freeze-controller-reference-derivatives",
        "summary": "Freeze physical-direction observation responses from foreign TRAIN episodes.",
        "example": "serie freeze-controller-reference-derivatives --recording RECORDING/manifest.json --reference-anchor 48 --factors 0.5,1,1.5 --epsilons 0.001,0.0001,0.00001 --selected-epsilon 0.0001 --direction-seed 12323 --out REFERENCE.npz",
    },
    {
        "group": "03 visualize",
        "command": "demo",
        "summary": "Generate an interactive 3D attractor HTML page with controllable noise.",
        "example": "serie demo lorenz --out-html lorenz_demo.html",
    },
    {
        "group": "05 audit",
        "command": "walkthrough",
        "summary": "Author/reviewer audit: generate arrays, plots, and verification metadata.",
        "example": "serie walkthrough --out tmp/walkthrough --array-format all --overwrite --format json",
    },
    {
        "group": "06 metadata",
        "command": "version",
        "summary": "Print the installed SERIE package version.",
        "example": "serie version",
    },
)


def _configure_logging() -> None:
    """Configure human-friendly CLI logs without changing library behavior."""
    root = logging.getLogger()
    if root.handlers:
        return
    try:
        from rich.logging import RichHandler
    except ImportError:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)],
    )


def _array_summary(values: Any, *, limit: int = 8) -> list[float]:
    flat = values.reshape(-1)
    return [float(x) for x in flat[:limit]]


def _require_partition(name: str) -> str:
    if name not in _PARTITIONS:
        valid = ", ".join(sorted(_PARTITIONS))
        raise SystemExit(f"unknown partition {name!r}; valid partitions: {valid}")
    return name


def _path_component(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_=-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "_adhoc"


_SHOCK_SUFFIX_BY_REQUEST: dict[ShockRequest, str] = {
    "none": "",
    "param": "_param_shock",
    "state": "_state_shock",
    "switch": "_switch",
}


def _normalize_source_name(source: str) -> str:
    return source.lower().replace("-", "_")


def _resolve_generate_source(source: str, *, shock: ShockRequest) -> tuple[str, TrajectorySpec]:
    """Resolve CLI source + intervention flag into a concrete trajectory spec."""
    from . import catalog

    base = _normalize_source_name(source)
    if shock == "none":
        try:
            return base, source_preset(base)()
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    suffix = _SHOCK_SUFFIX_BY_REQUEST[shock]
    target = base if base.endswith(suffix) else f"{base}{suffix}"
    factory_name = f"{target}_spec"
    factory = getattr(catalog, factory_name, None)
    if factory is None:
        try:
            return target, source_preset(target)()
        except ValueError as exc:
            valid = ", ".join(list_sources())
            raise SystemExit(
                f"source {source!r} has no built-in {shock!r} intervention preset; "
                f"valid base sources: {valid}"
            ) from exc
    return target, factory()


def _reference_source_for_plot(source: TrajectorySpec) -> TrajectorySpec | None:
    if source.shocks.is_stationary:
        return None
    return replace(source, shocks=ShockSchedule.none())


def _dgp_bank_metadata(bank) -> dict[str, Any]:
    return {
        "name": bank.name,
        "noise_kind": bank.noise_kind,
        "T": int(bank.T),
        "D": int(bank.D),
        "n_trajectory_draws": int(bank.n_trajectory_draws),
        "n_noise_draws": int(bank.n_noise_draws),
        "is_stochastic": bool(bank.is_stochastic),
        "trajectory_seed_base": int(bank.trajectory_seed_base),
        "noise_seed_base": int(bank.noise_seed_base),
        "trajectory_draw_seeds": list(bank.trajectory_draw_seeds),
        "noise_draw_seeds": list(bank.noise_draw_seeds),
        "feature_names": list(bank.feature_names),
    }


def _generate_cell_identity(
    *,
    source_key: str,
    source_spec: TrajectorySpec,
    shock: ShockRequest,
    sigma: float,
    sigma_scale: Literal["unit", "train_std"],
    partition_name: str,
    seed: int,
    n_trajectory_draws: int,
    n_noise_draws: int,
    noise_seed_base: int,
    noise_kind: str,
    array_format: ArrayRequest,
    draw_design: DrawDesign | str,
    plot: PlotRequest,
    series_feature_idx: int,
    attractor_feature_idxs: tuple[int, int, int],
    max_points: int,
    image_format: PlotFormat,
    dpi: int,
) -> dict[str, Any]:
    """Path-independent identity for one generated SERIE cell and artifact request."""
    return {
        "schema": "serie.generate_cell_identity.v1",
        "source": {
            "requested": source_key,
            "resolved": source_spec.tag,
            "spec": canonical_value(source_spec),
        },
        "cell": {
            "shock": shock,
            "sigma": float(sigma),
            "sigma_scale": sigma_scale,
            "partition": partition_name,
            "seed": int(seed),
            "n_trajectory_draws": int(n_trajectory_draws),
            "n_noise_draws": int(n_noise_draws),
            "noise_seed_base": int(noise_seed_base),
            "noise_kind": noise_kind,
        },
        "artifacts": {
            "array_format": array_format,
            "draw_design": None
            if array_format == "none"
            else DrawDesign(draw_design).value,
            "plot": plot,
            "plot_options": None
            if plot == "none"
            else {
                "series_feature_idx": int(series_feature_idx),
                "attractor_feature_idxs": [int(idx) for idx in attractor_feature_idxs],
                "max_points": int(max_points),
                "image_format": image_format,
                "dpi": int(dpi),
            },
        },
    }


def _write_generate_cell_identity(out: Path, identity: dict[str, Any], cell_hash: str) -> Path:
    path = out / "serie_cell_identity.json"
    record = {
        "schema": "serie.generate_cell_invocation.v1",
        "cell_hash": cell_hash,
        "cell_hash_algorithm": "sha256[:16]",
        "identity": identity,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _split_summary_payload(
    *,
    source_key: str,
    bank,
    sigma: float,
    sigma_scale: Literal["unit", "train_std"],
    partition_name: str,
    split,
    shock: ShockRequest,
) -> dict[str, Any]:
    return {
        "source": source_key,
        "name": bank.name,
        "shock": shock,
        "sigma": float(sigma),
        "sigma_scale": sigma_scale,
        "noise_kind": bank.noise_kind,
        "partition": partition_name,
        "cell_tag": split.cell.tag if split.cell is not None else None,
        "dgp_bank": _dgp_bank_metadata(bank),
        "shape": {
            "full": list(split.values.shape),
            "train": list(split.train_values.shape),
            "val": list(split.val_values.shape),
            "test": list(split.test_values.shape),
        },
        "train_stats": {
            "mean_head": _array_summary(split.train_stats.mean),
            "std_head": _array_summary(split.train_stats.std),
        },
    }


def _generate_payload(
    source: str,
    *,
    shock: ShockRequest,
    out: Path | None,
    seed: int,
    n_trajectory_draws: int,
    n_noise_draws: int,
    noise_seed_base: int,
    sigma: float,
    sigma_scale: Literal["unit", "train_std"],
    partition_name: str,
    noise_kind: str,
    array_format: ArrayRequest,
    draw_design: DrawDesign | str,
    plot: PlotRequest,
    series_feature_idx: int,
    attractor_feature_idxs: str,
    max_points: int,
    image_format: PlotFormat,
    dpi: int,
) -> dict[str, Any]:
    if out is None and (array_format != "none" or plot != "none"):
        raise SystemExit("--out is required when writing arrays or plots")
    if sigma_scale != "unit" and (array_format != "none" or plot != "none"):
        raise SystemExit(
            "--sigma-scale train_std is supported for the JSON split summary. "
            "File exports and plots currently require --sigma-scale unit because "
            "they are written before partition-relative scaling can be represented."
        )

    source_key, source_spec = _resolve_generate_source(source, shock=shock)
    partition_name = _require_partition(partition_name)
    parsed_attractor_feature_idxs = _parse_attractor_feature_idxs(attractor_feature_idxs)
    cell_identity = _generate_cell_identity(
        source_key=source_key,
        source_spec=source_spec,
        shock=shock,
        sigma=sigma,
        sigma_scale=sigma_scale,
        partition_name=partition_name,
        seed=seed,
        n_trajectory_draws=n_trajectory_draws,
        n_noise_draws=n_noise_draws,
        noise_seed_base=noise_seed_base,
        noise_kind=noise_kind,
        array_format=array_format,
        draw_design=draw_design,
        plot=plot,
        series_feature_idx=series_feature_idx,
        attractor_feature_idxs=parsed_attractor_feature_idxs,
        max_points=max_points,
        image_format=image_format,
        dpi=dpi,
    )
    cell_hash = canonical_hash_16(cell_identity)
    cell_id = f"{_path_component(source_key)}_{cell_hash[:12]}"
    bank = build_dgp_bank(
        source_spec,
        seed=seed,
        n_trajectory_draws=n_trajectory_draws,
        n_noise_draws=n_noise_draws,
        noise_seed_base=noise_seed_base,
        noise_kind=noise_kind,
        verbose=False,
    )
    partition = _PARTITIONS[partition_name]()
    split = inject_noise(bank, sigma=sigma, sigma_scale=sigma_scale, partition=partition)
    payload = _split_summary_payload(
        source_key=source_key,
        bank=bank,
        sigma=sigma,
        sigma_scale=sigma_scale,
        partition_name=partition_name,
        split=split,
        shock=shock,
    )
    payload["cell_hash"] = cell_hash
    payload["cell_id"] = cell_id

    if out is None:
        payload["artifacts"] = None
        return payload

    artifact_dir = out / cell_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    identity_path = _write_generate_cell_identity(artifact_dir, cell_identity, cell_hash)
    artifacts: dict[str, Any] = {
        "requested_out_dir": str(out),
        "out_dir": str(artifact_dir),
        "cell_id": cell_id,
        "cell_hash": cell_hash,
        "cell_identity": str(identity_path),
    }
    if array_format != "none":
        manifest = export_dgp_bank_files(
            bank,
            artifact_dir,
            sigmas=(sigma,),
            array_format=array_format,
            draw_design=draw_design,
            overwrite=True,
        )
        manifest_files = cast(dict[str, object], manifest["files"])
        manifest_shape = cast(dict[str, object], manifest["shape"])
        manifest_noisy = cast(list[dict[str, object]], manifest["noisy"])
        artifacts["arrays"] = {
            "array_format": array_format,
            "draw_design": DrawDesign(draw_design).value,
            "files": manifest_files,
            "manifest": str(artifact_dir / "manifest.json"),
            "n_noisy": len(manifest_noisy),
            "shape": manifest_shape,
        }

    if plot != "none":
        from .plots import plot_dgp_bank_saves

        reference_source = _reference_source_for_plot(source_spec)
        reference_bank = (
            build_dgp_bank(
                reference_source,
                seed=seed,
                n_trajectory_draws=n_trajectory_draws,
                n_noise_draws=n_noise_draws,
                noise_seed_base=noise_seed_base,
                noise_kind=noise_kind,
                verbose=False,
            )
            if reference_source is not None
            else None
        )
        plot_files = plot_dgp_bank_saves(
            bank,
            out=artifact_dir,
            kind=plot,
            source=source_spec,
            reference_bank=reference_bank,
            sigma=sigma,
            feature_idx=series_feature_idx,
            features_idx=parsed_attractor_feature_idxs,
            max_points=max_points,
            plot_format=image_format,
            dpi=dpi,
            overwrite=True,
        )
        artifacts["plots"] = {name: str(path) for name, path in plot_files.items()}

    payload["artifacts"] = artifacts
    return payload


def _parse_attractor_feature_idxs(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 3:
        raise SystemExit(
            "--attractor-feature-idxs must contain exactly three comma-separated indices, e.g. 0,1,2"
        )
    return parts


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def _walkthrough_payload(
    *,
    out: Path,
    seed: int,
    sigma: float,
    array_format: ArrayFormat,
    kind: PlotKind,
    max_points: int,
    overwrite: bool,
) -> dict[str, Any]:
    from .catalog import source_preset
    from .plots import plot_dgp_bank_saves

    out.mkdir(parents=True, exist_ok=True)
    arrays_dir = out / "arrays"
    plots_dir = out / "plots"

    regular_source = source_preset("lorenz")()
    shifted_source = source_preset("lorenz_param_shock")()
    regular_bank = build_dgp_bank(regular_source, seed=seed, verbose=False)
    shifted_bank = build_dgp_bank(shifted_source, seed=seed, verbose=False)

    regular_manifest = export_dgp_bank_files(
        regular_bank,
        arrays_dir / "lorenz",
        sigmas=(sigma,),
        array_format=array_format,
        overwrite=overwrite,
    )
    shifted_manifest = export_dgp_bank_files(
        shifted_bank,
        arrays_dir / "lorenz_param_shock",
        sigmas=(sigma,),
        array_format=array_format,
        overwrite=overwrite,
    )

    cases = (
        ("regular", regular_bank, regular_source, None, 0.0),
        ("regular_noise", regular_bank, regular_source, None, sigma),
        ("param_shift", shifted_bank, shifted_source, regular_bank, 0.0),
        ("param_shift_noise", shifted_bank, shifted_source, regular_bank, sigma),
    )
    plot_files: dict[str, dict[str, str]] = {}
    for name, bank, source, reference_bank, case_sigma in cases:
        written = plot_dgp_bank_saves(
            bank,
            out=plots_dir / f"{name}.png",
            kind=kind,
            source=source,
            reference_bank=reference_bank,
            sigma=case_sigma,
            max_points=max_points,
            overwrite=overwrite,
        )
        plot_files[name] = {plot_kind: str(path) for plot_kind, path in written.items()}

    files = []
    for export_dir, manifest in (
        (arrays_dir / "lorenz", regular_manifest),
        (arrays_dir / "lorenz_param_shock", shifted_manifest),
    ):
        manifest_files = cast(dict[str, object], manifest["files"])
        manifest_noisy = cast(list[dict[str, object]], manifest["noisy"])
        for rel in manifest_files.values():
            files.append(export_dir / str(rel))
        files.append(export_dir / "manifest.json")
        for noisy in manifest_noisy:
            noisy_files = cast(dict[str, object], noisy.get("files", {}))
            for rel in noisy_files.values():
                files.append(export_dir / str(rel))
    for per_case in plot_files.values():
        files.extend(Path(path) for path in per_case.values())

    regular_files = cast(dict[str, object], regular_manifest["files"])
    regular_noisy = cast(list[dict[str, object]], regular_manifest["noisy"])
    shifted_files = cast(dict[str, object], shifted_manifest["files"])
    shifted_noisy = cast(list[dict[str, object]], shifted_manifest["noisy"])
    return {
        "out_dir": str(out),
        "seed": seed,
        "sigma": float(sigma),
        "array_format": array_format,
        "plot_kind": kind,
        "steps": [
            "Build a clean Lorenz DGP bank and a Lorenz parameter-shift DGP bank.",
            "Export clean/noise arrays plus optional noisy cells.",
            "Render four visual cases: regular, regular_noise, param_shift, param_shift_noise.",
            "Check that every declared file exists and report byte sizes.",
        ],
        "sources": {
            "regular": regular_bank.name,
            "param_shift": shifted_bank.name,
        },
        "exports": {
            "regular": {
                "dir": str(arrays_dir / "lorenz"),
                "schema": regular_manifest["schema"],
                "n_noisy": len(regular_noisy),
                "files": regular_files,
            },
            "param_shift": {
                "dir": str(arrays_dir / "lorenz_param_shock"),
                "schema": shifted_manifest["schema"],
                "n_noisy": len(shifted_noisy),
                "files": shifted_files,
            },
        },
        "plots": plot_files,
        "files": [_file_record(path) for path in files],
        "explanation": {
            "regular": "baseline Lorenz clean trajectory",
            "regular_noise": "same Lorenz trajectory with the frozen noise draw scaled by sigma",
            "param_shift": "Lorenz trajectory whose parameters change at the declared shock time",
            "param_shift_noise": "same parameter-shift trajectory with the same sigma-scale noise protocol",
        },
    }


def _emit(payload: Any, fmt: OutputFormat) -> None:
    if fmt == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    for item in payload:
        print(item)


def _emit_payload(build_payload, fmt: OutputFormat) -> None:
    if fmt == "json":
        with contextlib.redirect_stdout(io.StringIO()):
            payload = build_payload()
        _emit(payload, fmt)
        return
    _emit(build_payload(), fmt)


def _commands_payload() -> dict[str, Any]:
    return {
        "commands": list(_CLI_COMMANDS),
        "help": {
            "all_commands": "serie --help",
            "one_command": "serie <command> --help",
        },
    }


@cappa.command(name="list-cli-commands")
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class Commands(DataclassRecord):
    """List SERIE CLI commands with examples."""

    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "text"

    def __call__(self) -> None:
        payload = _commands_payload()
        if self.format == "json":
            _emit(payload, self.format)
            return

        current_group = None
        for item in payload["commands"]:
            group = item["group"]
            if group != current_group:
                if current_group is not None:
                    print()
                print(f"[{group}]")
                current_group = group
            print(f"  {item['command']}: {item['summary']}")
            print(f"    {item['example']}")
        print()
        print("Help:")
        print(f"  {payload['help']['all_commands']}")
        print(f"  {payload['help']['one_command']}")


@cappa.command(name="list-data-ids")
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class List(DataclassRecord):
    """List built-in synthetic and known source names."""

    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "text"

    def __call__(self) -> None:
        _emit(
            {
                "synthetic": list(list_sources()),
                "known": sorted(item.value for item in KnownDataID),
            },
            self.format,
        )


@cappa.command(name="show-data-dir")
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class DataDir(DataclassRecord):
    """Print the external dataset directory."""

    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "text"

    def __call__(self) -> None:
        root = known_data_root()
        _emit({"data_dir": str(root), "exists": root.exists()}, self.format)


@cappa.command
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class Generate(DataclassRecord):
    """Generate one SERIE dataset cell.

    This is the main SERIE data command. It builds one source, optionally applies
    a built-in intervention, injects observation noise, creates a train/val/test
    split, and optionally writes arrays or static plots.

    Examples: JSON summary only = serie generate lorenz; noisy split =
    serie generate lorenz --shock param --sigma 0.1; files =
    serie generate lorenz --out tmp/cells --array-format all --plot both.

    Defaults are chosen for a quick JSON-summary-only cell: clean baseline source,
    no shock, no file writes, p70_20_10 split, seed 42, and JSON output.
    """

    # Source and intervention axis. `--shock param` turns `lorenz` into the
    # built-in `lorenz_param_shock` preset; `--shock switch` maps to
    # `lorenz_switch` where available. Use `--shock none` for the baseline DGP.
    source: Annotated[
        str,
        cappa.Arg(
            help=(
                "Source data ID to generate, e.g. lorenz, ou, rossler, or a known-data ID. "
                "Run list-data-ids to see choices. Required; no default."
            ),
        ),
    ]
    shock: Annotated[
        ShockRequest,
        cappa.Arg(
            long=True,
            help=(
                "Intervention preset for SOURCE: none keeps the baseline; param "
                "uses a parameter-step preset; state applies a state perturbation "
                "preset; switch splices in a second trajectory. Unsupported pairs fail."
            ),
        ),
    ] = "none"

    # Observation-noise axis. `sigma` is the dose; `sigma_scale=train_std`
    # makes that dose relative to each channel's train-split standard deviation.
    sigma: Annotated[
        float,
        cappa.Arg(
            long=True,
            default=0.0,
            help=(
                "Observation-noise dose used for the JSON split summary and optional noisy "
                "exports. With --sigma-scale unit, this is the injected standard deviation."
            ),
        ),
    ] = 0.0
    sigma_scale: Annotated[
        Literal["unit", "train_std"],
        cappa.Arg(
            long=True,
            help=(
                "Noise scaling convention: unit uses frozen unit-noise arrays; "
                "train_std scales by each channel's train-split standard deviation."
            ),
        ),
    ] = "unit"

    # Dataset split. Train statistics are computed before train_std scaling, so
    # the split choice is part of the generated cell definition.
    partition: Annotated[
        str,
        cappa.Arg(
            long=True,
            help=(
                "Train/validation/test split preset. Examples: p70_20_10, "
                "p70_10_20, p70_gap_10_20. Also defines train_std noise scaling."
            ),
        ),
    ] = "p70_20_10"

    # Artifact controls. With no `--out`, generate prints the JSON summary only. With
    # `--out`, choose arrays (`npy`, `csv`, `ipc`, `parquet`, `table`, `all`)
    # and/or static plots.
    out: Annotated[
        Path | None,
        cappa.Arg(
            long=True,
            help=(
                "Root directory for generated arrays, manifest, or plots. Files are "
                "written under a stable source_hash child directory. "
                "Omit to print only the generated-cell summary. Default: no files are written."
            ),
        ),
    ] = None
    array_format: Annotated[
        ArrayRequest,
        cappa.Arg(
            long=True,
            help=(
                "File format for arrays written under --out: none, npy, csv, ipc, "
                "parquet, table, safetensors, or all. Stdout is always JSON."
            ),
        ),
    ] = "none"
    plot: Annotated[
        PlotRequest,
        cappa.Arg(
            long=True,
            help=(
                "Static figure export: none, timeseries, attractor, or both. "
                "Requires --out unless set to none. Uses --series-feature-idx "
                "and --attractor-feature-idxs."
            ),
        ),
    ] = "none"

    # Reproducibility controls. Root seeds are spawned into per-draw child seeds;
    # draw counts control the frozen bank used by exports and stochastic systems.
    seed: Annotated[
        int,
        cappa.Arg(long=True, help="Root seed for trajectory/process draws."),
    ] = 42
    n_trajectory_draws: Annotated[
        int,
        cappa.Arg(
            long=True,
            help="Number of trajectory/process draws in the generated bank.",
        ),
    ] = 1
    n_noise_draws: Annotated[
        int,
        cappa.Arg(
            long=True,
            help="Number of frozen observation-noise draws in the generated bank.",
        ),
    ] = 1
    noise_seed_base: Annotated[
        int,
        cappa.Arg(long=True, help="Root seed for observation-noise draws."),
    ] = 1000
    draw_design: Annotated[
        DrawDesign,
        cappa.Arg(
            long=True,
            help=(
                "How exported noisy array cells pair trajectory draws with noise draws: "
                "auto, noise_only, joint, trajectory_fixed, or cross_product."
            ),
        ),
    ] = DrawDesign.AUTO
    noise_kind: Annotated[
        str,
        cappa.Arg(
            long=True,
            help="Distribution for frozen observation-noise draws: gaussian or laplace.",
        ),
    ] = "gaussian"

    # Plot-only options. `series_feature_idx` is for time-series plots;
    # `attractor_feature_idxs` selects the three coordinates for attractor plots.
    series_feature_idx: Annotated[
        int,
        cappa.Arg(
            long="--series-feature-idx",
            default=0,
            help="Channel index for time-series plots; 0 means the first feature.",
        ),
    ] = 0
    attractor_feature_idxs: Annotated[
        str,
        cappa.Arg(
            long="--attractor-feature-idxs",
            help=(
                "Three comma-separated channel indices for attractor plots. "
                "0,1,2 means the first three features."
            ),
        ),
    ] = "0,1,2"
    max_points: Annotated[
        int,
        cappa.Arg(long=True, help="Maximum points rendered in static plots."),
    ] = 5000
    image_format: Annotated[
        PlotFormat,
        cappa.Arg(long=True, help="Static plot file format: png, pdf, or svg."),
    ] = "png"
    dpi: Annotated[
        int,
        cappa.Arg(long=True, help="DPI for static plot output."),
    ] = 300

    def __call__(self) -> None:
        _emit_payload(
            lambda: _generate_payload(
                self.source,
                shock=self.shock,
                out=self.out,
                seed=self.seed,
                n_trajectory_draws=self.n_trajectory_draws,
                n_noise_draws=self.n_noise_draws,
                noise_seed_base=self.noise_seed_base,
                sigma=self.sigma,
                sigma_scale=self.sigma_scale,
                partition_name=self.partition,
                noise_kind=self.noise_kind,
                array_format=self.array_format,
                draw_design=self.draw_design,
                plot=self.plot,
                series_feature_idx=self.series_feature_idx,
                attractor_feature_idxs=self.attractor_feature_idxs,
                max_points=self.max_points,
                image_format=self.image_format,
                dpi=self.dpi,
            ),
            "json",
        )


@cappa.command
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class Walkthrough(DataclassRecord):
    """Generate arrays and figures for the standard SERIE example."""

    out: Annotated[Path, cappa.Arg(long=True)] = Path("serie_walkthrough")
    seed: Annotated[int, cappa.Arg(long=True)] = 42
    sigma: Annotated[float, cappa.Arg(long=True)] = 0.1
    array_format: Annotated[ArrayFormat, cappa.Arg(long=True)] = "all"
    kind: Annotated[PlotKind, cappa.Arg(long=True)] = "both"
    max_points: Annotated[int, cappa.Arg(long=True)] = 2000
    overwrite: Annotated[bool, cappa.Arg(long=True)] = False
    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "json"

    def __call__(self) -> None:
        _emit_payload(
            lambda: _walkthrough_payload(
                out=self.out,
                seed=self.seed,
                sigma=self.sigma,
                array_format=self.array_format,
                kind=self.kind,
                max_points=self.max_points,
                overwrite=self.overwrite,
            ),
            self.format,
        )


def _demo_template_dir() -> Path:
    """Resolve the pre-built demo_dist/ shipped as package data."""
    return Path(__file__).resolve().parent / "demo_dist"


def _demo_source_spec(source: str) -> TrajectorySpec:
    """Return a spec for the browser demo.

    The browser viewer always receives a 3D projection. Native 1D/2D systems
    are padded or projected so the same React/Three template can render them.
    """
    source = _normalize_source_name(source)
    if source in _DEMO_SOURCES or source in _DEMO_CANDIDATE_SOURCES:
        return source_preset(source)()

    groups = tuple(_DEMO_SOURCE_GROUPS)
    valid = ", ".join(_DEMO_SOURCES + _DEMO_CANDIDATE_SOURCES + groups)
    raise SystemExit(f"unknown demo source {source!r}; valid demo sources: {valid}")


def _resolve_demo_sources(source: str, systems: str | None) -> tuple[str, ...]:
    """Resolve a single source, named source group, or comma-separated list."""
    raw = systems if systems is not None else source
    parts = tuple(_normalize_source_name(part.strip()) for part in raw.split(",") if part.strip())
    if not parts:
        raise SystemExit("demo source list is empty")

    resolved: list[str] = []
    for part in parts:
        members = _DEMO_SOURCE_GROUPS.get(part, (part,))
        for member in members:
            if member not in resolved:
                resolved.append(member)

    for member in resolved:
        _demo_source_spec(member)
    return tuple(resolved)


def _project_demo_arrays(
    clean: np.ndarray,
    draws: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Project any source dimension into the 3D browser viewer contract."""
    T, D = clean.shape
    if D >= 3:
        return (
            np.ascontiguousarray(clean[:, :3], dtype=np.float32),
            np.ascontiguousarray(draws[:, :, :3], dtype=np.float32),
            list(feature_names[:3]),
            "first_three_channels" if D > 3 else "native_3d",
        )
    if D == 2:
        zeros = np.zeros((T, 1), dtype=np.float32)
        draw_zeros = np.zeros((draws.shape[0], T, 1), dtype=np.float32)
        return (
            np.ascontiguousarray(np.concatenate([clean, zeros], axis=1), dtype=np.float32),
            np.ascontiguousarray(np.concatenate([draws, draw_zeros], axis=2), dtype=np.float32),
            [*feature_names, "zero"],
            "2d_padded_z0",
        )
    if D == 1:
        time_axis = np.linspace(-1.0, 1.0, T, dtype=np.float32)[:, None]
        zeros = np.zeros((T, 1), dtype=np.float32)
        draw_zeros = np.zeros((draws.shape[0], T, 1), dtype=np.float32)
        return (
            np.ascontiguousarray(
                np.concatenate([time_axis, clean, zeros], axis=1), dtype=np.float32
            ),
            np.ascontiguousarray(
                np.concatenate([draw_zeros, draws, draw_zeros], axis=2), dtype=np.float32
            ),
            ["time", feature_names[0], "zero"],
            "1d_time_value_z0",
        )
    raise SystemExit(f"demo source has invalid D={D}")


def _load_demo_values(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load a local array file for the demo exporter.

    This is intentionally CLI-side.  The browser viewer consumes one explicit
    JSON contract; CSV/NPY parsing stays in Python where Polars/NumPy own the
    format details.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.asarray(np.load(path), dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise SystemExit(f"{path} must contain a [T, D] array, got shape {values.shape}")
        return values, tuple(f"f{i}" for i in range(values.shape[1]))

    if suffix == ".csv":
        import polars as pl

        df = pl.read_csv(path)
        ignored = {"t", "time", "timestamp", "date", "datetime", "index"}
        numeric_columns: list[str] = []
        for column in df.columns:
            if column.lower() in ignored:
                continue
            try:
                df[column].cast(pl.Float32, strict=True)
            except (pl.exceptions.PolarsError, TypeError, ValueError):
                continue
            numeric_columns.append(column)
        if not numeric_columns:
            raise SystemExit(f"{path} has no numeric value columns for the demo viewer")
        values = np.asarray(df.select(numeric_columns).to_numpy(), dtype=np.float32)
        return values, tuple(numeric_columns)

    raise SystemExit(f"unsupported demo input {path}; expected .csv or .npy")


def _bake_demo_json(
    source: str,
    *,
    seed: int,
    n_points: int,
    n_draws: int,
) -> dict[str, Any]:
    """Bake one trajectory + frozen noise draws into the 3D viewer contract."""
    if n_points <= 1:
        raise SystemExit("--n-points must be greater than 1")
    if n_draws < 1:
        raise SystemExit("--n-draws must be at least 1")

    source = _normalize_source_name(source)
    spec = _demo_source_spec(source)
    bank = build_dgp_bank(spec, seed=seed, n_noise_draws=n_draws)

    clean = np.asarray(bank.clean[0], dtype=np.float32)
    draws = np.asarray(bank.noise_draws, dtype=np.float32)

    if clean.shape[0] > n_points:
        idx = np.linspace(0, clean.shape[0] - 1, n_points).astype(np.int64)
        clean = clean[idx]
        draws = draws[:, idx, :]

    clean, draws, feature_names, projection = _project_demo_arrays(clean, draws, bank.feature_names)

    return {
        "system": source,
        "seed": int(seed),
        "source_dim": int(bank.D),
        "projection": projection,
        "feature_names": feature_names,
        "dt": float(spec.dt),
        "n_points": int(clean.shape[0]),
        "shape": {
            "clean": [int(clean.shape[0]), int(clean.shape[1])],
            "draws": [int(draws.shape[0]), int(draws.shape[1]), int(draws.shape[2])],
        },
        "clean": clean.tolist(),
        "draws": draws.tolist(),
    }


def _bake_demo_bundle_json(
    sources: tuple[str, ...],
    *,
    seed: int,
    n_points: int,
    n_draws: int,
) -> dict[str, Any]:
    """Bake one or more demo systems into the browser viewer contract."""
    systems = [
        _bake_demo_json(source, seed=seed, n_points=n_points, n_draws=n_draws)
        for source in sources
    ]
    if len(systems) == 1:
        return systems[0]
    return {
        "kind": "serie_demo_bundle",
        "default_system": systems[0]["system"],
        "systems": systems,
    }


def _bake_demo_json_from_file(
    input_path: Path,
    *,
    name: str | None,
    seed: int,
    n_points: int,
    n_draws: int,
) -> dict[str, Any]:
    """Bake a local CSV/NPY array into the browser demo JSON contract."""
    if n_points <= 1:
        raise SystemExit("--n-points must be greater than 1")
    if n_draws < 1:
        raise SystemExit("--n-draws must be at least 1")
    if not input_path.exists():
        raise SystemExit(f"{input_path} does not exist")

    values, feature_names = _load_demo_values(input_path)
    bank = build_empirical_dgp_bank(
        values,
        name=name or input_path.stem,
        noise_seed_base=seed,
        n_noise_draws=n_draws,
        feature_names=feature_names,
    )
    clean = np.asarray(bank.clean[0], dtype=np.float32)
    draws = np.asarray(bank.noise_draws, dtype=np.float32)
    if clean.shape[0] > n_points:
        idx = np.linspace(0, clean.shape[0] - 1, n_points).astype(np.int64)
        clean = clean[idx]
        draws = draws[:, idx, :]
    clean, draws, projected_names, projection = _project_demo_arrays(
        clean, draws, bank.feature_names
    )
    return {
        "system": bank.name,
        "seed": int(seed),
        "source_dim": int(bank.D),
        "projection": projection,
        "feature_names": projected_names,
        "dt": 1.0,
        "n_points": int(clean.shape[0]),
        "shape": {
            "clean": [int(clean.shape[0]), int(clean.shape[1])],
            "draws": [int(draws.shape[0]), int(draws.shape[1]), int(draws.shape[2])],
        },
        "clean": clean.tolist(),
        "draws": draws.tolist(),
    }


def _demo_payload_systems(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "systems" in payload:
        return cast(list[dict[str, Any]], payload["systems"])
    return [payload]


def _demo_payload(
    source: str,
    *,
    out: Path,
    input_path: Path | None,
    input_name: str | None,
    systems: str | None,
    data_out: Path | None,
    seed: int,
    n_points: int,
    n_draws: int,
    overwrite: bool,
) -> dict[str, Any]:
    if input_path is not None and systems is not None:
        raise SystemExit("--systems is only supported for built-in demo sources, not --input")
    if out.exists() and not overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to replace it")
    if data_out is not None and data_out.exists() and not overwrite:
        raise SystemExit(f"{data_out} already exists; pass --overwrite to replace it")

    template_dir = _demo_template_dir()
    index_html = template_dir / "index.html"
    if not index_html.exists():
        raise SystemExit(
            f"Pre-built demo template not found at {index_html}. "
            f"If installing from source, run: cd src/serie/demo && bun run build && cp -R dist ../demo_dist"
        )

    resolved_sources = _resolve_demo_sources(source, systems)
    payload = (
        _bake_demo_json_from_file(
            input_path,
            name=input_name,
            seed=seed,
            n_points=n_points,
            n_draws=n_draws,
        )
        if input_path is not None
        else _bake_demo_bundle_json(resolved_sources, seed=seed, n_points=n_points, n_draws=n_draws)
    )
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    if data_out is not None:
        data_out.parent.mkdir(parents=True, exist_ok=True)
        data_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    html = index_html.read_text(encoding="utf-8")

    # Inline CSS and JS assets so the output is a single self-contained HTML file.
    assets_dir = template_dir / "assets"
    injected_scripts = 0
    for asset_file in sorted(assets_dir.iterdir()):
        name = asset_file.name
        content = asset_file.read_text(encoding="utf-8").replace("</", "<\\/")
        if name.endswith(".css"):
            html = html.replace(
                f'<link rel="stylesheet" crossorigin href="./assets/{name}">',
                f"<style>{content}</style>",
            )
        elif name.endswith(".js"):
            html = html.replace(
                f'<script type="module" crossorigin src="./assets/{name}"></script>',
                "",
            )
            inject = (
                f"<script>window.__SERIE_DEMO_DATA__={data_json};</script>\n"
                f'<script type="module">{content}</script>'
            )
            html = html.replace("</body>", f"{inject}\n</body>")
            injected_scripts += 1

    if injected_scripts == 0 or "window.__SERIE_DEMO_DATA__" not in html:
        raise SystemExit("failed to inline SERIE demo template; rebuild src/serie/demo_dist")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    payload_systems = _demo_payload_systems(payload)
    return {
        "out": str(out),
        "source": str(payload_systems[0]["system"]),
        "sources": [str(item["system"]) for item in payload_systems],
        "shapes": {str(item["system"]): item["shape"] for item in payload_systems},
        "input": None if input_path is None else str(input_path),
        "seed": seed,
        "n_points": payload_systems[0]["n_points"],
        "n_draws": n_draws,
        "data_out": None if data_out is None else str(data_out),
        "bytes": len(html),
    }


def _demo_pack_payload(
    source: str,
    *,
    out_dir: Path,
    input_path: Path | None,
    input_name: str | None,
    systems: str | None,
    data_out: Path | None,
    array_format: ArrayRequest,
    seed: int,
    n_points: int,
    n_draws: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Write an inspectable demo folder: HTML + arrays + static plots."""
    resolved_sources = _resolve_demo_sources(source, systems)
    if len(resolved_sources) > 1:
        raise SystemExit("--out-dir writes source-specific arrays/plots; use one demo source")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite to replace files")

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "index.html"
    payload = _demo_payload(
        source,
        out=html_path,
        input_path=input_path,
        input_name=input_name,
        systems=systems,
        data_out=data_out,
        seed=seed,
        n_points=n_points,
        n_draws=n_draws,
        overwrite=overwrite,
    )

    arrays: dict[str, Any] | None = None
    if array_format != "none":
        if input_path is None:
            spec = _demo_source_spec(source)
            bank = build_dgp_bank(spec, seed=seed, n_noise_draws=n_draws, verbose=False)
        else:
            values, feature_names = _load_demo_values(input_path)
            bank = build_empirical_dgp_bank(
                values,
                name=input_name or input_path.stem,
                noise_seed_base=seed,
                n_noise_draws=n_draws,
                feature_names=feature_names,
            )
        manifest = export_dgp_bank_files(
            bank,
            out_dir / "arrays",
            array_format=cast(ArrayFormat, array_format),
            overwrite=overwrite,
        )
        arrays = {
            "array_format": array_format,
            "dir": str(out_dir / "arrays"),
            "files": manifest["files"],
            "manifest": str(out_dir / "arrays" / "manifest.json"),
        }

    plots: dict[str, str] = {}
    if input_path is None:
        from .plots import plot_source_saves

        try:
            written = plot_source_saves(
                _normalize_source_name(source),
                out=out_dir / "static",
                kind="both",
                seed=seed,
                sigma=0.24,
                max_points=n_points,
                overwrite=overwrite,
            )
        except ValueError as exc:
            plots["error"] = str(exc)
        else:
            plots = {name: str(path) for name, path in written.items()}

    return {
        "out_dir": str(out_dir),
        "html": str(html_path),
        "data": None if data_out is None else str(data_out),
        "arrays": arrays,
        "plots": plots,
        **payload,
    }


@cappa.command
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class Demo(DataclassRecord):
    """Generate an interactive 3D trajectory HTML page with controllable noise."""

    source: Annotated[str, cappa.Arg(default="lorenz")]
    out_html: Annotated[
        Path,
        cappa.Arg(long=True, help="Output path for one self-contained viewer HTML file."),
    ] = Path("serie_demo.html")
    out_dir: Annotated[
        Path | None,
        cappa.Arg(
            long=True,
            help="Optional demo folder containing index.html, static plots, and array artifacts.",
        ),
    ] = None
    input: Annotated[
        Path | None,
        cappa.Arg(
            long=True,
            help="Optional local .csv or .npy [T, D] array to visualize instead of a built-in source.",
        ),
    ] = None
    input_name: Annotated[
        str | None,
        cappa.Arg(long=True, help="Display name for --input data. Defaults to the file stem."),
    ] = None
    data_out: Annotated[Path | None, cappa.Arg(long=True)] = None
    systems: Annotated[
        str | None,
        cappa.Arg(
            long=True,
            help=(
                "Comma-separated built-in demo sources, or a group such as 'shapes'. "
                "Use with --out-html to bake a dropdown viewer."
            ),
        ),
    ] = None
    array_format: Annotated[
        ArrayRequest,
        cappa.Arg(
            long=True,
            help="Array artifacts for --out-dir: none, npy, csv, ipc, parquet, table, safetensors, or all.",
        ),
    ] = "ipc"
    seed: Annotated[int, cappa.Arg(long=True)] = 42
    n_points: Annotated[int, cappa.Arg(long=True)] = 4000
    n_draws: Annotated[int, cappa.Arg(long=True)] = 5
    overwrite: Annotated[bool, cappa.Arg(long=True)] = False
    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "json"

    def __call__(self) -> None:
        if self.out_dir is not None:
            out_dir = self.out_dir
            if self.out_html != Path("serie_demo.html"):
                raise SystemExit(
                    "use either --out-html for one HTML file or --out-dir for a demo folder"
                )
            _emit_payload(
                lambda: _demo_pack_payload(
                    self.source,
                    out_dir=out_dir,
                    input_path=self.input,
                    input_name=self.input_name,
                    systems=self.systems,
                    data_out=self.data_out,
                    array_format=self.array_format,
                    seed=self.seed,
                    n_points=self.n_points,
                    n_draws=self.n_draws,
                    overwrite=self.overwrite,
                ),
                self.format,
            )
            return

        _emit_payload(
            lambda: _demo_payload(
                self.source,
                out=self.out_html,
                input_path=self.input,
                input_name=self.input_name,
                systems=self.systems,
                data_out=self.data_out,
                seed=self.seed,
                n_points=self.n_points,
                n_draws=self.n_draws,
                overwrite=self.overwrite,
            ),
            self.format,
        )


@cappa.command
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class Version(DataclassRecord):
    """Print package version."""

    def __call__(self) -> None:
        try:
            print(version("serie"))
        except PackageNotFoundError:
            print("0.1.1+local")


@cappa.command(name="freeze-controller-candidate-bank")
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class FreezeControllerCandidateBank(DataclassRecord):
    """Freeze controller gear branches and independent reference examples without training."""

    out: Annotated[Path, cappa.Arg(long=True)]
    factors: Annotated[str | None, cappa.Arg(long=True)] = None
    history: Annotated[int | None, cappa.Arg(long=True)] = None
    horizon: Annotated[int | None, cappa.Arg(long=True)] = None
    source_root: Annotated[Path | None, cappa.Arg(long=True)] = None
    recording: Annotated[Path | None, cappa.Arg(long=True)] = None
    reference_anchor: Annotated[int | None, cappa.Arg(long=True)] = None
    event_sequence: Annotated[Path | None, cappa.Arg(long=True)] = None

    def __call__(self) -> None:
        from .controller_fork_bank import (
            freeze_controller_candidate_origins, load_controller_candidate_bank, load_controller_recording_bank,
            ControllerTwoEventRequest, freeze_controller_two_event_origins,
        )

        # CHG: this arm is query truth only. Its complete typed request owns
        # origins/windows/events; single-event/reference flags cannot override it.
        if self.event_sequence is not None:
            if (self.recording is None or self.source_root is not None or self.reference_anchor is not None
                    or self.factors is not None or self.history is not None or self.horizon is not None):
                raise ValueError("--event-sequence requires --recording and excludes the one-event factors/history/horizon/reference flags")
            request = msgspec.json.decode(self.event_sequence.read_bytes(), type=ControllerTwoEventRequest)
            frozen = freeze_controller_two_event_origins(self.recording, request=request, output=self.out)
            _emit({"path": str(self.out.resolve()), "schema": frozen.metadata["schema"],
                   "episodes": len(request.episodes), "origins_per_episode": len(request.anchors),
                   "candidate_counts": [len(axis.factors) for axis in request.events],
                   "history": request.history, "horizon": request.horizon,
                   "shape": list(frozen.rollouts.shape), "meaning": "two-event query truth; no reference or prediction"}, "json")
            return
        if self.factors is None or self.history is None or self.horizon is None:
            raise ValueError("one-event freezing requires explicit --factors, --history and --horizon")
        tokens = tuple(token.strip() for token in self.factors.split(","))
        if len(tokens) < 2 or any(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token) is None for token in tokens):
            raise ValueError("--factors needs comma-separated positive decimal values")
        values = tuple(float(token) for token in tokens)
        if (self.source_root is None) == (self.recording is None):
            raise ValueError("choose exactly one --source-root or --recording")
        if self.recording is not None:
            if self.reference_anchor is None:
                raise ValueError("--recording requires an explicit --reference-anchor")
            bank = load_controller_recording_bank(self.recording, factors=values, reference_anchor=self.reference_anchor)
        else:
            if self.reference_anchor is not None:
                raise ValueError("matched source-root uses its recorded event anchor")
            paths = tuple(self.source_root / f"factor-{token}" / "manifest.json" for token in tokens)
            bank = load_controller_candidate_bank(paths, factors=values)
        frozen = freeze_controller_candidate_origins(
            bank, history=self.history, horizon=self.horizon, output=self.out,
        )
        _emit({
            "path": str(self.out.resolve()),
            "episodes": len(frozen.splits),
            "origins_per_episode": len(frozen.anchors),
            "candidates": len(frozen.factors),
            "history": frozen.history,
            "horizon": frozen.horizon,
            "shape": list(frozen.rollouts.shape),
        }, "json")


# CHG 2026-09-30: extracting existing foreign examples never calls a simulator.
@cappa.command(name="freeze-recorded-controller-references")
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class FreezeRecordedControllerReferences(DataclassRecord):
    """Freeze actual TRAIN before/after examples from declared gear recordings."""

    out: Annotated[Path, cappa.Arg(long=True)]
    # CHG: event-transition requests bind their own sources; absence distinguishes
    # this typed arm from the old command, whose matched-layout default is unchanged.
    recordings: Annotated[str | None, cappa.Arg(long=True)] = None
    # CHG: the old --factors operation stays matched; independent layout is explicit.
    factors: Annotated[str | None, cappa.Arg(long=True)] = None
    source_layout: Annotated[RecordedReferenceLayout | None, cappa.Arg(long=True)] = None
    event_transitions: Annotated[Path | None, cappa.Arg(long=True)] = None

    def __call__(self) -> None:
        from .controller_fork_bank import (
            IndependentControllerReferenceSource, MatchedControllerReferenceSource, freeze_recorded_controller_references,
            RecordedControllerEventRequest, freeze_recorded_controller_event_transitions,
        )

        if self.event_transitions is not None:
            if self.recordings is not None or self.factors is not None or self.source_layout is not None:
                raise ValueError("--event-transitions binds its own sources/events; legacy reference-layout flags cannot also select them")
            request = msgspec.json.decode(self.event_transitions.read_bytes(), type=RecordedControllerEventRequest)
            frozen = freeze_recorded_controller_event_transitions(request, output=self.out)
            _emit({"path": str(self.out.resolve()), "schema": frozen.metadata["schema"],
                "world": frozen.metadata["world"], "train_references": len(frozen.episodes),
                "interval_start_shape": list(frozen.interval_start.shape),
                "interval_end_shape": list(frozen.interval_end.shape), "action_shape": list(frozen.actions.shape),
                "meaning": frozen.metadata["state_meaning"],
                "events": [item["selected_event"]["event_id"] for item in frozen.metadata["events"]]}, "json")
            return
        if self.recordings is None:
            raise ValueError("legacy recorded references require --recordings; event-transition sources belong in --event-transitions")
        # The old CLI builder's sole matched default remains on its original arm.
        layout = RecordedReferenceLayout.MATCHED_FACTOR_RECORDINGS if self.source_layout is None else self.source_layout
        paths = tuple(Path(token.strip()) for token in self.recordings.split(",") if token.strip())
        if layout is RecordedReferenceLayout.INDEPENDENT_EPISODES:
            if len(paths) != 1 or self.factors is not None:
                raise ValueError("independent references require one mixed manifest and no per-manifest --factors")
            source = IndependentControllerReferenceSource(paths[0])
        else:
            if self.factors is None:
                raise ValueError("matched reference recordings require their explicit --factors")
            source = MatchedControllerReferenceSource(paths, tuple(float(token.strip()) for token in self.factors.split(",")))
        frozen = freeze_recorded_controller_references(source, output=self.out)
        _emit({"path": str(self.out.resolve()), "schema": frozen.metadata["schema"],
               "world": frozen.metadata["world"], "train_references": len(frozen.episodes),
               "pair_shape": list(frozen.pairs.shape), "action_shape": list(frozen.actions.shape),
               "references_per_actual_gear": {str(factor): int(np.count_nonzero(frozen.factors == factor))
                                              for factor in np.unique(frozen.factors)},
               "selection_rule": frozen.metadata["selection_rule"],
               "meaning": "actual recorded transitions; no simulation, replay or reference gear change"}, "json")


# CHG 2026-09-30: one typed source producer for the foreign derivative bank;
# all scientific numerical settings are explicit CLI arguments.
@cappa.command(name="freeze-controller-reference-derivatives")
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class FreezeControllerReferenceDerivatives(DataclassRecord):
    """Freeze one-record directional responses in physical tangent coordinates."""

    factors: Annotated[str, cappa.Arg(long=True)]
    epsilons: Annotated[str, cappa.Arg(long=True)]
    selected_epsilon: Annotated[float, cappa.Arg(long=True)]
    direction_seed: Annotated[int, cappa.Arg(long=True)]
    out: Annotated[Path, cappa.Arg(long=True)]
    source_root: Annotated[Path | None, cappa.Arg(long=True)] = None
    recording: Annotated[Path | None, cappa.Arg(long=True)] = None
    reference_anchor: Annotated[int | None, cappa.Arg(long=True)] = None

    def __call__(self) -> None:
        from .controller_fork_bank import (
            freeze_controller_reference_derivatives, load_controller_candidate_bank, load_controller_recording_bank,
        )

        tokens = tuple(token.strip() for token in self.factors.split(","))
        factors = tuple(float(token) for token in tokens)
        epsilons = tuple(float(token.strip()) for token in self.epsilons.split(","))
        if (self.source_root is None) == (self.recording is None):
            raise ValueError("choose exactly one --source-root or --recording")
        if self.recording is not None:
            if self.reference_anchor is None:
                raise ValueError("--recording requires an explicit --reference-anchor")
            bank = load_controller_recording_bank(self.recording, factors=factors, reference_anchor=self.reference_anchor)
        else:
            if self.reference_anchor is not None:
                raise ValueError("matched source-root uses its recorded event anchor")
            paths = tuple(self.source_root / f"factor-{token}" / "manifest.json" for token in tokens)
            bank = load_controller_candidate_bank(paths, factors=factors)
        frozen = freeze_controller_reference_derivatives(
            bank, epsilons=epsilons, selected_epsilon=self.selected_epsilon,
            direction_seed=self.direction_seed, output=self.out,
        )
        absolute = np.abs(frozen.responses[:, :, 1:] - frozen.responses[:, :, :-1])
        denominator = np.abs(frozen.responses[:, :, 1:]) + np.abs(frozen.responses[:, :, :-1])
        relative = np.divide(absolute, denominator, out=np.zeros_like(absolute), where=denominator != 0)
        if frozen.contact_trace is None:
            raise ValueError("new derivative producer must return v2 native contact diagnostics")
        _emit({"path": str(self.out.resolve()), "train_reference_episodes": len(frozen.episodes),
               "physical_tangent_size": frozen.tangent_size, "observed_channels": frozen.reference_states.shape[-1],
               "response_shape": list(frozen.responses.shape), "selected_epsilon": frozen.selected_epsilon,
               "epsilons": list(frozen.epsilons), "max_adjacent_step_absolute_difference": float(absolute.max()),
               "max_adjacent_step_relative_difference": float(relative.max()),
               "schema": frozen.metadata["schema"], "contact_trace_shape": list(frozen.contact_trace.shape),
               "contact_pair_observations": len(frozen.contact_trace.geometry_pairs),
               "contact_sampling": "native-command ends only; physics substeps not traced",
               "meaning": "finite central responses; step-size diagnostics are not proof of differentiability"}, "json")


@cappa.command(name="freeze-recorded-controller-reference-derivatives")
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class FreezeRecordedControllerReferenceDerivatives(DataclassRecord):
    """Freeze physical responses at actual foreign recorded intervals, without changing gear."""

    recordings: Annotated[str, cappa.Arg(long=True)]
    source_layout: Annotated[RecordedReferenceLayout, cappa.Arg(long=True)]
    reference_file: Annotated[Path, cappa.Arg(long=True)]
    epsilons: Annotated[str, cappa.Arg(long=True)]
    selected_epsilon: Annotated[float, cappa.Arg(long=True)]
    direction_seed: Annotated[int, cappa.Arg(long=True)]
    physical_precision: Annotated[str, cappa.Arg(long=True)]
    out: Annotated[Path, cappa.Arg(long=True)]
    factors: Annotated[str | None, cappa.Arg(long=True)] = None

    def __call__(self) -> None:
        from .controller_fork_bank import (IndependentControllerReferenceSource,
            MatchedControllerReferenceSource, RecordedDerivativePrecision, RecordedDerivativeSettings,
            freeze_recorded_controller_reference_derivatives)

        paths = tuple(Path(token.strip()) for token in self.recordings.split(",") if token.strip())
        if self.source_layout is RecordedReferenceLayout.INDEPENDENT_EPISODES:
            if len(paths) != 1 or self.factors is not None:
                raise ValueError("independent references require one actual source and no per-manifest factors")
            source = IndependentControllerReferenceSource(paths[0])
        else:
            if self.factors is None:
                raise ValueError("matched references require the recorded factors to identify their source files")
            source = MatchedControllerReferenceSource(paths, tuple(float(value) for value in self.factors.split(",")))
        settings = RecordedDerivativeSettings(tuple(float(value) for value in self.epsilons.split(",")),
            self.selected_epsilon, self.direction_seed, RecordedDerivativePrecision(self.physical_precision))
        result = freeze_recorded_controller_reference_derivatives(source, reference_file=self.reference_file,
            settings=settings, output=self.out)
        difference = np.abs(result.responses[:, 1:] - result.responses[:, :-1])
        _emit({"path": str(self.out.resolve()), "schema": result.metadata["schema"],
            "references": len(result.directions), "response_shape": list(result.responses.shape),
            "selected_epsilon": settings.selected_epsilon, "epsilons": list(settings.epsilons),
            "physical_precision": settings.physical_precision.value,
            "max_adjacent_step_absolute_difference": float(difference.max()),
            "reference_gear_regenerated": False, "query_inputs_used": False,
            "meaning": "actual R central responses; finite steps do not establish differentiability"}, "json")


@cappa.command(name="serie")
@dataclass(frozen=False, slots=True, kw_only=False, eq=False, weakref_slot=True)
class Serie(DataclassRecord):
    """SERIE: generate controlled time-series cells and demos."""

    cmd: cappa.Subcommands[
        Commands
        | List
        | DataDir
        | Generate
        | FreezeControllerCandidateBank
        | FreezeRecordedControllerReferences
        | FreezeControllerReferenceDerivatives
        | FreezeRecordedControllerReferenceDerivatives
        | Walkthrough
        | Demo
        | Version
    ]


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    cappa.invoke(Serie, argv=argv)


if __name__ == "__main__":
    main()
