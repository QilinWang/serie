"""SERIE command-line surface."""

from __future__ import annotations

import contextlib
import io
import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, Literal

import attrs
import cappa

from . import PartitionSpec, export_dgp_bank_files, build_dgp_bank, inject_noise
from .catalog import list_sources
from .pipeline import KnownDataID, DrawDesign, known_data_root


OutputFormat = Literal["text", "json"]
PlotKind = Literal["timeseries", "attractor", "both"]
PlotFormat = Literal["png", "pdf", "svg"]
ArrayFormat = Literal["npy", "csv", "both"]
_DEMO_SOURCES = ("lorenz", "rossler", "chua")


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
        "command": "commands",
        "summary": "List SERIE CLI commands with examples.",
        "example": "serie commands --format json",
    },
    {
        "group": "01 discovery",
        "command": "list",
        "summary": "List built-in synthetic sources and known external dataset names.",
        "example": "serie list --format json",
    },
    {
        "group": "01 discovery",
        "command": "data-dir",
        "summary": "Print the external dataset directory used for known datasets.",
        "example": "serie data-dir --format json",
    },
    {
        "group": "02 inspect",
        "command": "inspect",
        "summary": "Build one DGP bank and report shape, feature names, and seed provenance.",
        "example": "serie inspect lorenz --format json",
    },
    {
        "group": "03 generate",
        "command": "inject-noise",
        "summary": "Build one DGP bank, inject one sigma dose, and return train/val/test split metadata.",
        "example": "serie inject-noise ou --sigma 0.1 --partition p70_20_10 --format json",
    },
    {
        "group": "03 generate",
        "command": "export",
        "summary": "Write generated clean/noise arrays, optional noisy arrays, CSVs, and a manifest.",
        "example": "serie export lorenz --out tmp/lorenz_export --sigma 0.1 --array-format both --format json",
    },
    {
        "group": "04 visualize",
        "command": "plot",
        "summary": "Render generated source figures: timeseries, attractor, or both.",
        "example": "serie plot lorenz --out tmp/lorenz.png --kind both --sigma 0.1 --format json",
    },
    {
        "group": "05 walkthrough",
        "command": "walkthrough",
        "summary": "Generate the standard SERIE example: arrays, plots, and verification metadata.",
        "example": "serie walkthrough --out tmp/walkthrough --array-format both --overwrite --format json",
    },
    {
        "group": "05 walkthrough",
        "command": "demo",
        "summary": "Generate an interactive 3D attractor HTML page with controllable noise.",
        "example": "serie demo lorenz --out lorenz_demo.html",
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


def _require_source(source: str) -> str:
    valid_sources = list_sources()
    if source not in valid_sources:
        valid = ", ".join(valid_sources)
        raise SystemExit(f"unknown source {source!r}; valid synthetic sources: {valid}")
    return source


def _require_partition(name: str) -> str:
    if name not in _PARTITIONS:
        valid = ", ".join(sorted(_PARTITIONS))
        raise SystemExit(f"unknown partition {name!r}; valid partitions: {valid}")
    return name


def _bank_payload(
    source: str, *, seed: int, noise_kind: str, verbose: bool
) -> dict[str, Any]:
    source = _require_source(source)
    bank = build_dgp_bank(source, seed=seed, noise_kind=noise_kind, verbose=verbose)
    return {
        "source": source,
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


def _split_payload(
    source: str,
    *,
    seed: int,
    sigma: float,
    sigma_scale: Literal["unit", "train_std"],
    partition_name: str,
    noise_kind: str,
    verbose: bool,
) -> dict[str, Any]:
    source = _require_source(source)
    partition_name = _require_partition(partition_name)
    bank = build_dgp_bank(source, seed=seed, noise_kind=noise_kind, verbose=verbose)
    partition = _PARTITIONS[partition_name]()
    split = inject_noise(bank, sigma=sigma, sigma_scale=sigma_scale, partition=partition)
    return {
        "source": source,
        "name": bank.name,
        "sigma": float(sigma),
        "sigma_scale": sigma_scale,
        "noise_kind": bank.noise_kind,
        "partition": partition_name,
        "cell_tag": split.cell.tag if split.cell is not None else None,
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


def _export_payload(
    source: str,
    *,
    out: Path,
    seed: int,
    n_trajectory_draws: int,
    n_noise_draws: int,
    noise_seed_base: int,
    sigma: float | None,
    array_format: ArrayFormat,
    draw_design: DrawDesign | str,
    noise_kind: str,
    overwrite: bool,
    verbose: bool,
) -> dict[str, Any]:
    source = _require_source(source)
    bank = build_dgp_bank(
        source,
        seed=seed,
        n_trajectory_draws=n_trajectory_draws,
        n_noise_draws=n_noise_draws,
        noise_seed_base=noise_seed_base,
        noise_kind=noise_kind,
        verbose=verbose,
    )
    manifest = export_dgp_bank_files(
        bank,
        out,
        sigmas=() if sigma is None else (sigma,),
        array_format=array_format,
        draw_design=draw_design,
        overwrite=overwrite,
    )
    return {
        "out_dir": str(out),
        "source": source,
        "name": bank.name,
        "array_format": array_format,
        "draw_design": DrawDesign(draw_design).value,
        "files": manifest["files"],
        "manifest": str(out / "manifest.json"),
        "n_noisy": len(manifest["noisy"]),
        "shape": manifest["shape"],
    }


def _parse_features_idx(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 3:
        raise SystemExit("--features-idx must contain exactly three comma-separated indices, e.g. 0,1,2")
    return parts


def _plot_payload(
    source: str,
    *,
    out: Path,
    kind: PlotKind,
    seed: int,
    sigma: float,
    noise_kind: str,
    feature_idx: int,
    features_idx: str,
    max_points: int,
    image_format: PlotFormat,
    dpi: int,
    overwrite: bool,
) -> dict[str, Any]:
    source = _require_source(source)
    from .plots import plot_source_saves

    written = plot_source_saves(
        source,
        out=out,
        kind=kind,
        seed=seed,
        sigma=sigma,
        noise_kind=noise_kind,
        feature_idx=feature_idx,
        features_idx=_parse_features_idx(features_idx),
        max_points=max_points,
        plot_format=image_format,
        dpi=dpi,
        overwrite=overwrite,
    )
    return {
        "source": source,
        "kind": kind,
        "sigma": float(sigma),
        "files": {name: str(path) for name, path in written.items()},
    }


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

    files = [
        arrays_dir / "lorenz" / "manifest.json",
        arrays_dir / "lorenz_param_shock" / "manifest.json",
    ]
    for export_dir, manifest in (
        (arrays_dir / "lorenz", regular_manifest),
        (arrays_dir / "lorenz_param_shock", shifted_manifest),
    ):
        for rel in manifest["files"].values():
            files.append(export_dir / str(rel))
        for noisy in manifest["noisy"]:
            for rel in noisy["files"].values():
                files.append(export_dir / str(rel))
    for per_case in plot_files.values():
        files.extend(Path(path) for path in per_case.values())

    return {
        "out_dir": str(out),
        "seed": seed,
        "sigma": float(sigma),
        "array_format": array_format,
        "plot_kind": kind,
        "steps": [
            "Build a clean Lorenz DGP bank and a Lorenz parameter-shift DGP bank.",
            "Export clean/noise arrays, optional noisy cells, and manifest.json.",
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
                "n_noisy": len(regular_manifest["noisy"]),
                "files": regular_manifest["files"],
            },
            "param_shift": {
                "dir": str(arrays_dir / "lorenz_param_shock"),
                "schema": shifted_manifest["schema"],
                "n_noisy": len(shifted_manifest["noisy"]),
                "files": shifted_manifest["files"],
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


@cappa.command
@attrs.define
class Commands:
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


@cappa.command
@attrs.define
class List:
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


@cappa.command(name="data-dir")
@attrs.define
class DataDir:
    """Print the external dataset directory."""

    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "text"

    def __call__(self) -> None:
        root = known_data_root()
        _emit({"data_dir": str(root), "exists": root.exists()}, self.format)


@cappa.command
@attrs.define
class Inspect:
    """Build one source and print its shape."""

    source: str
    seed: Annotated[int, cappa.Arg(long=True)] = 42
    noise_kind: Annotated[str, cappa.Arg(long=True)] = "gaussian"
    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "text"
    verbose: Annotated[bool, cappa.Arg(long=True)] = False

    def __call__(self) -> None:
        _emit_payload(
            lambda: _bank_payload(
                self.source,
                seed=self.seed,
                noise_kind=self.noise_kind,
                verbose=self.verbose,
            ),
            self.format,
        )


@cappa.command
@attrs.define
class InjectNoise:
    """Build, perturb, and split one source."""

    source: str
    sigma: Annotated[float, cappa.Arg(long=True, default=0.0)] = 0.0
    sigma_scale: Annotated[Literal["unit", "train_std"], cappa.Arg(long=True)] = "unit"
    partition: Annotated[str, cappa.Arg(long=True)] = "p70_20_10"
    seed: Annotated[int, cappa.Arg(long=True)] = 42
    noise_kind: Annotated[str, cappa.Arg(long=True)] = "gaussian"
    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "json"
    verbose: Annotated[bool, cappa.Arg(long=True)] = False

    def __call__(self) -> None:
        _emit_payload(
            lambda: _split_payload(
                self.source,
                seed=self.seed,
                sigma=self.sigma,
                sigma_scale=self.sigma_scale,
                partition_name=self.partition,
                noise_kind=self.noise_kind,
                verbose=self.verbose,
            ),
            self.format,
        )


@cappa.command
@attrs.define
class Export:
    """Export generated clean/noise arrays to NPY/CSV files."""

    source: str
    out: Annotated[Path, cappa.Arg(long=True)]
    seed: Annotated[int, cappa.Arg(long=True)] = 42
    n_trajectory_draws: Annotated[int, cappa.Arg(long=True)] = 1
    n_noise_draws: Annotated[int, cappa.Arg(long=True)] = 1
    noise_seed_base: Annotated[int, cappa.Arg(long=True)] = 1000
    sigma: Annotated[float | None, cappa.Arg(long=True)] = None
    array_format: Annotated[ArrayFormat, cappa.Arg(long=True)] = "both"
    draw_design: Annotated[DrawDesign, cappa.Arg(long=True)] = DrawDesign.AUTO
    noise_kind: Annotated[str, cappa.Arg(long=True)] = "gaussian"
    overwrite: Annotated[bool, cappa.Arg(long=True)] = False
    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "json"
    verbose: Annotated[bool, cappa.Arg(long=True)] = False

    def __call__(self) -> None:
        _emit_payload(
            lambda: _export_payload(
                self.source,
                out=self.out,
                seed=self.seed,
                n_trajectory_draws=self.n_trajectory_draws,
                n_noise_draws=self.n_noise_draws,
                noise_seed_base=self.noise_seed_base,
                sigma=self.sigma,
                array_format=self.array_format,
                draw_design=self.draw_design,
                noise_kind=self.noise_kind,
                overwrite=self.overwrite,
                verbose=self.verbose,
            ),
            self.format,
        )


@cappa.command
@attrs.define
class Plot:
    """Render a generated source plot."""

    source: str
    out: Annotated[Path, cappa.Arg(long=True)]
    kind: Annotated[PlotKind, cappa.Arg(long=True)] = "timeseries"
    seed: Annotated[int, cappa.Arg(long=True)] = 42
    sigma: Annotated[float, cappa.Arg(long=True, default=0.0)] = 0.0
    noise_kind: Annotated[str, cappa.Arg(long=True)] = "gaussian"
    feature_idx: Annotated[int, cappa.Arg(long=True, default=0)] = 0
    features_idx: Annotated[str, cappa.Arg(long=True)] = "0,1,2"
    max_points: Annotated[int, cappa.Arg(long=True)] = 5000
    image_format: Annotated[PlotFormat, cappa.Arg(long=True)] = "png"
    dpi: Annotated[int, cappa.Arg(long=True)] = 300
    overwrite: Annotated[bool, cappa.Arg(long=True)] = False
    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "json"

    def __call__(self) -> None:
        _emit_payload(
            lambda: _plot_payload(
                self.source,
                out=self.out,
                kind=self.kind,
                seed=self.seed,
                sigma=self.sigma,
                noise_kind=self.noise_kind,
                feature_idx=self.feature_idx,
                features_idx=self.features_idx,
                max_points=self.max_points,
                image_format=self.image_format,
                dpi=self.dpi,
                overwrite=self.overwrite,
            ),
            self.format,
        )


@cappa.command
@attrs.define
class Walkthrough:
    """Generate arrays and figures for the standard SERIE example."""

    out: Annotated[Path, cappa.Arg(long=True)] = Path("serie_walkthrough")
    seed: Annotated[int, cappa.Arg(long=True)] = 42
    sigma: Annotated[float, cappa.Arg(long=True)] = 0.1
    array_format: Annotated[ArrayFormat, cappa.Arg(long=True)] = "both"
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


def _bake_demo_json(
    source: str, *, seed: int, n_points: int, n_draws: int,
) -> dict[str, Any]:
    """Bake one 3D attractor + frozen noise draws into a JSON-serializable dict."""
    import numpy as np
    from .catalog import source_preset

    if source not in _DEMO_SOURCES:
        valid = ", ".join(_DEMO_SOURCES)
        raise SystemExit(f"unknown demo source {source!r}; valid demo sources: {valid}")
    if n_points <= 1:
        raise SystemExit("--n-points must be greater than 1")
    if n_draws < 1:
        raise SystemExit("--n-draws must be at least 1")

    spec = source_preset(source)()
    bank = build_dgp_bank(spec, seed=seed, n_noise_draws=n_draws)

    clean = np.asarray(bank.clean[0], dtype=np.float32)
    draws = np.asarray(bank.noise_draws, dtype=np.float32)

    if clean.shape[1] < 3:
        raise SystemExit(f"{source!r} has D={clean.shape[1]}; the demo requires a 3D source")

    if clean.shape[0] > n_points:
        idx = np.linspace(0, clean.shape[0] - 1, n_points).astype(np.int64)
        clean = clean[idx]
        draws = draws[:, idx, :]

    clean = np.ascontiguousarray(clean[:, :3], dtype=np.float32)
    draws = np.ascontiguousarray(draws[:, :, :3], dtype=np.float32)

    return {
        "system": source,
        "seed": int(seed),
        "feature_names": list(bank.feature_names[:3]),
        "dt": float(spec.dt),
        "n_points": int(clean.shape[0]),
        "clean": clean.tolist(),
        "draws": draws.tolist(),
    }


def _demo_payload(
    source: str,
    *,
    out: Path,
    seed: int,
    n_points: int,
    n_draws: int,
    overwrite: bool,
) -> dict[str, Any]:
    if out.exists() and not overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to replace it")

    template_dir = _demo_template_dir()
    index_html = template_dir / "index.html"
    if not index_html.exists():
        raise SystemExit(
            f"Pre-built demo template not found at {index_html}. "
            f"If installing from source, run: cd serie/demo && bun run build && cp -R dist ../demo_dist"
        )

    payload = _bake_demo_json(source, seed=seed, n_points=n_points, n_draws=n_draws)
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

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
                f"<script type=\"module\">{content}</script>"
            )
            html = html.replace("</body>", f"{inject}\n</body>")
            injected_scripts += 1

    if injected_scripts == 0 or "window.__SERIE_DEMO_DATA__" not in html:
        raise SystemExit("failed to inline SERIE demo template; rebuild serie/demo_dist")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return {
        "out": str(out),
        "source": source,
        "seed": seed,
        "n_points": payload["n_points"],
        "n_draws": n_draws,
        "bytes": len(html),
    }


@cappa.command
@attrs.define
class Demo:
    """Generate an interactive 3D attractor HTML page with controllable noise."""

    source: Annotated[str, cappa.Arg(default="lorenz")]
    out: Annotated[Path, cappa.Arg(long=True)] = Path("serie_demo.html")
    seed: Annotated[int, cappa.Arg(long=True)] = 42
    n_points: Annotated[int, cappa.Arg(long=True)] = 4000
    n_draws: Annotated[int, cappa.Arg(long=True)] = 5
    overwrite: Annotated[bool, cappa.Arg(long=True)] = False
    format: Annotated[OutputFormat, cappa.Arg(long=True)] = "json"

    def __call__(self) -> None:
        _emit_payload(
            lambda: _demo_payload(
                self.source,
                out=self.out,
                seed=self.seed,
                n_points=self.n_points,
                n_draws=self.n_draws,
                overwrite=self.overwrite,
            ),
            self.format,
        )


@cappa.command
@attrs.define
class Version:
    """Print package version."""

    def __call__(self) -> None:
        try:
            print(version("serie"))
        except PackageNotFoundError:
            print("0.1.1+local")


@cappa.command(name="serie")
@attrs.define
class Serie:
    """SERIE: time-series source inspection and one-cell noise injection."""

    cmd: cappa.Subcommands[Commands | List | DataDir | Inspect | InjectNoise | Export | Plot | Walkthrough | Demo | Version]


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    cappa.invoke(Serie, argv=argv)


if __name__ == "__main__":
    main()
