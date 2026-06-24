"""Source-level visualization for SERIE DGP banks.

This module contains the generic drawing choices for source inspection: paper
rcParams, clean/noisy colors, and attractor colormaps. It does not depend on
training results, prediction windows, or model internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from attrs import evolve

from .api import build_dgp_bank
from .specs import ShockEvent, ShockSchedule, TrajectorySpec, DGPBank

PlotKind = Literal["timeseries", "attractor", "both"]
PlotFormat = Literal["png", "pdf", "svg"]

CLEAN_COLOR = "#56B4E9"
NOISY_COLOR = "#E69F00"
REFERENCE_COLOR = "#D55E00"
CLEAN_CMAP = "Blues"
NOISY_CMAP = "Oranges"


def _import_matplotlib() -> tuple[Any, Any, Any, Any]:
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for SERIE plotting. It is a SERIE runtime dependency; "
            "reinstall the package if this import is missing."
        ) from exc
    return mpl, mcolors, plt, Line3DCollection


def _apply_paper_theme(*, dpi: int) -> None:
    mpl, _, plt, _ = _import_matplotlib()
    plt.style.use("default")
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "lines.linewidth": 1.8,
            "axes.linewidth": 1.5,
            "figure.dpi": dpi,
            "figure.facecolor": "white",
            "savefig.dpi": dpi,
        }
    )


def _event_label(event: ShockEvent) -> str:
    if event.t_start == event.t_end:
        timing = f"t={event.t_start:g}"
    else:
        timing = f"t={event.t_start:g}-{event.t_end:g}"
    return f"{event.shocktype.value}@{timing}"


def _shock_events(source: TrajectorySpec | None) -> tuple[ShockEvent, ...]:
    if source is None or source.shocks.is_stationary:
        return ()
    return source.shocks.events


def _unshocked_reference_source(source: TrajectorySpec | None) -> TrajectorySpec | None:
    if source is None or source.shocks.is_stationary:
        return None
    return evolve(source, shocks=ShockSchedule.none())


def _title(*, bank: DGPBank, sigma: float, source: TrajectorySpec | None) -> str:
    return " | ".join([bank.name, f"sigma={sigma:g}"])


def _shock_label_y(*, time: np.ndarray, values: np.ndarray, x: float) -> float:
    if values.size == 0:
        return 0.95
    radius = max(3, int(time.shape[0] * 0.015))
    idx = int(np.argmin(np.abs(time - x)))
    window = values[max(0, idx - radius) : min(values.shape[0], idx + radius + 1)]
    if window.size == 0:
        return 0.95
    global_mid = float(np.nanmedian(values))
    local_mid = float(np.nanmedian(window))
    return 0.06 if local_mid >= global_mid else 0.94


def _draw_shock_markers(ax, *, source: TrajectorySpec | None, time: np.ndarray, values: np.ndarray) -> None:
    events = _shock_events(source)
    if not events or time.size == 0:
        return

    transform = ax.get_xaxis_transform()
    for event in events:
        x_start = float(event.t_start * max(0, time.shape[0] - 1))
        x_end = float(event.t_end * max(0, time.shape[0] - 1))
        if event.t_end > event.t_start:
            ax.axvspan(x_start, x_end, color="#CC79A7", alpha=0.12, zorder=0)
            x_label = (x_start + x_end) / 2.0
        else:
            ax.axvline(x_start, color="#CC79A7", linestyle="--", linewidth=1.4, alpha=0.9, zorder=1)
            x_label = x_start
        y_label = _shock_label_y(time=time, values=values, x=x_label)
        va = "bottom" if y_label > 0.5 else "top"
        ax.text(
            x_label,
            y_label,
            _event_label(event),
            transform=transform,
            ha="center",
            va=va,
            fontsize=11,
            color="#8F2D64",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#CC79A7", "alpha": 0.9},
            zorder=5,
        )


def _post_shock_mask(*, source: TrajectorySpec | None, time: np.ndarray) -> np.ndarray | None:
    events = _shock_events(source)
    if not events or time.size == 0:
        return None
    first = min(event.t_start for event in events)
    x_start = float(first * max(0, time.shape[0] - 1))
    return time >= x_start


def _draw_reference_after_shock(
    ax,
    *,
    source: TrajectorySpec | None,
    time: np.ndarray,
    current: np.ndarray,
    reference: np.ndarray | None,
    current_label: str,
    reference_label: str,
) -> None:
    if reference is None:
        return
    mask = _post_shock_mask(source=source, time=time)
    if mask is None or not np.any(mask):
        return
    ax.plot(
        time[mask],
        current[mask],
        label=current_label,
        color=REFERENCE_COLOR,
        linewidth=2.0,
        alpha=0.70,
        zorder=4,
    )
    ax.plot(
        time[mask],
        reference[mask],
        label=reference_label,
        color=REFERENCE_COLOR,
        linewidth=2.0,
        alpha=0.55,
        linestyle=(0, (5, 3)),
        zorder=4,
    )


def _check_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)


def _downsample(values: np.ndarray, *, max_points: int) -> np.ndarray:
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if values.shape[0] <= max_points:
        return values
    idx = np.linspace(0, values.shape[0] - 1, max_points).astype(np.int64)
    return values[idx]


def _plot_timeseries(
    *,
    bank: DGPBank,
    source: TrajectorySpec | None,
    reference_bank: DGPBank | None,
    save_to: Path,
    sigma: float,
    feature_idx: int,
    trajectory_draw_ordinal: int,
    noise_draw_ordinal: int,
    max_points: int,
    dpi: int,
    overwrite: bool,
) -> Path:
    _, _, plt, _ = _import_matplotlib()
    _apply_paper_theme(dpi=dpi)
    _check_write(save_to, overwrite=overwrite)

    if not (0 <= feature_idx < bank.D):
        raise ValueError(f"feature_idx={feature_idx} out of range for D={bank.D}")

    clean = bank.clean[trajectory_draw_ordinal, :, feature_idx]
    noisy = bank.build_observed_series(
        noise_level=sigma,
        trajectory_draw_ordinal=trajectory_draw_ordinal,
        noise_draw_ordinal=noise_draw_ordinal,
    )[:, feature_idx]
    clean_reference = None
    noisy_reference = None
    if reference_bank is not None:
        if reference_bank.T != bank.T or reference_bank.D != bank.D:
            raise ValueError(
                "reference_bank must match bank shape: "
                f"got T={reference_bank.T}, D={reference_bank.D}; "
                f"expected T={bank.T}, D={bank.D}"
            )
        clean_reference = reference_bank.clean[trajectory_draw_ordinal, :, feature_idx]
        if sigma != 0.0:
            noisy_reference = reference_bank.build_observed_series(
                noise_level=sigma,
                trajectory_draw_ordinal=trajectory_draw_ordinal,
                noise_draw_ordinal=noise_draw_ordinal,
            )[:, feature_idx]
    clean = _downsample(clean[:, None], max_points=max_points)[:, 0]
    noisy = _downsample(noisy[:, None], max_points=max_points)[:, 0]
    if clean_reference is not None:
        clean_reference = _downsample(clean_reference[:, None], max_points=max_points)[:, 0]
    if noisy_reference is not None:
        noisy_reference = _downsample(noisy_reference[:, None], max_points=max_points)[:, 0]
    time = np.arange(clean.shape[0])
    marker_values = noisy if sigma != 0.0 else clean
    comparison_current = noisy if sigma != 0.0 else clean
    comparison_reference = noisy_reference if sigma != 0.0 else clean_reference
    current_label = "Intervened Noisy Source" if sigma != 0.0 else "Intervened Source"
    reference_label = (
        "No-intervention Noisy Reference" if sigma != 0.0 else "No-intervention Reference"
    )

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(time, clean, label="Clean Source", color=CLEAN_COLOR, linewidth=1.9, alpha=0.92, zorder=3)
    if sigma != 0.0:
        ax.plot(time, noisy, label="Noisy Source", color=NOISY_COLOR, linewidth=1.15, alpha=0.42, zorder=2)
    _draw_reference_after_shock(
        ax,
        source=source,
        time=time,
        current=comparison_current,
        reference=comparison_reference,
        current_label=current_label,
        reference_label=reference_label,
    )
    _draw_shock_markers(ax, source=source, time=time, values=marker_values)
    ax.set_title(_title(bank=bank, sigma=sigma, source=source), weight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel(bank.feature_names[feature_idx] if feature_idx < len(bank.feature_names) else f"Dim {feature_idx}")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.margins(x=0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, max(0, clean.shape[0] - 1))
    fig.savefig(save_to, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to


def _create_colored_line_segments(
    xyz_data: np.ndarray,
    color_data: np.ndarray,
    cmap_name: str,
    *,
    lw: float = 1.0,
    alpha: float = 0.9,
):
    _, mcolors, plt, Line3DCollection = _import_matplotlib()
    if xyz_data.ndim != 2 or xyz_data.shape[0] != 3:
        raise ValueError(
            f"_create_colored_line_segments expects xyz_data shape [3, T], got {tuple(xyz_data.shape)}"
        )
    T = xyz_data.shape[1]
    if T < 2:
        raise ValueError("_create_colored_line_segments requires at least 2 points")
    if color_data.ndim != 1 or color_data.shape[0] != T:
        raise ValueError(f"color_data must have shape [T={T}], got {tuple(color_data.shape)}")

    points = xyz_data.T
    segments = np.stack([points[:-1], points[1:]], axis=1)
    valid_c = color_data[np.isfinite(color_data)]
    if len(valid_c) > 0:
        vmin, vmax = np.percentile(valid_c, 2), np.percentile(valid_c, 98)
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = mcolors.Normalize(vmin=cast(float, vmin), vmax=cast(float, vmax))
    cmap = plt.get_cmap(cmap_name)
    colors_rgba = cmap(norm(color_data))
    colors_rgba[np.isnan(color_data)] = [0, 0, 0, 0]
    seg_colors = (colors_rgba[:-1] + colors_rgba[1:]) / 2.0
    return Line3DCollection(segments, colors=seg_colors, linewidths=lw, alpha=alpha), norm


def _expand_degenerate_limit(limit: tuple[float, float]) -> tuple[float, float]:
    low, high = limit
    if high > low:
        return limit
    pad = max(abs(low) * 0.05, 1e-3)
    return low - pad, high + pad


def _attractor_xyz_and_labels(
    *,
    values: np.ndarray,
    feature_names: tuple[str, ...],
    features_idx: tuple[int, int, int],
    allow_default_padding: bool,
) -> tuple[np.ndarray, tuple[str, str, str]]:
    if values.ndim != 2:
        raise ValueError(f"attractor values must have shape [T, D], got {tuple(values.shape)}")
    if values.shape[1] < 2:
        raise ValueError("attractor plot requires at least two channels")

    coords: list[np.ndarray] = []
    labels: list[str] = []
    for idx in features_idx:
        if 0 <= idx < values.shape[1]:
            coords.append(values[:, idx])
            labels.append(feature_names[idx] if idx < len(feature_names) else f"Dim {idx}")
        elif idx >= values.shape[1] and allow_default_padding:
            coords.append(np.zeros(values.shape[0], dtype=values.dtype))
            labels.append("0")
        else:
            raise ValueError(f"features_idx={features_idx} out of range for D={values.shape[1]}")
    return np.stack(coords, axis=0), cast(tuple[str, str, str], tuple(labels))


def _format_attractor_axis(ax, *, labels: tuple[str, str, str], limits: tuple[tuple[float, float], ...]) -> None:
    ax.set_xlabel(labels[0], labelpad=8)
    ax.set_ylabel(labels[1], labelpad=8)
    ax.set_zlabel(labels[2], labelpad=8)
    ax.set_xlim(_expand_degenerate_limit(limits[0]))
    ax.set_ylim(_expand_degenerate_limit(limits[1]))
    ax.set_zlim(_expand_degenerate_limit(limits[2]))
    if labels[0] == "0":
        ax.set_xticks([0.0])
    if labels[1] == "0":
        ax.set_yticks([0.0])
    if labels[2] == "0":
        ax.set_zticks([0.0])
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.view_init(elev=25, azim=-60)


def _plot_attractor(
    *,
    bank: DGPBank,
    source: TrajectorySpec | None,
    save_to: Path,
    sigma: float,
    features_idx: tuple[int, int, int],
    trajectory_draw_ordinal: int,
    noise_draw_ordinal: int,
    max_points: int,
    dpi: int,
    overwrite: bool,
) -> Path:
    _, _, plt, _ = _import_matplotlib()
    _apply_paper_theme(dpi=dpi)
    _check_write(save_to, overwrite=overwrite)

    if bank.D < 2:
        raise ValueError(f"attractor plot requires at least two channels, got D={bank.D}")
    allow_default_padding = features_idx == (0, 1, 2)
    if any(i < 0 or (i >= bank.D and not allow_default_padding) for i in features_idx):
        raise ValueError(f"features_idx={features_idx} out of range for D={bank.D}")

    clean = bank.clean[trajectory_draw_ordinal]
    noisy = bank.build_observed_series(noise_level=sigma, trajectory_draw_ordinal=trajectory_draw_ordinal, noise_draw_ordinal=noise_draw_ordinal)
    clean = _downsample(clean, max_points=max_points)
    noisy = _downsample(noisy, max_points=max_points)
    clean_xyz, labels = _attractor_xyz_and_labels(
        values=clean,
        feature_names=bank.feature_names,
        features_idx=features_idx,
        allow_default_padding=allow_default_padding,
    )
    noisy_xyz, _ = _attractor_xyz_and_labels(
        values=noisy,
        feature_names=bank.feature_names,
        features_idx=features_idx,
        allow_default_padding=allow_default_padding,
    )
    clean_speed = np.linalg.norm(np.diff(clean_xyz, axis=1), axis=0)
    clean_speed = np.concatenate(([clean_speed[0]], clean_speed))

    if sigma == 0.0:
        fig = plt.figure(figsize=(14, 10), layout="tight")
        ax_clean = fig.add_subplot(1, 1, 1, projection="3d")
        ax_noisy = None
    else:
        fig = plt.figure(figsize=(18, 8), layout="constrained")
        ax_clean = fig.add_subplot(1, 2, 1, projection="3d")
        ax_noisy = fig.add_subplot(1, 2, 2, projection="3d")

    fig.suptitle(_title(bank=bank, sigma=sigma, source=source), y=0.99, weight="bold", fontsize=16)

    lc_clean, norm_clean = _create_colored_line_segments(clean_xyz, clean_speed, CLEAN_CMAP, lw=1.2)
    ax_clean.add_collection(lc_clean)
    sm_clean = plt.cm.ScalarMappable(cmap=plt.get_cmap(CLEAN_CMAP), norm=norm_clean)
    cbar_clean = fig.colorbar(sm_clean, ax=ax_clean, shrink=0.6, pad=0.02, aspect=20)
    cbar_clean.set_label("Speed")
    ax_clean.set_title("Clean Source (Color: Speed)", weight="bold")

    if ax_noisy is not None:
        # Noise is not system flow. Plot it as samples rather than
        # connecting adjacent noisy observations into artificial trajectories.
        ax_noisy.scatter(*noisy_xyz, s=1.2, color=NOISY_COLOR, alpha=0.45, depthshade=False)
        ax_noisy.set_title("Noisy Source", weight="bold")

    all_xyz = clean_xyz if ax_noisy is None else np.hstack((clean_xyz, noisy_xyz))
    limits = (
        (float(all_xyz[0].min()), float(all_xyz[0].max())),
        (float(all_xyz[1].min()), float(all_xyz[1].max())),
        (float(all_xyz[2].min()), float(all_xyz[2].max())),
    )
    for ax in (ax_clean,) if ax_noisy is None else (ax_clean, ax_noisy):
        _format_attractor_axis(ax, labels=labels, limits=limits)

    fig.savefig(save_to, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to


def _resolve_plot_paths(out: str | Path, *, source_name: str, kind: PlotKind, plot_format: PlotFormat) -> dict[str, Path]:
    path = Path(out)
    suffix = path.suffix or f".{plot_format}"
    if kind != "both":
        if path.suffix:
            return {kind: path}
        return {kind: path / f"{source_name}_{kind}{suffix}"}
    if path.suffix:
        return {
            "timeseries": path.with_name(f"{path.stem}_timeseries{path.suffix}"),
            "attractor": path.with_name(f"{path.stem}_attractor{path.suffix}"),
        }
    return {
        "timeseries": path / f"{source_name}_timeseries{suffix}",
        "attractor": path / f"{source_name}_attractor{suffix}",
    }


def plot_dgp_bank_saves(
    bank: DGPBank,
    *,
    out: str | Path,
    kind: PlotKind = "timeseries",
    source: TrajectorySpec | None = None,
    reference_bank: DGPBank | None = None,
    sigma: float = 0.0,
    feature_idx: int = 0,
    features_idx: tuple[int, int, int] = (0, 1, 2),
    trajectory_draw_ordinal: int = 0,
    noise_draw_ordinal: int = 0,
    max_points: int = 5000,
    plot_format: PlotFormat = "png",
    dpi: int = 300,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save one or more source-level SERIE plots for a DGP bank."""
    paths = _resolve_plot_paths(out, source_name=bank.name, kind=kind, plot_format=plot_format)
    written: dict[str, Path] = {}
    if "timeseries" in paths:
        written["timeseries"] = _plot_timeseries(
            bank=bank,
            source=source,
            reference_bank=reference_bank,
            save_to=paths["timeseries"],
            sigma=sigma,
            feature_idx=feature_idx,
            trajectory_draw_ordinal=trajectory_draw_ordinal,
            noise_draw_ordinal=noise_draw_ordinal,
            max_points=max_points,
            dpi=dpi,
            overwrite=overwrite,
        )
    if "attractor" in paths and (bank.D >= 2 or kind == "attractor"):
        written["attractor"] = _plot_attractor(
            bank=bank,
            source=source,
            save_to=paths["attractor"],
            sigma=sigma,
            features_idx=features_idx,
            trajectory_draw_ordinal=trajectory_draw_ordinal,
            noise_draw_ordinal=noise_draw_ordinal,
            max_points=max_points,
            dpi=dpi,
            overwrite=overwrite,
        )
    return written


def plot_source_saves(
    source_name: str,
    *,
    out: str | Path,
    kind: PlotKind = "timeseries",
    seed: int = 42,
    sigma: float = 0.0,
    noise_kind: str = "gaussian",
    feature_idx: int = 0,
    features_idx: tuple[int, int, int] = (0, 1, 2),
    max_points: int = 5000,
    plot_format: PlotFormat = "png",
    dpi: int = 300,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Build one catalog DGP bank and save source-level visualization files."""
    from .catalog import source_preset

    source = source_preset(source_name)()
    bank = build_dgp_bank(source, seed=seed, noise_kind=noise_kind, verbose=False)
    reference_source = _unshocked_reference_source(source)
    reference_bank = (
        build_dgp_bank(reference_source, seed=seed, noise_kind=noise_kind, verbose=False)
        if reference_source is not None
        else None
    )
    return plot_dgp_bank_saves(
        bank,
        out=out,
        kind=kind,
        source=source,
        reference_bank=reference_bank,
        sigma=sigma,
        feature_idx=feature_idx,
        features_idx=features_idx,
        max_points=max_points,
        plot_format=plot_format,
        dpi=dpi,
        overwrite=overwrite,
    )
