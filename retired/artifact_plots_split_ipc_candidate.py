"""Artifact-level plots for forecast result folders.

This module reads a data-only artifact schema:

    pred.ipc          -> forecast samples or point forecasts
    truth.ipc         -> target trajectory
    eigenvalue.ipc    -> optional uncertainty streams, one eigen_summary column

It imports no Fern model, trainer, or database code. Fern-specific rendering is
only a data-presence rule: if eigen streams exist, add eigen panels or use them
to drive prediction-line alpha/width.
"""

from __future__ import annotations

import json
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from attrs import define

ArtifactPlotKind = Literal["diagnostics", "sequence", "attractor"]


class ArtifactStream(StrEnum):
    """Sequence stream names stored in artifact plot frames."""

    PRED = auto()
    TRUE = auto()
    FERN_MAX = auto()
    FERN_SUM = auto()
    FERN_LOGDET = auto()


_STREAM_FILE_BY_KIND: dict[ArtifactStream, str] = {
    ArtifactStream.PRED: "pred.ipc",
    ArtifactStream.TRUE: "truth.ipc",
}

_EIGENVALUE_FILE = "eigenvalue.ipc"
_EIGENVALUE_SUMMARY_COL = "eigen_summary"
_EIGEN_SUMMARY_TO_STREAM: dict[str, ArtifactStream] = {
    "max": ArtifactStream.FERN_MAX,
    "sum": ArtifactStream.FERN_SUM,
    "logdet": ArtifactStream.FERN_LOGDET,
}

_EIGEN_KINDS: tuple[ArtifactStream, ...] = (
    ArtifactStream.FERN_SUM,
    ArtifactStream.FERN_MAX,
    ArtifactStream.FERN_LOGDET,
)

_EIG_PANEL_LABEL_BY_KIND: dict[str, str] = {
    ArtifactStream.FERN_SUM.value: "Sum of\nEigenvalues",
    ArtifactStream.FERN_MAX.value: "Max\nEigenvalue",
    ArtifactStream.FERN_LOGDET.value: "Log\nDeterminant",
}

_EIG_DISPLAY_LABEL_BY_KIND: dict[str, str] = {
    ArtifactStream.FERN_SUM.value: "Sum of Eigenvalues",
    ArtifactStream.FERN_MAX.value: "Max Eigenvalue",
    ArtifactStream.FERN_LOGDET.value: "Log Determinant",
}


class PlotTheme(StrEnum):
    """Matplotlib style preset."""

    PAPER = auto()
    NOTEBOOK = auto()
    DARK = auto()


@define(frozen=True, slots=True)
class ArtifactSequence:
    """Loaded split IPC forecast artifact sequence.

    Args:
        run_dir: Folder containing split IPC files.
        seq: Sequence frame with columns ``kind``, ``sample_idx``,
            ``timestep_idx``, and features.
        feature_names: Feature columns to plot, in display order.

    Example:
        >>> artifacts = load_artifact_sequence("results/run_001")
        >>> sorted(artifacts.kinds_present)[:2]
        ['pred', 'true']
    """

    run_dir: Path
    seq: pl.DataFrame
    feature_names: tuple[str, ...]

    @property
    def kinds_present(self) -> set[str]:
        """Return the stream names present in the loaded frame."""
        if "kind" not in self.seq.columns:
            return set()
        return set(self.seq["kind"].unique().to_list())

    @property
    def has_eigen_streams(self) -> bool:
        """Return True when any eigen uncertainty stream is available."""
        present = self.kinds_present
        return any(kind.value in present for kind in _EIGEN_KINDS)


@define(frozen=True, slots=True)
class ForecastPanel:
    """Top panel: pred vs truth lines."""

    height_ratio: int = 3


@define(frozen=True, slots=True)
class ErrorPanel:
    """Heatmap of |pred - truth|."""

    cmap: str = "Reds"
    log_scale: bool = True
    height_ratio: int = 1


@define(frozen=True, slots=True)
class EigenPanel:
    """One heatmap row for an eigen stream."""

    kind: str
    label: str
    cmap: str = "Blues"
    height_ratio: int = 1


Panel = ForecastPanel | ErrorPanel | EigenPanel


@define(frozen=True, slots=True)
class DiagSpec:
    """Panel-stack diagnostics figure: forecast, error, and eigen heatmaps."""

    panels: tuple[Panel, ...]
    feature_idx: int = 0
    theme: PlotTheme = PlotTheme.PAPER
    title: str | None = None
    save_path: str | None = None
    dpi: int = 300
    width: float = 16.0


@define(frozen=True, slots=True)
class SeqSpec:
    """Gradient-line sequence plot with optional eigen-driven styling."""

    feature_idx: int = 0
    alpha_kind: str | None = None
    width_kind: str | None = None
    cmap: str = "viridis"
    draw_base_pred: bool = False
    theme: PlotTheme = PlotTheme.PAPER
    title: str | None = None
    save_path: str | None = None
    dpi: int = 300


@define(frozen=True, slots=True)
class AttractorSpec:
    """3D phase-space comparison plot."""

    features_idx: tuple[int, int, int] = (0, 1, 2)
    eig_kind: str | None = None
    eig_label: str = "Max Eigenvalue"
    mode: Literal["side_by_side", "overlay"] = "side_by_side"
    skip_transient: int = 0
    cmap_truth: str = "Blues"
    cmap_pred: str = "Oranges"
    theme: PlotTheme = PlotTheme.PAPER
    title: str | None = None
    save_path: str | None = None
    dpi: int = 300


VisSpec = DiagSpec | SeqSpec | AttractorSpec


def load_artifact_sequence(run_dir: str | Path) -> ArtifactSequence:
    """Load split IPC forecast artifacts from a run directory.

    Args:
        run_dir: Folder containing ``pred.ipc`` and ``truth.ipc``. Optional
            eigen summaries are stored in ``eigenvalue.ipc``.

    Returns:
        ArtifactSequence with one sequence frame and feature names.

    Example:
        >>> artifacts = load_artifact_sequence("results/example_run")
        >>> "pred" in artifacts.kinds_present
        True
    """
    base = Path(run_dir)
    frames: list[pl.DataFrame] = []
    for kind, filename in _STREAM_FILE_BY_KIND.items():
        path = base / filename
        if path.exists():
            df = pl.read_ipc(path)
            frames.append(
                df.with_columns(pl.lit(kind.value).alias("kind")).select(["kind", *df.columns])
            )

    eigenvalue_path = base / _EIGENVALUE_FILE
    if eigenvalue_path.exists():
        eigenvalue = pl.read_ipc(eigenvalue_path)
        if _EIGENVALUE_SUMMARY_COL not in eigenvalue.columns:
            raise ValueError(f"{eigenvalue_path} must contain {_EIGENVALUE_SUMMARY_COL!r}")
        for summary, kind in _EIGEN_SUMMARY_TO_STREAM.items():
            frame = eigenvalue.filter(pl.col(_EIGENVALUE_SUMMARY_COL) == summary)
            if frame.height:
                frame = frame.drop(_EIGENVALUE_SUMMARY_COL)
                frames.append(
                    frame.with_columns(pl.lit(kind.value).alias("kind")).select(["kind", *frame.columns])
                )

    if not frames:
        expected = ", ".join(_STREAM_FILE_BY_KIND.values())
        raise FileNotFoundError(
            f"no split IPC artifacts found in {base}; expected one of: {expected}"
        )

    seq = pl.concat(frames, how="diagonal_relaxed")
    feature_names = _load_feature_names(base, seq)
    return ArtifactSequence(run_dir=base, seq=seq, feature_names=tuple(feature_names))


def diagnostics(
    artifacts: ArtifactSequence | str | Path,
    *,
    feature_idx: int = 0,
    save_path: str | None = None,
    theme: PlotTheme = PlotTheme.PAPER,
    title: str | None = None,
    cmap_error: str = "Reds",
    cmap_eig: str = "Blues",
) -> DiagSpec:
    """Build a diagnostics plot spec from loaded artifacts.

    Eigen panels are added when eigen streams are present; no model name is
    checked.

    Example:
        >>> spec = diagnostics(load_artifact_sequence("results/example_run"))
        >>> [type(panel).__name__ for panel in spec.panels]
        ['ForecastPanel', 'ErrorPanel']
    """
    data = _coerce_artifacts(artifacts)
    _validate_feature_idx(feature_idx, list(data.feature_names), where="diagnostics")
    present = data.kinds_present
    panels: list[Panel] = [ForecastPanel(), ErrorPanel(cmap=cmap_error)]
    for kind in _EIGEN_KINDS:
        if kind.value in present:
            panels.append(
                EigenPanel(
                    kind=kind.value,
                    label=_EIG_PANEL_LABEL_BY_KIND[kind.value],
                    cmap=cmap_eig,
                )
            )
    return DiagSpec(
        panels=tuple(panels),
        feature_idx=feature_idx,
        theme=theme,
        title=title,
        save_path=save_path,
    )


def seq_plot(
    artifacts: ArtifactSequence | str | Path,
    *,
    feature_idx: int = 0,
    save_path: str | None = None,
    theme: PlotTheme = PlotTheme.PAPER,
    title: str | None = None,
    cmap: str = "viridis",
    draw_base_pred: bool = False,
) -> SeqSpec:
    """Build a sequence plot spec from loaded artifacts.

    ``fern_sum`` drives alpha and ``fern_max`` drives line width when present.

    Example:
        >>> spec = seq_plot(load_artifact_sequence("results/example_run"))
        >>> spec.feature_idx
        0
    """
    data = _coerce_artifacts(artifacts)
    _validate_feature_idx(feature_idx, list(data.feature_names), where="seq_plot")
    present = data.kinds_present
    alpha_kind = ArtifactStream.FERN_SUM.value if ArtifactStream.FERN_SUM.value in present else None
    width_kind = ArtifactStream.FERN_MAX.value if ArtifactStream.FERN_MAX.value in present else None
    return SeqSpec(
        feature_idx=feature_idx,
        alpha_kind=alpha_kind,
        width_kind=width_kind,
        cmap=cmap,
        draw_base_pred=draw_base_pred,
        theme=theme,
        title=title,
        save_path=save_path,
    )


def attractor(
    artifacts: ArtifactSequence | str | Path,
    *,
    eig_kind: str | None = None,
    features_idx: tuple[int, int, int] = (0, 1, 2),
    save_path: str | None = None,
    theme: PlotTheme = PlotTheme.PAPER,
    title: str | None = None,
    mode: Literal["side_by_side", "overlay"] = "side_by_side",
    skip_transient: int = 0,
) -> AttractorSpec:
    """Build a 3D attractor comparison spec from loaded artifacts.

    If ``eig_kind`` is omitted, ``fern_max`` is used when present.

    Example:
        >>> spec = attractor(load_artifact_sequence("results/example_run"))
        >>> spec.features_idx
        (0, 1, 2)
    """
    data = _coerce_artifacts(artifacts)
    if len(features_idx) != 3:
        raise ValueError(
            f"attractor: features_idx must contain exactly 3 indices, got {features_idx}"
        )
    if len(data.feature_names) < 3:
        raise ValueError(
            f"attractor plot requires at least 3 channels, got D={len(data.feature_names)} "
            f"({list(data.feature_names)})"
        )
    if any(i < 0 or i >= len(data.feature_names) for i in features_idx):
        raise ValueError(
            f"features_idx={features_idx} out of range for D={len(data.feature_names)}"
        )
    if skip_transient < 0:
        raise ValueError(f"attractor: skip_transient must be >= 0, got {skip_transient}")

    present = data.kinds_present
    chosen_eig_kind = eig_kind
    if chosen_eig_kind is None and ArtifactStream.FERN_MAX.value in present:
        chosen_eig_kind = ArtifactStream.FERN_MAX.value
    if chosen_eig_kind is not None and chosen_eig_kind not in present:
        raise ValueError(
            f"attractor: eig_kind={chosen_eig_kind!r} not found in streams {sorted(present)}"
        )

    return AttractorSpec(
        features_idx=features_idx,
        eig_kind=chosen_eig_kind,
        eig_label=eig_display_label(chosen_eig_kind),
        mode=mode,
        skip_transient=skip_transient,
        theme=theme,
        title=title,
        save_path=save_path,
    )


def draw(
    spec: VisSpec,
    artifacts: ArtifactSequence | str | Path,
    *,
    return_fig: bool = False,
) -> tuple[Any, dict[str, Any]] | None:
    """Render one artifact plot spec.

    Args:
        spec: DiagSpec, SeqSpec, or AttractorSpec.
        artifacts: ArtifactSequence or run directory path.
        return_fig: When True, return ``(fig, handles)`` for notebook/test
            post-processing. When False, save if ``spec.save_path`` is set;
            otherwise call ``plt.show()``.

    Example:
        >>> artifacts = load_artifact_sequence("results/example_run")
        >>> draw(diagnostics(artifacts, save_path="diag.png"), artifacts)
    """
    data = _coerce_artifacts(artifacts)
    match spec:
        case DiagSpec():
            return _draw_diagnostics(spec, data, return_fig=return_fig)
        case SeqSpec():
            return _draw_seq(spec, data, return_fig=return_fig)
        case AttractorSpec():
            return _draw_attractor(spec, data, return_fig=return_fig)
        case _:
            raise TypeError(f"unknown artifact plot spec type: {type(spec)}")


def plot_artifact_saves(
    run_dir: str | Path,
    *,
    out: str | Path,
    kind: ArtifactPlotKind = "diagnostics",
    feature_idx: int = 0,
    features_idx: tuple[int, int, int] = (0, 1, 2),
    theme: PlotTheme = PlotTheme.PAPER,
    dpi: int = 300,
    overwrite: bool = False,
) -> Path:
    """Load a run artifact folder and save one plot file.

    Args:
        run_dir: Folder containing split IPC forecast artifacts.
        out: Output figure path.
        kind: One of ``"diagnostics"``, ``"sequence"``, or ``"attractor"``.
        feature_idx: Channel index for diagnostics and sequence plots.
        features_idx: Three channel indices for attractor plots.
        theme: PlotTheme value.
        dpi: Save resolution.
        overwrite: Replace ``out`` when it already exists.

    Example:
        >>> plot_artifact_saves("results/run_001", out="diag.png", kind="diagnostics")
        PosixPath('diag.png')
    """
    out_path = Path(out)
    _check_write(out_path, overwrite=overwrite)
    artifacts = load_artifact_sequence(run_dir)
    if kind == "diagnostics":
        spec: VisSpec = diagnostics(
            artifacts,
            feature_idx=feature_idx,
            save_path=str(out_path),
            theme=theme,
        )
    elif kind == "sequence":
        spec = seq_plot(
            artifacts,
            feature_idx=feature_idx,
            save_path=str(out_path),
            theme=theme,
        )
    elif kind == "attractor":
        spec = attractor(
            artifacts,
            features_idx=features_idx,
            save_path=str(out_path),
            theme=theme,
        )
    else:
        raise ValueError("kind must be one of: diagnostics, sequence, attractor")

    fig_and_handles = draw(spec, artifacts, return_fig=True)
    if fig_and_handles is not None:
        fig, _ = fig_and_handles
        _, _, plt, _, _ = _import_matplotlib()
        plt.close(fig)
    return out_path


def eig_display_label(kind: str | None, default: str = "Max Eigenvalue") -> str:
    """Return the display label for an eigen stream key."""
    if kind is None:
        return default
    return _EIG_DISPLAY_LABEL_BY_KIND.get(kind, kind)


def _coerce_artifacts(artifacts: ArtifactSequence | str | Path) -> ArtifactSequence:
    if isinstance(artifacts, ArtifactSequence):
        return artifacts
    return load_artifact_sequence(artifacts)


def _load_feature_names(base: Path, seq: pl.DataFrame) -> list[str]:
    feature_path = base / "feature_names.json"
    if feature_path.exists():
        names = json.loads(feature_path.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{feature_path} must contain a JSON list of feature-name strings")
        return names
    return [c for c in seq.columns if c not in {"kind", "sample_idx", "timestep_idx"}]


def _check_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)


def _import_matplotlib() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from matplotlib.collections import LineCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError as exc:
        raise ImportError("matplotlib is required for artifact plotting") from exc
    return mpl, mcolors, plt, ticker, (LineCollection, Line3DCollection)


def _apply_theme(theme: PlotTheme, dpi: int) -> None:
    mpl, _, plt, _, _ = _import_matplotlib()
    plt.style.use("default")
    mpl.rcParams.update(mpl.rcParamsDefault)
    if theme is PlotTheme.PAPER:
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
    elif theme is PlotTheme.NOTEBOOK:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 12,
                "lines.linewidth": 1.5,
                "figure.dpi": 100,
                "savefig.dpi": 150,
            }
        )
    elif theme is PlotTheme.DARK:
        plt.style.use("dark_background")
        plt.rcParams.update({"font.size": 13, "figure.dpi": dpi, "savefig.dpi": dpi})
    else:
        raise ValueError(f"unknown PlotTheme: {theme!r}")


def _extract(df: pl.DataFrame, kind_val: str, cols: list[str]) -> np.ndarray:
    """Filter by kind, average forecast samples, return timestep-major [T, D]."""
    sub = df.filter(pl.col("kind") == kind_val)
    if sub.height == 0:
        raise ValueError(f"no rows for kind={kind_val!r}")

    missing_cols = [col for col in cols if col not in sub.columns]
    if missing_cols:
        fallback_cols = [col for col in sub.columns if col not in {"kind", "sample_idx", "timestep_idx"}]
        if len(fallback_cols) != len(cols):
            raise ValueError(
                f"_extract: requested cols {cols} missing {missing_cols}, and fallback cols "
                f"{fallback_cols} do not match expected width"
            )
        cols = fallback_cols

    return sub.group_by("timestep_idx").agg(pl.col(cols).mean()).sort("timestep_idx").select(cols).to_numpy()


def _extract_if(
    df: pl.DataFrame,
    kind_val: str | None,
    cols: list[str],
    present: set[str],
) -> np.ndarray | None:
    """Extract if kind exists and is not None, else None."""
    if kind_val is None or kind_val not in present:
        return None
    return _extract(df, kind_val, cols)


def _validate_feature_idx(feature_idx: int, feature_cols: list[str], *, where: str) -> None:
    d_dim = len(feature_cols)
    if not (0 <= feature_idx < d_dim):
        raise ValueError(
            f"{where}: feature_idx={feature_idx} out of range for D={d_dim} ({feature_cols})"
        )


def clean_and_normalize_seq(
    w: np.ndarray,
    T: int | None = None,
    *,
    clip_lo_pct: float = 5.0,
    clip_hi_pct: float = 95.0,
    fixed_min: float | None = None,
    fixed_max: float | None = None,
    gamma: float = 0.8,
    smooth_alpha: float = 0.0,
) -> np.ndarray:
    """Robustly normalize one visualization driver to [0, 1]."""
    w = np.asarray(w, dtype=np.float32).ravel()

    if T is not None and w.size != T:
        if w.size == T - 1:
            w = np.concatenate([w, w[-1:]])
        else:
            w = np.interp(np.linspace(0, w.size - 1, T), np.arange(w.size), w)

    if not np.isfinite(w).all():
        finite_mask = np.isfinite(w)
        if finite_mask.any():
            r_nan = np.nanmedian(w)
            r_pos = w[finite_mask].max()
            r_neg = w[finite_mask].min()
        else:
            r_nan, r_pos, r_neg = 0.0, 0.0, 0.0
        w = np.nan_to_num(w, nan=float(r_nan), posinf=float(r_pos), neginf=float(r_neg))

    if smooth_alpha > 0:
        out = np.empty_like(w)
        out[0] = w[0]
        for i in range(1, len(w)):
            out[i] = smooth_alpha * out[i - 1] + (1 - smooth_alpha) * w[i]
        w = out

    if fixed_min is not None and fixed_max is not None:
        lo, hi = fixed_min, fixed_max
    else:
        lo = np.percentile(w, clip_lo_pct)
        hi = np.percentile(w, clip_hi_pct)
    if hi <= lo:
        hi = lo + 1e-6

    w_norm = (np.clip(w, lo, hi) - lo) / (hi - lo)
    if gamma != 1.0:
        w_norm = np.power(np.clip(w_norm, 1e-6, 1.0), gamma)
    return w_norm


def _add_gradient_line(
    ax: Any,
    y: np.ndarray,
    *,
    w_alpha: np.ndarray | None = None,
    w_lw: np.ndarray | None = None,
    color: str = "#E69F00",
    cmap: str | None = None,
    alpha_base: float = 0.15,
    alpha_scale: float = 0.85,
    lw_min: float = 1.25,
    lw_max: float = 3.25,
    zorder: int = 4,
) -> None:
    """Draw a polyline whose alpha and line width are modulated by weights."""
    _, mcolors, plt, _, collections = _import_matplotlib()
    LineCollection, _ = collections
    t_len = len(y)
    x = np.arange(t_len, dtype=np.float32)

    def _align_weight(w: np.ndarray | None) -> np.ndarray | None:
        if w is None:
            return None
        w = np.asarray(w).reshape(-1)
        if len(w) == t_len:
            return w
        if len(w) == t_len - 1:
            return np.concatenate([w, w[-1:]], axis=0)
        return np.interp(
            np.linspace(0, len(w) - 1, t_len),
            np.arange(len(w), dtype=np.float32),
            w.astype(np.float32),
        )

    w_alpha = _align_weight(w_alpha)
    w_lw = _align_weight(w_lw)
    segs = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
    nseg = segs.shape[0]

    if cmap:
        cmap_obj = plt.get_cmap(cmap)
        weights = w_alpha[:-1] if w_alpha is not None else np.zeros(nseg)
        colors = cmap_obj(weights)
        colors[:, 3] = (alpha_base + alpha_scale * weights).clip(0, 1)
    else:
        rgba = np.array(mcolors.to_rgba(color), dtype=np.float32)
        colors = np.tile(rgba, (nseg, 1))
        if w_alpha is not None:
            colors[:, 3] = (alpha_base + alpha_scale * w_alpha[:-1]).clip(0, 1)

    lws = np.full(nseg, lw_min, dtype=np.float32)
    if w_lw is not None:
        lws = lw_min + (lw_max - lw_min) * w_lw[:-1]
    ax.add_collection(LineCollection(cast(Any, segs), colors=colors, linewidths=lws, zorder=zorder))


def _create_colored_line_segments(
    xyz_data: np.ndarray,
    color_data: np.ndarray,
    cmap_name: str,
    *,
    lw: float = 1.0,
    alpha: float = 0.9,
) -> tuple[Any, Any, Any]:
    _, mcolors, plt, _, collections = _import_matplotlib()
    _, Line3DCollection = collections
    if xyz_data.ndim != 2 or xyz_data.shape[0] != 3:
        raise ValueError(
            f"_create_colored_line_segments expects xyz_data shape [3, T], got {tuple(xyz_data.shape)}"
        )
    t_len = xyz_data.shape[1]
    if t_len < 2:
        raise ValueError("_create_colored_line_segments requires at least 2 points")
    if color_data.ndim != 1 or color_data.shape[0] != t_len:
        raise ValueError(f"color_data must have shape [T={t_len}], got {tuple(color_data.shape)}")

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
    return Line3DCollection(segments, colors=seg_colors, linewidths=lw, alpha=alpha), norm, cmap


def _draw_diagnostics(
    spec: DiagSpec,
    artifacts: ArtifactSequence,
    *,
    return_fig: bool = False,
) -> tuple[Any, dict[str, Any]] | None:
    """Render DiagSpec: gridspec with forecast, error, and eigen heatmaps."""
    mpl, mcolors, plt, ticker, _ = _import_matplotlib()
    _apply_theme(spec.theme, spec.dpi)

    seq = artifacts.seq
    feature_cols = list(artifacts.feature_names)
    present = artifacts.kinds_present
    fi = spec.feature_idx
    _validate_feature_idx(fi, feature_cols, where="_draw_diagnostics")

    pred_td = _extract(seq, ArtifactStream.PRED.value, feature_cols)
    truth_td = _extract(seq, ArtifactStream.TRUE.value, feature_cols)
    pred_1d = pred_td[:, fi]
    truth_1d = truth_td[:, fi]
    if len(pred_1d) == 0 or len(truth_1d) == 0:
        raise ValueError("_draw_diagnostics: empty prediction/truth sequence after extraction")
    if len(pred_1d) != len(truth_1d):
        raise ValueError(
            f"_draw_diagnostics: pred/truth length mismatch ({len(pred_1d)} vs {len(truth_1d)})"
        )

    t_len = len(pred_1d)
    x_edges = np.arange(t_len + 1) - 0.5
    y_edges = np.array([0, 1])
    panels = spec.panels
    ratios = [p.height_ratio for p in panels]
    total_height = sum(ratios) * 1.4

    fig = plt.figure(figsize=(spec.width, total_height), layout="constrained")
    gs = fig.add_gridspec(len(panels), 2, width_ratios=[50, 1], height_ratios=ratios)
    if spec.title is not None:
        fig.suptitle(spec.title, weight="bold")

    panel_axes: list[Any] = []
    cbar_axes: list[Any] = []
    prev_ax = None
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0], sharex=prev_ax)
        cax = fig.add_subplot(gs[i, 1])
        panel_axes.append(ax)
        cbar_axes.append(cax)

        match panel:
            case ForecastPanel():
                _render_forecast_panel(ax, cax, pred_1d, truth_1d, t_len)
            case ErrorPanel() as p:
                _render_error_panel(
                    ax, cax, pred_1d, truth_1d, x_edges, y_edges, p, mcolors, plt, ticker
                )
            case EigenPanel() as p:
                eig_td = _extract_if(seq, p.kind, feature_cols, present)
                if eig_td is not None:
                    _render_eigen_panel(ax, cax, eig_td[:, fi], x_edges, y_edges, p, mcolors, plt)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"{p.kind} not available",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    cax.set_visible(False)

        if i < len(panels) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time Step")
        prev_ax = ax

    if spec.save_path:
        fig.savefig(spec.save_path, dpi=spec.dpi, bbox_inches="tight")
    if return_fig:
        return fig, {"panel_axes": panel_axes, "cbar_axes": cbar_axes}
    plt.show()
    return None


def _render_forecast_panel(
    ax: Any, cax: Any, pred_1d: np.ndarray, truth_1d: np.ndarray, t_len: int
) -> None:
    """Panel: pred vs truth line plot."""
    time = np.arange(t_len)
    ax.plot(time, truth_1d, label="Ground Truth", color="#1f77b4", lw=2.5, alpha=0.8)
    ax.plot(time, pred_1d, label="Prediction", color="#ff7f0e", lw=2.5, alpha=0.9, ls="--")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylabel("Amplitude")
    ax.margins(x=0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, t_len - 1)
    cax.set_visible(False)


def _render_error_panel(
    ax: Any,
    cax: Any,
    pred_1d: np.ndarray,
    truth_1d: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    panel: ErrorPanel,
    mcolors: Any,
    plt: Any,
    ticker: Any,
) -> None:
    """Panel: error heatmap with log norm."""
    abs_error = np.abs(truth_1d - pred_1d)
    valid_err = abs_error[abs_error > 0]
    vmin = float(np.quantile(valid_err, 0.05)) if len(valid_err) > 0 else 1e-6
    vmax = float(np.quantile(valid_err, 0.95)) if len(valid_err) > 0 else 1.0
    if panel.log_scale and vmin <= 0:
        vmin = 1e-6
    if vmax <= vmin:
        vmax = vmin * (1.0 + 1e-6) if panel.log_scale else vmin + 1e-6
    norm = (
        mcolors.LogNorm(vmin=vmin, vmax=vmax)
        if panel.log_scale
        else mcolors.Normalize(vmin=vmin, vmax=vmax)
    )

    im = ax.pcolormesh(
        x_edges, y_edges, abs_error.reshape(1, -1), cmap=panel.cmap, norm=norm, rasterized=True
    )
    ax.set_yticks([])
    ax.set_ylabel(
        "Error\n(Log Scale)" if panel.log_scale else "Error", rotation=0, labelpad=38, va="center"
    )
    cb = plt.colorbar(im, cax=cax)
    if panel.log_scale:
        cb.ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=15))
        fmt = ticker.ScalarFormatter()
        fmt.set_scientific(False)
        cb.ax.yaxis.set_major_formatter(fmt)


def _render_eigen_panel(
    ax: Any,
    cax: Any,
    data_1d: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    panel: EigenPanel,
    mcolors: Any,
    plt: Any,
) -> None:
    """Panel: eigen heatmap with auto-diverging colormap."""
    d_min, d_max = float(np.nanmin(data_1d)), float(np.nanmax(data_1d))
    if d_min < 0 < d_max:
        limit = max(abs(d_min), abs(d_max), 1e-6)
        norm = mcolors.Normalize(vmin=-limit, vmax=limit)
        cmap_actual = "coolwarm"
    else:
        if d_max <= d_min:
            d_max = d_min + 1e-6
        norm = mcolors.Normalize(vmin=d_min, vmax=d_max)
        cmap_actual = panel.cmap

    im = ax.pcolormesh(
        x_edges, y_edges, data_1d.reshape(1, -1), cmap=cmap_actual, norm=norm, rasterized=True
    )
    ax.set_yticks([])
    ax.set_ylabel(panel.label, rotation=0, labelpad=38, va="center")
    plt.colorbar(im, cax=cax)


def _draw_seq(
    spec: SeqSpec,
    artifacts: ArtifactSequence,
    *,
    return_fig: bool = False,
) -> tuple[Any, dict[str, Any]] | None:
    """Render SeqSpec: single axes with gradient-line overlay."""
    _, _, plt, _, _ = _import_matplotlib()
    _apply_theme(spec.theme, spec.dpi)

    seq = artifacts.seq
    feature_cols = list(artifacts.feature_names)
    present = artifacts.kinds_present
    fi = spec.feature_idx
    _validate_feature_idx(fi, feature_cols, where="_draw_seq")

    pred_td = _extract(seq, ArtifactStream.PRED.value, feature_cols)
    truth_td = _extract(seq, ArtifactStream.TRUE.value, feature_cols)
    y_pred = pred_td[:, fi]
    y_truth = truth_td[:, fi]
    if len(y_pred) == 0 or len(y_truth) == 0:
        raise ValueError("_draw_seq: empty prediction/truth sequence after extraction")
    if len(y_pred) != len(y_truth):
        raise ValueError(f"_draw_seq: pred/truth length mismatch ({len(y_pred)} vs {len(y_truth)})")

    alpha_td = _extract_if(seq, spec.alpha_kind, feature_cols, present)
    width_td = _extract_if(seq, spec.width_kind, feature_cols, present)
    w_alpha = (
        clean_and_normalize_seq(alpha_td[:, fi], T=len(y_pred)) if alpha_td is not None else None
    )
    w_lw = clean_and_normalize_seq(width_td[:, fi], T=len(y_pred)) if width_td is not None else None

    pred_color = "#E69F00"
    truth_color = "#56B4E9"
    fig, ax = plt.subplots(figsize=(12, 6))
    if spec.draw_base_pred or (w_alpha is None and w_lw is None):
        ax.plot(y_pred, label="Prediction", color=pred_color, linewidth=1.5, zorder=3)
    ax.plot(y_truth, label="Ground Truth", color=truth_color, linewidth=1.8, alpha=0.8, zorder=2)
    if w_alpha is not None or w_lw is not None:
        _add_gradient_line(
            ax,
            y_pred,
            w_alpha=w_alpha,
            w_lw=w_lw,
            color=pred_color,
            cmap=spec.cmap,
            alpha_base=0.22,
            alpha_scale=0.88,
            lw_min=1.2,
            lw_max=2.9,
            zorder=5,
        )
        ax.autoscale_view()

    if spec.title is not None:
        ax.set_title(spec.title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if spec.save_path:
        fig.savefig(spec.save_path, dpi=spec.dpi, bbox_inches="tight")
    if return_fig:
        return fig, {"ax": ax}
    plt.show()
    return None


def _draw_attractor(
    spec: AttractorSpec,
    artifacts: ArtifactSequence,
    *,
    return_fig: bool = False,
) -> tuple[Any, dict[str, Any]] | None:
    """Render AttractorSpec: 3D phase-space plot."""
    _, _, plt, _, _ = _import_matplotlib()
    _apply_theme(spec.theme, spec.dpi)

    seq = artifacts.seq
    feature_cols = list(artifacts.feature_names)
    present = artifacts.kinds_present
    pred_td = _extract(seq, ArtifactStream.PRED.value, feature_cols)[spec.skip_transient :]
    truth_td = _extract(seq, ArtifactStream.TRUE.value, feature_cols)[spec.skip_transient :]
    if pred_td.shape[0] < 2 or truth_td.shape[0] < 2:
        raise ValueError(
            f"_draw_attractor: need at least 2 timesteps after skip_transient={spec.skip_transient}, "
            f"got pred={pred_td.shape[0]}, truth={truth_td.shape[0]}"
        )

    fi = list(spec.features_idx)
    truth_xyz = truth_td[:, fi].T
    pred_xyz = pred_td[:, fi].T
    eig_td = _extract_if(seq, spec.eig_kind, feature_cols, present)
    pred_color_data = None
    if eig_td is not None:
        pred_color_data = np.mean(eig_td[spec.skip_transient :, fi], axis=1)

    truth_speed = np.linalg.norm(np.diff(truth_xyz, axis=1), axis=0)
    truth_speed = np.concatenate(([truth_speed[0]], truth_speed))
    labels = [feature_cols[i] if i < len(feature_cols) else f"Dim {i}" for i in fi]

    if spec.mode == "overlay":
        fig = plt.figure(figsize=(14, 10), layout="tight")
        ax_truth = fig.add_subplot(1, 1, 1, projection="3d")
        ax_pred = ax_truth
    else:
        fig = plt.figure(figsize=(18, 8), layout="constrained")
        ax_truth = fig.add_subplot(1, 2, 1, projection="3d")
        ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
    if spec.title is not None:
        fig.suptitle(spec.title, y=0.99, weight="bold", fontsize=16)

    lc_truth, norm_truth, _ = _create_colored_line_segments(
        truth_xyz,
        truth_speed,
        spec.cmap_truth,
        lw=1.5 if spec.mode == "overlay" else 1.2,
    )
    ax_truth.add_collection(lc_truth)
    sm_truth = plt.cm.ScalarMappable(cmap=plt.get_cmap(spec.cmap_truth), norm=norm_truth)
    cbar_truth = fig.colorbar(
        sm_truth, ax=ax_truth, shrink=0.5 if spec.mode == "overlay" else 0.6, pad=0.02, aspect=20
    )
    cbar_truth.set_label("Speed")
    if spec.mode == "side_by_side":
        ax_truth.set_title("Ground Truth (Color: Speed)", weight="bold")

    if pred_color_data is not None:
        lc_pred, norm_pred, _ = _create_colored_line_segments(
            pred_xyz,
            pred_color_data,
            spec.cmap_pred,
            lw=1.5 if spec.mode == "overlay" else 1.2,
        )
        ax_pred.add_collection(lc_pred)
        sm_pred = plt.cm.ScalarMappable(cmap=plt.get_cmap(spec.cmap_pred), norm=norm_pred)
        cbar_pred = fig.colorbar(
            sm_pred, ax=ax_pred, shrink=0.5 if spec.mode == "overlay" else 0.6, pad=0.02, aspect=20
        )
        cbar_pred.set_label(spec.eig_label)
        if spec.mode == "side_by_side":
            ax_pred.set_title(f"Prediction (Color: {spec.eig_label})", weight="bold")
    else:
        ax_pred.plot(*pred_xyz, lw=1.0, color="#ff7f0e", alpha=0.9, label="Prediction", ls="--")
        if spec.mode == "side_by_side":
            ax_pred.set_title("Prediction (Solid)", weight="bold")

    all_data = np.hstack((truth_xyz, pred_xyz))
    limits = (
        (float(all_data[0].min()), float(all_data[0].max())),
        (float(all_data[1].min()), float(all_data[1].max())),
        (float(all_data[2].min()), float(all_data[2].max())),
    )
    for ax in [ax_truth] if spec.mode == "overlay" else [ax_truth, ax_pred]:
        ax.set_xlabel(labels[0], labelpad=10)
        ax.set_ylabel(labels[1], labelpad=10)
        ax.set_zlabel(labels[2], labelpad=10)
        ax.set_xlim(_expand_degenerate_limit(limits[0]))
        ax.set_ylim(_expand_degenerate_limit(limits[1]))
        ax.set_zlim(_expand_degenerate_limit(limits[2]))
        ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.view_init(elev=25, azim=-60)

    if spec.save_path:
        fig.savefig(spec.save_path, dpi=spec.dpi, bbox_inches="tight")
    if return_fig:
        return fig, {"ax_truth": ax_truth, "ax_pred": ax_pred}
    plt.show()
    return None


def _expand_degenerate_limit(limit: tuple[float, float]) -> tuple[float, float]:
    low, high = limit
    if high > low:
        return limit
    pad = max(abs(low) * 0.05, 1e-3)
    return low - pad, high + pad
