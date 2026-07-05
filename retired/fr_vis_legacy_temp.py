from __future__ import annotations
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib.axes import Axes  # <--- Import the type explicitly
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import List, Optional, Literal, Any, Tuple
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
from typing import cast, Any, TYPE_CHECKING, Final
import polars as pl
from fr_tr import SeqKind

from enum import StrEnum, auto
from attrs import define, field, Factory

if TYPE_CHECKING:
    from fr_tr import RunResult

"""call site comparison:
# ═══ OLD (scattered, untyped, numpy-passing) ═══
from fr_vis import plot_forecast_diagnostics, _pl_seq_to_numpy_td

feature_cols = [f"feat_{d}" for d in range(db.channels)]
pred = _pl_seq_to_numpy_td(tr.final_seq, kind="pred", feature_cols=feature_cols)
truth = _pl_seq_to_numpy_td(tr.final_seq, kind="true", feature_cols=feature_cols)
eig_sum = _pl_seq_to_numpy_td(tr.final_seq, kind="eigenvalue_sum", feature_cols=feature_cols)
eig_max = _pl_seq_to_numpy_td(tr.final_seq, kind="eigenvalue_max", feature_cols=feature_cols)
eig_logdet = _pl_seq_to_numpy_td(tr.final_seq, kind="eigenvalue_logdet", feature_cols=feature_cols)

plot_forecast_diagnostics(
    pred, truth, feature_idx=0,
    eig_sum=eig_sum, eig_max=eig_max, eig_logdet=eig_logdet,
    save_path="diag.png"
)

# ═══ NEW (one line, typed, auto-configured) ═══
from fr_vis import diagnostics, draw

draw(diagnostics(result, save_path="diag.png"), result)

# Override anything:
draw(diagnostics(result, feature_idx=2, theme=PlotTheme.DARK), result)

# Non-FERN models → factory auto-skips eigen panels:
draw(diagnostics(patchtst_result), patchtst_result)

# Seq plot:
from fr_vis import seq_plot
draw(seq_plot(result), result)

# Attractor:
from fr_vis import attractor
draw(attractor(result, save_path="attractor.png"), result)

# From disk (no Trainer needed):
result = RunResult.load(Path("results/lorenz_fern_s42_p96"))
draw(diagnostics(result), result)
"""

"""
fr_vis.py — Typed visualization specs and factory constructors.

Three figure families (each has its own renderer in fr_vis.py):
  DiagSpec    — gridspec panel stack: forecast + error + eigen heatmaps
  SeqSpec     — single axes with gradient-line overlay (alpha/width from eigen)
  AttractorSpec — 3D phase-space reconstruction

Factory constructors (consume RunResult → auto-configured spec):
  diagnostics(result, ...)   → DiagSpec
  seq_plot(result, ...)      → SeqSpec
  attractor(result, ...)     → AttractorSpec
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PlotTheme(StrEnum):
    PAPER    = auto()   # serif, white bg, constrained_layout
    NOTEBOOK = auto()   # sans-serif, looser spacing
    DARK     = auto()   # dark_background for slides


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel specs (for DiagSpec only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    kind: str                  # SeqKind value: "eigenvalue_sum", "eigenvalue_max", "eigenvalue_logdet"
    label: str                 # y-axis label: "Sum of\nEigenvalues"
    cmap: str = "Blues"        # overridden to 'coolwarm' if data crosses zero
    height_ratio: int = 1

# Exhaustive union
Panel = ForecastPanel | ErrorPanel | EigenPanel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figure specs (one per figure family)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@define(frozen=True, slots=True)
class DiagSpec:
    """Panel-stack diagnostics figure (forecast + error + eigen heatmaps)."""
    panels: tuple[Panel, ...]
    feature_idx: int = 0
    theme: PlotTheme = PlotTheme.PAPER
    title: str | None = None        # #CHG: default None (no title for papers)
    save_path: str | None = None
    dpi: int = 300
    width: float = 16.0


@define(frozen=True, slots=True)
class SeqSpec:
    """Gradient-line sequence plot (alpha/width driven by eigen streams)."""
    feature_idx: int = 0
    alpha_kind: str | None = None    # SeqKind value or None (no driver)
    width_kind: str | None = None    # SeqKind value or None
    cmap: str = "viridis"
    draw_base_pred: bool = False
    theme: PlotTheme = PlotTheme.PAPER
    title: str | None = None         # #CHG: default None
    save_path: str | None = None
    dpi: int = 300


# CHG: `Final` marks "constant by contract" values for static tools/AI readers.
# Runtime Python will not freeze these values; this is a type-level intent signal.
# Motivation: keep API defaults/sentinels stable and self-documenting.
# CHG: explicit constants avoid magic strings/None sentinels in public API defaults.
AUTO_EIG_LABEL: Final[str] = "__AUTO_EIG_LABEL__"
DEFAULT_EIG_LABEL: Final[str] = "Max Eigenvalue"


@define(frozen=True, slots=True)
class AttractorSpec:
    """3D phase-space attractor reconstruction."""
    features_idx: tuple[int, int, int] = (0, 1, 2)
    eig_kind: str | None = None      # SeqKind value driving pred color
    eig_label: str = DEFAULT_EIG_LABEL  # CHG: never default to opaque "Eigen Metric"
    mode: str = "side_by_side"       # "side_by_side" | "overlay"
    skip_transient: int = 0
    cmap_truth: str = "Blues"
    cmap_pred: str = "Oranges"
    theme: PlotTheme = PlotTheme.PAPER
    title: str | None = None         # #CHG: default None
    save_path: str | None = None
    dpi: int = 300


# CHG: canonical label maps prevent drift between selected eigen stream and displayed text.
_EIG_PANEL_LABEL_BY_KIND: dict[str, str] = {
    SeqKind.EIGENVALUE_SUM.value: "Sum of\nEigenvalues",
    SeqKind.EIGENVALUE_MAX.value: "Max\nEigenvalue",
    SeqKind.EIGENVALUE_LOGDET.value: "Log\nDeterminant",
}

# CHG: display labels (single-line) for legends/colorbars.
_EIG_DISPLAY_LABEL_BY_KIND: dict[str, str] = {
    SeqKind.EIGENVALUE_SUM.value: "Sum of Eigenvalues",
    SeqKind.EIGENVALUE_MAX.value: "Max Eigenvalue",
    SeqKind.EIGENVALUE_LOGDET.value: "Log Determinant",
}


def eig_display_label(kind: str | None, default: str = DEFAULT_EIG_LABEL) -> str:
    # CHG: one helper so all call-sites format eigen labels consistently.
    if kind is None:
        return default
    return _EIG_DISPLAY_LABEL_BY_KIND.get(kind, kind)


# Top-level spec union
VisSpec = DiagSpec | SeqSpec | AttractorSpec


def _ensure_seq_result(result: "RunResult") -> tuple[pl.DataFrame, list[str]]:
    """Load/memoize current DB/raw-window sequence artifacts for retired callers."""
    seq = result.seq
    if seq is None:
        from fr_tr import load_sequence_frame

        seq = load_sequence_frame(result.run_id.run_dir)
        if seq is None:
            raise ValueError(f"No sequence artifacts found in {result.run_id.run_dir}")
        result.seq = seq

    feature_cols = list(result.feature_names or [])
    if not feature_cols:
        feature_cols = [c for c in seq.columns if c not in {"kind", "sample_idx", "timestep_idx"}]
        result.feature_names = feature_cols
    return seq, feature_cols


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory constructors — consume RunResult, produce specs with smart defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def diagnostics(
    result: "RunResult",
    *,
    feature_idx: int = 0,
    save_path: str | None = None,
    theme: PlotTheme = PlotTheme.PAPER,
    title: str | None = None,
    cmap_error: str = "Reds",
    cmap_eig: str = "Blues",
) -> DiagSpec:
    """
    Auto-build diagnostics spec from RunResult.
    FERN runs get eigen panels; non-FERN runs get forecast+error only.
    """
    from fr_tr import SeqKind
    seq, feature_cols = _ensure_seq_result(result)

    panels: list[Panel] = [
        ForecastPanel(),
        ErrorPanel(cmap=cmap_error),
    ]
    # CHG: validate feature index early at factory boundary for faster feedback.
    if not (0 <= feature_idx < len(feature_cols)):
        raise ValueError(
            f"diagnostics: feature_idx={feature_idx} out of range for "
            f"D={len(feature_cols)} ({feature_cols})"
        )

    # CHG: diagnostics() has no eig_kind argument; gate only on model type.
    if result.is_fern:
        kinds_present = set(seq["kind"].unique().to_list())
        # _eigen_defs = [
        #     (SeqKind.EIGENVALUE_SUM,    "Sum of\nEigenvalues"),
        #     (SeqKind.EIGENVALUE_MAX,    "Max\nEigenvalue"),
        #     (SeqKind.EIGENVALUE_LOGDET, "Log\nDeterminant"),
        # ]
        # for kind, label in _eigen_defs:
        #     if kind.value in kinds_present:
        #         panels.append(EigenPanel(kind=kind.value, label=label, cmap=cmap_eig))

        for kind in (SeqKind.EIGENVALUE_SUM, SeqKind.EIGENVALUE_MAX, SeqKind.EIGENVALUE_LOGDET):
            if kind.value in kinds_present:
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
    result: "RunResult",
    *,
    feature_idx: int = 0,
    save_path: str | None = None,
    theme: PlotTheme = PlotTheme.PAPER,
    title: str | None = None,
    cmap: str = "viridis",
    draw_base_pred: bool = False,
) -> SeqSpec:
    """
    Auto-build seq spec from RunResult.
    Fern runs use eigenvalue_sum→alpha and eigenvalue_max→width. Non-Fern gets plain lines.
    """
    from fr_tr import SeqKind
    seq, feature_cols = _ensure_seq_result(result)

    alpha_kind = None
    width_kind = None
    # CHG: validate feature index early at factory boundary for faster feedback.
    if not (0 <= feature_idx < len(feature_cols)):
        raise ValueError(
            f"seq_plot: feature_idx={feature_idx} out of range for "
            f"D={len(feature_cols)} ({feature_cols})"
        )

    # CHG: seq_plot() also has no eig_kind argument; same fix as diagnostics().
    if result.is_fern:
        kinds_present = set(seq["kind"].unique().to_list())
        if SeqKind.EIGENVALUE_SUM.value in kinds_present:
            alpha_kind = SeqKind.EIGENVALUE_SUM.value
        if SeqKind.EIGENVALUE_MAX.value in kinds_present:
            width_kind = SeqKind.EIGENVALUE_MAX.value

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

from fr_tr import SeqKind
def attractor(
    result: "RunResult",
    *,
    eig_kind: str | None = None,
    features_idx: tuple[int, int, int] = (0, 1, 2),
    save_path: str | None = None,
    theme: PlotTheme = PlotTheme.PAPER,
    title: str | None = None,
    eig_label: str = AUTO_EIG_LABEL,  # CHG: explicit constant sentinel; no None in public arg
    mode: str = "side_by_side",
    skip_transient: int = 0,
) -> AttractorSpec:
    """
    Auto-build attractor spec from RunResult.
    Fern runs default to eigenvalue_max coloring unless eig_kind is provided.
    """
    from fr_tr import SeqKind
    seq, feature_cols = _ensure_seq_result(result)
    D = len(feature_cols)
    # CHG: require exactly 3 selected channels for 3D attractor API clarity.
    if len(features_idx) != 3:
        raise ValueError(f"attractor: features_idx must contain exactly 3 indices, got {features_idx}")
    if D < 3:
        raise ValueError(
            f"Attractor plot requires ≥3 channels, got {D} "
            f"({feature_cols}). "
            f"Use diagnostics() or seq_plot() for low-dimensional systems."
        )
    # CHG: explicit index guard to fail loud on invalid channel selection.
    if any(i < 0 or i >= D for i in features_idx):
        raise ValueError(f"features_idx={features_idx} out of range for D={D}")
    # CHG: validate UI mode explicitly instead of silently treating unknown mode as side_by_side.
    if mode not in {"side_by_side", "overlay"}:
        raise ValueError(f"attractor: mode={mode!r} must be 'side_by_side' or 'overlay'")
    # CHG: negative skip is always invalid and usually a caller bug.
    if skip_transient < 0:
        raise ValueError(f"attractor: skip_transient must be >= 0, got {skip_transient}")

    chosen_eig_kind = eig_kind
    kinds_present = set(seq["kind"].unique().to_list())  # CHG: used for auto-select + validation
    # CHG: only auto-select if caller did not explicitly pass eig_kind.
    if chosen_eig_kind is None and result.is_fern:
        if SeqKind.EIGENVALUE_MAX.value in kinds_present:
            chosen_eig_kind = SeqKind.EIGENVALUE_MAX.value
            print(f"ℹ️  attractor: auto-selected eig_kind={chosen_eig_kind!r}.")
        elif SeqKind.EIGENVALUE_SUM.value in kinds_present:
            chosen_eig_kind = SeqKind.EIGENVALUE_SUM.value
            print(
                f"⚠️  attractor: auto-selected fallback eig_kind={chosen_eig_kind!r} "
                f"(preferred {SeqKind.EIGENVALUE_MAX.value!r} missing)."
            )
        elif SeqKind.EIGENVALUE_LOGDET.value in kinds_present:
            chosen_eig_kind = SeqKind.EIGENVALUE_LOGDET.value
            print(
                f"⚠️  attractor: auto-selected fallback eig_kind={chosen_eig_kind!r} "
                f"(preferred {SeqKind.EIGENVALUE_MAX.value!r} missing)."
            )
        else:
            print(
                "⚠️  attractor: Fern run has no eigen stream "
                f"({SeqKind.EIGENVALUE_MAX.value!r}, {SeqKind.EIGENVALUE_SUM.value!r}, "
                f"{SeqKind.EIGENVALUE_LOGDET.value!r}); "
                "prediction will be drawn as a solid line."
            )
    # CHG: explicit caller-provided eig_kind must exist; fail loud with available kinds.
    if chosen_eig_kind is not None and chosen_eig_kind not in kinds_present:
        raise ValueError(
            f"attractor: eig_kind={chosen_eig_kind!r} not found in seq kinds {sorted(kinds_present)}"
        )

    # CHG: resolve label by explicit override-or-auto, robust to future text changes.
    chosen_eig_label = (
        eig_display_label(chosen_eig_kind, default=DEFAULT_EIG_LABEL)
        if eig_label == AUTO_EIG_LABEL
        else eig_label
    )

    return AttractorSpec(
        features_idx=features_idx,
        eig_kind=chosen_eig_kind,
        eig_label=chosen_eig_label,
        mode=mode,
        skip_transient=skip_transient,
        theme=theme,
        title=title,
        save_path=save_path,
    )
#region [Plot] Forecast Diagnostics
# ==============================================================================
# 2. FORECAST DIAGNOSTICS (Killer Plot)
# ==============================================================================


 


 
#region [Plot] Attractor Comparison (Combined)
# ==============================================================================
# 3. ATTRACTOR COMPARISON (Combined 3D Plot)
# ==============================================================================





 
# ICLR plot example
#region ICLR plot example
def plot_final_iclr_figure(full_series, input_len, patch_len):
    """
    Generates the final, publication-quality figure with a more plausible
    forecast series that aligns better with the geometric guides.
    """
    # --- 1. ICLR-Ready Setup and Styling ---
    plt.style.use("seaborn-v0_8-paper")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 20,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("FERN's Geometric Forecasting Mechanism", weight="bold")

    # Colorblind-friendly palette
    COLOR_INPUT = "#377eb8"  # Blue
    COLOR_PREDICT = "#ff7f00"  # Orange
    COLOR_TRUTH = "#4daf4a"  # Green
    COLOR_LATENT = "#984ea3"  # Purple
    LINEWIDTH = 2.2

    # --- 2. Panel (a): Input Series ---
    ax1 = axes[0]
    input_series = full_series[:input_len]
    ax1.plot(
        np.arange(input_len),
        input_series,
        color=COLOR_INPUT,
        linewidth=LINEWIDTH,
        label="Input Series",
    )
    num_input_patches = input_len // patch_len
    for i in range(1, num_input_patches):
        ax1.axvline(
            x=i * patch_len - 0.5,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
    ax1.set_title("(a) Input Series", weight="bold")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left")
    ax1.set_xlim(0, input_len)

    # --- 3. Panel (b): Latent Space Transformation ---
    ax2 = axes[1]
    np.random.seed(0)
    points = np.random.randn(200, 2)
    scale = np.array([[2.8, 0], [0, 0.9]])
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transform = rotation @ scale
    transformed_points = (transform @ points.T).T
    ax2.scatter(
        points[:, 0],
        points[:, 1],
        alpha=0.6,
        s=20,
        color=COLOR_INPUT,
        label="Isotropic Gaussian Latent",
    )
    ax2.scatter(
        transformed_points[:, 0],
        transformed_points[:, 1],
        alpha=0.6,
        s=20,
        color=COLOR_LATENT,
        label="Anisotropic Encoded Latent",
    )
    iso_radius = 2.5
    circ = Ellipse(
        xy=(0, 0),
        width=2 * iso_radius,
        height=2 * iso_radius,
        angle=0,
        facecolor=COLOR_INPUT,
        alpha=0.1,
    )
    ax2.add_patch(circ)
    cov = np.cov(transformed_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    ell_scale = 2.5
    ell_width = 2 * ell_scale * np.sqrt(eigvals.max())
    ell_height = 2 * ell_scale * np.sqrt(eigvals.min())
    ell_angle_deg = np.degrees(np.arctan2(eigvecs[1, eigvals.argmax()], eigvecs[0, eigvals.argmax()]))
    ell = Ellipse(
        xy=(0, 0),
        width=ell_width,
        height=ell_height,
        angle=ell_angle_deg,
        facecolor=COLOR_LATENT,
        alpha=0.15,
    )
    ax2.add_patch(ell)
    ax2.set_title("(b) Latent Space Transformation", weight="bold")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("Latent Dimension 1")
    ax2.set_ylabel("Latent Dimension 2")
    ax2.legend(loc="upper right")

    # --- 4. Panel (c): Patchwise Geometric Forecast ---
    ax3 = axes[2]
    forecast_len = patch_len * 3
    forecast_x = np.arange(input_len, input_len + forecast_len)
    last_val = input_series[-1]

    # **MODIFIED LINE**: Create a U-shaped forecast to better match the ground truth trend
    # This ensures the forecast line aligns visually with all three ellipsoids.
    t_poly = np.linspace(-1.5, 1.5, forecast_len)
    parabolic_trend = 3.0 * (t_poly**2)
    t_sin = np.linspace(0, 3 * np.pi, forecast_len)
    forecast_y = last_val + parabolic_trend - 2.5 + np.sin(t_sin * 1.2) * 0.8 + (np.random.randn(forecast_len) * 0.4)

    ground_truth_y = full_series[input_len : input_len + forecast_len]

    context_x = np.arange(input_len - patch_len, input_len + 1)
    context_y = full_series[input_len - patch_len : input_len + 1]
    ax3.plot(context_x, context_y, color=COLOR_INPUT, linewidth=LINEWIDTH, label="Input")
    ax3.plot(
        forecast_x,
        forecast_y,
        color=COLOR_PREDICT,
        linewidth=LINEWIDTH,
        linestyle="--",
        label="Forecast",
    )
    ax3.plot(
        forecast_x,
        ground_truth_y,
        color=COLOR_TRUTH,
        linewidth=LINEWIDTH,
        linestyle="-",
        label="Ground Truth",
    )

    global_pred_std = np.std(forecast_y) + 1e-6
    for i in range(forecast_len // patch_len):
        start_idx, end_idx = i * patch_len, (i + 1) * patch_len
        patch_x_coords = forecast_x[start_idx:end_idx]
        patch_y_truth = ground_truth_y[start_idx:end_idx]
        patch_y_pred = forecast_y[start_idx:end_idx]

        center_x, center_y = np.mean(patch_x_coords), np.mean(patch_y_pred)

        slope = np.polyfit(patch_x_coords, patch_y_truth, 1)[0]
        angle_deg = np.degrees(np.arctan(slope))

        patch_pred_std = np.std(patch_y_pred)
        rel_unc = np.clip(patch_pred_std / global_pred_std, 0.7, 1.5)
        height = 2.5 * rel_unc

        ellipse = Ellipse(
            xy=(center_x, center_y),
            width=patch_len * 1.2,
            height=float(height),
            angle=angle_deg,
            facecolor=COLOR_PREDICT,
            alpha=0.3,
        )
        ax3.add_patch(ellipse)
        ax3.axvline(
            x=end_idx + input_len - patch_len - 0.5,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

    ax3.set_title("(c) Patchwise Geometric Forecast", weight="bold")
    ax3.set_xlabel("Time Step")
    ax3.legend(loc="upper left")
    ax3.set_xlim(input_len - patch_len, input_len + forecast_len)

    # --- 5. Final Polishing ---
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(width=1.2)
        ax.grid(True, which="major", linestyle=":", linewidth=0.5, color="gainsboro")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig("fern_final_figure.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


"""
if __name__ == '__main__':
    np.random.seed(42)
    input_length = 96
    patch_length = 24
    total_length = 200
    t = np.linspace(0, 10 * np.pi, total_length)
    # Using the same data generation as before
    time_series = 3 * np.sin(t * 0.5) + np.cos(t * 2.5) + t * 0.15 + np.random.randn(total_length) * 0.4

    print("Generating the final, revised ICLR-style figure...")
    plot_final_iclr_figure(time_series, input_length, patch_length)
    print("Figure saved as 'fern_final_figure.pdf'")
"""
#endregion

#region [Helpers] Styles & Normalization
# ==============================================================================
# 1. STYLE & HELPER FUNCTIONS
# ==============================================================================



def _create_colored_line_segments(
    xyz_data: np.ndarray, 
    color_data: np.ndarray, 
    cmap_name: str, 
    lw: float = 1.0, 
    alpha: float = 0.9
) -> Tuple[Line3DCollection, mcolors.Normalize, Any]:
    # CHG: fail loud on malformed inputs (better ergonomics than cryptic numpy/mpl errors).
    if xyz_data.ndim != 2 or xyz_data.shape[0] != 3:
        raise ValueError(
            f"_create_colored_line_segments expects xyz_data shape [3, T], got {tuple(xyz_data.shape)}"
        )
    T = xyz_data.shape[1]
    if T < 2:
        raise ValueError("_create_colored_line_segments requires at least 2 points (T >= 2)")
    if color_data.ndim != 1 or color_data.shape[0] != T:
        raise ValueError(
            f"_create_colored_line_segments expects color_data shape [T={T}], got {tuple(color_data.shape)}"
        )

    # 1. Transpose [3, T] -> [T, 3] to get points (x, y, z)
    points = xyz_data.T 
    # 2. Create segments: Shape [T-1, 2, 3]
    # Each segment is [(x_i, y_i, z_i), (x_{i+1}, y_{i+1}, z_{i+1})]
    # segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    
    # 3. Robust Normalization (2nd/98th percentile) to ignore outliers
    valid_c = color_data[np.isfinite(color_data)]
    if len(valid_c) > 0:
        vmin, vmax = np.percentile(valid_c, 2), np.percentile(valid_c, 98)
    else:
        vmin, vmax = 0.0, 1.0
    
    if vmax <= vmin:
        vmax = vmin + 1e-6  # GUARD: constant color driver still needs non-zero range

    norm = mcolors.Normalize(vmin=cast(float, vmin), vmax=cast(float, vmax))
    cmap = plt.get_cmap(cmap_name)
    # 4. Map values to RGBA
    # We map the raw values first, then average the RGBA colors
    colors_rgba = cmap(norm(color_data)) # [T, 4]
    
    # Handle NaNs (make transparent)
    nan_mask = np.isnan(color_data)
    colors_rgba[nan_mask] = [0, 0, 0, 0]
    
    # 5. Segment Color = Average of Endpoints
    # We have T points but T-1 segments. 
    # Color of segment i is average of point i and point i+1
    seg_colors = (colors_rgba[:-1] + colors_rgba[1:]) / 2.0
    
    # 6. Return LineCollection, Normalization, and Colormap
    lc = Line3DCollection(segments, colors=seg_colors, linewidths=lw, alpha=alpha)
    return lc, norm, cmap
 #endregion


#region helper functions

def _add_gradient_line(
    ax: Axes,                 # where to draw
    y: np.ndarray,                # 1D array of y-values (length T)
    *,
    w_alpha: np.ndarray | None = None,  # per-point weights in [0,1] to drive transparency
    w_lw:    np.ndarray | None = None,  # per-point weights in [0,1] to drive thickness
    color: str = "#E69F00",             # base color when not using a colormap
    cmap: str | None = None,            # optional colormap name; if set, color varies by w_alpha
    alpha_base: float = 0.15,           # minimum transparency
    alpha_scale: float = 0.85,          # how much transparency can increase from base
    lw_min: float = 1.25,               # minimum line width
    lw_max: float = 3.25,               # maximum line width
    zorder: int = 4,                    # draw above curves with lower zorder
):
    """
    _add_gradient_line
    Intent/contract:
    - Draw a polyline whose per-segment alpha/line-width are modulated by weights.
    - y: length T; there are (T-1) segments.
    - w_alpha / w_lw: accept length T or length T-1. If neither matches, resample to T.
    - This guard prevents broadcast errors if upstream series lengths ever drift.
    """
    T = len(y)
    x = np.arange(T, dtype=np.float32)

    def _align_weight(w, name):
        if w is None:
            return None
        w = np.asarray(w).reshape(-1)
        if len(w) == T:
            return w
        if len(w) == T - 1:
            # per-segment → per-point by padding last
            return np.concatenate([w, w[-1:]], axis=0)
        # Fallback: resample to T with linear interpolation
        w_lin = np.interp(
            np.linspace(0, len(w) - 1, T),
            np.arange(len(w), dtype=np.float32),
            w.astype(np.float32),
        )
        print(f"⚠️  {name} length {len(w)} ≠ T={T}; resampled to T.")
        return w_lin

    w_alpha = _align_weight(w_alpha, "w_alpha")
    w_lw    = _align_weight(w_lw,    "w_lw")

    if w_alpha is not None:
        assert len(w_alpha) == T, f"w_alpha has len {len(w_alpha)}, expected {T}"
    if w_lw is not None:
        assert len(w_lw) == T, f"w_lw has len {len(w_lw)}, expected {T}"

    """
    Convert the polyline into segments for LineCollection.
    Shape becomes [Nseg, 2, 2] where Nseg = T-1. Each segment is [(x_k,y_k), (x_{k+1},y_{k+1})].
    This is why later we index weights with [:-1]: there are T-1 segments.
    """
    # segments: [Nsegments, 2, 2] → [[(x0,y0),(x1,y1)], ...]
    segs = np.stack([np.column_stack([x[:-1], y[:-1]]),
                     np.column_stack([x[1:],  y[1:]])], axis=1)
    
    nseg = segs.shape[0] # Cache the number of segments.
    assert nseg == T - 1, f"nseg={nseg}, but T-1={T-1}"
    """
    Colormap branch: if a cmap is provided, color each segment by w_alpha (after slicing to T-1).
    colors = cmap_obj(weights) returns RGBA per segment based on the colormap.

    Then we override the alpha channel (colors[:, 3]) to be alpha_base + alpha_scale * weight, 
    clipped to [0,1]. This keeps transparency meaningful even when the colormap has its own alpha.
    """
    if cmap:
        cmap_obj = plt.get_cmap(cmap) 
        colors = cmap_obj((w_alpha[:-1] if w_alpha is not None else np.zeros(nseg)))
        # override alpha from weights
        colors[:, 3] = (alpha_base + alpha_scale * (w_alpha[:-1] if w_alpha is not None else 0)).clip(0, 1) # pyright: ignore
    else: 
        """
        Solid-color branch: all segments start with the same RGBA. If w_alpha is given, 
        vary only the alpha channel per segment using the same base + scale * weight rule.
        """
        rgba = np.array(mcolors.to_rgba(color), dtype=np.float32)
        colors = np.tile(rgba, (nseg, 1))
        if w_alpha is not None:
            colors[:, 3] = (alpha_base + alpha_scale * w_alpha[:-1]).clip(0, 1) # pyright: ignore
    """
    Compute per-segment line widths. Default to lw_min. If w_lw is supplied, map it linearly 
    into [lw_min, lw_max] (again using [:-1] to match segment count).
    """
    lws = np.full(nseg, lw_min, dtype=np.float32)
    if w_lw is not None:
        lws = lw_min + (lw_max - lw_min) * w_lw[:-1]
    """
    Create the LineCollection with your segments, per-segment colors, and per-segment widths.
    Add it to the Axes. (You can call ax.autoscale_view() after this if there 
    wasn’t already a line plotted that set limits.)
    """
    lc = LineCollection(cast(Any, segs), colors=colors, linewidths=lws, zorder=zorder)
    ax.add_collection(lc)

def clean_and_normalize_seq(
    w: np.ndarray, 
    T: int | None = None, 
    *,
    clip_lo_pct: float = 5.0,
    clip_hi_pct: float = 95.0,
    fixed_min: float | None = None,
    fixed_max: float | None = None,
    gamma: float = 0.8,
    smooth_alpha: float = 0.0
) -> np.ndarray:
    """
    Robust normalization for visualization.
    1. Resamples to length T (if needed).
    2. Replaces NaNs/Infs safely.
    3. Clips outliers (Percentile) OR uses fixed range.
    4. Applies Gamma correction for better contrast.
    """
    w = np.asarray(w, dtype=np.float32).ravel()
    
    # 1. Alignment (Resample to T if mismatch)
    if T is not None and w.size != T:
        if w.size == T - 1:
            w = np.concatenate([w, w[-1:]]) # Simple pad
        else:
            # Linear interpolation for size mismatch
            w = np.interp(
                np.linspace(0, w.size - 1, T),
                np.arange(w.size),
                w
            )

    # 2. Safety (NaN/Inf removal)
    # logic: Replace NaN with median, +/-Inf with min/max of valid data
    if not np.isfinite(w).all():
        finite_mask = np.isfinite(w)
        if finite_mask.any():
            r_nan = np.nanmedian(w)
            r_pos = w[finite_mask].max()
            r_neg = w[finite_mask].min()
        else:
            r_nan, r_pos, r_neg = 0.0, 0.0, 0.0
        w = np.nan_to_num(
            w, nan=float(r_nan), posinf=float(r_pos), neginf=float(r_neg))

    # 3. Smoothing (Optional EMA)
    if smooth_alpha > 0:
        # Vectorized approx or simple loop
        out = np.empty_like(w)
        out[0] = w[0]
        for i in range(1, len(w)):
            out[i] = smooth_alpha * out[i-1] + (1 - smooth_alpha) * w[i]
        w = out

    # 4. Normalization Range
    if fixed_min is not None and fixed_max is not None: 
        lo, hi = fixed_min, fixed_max # Comparison Mode: Use fixed shared scale
    else:  # Adaptive Mode: Use percentiles to ignore outliers
        lo = np.percentile(w, clip_lo_pct) 
        hi = np.percentile(w, clip_hi_pct)

    # Prevent div/0
    if hi <= lo: hi = lo + 1e-6

    # 5. Apply Norm + Gamma
    w_norm = (np.clip(w, lo, hi) - lo) / (hi - lo)
    
    if gamma != 1.0: w_norm = np.power(np.clip(w_norm, 1e-6, 1.0), gamma)
        
    return w_norm
#endregion helper functions




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _apply_theme(theme, dpi: int) -> None:
    """Replaces _apply_iclr_style.  Match on PlotTheme."""
    # Import locally so this block is self-contained during transition 

    plt.style.use("default")  # CHG: reset sticky matplotlib state
    mpl.rcParams.update(mpl.rcParamsDefault)  # CHG

    match theme:
        case PlotTheme.PAPER:
            plt.rcParams.update({
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
            })
        case PlotTheme.NOTEBOOK:
            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.size": 12,
                "lines.linewidth": 1.5,
                "figure.dpi": 100,
                "savefig.dpi": 150,
            })
        case PlotTheme.DARK:
            plt.style.use("dark_background")
            plt.rcParams.update({"font.size": 13, "figure.dpi": dpi, "savefig.dpi": dpi})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extraction helpers (Polars → numpy boundary)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#NOTE only implemented the mean ver, not eg the first sample ver

def _extract(df: pl.DataFrame, kind_val: str, cols: list[str]) -> np.ndarray:
    """Filter by kind, average forecast samples, return timestep-major [T, D]."""
    sub = df.filter(pl.col("kind") == kind_val)
    if sub.height == 0:
        raise ValueError(f"No rows for kind={kind_val!r}")

    missing_cols = [col for col in cols if col not in sub.columns]  # CHG
    if missing_cols:
        fallback_cols = [col for col in sub.columns if col not in {"kind", "sample_idx", "timestep_idx"}]  # CHG
        if len(fallback_cols) != len(cols):
            raise ValueError(f"_extract: requested cols {cols} missing {missing_cols}, and fallback cols {fallback_cols} do not match expected width")
        print(f"⚠️  _extract: missing {missing_cols}; using stored seq columns {fallback_cols} instead.")  # CHG
        cols = fallback_cols

    sub = sub.group_by("timestep_idx").agg(pl.col(cols).mean()).sort("timestep_idx")
    return sub.select(cols).to_numpy()


def _extract_if(
    df: pl.DataFrame, kind_val: str | None, cols: list[str], present: set[str]
) -> np.ndarray | None:
    """Extract if kind exists and is not None, else None.  The ONLY |None in public path."""
    if kind_val is None or kind_val not in present:
        return None
    return _extract(df, kind_val, cols)


def _pl_seq_to_numpy_td(seq: pl.DataFrame, kind: str, feature_cols: list[str]) -> np.ndarray:
    """Compatibility helper from the retired API."""
    return _extract(seq, kind, feature_cols)


def plot_forecast_diagnostics(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    feature_idx: int = 0,
    eig_sum: np.ndarray | None = None,
    eig_max: np.ndarray | None = None,
    eig_logdet: np.ndarray | None = None,
    save_path: str | None = None,
    dpi: int = 300,
) -> Figure:
    """Compatibility plotting helper for callers that already have numpy arrays."""
    panels = [("Forecast", None), ("Abs Error", None)]
    if eig_sum is not None:
        panels.append(("Eigen Sum", eig_sum[:, feature_idx]))
    if eig_max is not None:
        panels.append(("Eigen Max", eig_max[:, feature_idx]))
    if eig_logdet is not None:
        panels.append(("Eigen LogDet", eig_logdet[:, feature_idx]))

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, max(4, 2 * len(panels))), sharex=True)
    axes = np.atleast_1d(axes)
    pred_1d = pred[:, feature_idx]
    truth_1d = truth[:, feature_idx]

    axes[0].plot(truth_1d, label="Ground Truth", color="#56B4E9")
    axes[0].plot(pred_1d, label="Prediction", color="#E69F00", linestyle="--")
    axes[0].legend()

    axes[1].imshow(np.abs(pred_1d - truth_1d)[None, :], aspect="auto", cmap="Reds")
    axes[1].set_ylabel("Abs Error")

    for ax, (label, data) in zip(axes[2:], panels[2:]):
        ax.imshow(np.asarray(data)[None, :], aspect="auto", cmap="Blues")
        ax.set_ylabel(label)

    axes[-1].set_xlabel("Time Step")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def _validate_feature_idx(feature_idx: int, feature_cols: list[str], *, where: str) -> None:
    # CHG: centralized guard so diagnostics/seq fail with a clear actionable message.
    D = len(feature_cols)
    if not (0 <= feature_idx < D):
        raise ValueError(f"{where}: feature_idx={feature_idx} out of range for D={D} ({feature_cols})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Top-level entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw(spec, result, *, return_fig: bool = False) -> tuple[Figure, dict[str, Any]] | None:
    """
    Single public entry point.  Dispatches on spec type.

    Why `return_fig` exists:
      - When False (default), this behaves like classic plotting code: render + `plt.show()`.
      - When True, caller receives `(fig, handles)` so notebook/tests can post-edit the figure
        (titles, annotations, extra artists) before showing/saving.

    Usage:
        draw(diagnostics(result, feature_idx=0), result)
        draw(seq_plot(result, save_path="seq.png"), result)
        draw(attractor(result), result)
        fig, handles = draw(attractor(result), result, return_fig=True)
        handles["ax_pred"].set_title("Custom title from caller")
        fig.savefig("custom.png", dpi=300, bbox_inches="tight")
    """ 

    match spec:
        case DiagSpec():
            return _draw_diagnostics(spec, result, return_fig=return_fig)
        case SeqSpec():
            return _draw_seq(spec, result, return_fig=return_fig)
        case AttractorSpec():
            return _draw_attractor(spec, result, return_fig=return_fig)
        case _:
            raise TypeError(f"Unknown spec type: {type(spec)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Diagnostics renderer (panel-based gridspec)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_diagnostics(spec, result, *, return_fig: bool = False) -> tuple[Figure, dict[str, Any]] | None:
    """Render DiagSpec: gridspec with forecast + error + eigen heatmaps."""
    from fr_tr import SeqKind
    seq, feature_cols = _ensure_seq_result(result)

    _apply_theme(spec.theme, spec.dpi)

    kinds_present = set(seq["kind"].unique().to_list())
    fi = spec.feature_idx
    _validate_feature_idx(fi, feature_cols, where="_draw_diagnostics")  # CHG

    # Data extraction (once)
    pred_td  = _extract(seq, SeqKind.PRED.value, feature_cols)
    truth_td = _extract(seq, SeqKind.TRUE.value, feature_cols)
    if pred_td.shape[0] == 0 or truth_td.shape[0] == 0:
        raise ValueError("_draw_diagnostics: empty prediction/truth sequence after extraction")  # CHG
    pred_1d  = pred_td[:, fi]
    truth_1d = truth_td[:, fi]
    if len(pred_1d) != len(truth_1d):
        raise ValueError(
            f"_draw_diagnostics: pred/truth length mismatch ({len(pred_1d)} vs {len(truth_1d)})"
        )  # CHG
    T = len(pred_1d)
    x_edges = np.arange(T + 1) - 0.5
    y_edges = np.array([0, 1])

    # Layout
    panels = spec.panels
    n = len(panels)
    ratios = [p.height_ratio for p in panels]
    # #CHG: height formula — 1.4 per ratio unit (was 6 + 1.2*n_eigen, equivalent)
    total_height = sum(ratios) * 1.4

    fig = plt.figure(figsize=(spec.width, total_height), layout="constrained")
    gs = fig.add_gridspec(n, 2, width_ratios=[50, 1], height_ratios=ratios)

    # #CHG: no suptitle by default (was always "Diagnostics: {feature}")
    if spec.title is not None:
        fig.suptitle(spec.title, weight="bold")

    # Why keep handle lists:
    # - lets callers surgically tweak specific panel/colorbar axes after draw(..., return_fig=True)
    panel_axes: list[Axes] = []
    cbar_axes: list[Axes] = []

    prev_ax = None
    for i, panel in enumerate(panels):
        ax  = fig.add_subplot(gs[i, 0], sharex=prev_ax)
        cax = fig.add_subplot(gs[i, 1])
        panel_axes.append(ax)
        cbar_axes.append(cax)

        match panel:
            case ForecastPanel():
                _render_forecast_panel(ax, cax, pred_1d, truth_1d, T)

            case ErrorPanel() as p:
                _render_error_panel(ax, cax, pred_1d, truth_1d, T, x_edges, y_edges, p)

            case EigenPanel() as p:
                eig_td = _extract_if(seq, p.kind, feature_cols, kinds_present)
                if eig_td is not None:
                    eig_1d = eig_td[:, fi]
                    _render_eigen_panel(ax, cax, eig_1d, T, x_edges, y_edges, p)
                else:
                    ax.text(0.5, 0.5, f"{p.kind} not available",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=12, color="gray")
                    cax.set_visible(False)

        # Hide x labels except last panel
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time Step")

        prev_ax = ax

    if spec.save_path:
        plt.savefig(spec.save_path, dpi=spec.dpi, bbox_inches="tight")
        print(f"🖼️ Saved diagnostics to: {spec.save_path}")
    if return_fig:
        return fig, {"panel_axes": panel_axes, "cbar_axes": cbar_axes}
    plt.show()
    return None


def _render_forecast_panel(
    ax: Axes, cax: Axes,
    pred_1d: np.ndarray, truth_1d: np.ndarray, T: int,
) -> None:
    """Panel: pred vs truth line plot."""
    time = np.arange(T)
    ax.plot(time, truth_1d, label="Ground Truth", color="#1f77b4", lw=2.5, alpha=0.8)
    ax.plot(time, pred_1d,  label="Prediction",   color="#ff7f0e", lw=2.5, alpha=0.9, ls="--")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylabel("Amplitude")
    # #CHG: removed sub-title "Forecast vs. Ground Truth" — redundant in paper context
    ax.margins(x=0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, T - 1)
    cax.set_visible(False)  # no colorbar for line plot


def _render_error_panel(
    ax: Axes, cax: Axes,
    pred_1d: np.ndarray, truth_1d: np.ndarray, T: int,
    x_edges: np.ndarray, y_edges: np.ndarray,
    panel,
) -> None:
    """Panel: error heatmap with log norm."""
    abs_error = np.abs(truth_1d - pred_1d)
    valid_err = abs_error[abs_error > 0]
    vmin = float(np.quantile(valid_err, 0.05)) if len(valid_err) > 0 else 1e-6
    vmax = float(np.quantile(valid_err, 0.95)) if len(valid_err) > 0 else 1.0
    # CHG: keep normalization numerically valid on constant-error or degenerate ranges.
    if panel.log_scale and vmin <= 0:
        vmin = 1e-6
    if vmax <= vmin:
        if panel.log_scale:
            vmax = vmin * (1.0 + 1e-6)
        else:
            vmax = vmin + 1e-6

    if panel.log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(
        x_edges, y_edges, abs_error.reshape(1, -1),
        cmap=panel.cmap, norm=norm, rasterized=True,
    )
    ax.set_yticks([])
    # #CHG: labelpad 50→38 (compress left whitespace)
    ax.set_ylabel("Error\n(Log Scale)" if panel.log_scale else "Error",
                  rotation=0, labelpad=38, va="center")

    cb = plt.colorbar(im, cax=cax)
    if panel.log_scale:
        cb.ax.yaxis.set_major_locator(
            ticker.LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=15)
        )
        fmt = ticker.ScalarFormatter()
        fmt.set_scientific(False)
        cb.ax.yaxis.set_major_formatter(fmt)


def _render_eigen_panel(
    ax: Axes, cax: Axes,
    data_1d: np.ndarray, T: int,
    x_edges: np.ndarray, y_edges: np.ndarray,
    panel,
) -> None:
    """Panel: eigen heatmap with auto-diverging colormap."""
    d_min, d_max = float(np.nanmin(data_1d)), float(np.nanmax(data_1d))

    # Smart colormap: diverging if data crosses zero (e.g. LogDet)
    if d_min < 0 < d_max:
        limit = max(abs(d_min), abs(d_max), 1e-6)  # GUARD: constant-zero case 
        norm = mcolors.Normalize(vmin=-limit, vmax=limit)
        cmap_actual = "coolwarm"
    else:
        # CHG: constant-valued panel needs non-zero width for stable normalization.
        if d_max <= d_min:
            d_max = d_min + 1e-6
        norm = mcolors.Normalize(vmin=d_min, vmax=d_max)
        cmap_actual = panel.cmap

    im = ax.pcolormesh(
        x_edges, y_edges, data_1d.reshape(1, -1),
        cmap=cmap_actual, norm=norm, rasterized=True,
    )
    ax.set_yticks([])
    # #CHG: labelpad 50→38 (compress left whitespace)
    ax.set_ylabel(panel.label, rotation=0, labelpad=38, va="center")
    plt.colorbar(im, cax=cax)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Seq renderer (gradient line)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_seq(spec, result, *, return_fig: bool = False) -> tuple[Figure, dict[str, Any]] | None:
    """Render SeqSpec: single axes with gradient-line overlay."""
    from fr_tr import SeqKind
    seq, feature_cols = _ensure_seq_result(result)

    _apply_theme(spec.theme, spec.dpi)

    kinds_present = set(seq["kind"].unique().to_list())
    fi = spec.feature_idx
    _validate_feature_idx(fi, feature_cols, where="_draw_seq")  # CHG

    pred_td  = _extract(seq, SeqKind.PRED.value, feature_cols)
    truth_td = _extract(seq, SeqKind.TRUE.value, feature_cols)

    # Slice to 1D
    def _to_1d(arr):
        if arr.ndim == 2:
            return arr[:, fi]
        return arr.ravel()

    y_pred  = _to_1d(pred_td)
    y_truth = _to_1d(truth_td)
    if len(y_pred) == 0 or len(y_truth) == 0:
        raise ValueError("_draw_seq: empty prediction/truth sequence after extraction")  # CHG
    if len(y_pred) != len(y_truth):
        raise ValueError(f"_draw_seq: pred/truth length mismatch ({len(y_pred)} vs {len(y_truth)})")  # CHG

    # Optional eigen drivers → normalize to [0,1]
    alpha_td = _extract_if(seq, spec.alpha_kind, feature_cols, kinds_present)
    width_td = _extract_if(seq, spec.width_kind, feature_cols, kinds_present)

    w_alpha = None
    if alpha_td is not None:
        w_alpha = clean_and_normalize_seq(_to_1d(alpha_td), T=len(y_pred))

    w_lw = None
    if width_td is not None:
        w_lw = clean_and_normalize_seq(_to_1d(width_td), T=len(y_pred))

    # --- Plot ---
    PRED_COLOR  = "#E69F00"
    TRUTH_COLOR = "#56B4E9"
 
    fig, ax = plt.subplots(figsize=(12, 6))

    # Base prediction line (if no gradient or explicitly requested)
    if spec.draw_base_pred or (w_alpha is None and w_lw is None):
        ax.plot(y_pred, label="Prediction", color=PRED_COLOR, linewidth=1.5, zorder=3)

    # Ground truth (always solid)
    ax.plot(y_truth, label="Ground Truth", color=TRUTH_COLOR, linewidth=1.8, alpha=0.8, zorder=2)

    # Gradient line (if any driver)
    if w_alpha is not None or w_lw is not None:
        _add_gradient_line(
            ax, y_pred,
            w_alpha=w_alpha, w_lw=w_lw,
            color=PRED_COLOR, cmap=spec.cmap,
            alpha_base=0.22, alpha_scale=0.88,
            lw_min=1.2, lw_max=2.9,
            zorder=5,
        )
        ax.autoscale_view()  # CHG: include gradient collection extents when base prediction line is suppressed.

    # #CHG: no title by default (was always "Sequence Comparison (Feature {idx})")
    if spec.title is not None:
        ax.set_title(spec.title, fontsize=16, fontweight="bold", pad=20)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)  # CHG
    ax.legend(fontsize=11)
    fig.tight_layout()

    if spec.save_path:
        plt.savefig(spec.save_path, dpi=spec.dpi, bbox_inches="tight")
        print(f"🖼️ Sequence plot saved to: {spec.save_path}")
    if return_fig:
        return fig, {"ax": ax}
    plt.show()
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Attractor renderer (3D)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_attractor(spec, result, *, return_fig: bool = False) -> tuple[Figure, dict[str, Any]] | None:
    """Render AttractorSpec: 3D phase-space plot."""
    from fr_tr import SeqKind
    seq, feature_cols = _ensure_seq_result(result)

    _apply_theme(spec.theme, spec.dpi)

    kinds_present = set(seq["kind"].unique().to_list())

    pred_td  = _extract(seq, SeqKind.PRED.value, feature_cols)
    truth_td = _extract(seq, SeqKind.TRUE.value, feature_cols)

    # Transient skip
    skip = spec.skip_transient
    pred_td  = pred_td[skip:]
    truth_td = truth_td[skip:]
    if pred_td.shape[0] < 2 or truth_td.shape[0] < 2:
        raise ValueError(
            f"_draw_attractor: need at least 2 timesteps after skip_transient={skip}, "
            f"got pred={pred_td.shape[0]}, truth={truth_td.shape[0]}"
        )  # CHG

    # [T, 3] → [3, T] for plotting
    fi = list(spec.features_idx)
    truth_xyz = truth_td[:, fi].T
    pred_xyz  = pred_td[:, fi].T

    # Eigen color data (for prediction)
    eig_td = _extract_if(seq, spec.eig_kind, feature_cols, kinds_present)
    pred_color_data = None
    if eig_td is not None:
        eig_td = eig_td[skip:]
        pred_color_data = np.mean(eig_td[:, fi], axis=1)  # [T]

    # Truth color = speed
    truth_speed = np.linalg.norm(np.diff(truth_xyz, axis=1), axis=0)
    truth_speed = np.concatenate(([truth_speed[0]], truth_speed))

    # Feature names for axis labels
    names = result.feature_names
    if all(i < len(names) for i in fi):
        lbl_x, lbl_y, lbl_z = [names[i] for i in fi]
    else:
        lbl_x, lbl_y, lbl_z = [f"Dim {i}" for i in fi]

    # --- Figure setup ---
    mode = spec.mode
    if mode == "overlay":
        fig = plt.figure(figsize=(14, 10), layout="tight")
        ax_truth = fig.add_subplot(1, 1, 1, projection="3d")
        ax_pred = ax_truth
    else:
        fig = plt.figure(figsize=(18, 8), layout="constrained")
        ax_truth = fig.add_subplot(1, 2, 1, projection="3d")
        ax_pred  = fig.add_subplot(1, 2, 2, projection="3d")

    # #CHG: no suptitle by default (was always a long multi-line title)
    if spec.title is not None:
        fig.suptitle(spec.title, y=0.99, weight="bold", fontsize=16)

    # --- Draw truth ---
    lc_truth, norm1, _ = _create_colored_line_segments(
        truth_xyz, truth_speed, spec.cmap_truth,
        lw=1.5 if mode == "overlay" else 1.2,
    )
    ax_truth.add_collection(lc_truth)
    sm1 = plt.cm.ScalarMappable(cmap=plt.get_cmap(spec.cmap_truth), norm=norm1)
    # fig.colorbar(sm1, ax=ax_truth,
    #              shrink=0.5 if mode == "overlay" else 0.6, pad=0.02, aspect=20)
    cbar_truth = fig.colorbar(
        sm1,
        ax=ax_truth,
        shrink=0.5 if mode == "overlay" else 0.6,
        pad=0.02,
        aspect=20,
    )  # CHG
    cbar_truth.set_label("Speed")  # CHG
    if mode == "side_by_side":
        ax_truth.set_title("Ground Truth (Color: Speed)", weight="bold")

    # --- Draw pred ---
    if pred_color_data is not None:
        lc_pred, norm2, _ = _create_colored_line_segments(
            pred_xyz, pred_color_data, spec.cmap_pred,
            lw=1.5 if mode == "overlay" else 1.2,
        )
        ax_pred.add_collection(lc_pred)
        sm2 = plt.cm.ScalarMappable(cmap=plt.get_cmap(spec.cmap_pred), norm=norm2)
        # fig.colorbar(sm2, ax=ax_pred,
        #              shrink=0.5 if mode == "overlay" else 0.6, pad=0.02, aspect=20)
        cbar_pred = fig.colorbar(
            sm2,
            ax=ax_pred,
            shrink=0.5 if mode == "overlay" else 0.6,
            pad=0.02,
            aspect=20,
        )  # CHG
        cbar_pred.set_label(spec.eig_label)  # CHG
        if mode == "side_by_side":
            ax_pred.set_title(f"Prediction (Color: {spec.eig_label})", weight="bold")  # CHG: human label over raw kind key

    else:
        ax_pred.plot(*pred_xyz, lw=1.0, color="#ff7f0e", alpha=0.9,
                     label="Prediction", ls="--")
        if mode == "side_by_side":
            ax_pred.set_title("Prediction (Solid)", weight="bold")

    # --- Unified formatting ---
    all_data = np.hstack((truth_xyz, pred_xyz))
    lim_x = (all_data[0].min(), all_data[0].max())
    lim_y = (all_data[1].min(), all_data[1].max())
    lim_z = (all_data[2].min(), all_data[2].max())

    axes_to_format = [ax_truth] if mode == "overlay" else [ax_truth, ax_pred]
    for ax in axes_to_format:
        ax.set_xlabel(lbl_x, labelpad=10)
        ax.set_ylabel(lbl_y, labelpad=10)
        ax.set_zlabel(lbl_z, labelpad=10)
        ax.set_xlim(lim_x)
        ax.set_ylim(lim_y)
        ax.set_zlim(lim_z)
        ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.view_init(elev=25, azim=-60)

    if spec.save_path:
        plt.savefig(spec.save_path, dpi=spec.dpi, bbox_inches="tight")
        print(f"🖼️ Saved 3D Attractor plot ({mode}) to: {spec.save_path}")
    if return_fig:
        # `ax_pred is ax_truth` in overlay mode; keys stay stable for caller code.
        return fig, {"ax_truth": ax_truth, "ax_pred": ax_pred}
    plt.show()
    return None
