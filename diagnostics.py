"""Split diagnostics — compute reports, render Rich tables, pipeline stage."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from attrs import define

if TYPE_CHECKING:
    from rich.console import Console
    from rich.table import Table

    from .pipeline import DataBundle, BundleStep

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reports (frozen data, no side effects)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, slots=True)
class CountsReport:
    train: int
    val: int
    test: int
    seq_len: int | None = None
    pred_len: int | None = None

    @property
    def total(self) -> int:
        return self.train + self.val + self.test

    @property
    def min_required(self) -> int | None:
        if self.seq_len and self.pred_len:
            return self.seq_len + self.pred_len
        return None


@define(frozen=True, slots=True)
class StatsReport:
    D: int
    col: int  # how many channels were requested
    mean: np.ndarray  # [limit]
    std: np.ndarray
    median: np.ndarray
    mad: np.ndarray
    min_: np.ndarray
    max_: np.ndarray
    q01: np.ndarray
    q25: np.ndarray
    q75: np.ndarray
    q99: np.ndarray


@define(frozen=True, slots=True)
class HealthReport:
    """Summary of train-split distribution issues."""

    D: int
    n_heavy: int
    n_const: int
    n_sparse: int
    heavy_channels: tuple[int, ...]
    const_channels: tuple[int, ...]
    sparse_channels: tuple[int, ...]
    sparse_thr: float


@define(frozen=True, slots=True)
class DriftReport:
    mean_shift_val: float  # normalized: avg |Δμ| / train_std
    mean_shift_test: float
    std_ratio_val: float  # median(σ_split / σ_train)
    std_ratio_test: float
    drift_thr_ratio: float

    @property
    def has_drift(self) -> bool:
        return abs(self.std_ratio_val - 1.0) > self.drift_thr_ratio or abs(self.std_ratio_test - 1.0) > self.drift_thr_ratio


@define(frozen=True, slots=True)
class SampleReport:
    data: np.ndarray  # [row, c_limit]
    feature_names: tuple[str, ...]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compute functions (DataBundle → Report)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_counts(db: "DataBundle") -> CountsReport:
    assert db.train_partition_stats is not None
    assert db.val_partition_stats is not None
    assert db.test_partition_stats is not None
    t, v, s = db.train_partition_stats, db.val_partition_stats, db.test_partition_stats
    return CountsReport(train=int(t.counts), val=int(v.counts), test=int(s.counts))


def compute_stats(db: "DataBundle", *, col: int = 5) -> StatsReport:
    assert db.train_partition_stats is not None
    t = db.train_partition_stats
    D = t.mean.shape[0]
    limit = min(D, col)
    return StatsReport(
        D=D,
        col=col,
        mean=t.mean[:limit],
        std=t.std[:limit],
        median=t.median[:limit],
        mad=t.mad[:limit],
        min_=t.min_[:limit],
        max_=t.max_[:limit],
        q01=t.q01[:limit],
        q25=t.q25[:limit],
        q75=t.q75[:limit],
        q99=t.q99[:limit],
    )


def compute_health(db: "DataBundle", *, sparse_thr=0.3, heavy_tail_thr_ratio=1.5) -> HealthReport:
    """Compute train-split distribution issues."""
    assert db.train_partition_stats is not None
    assert db.series is not None
    assert db.indices is not None
    t = db.train_partition_stats
    D = t.mean.shape[0]
    iqr = t.q75 - t.q25
    denom = np.maximum(iqr, 1e-8)
    right_tail = (t.q99 - t.q95) / denom
    left_tail = (t.q05 - t.q01) / denom
    mask_heavy = (right_tail > heavy_tail_thr_ratio) | (left_tail > heavy_tail_thr_ratio)
    mask_const = t.std < 1e-6
    train_data = db.series[db.indices[0] : db.indices[1]]
    mask_sparse = (train_data == 0).astype(np.float32).mean(axis=0) > sparse_thr
    return HealthReport(
        D=D,
        n_heavy=int(mask_heavy.sum()),
        n_const=int(mask_const.sum()),
        n_sparse=int(mask_sparse.sum()),
        heavy_channels=tuple(int(i) for i in np.flatnonzero(mask_heavy)),
        const_channels=tuple(int(i) for i in np.flatnonzero(mask_const)),
        sparse_channels=tuple(int(i) for i in np.flatnonzero(mask_sparse)),
        sparse_thr=sparse_thr,
    )


def compute_drift(db: "DataBundle", *, drift_thr_ratio: float = 0.3) -> DriftReport:
    assert db.train_partition_stats is not None
    assert db.val_partition_stats is not None
    assert db.test_partition_stats is not None
    t, v, s = db.train_partition_stats, db.val_partition_stats, db.test_partition_stats
    denom = np.maximum(t.std, 1e-8)
    return DriftReport(
        mean_shift_val=float(np.mean(np.abs(v.mean - t.mean) / denom)),
        mean_shift_test=float(np.mean(np.abs(s.mean - t.mean) / denom)),
        std_ratio_val=float(np.quantile(v.std / denom, 0.5, method="linear")),
        std_ratio_test=float(np.quantile(s.std / denom, 0.5, method="linear")),
        drift_thr_ratio=drift_thr_ratio,
    )


def compute_sample(db: "DataBundle", *, col: int = 10, row: int = 5) -> SampleReport:
    assert db.series is not None
    assert db.indices is not None
    assert db.feature_names is not None
    X = db.series
    D = X.shape[1]
    c_limit = min(col, D)
    sample = X[db.indices[0] : db.indices[0] + row, :c_limit]
    names = tuple(db.feature_names[:c_limit])
    return SampleReport(data=sample, feature_names=names)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Renderers (Report → Rich Table)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_counts(r: CountsReport) -> "Table":
    from rich.table import Table
    from rich.text import Text

    tbl = Table(title="Split Sizes", show_edge=False, pad_edge=False)
    tbl.add_column("Split", style="bold")
    tbl.add_column("Count", justify="right")
    tbl.add_column("Fraction", justify="right")
    for name, count, color in [("Train", r.train, "green"), ("Val", r.val, "yellow"), ("Test", r.test, "red")]:
        tbl.add_row(Text(name, style=color), f"{count:,}", f"{count / r.total:.1%}")
    tbl.add_row("Total", f"{r.total:,}", "1.000", style="dim")

    if r.min_required:
        min_split = min(r.train, r.val, r.test)
        if min_split < r.min_required:
            tbl.add_row(
                Text("⚠", style="red bold"),
                f"min split={min_split:,}",
                f"< seq+pred={r.min_required:,}",
            )
    return tbl


def render_stats(r: StatsReport) -> "Table":
    from rich.table import Table

    limit = r.mean.shape[0]
    tbl = Table(title=f"Channel Statistics — Train (D={r.D})", show_edge=False, pad_edge=False)
    tbl.add_column("Ch", justify="right", style="dim")
    for h in ["Mean", "Std", "Median", "MAD", "Min", "1%", "25%", "75%", "99%", "Max"]:
        tbl.add_column(h, justify="right")
    for i in range(limit):
        tbl.add_row(
            str(i),
            f"{float(r.mean[i]):.3f}",
            f"{float(r.std[i]):.3f}",
            f"{float(r.median[i]):.3f}",
            f"{float(r.mad[i]):.3f}",
            f"{float(r.min_[i]):.3f}",
            f"{float(r.q01[i]):.3f}",
            f"{float(r.q25[i]):.3f}",
            f"{float(r.q75[i]):.3f}",
            f"{float(r.q99[i]):.3f}",
            f"{float(r.max_[i]):.3f}",
        )
    if r.D > limit:
        tbl.add_row(f"... +{r.D - limit}", *[""] * 10, style="dim")
    return tbl


def render_health(report: HealthReport, console: "Console | None" = None) -> "Table":
    """Render a `HealthReport` as a Rich table."""
    from rich.table import Table
    from rich.text import Text

    def _ex(channels, n=10):
        if not channels:
            return Text("none", style="green")
        s = str(list(channels[:n]))
        if len(channels) > n:
            s += "..."
        return Text(s, style="red")

    tbl = Table(title="Distribution Health — Train", show_edge=False, pad_edge=False)
    tbl.add_column("Issue")
    tbl.add_column(f"Count / {report.D}", justify="right")
    tbl.add_column("Channels")
    tbl.add_row("Heavy tails", str(report.n_heavy), _ex(report.heavy_channels))
    tbl.add_row("Near constant", str(report.n_const), _ex(report.const_channels))
    tbl.add_row(f"Sparse (>{report.sparse_thr:.0%} zeros)", str(report.n_sparse), _ex(report.sparse_channels))

    if console:
        console.print(tbl)
    return tbl


def render_drift(r: DriftReport) -> "Table":
    from rich.table import Table
    from rich.text import Text

    def _style(ratio):
        return "red bold" if abs(ratio - 1.0) > r.drift_thr_ratio else ""

    status = "✓" if not r.has_drift else "⚠ drift"
    tbl = Table(title=f"Cross-Split Drift (ref: Train)  [{status}]", show_edge=False, pad_edge=False)
    tbl.add_column("Metric")
    tbl.add_column("Val", justify="right")
    tbl.add_column("Test", justify="right")
    tbl.add_row("Mean shift / train std", f"{r.mean_shift_val:.4f}", f"{r.mean_shift_test:.4f}")
    tbl.add_row(
        "Std ratio (median)",
        Text(f"{r.std_ratio_val:.3f}", style=_style(r.std_ratio_val)),
        Text(f"{r.std_ratio_test:.3f}", style=_style(r.std_ratio_test)),
    )
    return tbl


def render_sample(r: SampleReport) -> "Table":
    from rich.table import Table

    rows, cols = r.data.shape
    tbl = Table(title=f"Sample (first {rows} rows × {cols} ch)", show_edge=False, pad_edge=False)
    for name in r.feature_names:
        tbl.add_column(name, justify="right")
    for i in range(rows):
        tbl.add_row(*[f"{float(r.data[i, j]):.4f}" for j in range(cols)])
    return tbl


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Composite render + pipeline stage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def print_diagnostics(
    db: "DataBundle",
    *,
    col: int = 5,
    row: int = 5,
    sparse_thr: float = 0.3,
    heavy_tail_thr_ratio: float = 1.5,
    drift_thr_ratio: float = 0.3,
    print_verbose: bool = False,
) -> None:
    """Compute and render diagnostics for a partitioned bundle."""
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    # Compute (pure data — collectible, serializable)
    counts = compute_counts(db)
    stats = compute_stats(db, col=col)
    health = compute_health(db, sparse_thr=sparse_thr, heavy_tail_thr_ratio=heavy_tail_thr_ratio)
    drift = compute_drift(db, drift_thr_ratio=drift_thr_ratio)
    sample = compute_sample(db, col=col, row=row)

    # Render
    console.print()
    console.print(
        Panel.fit(
            Columns(
                [
                    render_counts(counts),
                    render_health(health),
                    render_drift(drift),
                ],
                padding=(0, 3),
            ),
            title="[bold]Data Diagnostics[/bold]",
            border_style="blue",
        )
    )
    console.print(
        Panel.fit(
            render_stats(stats),
            title="[bold]Statistics[/bold]",
            border_style="blue",
        )
    )
    if print_verbose:
        console.print(render_sample(sample))

    if drift.has_drift:
        console.print("[bold red]⚠  Significant variance drift detected[/bold red]")


def make_stage_print_diagnostics(
    col: int = 10,
    row: int = 5,
    sparse_thr: float = 0.3,
    heavy_tail_thr_ratio: float = 1.5,
    drift_thr_ratio: float = 0.3,
    print_verbose: bool = False,
) -> Callable[[DataBundle], DataBundle]:
    """Return a BundleStep-compatible callable that prints diagnostics.

    Import-free at call time — avoids circular import with pipeline.BundleStep.
    Use: `db | make_stage_print_diagnostics()`
    """

    def _apply(db: "DataBundle") -> "DataBundle":
        print_diagnostics(
            db,
            col=col,
            row=row,
            sparse_thr=sparse_thr,
            heavy_tail_thr_ratio=heavy_tail_thr_ratio,
            drift_thr_ratio=drift_thr_ratio,
            print_verbose=print_verbose,
        )
        logger.info("data partitioned successfully")
        return db

    return _apply
