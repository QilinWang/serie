"""Retired tests for the retired serie artifact-plots candidate.

Frozen beside serie/retired/artifact_plots_split_ipc_candidate.py, which holds
the `serie.artifact_plots` module these tests import. The module left the
active package before its tests did; the orphaned test file broke collection
of serie/tests/. Renamed off the test_* pattern so pytest never collects it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from serie.artifact_plots import (
    ArtifactStream,
    EigenPanel,
    diagnostics,
    load_artifact_sequence,
    plot_artifact_saves,
    seq_plot,
)


def _write_stream(path: Path, values: np.ndarray) -> None:
    rows: list[dict[str, float | int]] = []
    for t_idx, row in enumerate(values):
        rows.append(
            {
                "sample_idx": 0,
                "timestep_idx": t_idx,
                "x": float(row[0]),
                "y": float(row[1]),
                "z": float(row[2]),
            }
        )
    pl.DataFrame(rows).write_ipc(path)


def _write_eigenvalue(path: Path, summaries: dict[str, np.ndarray]) -> None:
    frames: list[pl.DataFrame] = []
    for summary, values in summaries.items():
        rows: list[dict[str, float | int | str]] = []
        for t_idx, row in enumerate(values):
            rows.append(
                {
                    "eigen_summary": summary,
                    "sample_idx": 0,
                    "timestep_idx": t_idx,
                    "x": float(row[0]),
                    "y": float(row[1]),
                    "z": float(row[2]),
                }
            )
        frames.append(pl.DataFrame(rows))
    pl.concat(frames, how="vertical").write_ipc(path)


def _write_artifact_folder(tmp_path: Path, *, with_eigen: bool) -> Path:
    run_dir = tmp_path / ("with_eigen" if with_eigen else "plain")
    run_dir.mkdir()
    t = np.linspace(0.0, 1.0, 8)
    truth = np.column_stack([t, np.sin(t), np.cos(t)])
    pred = truth + 0.05
    _write_stream(run_dir / "pred.ipc", pred)
    _write_stream(run_dir / "truth.ipc", truth)
    (run_dir / "feature_names.json").write_text(
        json.dumps(["x", "y", "z"]) + "\n",
        encoding="utf-8",
    )
    if with_eigen:
        _write_eigenvalue(
            run_dir / "eigenvalue.ipc",
            {
                "sum": np.abs(truth) + 0.1,
                "max": np.abs(pred) + 0.2,
                "logdet": truth - 0.5,
            },
        )
    return run_dir


def test_load_artifact_sequence_reads_split_ipc_and_eigen_streams(tmp_path: Path) -> None:
    run_dir = _write_artifact_folder(tmp_path, with_eigen=True)

    artifacts = load_artifact_sequence(run_dir)

    assert artifacts.feature_names == ("x", "y", "z")
    assert ArtifactStream.PRED.value in artifacts.kinds_present
    assert ArtifactStream.TRUE.value in artifacts.kinds_present
    assert ArtifactStream.FERN_MAX.value in artifacts.kinds_present
    assert artifacts.has_eigen_streams


def test_specs_use_eigen_presence_not_model_branch(tmp_path: Path) -> None:
    with_eigen = load_artifact_sequence(_write_artifact_folder(tmp_path, with_eigen=True))
    plain = load_artifact_sequence(_write_artifact_folder(tmp_path, with_eigen=False))

    eigen_diag = diagnostics(with_eigen)
    plain_diag = diagnostics(plain)
    eigen_seq = seq_plot(with_eigen)
    plain_seq = seq_plot(plain)

    assert sum(isinstance(panel, EigenPanel) for panel in eigen_diag.panels) == 3
    assert sum(isinstance(panel, EigenPanel) for panel in plain_diag.panels) == 0
    assert eigen_seq.alpha_kind == ArtifactStream.FERN_SUM.value
    assert eigen_seq.width_kind == ArtifactStream.FERN_MAX.value
    assert plain_seq.alpha_kind is None
    assert plain_seq.width_kind is None


def test_plot_artifact_saves_writes_diagnostics_file(tmp_path: Path) -> None:
    run_dir = _write_artifact_folder(tmp_path, with_eigen=True)
    out = tmp_path / "diagnostics.png"

    written = plot_artifact_saves(run_dir, out=out, kind="diagnostics")

    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0
