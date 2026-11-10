"""Smoke test — run every public API path end-to-end in one test function.

This is the runnable companion to ``docs/python_walkthrough.md``.  The
walkthrough explains concepts; this test proves the same code paths still
compose without crashing.

It exercises:
    - build_dgp_bank with string name, TrajectorySpec, and intervention preset
    - inject_noise with unit and train_std sigma scaling
    - multiple sources: Lorenz, Rössler, Chua, GARCH, OU
    - iter_cells with stochastic paired draws (DrawDesign.AUTO → JOINT)
    - build_empirical_dgp_bank with custom feature names
    - build_clean_split_bundle + build_noisy_bundle_for_cell (low-level pipeline)
    - export_dgp_bank_files with manifest round-trip

If this test passes, the public API is wired correctly.  Individual
correctness invariants (determinism, injection identity, CRN, etc.) are
tested in test_titration_edges.py and test_public_contracts.py.
"""

from __future__ import annotations

from msgspec.structs import replace

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from serie import (
    DrawDesign,
    PartitionSpec,
    TitrationPlan,
    TrajectorySpec,
    build_clean_split_bundle,
    build_dgp_bank,
    build_empirical_dgp_bank,
    build_noisy_bundle_for_cell,
    export_dgp_bank_files,
    inject_noise,
    iter_cells,
)


def _line(label: str, **items: object) -> None:
    fields = " ".join(f"{key}={value}" for key, value in items.items())
    print(f"SMOKE {label}: {fields}")


def test_public_api_smoke_flow(capsys) -> None:
    """Executable version of the Python walkthrough's core API path.

    The Markdown walkthrough explains the concepts.  This test proves the same
    public primitives still compose end-to-end.
    """
    bank = build_dgp_bank(TrajectorySpec.lorenz(steps=199), seed=42, n_noise_draws=2)
    split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
    assert bank.clean.shape == (1, 200, 3)
    assert split.train_values.shape == (140, 3)
    _line("basic", source=bank.name, clean=bank.clean.shape, train=split.train_values.shape)

    alternate_noise = inject_noise(
        bank,
        sigma=0.25,
        partition=PartitionSpec.p70_20_10(),
        noise_draw_ordinal=1,
    )
    assert alternate_noise.metadata["trajectory_draw_ordinal"] == 0
    assert alternate_noise.metadata["noise_draw_ordinal"] == 1
    _line("deterministic-replicate", noise_draw=alternate_noise.metadata["noise_draw_ordinal"])

    for spec in (
        TrajectorySpec.rossler(steps=99),
        TrajectorySpec.chua(steps=99),
        TrajectorySpec.garch(steps=99),
    ):
        source_bank = build_dgp_bank(spec, seed=7)
        source_split = inject_noise(source_bank, sigma=0.1)
        assert source_split.train_values.ndim == 2
        _line("source", source=source_bank.name, clean=source_bank.clean.shape)

    shifted = replace(TrajectorySpec.lorenz_param_shock(), steps=199)
    shifted_bank = build_dgp_bank(shifted, seed=42)
    clean = inject_noise(shifted_bank, sigma=0.0)
    noisy = inject_noise(shifted_bank, sigma=0.25)
    assert clean.source_id == "lorenz_param_shock"
    assert noisy.train_values.shape == clean.train_values.shape
    _line("intervention", source=shifted_bank.name, sigma=noisy.metadata["sigma"])

    stochastic_bank = build_dgp_bank(
        TrajectorySpec.ou(steps=99),
        seed=11,
        noise_seed_base=4000,
        n_trajectory_draws=2,
        n_noise_draws=2,
    )
    plan = TitrationPlan(
        noise_levels=(0.0, 0.5),
        n_trajectory_draws=2,
        n_noise_draws=2,
        draw_design=DrawDesign.AUTO,
    )
    cells = tuple(iter_cells(stochastic_bank, plan))
    assert cells
    assert all(cell.trajectory_draw_ordinal == cell.noise_draw_ordinal for cell in cells)
    _line("paired", cells=len(cells), first=cells[0].tag)

    t = np.linspace(0.0, 2.0 * np.pi, 128, dtype=np.float32)
    values = np.stack([np.sin(t), np.cos(t)], axis=1)
    empirical_bank = build_empirical_dgp_bank(
        values,
        name="toy_recording",
        feature_names=("sin", "cos"),
    )
    empirical_split = inject_noise(empirical_bank, sigma=1.0, sigma_scale="train_std")
    assert empirical_bank.clean.shape == (1, 128, 2)
    assert empirical_split.feature_names == ("sin", "cos")
    _line("empirical", source=empirical_bank.name, train=empirical_split.train_values.shape)

    pipeline_bank = build_dgp_bank(TrajectorySpec.lorenz(steps=99), seed=0)
    partition = PartitionSpec.p70_20_10()
    cell = pipeline_bank.cell(noise_level=0.2)
    clean_db = build_clean_split_bundle(
        pipeline_bank,
        trajectory_draw_ordinal=0,
        partition_spec=partition,
        print_diagnostics=False,
    )
    noisy_db = build_noisy_bundle_for_cell(pipeline_bank, clean_db, cell)
    assert clean_db.indices is not None
    assert noisy_db.series is not None
    _line("pipeline", indices=clean_db.indices, series=noisy_db.series.shape)

    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        manifest = export_dgp_bank_files(
            pipeline_bank,
            out,
            sigmas=(0.1,),
            array_format="npy",
            overwrite=True,
        )
        assert (out / "clean.npy").exists()
        assert (out / "noise_draws.npy").exists()
        assert manifest["schema"] == "serie.dgp_bank_file_export.v1"
        _line("export", noisy=len(manifest["noisy"]))

    print("SMOKE OK")
    out = capsys.readouterr().out
    assert "SMOKE basic:" in out
    assert "SMOKE pipeline:" in out
    assert "SMOKE OK" in out
