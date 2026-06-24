"""One-file SERIE example.

Run from the study checkout:

    python serie/example.py

Run after installing the package:

    python -m serie.example

This file is intentionally small. The installed CLI walkthrough is still the
full artifact-producing path; this example is for reading the Python API in one
place.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    # Let `python serie/example.py` work from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serie import (  # noqa: E402
    DrawDesign,
    PartitionSpec,
    TitrationPlan,
    TrajectorySpec,
    build_dgp_bank,
    inject_noise,
    iter_cells,
)


def describe_split(label, split) -> None:
    """Print the parts a model usually consumes."""
    print(f"\n{label}")
    print(f"  source: {split.source_id}")
    print(f"  train/val/test: {split.train_values.shape} / {split.val_values.shape} / {split.test_values.shape}")
    print(f"  sigma: {split.metadata['sigma']} ({split.metadata['sigma_scale']})")
    print(
        "  draw ordinals: "
        f"trajectory={split.metadata['trajectory_draw_ordinal']}, "
        f"noise={split.metadata['noise_draw_ordinal']}"
    )
    print(
        "  draw seeds: "
        f"trajectory={split.metadata['trajectory_draw_seed']}, "
        f"noise={split.metadata['noise_draw_seed']}"
    )


def deterministic_lorenz_example() -> None:
    """Deterministic system: one clean trajectory, many observation-noise draws."""
    print("== deterministic Lorenz ==")
    spec = TrajectorySpec.lorenz(steps=1999)
    bank = build_dgp_bank(
        spec,
        seed=42,
        noise_seed_base=3000,
        n_trajectory_draws=1,
        n_noise_draws=3,
    )
    partition = PartitionSpec.p70_20_10()

    clean = inject_noise(bank, sigma=0.0, partition=partition, noise_draw_ordinal=0)
    noisy = inject_noise(bank, sigma=0.25, partition=partition, noise_draw_ordinal=1)

    describe_split("clean split", clean)
    describe_split("same trajectory, different frozen observation-noise draw", noisy)


def stochastic_ou_example() -> None:
    """Stochastic system: trajectory and observation-noise draws are paired."""
    print("\n== stochastic OU ==")
    spec = TrajectorySpec.ou(steps=999)
    bank = build_dgp_bank(
        spec,
        seed=11,
        noise_seed_base=4000,
        n_trajectory_draws=3,
        n_noise_draws=3,
    )
    partition = PartitionSpec.p70_20_10()
    plan = TitrationPlan(
        noise_levels=(0.0, 0.5),
        n_trajectory_draws=3,
        n_noise_draws=3,
        draw_design=DrawDesign.AUTO,
    )

    print("cells selected by AUTO for a stochastic source:")
    for cell in iter_cells(bank, plan):
        print(
            "  "
            f"sigma={cell.noise_level:<3} "
            f"trajectory_ordinal={cell.trajectory_draw_ordinal} "
            f"noise_ordinal={cell.noise_draw_ordinal}"
        )

    cell = next(iter_cells(bank, plan))
    split = inject_noise(
        bank,
        sigma=cell.noise_level,
        partition=partition,
        trajectory_draw_ordinal=cell.trajectory_draw_ordinal,
        noise_draw_ordinal=cell.noise_draw_ordinal,
    )
    describe_split("first paired stochastic cell", split)


def main() -> None:
    deterministic_lorenz_example()
    stochastic_ou_example()


if __name__ == "__main__":
    main()
