# Python Walkthrough

This page explains the Python API behind the CLI.  It is the human-facing
walkthrough: use it when you want the concepts and snippets.  The executable
smoke version lives in `tests/test_example.py`, so CI proves these public API
paths still compose.

For the visual first impression, use the CLI demo instead:

```bash
serie demo lorenz --out-html tmp/lorenz_demo.html
serie demo rossler --out-html tmp/rossler_demo.html
serie demo chua --out-html tmp/chua_demo.html
```

For the main data workflow, use `generate`:

```bash
serie generate lorenz --sigma 0.1
serie generate lorenz --shock param --sigma 0.1
```

The Python API explains what those commands compose internally.

## The Core Composition

SERIE has one main rhythm:

```text
source/spec -> build_dgp_bank(...) -> inject_noise(...)
```

In Python:

```python
from serie import PartitionSpec, TrajectorySpec, build_dgp_bank, inject_noise

source = TrajectorySpec.lorenz(steps=1999)
partition = PartitionSpec.p70_20_10()

bank = build_dgp_bank(source, seed=42, n_noise_draws=2)
split = inject_noise(bank, sigma=0.1, partition=partition)

print(bank.clean.shape)
print(bank.noise_draws.shape)
print(split.train_values.shape)
```

`build_dgp_bank(...)` is the expensive step.  It creates clean trajectory draws
and freezes observation-noise draws.  `inject_noise(...)` is the cheap step.  It
reuses the bank and returns one train/validation/test split.

## Why Build Once?

Every sigma from the same bank uses the same frozen noise draw:

```python
for sigma in (0.0, 0.1, 0.5):
    split = inject_noise(bank, sigma=sigma, partition=partition)
    print(sigma, split.train_values.shape)
```

That is common random numbers (CRN): when sigma changes, the random draw does
not.  This makes the dose-response comparison cleaner because the difference is
caused by the noise level, not by a new random sample.

## Deterministic Systems: One Trajectory, Many Noise Draws

For deterministic systems like Lorenz, the clean future is fixed once the
initial condition and parameters are fixed.  Multiple samples usually mean
multiple observation-noise draws around that one clean path:

```python
bank = build_dgp_bank(
    TrajectorySpec.lorenz(steps=1999),
    seed=42,
    noise_seed_base=3000,
    n_trajectory_draws=1,
    n_noise_draws=3,
)

clean = inject_noise(bank, sigma=0.0, partition=partition, noise_draw_ordinal=0)
noisy = inject_noise(bank, sigma=0.25, partition=partition, noise_draw_ordinal=1)

print(clean.metadata["noise_draw_ordinal"])
print(noisy.metadata["noise_draw_ordinal"])
```

The two calls share the same deterministic trajectory draw, but use different
frozen observation-noise draws.  This is useful for repeated measurement-noise
replicates around one known clean system.

## Multiple Source Families

SERIE is not Lorenz-only.  The same primitive composition works for attractors,
stochastic processes, volatility systems, and your own recordings:

```python
sources = [
    TrajectorySpec.lorenz(steps=999),
    TrajectorySpec.rossler(steps=999),
    TrajectorySpec.chua(steps=999),
    TrajectorySpec.ou(steps=799),
    TrajectorySpec.garch(steps=799),
]

for source in sources:
    bank = build_dgp_bank(source, seed=7)
    split = inject_noise(bank, sigma=0.1)
    print(bank.name, bank.clean.shape, split.train_values.shape)
```

## Intervention And Noise Are Separate Axes

An intervention changes the system.  Observation noise changes the measurement.
You can test either one alone, or both together:

```python
from attrs import evolve

clean_spec = TrajectorySpec.lorenz(steps=1999)
shifted_spec = evolve(TrajectorySpec.lorenz_param_shock(), steps=1999)

clean_bank = build_dgp_bank(clean_spec, seed=42)
shifted_bank = build_dgp_bank(shifted_spec, seed=42)

regular_clean = inject_noise(clean_bank, sigma=0.0)
regular_noisy = inject_noise(clean_bank, sigma=0.25)
shifted_clean = inject_noise(shifted_bank, sigma=0.0)
shifted_noisy = inject_noise(shifted_bank, sigma=0.25)
```

CLI equivalents:

```bash
serie generate lorenz --sigma 0.25
serie generate lorenz --shock param --sigma 0.25
```

## Stochastic Draw Pairing

For deterministic systems, one clean trajectory can be paired with many
observation-noise draws.  For stochastic systems, each trajectory draw is a
different possible future, so trajectory draw `r` should pair with noise draw
`r` in the main predictive-distribution design.

```python
from serie import DrawDesign, TitrationPlan, iter_cells

bank = build_dgp_bank(
    TrajectorySpec.ou(steps=999),
    seed=11,
    noise_seed_base=4000,
    n_trajectory_draws=3,
    n_noise_draws=3,
)

plan = TitrationPlan(
    noise_levels=(0.0, 0.5),
    n_trajectory_draws=3,
    n_noise_draws=3,
    draw_design=DrawDesign.AUTO,
)

for cell in iter_cells(bank, plan):
    print(cell.noise_level, cell.trajectory_draw_ordinal, cell.noise_draw_ordinal)
```

`DrawDesign.AUTO` uses paired diagonal draws for stochastic sources and
noise-only expansion for deterministic sources.

## Low-Level Pipeline Composition

Most users should call `inject_noise(...)`.  The lower-level API exists for
trainer integration and for readers who want to see the stage order explicitly.

The important ordering is:

```text
load clean tensor -> split by fraction -> compute train statistics -> inject noise
```

That order matters for `sigma_scale="train_std"`: the training-set standard
deviation must be computed from the clean training rows before noise can be
scaled by it.  Computing it over the whole series would leak validation/test
information.

In code:

```python
from serie import build_clean_split_bundle, build_noisy_bundle_for_cell

bank = build_dgp_bank(TrajectorySpec.lorenz(steps=999), seed=0)
partition = PartitionSpec.p70_20_10()
cell = bank.cell(noise_level=0.2)

clean_db = build_clean_split_bundle(
    bank,
    trajectory_draw_ordinal=0,
    partition_spec=partition,
    print_diagnostics=False,
)

noisy_db = build_noisy_bundle_for_cell(bank, clean_db, cell)

print(clean_db.indices)
print(noisy_db.series.shape)
```

If you read `pipeline.py`, this is implemented with a pipeable `DataBundle`:

```text
DataBundle()
  | StageLoadTensor(...)
  | StageSplitByFraction(...)
  | StageComputeSplitStats(...)
  | StageInjectNoise(...)
```

That pipe syntax is an internal staging pattern.  The stable public workflow is
still `build_dgp_bank(...)` followed by `inject_noise(...)`.

## Bring Your Own Clean Recording

If you already have a clean `[T, D]` NumPy array, wrap it as an empirical DGP
bank:

```python
import numpy as np
from serie import build_empirical_dgp_bank

t = np.linspace(0.0, 8.0 * np.pi, 512, dtype=np.float32)
values = np.stack(
    [
        np.sin(t),
        0.2 * np.cos(3.0 * t),
        np.sin(0.5 * t) + 0.1 * np.cos(7.0 * t),
    ],
    axis=1,
)

bank = build_empirical_dgp_bank(
    values,
    name="toy_wearable_gait",
    feature_names=("hip", "ankle", "load"),
)

split = inject_noise(bank, sigma=1.0, sigma_scale="train_std")
```

`sigma_scale="train_std"` means sigma is measured per channel in units of that
channel's clean training-set standard deviation.

## Executable Smoke Coverage

The snippets above are documentation.  The executable version lives in
`tests/test_example.py` and is run by pytest.  It exercises the same main API
paths without creating a separate public example module:

```bash
python -m pytest tests/test_example.py -q
```
