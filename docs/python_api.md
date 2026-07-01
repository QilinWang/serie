[← Back to the README](../README.md)

## Python API

The CLI is the easiest place to start, but the Python API is the programmable
surface for notebooks, experiments, and trainer integrations.

For a plain Python teaching walkthrough, read
[`docs/python_walkthrough.md`](https://github.com/QilinWang/serie/blob/main/docs/python_walkthrough.md)
This does not replace the visual `serie demo`; it explains the public primitive
composition behind `serie generate`, several source families, intervention/noise
as separate axes, stochastic draw pairing, and wrapping your own clean array.

Most users need this import line:

```python
from serie import PartitionSpec, ShockEvent, ShockSchedule, TrajectorySpec, build_dgp_bank, build_empirical_dgp_bank, inject_noise
```

### Build Once, Inject Noise Many Times

Build one DGP bank, then reuse it for every noise dose. Reusing the same bank
applies common random numbers (CRN): every sigma sees the
same noise realization, so differences across `sigma` are not confounded by a
fresh random draw.

```python
from serie import PartitionSpec, build_dgp_bank, inject_noise

bank = build_dgp_bank("lorenz", seed=42)

split = inject_noise(
    bank,
    sigma=0.1,
    partition=PartitionSpec.p70_20_10(),
)

print(split.train_values.shape)
print(split.train_stats.std)
```

### Clean Recorded Data

For clean recorded data, wrap a `[T, D]` NumPy array directly:

```python
from serie import build_empirical_dgp_bank, inject_noise

bank = build_empirical_dgp_bank(values, name="wearable_gait")
split = inject_noise(bank, sigma=1.0, sigma_scale="train_std")
```

`sigma_scale="train_std"` means each channel receives noise scaled
by its clean train-split standard deviation. This is the public form of the
signal-relative dosing used in the papers.

### Noise Laws

Noise is not limited to Gaussian draws:

```python
bank = build_dgp_bank("lorenz", seed=42, noise_kind="laplace")
```

`noise_kind` accepts `"gaussian"`, `"laplace"`, or a callable
`(rng, shape) -> array`. The built-in laws are unit-variance by convention, so
`sigma` reads as the injected noise standard deviation and different laws are
matched by noise power. Custom laws may use any known physical scale; then
`sigma` is a raw multiplier and the draw distribution owns the scale. The hard
requirement is that the injected law is exactly known and reported.

### Synthetic Parameters

Use `TrajectorySpec` when you want explicit synthetic parameters:

```python
from serie import TrajectorySpec, build_dgp_bank

spec = TrajectorySpec.lorenz(steps=50_000)
bank = build_dgp_bank(spec, seed=42)
```

### Interventions

An intervention changes the clean source process before noise is
added. Presets such as `TrajectorySpec.lorenz_param_shock()` are the shortest
entry point:

```python
from serie import TrajectorySpec, build_dgp_bank

source = TrajectorySpec.lorenz_param_shock()
bank = build_dgp_bank(source, seed=42)
```

For a custom intervention, build a `ShockEvent` and attach it to the source:

```python
from attrs import evolve
from serie import ShockEvent, ShockSchedule, TrajectorySpec, build_dgp_bank

base = TrajectorySpec.lorenz()
source = evolve(
    base,
    shocks=ShockSchedule.single(
        ShockEvent.state_perturbation(state_idx=0, eps=0.9, t_start=0.35)
    ),
)
bank = build_dgp_bank(source, seed=42)
```

Use `ShockSchedule.multiple(event_a, event_b, ...)` for multiple
time-ordered parameter/state interventions inside one clean trajectory:

```python
source = evolve(
    base,
    shocks=ShockSchedule.multiple(
        ShockEvent.params_stepped_transition(
            params_after=evolve(base.params, rho=32.0),
            t_start=0.30,
            t_end=0.45,
            n_inserts=4,
        ),
        ShockEvent.state_perturbation(state_idx=0, eps=0.9, t_start=0.60),
    ),
)
```

SERIE requires events to be declared in time order and applies each event when
the integration reaches its declared fraction of the usable trajectory.
Trajectory switch events remain single-event because they cut between
independently generated trajectories instead of perturbing one integration in
place.

### Seed Provenance

The seed you pass to `build_dgp_bank(...)` is root entropy. SERIE uses NumPy
`SeedSequence(root).spawn(n)` to create independent child streams for
trajectory and noise draws. That means the seed a user types is not
necessarily the actual seed used by one draw.

```python
bank = build_dgp_bank("ou", seed=42, noise_seed_base=1000, n_trajectory_draws=3, n_noise_draws=3)
```

Interpret the fields this way:

- `bank.trajectory_seed_base == 42`: the root you passed as `seed`.
- `bank.trajectory_draw_seeds`: child seeds generated for trajectory draws.
- `bank.noise_seed_base == 1000`: the root you passed as
  `noise_seed_base`.
- `bank.noise_draw_seeds`: child seeds generated for observation-noise draws.

For one sweep cell, `trajectory_draw_ordinal` and `noise_draw_ordinal` are
spawn ordinals. They also select the array slot in the materialized bank. For
example, `noise_draw_ordinal=1` means "use the second spawned noise draw,"
materialized as `bank.noise_draws[1]`. The generated child seed for that draw is
`bank.noise_draw_seeds[1]`, also recorded on the cell as
`cell.noise_draw_seed`.

The reproducible recipe is `(seed_base, draw_ordinal)`. The child seed is a
derived fingerprint of that recipe, useful for logs and cross-checks, but the
ordinal is not disposable metadata.

### Draw Designs

SERIE separates two kinds of randomness:

- trajectory randomness: the future trajectory itself, present in stochastic
  systems such as OU, double-well, SLDS, seasonal AR, and GARCH;
- noise randomness: the sensor/noise draw injected after the clean source
  exists.

`TitrationPlan(draw_design="auto")` is the default and chooses the
statistically clean design from the source:

- deterministic or recorded source: `noise_only`, one clean trajectory path
  crossed with `n_noise_draws` noise draws;
- stochastic source: `joint`, one joint future per draw, pairing
  trajectory draw `r` with noise draw `r`.

That paired stochastic design is the main benchmark path: every ensemble member
is one draw from the full future law, and the same paired draws are reused across
all sigma levels. Two diagnostic designs are explicit:

- `trajectory_fixed`: one stochastic trajectory path with many noise
  draws; useful for measurement-noise-only diagnostics, not the full SDE law.
- `cross_product`: the full trajectory × noise grid; useful when that grid is
  the experiment, but not the default unit of replication.

### Stable Import Surface

These names are intended as public imports:

```python
from serie import (
    DataBundle,
    KnownCleanConfig,
    KnownDataID,
    KnownLoadMode,
    KnownMeta,
    PartitionSpec,
    DrawDesign,
    ShockEvent,
    ShockSchedule,
    ShockType,
    SplitTimeSeriesData,
    TimeSeriesData,
    TitrationCell,
    TitrationPlan,
    DGPBank,
    TrajectorySpec,
    build_clean_split_bundle,
    build_noisy_bundle_for_cell,
    export_dgp_bank_files,
    iter_cells,
    load_data,
    build_dgp_bank,
    build_empirical_dgp_bank,
    inject_noise,
)
```

Most users should stay with `build_dgp_bank(...)`, `build_empirical_dgp_bank(...)`, and
`inject_noise(...)`. The bundle builders and `iter_cells(...)` are low-level
integration hooks for trainer or sweep code.

