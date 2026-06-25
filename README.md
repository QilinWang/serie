# SERIE

[![CI](https://github.com/QilinWang/serie/actions/workflows/ci.yml/badge.svg)](https://github.com/QilinWang/serie/actions/workflows/ci.yml)
[![Pages Demo](https://github.com/QilinWang/serie/actions/workflows/pages-demo.yml/badge.svg)](https://github.com/QilinWang/serie/actions/workflows/pages-demo.yml)
[![PyPI](https://img.shields.io/pypi/v/serie.svg)](https://pypi.org/project/serie/)
[![Downloads](https://static.pepy.tech/badge/serie/month)](https://pepy.tech/project/serie)

[中文说明](README.zh-CN.md)

**Stress-test a time-series forecaster before you ship it — against the regime shifts and measurement noise *you* know it will face, with ground truth known exactly.**


You've trained a forecasting model and the validation numbers look good. Now the
real question: **how do you prove it's safe to put into production?** Before a
bridge carries traffic or a chip leaves the fab, it is stress-tested against
conditions its engineers chose on purpose. A model about to drive trades,
dispatch power, or flag a failing machine deserves the same — a factory seal
that says *tested, and here is exactly what it can take.*

The usual answer is to run the model on a standard benchmark and report the
error. But that is a grade on one slice of history, and it stays silent on the
two things that actually break forecasters in the field. The first is when the
world **shifts** under the model — a market regime turns, a machine's dynamics
drift, a shock lands. The second is when the **signal degrades** — sensors get
noisier, readings go missing, sampling coarsens. On real-world data you cannot
even ask these questions cleanly: you do not control when the regime changes and
you do not know the true process, so you can never separate "the model failed"
from "the data was ambiguous." In the lab there is only so much you can do — and
right now the field is not even doing that much.

SERIE is that lab — and the crucial part is that *you* design the stresses, from
what you know about your own system. You choose which regime shift to inject and
exactly when; you supply the noise law your channel actually exhibits, not a
textbook default; and because the underlying system is synthetic, its ground
truth is known *exactly*. So the conditions you test under aren't someone else's
idea of failure — they are the ones your forecaster will really meet in your
domain — and what comes out is a falsifiable, repeatable seal rather than a
leaderboard number. That is the step beyond benchmarking: not "what score did it
get," but "what, precisely, can it withstand under the conditions *I* know
matter — and can anyone re-verify that?"

## What it lets you do

Two of these stresses are yours to author — and the one who authors them can be a
domain specialist, or a model trained to know that domain's failure modes:

- **Design the regime change you actually fear.** SERIE doesn't hand you a fixed
  menu of shocks — you specify the intervention yourself: which parameter drifts
  and by how much, whether the state takes a sudden perturbation or the trajectory
  switches into a new regime, and the exact moment it strikes. That is where your domain knowledge
  enters. A quant encodes the volatility spike or correlation breakdown that has
  burned the desk before; a controls engineer encodes the actuator fault; a grid
  operator encodes the load shock that follows a heatwave. The stress is no longer
  generic — it is the specific failure your system will meet in deployment — and
  because you declared exactly what changed and when, "the model breaks under
  *this* shift" emerges as a measured, falsifiable result instead of a standing
  worry.

- **Dial up your noise, not a textbook's.** Real measurement noise is rarely clean
  Gaussian: it is heavy-tailed on outlier-prone sensors, bursty on a lossy link,
  skewed, sometimes carried in real physical units. SERIE lets you supply the law
  you know your channel actually has — any `(rng, shape) -> array` you can write —
  and then turn it up in measured doses, from pristine to severe, tracing where the
  model's forecasts come apart. What you get is the model's tolerance to the
  degradation that will *really* happen, not to an idealization — and that
  degradation curve is the operating envelope you stamp on the seal: rated to here,
  no further.

- **Test the model, not the data.** Every synthetic source — Lorenz,
  Ornstein–Uhlenbeck, GARCH, and more — has dynamics whose statistics are known
  in closed form. So when a forecast is wrong, the error is unambiguously the
  model's: there is no half-corrupted real series to argue about. You are
  certifying against a known spec, which is the only way a seal means anything.

- **Compare models and doses fairly.** When you pit two models against each other,
  or one model at two noise levels, the difference has to be real — not an
  artifact of which random numbers you happened to draw. SERIE reuses the *same*
  frozen noise across every sigma (a technique called common random numbers), so
  two settings differ only by the thing you changed. That is what lets you run an
  actual significance test and trust the verdict, instead of re-running and
  hoping.

- **Check whether the confidence is honest.** A forecast is a number *and* a
  stated confidence, and a model that is confidently wrong is the dangerous kind —
  because you size a position or dispatch a crew based on that confidence. To
  audit it you must know the true distribution of the noise you injected, which
  SERIE does exactly. Then you can ask the questions that matter: does the model's
  90% interval actually contain the truth 90% of the time (**coverage**); is its
  predicted distribution the right shape (the **probability integral transform**,
  PIT); how surprised is it by reality on average (**negative log-likelihood**,
  NLL). Honest uncertainty is half of what "ready to ship" means, and SERIE is
  built to measure it.

- **Issue a seal anyone can re-verify.** A certification no one can reproduce is
  worthless. The same seed produces byte-identical data on any machine, and every
  dataset is stamped with a protocol version — so "tested on SERIE, Lorenz,
  sigma 0.3" denotes one exact thing that you, a teammate, or a reviewer can
  regenerate and re-check months later.

- **Bring your own system — it is not synthetic-only.** SERIE is not limited to
  simulated data. Hand it a clean recording — a wearable gait trace, sensor logs,
  your own production series as a `[T, D]` array — and it stress-tests that the
  same way: perturb it with controlled noise, split it, certify against it. The lab
  comes to your domain, not the other way around.

## How it works — a five-minute tour

The whole library is three nouns and two verbs. Hold this picture and the rest is detail:

    source  ──build_dgp_bank()──▶  DGP bank  ──inject_noise()──▶  split
   (what system)             (built once)              (one dosed dataset)

- A **source** is the system you want to test — a name like `"lorenz"`, a
  `TrajectorySpec` you configured, or a real recording you hand in.
- **`build_dgp_bank(source)`** does the expensive work *once*: it runs the clean
  trajectory and draws a frozen reservoir of noise. The result is a
  `DGPBank` — a bank of clean trajectory draws plus reusable noise
  draws, ready to be dosed.
- **`inject_noise(bank, sigma=...)`** is cheap and repeatable: it adds `sigma`
  worth of that frozen noise to the clean signal and hands back a ready-to-use
  train / validation / test split.

So the rhythm is always **build_dgp_bank once, inject_noise many times**:

```python
from serie import PartitionSpec, build_dgp_bank, inject_noise

# 1. Build a DGP bank once — the expensive call. It runs the clean Lorenz
#    trajectory and freezes a reservoir of noise draws.
bank = build_dgp_bank("lorenz", seed=42)

# 2. Inject a chosen noise dose. Cheap — call it as often as you like.
split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())

# 3. `split` is a partitioned dataset; stats come from the TRAIN split only.
print(split.train_values.shape)   # (T_train, D) — feed this to your model
print(split.train_stats.std)      # per-channel std, train only (no leakage)
```

### Why "build_dgp_bank once"?

The split between `build_dgp_bank` and `inject_noise` isn't only about speed — it's what makes
a comparison trustworthy. Because the noise reservoir is *frozen* inside
`bank`, every dose drawn from the same `bank` reuses the **same**
underlying random draw:

```python
clean = inject_noise(bank, sigma=0.0)   # the pristine signal
mild  = inject_noise(bank, sigma=0.1)   # same signal + 0.1 × (the same noise)
harsh = inject_noise(bank, sigma=0.5)   # same signal + 0.5 × (the same noise)
```

The *only* thing that changed between `mild` and `harsh` is the dose — not which
random numbers came up. That's the experimental-control trick called **common
random numbers**: a difference in your model's error across these doses is caused
by the noise *level*, not by luck. Re-running `build_dgp_bank` per dose would quietly
break that guarantee.

(Want independent repeats on top? For deterministic sources, that means one
clean trajectory plus many noise draws. For stochastic sources, it
means paired trajectory and noise draws: `(trajectory_draw_ordinal=0,
noise_draw_ordinal=0)`, `(1, 1)`, and so on. That pairing is the unit of replication for
predictive-distribution tests.)

### Finding a breaking point

Because dosing is cheap, the natural move is to sweep — turn the noise up step by
step and watch where your model's forecasts fall apart:

```python
bank = build_dgp_bank("lorenz", seed=42)
for sigma in [0.0, 0.1, 0.25, 0.5, 1.0]:
    split = inject_noise(bank, sigma=sigma)
    error = evaluate(your_model, split)        # your own eval loop
    print(f"sigma={sigma}: error={error:.4f}")
```

The curve of error vs `sigma` is the model's tolerance — the operating envelope
from above, made concrete.

### Bringing your own data

All of this works on a real recording too — hand `build_empirical_dgp_bank` a clean
`[T, D]` array instead of a system name:

```python
from serie import build_empirical_dgp_bank, inject_noise

bank = build_empirical_dgp_bank(my_array, name="wearable_gait")
split = inject_noise(bank, sigma=1.0, sigma_scale="train_std")
```

(`sigma_scale="train_std"` scales noise per channel by that channel's
training-split standard deviation — useful when channels have different physical
units.)

## Install

After publication:

```bash
python -m pip install serie
```

From this repository:

```bash
cd serie
python -m pip install -e ".[dev,diagnostics]"
```

With `uv`, a checkout can run the CLI without syncing a project environment.
The `--no-project --with-editable .` form builds a temporary/cached execution
environment from the local checkout and avoids touching a shared
`UV_PROJECT_ENVIRONMENT`:

```bash
cd serie
uv run --no-project --with-editable . serie walkthrough --out tmp/walkthrough --overwrite --format json
```

After installing from PyPI, use the shorter installed command:

```bash
serie walkthrough --out tmp/walkthrough --overwrite --format json
```

Runtime dependencies are `attrs`, `cappa`, `matplotlib`, `numpy`, and `polars`.
`cappa` provides the CLI, `matplotlib` renders source figures, and `polars` is
part of the known-recording loader path. `rich` is optional and only used for
diagnostic table rendering.

## Command Line API

The package includes a small Cappa CLI. This is the installed user surface:
after `pip install serie` or `uv tool install serie`, run `serie ...` directly.
The CLI does not assume you have this repository, a Justfile, or `uv`.

A command has this shape:

```bash
serie <command> <positional-arguments> --option value --flag
```

For example:

```bash
serie generate lorenz --shock param --sigma 0.1 --format json
```

Read that as:

- `serie`: run the installed SERIE command.
- `generate`: generate one SERIE dataset cell.
- `lorenz`: the source name.
- `--shock param`: use the built-in parameter-shift intervention preset.
- `--sigma 0.1`: set the observation-noise dose.
- `--format json`: print machine-readable output for humans, scripts, and
  agents.

Options beginning with `--` are named options. Some options take values
(`--sigma 0.1`); some are switches (`--overwrite`).

To print the command menu in a form that is easy for humans and agents to scan:

```bash
serie commands
serie commands --format json
```

For detailed options on one command, use normal CLI help:

```bash
serie generate --help
```

### First Run: See It Work

Start with three commands. They show the available sources, create an
interactive visual demo, and generate one train/validation/test split:

```bash
serie list
serie demo lorenz --out tmp/lorenz_demo.html
serie generate lorenz --sigma 0.1 --format json
```

Read that as:

1. `serie list`: see which built-in sources are available.
2. `serie demo lorenz ...`: write one offline HTML file with a draggable Lorenz
   attractor and a live noise slider.
3. `serie generate lorenz ...`: generate one dataset cell and return
   train/validation/test split metadata.

Once those work, use the command-specific sections below.

### Full Verification Walkthrough

For a fuller install audit, run these commands in order:

```bash
serie commands --format json
serie list --format json
serie data-dir --format json
serie inspect ou --format json
serie generate ou --sigma 0.1 --partition p70_20_10 --format json
serie generate lorenz --shock param --sigma 0.1 --out tmp/lorenz_cell --array-format all --plot both --overwrite --format json
serie walkthrough --out tmp/walkthrough --array-format all --overwrite --format json
serie demo lorenz --out tmp/lorenz_demo.html --overwrite
```

This answers eight questions:

1. What commands does SERIE expose?
2. What source names does SERIE know?
3. Where should external dataset files live?
4. Can SERIE generate one clean source and report its metadata?
5. Can SERIE generate one cell and split the series?
6. Can one command combine intervention, noise, arrays, and figures?
7. Can SERIE write the standard audit walkthrough?
8. Can SERIE write one offline interactive attractor demo?

### Command Reference

| Command | What it does |
|---------|-------------|
| `serie commands` | List all CLI commands with examples |
| `serie list` | List available source names (synthetic + known) |
| `serie data-dir` | Show where external dataset files should live |
| `serie inspect <source>` | Build one source and print its metadata |
| `serie generate <source>` | The main workflow: source + intervention + noise + split + optional files |
| `serie demo <source>` | Write a self-contained interactive HTML with 3D attractor + noise slider |
| `serie walkthrough` | Full end-to-end audit: arrays, manifests, and figures for two sources |

Add `--format json` to any command for machine-readable output. Use
`serie <command> --help` for full options.

**`generate`** is the command most users need. Examples:

```bash
serie generate lorenz --sigma 0.1 --format json
serie generate lorenz --shock param --sigma 0.1 --format json
serie generate lorenz --sigma 0.1 --out tmp/cell --array-format all --plot both --overwrite --format json
```

Key options: `--shock none|param|state|switch` for built-in interventions,
`--sigma` for noise dose, `--sigma-scale unit|train_std` for scaling mode,
`--partition` for split convention, `--out` for artifact directory,
`--array-format` and `--plot` for file outputs.

**`demo`** writes a single self-contained HTML file — open it in a browser,
drag the 3D trajectory, and slide the observation-noise dose. Supported demo
sources: `lorenz`, `rossler`, `chua`, `vanderpol`, `henon`, `logistic`.

```bash
serie demo lorenz --out tmp/lorenz_demo.html
serie demo henon --out tmp/henon_demo.html --data-out tmp/henon_demo.json
```

`--data-out` writes the browser payload as JSON. That JSON is the viewer
contract only: it stores `clean` points, frozen noise `draws`, feature names,
projection metadata, and seed/source metadata so a hosted or local demo page can
load it with the **Load local JSON** button. SERIE's data interchange artifacts
remain `npy`/CSV/IPC/Parquet exports from `serie generate`/`serie export`; Fern-side
high-throughput handoff can keep using IPC. JSON is for browser visualization
and metadata, not the preferred bulk array format.

**`walkthrough`** is the full package verification — it writes arrays, CSVs,
manifests, and figures for both a regular and parameter-shifted Lorenz source.

```bash
serie walkthrough --out tmp/walkthrough --overwrite --format json
```

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

## Release Boundary

Runtime dependencies are intentionally small and trainer-independent:

- `attrs`
- `cappa`
- `matplotlib`
- `numpy`
- `polars`

Optional diagnostic dependency:

- `rich`

SERIE must not depend on `torch`, Fern model modules, experiment registries,
result databases, or dashboard crates.

The default wheel ships one small prebuilt offline viewer template:
`serie/demo_dist/`, used by `serie demo` to produce a self-contained HTML
inspection artifact. That template is the only bundled frontend artifact. The
wheel does not bundle dashboards, `node_modules`, generated demo data, large
known datasets, generated arrays, model checkpoints, prediction windows,
reports, or run folders. User-requested artifacts from `serie generate --out`
or `export_dgp_bank_files(...)` are local verification artifacts owned by the
caller.

## Known Datasets

The default Python wheel does not bundle large known datasets. This keeps the
install small and keeps dataset provenance separate from the library release.

Known datasets are not second-class examples. When a recording is clean enough
to serve as a stable baseline process, titration is meaningful: it asks how far
the forecasting or calibration pipeline can be pushed as noise is
added under controlled conditions.

Current known-data source links:

- ETT-small (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`):
  <https://github.com/zhouhaoyi/ETDataset>
- HuGaDB gait / accel / EMG recordings:
  <https://github.com/romanchereshnev/HuGaDB>

These links identify the upstream sources. The package currently expects local
prepared IPC or CSV files for the built-in `KnownMeta` presets.

Built-in known-data presets look under:

```text
$SERIE_KNOWN_ROOT/
```

or, when that environment variable is unset:

```text
~/.cache/serie/known/
```

Expected prepared layout:

```text
known/
  ett/ETTh1.ipc
  ett/ETTh2.ipc
  ett/ETTm1.ipc
  ett/ETTm2.ipc
  hugadb/hugadb_accel_emg.ipc
  hugadb/hugadb_walk_bicycling_switch.ipc
```

Use `KnownMeta.custom(path=...)` when pointing at an externally downloaded CSV
or IPC file outside this cache layout.

Simulated systems such as Lorenz, Rossler, Chua, OU, DoubleWell, SLDS, GARCH,
and KS work without external data files. Recorded systems require explicit
local data preparation.

## Tests

```bash
cd serie
python -m pytest tests -q
```
