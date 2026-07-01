# SERIE

[![CI](https://github.com/QilinWang/serie/actions/workflows/ci.yml/badge.svg)](https://github.com/QilinWang/serie/actions/workflows/ci.yml)
[![Pages Demo](https://github.com/QilinWang/serie/actions/workflows/pages-demo.yml/badge.svg)](https://github.com/QilinWang/serie/actions/workflows/pages-demo.yml)
[![PyPI](https://img.shields.io/pypi/v/serie.svg)](https://pypi.org/project/serie/)
[![Downloads](https://static.pepy.tech/badge/serie/month)](https://pepy.tech/project/serie)

[中文说明](README.zh-CN.md) · Rust engine: [**serie-rs**](https://crates.io/crates/serie-rs) ([source](https://github.com/qilinwang/serie-rs))

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

## Documentation

Detailed references live under [`docs/`](docs/):

- [Command line API](docs/command_line.md) — every `serie` command, its options, and examples.
- [Python API](docs/python_api.md) — the programmable surface: `build_dgp_bank`, `inject_noise`, interventions, seed provenance, draw designs, and the stable import list.
- [Python walkthrough](docs/python_walkthrough.md) — a plain teaching tour of the primitives behind the CLI.
- [System presets](docs/system_presets.md) — stability notes and parameter choices for every built-in source.
- [Known datasets](docs/known_datasets.md) — bring-your-own recorded data and the expected on-disk layout.

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

## Rust engine (serie-rs)

SERIE has a Rust sibling, [**serie-rs**](https://crates.io/crates/serie-rs)
([source](https://github.com/qilinwang/serie-rs)). The two are siblings over one
protocol, not a port: this Python package is the published, documented benchmark
surface (CLI, demos, the vocabulary papers cite), while `serie-rs` is the
data-plane engine — the same staged pipeline (`load → clean → inject_noise →
partition → ready`) and the same system family, implemented in Rust. It also reads
the grouped `dgp_bank.safetensors` this package writes (`serie generate
--array-format safetensors`), so you can build a bank once in Python and consume it
from Rust. (`serie` is already taken on crates.io, hence the `-rs` suffix.)

## Tests

```bash
cd serie
python -m pytest tests -q
```
