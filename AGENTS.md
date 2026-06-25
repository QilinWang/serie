# SERIE Agent Guide

Use this file to orient coding agents that need to operate the installed
`serie` command. Keep it short. Human onboarding belongs in `README.md`;
maintainer and release notes live outside this package repo in
`../docs/dev/serie_release/developer.md`.

## What SERIE Does

SERIE builds time-series robustness test data. It starts from a clean source,
optionally applies an intervention, injects observation noise, and returns
train/validation/test split metadata or files.

The shell-first workflow is:

```text
source + optional shock + sigma -> serie generate -> split metadata/files
```

For Python code, the underlying primitives are `build_dgp_bank(...)` and
`inject_noise(...)`. For CLI work, prefer the higher-level command
`serie generate`.

## Core CLI Commands

Start by discovering the available surface:

```bash
serie commands
serie list
serie generate --help
```

Use `--format json` when another program or agent needs to parse the output.

```bash
serie commands --format json
serie list --format json
```

## Generate One Cell

`generate` is the command most agents should use. It resolves the source,
applies a built-in shock preset if requested, injects noise, splits the series,
and prints or writes the result.

```bash
serie generate lorenz --sigma 0.1 --format json
serie generate lorenz --shock param --sigma 0.1 --format json
serie generate ou --sigma 1.0 --sigma-scale train_std --format json
```

Useful options:

- `--shock none|param|state|switch`: choose a built-in intervention preset.
- `--sigma`: observation-noise dose.
- `--sigma-scale unit|train_std`: raw unit noise or train-std-scaled noise.
- `--partition`: split convention, such as `p70_20_10`.
- `--out`: write artifacts to a directory.
- `--array-format none|npy|csv|ipc|parquet|table|all`: choose array outputs.
- `--plot none|timeseries|attractor|both`: choose figure outputs.
- `--overwrite`: replace an existing output directory.

Example with files:

```bash
serie generate lorenz --shock param --sigma 0.1 \
  --out tmp/lorenz_cell --array-format all --plot both \
  --overwrite --format json
```

## Inspect And Visualize

Use these for quick checks:

```bash
serie inspect lorenz --format json
serie data-dir --format json
serie demo lorenz --out tmp/lorenz_demo.html --overwrite
serie walkthrough --out tmp/walkthrough --overwrite --format json
```

`demo` writes a self-contained HTML file for `lorenz`, `rossler`, `chua`,
`vanderpol`, `henon`, or `logistic`.
`walkthrough` writes a fuller audit with arrays, manifests, and figures.

## Concepts To Preserve

When interpreting outputs:

- `sigma` is the observation-noise dose.
- `trajectory_draw_ordinal` and `noise_draw_ordinal` are draw identities.
- deterministic sources default to one clean trajectory with many noise draws.
- stochastic sources default to paired trajectory/noise draws by ordinal.
- split statistics are train-only; do not compute scale from val/test rows.

Do not reintroduce old public names like `prepare` or `titrate`. Retired docs
may still contain them.

## If Editing Code

Keep changes scoped to the public data layer. Do not move Fern trainer,
dashboard, run-result, or checkpoint concepts back into SERIE.

Use `just ci-check` before handing work back. It runs Ruff, Pyright, and pytest.

For browser-template work, use `just dev-demo-template` to rebuild
`demo_dist/`, then `just dev-demo-inspect` to write ignored inspection HTML/JSON
files under `tmp/candidate_demos/`. Vite/esbuild has repeatedly hung inside the
Codex filesystem sandbox; if a sandboxed `bun run build` stalls after printing
`vite build`, rerun the Justfile recipe or `bun run build` outside the sandbox
instead of debugging frontend code that has not actually failed.

The maintainer may use Nushell or Fish interactively, but repo commands should
stay shell-simple. Put repeatable logic in `Justfile`, Python scripts, or CI
YAML instead of relying on shell-specific tricks.
