# SERIE Agent Guide

Use this file to orient coding agents that need to operate the installed
`serie` command. Keep it short. Human onboarding belongs in `README.md`;
maintainer and release notes live outside this package repo in
`../docs/serie_release/developer.md`.

## What SERIE Does

SERIE builds time-series robustness test data. It starts from a clean source,
optionally applies an intervention, injects observation noise, and returns
the JSON DGP-bank summary, train/validation/test split summary, or files.

The shell-first workflow is:

```text
source + optional shock + sigma -> serie generate -> JSON summary/files
```

For Python code, the underlying primitives are `build_dgp_bank(...)` and
`inject_noise(...)`. For CLI work, prefer the higher-level command
`serie generate`.

## Core CLI Commands

Start by discovering the available surface:

```bash
serie list-cli-commands
serie list-data-ids
serie generate --help
```

`generate` always prints JSON. Use `--format json` on discovery, demo, or
walkthrough commands when another program or agent needs to parse their output.

```bash
serie list-cli-commands --format json
serie list-data-ids --format json
```

## Generate One Cell

`generate` is the command most agents should use. It resolves the source,
applies a built-in shock preset if requested, injects noise, splits the series,
and prints a JSON DGP-bank summary plus split summary.

```bash
serie generate lorenz --sigma 0.1
serie generate lorenz --shock param --sigma 0.1
serie generate ou --sigma 1.0 --sigma-scale train_std
```

Useful options:

- `--shock none|param|state|switch`: choose a built-in intervention preset.
- `--sigma`: observation-noise dose.
- `--sigma-scale unit|train_std`: raw unit noise or train-std-scaled noise.
- `--partition`: split convention, such as `p70_20_10`.
- `--out`: write artifacts under a stable `<source>_<cell-hash>` child directory.
- `--array-format none|npy|csv|ipc|parquet|table|safetensors|all`: choose array outputs.
  `safetensors` writes one grouped `dgp_bank.safetensors`; every export folder
  still writes `manifest.json` as the file index.
- `--plot none|timeseries|attractor|both`: choose figure outputs.

Example with files:

```bash
serie generate lorenz --shock param --sigma 0.1 \
  --out tmp/lorenz_cell --array-format all --plot both
```

## Inspect And Visualize

Use these for quick checks:

```bash
serie show-data-dir --format json
serie generate lorenz
serie demo lorenz --out-html tmp/lorenz_demo.html --overwrite
serie demo henon --out-dir tmp/henon_demo_pack --overwrite --format json
serie walkthrough --out tmp/walkthrough --overwrite --format json
```

`demo` writes a self-contained HTML file for `lorenz`, `rossler`, `chua`,
`vanderpol`, `henon`, or `logistic`.
Use `demo --out-dir` when a folder with HTML, IPC arrays, and static PNG plots
is more useful than one shareable HTML file. Add `--data-out` only when a
separate browser JSON payload is needed for inspection.
`walkthrough` writes a fuller audit with arrays, manifests, and figures.

## Concepts To Preserve

When interpreting outputs:

- `sigma` is the observation-noise dose.
- `trajectory_draw_ordinal` and `noise_draw_ordinal` are draw identities.
- deterministic sources default to one clean trajectory with many noise draws.
- stochastic sources default to paired trajectory/noise draws by ordinal.
- split statistics are train-only; do not compute scale from val/test rows.

Do not reintroduce old public names like `prepare` or `titrate`. Deleted or
retired docs may contain them in history; do not recreate those docs.

## If Editing Code

Keep changes scoped to the public data layer. Do not move Fern trainer,
dashboard, run-result, or checkpoint concepts back into SERIE.

Use `just check` before handing work back. It runs Ruff, Pyright, pytest with
coverage, builds the wheel, installs that wheel in a clean temp venv, and
smoke-tests the installed CLI. `just publish-release` runs this same check
before uploading to PyPI.

For browser-template work, use `just dev-demo-template` to rebuild
`demo_dist/`, then `just dev-demo-inspect` to write ignored inspection HTML/JSON
files under `tmp/candidate_demos/`. Vite/esbuild has repeatedly hung inside the
Codex filesystem sandbox; if a sandboxed `bun run build` stalls after printing
`vite build`, rerun the Justfile recipe or `bun run build` outside the sandbox
instead of debugging frontend code that has not actually failed.

The maintainer may use Nushell or Fish interactively, but repo commands should
stay shell-simple. Put repeatable logic in `Justfile`, Python scripts, or CI
YAML instead of relying on shell-specific tricks.
