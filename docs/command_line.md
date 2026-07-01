[← Back to the README](../README.md)

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
serie generate lorenz --shock param --sigma 0.1
```

Read that as:

- `serie`: run the installed SERIE command.
- `generate`: generate one SERIE dataset cell.
- `lorenz`: the source name.
- `--shock param`: use the built-in parameter-shift intervention preset.
- `--sigma 0.1`: set the observation-noise dose.
- `generate` always prints machine-readable JSON for humans, scripts, and
  agents.

Options beginning with `--` are named options. Some options take values
(`--sigma 0.1`); some are switches (`--overwrite`).

To print the command menu in a form that is easy for humans and agents to scan:

```bash
serie list-cli-commands
serie list-cli-commands --format json
```

For detailed options on one command, use normal CLI help:

```bash
serie generate --help
```

### First Run: See It Work

Start with three commands. They show the available sources, create an
interactive visual demo, and generate one train/validation/test split:

```bash
serie list-data-ids
serie demo lorenz --out-html tmp/lorenz_demo.html
serie generate lorenz --sigma 0.1
```

Read that as:

1. `serie list-data-ids`: see which built-in sources are available.
2. `serie demo lorenz ...`: write one offline HTML file with a draggable Lorenz
   attractor and a live noise slider.
3. `serie generate lorenz ...`: generate one dataset cell and return
   train/validation/test split summary.

Once those work, use the command-specific sections below.

### Full Verification Walkthrough

For a fuller install audit, run these commands in order:

```bash
serie list-cli-commands --format json
serie list-data-ids --format json
serie show-data-dir --format json
serie generate ou --sigma 0.1 --partition p70_20_10
serie generate lorenz --shock param --sigma 0.1 --out tmp/lorenz_cell --array-format all --plot both
serie walkthrough --out tmp/walkthrough --array-format all --overwrite --format json
serie demo lorenz --out-html tmp/lorenz_demo.html --overwrite
```

This answers seven questions:

1. What commands does SERIE expose?
2. What source names does SERIE know?
3. Where should external dataset files live?
4. Can SERIE generate one cell and report both the DGP-bank summary and split summary?
5. Can one command combine intervention, noise, arrays, and figures?
6. Can SERIE write the standard audit walkthrough?
7. Can SERIE write one offline interactive attractor demo?

### Command Reference

| Command | What it does |
|---------|-------------|
| `serie list-cli-commands` | List all CLI commands with examples |
| `serie list-data-ids` | List available source names (synthetic + known) |
| `serie show-data-dir` | Show where external dataset files should live |
| `serie generate <source>` | The main workflow: source + intervention + noise + split + optional files |
| `serie demo <source>` | Write a self-contained interactive HTML with 3D attractor + noise slider |
| `serie walkthrough` | Full end-to-end audit: arrays, manifests, and figures for two sources |

`generate` always prints JSON. Discovery, walkthrough, and demo commands also
accept `--format json` when another program needs to parse their output. Use
`serie <command> --help` for full options.

**`generate`** is the command most users need. Examples:

```bash
serie generate lorenz --sigma 0.1
serie generate lorenz --shock param --sigma 0.1
serie generate lorenz --sigma 0.1 --out tmp/cell --array-format all --plot both
```

Key options: `--shock none|param|state|switch` for built-in interventions,
`--sigma` for noise dose, `--sigma-scale unit|train_std` for scaling mode,
`--partition` for split convention, `--out` for artifact directory,
`--array-format none|npy|csv|ipc|parquet|table|safetensors|all` and
`--plot` for file outputs.
When `--out tmp/cell` is used, files are written under a stable child directory
such as `tmp/cell/lorenz_<cell-hash>/`, not directly into `tmp/cell/`.

`--array-format safetensors` writes one grouped `dgp_bank.safetensors` with
`clean`, `noise`, and optional `noisy_sigma_*` tensors. Every export folder,
including safetensors-only exports, also writes `manifest.json` as the human and
agent-readable file index. Use `safetensors` for machine handoff; use CSV/IPC/
Parquet when you want table inspection.

**`demo`** writes a single self-contained HTML file — open it in a browser,
drag the 3D trajectory, and slide the observation-noise dose. Supported demo
sources: `lorenz`, `rossler`, `chua`, `vanderpol`, `henon`, `logistic`.

```bash
serie demo lorenz --out-html tmp/lorenz_demo.html
serie demo henon --out-html tmp/henon_demo.html --data-out tmp/henon_demo.json
serie demo henon --out-dir tmp/henon_demo_pack
```

`--data-out` writes the browser payload as JSON. That JSON is the viewer
contract only: it stores `clean` points, frozen noise `draws`, feature names,
projection metadata, and seed/source metadata so a hosted or local demo page can
load it with the **Load local JSON** button. SERIE's data interchange artifacts
remain `dgp_bank.safetensors` or table/array exports from `serie generate`.
JSON is for browser visualization and metadata, not the preferred bulk array
format.

Use `--out-dir` when you want an inspection folder instead of one HTML file.
It writes `index.html`, IPC array artifacts under `arrays/`, and static PNG
plots under `static/`. Add `--data-out path/to/demo.json` only when you also
want the browser JSON payload as a separate inspection file. The single-file
`--out-html` path remains the best default for sharing.

**`walkthrough`** is the full package verification — it writes arrays, CSVs,
manifests, and figures for both a regular and parameter-shifted Lorenz source.

```bash
serie walkthrough --out tmp/walkthrough --overwrite --format json
```

