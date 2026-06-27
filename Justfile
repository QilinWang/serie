set dotenv-load := true

version := "0.1.1"
uv_serie := "UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable . serie"
uv_dev := "UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable '.[dev,diagnostics]'"

# Print the CLI command menu as JSON; no args. JSON is for scripts/agents to parse.
[group("01 commands-and-data")]
list-cli-commands:
    {{uv_serie}} list-cli-commands --format json

# Print synthetic source names and known dataset IDs as JSON; no args.
[group("01 commands-and-data")]
list-data-ids:
    {{uv_serie}} list-data-ids --format json

# Print the known-data folder path/existence as JSON; no args.
[group("01 commands-and-data")]
show-data-dir:
    {{uv_serie}} show-data-dir --format json

# Print generated-cell JSON; args: source=data ID, sigma=noise dose, shock=intervention, partition=split. Defaults: ou, 0.1, none, p70_20_10.
[group("01 commands-and-data")]
build-cell source="ou" sigma="0.1" shock="none" partition="p70_20_10":
    {{uv_serie}} generate {{source}} --shock {{shock}} --sigma {{sigma}} --partition {{partition}}

# Run the walkthrough and print JSON; default out=tmp/walkthrough, overwrites that output.
[group("02 author-walkthrough")]
walkthrough-run out="tmp/walkthrough":
    {{uv_serie}} walkthrough --out {{out}} --overwrite --format json

# Meaningful every-system visual audit: clean/noise plus intervention cases, PNGs only.
[group("02 author-walkthrough")]
system-check out="tmp/system_check":
    just clean-system-check {{out}}
    {{uv_dev}} python ../docs/serie_release/system_check.py --out {{out}}

# Check the checkout end-to-end: source quality, tests, wheel build, and clean install.
[group("03 check")]
check:
    just check-source
    just check-wheel

# Unit under `check`: Ruff, Pyright, and pytest coverage.
[group("03 check")]
check-source:
    {{uv_dev}} ruff check .
    {{uv_dev}} pyright
    {{uv_dev}} python -m pytest tests -q --cov --cov-report=term-missing --cov-report=xml

# Unit under `check`: build dist, install the wheel cleanly, and smoke-test the installed CLI.
[group("03 check")]
check-wheel tmp="tmp/release_check":
    just clean-release-tmp {{tmp}}
    just build-dist-release
    mkdir -p {{tmp}}
    env -u UV_PROJECT_ENVIRONMENT uv venv {{tmp}}/.venv
    uv pip install --python {{tmp}}/.venv/bin/python dist/serie-{{version}}-py3-none-any.whl
    {{tmp}}/.venv/bin/serie version
    {{tmp}}/.venv/bin/serie generate lorenz
    {{tmp}}/.venv/bin/serie generate lorenz --out {{tmp}}/export --sigma 0.1 --array-format all
    {{tmp}}/.venv/bin/serie walkthrough --out {{tmp}}/walkthrough --overwrite --format json
    just check-wheel-files {{tmp}}
    just clean-release-tmp {{tmp}}

# Unit under `check-wheel`: assert expected generated files exist.
[group("03 check")]
check-wheel-files tmp="tmp/release_check":
    test -f {{tmp}}/export/lorenz_*/serie_cell_identity.json
    test -f {{tmp}}/export/lorenz_*/clean.npy
    test -f {{tmp}}/export/lorenz_*/clean.csv
    test -f {{tmp}}/export/lorenz_*/clean.ipc
    test -f {{tmp}}/export/lorenz_*/clean.parquet
    test -f {{tmp}}/export/lorenz_*/noise_draws.npy
    test -f {{tmp}}/export/lorenz_*/noise_draws.csv
    test -f {{tmp}}/export/lorenz_*/noise_draws.ipc
    test -f {{tmp}}/export/lorenz_*/noise_draws.parquet
    test -f {{tmp}}/export/lorenz_*/manifest.json
    test -f {{tmp}}/walkthrough/arrays/lorenz/clean.npy
    test -f {{tmp}}/walkthrough/arrays/lorenz/clean.csv
    test -f {{tmp}}/walkthrough/arrays/lorenz_param_shock/manifest.json
    test -f {{tmp}}/walkthrough/arrays/lorenz_param_shock/noise_draws.csv
    test -f {{tmp}}/walkthrough/plots/regular_timeseries.png
    test -f {{tmp}}/walkthrough/plots/regular_noise_timeseries.png
    test -f {{tmp}}/walkthrough/plots/param_shift_timeseries.png
    test -f {{tmp}}/walkthrough/plots/param_shift_noise_timeseries.png
    test -f {{tmp}}/walkthrough/plots/regular_attractor.png
    test -f {{tmp}}/walkthrough/plots/regular_noise_attractor.png
    test -f {{tmp}}/walkthrough/plots/param_shift_attractor.png
    test -f {{tmp}}/walkthrough/plots/param_shift_noise_attractor.png

# Build the PyPI upload files into dist/.
[group("04 release")]
build-dist-release:
    rm -rf dist
    just clean-egg-info
    UV_CACHE_DIR=/tmp/uv-cache uv build . --out-dir dist
    just clean-egg-info

# Build standalone static demo HTML; output is tmp/deploy_site/index.html, no args.
[group("05 demo")]
build-demo-static:
    mkdir -p tmp/deploy_site
    {{uv_dev}} serie demo shapes --out-html tmp/deploy_site/index.html --overwrite --format json

# Optional: create/update persistent local .venv; normal recipes do not need it.
[group("05 dev")]
dev-env:
    env -u UV_PROJECT_ENVIRONMENT UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
    UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -e ".[dev,diagnostics]"

# Rebuild packaged browser viewer template from demo/; run outside sandbox if Vite stalls.
[group("05 dev")]
dev-demo-template:
    cd demo && XDG_RUNTIME_DIR=/tmp TMPDIR=/tmp bun run build
    rm -rf demo_dist
    cp -R demo/dist demo_dist
    rm -f demo_dist/attractor_data.json

# Generate local demo HTML+JSON files; default out=tmp/candidate_demos, outputs are ignored.
[group("05 dev")]
dev-demo-inspect out="tmp/candidate_demos":
    rm -rf {{out}}
    mkdir -p {{out}}
    {{uv_dev}} serie demo lorenz --out-html {{out}}/lorenz.html --data-out {{out}}/lorenz.json --n-points 4000 --n-draws 5 --overwrite --format json
    {{uv_dev}} serie demo rossler --out-html {{out}}/rossler.html --data-out {{out}}/rossler.json --n-points 4000 --n-draws 5 --overwrite --format json
    {{uv_dev}} serie demo chua --out-html {{out}}/chua.html --data-out {{out}}/chua.json --n-points 4000 --n-draws 5 --overwrite --format json
    {{uv_dev}} serie demo vanderpol --out-html {{out}}/vanderpol.html --data-out {{out}}/vanderpol.json --n-points 4000 --n-draws 5 --overwrite --format json
    {{uv_dev}} serie demo henon --out-html {{out}}/henon.html --data-out {{out}}/henon.json --n-points 4000 --n-draws 5 --overwrite --format json
    {{uv_dev}} serie demo logistic --out-html {{out}}/logistic.html --data-out {{out}}/logistic.json --n-points 4000 --n-draws 5 --overwrite --format json

# Run the full check, then upload dist/ files to PyPI. Requires UV_PUBLISH_TOKEN.
[group("04 release")]
publish-release:
    just check
    test -n "$UV_PUBLISH_TOKEN"
    uv publish dist/serie-{{version}}.tar.gz dist/serie-{{version}}-py3-none-any.whl

# List maintainer notes that are intentionally outside the PyPI package.
[group("06 author-docs")]
docs-list:
    find ../docs/dev/serie_release -maxdepth 2 -type f

# Unit under clean/build/check: remove setuptools egg-info metadata.
[group("07 author-clean")]
clean-egg-info:
    rm -rf serie.egg-info

# Remove build/, dist/, and egg-info; source and .venv are untouched.
[group("07 author-clean")]
clean-build:
    rm -rf build
    rm -rf dist
    just clean-egg-info

# Unit under check-wheel: remove temp install audit dir and egg-info.
[group("07 author-clean")]
clean-release-tmp tmp="tmp/release_check":
    rm -rf {{tmp}}
    just clean-egg-info

# Remove generated tmp/ outputs and egg-info; source, dist/, and .venv stay.
[group("07 author-clean")]
clean-tmp:
    rm -rf tmp
    just clean-egg-info

# Remove only the every-system visual audit folder.
[group("07 author-clean")]
clean-system-check out="tmp/system_check":
    rm -rf {{out}}
    just clean-egg-info

# Remove local development .venv; uv can recreate it on the next run.
[group("07 author-clean")]
clean-env:
    rm -rf .venv

# Critical reset: remove tmp outputs, build artifacts, dist/, egg-info, and .venv.
[group("07 author-clean")]
clean-all:
    just clean-tmp
    just clean-build
    just clean-env
