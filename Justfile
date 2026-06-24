set dotenv-load := true

version := "0.1.1"

# Author CLI check: list SERIE CLI commands with examples.
[group("01 author-cli")]
cli-commands:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable . serie commands --format json

# Author CLI check: list source names.
[group("01 author-cli")]
cli-list:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable . serie list --format json

# Author CLI check: show external dataset directory.
[group("01 author-cli")]
cli-data-dir:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable . serie data-dir --format json

# Author CLI check: generate one source and print metadata.
[group("01 author-cli")]
cli-inspect source="ou":
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable . serie inspect {{source}} --format json

# Author CLI check: inject sigma noise and split train/val/test.
[group("01 author-cli")]
cli-inject-noise source="ou" sigma="0.1":
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable . serie inject-noise {{source}} --sigma {{sigma}} --partition p70_20_10 --format json

# High-level walkthrough: run the installed CLI end-to-end under tmp/.
[group("02 author-walkthrough")]
walkthrough-run out="tmp/walkthrough":
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable . serie walkthrough --out {{out}} --overwrite --format json

# Meaningful every-system visual audit: clean/noise plus intervention cases, PNGs only.
[group("02 author-walkthrough")]
system-check out="tmp/system_check":
    just clean-system-check {{out}}
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable ".[dev,diagnostics]" python ../docs/dev/serie_release/system_check.py --out {{out}}

# Source-only Python example; hidden because it is not an installed CLI command.
[group("03 author-dev")]
[private]
example-run:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable . python -m serie.example

# Run the SERIE test suite against the editable checkout.
[group("03 author-dev")]
dev-test:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with-editable ".[dev,diagnostics]" python -m pytest tests -q

# Optional: create/update persistent local .venv; normal recipes do not need it.
[group("03 author-dev")]
dev-env:
    env -u UV_PROJECT_ENVIRONMENT UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
    UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -e ".[dev,diagnostics]"

# Build sdist and wheel into dist/, then remove temporary egg-info.
[group("04 author-release")]
release-build:
    just clean-egg-info
    UV_CACHE_DIR=/tmp/uv-cache uv build . --out-dir dist
    just clean-egg-info

# Clean-install audit: build wheel, install in temp venv, verify CLI/files, clean.
[group("04 author-release")]
release-check tmp="tmp/release_check":
    just clean-release-check {{tmp}}
    mkdir -p {{tmp}}
    just clean-egg-info
    UV_CACHE_DIR=/tmp/uv-cache uv build . --out-dir dist
    env -u UV_PROJECT_ENVIRONMENT uv venv {{tmp}}/.venv
    uv pip install --python {{tmp}}/.venv/bin/python dist/serie-{{version}}-py3-none-any.whl
    {{tmp}}/.venv/bin/serie version
    {{tmp}}/.venv/bin/serie inspect lorenz --format json
    {{tmp}}/.venv/bin/serie export lorenz --out {{tmp}}/export --sigma 0.1 --format json
    {{tmp}}/.venv/bin/serie walkthrough --out {{tmp}}/walkthrough --overwrite --format json
    just release-check-files {{tmp}}
    just clean-release-check {{tmp}}

# Full local gate before publishing: test editable code, then test the built wheel.
[group("04 author-release")]
release-ready:
    just dev-test
    just release-check

# Print the publish command without uploading anything.
[group("04 author-release")]
release-preview-publish:
    just --dry-run release-publish

# Internal release-check helper: assert expected NPY/CSV/manifest/plot files exist.
[group("05 author-release-units")]
[private]
release-check-files tmp="tmp/release_check":
    test -f {{tmp}}/export/clean.npy
    test -f {{tmp}}/export/clean.csv
    test -f {{tmp}}/export/noise_draws.npy
    test -f {{tmp}}/export/noise_draws.csv
    test -f {{tmp}}/export/manifest.json
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

# Publish current dist/ artifacts to PyPI; requires UV_PUBLISH_TOKEN.
[group("04 author-release")]
release-publish:
    test -n "$UV_PUBLISH_TOKEN"
    uv publish dist/serie-{{version}}.tar.gz dist/serie-{{version}}-py3-none-any.whl

# List maintainer notes that are intentionally outside the PyPI package.
[group("06 author-docs")]
docs-list:
    find ../docs/dev/serie_release -maxdepth 2 -type f

# Internal helper: remove setuptools egg-info metadata.
[group("07 author-clean")]
[private]
clean-egg-info:
    rm -rf serie.egg-info

# Remove build/, dist/, and egg-info; source and .venv are untouched.
[group("07 author-clean")]
clean-build:
    rm -rf build
    rm -rf dist
    just clean-egg-info

# Internal release-check cleanup: remove temp install audit dir and egg-info.
[group("07 author-clean")]
[private]
clean-release-check tmp="tmp/release_check":
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
