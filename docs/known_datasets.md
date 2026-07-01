[← Back to the README](../README.md)

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

