# fr_vis Migration TODO

This is a temporary landing zone for the former root `fr_vis.py`. The file was
moved verbatim to `serie/retired/fr_vis_legacy_temp.py` before trimming active
Python compatibility helpers.

## Intended Shape

`serie` should not depend on Fern model or trainer code. The reusable plotting
surface should read artifact data, not `RunResult` objects:

- input: a run directory or explicit `sequence_tensors.safetensors`;
- required tensors: `pred [S,T,D]`, `truth [1,T,D]`;
- optional tensors: `eigenvalue_max`, `eigenvalue_sum`, `eigenvalue_logdet`;
- output: static matplotlib figures for paper/export use, or a small derived
  table only at the rendering boundary.

## Migration Tasks

1. Replace `RunResult` and `SeqKind` imports with a tensor-native loader.
2. Keep eigen rendering as a data-presence check: if eigenvalue tensors exist,
   draw eigen panels; otherwise draw forecast/error only.
3. Avoid restoring the old melted DataFrame contract as the internal API.
4. Decide public packaging deliberately: either `serie[viz]` one-shot CLI or a
   small internal plotting module. Do not silently add trainer dependencies.
5. `artifact_plots_split_ipc_candidate.py` is the retired split-IPC candidate.
   Replace it with a tensor-native reader before treating it as the final viz
   path.
6. After the tensor-native rewrite exists, delete this temporary retired copy.

## What Not To Preserve

- Three separate eigen IPC files.
- A `kind` discriminator table as the canonical artifact.
- `RunResult`/`RunId`/trainer imports inside `serie`.
