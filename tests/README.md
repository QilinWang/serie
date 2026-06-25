# SERIE Test Suite

Keep the test suite flat until it grows large enough to need subdirectories.
Use file names to show intent.

## Files

- `test_gen.py` — integrator and generator correctness: shapes, finiteness,
  burn-in, shocks, diagonal ground truth, SDE moments, and source presets.
- `test_titration_edges.py` — DGP-bank and titration invariants: determinism,
  common random numbers, noise laws, draw pairing, train-only statistics,
  provenance, and export round trips.
- `test_properties.py` — randomized property tests for invariants that should
  hold across many seeds, sigma values, and legal partition boundaries.
- `test_public_contracts.py` — public API contracts and defensive behavior:
  renamed fields, shape validation, alias safety, feature-name flow, split
  slicing, and high-level edge cases.
- `test_cli.py` — installed CLI behavior and JSON payloads.
- `test_known_data_boundary.py` — known-data files are not bundled and missing
  data errors are explicit.

## Where To Add A New Test

- Generator math or an equation-level property: `test_gen.py`.
- Noise injection, draw semantics, or protocol invariants:
  `test_titration_edges.py`.
- Algebra or split invariants that should hold for many generated inputs:
  `test_properties.py`.
- Public API behavior that protects users from silent mistakes:
  `test_public_contracts.py`.
- Command-line behavior: `test_cli.py`.
- Known-recording packaging or local-data boundary: `test_known_data_boundary.py`.

Prefer fast, explicit contract tests over broad slow sweeps. Slow visual or
every-system audits belong in maintainer scripts, not the default pytest suite.
