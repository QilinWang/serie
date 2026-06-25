"""Property-based tests for SERIE's core invariants.

Most tests in this package are example-based: they pick one source, one seed,
one sigma, and assert the expected result.  That is the right style for named
regressions.  This file is different.  Hypothesis generates many small legal
inputs, so each test checks a rule that should be true for a whole family of
cases.

These tests are intentionally small and algebraic.  They are not slow sweeps.
The goal is to catch mistakes like:

- sigma=0 accidentally still touches the noise draw
- `clean + sigma * noise` changes under refactors
- split indices no longer match the documented fraction math
- seed provenance drifts so an ordinal no longer points to the spawned child
  seed it is supposed to identify
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import given, settings, strategies as st

from serie import PartitionSpec, build_empirical_dgp_bank, inject_noise
from serie.specs import spawn_draw_seeds


_SEED = st.integers(min_value=0, max_value=2**32 - 1)
_SIGMA = st.floats(
    min_value=-5.0,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
    exclude_min=True,
    exclude_max=True,
)


@st.composite
def _partition_cases(draw: Any) -> tuple[int, PartitionSpec]:
    """Generate legal split boundaries and a compatible series length.

    Hypothesis strategies are allowed to depend on earlier draws.  We first
    choose `train_end`, then choose `val_end` after it, so every emitted
    PartitionSpec is valid by construction.  That lets the test focus on split
    behavior instead of mostly testing invalid-input rejection.
    """
    T = draw(st.integers(min_value=90, max_value=180))
    train_end = draw(st.floats(min_value=0.25, max_value=0.65, allow_nan=False))
    val_end = draw(st.floats(min_value=train_end + 0.15, max_value=0.85, allow_nan=False))
    return T, PartitionSpec(
        train_start=0.0,
        train_end=float(train_end),
        val_start=float(train_end),
        val_end=float(val_end),
        test_start=float(val_end),
        test_end=1.0,
    )


@settings(max_examples=25, deadline=None)
@given(noise_seed_base=_SEED)
def test_sigma_zero_is_clean_for_many_noise_roots(noise_seed_base: int) -> None:
    """Zero dose must ignore the frozen observation draw completely.

    Why this matters:
    `sigma=0` is the control cell.  It should return the clean signal exactly,
    regardless of which noise root was used to build the bank.  If this fails,
    the baseline is no longer a true baseline, and every dose-response curve is
    contaminated at its origin.
    """
    values = np.arange(40, dtype=np.float32).reshape(20, 2)
    bank = build_empirical_dgp_bank(
        values,
        name="property_clean",
        noise_seed_base=noise_seed_base,
    )

    observed = bank.build_observed_series(noise_level=0.0)

    np.testing.assert_array_equal(observed, bank.clean[0])


@settings(max_examples=35, deadline=None)
@given(noise_seed_base=_SEED, sigma=_SIGMA)
def test_unit_scale_injection_is_linear_for_many_sigmas(
    noise_seed_base: int,
    sigma: float,
) -> None:
    """Unit-scale observations are exactly clean + sigma * frozen_noise.

    Why this matters:
    SERIE's controlled-noise guarantee is algebraic.  Once a bank is built, the
    only thing that should change across sigma values is the scalar multiplier.
    Hypothesis tries many positive and negative sigma values and many noise
    roots; the expected array is still the same simple equation every time.
    """
    values = np.arange(60, dtype=np.float32).reshape(30, 2) / 10.0
    bank = build_empirical_dgp_bank(
        values,
        name="property_linear",
        noise_seed_base=noise_seed_base,
    )

    observed = bank.build_observed_series(noise_level=sigma)
    expected = bank.clean[0] + np.float32(sigma) * bank.noise_draws[0]

    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


@settings(max_examples=30, deadline=None)
@given(case=_partition_cases())
def test_partition_indices_and_slices_match_manual_fraction_math(
    case: tuple[int, PartitionSpec],
) -> None:
    """Any legal split should expose the exact slices implied by its fractions.

    Why this matters:
    PartitionSpec is public API.  A user reading `train_end=0.7` expects the
    training rows to be `values[: int(T * 0.7)]`.  This property checks many
    legal boundaries and lengths, then compares SERIE's train/val/test accessors
    with the manual index math.
    """
    T, partition = case
    values = np.arange(T * 3, dtype=np.float32).reshape(T, 3)
    bank = build_empirical_dgp_bank(values, name="property_partition")

    split = inject_noise(bank, sigma=0.0, partition=partition)

    expected_indices = (
        int(T * partition.train_start),
        int(T * partition.train_end),
        int(T * partition.val_start),
        int(T * partition.val_end),
        int(T * partition.test_start),
        int(T * partition.test_end),
    )
    assert split.indices == expected_indices
    train_start, train_end, val_start, val_end, test_start, test_end = expected_indices
    np.testing.assert_array_equal(split.train_values, split.values[train_start:train_end])
    np.testing.assert_array_equal(split.val_values, split.values[val_start:val_end])
    np.testing.assert_array_equal(split.test_values, split.values[test_start:test_end])


@settings(max_examples=25, deadline=None)
@given(noise_seed_base=_SEED, n_noise_draws=st.integers(min_value=1, max_value=6))
def test_noise_draw_ordinal_matches_spawned_child_seed(
    noise_seed_base: int,
    n_noise_draws: int,
) -> None:
    """The draw ordinal is a spawn index; the child seed is its fingerprint.

    Why this matters:
    We store both the root seed and the draw ordinal because NumPy's
    `SeedSequence(root).spawn(n)` defines draw `i` by its spawn position.  The
    derived child seed is useful provenance, but it must stay synchronized with
    the ordinal.  This test rebuilds the expected spawn sequence and checks that
    every TitrationCell reports the matching ordinal/child-seed pair.
    """
    values = np.arange(24, dtype=np.float32).reshape(12, 2)
    bank = build_empirical_dgp_bank(
        values,
        name="property_seed_recipe",
        noise_seed_base=noise_seed_base,
        n_noise_draws=n_noise_draws,
    )

    expected_noise_seeds = spawn_draw_seeds(noise_seed_base, n_noise_draws)

    assert bank.noise_draw_seeds == expected_noise_seeds
    for ordinal, child_seed in enumerate(expected_noise_seeds):
        cell = bank.cell(noise_level=0.25, noise_draw_ordinal=ordinal)
        assert cell.noise_draw_ordinal == ordinal
        assert cell.noise_draw_seed == child_seed
