"""Edge-case tests for the titration layer: build_dgp_bank / inject_noise / DGPBank.

test_gen.py covers the integrator layer (shapes, finiteness, shock divergence).
This file covers the layer above it — the contracts that make titration sweeps
scientifically valid:

    1. Determinism      — same seed, same bits; different seed, different data
    2. Injection        — noisy == clean + sigma * noise_draw, exactly; purity
    3. Shared draws     — sigmas differ only in amplitude, never in realization
    4. Partition/stats  — split bounds honored; stats computed on train only
    5. Error contracts  — wrong inputs fail loudly with actionable messages

These tests use the public SERIE names (build_dgp_bank / inject_noise / sigma).
"""

from __future__ import annotations

from msgspec.structs import replace

import copy
import dataclasses
import json

import numpy as np
import polars as pl
import pytest

from serie import (
    PROTOCOL_VERSION,
    DGPBank,
    CellMaterializationSpec,
    MaterializedCell,
    TitrationCell,
    PartitionSpec,
    DrawDesign,
    SigmaScale,
    SplitTreatmentScheme,
    ShockEvent,
    ShockSchedule,
    TitrationPlan,
    TrajectorySpec,
    __version__,
    export_dgp_bank_files,
    load_materialized_cell,
    load_dgp_bank_safetensors,
    materialize_cell,
    save_materialized_cell,
    build_dgp_bank,
    build_empirical_dgp_bank,
    inject_noise,
)
from serie.api import (
    build_clean_split_bundle,
    dgp_bank_from_tensors,
    export_dgp_bank_safetensors,
    iter_cells,
)
from serie.catalog import LorenzParams, OUParams, SeasonalARParams
from serie.integrators import _interpolate_params
from serie.specs import spawn_draw_seeds
from serie.noise import (
    NoiseKind,
    NoiseTemporalProcess,
    ObservationNoiseProcess,
    impose_temporal_dependence,
)


# Small, fast specs — unit tests must not integrate 36k steps.
def small_lorenz(steps: int = 400) -> TrajectorySpec:
    return TrajectorySpec.lorenz(steps=steps)


def small_ou(steps: int = 300) -> TrajectorySpec:
    return TrajectorySpec.ou(steps=steps)


def test_stage_params_interpolate_at_progress_one_lands_exactly_on_after():
    before = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    after = LorenzParams(sigma=10.1, rho=28.1, beta=8.1 / 3.0)

    assert _interpolate_params(before, after, 1.0) == after


def test_stage_params_interpolate_matches_two_product_lerp_form():
    before = OUParams(theta=0.2, mu=0.0, sigma_noise=0.3)
    after = OUParams(theta=0.8, mu=0.5, sigma_noise=0.6)
    alpha = 0.25

    result = _interpolate_params(before, after, alpha)

    assert isinstance(result, OUParams)
    assert result.theta == before.theta * (1.0 - alpha) + after.theta * alpha
    assert result.mu == before.mu * (1.0 - alpha) + after.mu * alpha
    assert result.sigma_noise == before.sigma_noise * (1.0 - alpha) + after.sigma_noise * alpha


def test_stage_params_seasonal_ar_period_stays_fixed_during_interpolation():
    before = SeasonalARParams(S=24, phi=0.5, sigma=0.2, a0=1.0, amp_drift=0.0)
    after = SeasonalARParams(S=48, phi=0.8, sigma=0.35, a0=1.4, amp_drift=0.0)

    for alpha in (0.5, 1.0):
        result = _interpolate_params(before, after, alpha)
        assert isinstance(result, SeasonalARParams)
        assert result.S == before.S
        assert result.phi == before.phi * (1.0 - alpha) + after.phi * alpha


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Determinism
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeterminism:
    def test_same_seed_bitwise_identical(self):
        a = build_dgp_bank(small_lorenz(), seed=7, verbose=False)
        b = build_dgp_bank(small_lorenz(), seed=7, verbose=False)
        np.testing.assert_array_equal(a.clean, b.clean)
        np.testing.assert_array_equal(a.noise_draws, b.noise_draws)

    def test_different_seed_changes_stochastic_clean_not_noise(self):
        # OU is an SDE — the trajectory seed drives its Wiener noise.
        a = build_dgp_bank(small_ou(), seed=7, verbose=False)
        b = build_dgp_bank(small_ou(), seed=8, verbose=False)
        assert not np.array_equal(a.clean, b.clean)
        # noise draws are seeded by noise_seed_base, independent
        # of the trajectory seed
        np.testing.assert_array_equal(a.noise_draws, b.noise_draws)

    def test_deterministic_ode_ignores_trajectory_seed(self):
        # PINNED CONTRACT (found by this test): deterministic ODEs have a
        # fixed initial condition, so the trajectory seed is a no-op and
        # n_trajectory_draws > 1 yields IDENTICAL draws. Trajectory diversity
        # for deterministic systems comes from n_noise_draws (this is how the
        # titration suites are configured: ROSSLER n_noise_draws=3 vs
        # OU n_trajectory_draws=3). If IC jitter is ever added, this test should
        # flip to inequality — intentionally.
        a = build_dgp_bank(small_lorenz(), seed=7, verbose=False)
        b = build_dgp_bank(small_lorenz(), seed=8, verbose=False)
        np.testing.assert_array_equal(a.clean, b.clean)

    def test_different_noise_seed_changes_noise_not_clean(self):
        a = build_dgp_bank(small_lorenz(), seed=7, noise_seed_base=1000, verbose=False)
        b = build_dgp_bank(small_lorenz(), seed=7, noise_seed_base=2000, verbose=False)
        np.testing.assert_array_equal(a.clean, b.clean)
        assert not np.array_equal(a.noise_draws, b.noise_draws)

    def test_stochastic_systems_are_seeded_too(self):
        # SDEs draw their own randomness — must still be reproducible by seed
        a = build_dgp_bank(small_ou(), seed=3, verbose=False)
        b = build_dgp_bank(small_ou(), seed=3, verbose=False)
        np.testing.assert_array_equal(a.clean, b.clean)

    def test_draw_seeds_are_spawned_not_consecutive(self):
        bank = build_dgp_bank(
            small_ou(),
            seed=5,
            noise_seed_base=1000,
            n_trajectory_draws=3,
            n_noise_draws=3,
            verbose=False,
        )
        assert bank.trajectory_seed_base == 5
        assert bank.noise_seed_base == 1000
        assert bank.trajectory_draw_seeds == spawn_draw_seeds(5, 3)
        assert bank.noise_draw_seeds == spawn_draw_seeds(1000, 3)
        assert bank.trajectory_draw_seeds != (5, 6, 7)
        assert bank.noise_draw_seeds != (1000, 1001, 1002)


class TestDrawProtocolPort:
    """Exact test-function ports from ``serie-rs/src/draw_protocol.rs``."""

    def test_spawned_draw_seeds_are_stable_and_not_consecutive(self):
        seeds = spawn_draw_seeds(42, 3)
        assert seeds == (803958421, 1820700080, 3421468131)
        assert seeds == spawn_draw_seeds(42, 3)
        assert seeds != (42, 43, 44)
        assert seeds[0] != seeds[1]

    def test_deterministic_auto_uses_one_trajectory_many_noise_draws(self):
        bank = build_dgp_bank(
            small_lorenz(),
            seed=7,
            noise_seed_base=1000,
            n_trajectory_draws=1,
            n_noise_draws=3,
            verbose=False,
        )
        plan = TitrationPlan(noise_levels=(0.0, 1.0), n_trajectory_draws=1, n_noise_draws=3)
        cells = list(iter_cells(bank, plan))
        assert [(cell.trajectory_draw_ordinal, cell.noise_draw_ordinal) for cell in cells] == [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 0),
            (0, 1),
            (0, 2),
        ]

    def test_deterministic_auto_rejects_multiple_trajectory_draws(self):
        bank = build_dgp_bank(
            small_lorenz(),
            seed=7,
            noise_seed_base=1000,
            n_trajectory_draws=2,
            n_noise_draws=2,
            verbose=False,
        )
        plan = TitrationPlan(noise_levels=(0.0,), n_trajectory_draws=2, n_noise_draws=2)
        with pytest.raises(ValueError, match="noise_only"):
            list(iter_cells(bank, plan))

    def test_stochastic_auto_pairs_trajectory_and_noise_draws(self):
        bank = build_dgp_bank(
            small_ou(),
            seed=7,
            noise_seed_base=1000,
            n_trajectory_draws=3,
            n_noise_draws=3,
            verbose=False,
        )
        plan = TitrationPlan(noise_levels=(0.0, 1.0), n_trajectory_draws=3, n_noise_draws=3)
        cells = list(iter_cells(bank, plan))
        assert [(cell.trajectory_draw_ordinal, cell.noise_draw_ordinal) for cell in cells] == [
            (0, 0),
            (1, 1),
            (2, 2),
            (0, 0),
            (1, 1),
            (2, 2),
        ]

    @pytest.mark.parametrize(("seed", "n_noise_draws"), [(7, 3), (0, 2)])
    def test_stochastic_auto_requires_equal_draw_counts(self, seed, n_noise_draws):
        bank = build_dgp_bank(
            small_ou(),
            seed=seed,
            noise_seed_base=1000,
            n_trajectory_draws=1,
            n_noise_draws=n_noise_draws,
            verbose=False,
        )
        plan = TitrationPlan(noise_levels=(0.0,), n_trajectory_draws=1, n_noise_draws=n_noise_draws)
        with pytest.raises(ValueError, match="n_trajectory_draws must equal n_noise_draws"):
            list(iter_cells(bank, plan))

    def test_stochastic_cross_product_is_explicit(self):
        bank = build_dgp_bank(
            small_ou(),
            seed=7,
            noise_seed_base=1000,
            n_trajectory_draws=2,
            n_noise_draws=3,
            verbose=False,
        )
        plan = TitrationPlan(
            noise_levels=(0.0,),
            n_trajectory_draws=2,
            n_noise_draws=3,
            draw_design=DrawDesign.CROSS_PRODUCT,
        )
        assert len(list(iter_cells(bank, plan))) == 6


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. The injection identity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInjection:
    def test_noisy_equals_clean_plus_sigma_times_draw(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        sigma = 0.37
        noisy = bank.build_observed_series(noise_level=sigma)
        expected = bank.clean[0] + sigma * bank.noise_draws[0]
        np.testing.assert_array_equal(noisy, expected)

    def test_sigma_zero_is_exactly_clean(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        np.testing.assert_array_equal(bank.build_observed_series(noise_level=0.0), bank.clean[0])

    def test_injection_is_pure(self):
        # building observed arrays must not mutate the bank; repeat calls agree
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        clean_before = bank.clean.copy()
        draws_before = bank.noise_draws.copy()
        first = bank.build_observed_series(noise_level=1.5)
        second = bank.build_observed_series(noise_level=1.5)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(bank.clean, clean_before)
        np.testing.assert_array_equal(bank.noise_draws, draws_before)

    def test_noise_draws_is_approximately_standard_normal(self):
        bank = build_dgp_bank(small_lorenz(steps=2000), seed=0, verbose=False)
        draw = bank.noise_draws[0]
        assert abs(float(draw.mean())) < 0.05
        assert abs(float(draw.std()) - 1.0) < 0.05

    def test_dtype_is_float32(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        assert bank.clean.dtype == np.float32
        assert bank.build_observed_series(noise_level=0.5).dtype == np.float32


class TestPrepareFromArray:
    def test_clean_recording_becomes_dgp_bank(self):
        values = np.arange(24, dtype=np.float32).reshape(12, 2)
        bank = build_empirical_dgp_bank(values, name="wearable_gait", feature_names=("x", "y"))
        assert bank.name == "wearable_gait"
        assert bank.clean.shape == (1, 12, 2)
        assert bank.feature_names == ("x", "y")
        np.testing.assert_array_equal(bank.clean[0], values)

    def test_train_std_scaled_sigma_uses_train_channel_scale(self):
        values = np.stack(
            [
                np.linspace(0.0, 10.0, 20, dtype=np.float32),
                np.linspace(0.0, 100.0, 20, dtype=np.float32),
            ],
            axis=1,
        )
        bank = build_empirical_dgp_bank(values, name="scaled", noise_seed_base=7)
        partition = PartitionSpec.p70_20_10()
        clean = inject_noise(bank, sigma=0.0, partition=partition)
        scaled = inject_noise(
            bank,
            sigma=1.0,
            sigma_scale="train_std",
            partition=partition,
        )
        draw = bank.noise_draws[0][: clean.train_values.shape[0]]
        expected = clean.train_values + clean.train_stats.std[None, :] * draw
        np.testing.assert_allclose(scaled.train_values, expected, rtol=1e-5, atol=1e-5)

    def test_an_observed_series_reads_back_as_the_bundle_it_was(self, tmp_path):
        from serie.api import load_observed_series, save_observed_series
        from serie.pipeline import NoisedSplits, StageInjectTrainScaledNoise

        values = np.stack([np.linspace(-2.0, 2.0, 40, dtype=np.float32), np.linspace(10.0, 50.0, 40, dtype=np.float32)], axis=1)
        bank = build_empirical_dgp_bank(values, name="observed", noise_seed_base=17)
        clean = build_clean_split_bundle(bank, 0, PartitionSpec.p70_20_10(), print_diagnostics=False)
        observed = copy.copy(clean) | StageInjectTrainScaledNoise(noise_level=0.5, noise_draw=bank.noise_draws[0], noise_on=NoisedSplits.of(("train", "val")))
        path = save_observed_series(observed, tmp_path / "observed_series.safetensors")
        back = load_observed_series(path)
        np.testing.assert_array_equal(back.series, observed.series)
        assert back.indices == tuple(observed.indices) and back.channels == 2 and back.injected_sigma == 0.5
        assert back.dataset_id == observed.dataset_id and list(back.feature_names) == list(observed.feature_names)
        np.testing.assert_allclose(back.train_partition_stats.std, observed.train_partition_stats.std)
        assert back.test_partition_stats.counts == observed.test_partition_stats.counts
        with pytest.raises(FileExistsError):
            save_observed_series(observed, path)

    def test_noise_on_names_the_splits_and_each_scheme_is_one_subset(self):
        from serie.pipeline import NoisedSplits, StageInjectNoise

        values = np.stack(
            [
                np.linspace(-2.0, 2.0, 40, dtype=np.float32),
                np.linspace(10.0, 50.0, 40, dtype=np.float32),
            ],
            axis=1,
        )
        bank = build_empirical_dgp_bank(values, name="noise_on", noise_seed_base=17)
        partition = PartitionSpec.p70_20_10()
        clean = build_clean_split_bundle(bank, 0, partition, print_diagnostics=False)
        draw = bank.noise_draws[0]
        t0, t1, v0, v1, s0, s1 = clean.indices
        for names, rows in (
            (("train",), [(t0, t1)]),
            (("val",), [(v0, v1)]),
            (("test",), [(s0, s1)]),
            (("val", "test"), [(v0, v1), (s0, s1)]),
            (("train", "val", "test"), [(0, values.shape[0])]),
        ):
            db = copy.copy(clean) | StageInjectNoise(
                noise_level=0.5, noise_draw=draw, noise_on=NoisedSplits.of(names)
            )
            expected = clean.series.copy()
            for a, b in rows:
                expected[a:b] += 0.5 * draw[a:b]
            np.testing.assert_allclose(db.series, expected, rtol=1e-6, atol=1e-6)
        for scheme, names in (
            (SplitTreatmentScheme.NOISE_MATCHED_RETRAINING, ("train", "val", "test")),
            (SplitTreatmentScheme.CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION, ("test",)),
            (SplitTreatmentScheme.NOISE_TRAIN_NOISE_SELECT_CLEAN_TEST, ("train", "val")),
        ):
            assert NoisedSplits.of_scheme(scheme).names() == names
        with pytest.raises(ValueError, match="non-empty subset"):
            NoisedSplits.of(())
        with pytest.raises(ValueError, match="non-empty subset"):
            NoisedSplits.of(("training",))

    def test_csr_scheme_keeps_train_and_validation_clean_and_titrates_only_test(self):
        values = np.stack(
            [
                np.linspace(-2.0, 2.0, 40, dtype=np.float32),
                np.linspace(10.0, 50.0, 40, dtype=np.float32),
            ],
            axis=1,
        )
        bank = build_empirical_dgp_bank(values, name="csr", noise_seed_base=17)
        partition = PartitionSpec.p70_20_10()
        clean = inject_noise(bank, sigma=0.0, partition=partition)
        dose = 0.5
        treated = inject_noise(
            bank,
            sigma=dose,
            sigma_scale="train_std",
            partition=partition,
            split_treatment_scheme=SplitTreatmentScheme.CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION,
        )

        np.testing.assert_array_equal(treated.train_values, clean.train_values)
        np.testing.assert_array_equal(treated.val_values, clean.val_values)
        test_start, test_end = clean.indices[4], clean.indices[5]
        expected_test = clean.test_values + dose * (
            clean.train_stats.std[None, :] * bank.noise_draws[0][test_start:test_end]
        )
        np.testing.assert_allclose(treated.test_values, expected_test, rtol=1e-5, atol=1e-5)
        assert (
            treated.metadata["split_treatment_scheme"] == "clean_train_clean_select_test_titration"
        )

    def test_return_to_clean_scheme_titrates_train_and_validation_only(self):
        values = np.stack(
            [
                np.linspace(-2.0, 2.0, 40, dtype=np.float32),
                np.linspace(10.0, 50.0, 40, dtype=np.float32),
            ],
            axis=1,
        )
        bank = build_empirical_dgp_bank(values, name="return_to_clean", noise_seed_base=17)
        partition = PartitionSpec.p70_20_10()
        clean = inject_noise(bank, sigma=0.0, partition=partition)
        dose = 0.5
        treated = inject_noise(
            bank,
            sigma=dose,
            sigma_scale="train_std",
            partition=partition,
            split_treatment_scheme=SplitTreatmentScheme.NOISE_TRAIN_NOISE_SELECT_CLEAN_TEST,
        )

        train_start, train_end, val_start, val_end, _, _ = clean.indices
        expected_train = clean.train_values + dose * (
            clean.train_stats.std[None, :] * bank.noise_draws[0][train_start:train_end]
        )
        expected_val = clean.val_values + dose * (
            clean.train_stats.std[None, :] * bank.noise_draws[0][val_start:val_end]
        )
        np.testing.assert_allclose(treated.train_values, expected_train, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(treated.val_values, expected_val, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(treated.test_values, clean.test_values)
        assert treated.metadata["split_treatment_scheme"] == "noise_train_noise_select_clean_test"

    def test_relative_test_titration_has_paired_algebra_and_no_scale_leakage(self):
        values = np.stack(
            [
                np.linspace(-3.0, 3.0, 100, dtype=np.float32),
                np.linspace(20.0, 80.0, 100, dtype=np.float32),
            ],
            axis=1,
        )
        partition = PartitionSpec.p70_20_10()
        bank = build_empirical_dgp_bank(values, name="paired_canary", noise_seed_base=31)
        clean = inject_noise(bank, sigma=0.0, partition=partition)
        dose_low = 0.25
        dose_high = 1.0
        low = inject_noise(
            bank,
            sigma=dose_low,
            sigma_scale="train_std",
            partition=partition,
            split_treatment_scheme=SplitTreatmentScheme.CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION,
        )
        high = inject_noise(
            bank,
            sigma=dose_high,
            sigma_scale="train_std",
            partition=partition,
            split_treatment_scheme=SplitTreatmentScheme.CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION,
        )

        test_start, test_end = clean.indices[4], clean.indices[5]
        scaled_test_draw = clean.train_stats.std[None, :] * bank.noise_draws[0][test_start:test_end]
        np.testing.assert_allclose(
            low.test_values - clean.test_values,
            dose_low * scaled_test_draw,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            high.test_values - clean.test_values,
            (dose_high / dose_low) * (low.test_values - clean.test_values),
            rtol=1e-5,
            atol=1e-5,
        )

        test_perturbed_values = values.copy()
        test_perturbed_values[test_start:test_end] += np.array(
            [10_000.0, -20_000.0], dtype=np.float32
        )
        perturbed_bank = build_empirical_dgp_bank(
            test_perturbed_values,
            name="paired_canary_test_perturbed",
            noise_seed_base=31,
        )
        perturbed = inject_noise(
            perturbed_bank,
            sigma=dose_low,
            sigma_scale="train_std",
            partition=partition,
            split_treatment_scheme=SplitTreatmentScheme.CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION,
        )
        np.testing.assert_array_equal(perturbed.train_stats.std, low.train_stats.std)


class TestNoiseKinds:
    """The protocol works with any exactly-known noise law.

    Built-ins are unit-variance by convention; custom callables may choose a
    different known scale.
    """

    def test_gaussian_is_the_default_bitwise(self):
        a = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        b = build_dgp_bank(small_lorenz(), seed=0, noise_kind="gaussian", verbose=False)
        np.testing.assert_array_equal(a.noise_draws, b.noise_draws)
        assert a.noise_kind == "gaussian"

    def test_laplace_draws_have_unit_variance(self):
        bank = build_dgp_bank(small_lorenz(steps=4000), seed=0, noise_kind="laplace", verbose=False)
        assert abs(float(bank.noise_draws.std()) - 1.0) < 0.05
        assert bank.noise_kind == "laplace"
        assert bank.noise_draws.dtype == np.float32

    def test_laplace_is_heavier_tailed_than_gaussian(self):
        # Excess kurtosis: Laplace = 3.0, Gaussian = 0.0. With 12k samples the
        # estimate separates them by a wide margin.
        gauss = build_dgp_bank(
            small_lorenz(steps=4000), seed=0, noise_kind="gaussian", verbose=False
        ).noise_draws
        lap = build_dgp_bank(
            small_lorenz(steps=4000), seed=0, noise_kind="laplace", verbose=False
        ).noise_draws

        def excess_kurtosis(x: np.ndarray) -> float:
            x = x.reshape(-1).astype(np.float64)
            z = (x - x.mean()) / x.std()
            return float((z**4).mean() - 3.0)

        assert excess_kurtosis(gauss) < 1.0
        assert excess_kurtosis(lap) > 1.5

    def test_laplace_is_seeded(self):
        a = build_dgp_bank(small_lorenz(), seed=0, noise_kind="laplace", verbose=False)
        b = build_dgp_bank(small_lorenz(), seed=0, noise_kind="laplace", verbose=False)
        np.testing.assert_array_equal(a.noise_draws, b.noise_draws)

    def test_custom_callable_law(self):
        # Uniform(-sqrt(3), sqrt(3)) has unit variance, so sigma reads as the
        # injected standard deviation for this custom law too.
        def uniform_unit(rng, shape):
            bound = np.sqrt(3.0)
            return rng.uniform(-bound, bound, size=shape).astype(np.float32)

        bank = build_dgp_bank(
            small_lorenz(steps=4000), seed=0, noise_kind=uniform_unit, verbose=False
        )
        assert bank.noise_kind == "custom"
        assert abs(float(bank.noise_draws.std()) - 1.0) < 0.05
        # Bounded support proves the custom law was actually used
        assert float(np.abs(bank.noise_draws).max()) <= np.sqrt(3.0) + 1e-6

    def test_custom_callable_can_have_non_unit_scale(self):
        # A physical sensor law may own its own scale. Here sigma=1.0 means
        # "one draw unit", not "one standard deviation".
        def two_sigma_gaussian(rng, shape):
            return (2.0 * rng.standard_normal(shape)).astype(np.float32)

        bank = build_dgp_bank(
            small_lorenz(steps=4000),
            seed=0,
            noise_kind=two_sigma_gaussian,
            verbose=False,
        )
        assert bank.noise_kind == "custom"
        assert abs(float(bank.noise_draws.std()) - 2.0) < 0.1
        noisy = bank.build_observed_series(noise_level=0.5)
        expected = bank.clean[0] + 0.5 * bank.noise_draws[0]
        np.testing.assert_array_equal(noisy, expected)

    def test_unknown_noise_kind_lists_valid_options(self):
        with pytest.raises(ValueError) as exc:
            build_dgp_bank(small_lorenz(), noise_kind="cauchy", verbose=False)
        message = str(exc.value)
        assert "gaussian" in message and "laplace" in message
        assert "callable" in message

    def test_custom_callable_shape_mismatch_is_rejected(self):
        def wrong_shape(rng, shape):
            return rng.standard_normal((1, 2, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="shape"):
            build_dgp_bank(small_lorenz(), noise_kind=wrong_shape, verbose=False)


def _stage_dgp_bank_noise(kind: str, *, timesteps: int = 10_000) -> DGPBank:
    """Use a fixed empirical clean path so the test isolates observation noise."""
    return build_empirical_dgp_bank(
        np.zeros((timesteps, 1), dtype=np.float32),
        name=f"pass8_{kind}",
        noise_kind=kind,
        noise_seed_base=1_000,
    )


def test_stage_dgp_bank_student_t3_variance_and_heavy_tails():
    bank = _stage_dgp_bank_noise("student_t3")
    values = bank.noise_draws.reshape(-1).astype(np.float64)
    assert bank.noise_kind == "student_t3"
    assert abs(float(values.mean())) < 0.05
    assert abs(float(values.var()) - 1.0) < 0.3
    assert int(np.count_nonzero(np.abs(values) > 4.0)) >= 5


def test_stage_dgp_bank_extending_noise_draws_preserves_prefix():
    values = np.zeros((257, 3), dtype=np.float32)
    for kind in ("gaussian", "laplace", "student_t3", "contaminated"):
        four = build_empirical_dgp_bank(
            values,
            name=f"pass8_prefix_{kind}",
            n_noise_draws=4,
            noise_seed_base=1_000,
            noise_kind=kind,
        )
        eight = build_empirical_dgp_bank(
            values,
            name=f"pass8_prefix_{kind}",
            n_noise_draws=8,
            noise_seed_base=1_000,
            noise_kind=kind,
        )
        np.testing.assert_array_equal(four.noise_draws, eight.noise_draws[:4])


def test_stage_dgp_bank_laplace_variance_and_middle_tails():
    bank = _stage_dgp_bank_noise("laplace")
    values = bank.noise_draws.reshape(-1).astype(np.float64)
    tail_count = int(np.count_nonzero(np.abs(values) > 4.0))
    assert bank.noise_kind == "laplace"
    assert abs(float(values.mean())) < 0.03
    assert abs(float(values.var()) - 1.0) < 0.08
    assert 5 <= tail_count <= 120


def test_stage_dgp_bank_contaminated_variance_and_sparse_tails():
    bank = _stage_dgp_bank_noise("contaminated", timesteps=200_000)
    values = bank.noise_draws.reshape(-1).astype(np.float64)
    tail_count = int(np.count_nonzero(np.abs(values) > 4.0))
    assert bank.noise_kind == "contaminated"
    assert abs(float(values.mean())) < 0.03
    assert abs(float(values.var()) - 1.0) < 0.08
    assert 1_200 <= tail_count <= 2_800


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Common random numbers across doses — the comparability property
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCommonRandomNumbers:
    def test_sigmas_differ_only_in_amplitude(self):
        # The design fact that makes titration sweeps comparable is common
        # random numbers: every sigma reuses the same noise draw.
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        recovered = [
            (bank.build_observed_series(noise_level=s) - bank.clean[0]) / s
            for s in (0.25, 1.0, 2.0)
        ]
        # float32: clean is O(30) for Lorenz, so (clean + s*u) - clean carries
        # cancellation error ~ |clean| * eps / s ≈ 2e-5. atol set accordingly.
        np.testing.assert_allclose(recovered[0], recovered[1], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(recovered[1], recovered[2], rtol=1e-3, atol=1e-4)

    def test_public_titrate_reuses_the_same_draw_between_sigmas(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        partition = PartitionSpec.p70_20_10()
        low = inject_noise(bank, sigma=0.25, partition=partition)
        high = inject_noise(bank, sigma=0.75, partition=partition)
        expected_delta = 0.5 * bank.noise_draws[0]
        np.testing.assert_allclose(high.values - low.values, expected_delta, rtol=1e-3, atol=1e-4)

    def test_draws_are_independent_for_stochastic_systems(self):
        # OU: trajectory draws differ (seed+i drives the SDE noise).
        # For deterministic ODEs they would be identical — see
        # TestDeterminism.test_deterministic_ode_ignores_trajectory_seed.
        bank = build_dgp_bank(
            small_ou(), seed=0, n_trajectory_draws=2, n_noise_draws=2, verbose=False
        )
        assert bank.clean.shape[0] == 2
        assert bank.noise_draws.shape[0] == 2
        assert not np.array_equal(bank.clean[0], bank.clean[1])
        assert not np.array_equal(bank.noise_draws[0], bank.noise_draws[1])

    def test_inject_selects_requested_draws(self):
        bank = build_dgp_bank(
            small_ou(), seed=0, n_trajectory_draws=2, n_noise_draws=2, verbose=False
        )
        sigma = 0.5
        noisy = bank.build_observed_series(
            noise_level=sigma, trajectory_draw_ordinal=1, noise_draw_ordinal=1
        )
        expected = bank.clean[1] + sigma * bank.noise_draws[1]
        np.testing.assert_array_equal(noisy, expected)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Partition + train-only stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPartition:
    def test_split_lengths_match_fractions(self):
        bank = build_dgp_bank(small_lorenz(steps=999), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.0, partition=PartitionSpec.p70_20_10())
        T = bank.T
        # rounding tolerance of 1 step per boundary
        assert abs(split.train_values.shape[0] - 0.7 * T) <= 1
        assert abs(split.val_values.shape[0] - 0.2 * T) <= 1
        assert abs(split.test_values.shape[0] - 0.1 * T) <= 1

    def test_contiguous_partition_reconstructs_series(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.0, partition=PartitionSpec.p70_20_10())
        glued = np.concatenate([split.train_values, split.val_values, split.test_values])
        np.testing.assert_array_equal(glued, bank.clean[0][: glued.shape[0]])

    def test_train_stats_use_train_slice_only(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.0, partition=PartitionSpec.p70_20_10())
        manual_mean = split.train_values.mean(axis=0)
        manual_std = split.train_values.std(axis=0)
        np.testing.assert_allclose(split.train_stats.mean, manual_mean, rtol=1e-4)
        np.testing.assert_allclose(split.train_stats.std, manual_std, rtol=1e-2)
        # and they must NOT equal full-series stats (Lorenz drifts across splits)
        full_mean = bank.clean[0].mean(axis=0)
        assert not np.allclose(split.train_stats.mean, full_mean, rtol=1e-6, atol=1e-8)

    def test_gap_partition_skips_boundary(self):
        bank = build_dgp_bank(small_lorenz(steps=1000), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.0, partition=PartitionSpec.p70_gap_10_20())
        # val starts at 71%, so train-end and val-start are not adjacent
        T = bank.T
        n_train = split.train_values.shape[0]
        val_start_expected = bank.clean[0][int(0.71 * T)]
        np.testing.assert_allclose(split.val_values[0], val_start_expected, rtol=1e-5)
        assert abs(n_train - 0.7 * T) <= 1

    def test_split_carries_noise(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        clean_split = inject_noise(bank, sigma=0.0, partition=PartitionSpec.p70_20_10())
        noisy_split = inject_noise(bank, sigma=2.0, partition=PartitionSpec.p70_20_10())
        assert not np.array_equal(clean_split.train_values, noisy_split.train_values)
        diff = noisy_split.train_values - clean_split.train_values
        n = diff.shape[0]
        np.testing.assert_allclose(diff, 2.0 * bank.noise_draws[0][:n], rtol=1e-4, atol=1e-5)


class TestPartitionValidation:
    @pytest.mark.parametrize(
        ("val_start", "test_start", "error"),
        [(0.6, 0.9, "Val bounds invalid"), (0.7, 0.85, "Test bounds invalid")],
    )
    def test_overlapping_partition_boundaries_are_rejected(self, val_start, test_start, error):
        with pytest.raises(ValueError, match=error):
            PartitionSpec(
                train_start=0.0,
                train_end=0.7,
                val_start=val_start,
                val_end=0.9,
                test_start=test_start,
                test_end=1.0,
            )

    def test_fraction_outside_unit_interval_rejected(self):
        with pytest.raises(ValueError):
            PartitionSpec(
                train_start=0.0,
                train_end=1.5,
                val_start=1.5,
                val_end=1.6,
                test_start=1.6,
                test_end=2.0,
            )

    def test_empty_train_rejected(self):
        with pytest.raises(ValueError, match="Train bounds"):
            PartitionSpec(
                train_start=0.5,
                train_end=0.5,
                val_start=0.5,
                val_end=0.8,
                test_start=0.8,
                test_end=1.0,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Error contracts + provenance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestErrorContracts:
    def test_unknown_system_lists_valid_names(self):
        with pytest.raises(ValueError) as exc:
            build_dgp_bank("not_a_system", verbose=False)
        message = str(exc.value)
        assert "not_a_system" in message
        assert "lorenz" in message  # the fix is in the message

    def test_source_catalog_accepts_case_and_dash_in_a_preset_name(self):
        from serie.catalog import source_preset

        assert source_preset("LORENZ-PARAM-SHOCK") is source_preset("lorenz_param_shock")

    def test_bank_string_coercion_is_case_insensitive(self):
        a = build_dgp_bank("LORENZ", seed=0, verbose=False)
        b = build_dgp_bank("lorenz", seed=0, verbose=False)
        np.testing.assert_array_equal(a.clean, b.clean)

    def test_wrong_source_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="TrajectorySpec"):
            build_dgp_bank(42, verbose=False)

    def test_known_data_rejects_multiple_trajectory_draw_ordinals(self):
        from serie import KnownMeta

        with pytest.raises(ValueError, match="n_trajectory_draws=1"):
            build_dgp_bank(KnownMeta.etth1(), seed=0, n_trajectory_draws=2, verbose=False)

    def test_known_data_requires_zero_trajectory_seed_before_asset_load(self):
        from serie import KnownMeta

        # The seed refusal happens before any file access. The zero-seed case
        # reaches the independent draw-count guard, proving it is not refused
        # merely for being a known recording.
        with pytest.raises(ValueError, match=r"trajectory_seed_base=0; got seed=7"):
            build_dgp_bank(KnownMeta.etth1(), seed=7, n_trajectory_draws=2, verbose=False)
        with pytest.raises(ValueError, match="n_trajectory_draws=1"):
            build_dgp_bank(KnownMeta.etth1(), seed=0, n_trajectory_draws=2, verbose=False)


class TestProvenance:
    def test_bank_rejects_seed_provenance_drift(self):
        clean = np.zeros((1, 8, 2), dtype=np.float32)
        noise_draws = np.zeros((1, 8, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="trajectory_draw_seeds"):
            DGPBank(
                name="bad",
                trajectory_seed_base=7,
                noise_seed_base=1000,
                trajectory_draw_seeds=(8,),
                noise_draw_seeds=spawn_draw_seeds(1000, 1),
                clean=clean,
                noise_draws=noise_draws,
                feature_names=("x", "y"),
                is_stochastic=False,
                noise_kind="gaussian",
                noise_ar1_phi=0.0,
            )

    def test_bank_rejects_out_of_range_draw_ordinals(self):
        bank = build_dgp_bank(
            small_ou(), seed=0, n_trajectory_draws=2, n_noise_draws=2, verbose=False
        )
        with pytest.raises(IndexError, match="trajectory_draw_ordinal"):
            bank.build_observed_series(noise_level=0.1, trajectory_draw_ordinal=2)
        with pytest.raises(IndexError, match="noise_draw_ordinal"):
            bank.build_observed_series(noise_level=0.1, noise_draw_ordinal=2)

    def test_titration_cell_records_the_cell(self):
        bank = build_dgp_bank(small_lorenz(), seed=5, n_noise_draws=2, verbose=False)
        cell = bank.cell(noise_level=0.25, trajectory_draw_ordinal=0, noise_draw_ordinal=1)
        assert cell.noise_level == 0.25
        assert isinstance(cell, TitrationCell)
        assert cell.trajectory_draw_seed == bank.trajectory_draw_seeds[0]
        assert cell.trajectory_seed_base == bank.trajectory_seed_base
        assert cell.noise_seed_base == bank.noise_seed_base
        assert cell.noise_draw_seed == bank.noise_draw_seeds[1]
        assert cell.noise_draw_ordinal == 1
        assert cell.T == bank.T and cell.D == bank.D
        assert "sig0.25" in cell.tag

    def test_split_metadata_records_protocol_version(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
        assert split.metadata["serie_version"] == __version__
        assert split.metadata["protocol_version"] == PROTOCOL_VERSION
        assert split.metadata["sigma"] == 0.1
        assert split.metadata["trajectory_draw_seed"] == bank.trajectory_draw_seeds[0]
        assert split.metadata["noise_draw_seed"] == bank.noise_draw_seeds[0]

    def test_synthetic_source_identity_survives_noise_injection(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
        assert bank.name == "lorenz"
        assert split.source_id == bank.name

    def test_shape_contract_includes_initial_condition_row(self):
        spec = TrajectorySpec.ou(steps=123, burn_in=7)
        bank = build_dgp_bank(spec, seed=0, verbose=False)
        assert bank.clean.shape == (1, spec.steps - spec.burn_in + 1, spec.dim)

    def test_iter_cells_covers_the_full_grid(self):
        bank = build_dgp_bank(
            small_lorenz(), seed=0, n_trajectory_draws=2, n_noise_draws=2, verbose=False
        )
        plan = TitrationPlan(
            noise_levels=(0.0, 1.0),
            n_trajectory_draws=2,
            n_noise_draws=2,
            draw_design=DrawDesign.CROSS_PRODUCT,
        )
        titration_cells = list(iter_cells(bank, plan))
        assert len(titration_cells) == 2 * 2 * 2
        cells = {
            (c.noise_level, c.trajectory_draw_ordinal, c.noise_draw_ordinal)
            for c in titration_cells
        }
        assert len(cells) == 8  # no duplicates

    def test_stochastic_path_fixed_is_explicit_diagnostic(self):
        bank = build_dgp_bank(
            small_ou(), seed=0, n_trajectory_draws=1, n_noise_draws=3, verbose=False
        )
        plan = TitrationPlan(
            noise_levels=(0.0,),
            n_trajectory_draws=1,
            n_noise_draws=3,
            draw_design="trajectory_fixed",
        )
        cells = list(iter_cells(bank, plan))
        assert [(c.trajectory_draw_ordinal, c.noise_draw_ordinal) for c in cells] == [
            (0, 0),
            (0, 1),
            (0, 2),
        ]

    def test_clean_bundle_must_match_cell_trajectory_draw(self):
        bank = build_dgp_bank(
            small_ou(), seed=0, n_trajectory_draws=2, n_noise_draws=2, verbose=False
        )
        partition = PartitionSpec.p70_20_10()
        from serie.api import build_clean_split_bundle, build_noisy_bundle_for_cell

        db_clean = build_clean_split_bundle(
            bank, trajectory_draw_ordinal=0, partition_spec=partition, print_diagnostics=False
        )
        cell = bank.cell(noise_level=0.1, trajectory_draw_ordinal=1, noise_draw_ordinal=1)
        with pytest.raises(ValueError, match="draw mismatch"):
            build_noisy_bundle_for_cell(bank, db_clean, cell)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"trajectory_draw_ordinal": True}, "trajectory_draw_ordinal"),
            ({"trajectory_draw_ordinal": -1}, "trajectory_draw_ordinal"),
            ({"noise_draw_ordinal": -1}, "noise_draw_ordinal"),
            ({"sigma": np.float32(-0.1)}, "sigma"),
            ({"sigma": np.float32(np.nan)}, "sigma"),
            ({"sigma_scale": "unknown"}, "unknown"),
        ],
    )
    def test_materialization_spec_refuses_invalid_boundary_values(self, override, message):
        fields = {
            "trajectory_draw_ordinal": 0,
            "noise_draw_ordinal": 0,
            "sigma": np.float32(0.25),
            "partition": PartitionSpec.p70_20_10(),
            "split_treatment": SplitTreatmentScheme.NOISE_MATCHED_RETRAINING,
            "sigma_scale": SigmaScale.UNIT,
        }
        fields.update(override)

        with pytest.raises((ValueError, TypeError), match=message):
            CellMaterializationSpec(**fields)

    def test_materialized_cell_is_a_standalone_serie_artifact(self, tmp_path):
        values = np.stack(
            [
                np.linspace(-3.0, 3.0, 100, dtype=np.float32),
                np.linspace(20.0, 80.0, 100, dtype=np.float32),
            ],
            axis=1,
        )
        bank = build_empirical_dgp_bank(values, name="standalone_cell", noise_seed_base=31)
        spec = CellMaterializationSpec(
            trajectory_draw_ordinal=0,
            noise_draw_ordinal=0,
            sigma=np.float32(0.25),
            partition=PartitionSpec.p70_20_10(),
            split_treatment=SplitTreatmentScheme.CLEAN_TRAIN_CLEAN_SELECT_TEST_TITRATION,
            sigma_scale=SigmaScale.TRAIN_STD,
        )

        materialized = materialize_cell(bank, spec)

        assert isinstance(materialized, MaterializedCell)
        np.testing.assert_array_equal(materialized.clean, values)
        np.testing.assert_array_equal(materialized.observed[:80], values[:80])
        test_start, test_end = materialized.split_indices[4:]
        expected_test = values[test_start:test_end] + spec.sigma * (
            materialized.clean_train_stats.std[None, :] * bank.noise_draws[0][test_start:test_end]
        )
        np.testing.assert_allclose(
            materialized.observed[test_start:test_end],
            expected_test,
            rtol=1e-5,
            atol=1e-5,
        )

        path = tmp_path / "materialized_cell.safetensors"
        manifest = save_materialized_cell(materialized, path)
        reopened = load_materialized_cell(path)

        assert manifest["schema"] == "serie.materialized_cell_safetensors.v1"
        assert manifest["stats_source"] == "clean_partitions"
        assert reopened.spec == spec
        assert reopened.cell == materialized.cell
        assert reopened.feature_names == materialized.feature_names
        np.testing.assert_array_equal(reopened.clean, materialized.clean)
        np.testing.assert_array_equal(reopened.observed, materialized.observed)
        np.testing.assert_array_equal(reopened.unit_noise, materialized.unit_noise)
        np.testing.assert_array_equal(
            reopened.clean_train_stats.std, materialized.clean_train_stats.std
        )
        second_path = tmp_path / "materialized_cell_second.safetensors"
        save_materialized_cell(materialized, second_path)
        assert second_path.read_bytes() == path.read_bytes()
        with pytest.raises(FileExistsError):
            save_materialized_cell(materialized, path)

    def test_export_dgp_bank_files_writes_arrays_tables_and_manifest(self, tmp_path):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        out = tmp_path / "export"
        manifest = export_dgp_bank_files(bank, out, sigmas=(0.25,))

        assert manifest["schema"] == "serie.dgp_bank_file_export.v1"
        assert manifest["array_format"] == "all"
        assert manifest["files"]["safetensors"] == "dgp_bank.safetensors"
        assert (out / "dgp_bank.safetensors").exists()
        np.testing.assert_array_equal(np.load(out / "clean.npy"), bank.clean)
        np.testing.assert_array_equal(np.load(out / "noise_draws.npy"), bank.noise_draws)
        clean_csv = pl.read_csv(out / "clean.csv")
        assert clean_csv.columns == ["trajectory_draw_ordinal", "t", "f0", "f1", "f2"]
        assert clean_csv.height == bank.n_trajectory_draws * bank.T
        assert pl.read_ipc(out / "clean.ipc").height == bank.n_trajectory_draws * bank.T
        assert pl.read_parquet(out / "clean.parquet").height == bank.n_trajectory_draws * bank.T

        entry = manifest["noisy"][0]
        noisy = np.load(out / entry["files"]["npy"])
        expected = bank.build_observed_series(noise_level=0.25)
        np.testing.assert_array_equal(noisy, expected)
        noisy_csv = pl.read_csv(out / entry["files"]["csv"])
        assert noisy_csv.columns == ["t", "f0", "f1", "f2"]
        assert noisy_csv.height == bank.T
        assert pl.read_ipc(out / entry["files"]["ipc"]).height == bank.T
        assert pl.read_parquet(out / entry["files"]["parquet"]).height == bank.T

        on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert on_disk["noisy"][0]["cell_tag"] == entry["cell_tag"]
        with pytest.raises(FileExistsError):
            export_dgp_bank_files(bank, out)

    def test_export_dgp_bank_files_writes_grouped_safetensors_and_manifest(self, tmp_path):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        out = tmp_path / "grouped"
        manifest = export_dgp_bank_files(bank, out, sigmas=(0.25,), array_format="safetensors")

        assert manifest["schema"] == "serie.dgp_bank_safetensors.v1"
        assert manifest["array_format"] == "safetensors"
        assert manifest["files"] == {"safetensors": "dgp_bank.safetensors"}
        assert (out / "manifest.json").exists()
        assert (out / "dgp_bank.safetensors").exists()

        artifact = load_dgp_bank_safetensors(out / "dgp_bank.safetensors")
        np.testing.assert_array_equal(artifact.clean, bank.clean)
        np.testing.assert_array_equal(artifact.noise, bank.noise_draws)
        assert artifact.feature_names == bank.feature_names

        noisy_entry = manifest["noisy"][0]
        tensor_name = noisy_entry["tensor"]
        assert tensor_name in artifact.tensors
        expected = bank.build_observed_series(noise_level=0.25)
        np.testing.assert_array_equal(artifact.tensors[tensor_name], expected)

        on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert on_disk["files"] == manifest["files"]
        assert on_disk["noisy"][0]["tensor"] == tensor_name
        with pytest.raises(FileExistsError):
            export_dgp_bank_files(bank, out, array_format="safetensors")

    def test_dgp_bank_round_trips_through_safetensors_to_dgp_bank(self, tmp_path):
        # export → load → convert must rebuild the SAME bank: this is the
        # contract the Rust-generated bank handoff relies on (the serie-rs
        # writer emits the same header, so a bank from either engine passes
        # DGPBank's spawn-consistency validators on load).
        bank = build_dgp_bank(
            small_ou(), seed=42, n_trajectory_draws=2, n_noise_draws=3, verbose=False
        )
        path = tmp_path / "dgp_bank.safetensors"
        export_dgp_bank_safetensors(bank, path)

        rebuilt = dgp_bank_from_tensors(load_dgp_bank_safetensors(path))

        assert rebuilt.name == bank.name
        assert rebuilt.trajectory_seed_base == bank.trajectory_seed_base
        assert rebuilt.noise_seed_base == bank.noise_seed_base
        assert rebuilt.trajectory_draw_seeds == bank.trajectory_draw_seeds
        assert rebuilt.noise_draw_seeds == bank.noise_draw_seeds
        assert rebuilt.feature_names == bank.feature_names
        assert rebuilt.is_stochastic == bank.is_stochastic
        assert rebuilt.noise_kind == bank.noise_kind
        np.testing.assert_array_equal(rebuilt.clean, bank.clean)
        np.testing.assert_array_equal(rebuilt.noise_draws, bank.noise_draws)

    def test_dgp_bank_from_tensors_requires_the_full_header(self, tmp_path):
        bank = build_dgp_bank(small_ou(), seed=42, verbose=False)
        path = tmp_path / "dgp_bank.safetensors"
        export_dgp_bank_safetensors(bank, path)
        artifact = load_dgp_bank_safetensors(path)
        stripped = dataclasses.replace(
            artifact,
            metadata={k: v for k, v in artifact.metadata.items() if k != "trajectory_draw_seeds"},
        )
        with pytest.raises(ValueError, match="trajectory_draw_seeds"):
            dgp_bank_from_tensors(stripped)

    def test_public_metadata_is_immutable(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
        with pytest.raises(TypeError):
            split.metadata["sigma"] = 999.0

        loaded = build_empirical_dgp_bank(np.ones((20, 2), dtype=np.float32), name="recording")
        split = inject_noise(loaded, sigma=0.0)
        with pytest.raises(TypeError):
            split.metadata["source_name"] = "mutated"

    def test_param_shift_matches_regular_before_shock_then_diverges(self):
        base = small_lorenz(steps=600)
        shifted = replace(
            base,
            shocks=ShockSchedule.single(
                ShockEvent.params_step_change(
                    params_after=LorenzParams(sigma=10.1, rho=28.1, beta=8.1 / 3.0),
                    t_start=0.35,
                )
            ),
        )

        regular_bank = build_dgp_bank(base, seed=42, verbose=False)
        shifted_bank = build_dgp_bank(shifted, seed=42, verbose=False)
        shock_idx = int(0.35 * regular_bank.T)

        np.testing.assert_array_equal(
            regular_bank.clean[0, : shock_idx - 2],
            shifted_bank.clean[0, : shock_idx - 2],
        )
        post_mean_abs = np.mean(
            np.abs(
                regular_bank.clean[0, shock_idx + 20 :] - shifted_bank.clean[0, shock_idx + 20 :]
            )
        )
        assert post_mean_abs > 1e-3

    def test_public_shock_types_are_exported(self):
        event = ShockEvent.state_perturbation(state_idx=0, eps=0.1, t_start=0.5)
        schedule = ShockSchedule.single(event)
        assert schedule.n_shocks == 1

    def test_public_schedule_accepts_multiple_param_or_state_events(self):
        schedule = ShockSchedule.multiple(
            ShockEvent.state_perturbation(state_idx=0, eps=0.1, t_start=0.2),
            ShockEvent.state_perturbation(state_idx=1, eps=-0.2, t_start=0.8),
        )
        assert schedule.n_shocks == 2
        assert schedule.shock_times == (0.2, 0.8)

    def test_public_schedule_rejects_unsorted_events(self):
        with pytest.raises(ValueError, match="time-ordered"):
            ShockSchedule.multiple(
                ShockEvent.state_perturbation(state_idx=0, eps=0.1, t_start=0.8),
                ShockEvent.state_perturbation(state_idx=1, eps=-0.2, t_start=0.2),
            )

    def test_public_schedule_rejects_trajectory_switch_with_other_events(self):
        with pytest.raises(ValueError, match="TRAJECTORY_SWITCH"):
            ShockSchedule.multiple(
                ShockEvent.trajectory_switch(ic_b=(0.1, 0.2, 0.3), t_start=0.5),
                ShockEvent.state_perturbation(state_idx=0, eps=0.1, t_start=0.8),
            )


class TestPlanValidation:
    def test_plan_rejects_empty_noise_levels(self):
        with pytest.raises(ValueError, match="noise_levels"):
            TitrationPlan(noise_levels=())

    def test_plan_rejects_zero_draw_counts(self):
        with pytest.raises(ValueError):
            TitrationPlan(n_trajectory_draws=0)
        with pytest.raises(ValueError):
            TitrationPlan(n_noise_draws=0)

    def test_plan_accepts_string_draw_design(self):
        plan = TitrationPlan(draw_design="cross_product", n_trajectory_draws=2, n_noise_draws=3)
        assert plan.draw_design is DrawDesign.CROSS_PRODUCT
        assert plan.cell_count_per_sigma(is_stochastic=False) == 6

    def test_plan_cell_count_matches_resolved_design(self):
        deterministic = TitrationPlan(n_trajectory_draws=1, n_noise_draws=4)
        stochastic = TitrationPlan(n_trajectory_draws=3, n_noise_draws=3)
        assert deterministic.cell_count_per_sigma(is_stochastic=False) == 4
        assert stochastic.cell_count_per_sigma(is_stochastic=True) == 3

    def test_plan_rejects_bank_with_too_few_draws(self):
        bank = build_dgp_bank(small_lorenz(), n_trajectory_draws=1, n_noise_draws=1, verbose=False)
        plan = TitrationPlan(
            noise_levels=(0.0,),
            n_trajectory_draws=1,
            n_noise_draws=2,
            draw_design=DrawDesign.NOISE_ONLY,
        )
        with pytest.raises(ValueError, match="noise draws"):
            list(iter_cells(bank, plan))


class TestStageOrder:
    def test_train_std_noise_uses_clean_train_stats_then_recomputes_stats(self):
        values = np.stack(
            [
                np.linspace(0.0, 10.0, 40, dtype=np.float32),
                np.linspace(0.0, 100.0, 40, dtype=np.float32),
            ],
            axis=1,
        )
        bank = build_empirical_dgp_bank(values, name="scaled", noise_seed_base=7)
        partition = PartitionSpec.p70_20_10()
        clean = inject_noise(bank, sigma=0.0, partition=partition)
        noisy = inject_noise(bank, sigma=1.0, sigma_scale="train_std", partition=partition)

        draw = bank.noise_draws[0][: clean.train_values.shape[0]]
        expected_train = clean.train_values + clean.train_stats.std[None, :] * draw
        np.testing.assert_allclose(noisy.train_values, expected_train, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            noisy.train_stats.mean,
            noisy.train_values.mean(axis=0),
            rtol=1e-5,
            atol=1e-5,
        )
        assert not np.allclose(noisy.train_stats.mean, clean.train_stats.mean)


# Exact semantic ports of the six Rust `dgp_bank::ar1_tests` functions.  The
# random stream is intentionally NumPy-owned for the `serie_py` engine; these
# assertions preserve the scientific process, not Rust PCG byte identity.
def _ar1_draws(phi: float) -> np.ndarray:
    innovations = np.random.default_rng(4242).standard_normal((1, 60_000, 1)).astype(np.float32)
    process = ObservationNoiseProcess.from_wire(NoiseKind.GAUSSIAN, phi)
    return impose_temporal_dependence(innovations, process).reshape(-1)


def test_ar1_preserves_unit_variance_at_every_phi():
    for phi in (0.0, 0.5, 0.9, -0.5):
        values = _ar1_draws(phi).astype(np.float64)
        variance = float(values.var())
        effective = values.size * (1.0 - phi) / (1.0 + phi)
        tolerance = 4.0 * np.sqrt(2.0 / effective)
        assert abs(variance - 1.0) < tolerance, (
            f"phi={phi} gave variance {variance}, outside {tolerance:.4f} of 1"
        )


def test_ar1_lag_one_autocorrelation_equals_phi():
    for phi in (0.5, 0.9, -0.5):
        values = _ar1_draws(phi).astype(np.float64)
        centred = values - values.mean()
        measured = float(np.mean(centred[:-1] * centred[1:]) / np.mean(centred * centred))
        assert abs(measured - phi) < 0.02, f"phi={phi} measured lag-1 autocorrelation {measured}"


def test_ar1_phi_zero_is_bitwise_identical_to_the_plain_gaussian_draw():
    plain = np.random.default_rng(99).standard_normal((2, 128, 3)).astype(np.float32)
    iid = ObservationNoiseProcess.from_wire(NoiseKind.GAUSSIAN, 0.0)
    filtered = impose_temporal_dependence(plain, iid)
    np.testing.assert_array_equal(plain, filtered)


def test_ar1_process_constructor_refuses_invalid_wire_combinations():
    with pytest.raises(ValueError, match="requires gaussian"):
        ObservationNoiseProcess.from_wire(NoiseKind.STUDENT_T3, 0.5)
    with pytest.raises(ValueError, match="requires gaussian"):
        ObservationNoiseProcess.from_wire(NoiseKind.LAPLACE, -0.5)
    with pytest.raises(ValueError, match=r"lie in \(-1, 1\)"):
        ObservationNoiseProcess.from_wire(NoiseKind.GAUSSIAN, 1.0)
    with pytest.raises(ValueError, match="finite"):
        ObservationNoiseProcess.from_wire(NoiseKind.GAUSSIAN, float("nan"))

    iid = ObservationNoiseProcess.from_wire(NoiseKind.STUDENT_T3, 0.0)
    assert iid.innovation_kind is NoiseKind.STUDENT_T3
    assert iid.temporal_process is NoiseTemporalProcess.IID


def test_ar1_metadata_is_explicit_and_iid_metadata_stays_legacy_compatible():
    iid = ObservationNoiseProcess.from_wire(NoiseKind.GAUSSIAN, 0.0)
    assert iid.artifact_metadata() == {}

    ar1 = ObservationNoiseProcess.from_wire(NoiseKind.GAUSSIAN, 0.9)
    assert ar1.artifact_metadata() == {
        "noise_temporal_process": "ar1",
        "noise_ar1_phi": 0.9,
    }


def test_ar1_filters_each_channel_independently():
    innovations = np.array(
        [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]],
        dtype=np.float32,
    )
    process = ObservationNoiseProcess.from_wire(NoiseKind.GAUSSIAN, 0.5)
    filtered = impose_temporal_dependence(innovations, process)
    np.testing.assert_array_equal(filtered[0, :, 1], np.zeros(4, dtype=np.float32))
    assert not np.array_equal(filtered[0, :, 0], innovations[0, :, 0])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
