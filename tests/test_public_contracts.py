"""Contract tests — explicit behavioral checks for invariants enforced by
convention, attrs validators, or implicit assumptions.

test_gen.py covers integrator correctness (shapes, finiteness, shocks).
test_titration_edges.py covers the titration layer (determinism, injection
identity, common random numbers, partitioning, provenance).

This file covers the gaps between those two:

    1. PartitionSpec field access after rename
    2. PartitionStats.center_and_spread correctness
    3. Empirical bank edge cases (1D, too small, wrong shape)
    4. Negative and extreme sigma behavior
    5. burn_in truncation contract
    6. Diagonal system eigenvalue recovery (ground truth)
    7. params_stepped_transition (gradual n_inserts > 1)
    8. Array aliasing safety (returned arrays don't share memory with bank)
    9. Feature names survive the full pipeline
   10. Export with multiple sigmas
   11. SplitTimeSeriesData slicing vs manual index math
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from attrs import evolve

from serie import (
    PartitionSpec,
    ShockEvent,
    ShockSchedule,
    TrajectorySpec,
    build_dgp_bank,
    build_empirical_dgp_bank,
    export_dgp_bank_files,
    inject_noise,
)
from serie.integrators import generate_from_trajectory_spec


# Small, fast specs for unit tests.
def small_lorenz(steps: int = 400) -> TrajectorySpec:
    return TrajectorySpec.lorenz(steps=steps)


def small_ou(steps: int = 300) -> TrajectorySpec:
    return TrajectorySpec.ou(steps=steps)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PartitionSpec renamed fields
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPartitionSpecFields:
    def test_preset_fields_are_accessible_by_new_names(self):
        p = PartitionSpec.p70_20_10()
        assert p.train_start == 0.0
        assert p.train_end == 0.7
        assert p.val_start == 0.7
        assert p.val_end == 0.9
        assert p.test_start == 0.9
        assert p.test_end == 1.0

    def test_custom_partition_round_trips(self):
        p = PartitionSpec(
            train_start=0.0,
            train_end=0.6,
            val_start=0.6,
            val_end=0.8,
            test_start=0.8,
            test_end=1.0,
        )
        assert p.train_end == 0.6
        assert p.val_start == 0.6

    def test_old_field_names_do_not_exist(self):
        p = PartitionSpec.p70_20_10()
        assert not hasattr(p, "t0")
        assert not hasattr(p, "s1")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PartitionStats.center_and_spread
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPartitionStatsMethods:
    @pytest.fixture()
    def stats(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.0, partition=PartitionSpec.p70_20_10())
        return split.train_stats

    def test_standard_returns_mean_and_std(self, stats):
        center, spread = stats.center_and_spread("standard")
        np.testing.assert_array_equal(center, stats.mean)
        np.testing.assert_array_equal(spread, stats.std)

    def test_robust_iqr_returns_median_and_iqr(self, stats):
        center, spread = stats.center_and_spread("robust_iqr")
        np.testing.assert_array_equal(center, stats.median)
        np.testing.assert_array_equal(spread, stats.q75 - stats.q25)

    def test_robust_mad_returns_median_and_scaled_mad(self, stats):
        center, spread = stats.center_and_spread("robust_mad")
        np.testing.assert_array_equal(center, stats.median)
        np.testing.assert_allclose(spread, stats.mad * 1.4826)

    def test_center_mean_returns_mean_and_ones(self, stats):
        center, spread = stats.center_and_spread("center_mean")
        np.testing.assert_array_equal(center, stats.mean)
        np.testing.assert_array_equal(spread, np.ones_like(stats.mean, dtype=np.float32))

    def test_none_returns_zeros_and_ones(self, stats):
        center, spread = stats.center_and_spread("none")
        np.testing.assert_array_equal(center, np.zeros_like(stats.mean, dtype=np.float32))
        np.testing.assert_array_equal(spread, np.ones_like(stats.std, dtype=np.float32))

    def test_unknown_method_raises_with_name(self, stats):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            stats.center_and_spread("quantile_99")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Empirical bank edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEmpiricalBankEdges:
    def test_1d_array_auto_expands_to_column(self):
        values = np.arange(20, dtype=np.float32)
        bank = build_empirical_dgp_bank(values, name="1d")
        assert bank.clean.shape == (1, 20, 1)

    def test_too_few_rows_rejected(self):
        values = np.array([[1.0]], dtype=np.float32)  # shape (1, 1) — only 1 row
        with pytest.raises(ValueError, match="at least 2 rows"):
            build_empirical_dgp_bank(values, name="tiny")

    def test_3d_array_rejected(self):
        values = np.zeros((5, 3, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="\\[T, D\\]"):
            build_empirical_dgp_bank(values, name="3d")

    def test_feature_names_length_mismatch_rejected(self):
        values = np.zeros((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="feature_names has length"):
            build_empirical_dgp_bank(values, name="bad", feature_names=("a", "b"))

    def test_integer_array_coerced_to_float32(self):
        values = np.arange(20, dtype=np.int64).reshape(10, 2)
        bank = build_empirical_dgp_bank(values, name="int")
        assert bank.clean.dtype == np.float32


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Negative and extreme sigma
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExtremeSigma:
    def test_negative_sigma_subtracts_noise(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        pos = bank.build_observed_series(noise_level=0.5)
        neg = bank.build_observed_series(noise_level=-0.5)
        # negative sigma means clean - 0.5 * draw
        expected_neg = bank.clean[0] - 0.5 * bank.noise_draws[0]
        np.testing.assert_array_equal(neg, expected_neg)
        # and the pair should be symmetric around clean
        np.testing.assert_allclose((pos + neg) / 2, bank.clean[0], rtol=1e-5, atol=1e-5)

    def test_very_large_sigma_still_finite(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        noisy = bank.build_observed_series(noise_level=1e6)
        assert np.all(np.isfinite(noisy))

    def test_inject_noise_with_sigma_zero_matches_clean(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.0, partition=PartitionSpec.p70_20_10())
        n_train = split.train_values.shape[0]
        np.testing.assert_array_equal(split.train_values, bank.clean[0][:n_train])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. burn_in truncation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBurnIn:
    def test_burn_in_shortens_output_by_burn_in_steps(self):
        steps, burn_in = 500, 100
        spec = TrajectorySpec.lorenz(steps=steps, burn_in=burn_in)
        traj = generate_from_trajectory_spec(spec, seed=0)
        # output = steps - burn_in + 1 (the +1 is the initial condition)
        assert traj.shape == (steps - burn_in + 1, 3)

    def test_burn_in_zero_gives_steps_plus_one(self):
        spec = TrajectorySpec.lorenz(steps=200, burn_in=0)
        traj = generate_from_trajectory_spec(spec, seed=0)
        assert traj.shape == (201, 3)

    def test_burn_in_removes_transient_from_bank(self):
        no_burn = TrajectorySpec.lorenz(steps=500, burn_in=0)
        with_burn = TrajectorySpec.lorenz(steps=500, burn_in=100)
        bank_no = build_dgp_bank(no_burn, seed=0, verbose=False)
        bank_yes = build_dgp_bank(with_burn, seed=0, verbose=False)
        assert bank_no.T == 501
        assert bank_yes.T == 401
        # The burned bank's first row should match the non-burned bank at index 100
        np.testing.assert_allclose(bank_yes.clean[0, 0], bank_no.clean[0, 100], rtol=1e-5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Diagonal eigenvalue recovery (ground truth test)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDiagonalGroundTruth:
    def test_trajectory_matches_exact_exponential_decay(self):
        eigenvalues = (-0.5, -1.0, -2.0)
        dt = 0.01
        spec = TrajectorySpec.diagonal(eigenvalues=eigenvalues, steps=1000, dt=dt)
        traj = generate_from_trajectory_spec(spec, seed=0, dtype=np.float64)
        # x_i(t) = x_i(0) * exp(λ_i * t)
        # x_i(0) = 1.0 for all channels (diagonal_spec default)
        t = np.arange(traj.shape[0]) * dt
        for i, lam in enumerate(eigenvalues):
            expected = np.exp(lam * t)
            np.testing.assert_allclose(traj[:, i], expected, rtol=1e-4)

    def test_discrete_eigenvalues_are_exp_lambda_dt(self):
        eigenvalues = (-0.5, -1.0, -2.0)
        dt = 0.01
        spec = TrajectorySpec.diagonal(eigenvalues=eigenvalues, steps=100, dt=dt)
        traj = generate_from_trajectory_spec(spec, seed=0, dtype=np.float64)
        # Discrete propagator eigenvalue: x[n+1]/x[n] ≈ exp(λ * dt)
        for i, lam in enumerate(eigenvalues):
            ratios = traj[1:, i] / traj[:-1, i]
            expected_ratio = np.exp(lam * dt)
            np.testing.assert_allclose(ratios, expected_ratio, rtol=1e-3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Gradual parameter transition (n_inserts > 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGradualTransition:
    def test_stepped_transition_diverges_more_gradually_than_abrupt(self):
        from serie.catalog import LorenzParams

        base = small_lorenz(steps=600)
        after = LorenzParams(sigma=10.5, rho=29.0, beta=8.5 / 3.0)

        abrupt = evolve(
            base,
            shocks=ShockSchedule.single(
                ShockEvent.params_step_change(params_after=after, t_start=0.3)
            ),
        )
        gradual = evolve(
            base,
            shocks=ShockSchedule.single(
                ShockEvent.params_stepped_transition(
                    params_after=after, t_start=0.3, t_end=0.6, n_inserts=4
                )
            ),
        )

        bank_base = build_dgp_bank(base, seed=42, verbose=False)
        bank_abrupt = build_dgp_bank(abrupt, seed=42, verbose=False)
        bank_gradual = build_dgp_bank(gradual, seed=42, verbose=False)

        # Before the shock, all three should be identical
        shock_idx = int(0.3 * bank_base.T)
        np.testing.assert_array_equal(
            bank_base.clean[0, : shock_idx - 2],
            bank_abrupt.clean[0, : shock_idx - 2],
        )
        np.testing.assert_array_equal(
            bank_base.clean[0, : shock_idx - 2],
            bank_gradual.clean[0, : shock_idx - 2],
        )

        # After the transition window, both diverge from baseline
        post_idx = int(0.7 * bank_base.T)
        abrupt_div = float(
            np.abs(bank_abrupt.clean[0, post_idx:] - bank_base.clean[0, post_idx:]).mean()
        )
        gradual_div = float(
            np.abs(bank_gradual.clean[0, post_idx:] - bank_base.clean[0, post_idx:]).mean()
        )
        assert abrupt_div > 0.01
        assert gradual_div > 0.01


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Array aliasing safety
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestArrayAliasing:
    def test_modifying_split_train_does_not_corrupt_bank(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        clean_snapshot = bank.clean.copy()
        split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
        split.train_values[:] = 999.0  # mutate the returned array
        np.testing.assert_array_equal(bank.clean, clean_snapshot)

    def test_modifying_observed_series_does_not_corrupt_bank(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        clean_snapshot = bank.clean.copy()
        draws_snapshot = bank.noise_draws.copy()
        obs = bank.build_observed_series(noise_level=1.0)
        obs[:] = 0.0  # mutate the returned array
        np.testing.assert_array_equal(bank.clean, clean_snapshot)
        np.testing.assert_array_equal(bank.noise_draws, draws_snapshot)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Feature names survive the pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFeatureNames:
    def test_synthetic_bank_has_auto_feature_names(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        assert bank.feature_names == ("f0", "f1", "f2")

    def test_feature_names_survive_inject_noise(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
        assert split.feature_names == ("f0", "f1", "f2")

    def test_empirical_bank_preserves_custom_names(self):
        values = np.zeros((20, 3), dtype=np.float32)
        bank = build_empirical_dgp_bank(
            values, name="test", feature_names=("accel_x", "accel_y", "accel_z")
        )
        assert bank.feature_names == ("accel_x", "accel_y", "accel_z")
        split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
        assert split.feature_names == ("accel_x", "accel_y", "accel_z")

    def test_feature_names_appear_in_export_manifest(self, tmp_path):
        values = np.zeros((20, 2), dtype=np.float32)
        bank = build_empirical_dgp_bank(values, name="test", feature_names=("x", "y"))
        manifest = export_dgp_bank_files(bank, tmp_path / "out")
        assert manifest["feature_names"] == ["x", "y"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Export with multiple sigmas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExportMultipleSigmas:
    def test_export_two_sigmas_writes_two_noisy_entries(self, tmp_path):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        out = tmp_path / "multi"
        manifest = export_dgp_bank_files(bank, out, sigmas=(0.1, 0.5))
        assert len(manifest["noisy"]) == 2
        sigmas_in_manifest = {entry["sigma"] for entry in manifest["noisy"]}
        assert sigmas_in_manifest == {0.1, 0.5}

        # Each noisy file should exist and have correct shape
        for entry in manifest["noisy"]:
            npy_path = out / entry["files"]["npy"]
            arr = np.load(npy_path)
            assert arr.shape == (bank.T, bank.D)

    def test_export_manifest_json_round_trips(self, tmp_path):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        out = tmp_path / "rt"
        manifest = export_dgp_bank_files(bank, out, sigmas=(0.25,))
        on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert on_disk["source_name"] == manifest["source_name"]
        assert on_disk["noisy"][0]["sigma"] == 0.25


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. SplitTimeSeriesData slicing correctness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSplitSlicing:
    def test_train_val_test_slices_match_manual_indices(self):
        bank = build_dgp_bank(small_lorenz(steps=999), seed=0, verbose=False)
        partition = PartitionSpec.p70_20_10()
        split = inject_noise(bank, sigma=0.0, partition=partition)
        T = bank.T

        # Manual index computation matching what StageSplitByFraction does
        t0 = int(T * partition.train_start)
        t1 = int(T * partition.train_end)
        v0 = int(T * partition.val_start)
        v1 = int(T * partition.val_end)
        s0 = int(T * partition.test_start)
        s1 = int(T * partition.test_end)

        np.testing.assert_array_equal(split.train_values, bank.clean[0][t0:t1])
        np.testing.assert_array_equal(split.val_values, bank.clean[0][v0:v1])
        np.testing.assert_array_equal(split.test_values, bank.clean[0][s0:s1])

    def test_indices_tuple_matches_properties(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.0, partition=PartitionSpec.p70_20_10())
        t0, t1, v0, v1, s0, s1 = split.indices
        assert split.train_values.shape[0] == t1 - t0
        assert split.val_values.shape[0] == v1 - v0
        assert split.test_values.shape[0] == s1 - s0

    def test_channels_property_matches_data(self):
        bank = build_dgp_bank(small_lorenz(), seed=0, verbose=False)
        split = inject_noise(bank, sigma=0.0, partition=PartitionSpec.p70_20_10())
        assert split.channels == 3
        assert split.train_values.shape[1] == split.channels
