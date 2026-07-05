"""
test_gen.py — Verify generator correctness for the pure-Python SERIE package.

Refactoring safety net: the checks below pin generator shape, finiteness,
shock behavior, and a few known qualitative or moment properties of the
reference dynamics.

Test matrix:
    ODE systems:  lorenz, rossler, hyper_rossler, chua, duffing, vanderpol, lorenz96
    Map systems:  henon, logistic
    Shock modes:  no_shock, param_shock, state_shock

"""

from __future__ import annotations

import numpy as np
import pytest
from attrs import evolve

from serie.catalog import (
    SYSTEMS,
    ODESystem,
    LorenzParams,
    RosslerParams,
    ChuaParams,
    VanderPolParams,
    DuffingParams,
    HyperRosslerParams,
    Lorenz96Params,
    HenonParams,
    LogisticParams,
    OUParams,
    DiagonalParams,
)
from serie import ShockEvent, ShockSchedule, TrajectorySpec
from serie.integrators import (
    StepShockSchedule,
    _params_as_args,
    _run_map_py,
    _run_ode_py,
    generate_from_trajectory_spec,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEPS = 10_000
DT = 0.01


def _np_ode(system_name, x0, steps, dt, params, schedule=None):
    spec = SYSTEMS[system_name]
    return _run_ode_py(
        f=spec.f_py,
        x0=x0,
        steps=steps,
        dt=dt,
        t0=0.0,
        params=params,
        method="rk4",
        time_dependent=spec.time_dependent,
        schedule=schedule,
    )


def _rk4_step(f, x: np.ndarray, dt: float, params) -> np.ndarray:
    args = _params_as_args(params)
    k1 = f(x, 0.0, *args)
    k2 = f(x + 0.5 * dt * k1, 0.0, *args)
    k3 = f(x + 0.5 * dt * k2, 0.0, *args)
    k4 = f(x + dt * k3, 0.0, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _estimate_lorenz_largest_lyapunov(
    *, steps=20_000, burn_steps=1_000, dt=0.01, delta0=1e-7
) -> float:
    # Largest Lyapunov exponent = exponential rate at which two trajectories
    # starting delta0 apart pull apart. Positive => chaos. Benettin method:
    # advance both, measure growth, then RESCALE the separation back to delta0
    # so it never saturates at the attractor's size — accumulate log-growth.
    f = SYSTEMS["lorenz"].f_py
    params = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    x = np.array([1.0, 0.98, 1.1], dtype=np.float64)
    direction = np.array([0.3, -0.4, 0.5], dtype=np.float64)
    direction *= delta0 / np.linalg.norm(direction)  # tiny initial separation
    y = x + direction
    log_growth, n = 0.0, 0
    for i in range(steps):
        x = _rk4_step(f, x, dt, params)
        y = _rk4_step(f, y, dt, params)
        diff = y - x
        dist = float(np.linalg.norm(diff))  # how far they've drifted
        if i >= burn_steps:  # skip the transient
            log_growth += np.log(dist / delta0)  # one step of exp-growth
            n += 1
        y = x + (delta0 / dist) * diff  # renormalize back to delta0
    return log_growth / (n * dt)  # average rate per unit time
    # classic Lorenz ~ 0.9056; ours ~0.914 (finite-dt bias is normal)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pure-Python integration tests (always run)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTrajectorySpecValidation:
    """Spec construction should reject system/method mismatches early."""

    def test_sde_rejects_rk4_method(self):
        with pytest.raises(ValueError, match="ou expects method 'euler'"):
            TrajectorySpec(
                system=ODESystem.OU,
                params=OUParams(theta=0.2, mu=0.0, sigma_noise=0.3),
                dt=0.5,
                steps=100,
                burn_in=0,
                dim=1,
                method="rk4",
                x0=(0.0,),
            )

    def test_ou_preset_sets_euler_method(self):
        assert TrajectorySpec.ou(steps=100).method == "euler"


class TestPythonODE:
    """Test that Python integrators produce finite, correctly-shaped trajectories."""

    def test_lorenz_shape_and_finite(self):
        x0 = np.array([1.0, 0.98, 1.1])
        params = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        traj = _np_ode("lorenz", x0, STEPS, DT, params)
        assert traj.shape == (STEPS + 1, 3)
        assert np.all(np.isfinite(traj))

    def test_lorenz_ic_preserved(self):
        x0 = np.array([1.0, 0.98, 1.1])
        params = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        traj = _np_ode("lorenz", x0, 100, DT, params)
        np.testing.assert_allclose(traj[0], x0, atol=1e-15)

    def test_rossler_finite(self):
        x0 = np.array([1.0, 1.0, 1.0])
        params = RosslerParams(a=0.2, b=0.2, c=5.7)
        traj = _np_ode("rossler", x0, STEPS, DT, params)
        assert np.all(np.isfinite(traj))

    def test_chua_finite(self):
        x0 = np.array([0.1, 0.0, 0.0])
        params = ChuaParams(alpha=15.6, beta=28.0, m1_inner_le1=-8.0 / 7.0, m0_outer_gt1=-5.0 / 7.0)
        traj = _np_ode("chua", x0, STEPS, 0.005, params)
        assert np.all(np.isfinite(traj))

    def test_lorenz96_variable_dim(self):
        dim = 6
        x0 = np.ones(dim)
        x0[0] += 0.01
        params = Lorenz96Params(forcing=8.0)
        traj = _np_ode("lorenz96", x0, STEPS, 0.007, params)
        assert traj.shape == (STEPS + 1, dim)
        assert np.all(np.isfinite(traj))

    def test_vanderpol_finite(self):
        x0 = np.array([1.0, 0.0])
        params = VanderPolParams(mu=2.0)
        traj = _np_ode("vanderpol", x0, STEPS, DT, params)
        assert np.all(np.isfinite(traj))

    def test_duffing_finite(self):
        x0 = np.array([0.1, 0.0])
        params = DuffingParams(alpha=1.0, beta=-1.0, delta=0.2, gamma=0.65, omega=1.2)
        traj = _np_ode("duffing", x0, 2000, 0.002, params)
        assert np.all(np.isfinite(traj))

    def test_hyper_rossler_finite(self):
        x0 = np.array([0.1, 0.1, 0.1, 0.1])
        params = HyperRosslerParams(a=0.25, b=3.0, c=0.5, d=0.05)
        traj = _np_ode("hyper_rossler", x0, 2000, 0.002, params)
        assert np.all(np.isfinite(traj))

    def test_lorenz_largest_lyapunov_matches_classic_attractor(self):
        lyap = _estimate_lorenz_largest_lyapunov()
        assert 0.75 < lyap < 1.05

    def test_burn_in_removes_raw_prefix_exactly(self):
        full_spec = TrajectorySpec.diagonal(
            eigenvalues=(-0.25, -0.5),
            steps=40,
            dt=0.01,
            burn_in=0,
        )
        burn_spec = TrajectorySpec.diagonal(
            eigenvalues=(-0.25, -0.5),
            steps=40,
            dt=0.01,
            burn_in=7,
        )
        full = generate_from_trajectory_spec(full_spec, seed=0)
        burned = generate_from_trajectory_spec(burn_spec, seed=0)
        np.testing.assert_array_equal(burned, full[7:])

    def test_diagonal_discrete_ratios_match_exp_lambda_dt(self):
        eigenvalues = (-0.2, -0.5, 0.1)
        dt = 1e-3
        spec = TrajectorySpec.diagonal(eigenvalues=eigenvalues, steps=4, dt=dt, burn_in=0)
        traj = generate_from_trajectory_spec(spec, seed=0).astype(np.float64)
        observed = traj[1] / traj[0]
        expected = np.exp(np.array(eigenvalues, dtype=np.float64) * dt)
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)


class TestPythonMap:
    def test_henon_bounded(self):
        x0 = np.array([0.1, 0.1])
        params = HenonParams(a=1.4, b=0.3)
        traj = _run_map_py(SYSTEMS["henon"].f_py, x0, STEPS, params, None)
        assert traj.shape == (STEPS + 1, 2)
        assert np.all(np.isfinite(traj))
        assert np.all(np.abs(traj) < 3.0)

    def test_logistic_in_unit_interval(self):
        x0 = np.array([0.5])
        params = LogisticParams(r=3.9)
        traj = _run_map_py(SYSTEMS["logistic"].f_py, x0, STEPS, params, None)
        assert traj.shape == (STEPS + 1, 1)
        assert np.all(traj >= 0.0) and np.all(traj <= 1.0)


class TestPythonShock:
    def test_lorenz_param_shock_differs(self):
        x0 = np.array([1.0, 0.98, 1.1])
        params = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        params_after = LorenzParams(sigma=10.1, rho=28.1, beta=8.1 / 3.0)
        shock_step = STEPS // 2

        baseline = _np_ode("lorenz", x0, STEPS, DT, params)
        schedule = StepShockSchedule.param_shock(shock_step, params, params_after)
        shocked = _np_ode("lorenz", x0, STEPS, DT, params, schedule=schedule)

        # Pre-shock identical
        np.testing.assert_array_equal(baseline[:shock_step], shocked[:shock_step])
        # Post-shock differs
        assert np.max(np.abs(baseline[shock_step + 10 :] - shocked[shock_step + 10 :])) > 1e-6

    def test_lorenz_state_shock_differs(self):
        x0 = np.array([1.0, 0.98, 1.1])
        params = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        shock_step = STEPS // 2

        baseline = _np_ode("lorenz", x0, STEPS, DT, params)
        schedule = StepShockSchedule.state_shock(shock_step, state_idx=0, eps=5.0, params=params)
        shocked = _np_ode("lorenz", x0, STEPS, DT, params, schedule=schedule)

        np.testing.assert_array_equal(baseline[:shock_step], shocked[:shock_step])
        assert np.max(np.abs(baseline[shock_step + 1 :] - shocked[shock_step + 1 :])) > 1.0

    def test_public_multi_state_schedule_applies_both_perturbations(self):
        base = TrajectorySpec.lorenz(steps=500, burn_in=0)
        source = evolve(
            base,
            shocks=ShockSchedule.multiple(
                ShockEvent.state_perturbation(state_idx=0, eps=0.5, t_start=0.2),
                ShockEvent.state_perturbation(state_idx=1, eps=-0.75, t_start=0.6),
            ),
        )

        baseline = generate_from_trajectory_spec(base, seed=0)
        shocked = generate_from_trajectory_spec(source, seed=0)

        assert source.tag == "lorenz_2shocks"
        np.testing.assert_array_equal(baseline[:100], shocked[:100])
        assert np.max(np.abs(baseline[105:] - shocked[105:])) > 1e-4
        assert np.max(np.abs(baseline[305:] - shocked[305:])) > 1e-3

    def test_step_shock_schedule_param_transition_interpolates_params(self):
        before = DiagonalParams(eigenvalues=(0.0, 2.0))
        after = DiagonalParams(eigenvalues=(2.0, 4.0))
        schedule = StepShockSchedule.param_transition(
            before=before,
            after=after,
            start_step=2,
            end_step=6,
            n_inserts=2,
        )

        assert [event.step for event in schedule.events] == [4, 6]
        assert [event.param_set_idx for event in schedule.events] == [1, 2]
        assert schedule.param_sets == (
            before,
            DiagonalParams(eigenvalues=(1.0, 3.0)),
            after,
        )

    def test_public_params_stepped_transition_changes_dynamics_gradually(self):
        base = TrajectorySpec.diagonal(eigenvalues=(0.0,), steps=20, dt=0.05, burn_in=0)
        transitioned = evolve(
            base,
            shocks=ShockSchedule.single(
                ShockEvent.params_stepped_transition(
                    params_after=DiagonalParams(eigenvalues=(2.0,)),
                    t_start=0.2,
                    t_end=0.6,
                    n_inserts=2,
                )
            ),
        )

        baseline = generate_from_trajectory_spec(base, seed=0)
        shocked = generate_from_trajectory_spec(transitioned, seed=0)
        np.testing.assert_array_equal(baseline[:8], shocked[:8])
        assert shocked[-1, 0] > baseline[-1, 0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Preset construction tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPresets:
    """Verify that catalog preset functions produce valid TrajectorySpecs."""

    def test_all_active_presets_construct(self):
        from serie.catalog import (
            lorenz_spec,
            rossler_spec,
            chua_spec,
            lorenz96_spec,
            ou_spec,
            doublewell_spec,
            lg_ssm_spec,
            lg_ssm_small_spec,
            slds_spec,
            seasonal_ar_spec,
            garch_spec,
            ks_spec,
            diagonal_spec,
        )

        for fn in [
            lorenz_spec,
            rossler_spec,
            chua_spec,
            lorenz96_spec,
            ou_spec,
            doublewell_spec,
            lg_ssm_spec,
            lg_ssm_small_spec,
            slds_spec,
            seasonal_ar_spec,
            garch_spec,
            ks_spec,
            diagonal_spec,
        ]:
            spec = fn()
            assert spec.steps > 0
            assert spec.dim > 0
            assert spec.dt > 0

    def test_shock_presets_construct(self):
        from serie.catalog import (
            lorenz_param_shock_spec,
            lorenz_state_shock_spec,
            lorenz_switch_spec,
            rossler_param_shock_spec,
            chua_param_shock_spec,
            chua_switch_spec,
            ou_param_shock_spec,
            doublewell_param_shock_spec,
            doublewell_switch_spec,
            slds_param_shock_spec,
            slds_switch_spec,
            seasonal_ar_param_shock_spec,
            garch_param_shock_spec,
            ks_param_shock_spec,
        )

        shock_fns = [
            lorenz_param_shock_spec,
            lorenz_state_shock_spec,
            lorenz_switch_spec,
            rossler_param_shock_spec,
            chua_param_shock_spec,
            chua_switch_spec,
            ou_param_shock_spec,
            doublewell_param_shock_spec,
            doublewell_switch_spec,
            slds_param_shock_spec,
            slds_switch_spec,
            seasonal_ar_param_shock_spec,
            garch_param_shock_spec,
            ks_param_shock_spec,
        ]
        for fn in shock_fns:
            spec = fn()
            assert spec.shocks is not None
            assert not spec.shocks.is_stationary

    def test_trajectory_spec_forwarding_matches_catalog(self):
        """TrajectorySpec.lorenz() should produce the same result as lorenz_spec()."""
        from serie.specs import TrajectorySpec
        from serie.catalog import lorenz_spec

        a = TrajectorySpec.lorenz()
        b = lorenz_spec()
        assert a.system == b.system
        assert a.dt == b.dt
        assert a.steps == b.steps
        assert a.dim == b.dim


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# generate_from_trajectory_spec end-to-end
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGenerateFromTrajectorySpec:
    def test_lorenz_e2e(self):
        from serie.catalog import lorenz_spec
        from serie.integrators import generate_from_trajectory_spec

        spec = lorenz_spec(steps=1000)
        traj = generate_from_trajectory_spec(spec, seed=0)
        assert traj.shape == (1001, 3)
        assert traj.dtype == np.float32
        assert np.all(np.isfinite(traj))

    def test_ou_e2e(self):
        from serie.catalog import ou_spec
        from serie.integrators import generate_from_trajectory_spec

        spec = ou_spec(steps=1000)
        traj = generate_from_trajectory_spec(spec, seed=42)
        assert traj.shape == (1001, 1)
        assert np.all(np.isfinite(traj))

    def test_ou_stationary_variance_matches_euler_maruyama_scheme(self):
        from serie.catalog import ou_spec
        from serie.integrators import generate_from_trajectory_spec

        # OU is an SDE: dx = -theta*x*dt + sigma*dW. Euler-Maruyama discretizes it as
        # an AR(1):  x[n+1] = (1 - theta*dt) x[n] + sigma*sqrt(dt)*eps[n].
        # "Correct" for a stochastic system = the right LONG-RUN distribution.
        # AR(1) stationary variance = b^2/(1-a^2) with a=1-theta*dt, b=sigma*sqrt(dt):
        spec = ou_spec(steps=30_000, burn_in=5_000)  # burn_in: reach stationarity
        traj = generate_from_trajectory_spec(spec, seed=42, dtype=np.float64)[:, 0]
        p = spec.params
        expected = (p.sigma_noise**2 * spec.dt) / (1.0 - (1.0 - p.theta * spec.dt) ** 2)
        assert abs(float(traj.mean()) - p.mu) < 0.05  # mean-reverts to mu
        assert 0.85 * expected < float(traj.var()) < 1.15 * expected  # right spread

    def test_doublewell_e2e(self):
        from serie.catalog import doublewell_spec
        from serie.integrators import generate_from_trajectory_spec

        spec = doublewell_spec(steps=1000)
        traj = generate_from_trajectory_spec(spec, seed=42)
        assert traj.shape == (1001, 1)
        assert np.all(np.isfinite(traj))
        assert float(traj.std()) > 0.05

    def test_seasonal_ar_e2e(self):
        from serie.catalog import seasonal_ar_spec
        from serie.integrators import generate_from_trajectory_spec

        spec = seasonal_ar_spec(steps=1000)
        traj = generate_from_trajectory_spec(spec, seed=42)
        assert traj.shape == (1001, 1)
        assert np.all(np.isfinite(traj))
        assert float(traj.std()) > 0.5

    def test_garch_e2e(self):
        from serie.catalog import garch_spec
        from serie.integrators import generate_from_trajectory_spec

        spec = garch_spec(steps=1000)
        traj = generate_from_trajectory_spec(spec, seed=42)
        assert traj.shape == (1001, 2)
        assert np.all(np.isfinite(traj))
        assert np.all(traj[:, 1] >= 0.0)
        assert float(traj[:, 0].std()) > 0.1

    def test_ks_e2e_nonzero_field(self):
        from serie.catalog import ks_spec
        from serie.integrators import generate_from_trajectory_spec

        # CHG: the preset default is now burn_in=2500; this short e2e case
        # overrides steps, so it must override burn_in too.
        nx = 32
        x = np.arange(nx, dtype=np.float64)
        spec = evolve(
            ks_spec(nx=nx, steps=200, dt=1e-2, burn_in=0),
            x0=tuple(0.1 * np.cos(2.0 * np.pi * x / nx)),
        )
        traj = generate_from_trajectory_spec(spec, seed=42, dtype=np.float64)
        assert traj.shape == (201, nx)
        assert np.all(np.isfinite(traj))
        assert float(traj.std()) > 0.01

    def test_ks_preset_seed_field_is_not_degenerate(self):
        # Pins the 2026-07-01 scientific fix: x0=zeros is a fixed point of the
        # KS flow, so the old preset generated an identically zero series. The
        # default seed field must be non-zero with the Kassam–Trefethen shape.
        from serie.catalog import ks_spec

        spec = ks_spec()
        x0 = np.asarray(spec.x0)
        assert float(np.abs(x0).max()) > 0.5
        assert spec.burn_in > 0

    def test_slds_e2e(self):
        from serie.catalog import slds_spec
        from serie.integrators import generate_from_trajectory_spec

        spec = slds_spec(steps=1000)
        traj = generate_from_trajectory_spec(spec, seed=42)
        assert traj.shape == (1001, 1)
        assert np.all(np.isfinite(traj))
        assert float(traj.std()) > 0.1

    def test_slds_alternates_between_regime_noise_scales(self):
        from serie.catalog import slds_spec
        from serie.integrators import generate_from_trajectory_spec

        # SLDS = a hidden chain switches which linear dynamics is active each step.
        # Rig it to isolate the SWITCHING: regime 1 is frozen at exactly 0 (A1=Q1=0),
        # regime 2 is pure noise (Q2=1), and p11=p22=0 forces strict 1,2,1,2 alternation.
        # If switching works, exactly half the points are bit-exact zero.
        spec = slds_spec(A1=0.0, Q1=0.0, A2=0.0, Q2=1.0, p11=0.0, p22=0.0, steps=2000)
        traj = generate_from_trajectory_spec(spec, seed=42, dtype=np.float64)[:, 0]
        zero_fraction = float(np.isclose(traj[1:], 0.0, atol=0.0).mean())
        assert zero_fraction == 0.5

    def test_slds_switch_drawn_at_step_t_drives_step_t(self):
        from serie.catalog import slds_spec
        from serie.integrators import generate_from_trajectory_spec

        # Regime-timing pin (user-approved 2026-07-01): the regime drawn at
        # step t drives the dynamics at step t itself, x_t = A_{z_t}·x_{t-1}.
        # Rig: seed chosen so the initial coin flip starts in regime 1
        # (premise asserted below), p11=0 forces an immediate 1→2 transition
        # at t=1, p22=1 keeps regime 2, and Q1=Q2=0 removes noise. New
        # convention: x_1 = A2·x_0 = 0. The pre-fix convention stepped with
        # the pre-transition regime and gave x_1 = A1·x_0 = 0.9.
        seed = 2
        assert np.random.default_rng(seed).random() < 0.5, "premise: initial regime must be 1"

        spec = slds_spec(A1=0.9, Q1=0.0, A2=0.0, Q2=0.0, p11=0.0, p22=1.0, steps=3)
        spec = evolve(spec, x0=(1.0,))
        traj = generate_from_trajectory_spec(spec, seed=seed, dtype=np.float64)[:, 0]

        assert traj[0] == 1.0
        assert traj[1] == 0.0, "switch drawn at step 1 must drive step 1 itself"
        assert np.all(traj[1:] == 0.0)

    def test_param_transition_rejects_more_inserts_than_window_steps(self):
        from serie.catalog import lorenz_spec
        from serie.integrators import generate_from_trajectory_spec

        # A stepped transition whose window collapses to 1 step gap cannot
        # honor n_inserts=5. This must raise (not silently clamp): steps=100,
        # t_start=0.5, t_end=0.5001 → window steps 50..51 → 1 gap < 5.
        spec = evolve(
            lorenz_spec(steps=100),
            shocks=ShockSchedule.single(
                ShockEvent.params_stepped_transition(
                    params_after=LorenzParams(sigma=10.0, rho=35.0, beta=8.0 / 3.0),
                    t_start=0.5,
                    t_end=0.5001,
                    n_inserts=5,
                )
            ),
        )
        with pytest.raises(ValueError, match="asks for 5 scheduled updates"):
            generate_from_trajectory_spec(spec, seed=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LG-SSM scientific validation (roadmap §15.2 / §15.3 / §3.5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLGSSM:
    """LG-SSM exact-discretization family, validated before the Rust mirror.

    The four designed checks:
      (a) stationary covariance of a long trajectory solves the discrete
          Lyapunov equation P = A·P·Aᵀ + Q (roadmap §3.4 / §22.2 state-space);
      (b) the Van Loan Q equals brute-force numerical integration of the
          covariance integral ∫₀^dt e^{Fs}·L Q_c Lᵀ·e^{Fᵀs} ds (§15.2 / §22.2
          continuous-time SDE);
      (c) with ill-conditioned V at FIXED spectral radius, transient
          amplification G_K(A) = max_k ‖Aᵏ‖₂ exceeds the normal case while
          both decay asymptotically (§3.5, transient-growth diagnostic);
      (d) determinism and draw design match the other stochastic sources.
    """

    def test_lg_ssm_e2e(self):
        from serie.catalog import lg_ssm_spec

        spec = lg_ssm_spec(steps=1000)
        traj = generate_from_trajectory_spec(spec, seed=42)
        assert traj.shape == (1001, 8)
        assert np.all(np.isfinite(traj))
        assert float(traj.std()) > 0.1

    def test_lg_ssm_spectral_radius_fixed_across_kappa(self):
        # The controlled non-normality design (§3.5): |eig(A)| = exp(-γ·dt)
        # is independent of kappa_v, so ρ(A) stays pinned while V's
        # conditioning varies over the §3.6 menu.
        from serie.catalog import lg_ssm_discretize_exact, lg_ssm_spec

        for kappa_v in (1.0, 5.0, 20.0, 100.0):
            spec = lg_ssm_spec(kappa_v=kappa_v)
            A, _ = lg_ssm_discretize_exact(spec.params, spec.dt)
            rho = float(np.max(np.abs(np.linalg.eigvals(A))))
            assert abs(rho - 0.95) < 1e-12, f"kappa_v={kappa_v}: rho={rho}"

    def test_lg_ssm_discretization_matches_closed_form_normal_case(self):
        # kappa_v=1 admits closed forms that validate the Van Loan route
        # independently: A = W·e^{B_c·dt}·Wᵀ with e^{B_c·dt} block-diagonal of
        # e^{-γ·dt}·R(ω·dt) rotations, and Q = q_c·(1 - e^{-2γ·dt})/(2γ)·I
        # (each §15.3 block is normal with F + Fᵀ = -2γ·I).
        from serie.catalog import (
            _lg_ssm_mixing_matrix,
            lg_ssm_discretize_exact,
            lg_ssm_spec,
        )

        spec = lg_ssm_spec()
        p = spec.params
        dt = spec.dt
        A, Q = lg_ssm_discretize_exact(p, dt)

        dim = 2 * len(p.gammas)
        e_b_dt = np.zeros((dim, dim))
        for i, (gamma, omega) in enumerate(zip(p.gammas, p.omegas, strict=True)):
            decay = np.exp(-gamma * dt)
            c, s = np.cos(omega * dt), np.sin(omega * dt)
            e_b_dt[2 * i, 2 * i] = decay * c
            e_b_dt[2 * i, 2 * i + 1] = -decay * s
            e_b_dt[2 * i + 1, 2 * i] = decay * s
            e_b_dt[2 * i + 1, 2 * i + 1] = decay * c
        w = _lg_ssm_mixing_matrix(dim)
        a_closed = w @ e_b_dt @ w.T

        gamma = p.gammas[0]
        q_closed = p.q_c * (1.0 - np.exp(-2.0 * gamma * dt)) / (2.0 * gamma) * np.eye(dim)

        assert np.abs(A - a_closed).max() < 1e-12
        assert np.abs(Q - q_closed).max() < 1e-12
        # §2.5 standardization pin: q_c defaults to 2γ so the kappa_v=1
        # stationary covariance is exactly the identity (Q = (1 - ρ²)·I and
        # A·Aᵀ = ρ²·I make P = I solve P = A·P·Aᵀ + Q).
        assert np.abs((A @ A.T + Q) - np.eye(dim)).max() < 1e-12

    def test_lg_ssm_stationary_covariance_matches_discrete_lyapunov(self):
        # Check (a): for stable A, the empirical covariance of a long
        # trajectory matches P solving P = A·P·Aᵀ + Q within sampling
        # tolerance. Run both the normal preset (where P = I by the §2.5
        # standardization) and a non-normal kappa_v=5 case (where P is far
        # from identity).
        from scipy.linalg import solve_discrete_lyapunov

        from serie.catalog import lg_ssm_discretize_exact, lg_ssm_small_spec

        for kappa_v in (1.0, 5.0):
            spec = lg_ssm_small_spec(kappa_v=kappa_v, steps=60_000)
            A, Q = lg_ssm_discretize_exact(spec.params, spec.dt)
            P = solve_discrete_lyapunov(A, Q)
            traj = generate_from_trajectory_spec(spec, seed=42, dtype=np.float64)
            empirical = np.cov(traj.T)
            relative_error = float(np.abs(empirical - P).max() / np.abs(P).max())
            assert relative_error < 0.05, f"kappa_v={kappa_v}: {relative_error}"
            if kappa_v == 1.0:
                assert np.abs(P - np.eye(4)).max() < 1e-12

    def test_lg_ssm_van_loan_q_matches_brute_force_covariance_integral(self):
        # Check (b): Q from the Van Loan block exponential equals direct
        # numerical quadrature of ∫₀^dt e^{Fs}·q_c I·e^{Fᵀs} ds (§15.2) for a
        # small non-normal case.
        from scipy.integrate import simpson
        from scipy.linalg import expm

        from serie.catalog import (
            lg_ssm_discretize_exact,
            lg_ssm_drift_matrix,
            lg_ssm_small_spec,
        )

        spec = lg_ssm_small_spec(kappa_v=5.0)
        p = spec.params
        dt = spec.dt
        _, Q = lg_ssm_discretize_exact(p, dt)

        F = lg_ssm_drift_matrix(p)
        diffusion = p.q_c * np.eye(F.shape[0])
        s_grid = np.linspace(0.0, dt, 2001)
        integrand = np.array([expm(F * s) @ diffusion @ expm(F.T * s) for s in s_grid])
        q_brute_force = simpson(integrand, x=s_grid, axis=0)

        assert np.abs(Q - q_brute_force).max() < 1e-12

    def test_lg_ssm_nonnormal_transient_growth_exceeds_normal_at_fixed_radius(self):
        # Check (c): §3.7 scenario 4 ("transient-risk system: ρ(A)=0.9 but
        # G_50(A) ≫ 1 via non-normal V"). At the same spectral radius, the
        # ill-conditioned-V system amplifies transients while the normal one
        # contracts monotonically (‖Aᵏ‖₂ = ρᵏ); both decay asymptotically.
        from serie.catalog import lg_ssm_discretize_exact, lg_ssm_spec

        def two_norms_of_powers(A: np.ndarray, k_max: int) -> np.ndarray:
            powers = []
            a_k = np.eye(A.shape[0])
            for _ in range(k_max):
                a_k = a_k @ A
                powers.append(np.linalg.norm(a_k, 2))
            return np.array(powers)

        spec_normal = lg_ssm_spec(spectral_radius=0.9, kappa_v=1.0)
        spec_nonnormal = lg_ssm_spec(spectral_radius=0.9, kappa_v=100.0)
        A_normal, _ = lg_ssm_discretize_exact(spec_normal.params, spec_normal.dt)
        A_nonnormal, _ = lg_ssm_discretize_exact(spec_nonnormal.params, spec_nonnormal.dt)

        norms_normal = two_norms_of_powers(A_normal, 300)
        norms_nonnormal = two_norms_of_powers(A_nonnormal, 300)
        g50_normal = float(norms_normal[:50].max())
        g50_nonnormal = float(norms_nonnormal[:50].max())

        assert g50_normal <= 0.9 + 1e-9, "normal case: G_50 = rho (no amplification)"
        assert g50_nonnormal > 5.0, f"designed transient amplification, got {g50_nonnormal}"
        assert g50_nonnormal > g50_normal
        assert norms_normal[-1] < 1e-2, "asymptotic decay (normal)"
        assert norms_nonnormal[-1] < 1e-2, "asymptotic decay (non-normal)"

    def test_lg_ssm_same_seed_reproduces(self):
        from serie.catalog import lg_ssm_spec

        spec = lg_ssm_spec(steps=500)
        a = generate_from_trajectory_spec(spec, seed=7, dtype=np.float64)
        b = generate_from_trajectory_spec(spec, seed=7, dtype=np.float64)
        assert np.array_equal(a, b)

    def test_lg_ssm_different_trajectory_seeds_differ(self):
        from serie.catalog import lg_ssm_spec

        spec = lg_ssm_spec(steps=500)
        a = generate_from_trajectory_spec(spec, seed=7, dtype=np.float64)
        b = generate_from_trajectory_spec(spec, seed=8, dtype=np.float64)
        assert not np.array_equal(a, b)

    def test_lg_ssm_bank_uses_paired_stochastic_draw_design(self):
        # Check (d): lg_ssm is registered stochastic, so build_dgp_bank spawns
        # one trajectory seed per draw (each draw a different noise path),
        # exactly like ou.
        import serie

        bank = serie.build_dgp_bank("lg_ssm_small", seed=0, n_trajectory_draws=2)
        assert bank.is_stochastic
        assert bank.clean.shape[0] == 2
        assert bank.clean.shape[2] == 4
        assert not np.array_equal(bank.clean[0], bank.clean[1])

    def test_lg_ssm_param_shock_rebuilds_discretization_at_exact_step(self):
        # An abrupt params_step_change re-discretizes (A, Q) at the event
        # step: pre-shock rows are bitwise identical to the unshocked run
        # (same rng stream, same draw count per step), the shock-step row is
        # the first to differ.
        from serie.catalog import LGSSMParams, lg_ssm_small_spec

        spec = lg_ssm_small_spec(steps=200)
        p = spec.params
        params_after = LGSSMParams(
            gammas=tuple(2.0 * g for g in p.gammas),  # faster damping, same block count
            omegas=p.omegas,
            kappa_v=p.kappa_v,
            q_c=p.q_c,
        )
        shocked_spec = evolve(
            spec,
            shocks=ShockSchedule.single(
                ShockEvent.params_step_change(params_after=params_after, t_start=0.5)
            ),
        )
        base = generate_from_trajectory_spec(spec, seed=3, dtype=np.float64)
        shocked = generate_from_trajectory_spec(shocked_spec, seed=3, dtype=np.float64)
        shock_step = int(0.5 * 200)
        assert np.array_equal(base[:shock_step], shocked[:shock_step])
        assert not np.array_equal(base[shock_step], shocked[shock_step])

    def test_lg_ssm_stepped_transition_interpolates_tuple_params(self):
        # LGSSMParams carries tuple fields (gammas, omegas); the transition
        # interpolation machinery must flatten and rebuild them (the
        # DiagonalParams precedent), keeping block count structural.
        from serie.catalog import LGSSMParams

        before = LGSSMParams(gammas=(0.1, 0.2), omegas=(0.5, 0.25), kappa_v=1.0, q_c=0.2)
        after = LGSSMParams(gammas=(0.3, 0.4), omegas=(0.7, 0.45), kappa_v=5.0, q_c=0.4)
        schedule = StepShockSchedule.param_transition(
            before, after, start_step=10, end_step=20, n_inserts=2
        )
        halfway = schedule.param_sets[1]
        assert isinstance(halfway, LGSSMParams)
        assert halfway.gammas == pytest.approx((0.2, 0.3))
        assert halfway.omegas == pytest.approx((0.6, 0.35))
        assert halfway.kappa_v == pytest.approx(3.0)
        assert halfway.q_c == pytest.approx(0.3)
        landed = schedule.param_sets[2]
        assert landed == after
