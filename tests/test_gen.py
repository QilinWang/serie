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
        f=spec.f_py, x0=x0, steps=steps, dt=dt, t0=0.0,
        params=params, method="rk4", time_dependent=spec.time_dependent,
        schedule=schedule,
    )


def _rk4_step(f, x: np.ndarray, dt: float, params) -> np.ndarray:
    args = _params_as_args(params)
    k1 = f(x, 0.0, *args)
    k2 = f(x + 0.5 * dt * k1, 0.0, *args)
    k3 = f(x + 0.5 * dt * k2, 0.0, *args)
    k4 = f(x + dt * k3, 0.0, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _estimate_lorenz_largest_lyapunov(*, steps=20_000, burn_steps=1_000,
                                      dt=0.01, delta0=1e-7) -> float:
    # Largest Lyapunov exponent = exponential rate at which two trajectories
    # starting delta0 apart pull apart. Positive => chaos. Benettin method:
    # advance both, measure growth, then RESCALE the separation back to delta0
    # so it never saturates at the attractor's size — accumulate log-growth.
    f = SYSTEMS["lorenz"].f_py
    params = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    x = np.array([1.0, 0.98, 1.1], dtype=np.float64)
    direction = np.array([0.3, -0.4, 0.5], dtype=np.float64)
    direction *= delta0 / np.linalg.norm(direction)   # tiny initial separation
    y = x + direction
    log_growth, n = 0.0, 0
    for i in range(steps):
        x = _rk4_step(f, x, dt, params)
        y = _rk4_step(f, y, dt, params)
        diff = y - x
        dist = float(np.linalg.norm(diff))            # how far they've drifted
        if i >= burn_steps:                           # skip the transient
            log_growth += np.log(dist / delta0)       # one step of exp-growth
            n += 1
        y = x + (delta0 / dist) * diff                # renormalize back to delta0
    return log_growth / (n * dt)                      # average rate per unit time
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
        assert np.max(np.abs(baseline[shock_step + 10:] - shocked[shock_step + 10:])) > 1e-6

    def test_lorenz_state_shock_differs(self):
        x0 = np.array([1.0, 0.98, 1.1])
        params = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        shock_step = STEPS // 2

        baseline = _np_ode("lorenz", x0, STEPS, DT, params)
        schedule = StepShockSchedule.state_shock(shock_step, state_idx=0, eps=5.0, params=params)
        shocked = _np_ode("lorenz", x0, STEPS, DT, params, schedule=schedule)

        np.testing.assert_array_equal(baseline[:shock_step], shocked[:shock_step])
        assert np.max(np.abs(baseline[shock_step + 1:] - shocked[shock_step + 1:])) > 1.0

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Preset construction tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPresets:
    """Verify that catalog preset functions produce valid TrajectorySpecs."""

    def test_all_active_presets_construct(self):
        from serie.catalog import (
            lorenz_spec, rossler_spec, chua_spec, lorenz96_spec,
            ou_spec, doublewell_spec, slds_spec, seasonal_ar_spec,
            garch_spec, ks_spec, diagonal_spec,
        )
        for fn in [lorenz_spec, rossler_spec, chua_spec, lorenz96_spec,
                    ou_spec, doublewell_spec, slds_spec, seasonal_ar_spec,
                    garch_spec, ks_spec, diagonal_spec]:
            spec = fn()
            assert spec.steps > 0
            assert spec.dim > 0
            assert spec.dt > 0

    def test_shock_presets_construct(self):
        from serie.catalog import (
            lorenz_param_shock_spec, lorenz_state_shock_spec, lorenz_switch_spec,
            rossler_param_shock_spec,
            chua_param_shock_spec, chua_switch_spec,
            ou_param_shock_spec,
            doublewell_param_shock_spec, doublewell_switch_spec,
            slds_param_shock_spec, slds_switch_spec,
            seasonal_ar_param_shock_spec,
            garch_param_shock_spec,
            ks_param_shock_spec,
        )
        shock_fns = [
            lorenz_param_shock_spec, lorenz_state_shock_spec, lorenz_switch_spec,
            rossler_param_shock_spec,
            chua_param_shock_spec, chua_switch_spec,
            ou_param_shock_spec,
            doublewell_param_shock_spec, doublewell_switch_spec,
            slds_param_shock_spec, slds_switch_spec,
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
        spec = ou_spec(steps=30_000, burn_in=5_000)        # burn_in: reach stationarity
        traj = generate_from_trajectory_spec(spec, seed=42, dtype=np.float64)[:, 0]
        p = spec.params
        expected = (p.sigma_noise**2 * spec.dt) / (1.0 - (1.0 - p.theta * spec.dt) ** 2)
        assert abs(float(traj.mean()) - p.mu) < 0.05       # mean-reverts to mu
        assert 0.85 * expected < float(traj.var()) < 1.15 * expected   # right spread


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

        nx = 32
        x = np.arange(nx, dtype=np.float64)
        spec = evolve(
            ks_spec(nx=nx, steps=200, dt=1e-2),
            x0=tuple(0.1 * np.cos(2.0 * np.pi * x / nx)),
        )
        traj = generate_from_trajectory_spec(spec, seed=42, dtype=np.float64)
        assert traj.shape == (201, nx)
        assert np.all(np.isfinite(traj))
        assert float(traj.std()) > 0.01

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
