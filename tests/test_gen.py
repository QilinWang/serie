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

from msgspec.structs import replace
import msgspec
import json


import numpy as np
import pytest

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
from serie import ShockEvent, ShockSchedule, TrajectorySpec, build_dgp_bank, load_data
from serie.api import dgp_bank_from_tensors, export_dgp_bank_files, export_dgp_bank_safetensors, load_dgp_bank_safetensors
from serie import (
    ImpulseCandidate,
    ImpulseClock,
    ImpulseClockKind,
    ImpulseFormulaKind,
    NamedStateTerm,
    RealizedImpulse,
    StateImpulsePlan,
    StateImpulseRealization,
    StateImpulseSource,
    realize_state_impulse_plan,
)
from serie.integrators import (
    ParamChangeEvent,
    ScalarParameterIncrementEvent,
    StateCoupledImpulseEvent,
    StateExpressionOperation,
    StatePerturbEvent,
    StateVariableTerm,
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
    def test_random_state_impulse_plan_records_and_replays_timing_and_choice(self, tmp_path):
        plan = StateImpulsePlan(
            system=ODESystem.LORENZ,
            variables=("x", "y", "z"),
            clock=ImpulseClock(kind=ImpulseClockKind.ONE_PER_BLOCK, first_step=24,
                               last_step=95, stride=24, probability=1.0),
            candidates=(
                ImpulseCandidate(targets=("y", "z"), formula=ImpulseFormulaKind.CONSTANT,
                                 terms=(), coefficient=0.01, probability=0.5),
                ImpulseCandidate(targets=("y", "z"), formula=ImpulseFormulaKind.STATE_SUM,
                                 terms=(NamedStateTerm(variable="y", lag=1, weight=1.0),
                                        NamedStateTerm(variable="z", lag=2, weight=1.0)),
                                 coefficient=1e-4, probability=0.5),
            ),
        )
        realization = realize_state_impulse_plan(plan, seed=0)
        assert realization == realize_state_impulse_plan(plan, seed=0)
        assert len(realization.events) == 3
        assert all(start <= event.step < start + 24 for start, event in zip(
            (24, 48, 72), realization.events, strict=True))
        assert all(event.candidate_index in (0, 1) for event in realization.events)
        assert {event.candidate_index for event in realization.events} == {0, 1}
        saved = msgspec.json.encode(realization)
        assert msgspec.json.decode(saved, type=StateImpulseRealization) == realization

        source = TrajectorySpec.lorenz(steps=96, burn_in=0)
        schedule = realization.to_step_schedule(source)
        first_step = realization.events[0].step
        baseline = generate_from_trajectory_spec(source, seed=3, dtype=np.float64)
        changed = generate_from_trajectory_spec(source, seed=3, dtype=np.float64, step_schedule=schedule)
        np.testing.assert_array_equal(changed[:first_step], baseline[:first_step])
        assert np.max(np.abs(changed[first_step] - baseline[first_step])) > 0
        np.testing.assert_array_equal(
            source.load(seed=3, step_schedule=schedule).series,
            generate_from_trajectory_spec(source, seed=3, step_schedule=schedule),
        )
        named = StateImpulseSource(
            dataset_id="lorenz.state_impulse.fixture",
            trajectory_spec=source,
            realization=realization,
        )
        with pytest.raises(ValueError, match="ordinary system dataset identity"):
            replace(named, dataset_id="lorenz.series")
        public_data = load_data(named, seed=3)
        assert public_data.source_id == named.dataset_id
        assert public_data.feature_names == ("x", "y", "z")
        assert public_data.metadata["state_impulse_realization"]["events"] == tuple(
            {"step": event.step, "candidate_index": event.candidate_index}
            for event in realization.events
        )
        np.testing.assert_array_equal(public_data.values, changed.astype(np.float32))
        bank = build_dgp_bank(named, seed=3, n_noise_draws=1)
        assert bank.name == named.dataset_id
        assert bank.feature_names == public_data.feature_names
        np.testing.assert_array_equal(bank.clean[0], public_data.values)
        assert json.loads(bank.source_declaration)["realization"]["seed"] == 0
        manifest = export_dgp_bank_files(bank, tmp_path / "arrays", array_format="npy")
        assert manifest["source_declaration"]["dataset_id"] == named.dataset_id
        safetensors_path = tmp_path / "bank.safetensors"
        export_dgp_bank_safetensors(bank, safetensors_path)
        restored = dgp_bank_from_tensors(load_dgp_bank_safetensors(safetensors_path))
        assert restored.source_declaration == bank.source_declaration
        with pytest.raises(ValueError, match="freezes one realized event schedule"):
            build_dgp_bank(named, seed=3, n_trajectory_draws=2)

    def test_random_clock_is_independent_of_candidate_choice_stream(self):
        candidate = ImpulseCandidate(targets=("y",), formula=ImpulseFormulaKind.CONSTANT,
                                     terms=(), coefficient=0.01, probability=1.0)
        clock = ImpulseClock(kind=ImpulseClockKind.BERNOULLI, first_step=24,
                             last_step=72, stride=1, probability=0.5)
        one = StateImpulsePlan(system=ODESystem.LORENZ, variables=("x", "y", "z"),
                               clock=clock, candidates=(candidate,))
        two = replace(one, candidates=(replace(candidate, probability=0.5),
                                       replace(candidate, targets=("z",), probability=0.5)))
        one_events = realize_state_impulse_plan(one, seed=17).events
        two_events = realize_state_impulse_plan(two, seed=17).events
        assert tuple(event.step for event in one_events) == tuple(event.step for event in two_events)
        assert all(24 <= event.step <= 72 for event in one_events)
        assert len(one_events) > 0

    def test_fixed_interval_clock_randomizes_formula_at_exact_steps(self):
        plan = StateImpulsePlan(
            system=ODESystem.LORENZ,
            variables=("x", "y", "z"),
            clock=ImpulseClock(kind=ImpulseClockKind.FIXED_INTERVAL,
                               first_step=24, last_step=96, stride=24, probability=1.0),
            candidates=(
                ImpulseCandidate(targets=("y", "z"), formula=ImpulseFormulaKind.CONSTANT,
                                 terms=(), coefficient=0.01, probability=0.5),
                ImpulseCandidate(targets=("y", "z"), formula=ImpulseFormulaKind.STATE_SUM,
                                 terms=(NamedStateTerm(variable="y", lag=0, weight=1.0),
                                        NamedStateTerm(variable="z", lag=0, weight=1.0)),
                                 coefficient=1e-4, probability=0.5),
            ),
        )
        events = realize_state_impulse_plan(plan, seed=0).events
        assert tuple(event.step for event in events) == (24, 48, 72, 96)
        assert {event.candidate_index for event in events} == {0, 1}

    def test_state_impulse_plan_refuses_undeclared_formula_and_invalid_replay(self):
        clock = ImpulseClock(kind=ImpulseClockKind.FIXED_INTERVAL, first_step=24,
                             last_step=48, stride=24, probability=1.0)
        candidate = ImpulseCandidate(targets=("y", "z"), formula=ImpulseFormulaKind.STATE_PRODUCT,
                                     terms=(NamedStateTerm(variable="y", lag=0, weight=1.0),
                                            NamedStateTerm(variable="z", lag=0, weight=1.0)),
                                     coefficient=1e-4, probability=1.0)
        plan = StateImpulsePlan(system=ODESystem.LORENZ, variables=("x", "y", "z"),
                                clock=clock, candidates=(candidate,))
        with pytest.raises(ValueError, match="formula"):
            replace(candidate, formula="power")
        with pytest.raises(ValueError, match="declared state variables"):
            replace(plan, candidates=(replace(candidate, targets=("w",)),))
        with pytest.raises(ValueError, match="every eligible"):
            StateImpulseRealization(plan=plan, seed=0,
                                    events=(RealizedImpulse(step=24, candidate_index=0),))
        with pytest.raises(ValueError, match="source system disagree"):
            realize_state_impulse_plan(plan, seed=0).to_step_schedule(TrajectorySpec.rossler(steps=48))

    def test_exact_step_state_and_parameter_changes_use_native_steps(self):
        base = TrajectorySpec.lorenz(steps=72, burn_in=0)
        schedule = StepShockSchedule(
            events=(
                StatePerturbEvent(step=24, state_idx=1, eps=0.01),
                StatePerturbEvent(step=24, state_idx=2, eps=0.01),
                ScalarParameterIncrementEvent(step=48, parameter_name="rho", delta=0.01),
            ),
            param_sets=(base.params,),
        )
        factual = generate_from_trajectory_spec(base, seed=0, dtype=np.float64)
        changed = generate_from_trajectory_spec(base, seed=0, dtype=np.float64, step_schedule=schedule)

        np.testing.assert_array_equal(changed[:24], factual[:24])
        pre_step = factual[23].copy()
        pre_step[1:] += 0.01
        np.testing.assert_allclose(changed[24], _rk4_step(SYSTEMS["lorenz"].f_py, pre_step, base.dt, base.params), rtol=0, atol=1e-13)
        after = replace(base.params, rho=base.params.rho + 0.01)
        np.testing.assert_allclose(changed[48], _rk4_step(SYSTEMS["lorenz"].f_py, changed[47], base.dt, after), rtol=0, atol=1e-13)

    def test_exact_step_delayed_perturbation_reads_the_declared_past(self):
        base = TrajectorySpec.lorenz(steps=72, burn_in=0)
        delayed = StateCoupledImpulseEvent(
            step=24,
            targets=(1, 2),
            terms=(StateVariableTerm(state_idx=1, lag=1, weight=1.0),
                   StateVariableTerm(state_idx=2, lag=2, weight=1.0)),
            operation=StateExpressionOperation.SUM,
            coefficient=1e-4,
        )
        schedule = StepShockSchedule(events=(delayed, replace(delayed, step=48)), param_sets=(base.params,))
        factual = generate_from_trajectory_spec(base, seed=0, dtype=np.float64)
        changed = generate_from_trajectory_spec(base, seed=0, dtype=np.float64, step_schedule=schedule)

        np.testing.assert_array_equal(changed[:24], factual[:24])
        for step in (24, 48):
            amount = 1e-4 * (changed[step - 2, 1] + changed[step - 3, 2])
            pre_step = changed[step - 1].copy()
            pre_step[1:] += amount
            np.testing.assert_allclose(
                changed[step], _rk4_step(SYSTEMS["lorenz"].f_py, pre_step, base.dt, base.params),
                rtol=0, atol=1e-13,
            )

    def test_exact_step_state_product_reads_current_pre_step_state(self):
        base = TrajectorySpec.lorenz(steps=24, burn_in=0)
        event = StateCoupledImpulseEvent(
            step=24, targets=(1, 2),
            terms=(StateVariableTerm(state_idx=1, lag=0, weight=1.0),
                   StateVariableTerm(state_idx=2, lag=0, weight=1.0)),
            operation=StateExpressionOperation.PRODUCT, coefficient=1e-4,
        )
        schedule = StepShockSchedule(events=(event,), param_sets=(base.params,))
        factual = generate_from_trajectory_spec(base, dtype=np.float64)
        changed = generate_from_trajectory_spec(base, dtype=np.float64, step_schedule=schedule)

        np.testing.assert_array_equal(changed[:24], factual[:24])
        pre_step = factual[23].copy()
        pre_step[1:] += 1e-4 * factual[23, 1] * factual[23, 2]
        np.testing.assert_allclose(changed[24], _rk4_step(SYSTEMS["lorenz"].f_py, pre_step, base.dt, base.params), rtol=0, atol=1e-13)

    def test_exact_step_schedule_refuses_missing_history_and_mixed_clocks(self):
        base = TrajectorySpec.lorenz(steps=72, burn_in=0)
        delayed = StateCoupledImpulseEvent(
            step=2, targets=(1, 2),
            terms=(StateVariableTerm(state_idx=1, lag=1, weight=1.0),
                   StateVariableTerm(state_idx=2, lag=2, weight=1.0)),
            operation=StateExpressionOperation.SUM,
            coefficient=1e-4,
        )
        schedule = StepShockSchedule(events=(delayed,), param_sets=(base.params,))
        with pytest.raises(ValueError, match="preceding state history"):
            generate_from_trajectory_spec(base, step_schedule=schedule)
        fraction_timed = replace(base, shocks=ShockSchedule.single(
            ShockEvent.state_perturbation(state_idx=1, eps=0.01, t_start=0.5)))
        with pytest.raises(ValueError, match="cannot share one trajectory"):
            generate_from_trajectory_spec(fraction_timed, step_schedule=schedule)

    def test_exact_step_events_reach_the_existing_serie_bundle(self):
        base = TrajectorySpec.lorenz(steps=48, burn_in=0)
        schedule = StepShockSchedule(
            events=(StatePerturbEvent(step=24, state_idx=1, eps=0.01),),
            param_sets=(base.params,),
        )
        bundle = base.load(seed=3, step_schedule=schedule)
        expected = generate_from_trajectory_spec(base, seed=3, step_schedule=schedule)

        np.testing.assert_array_equal(bundle.series, expected)
        assert bundle.params.step_schedule is schedule

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
        source = replace(
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

    def test_public_stepped_transition_preserves_prefix_and_changes_final_value(self):
        base = TrajectorySpec.diagonal(eigenvalues=(0.0,), steps=20, dt=0.05, burn_in=0)
        transitioned = replace(
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
            doublewell_barrier_collapse_spec,
            lg_ssm_spec,
            lg_ssm_small_spec,
            lg_ssm_switch_spec,
            lg_ssm_regime_switch_spec,
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
            doublewell_barrier_collapse_spec,
            lg_ssm_spec,
            lg_ssm_small_spec,
            lg_ssm_switch_spec,
            lg_ssm_regime_switch_spec,
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
            doublewell_barrier_collapse_spec,
            doublewell_param_shock_spec,
            doublewell_switch_spec,
            lg_ssm_switch_spec,
            lg_ssm_regime_switch_spec,
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
            doublewell_barrier_collapse_spec,
            doublewell_param_shock_spec,
            doublewell_switch_spec,
            lg_ssm_switch_spec,
            lg_ssm_regime_switch_spec,
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

    def test_doublewell_barrier_collapse_changes_only_the_barrier_parameter(self):
        from serie.catalog import DoubleWellParams, doublewell_barrier_collapse_spec

        spec = doublewell_barrier_collapse_spec()
        assert spec.params == DoubleWellParams(a=1.5, sigma_noise=0.25)
        event = spec.shocks.events[0]
        assert event.t_start == 0.35
        assert event.params_after == DoubleWellParams(a=0.5, sigma_noise=0.25)

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

    def test_named_initial_condition_sources_are_closed_and_pinned(self):
        from serie.catalog import ODESystem, list_sources, source_preset

        holdout = [source_preset(f"rossler_ic_holdout_{ordinal}")() for ordinal in range(8)]
        assert holdout[0].x0 == (0.99, 0.99, 0.99)
        assert holdout[7].x0 == (1.01, 1.01, 1.01)
        for axis in range(3):
            assert sum(spec.x0[axis] == 0.99 for spec in holdout) == 4
            assert sum(spec.x0[axis] == 1.01 for spec in holdout) == 4

        rossler_random = [source_preset(f"rossler_ic_random_{ordinal}")() for ordinal in range(16)]
        assert rossler_random[0].x0 == (
            0.9913589285360874,
            0.9964693520999365,
            0.9969714152588378,
        )
        assert rossler_random[15].x0 == (
            1.0037225046260485,
            0.9921321284829052,
            1.0048187731345346,
        )
        assert len({spec.x0 for spec in rossler_random}) == 16
        assert all(spec.system is ODESystem.ROSSLER for spec in holdout + rossler_random)
        assert all(abs(value - 1.0) <= 0.01 for spec in rossler_random for value in spec.x0)

        lorenz_random = [source_preset(f"lorenz_ic_{ordinal}")() for ordinal in range(16)]
        assert lorenz_random[0].x0 == (
            0.42436157047691725,
            0.13387029225556435,
            1.8456599328215408,
        )
        assert lorenz_random[15].x0 == (
            0.07975996295866583,
            1.3642066367795043,
            1.7588223539298788,
        )
        assert len({spec.x0 for spec in lorenz_random}) == 16
        assert all(spec.system is ODESystem.LORENZ for spec in lorenz_random)
        lorenz_base = (1.0, 0.98, 1.1)
        assert all(
            abs(value - lorenz_base[axis]) <= 1.0
            for spec in lorenz_random
            for axis, value in enumerate(spec.x0)
        )

        names = set(list_sources())
        assert all(f"rossler_ic_holdout_{ordinal}" in names for ordinal in range(8))
        assert all(f"rossler_ic_random_{ordinal}" in names for ordinal in range(16))
        assert all(f"lorenz_ic_{ordinal}" in names for ordinal in range(16))
        for unsupported in [
            "rossler_ic_holdout_8",
            "rossler_ic_random_16",
            "lorenz_ic_16",
            "controlled_pendulum",
            "controlled_lgssm_controlled_lgssm",
            "radial_attenuation_gamma4",
            "radial_attenuation_gaussian",
        ]:
            with pytest.raises(ValueError, match="Unknown system name"):
                source_preset(unsupported)

    def test_named_initial_condition_bank_preserves_the_requested_source_identity(self):
        from serie.api import build_dgp_bank

        bank = build_dgp_bank(
            "rossler_ic_holdout_0",
            seed=42,
            n_trajectory_draws=1,
            n_noise_draws=1,
            noise_seed_base=11_000,
        )

        assert bank.name == "rossler_ic_holdout_0"
        assert tuple(float(value) for value in bank.clean[0, 0]) == pytest.approx((0.99, 0.99, 0.99))


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
        spec = replace(
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
        spec = replace(spec, x0=(1.0,))
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
        spec = replace(
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

    def test_stage_synthetic_lorenz_rk4_golden_rows(self):
        """Pin the shared f64 RK4 Lorenz trajectory before dtype conversion."""
        trajectory = generate_from_trajectory_spec(
            TrajectorySpec.lorenz(steps=8), seed=0, dtype=np.float64
        )
        expected = np.array(
            [
                [1.0, 0.98, 1.1],
                [1.0106171380832871, 1.2388683281370947, 1.0820453023442074],
                [1.0449852330283642, 1.501400454999214, 1.0674511464470786],
                [1.1014901367567624, 1.7736620877262035, 1.0566852170181364],
                [1.1792277940744504, 2.0613208903755176, 1.050435417714531],
                [1.2779072863287948, 2.3698172509608684, 1.0496302483769067],
                [1.3977780190154419, 2.7045083613021608, 1.0554776691236423],
                [1.539576445052988, 3.070790073021694, 1.0695221920723448],
                [1.7044885000029064, 3.4741993644492126, 1.0937215315961872],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(trajectory, expected, rtol=0.0, atol=2e-15)

    def test_stage_synthetic_transition_uses_elapsed_step_progress(self):
        """A rounded 7/10 schedule uses elapsed 2/5 then full progress."""
        before = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        after = LorenzParams(sigma=10.0, rho=35.0, beta=8.0 / 3.0)
        schedule = StepShockSchedule.param_transition(
            before, after, start_step=5, end_step=10, n_inserts=2
        )
        assert [(event.step, event.param_set_idx) for event in schedule.events] == [
            (7, 1),
            (10, 2),
        ]
        assert isinstance(schedule.param_sets[1], LorenzParams)
        # Rust currently advances by update ordinal and incorrectly gives 31.5
        # here. A transition names elapsed time, so its parameters must follow
        # the actual rounded event time: (7 - 5) / (10 - 5) = 2/5.
        assert schedule.param_sets[1].rho == pytest.approx(30.8, abs=1e-12)
        assert 28.0 < schedule.param_sets[1].rho < after.rho
        assert schedule.param_sets[2] == after

    def test_stage_synthetic_one_update_transition_is_abrupt(self):
        """A step-change starts at floor(0.25 * 20) = 5, never at its end time."""
        base = TrajectorySpec.lorenz(steps=20)
        shocked = replace(
            base,
            shocks=ShockSchedule.single(
                ShockEvent.params_step_change(
                    params_after=LorenzParams(
                        sigma=10.0, rho=35.0, beta=8.0 / 3.0
                    ),
                    t_start=0.25,
                )
            ),
        )
        nominal = generate_from_trajectory_spec(base, seed=0, dtype=np.float64)
        changed = generate_from_trajectory_spec(shocked, seed=0, dtype=np.float64)
        np.testing.assert_array_equal(changed[:5], nominal[:5])
        assert not np.array_equal(changed[5], nominal[5])

    @pytest.mark.parametrize("burn_in", [10, 11], ids=["equal", "greater"])
    def test_stage_synthetic_rejects_burn_in_at_or_beyond_steps(self, burn_in: int):
        spec = replace(TrajectorySpec.lorenz(steps=10), burn_in=burn_in)
        with pytest.raises(
            ValueError, match=rf"burn_in \({burn_in}\) must be < steps \(10\)"
        ):
            generate_from_trajectory_spec(spec, seed=0)

    def test_stage_synthetic_slds_burn_in_shape(self):
        from serie.catalog import slds_spec

        spec = replace(slds_spec(steps=10), burn_in=3)
        trajectory = generate_from_trajectory_spec(spec, seed=0, dtype=np.float64)
        assert trajectory.shape == (8, 1)
        assert np.isfinite(trajectory).all()

    def test_stage_synthetic_garch_initial_state_golden(self):
        from serie.catalog import garch_spec

        spec = garch_spec(steps=10)
        trajectory = generate_from_trajectory_spec(spec, seed=0, dtype=np.float64)
        expected_variance = spec.params.omega / (
            1.0 - spec.params.alpha - spec.params.beta
        )
        np.testing.assert_allclose(
            trajectory[0], [0.0, expected_variance], rtol=0.0, atol=1e-15
        )
        assert trajectory.shape == (11, 2)

    def test_stage_synthetic_discrete_seed_contracts(self):
        from serie.catalog import (
            garch_spec,
            lg_ssm_small_spec,
            seasonal_ar_spec,
            slds_spec,
        )

        for spec in (
            slds_spec(steps=40),
            seasonal_ar_spec(steps=40),
            garch_spec(steps=40),
            lg_ssm_small_spec(steps=40),
        ):
            first = generate_from_trajectory_spec(spec, seed=7, dtype=np.float64)
            same_seed = generate_from_trajectory_spec(spec, seed=7, dtype=np.float64)
            other_seed = generate_from_trajectory_spec(spec, seed=8, dtype=np.float64)
            np.testing.assert_array_equal(first, same_seed)
            assert not np.array_equal(first, other_seed)

    def test_stage_synthetic_discrete_param_shock_presets_generate(self):
        from serie.catalog import (
            garch_param_shock_spec,
            seasonal_ar_param_shock_spec,
            slds_param_shock_spec,
        )

        for spec in (
            replace(slds_param_shock_spec(), steps=40),
            replace(seasonal_ar_param_shock_spec(), steps=40),
            replace(garch_param_shock_spec(), steps=40),
        ):
            trajectory = generate_from_trajectory_spec(spec, seed=0, dtype=np.float64)
            assert trajectory.shape == (41, spec.dim)
            assert np.isfinite(trajectory).all()

    @pytest.mark.parametrize(
        "switch_spec_factory",
        [
            pytest.param("slds", id="slds"),
            pytest.param("doublewell", id="doublewell"),
        ],
    )
    def test_stage_synthetic_stochastic_switch_splices_seed_plus_one(
        self, switch_spec_factory: str
    ):
        from serie.catalog import doublewell_switch_spec, slds_switch_spec

        factory = {
            "slds": slds_switch_spec,
            "doublewell": doublewell_switch_spec,
        }[switch_spec_factory]
        switched_spec = replace(factory(), steps=20)
        event = switched_spec.shocks.events[0]
        assert event.ic_b is not None
        assert event.params_after is not None
        splice_step = int(event.t_start * switched_spec.steps)
        before = replace(switched_spec, shocks=ShockSchedule.none())
        after = replace(
            before,
            params=event.params_after,
            x0=event.ic_b,
            shocks=ShockSchedule.none(),
        )
        switched = generate_from_trajectory_spec(
            switched_spec, seed=5, dtype=np.float64
        )
        trajectory_a = generate_from_trajectory_spec(before, seed=5, dtype=np.float64)
        trajectory_b = generate_from_trajectory_spec(after, seed=6, dtype=np.float64)
        np.testing.assert_array_equal(
            switched[: splice_step + 1], trajectory_a[: splice_step + 1]
        )
        np.testing.assert_array_equal(
            switched[splice_step + 1 :], trajectory_b[splice_step + 1 :]
        )

    def test_stage_synthetic_ou_seed_changes_trajectory(self):
        first = generate_from_trajectory_spec(
            TrajectorySpec.ou(steps=100), seed=7, dtype=np.float64
        )
        other = generate_from_trajectory_spec(
            TrajectorySpec.ou(steps=100), seed=8, dtype=np.float64
        )
        assert not np.array_equal(first, other)

    def test_stage_generate_ode_switch_splices_prefix_onto_b(self):
        """A deterministic switch is A through its cut row, then B's suffix."""
        from serie.catalog import lorenz_switch_spec

        switched_spec = replace(lorenz_switch_spec(), steps=20)
        event = switched_spec.shocks.events[0]
        assert event.ic_b is not None
        assert event.params_after is not None
        splice_step = int(event.t_start * switched_spec.steps)
        before = replace(switched_spec, shocks=ShockSchedule.none())
        after = replace(
            before,
            params=event.params_after,
            x0=event.ic_b,
            shocks=ShockSchedule.none(),
        )
        switched = generate_from_trajectory_spec(switched_spec, seed=5, dtype=np.float64)
        trajectory_a = generate_from_trajectory_spec(before, seed=5, dtype=np.float64)
        trajectory_b = generate_from_trajectory_spec(after, seed=6, dtype=np.float64)
        np.testing.assert_array_equal(
            switched[: splice_step + 1], trajectory_a[: splice_step + 1]
        )
        np.testing.assert_array_equal(
            switched[splice_step + 1 :], trajectory_b[splice_step + 1 :]
        )

    def test_stage_generate_param_set_index_selects_declared_regime(self):
        """An event selects its declared parameter-set index, not its ordinal."""
        before = LorenzParams(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        skipped = LorenzParams(sigma=10.0, rho=29.0, beta=8.0 / 3.0)
        selected = LorenzParams(sigma=10.0, rho=35.0, beta=8.0 / 3.0)
        schedule = StepShockSchedule(
            events=(ParamChangeEvent(step=1, param_set_idx=2),),
            param_sets=(before, skipped, selected),
        )
        indexed = _np_ode(
            "lorenz", np.array([1.0, 0.98, 1.1]), 3, 0.01, before, schedule
        )
        direct = _np_ode(
            "lorenz", np.array([1.0, 0.98, 1.1]), 3, 0.01, selected
        )
        np.testing.assert_array_equal(indexed, direct)

    def test_stage_generate_ou_zero_noise_matches_euler_formula(self):
        """With sigma=0, OU is its stated Euler recurrence, independent of seed."""
        from serie.catalog import ou_spec

        spec = ou_spec(theta=0.5, mu=2.0, sigma_noise=0.0, dt=0.25, steps=2)
        trajectory = generate_from_trajectory_spec(spec, seed=17, dtype=np.float64)
        np.testing.assert_allclose(trajectory[:, 0], [0.0, 0.25, 0.46875], atol=1e-15)

    def test_stage_generate_doublewell_zero_noise_matches_euler_formula(self):
        """Zero-noise double-well follows x += (-x^3 + a*x) dt toward +sqrt(a)."""
        from serie.catalog import doublewell_spec

        spec = replace(
            doublewell_spec(a=1.5, sigma_noise=0.0, dt=0.1, steps=2), x0=(0.2,)
        )
        trajectory = generate_from_trajectory_spec(spec, seed=17, dtype=np.float64)
        first = 0.2 + (-0.2**3 + 1.5 * 0.2) * 0.1
        second = first + (-first**3 + 1.5 * first) * 0.1
        np.testing.assert_allclose(trajectory[:, 0], [0.2, first, second], atol=1e-15)
        assert trajectory[-1, 0] > trajectory[0, 0]

    def test_stage_generate_ou_parameter_and_state_shocks_land_on_event_row(self):
        """Events apply before their Euler step, so both alter that output row."""
        from serie.catalog import OUParams, ou_spec

        before = OUParams(theta=0.5, mu=0.0, sigma_noise=0.0)
        after = OUParams(theta=0.5, mu=4.0, sigma_noise=0.0)
        spec = replace(
            ou_spec(theta=before.theta, mu=before.mu, sigma_noise=0.0, dt=1.0, steps=2),
            shocks=ShockSchedule.multiple(
                ShockEvent.params_step_change(after, t_start=0.5),
                ShockEvent.state_perturbation(state_idx=0, eps=3.0, t_start=1.0),
            ),
        )
        trajectory = generate_from_trajectory_spec(spec, seed=0, dtype=np.float64)
        np.testing.assert_allclose(trajectory[:, 0], [0.0, 2.0, 4.5], atol=1e-15)

    def test_stage_generate_slds_param_shock_lands_on_event_row(self):
        """A zero-noise SLDS parameter replacement changes the declared row itself."""
        from serie.catalog import SLDSParams, slds_spec

        before = SLDSParams(A1=1.0, A2=1.0, Q1=0.0, Q2=0.0, p11=1.0, p22=1.0)
        after = SLDSParams(A1=2.0, A2=2.0, Q1=0.0, Q2=0.0, p11=1.0, p22=1.0)
        spec = replace(
            slds_spec(steps=3),
            params=before,
            x0=(1.0,),
            shocks=ShockSchedule.single(ShockEvent.params_step_change(after, t_start=2 / 3)),
        )
        trajectory = generate_from_trajectory_spec(spec, seed=0, dtype=np.float64)
        np.testing.assert_allclose(trajectory[:, 0], [1.0, 1.0, 2.0, 4.0], atol=1e-15)

    def test_stage_generate_seasonal_ar_periodicity_and_state_kick(self):
        """The recurrence is periodic without noise, and a state kick is pre-step."""
        from serie.catalog import seasonal_ar_spec

        periodic = generate_from_trajectory_spec(
            seasonal_ar_spec(S=4, phi=0.0, sigma=0.0, a0=1.0, steps=4),
            seed=0,
            dtype=np.float64,
        )
        np.testing.assert_allclose(periodic[:, 0], [0.0, 1.0, 0.0, -1.0, 0.0], atol=1e-15)

        kicked_spec = replace(
            seasonal_ar_spec(S=4, phi=1.0, sigma=0.0, a0=0.0, steps=4),
            shocks=ShockSchedule.single(
                ShockEvent.state_perturbation(state_idx=0, eps=3.0, t_start=0.5)
            ),
        )
        kicked = generate_from_trajectory_spec(kicked_spec, seed=0, dtype=np.float64)
        np.testing.assert_allclose(kicked[:, 0], [0.0, 0.0, 3.0, 3.0, 3.0], atol=1e-15)

    def test_stage_generate_garch_param_and_state_shocks_land_on_event_row(self):
        """GARCH updates parameters and its stored prior return before recurrence."""
        from serie.catalog import GARCHParams, garch_spec

        after = GARCHParams(omega=0.1, alpha=0.5, beta=0.0)
        base = replace(
            garch_spec(steps=4),
            shocks=ShockSchedule.single(ShockEvent.params_step_change(after, t_start=0.5)),
        )
        kicked_spec = replace(
            garch_spec(steps=4),
            shocks=ShockSchedule.multiple(
                ShockEvent.params_step_change(after, t_start=0.5),
                ShockEvent.state_perturbation(state_idx=0, eps=2.0, t_start=0.75),
            ),
        )
        nominal = generate_from_trajectory_spec(base, seed=4, dtype=np.float64)
        kicked = generate_from_trajectory_spec(kicked_spec, seed=4, dtype=np.float64)
        shock_step = 2
        assert kicked[shock_step, 0] == pytest.approx(nominal[shock_step, 0] + 2.0)
        assert kicked[3, 1] == pytest.approx(0.1 + 0.5 * kicked[2, 0] ** 2)

    def test_stage_generate_garch_unconditional_variance(self):
        """A stationary GARCH return's long-run second moment targets omega/(1-a-b)."""
        from serie.catalog import garch_spec

        spec = garch_spec(omega=0.01, alpha=0.06, beta=0.90, steps=30_000)
        trajectory = generate_from_trajectory_spec(spec, seed=17, dtype=np.float64)
        expected = spec.params.omega / (1.0 - spec.params.alpha - spec.params.beta)
        observed = float(np.mean(trajectory[5_000:, 0] ** 2))
        assert observed == pytest.approx(expected, rel=0.15)

    def test_stage_generate_lg_ssm_param_shock_rejects_block_count_change(self):
        """State dimension is structural: a parameter event cannot add/remove blocks."""
        from serie.catalog import LGSSMParams, lg_ssm_small_spec

        spec = lg_ssm_small_spec(steps=4)
        p = spec.params
        invalid = LGSSMParams(
            gammas=p.gammas[:1], omegas=p.omegas[:1], kappa_v=p.kappa_v, q_c=p.q_c
        )
        shocked = replace(
            spec,
            shocks=ShockSchedule.single(
                ShockEvent.params_step_change(invalid, t_start=0.5)
            ),
        )
        with pytest.raises(ValueError, match="must keep the block count"):
            generate_from_trajectory_spec(shocked, seed=0, dtype=np.float64)

    def test_stage_generate_switch_rejects_step_beyond_trajectory(self):
        """The public fractional switch boundary rejects a cut beyond the final row."""
        with pytest.raises(ValueError):
            ShockEvent.trajectory_switch(ic_b=(1.0, 1.0, 1.0), t_start=1.1)


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

    def test_lg_ssm_regime_switch_changes_frequency_memory_and_geometry(self):
        """The strong world is multi-axis but stays exactly linear-Gaussian."""
        from serie.catalog import LGSSMParams, lg_ssm_regime_switch_spec

        spec = lg_ssm_regime_switch_spec(steps=20)
        before = spec.params
        after = spec.shocks.events[0].params_after
        assert isinstance(before, LGSSMParams)
        assert isinstance(after, LGSSMParams)
        assert after.omegas == pytest.approx(tuple(1.5 * omega for omega in before.omegas))
        assert np.exp(-after.gammas[0] * spec.dt) == pytest.approx(0.90)
        assert after.kappa_v == pytest.approx(5.0)
        assert after.q_c == pytest.approx(2.0 * after.gammas[0])

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
        shocked_spec = replace(
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

    @pytest.mark.parametrize(("context_steps", "horizon"), [(6, 4), (12, 3)])
    def test_lg_ssm_switch_truth_matches_conditional_monte_carlo(self, context_steps, horizon):
        """The analytic predictor matches forward simulation across/after the switch."""
        from serie.catalog import (
            LGSSMParams,
            lg_ssm_discretize_exact,
            lg_ssm_switch_spec,
            lgssm_switch_truth,
        )

        switch_step = 8
        spec = lg_ssm_switch_spec(
            periods=(12.0, 24.0),
            switch_fraction=switch_step / 20,
            steps=20,
        )
        params_before = spec.params
        params_after = spec.shocks.events[0].params_after
        assert isinstance(params_before, LGSSMParams)
        assert isinstance(params_after, LGSSMParams)

        clean = generate_from_trajectory_spec(spec, seed=17, dtype=np.float64)
        observation_std = 0.2
        observation_covariance = observation_std**2 * np.eye(spec.dim)
        observation_rng = np.random.default_rng(23)
        observations = clean + observation_std * observation_rng.standard_normal(clean.shape)
        law = lgssm_switch_truth(
            observations[:context_steps],
            params_before=params_before,
            params_after=params_after,
            dt=spec.dt,
            switch_step=switch_step,
            horizon=horizon,
            observation_covariance_dd=observation_covariance,
        )

        before_transition, before_process_covariance = lg_ssm_discretize_exact(params_before, spec.dt)
        after_transition, after_process_covariance = lg_ssm_discretize_exact(params_after, spec.dt)
        monte_carlo_rng = np.random.default_rng(29 + context_steps)
        draws = 30_000
        latent = monte_carlo_rng.multivariate_normal(
            law.filtered_mean_d,
            law.filtered_covariance_dd,
            size=draws,
        )
        future_observations = []
        last_context_step = context_steps - 1
        for horizon_idx in range(horizon):
            destination_step = last_context_step + horizon_idx + 1
            if destination_step < switch_step:
                transition = before_transition
                process_covariance = before_process_covariance
            else:
                transition = after_transition
                process_covariance = after_process_covariance
            latent = (
                latent @ transition.T
                + monte_carlo_rng.multivariate_normal(
                    np.zeros(spec.dim),
                    process_covariance,
                    size=draws,
                )
            )
            future_observations.append(
                latent
                + monte_carlo_rng.multivariate_normal(
                    np.zeros(spec.dim),
                    observation_covariance,
                    size=draws,
                )
            )

        simulated = np.stack(future_observations, axis=1).reshape(draws, horizon * spec.dim)
        assert np.allclose(simulated.mean(axis=0), law.mean_hd.reshape(-1), atol=0.025)
        assert np.allclose(
            np.cov(simulated, rowvar=False, ddof=0),
            law.covariance_pp,
            atol=0.035,
        )
