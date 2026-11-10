"""
integrators.py - Unified dynamical systems integration.

Design principles:
  1. Pure computation in f_* functions (no defaults)
  2. SystemSpec registry (structure only, no defaults)
  3. Functional runtime entrypoint from TrajectorySpec (no config bridge class)
  4. Concrete parameter values defined by catalog presets or caller-owned specs

Supports:
  - ODEs: Lorenz, Rössler, Hyper-Rössler, Chua, Duffing, Van der Pol, Lorenz96
  - Maps: Hénon, Logistic
  - SDEs: Ornstein-Uhlenbeck, Double-Well Langevin
  - Discrete: SLDS, Seasonal AR, GARCH
  - PDEs: Kuramoto-Sivashinsky (ETDRK4)

Shock semantics (all systems):
  - "state": perturb state += eps at shock_step (global perturbation)
  - "param": abruptly replace params, or schedule a param transition window
  - "switch": replace the active trajectory with an independent trajectory at shock_step

NOTE: steps is the Number of integration steps.
Output shape is (steps + 1, dim), including initial condition.
This is the transpose of scipy.integrate.solve_ivp's `y`, which is (dim, steps + 1).

Behavior:

t_start == t_end and n_inserts=1 -> abrupt parameter step change.
t_start < t_end and n_inserts=k -> exactly k uniformly spaced param updates across the stepped transition window.
Concrete example:

start_step=500, end_step=1000, n_inserts=4 gives update steps:
[625, 750, 875, 1000]
Usage example:

event = ShockEvent.params_stepped_transition(
    params_after=after_params,
    t_start=0.35,
    t_end=0.5,
    n_inserts=4,
)
spec = evolve(base_spec, shocks=ShockSchedule.single(event))
"""

from __future__ import annotations

from enum import StrEnum
from msgspec.structs import replace

from msgspec import Struct
from serie.records import StructRecord, record_hash

from typing import Callable, cast

import numpy as np
from numpy.typing import DTypeLike

from .catalog import (
    DISCRETE_SYSTEMS,
    SYSTEMS,
    DiagonalParams,
    DoubleWellParams,
    GARCHParams,
    KSParams,
    LGSSMParams,
    Lorenz96Params,
    OUParams,
    SLDSParams,
    SeasonalARParams,
    SeasonalARBJParams,
    SystemParams,
    lg_ssm_discretize_exact,
)
from .specs import ShockType, TrajectorySpec


class ParamChangeEvent(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=False,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Switch to a different parameter set at a specific timestep."""

    step: int
    param_set_idx: int  # index into StepShockSchedule.param_sets

    __hash__ = record_hash


class StatePerturbEvent(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=False,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Additive state perturbation: x[state_idx] += eps at step."""

    step: int
    state_idx: int
    eps: float

    __hash__ = record_hash


class StateVariableTerm(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """One current or past coordinate, with lag counted from the pre-step state."""

    state_idx: int
    lag: int
    weight: float

    def __post_init__(self):
        if type(self.state_idx) is not int or self.state_idx < 0:
            raise ValueError("state variable term needs a nonnegative coordinate index")
        if type(self.lag) is not int or self.lag < 0:
            raise ValueError("state variable term needs a nonnegative integer lag")
        if not np.isfinite(self.weight):
            raise ValueError("state variable term weight must be finite")

    __hash__ = record_hash


class StateExpressionOperation(StrEnum):
    SUM = "sum"
    PRODUCT = "product"


class StateCoupledImpulseEvent(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """At one external event step, add a state-derived amount to each target."""

    step: int
    targets: tuple[int, ...]
    terms: tuple[StateVariableTerm, ...]
    operation: StateExpressionOperation
    coefficient: float

    def __post_init__(self):
        if type(self.step) is not int or self.step < 1:
            raise ValueError("state-coupled impulse needs a positive integer step")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("state-coupled impulse needs distinct target coordinates")
        if any(type(index) is not int or index < 0 for index in self.targets):
            raise ValueError("state-coupled impulse targets must be nonnegative integers")
        if not self.terms or not np.isfinite(self.coefficient):
            raise ValueError("state-coupled impulse needs terms and a finite coefficient")
        if not isinstance(self.operation, StateExpressionOperation):
            raise ValueError("state-coupled impulse operation must be sum or product")

    __hash__ = record_hash


class ScalarParameterIncrementEvent(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """At one exact step, add delta to one scalar parameter of the active law."""

    step: int
    parameter_name: str
    delta: float

    def __post_init__(self):
        if type(self.step) is not int or self.step < 1:
            raise ValueError("parameter increment needs a positive integer step")
        if not self.parameter_name or not np.isfinite(self.delta):
            raise ValueError("parameter increment needs a name and a finite delta")

    __hash__ = record_hash


def _interpolate_scalar(before: float, after: float, alpha: float) -> float:
    """Linear interpolation between two scalar values."""
    return before * (1.0 - alpha) + after * alpha


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# parameter stepped-transition values
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _transition_field_names(params: SystemParams) -> tuple[str, ...]:
    """Names of the fields that are allowed to move during a param transition.

    Structural fields stay fixed on the original object.
    Examples:
    - Lorenz96 `forcing` can move, but not lattice shape
    - SeasonalAR `S` stays fixed
    - KS `nx`, `dealias`, and `Lx` stay fixed
    """
    if isinstance(params, Lorenz96Params):
        return ("forcing",)
    if isinstance(params, SeasonalARParams):
        return ("phi", "sigma", "a0", "amp_drift")
    if isinstance(params, SeasonalARBJParams):
        # CHG 2026-10-06: the seasonal lag S indexes the past; it stays fixed.
        return ("phi", "seasonal_phi", "sigma")
    if isinstance(params, KSParams):
        # CHG (2026-07-02, user-approved): Lx removed from the transition set.
        # The ETDRK4 runner builds its spectral grid k from Lx ONCE and
        # build_coeffs() rebuilds coefficients from nu only, so an Lx change
        # would silently never take effect. Domain length is structural for a
        # fixed-nx grid: only the viscosity nu may move.
        return ("nu",)
    if isinstance(params, DiagonalParams):
        return ("eigenvalues",)
    if isinstance(params, LGSSMParams):
        # Block count (len of gammas/omegas) is structural; the per-block
        # rates/frequencies and the scalar knobs may move.
        return ("gammas", "omegas", "kappa_v", "q_c")
    # CHG: declared Struct order is the original parameter order; no copying.
    return params.__struct_fields__


def _transition_field_values(params: SystemParams) -> tuple[float, ...]:
    """Ordered numeric values that actually participate in a transition."""
    if isinstance(params, DiagonalParams):
        return tuple(float(value) for value in params.eigenvalues)
    if isinstance(params, LGSSMParams):
        # Flattened as [gammas…, omegas…, kappa_v, q_c] — the same order the
        # Rust SystemParams::as_f64_vec uses for LgSsmParams.
        return (
            *(float(value) for value in params.gammas),
            *(float(value) for value in params.omegas),
            float(params.kappa_v),
            float(params.q_c),
        )

    return tuple(
        float(getattr(params, field_name)) for field_name in _transition_field_names(params)
    )


def _params_with_transition_field_values(
    params: SystemParams,
    transition_values: tuple[float, ...],
) -> SystemParams:
    """Rebuild params from transition values while preserving structural fields."""
    if isinstance(params, DiagonalParams):
        return replace(params, eigenvalues=tuple(float(value) for value in transition_values))
    if isinstance(params, LGSSMParams):
        n_blocks = (len(transition_values) - 2) // 2
        return replace(
            params,
            gammas=tuple(float(value) for value in transition_values[:n_blocks]),
            omegas=tuple(float(value) for value in transition_values[n_blocks : 2 * n_blocks]),
            kappa_v=float(transition_values[2 * n_blocks]),
            q_c=float(transition_values[2 * n_blocks + 1]),
        )

    field_names = _transition_field_names(params)
    if len(field_names) != len(transition_values):
        raise ValueError(
            f"expected {len(field_names)} transition values for {type(params).__name__}, got {len(transition_values)}"
        )

    updates = {
        field_name: float(field_value)
        for field_name, field_value in zip(field_names, transition_values, strict=True)
    }
    return replace(params, **updates)


def _interpolate_params(before: SystemParams, after: SystemParams, alpha: float) -> SystemParams:
    """Build the intermediate parameter object at interpolation weight alpha."""
    if type(before) is not type(after):
        raise TypeError(
            f"param transition requires matching param types, got {type(before).__name__} vs {type(after).__name__}"
        )

    before_values = _transition_field_values(before)
    after_values = _transition_field_values(after)
    if len(before_values) != len(after_values):
        raise ValueError(
            f"param transition requires matching transition widths, got {len(before_values)} vs {len(after_values)}"
        )

    interpolated_values = tuple(
        _interpolate_scalar(before_value, after_value, alpha)
        for before_value, after_value in zip(before_values, after_values, strict=True)
    )
    return _params_with_transition_field_values(before, interpolated_values)


StepShockEvent = (
    ParamChangeEvent | StatePerturbEvent | StateCoupledImpulseEvent | ScalarParameterIncrementEvent
)


class StepShockSchedule(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=False,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Compiled shock schedule: events sorted by step + param sets.

    param_sets[0] is always the initial params.
    ParamChangeEvent.param_set_idx indexes into param_sets.
    """

    events: tuple[StepShockEvent, ...]
    param_sets: tuple[SystemParams, ...]  # [initial, after_1, after_2, ...]

    def __post_init__(self):
        steps = [e.step for e in self.events]
        if steps != sorted(steps):  # Validate sorted by step
            raise ValueError("StepShockSchedule events must be sorted by step")

        n = len(self.param_sets)
        for e in self.events:  # Validate param_set_idx in range
            if isinstance(e, ParamChangeEvent) and not (0 <= e.param_set_idx < n):
                raise ValueError(f"param_set_idx {e.param_set_idx} out of range [0, {n})")

    # ── Constructors (high-level → compiled schedule) ──
    @staticmethod
    def no_shock(params: SystemParams) -> StepShockSchedule:
        """No perturbation. Just wraps params for uniform API."""
        return StepShockSchedule(events=(), param_sets=(params,))

    @staticmethod
    def param_shock(
        step: int,
        before: SystemParams,
        after: SystemParams,
    ) -> StepShockSchedule:
        """Single regime change at `step`."""
        return StepShockSchedule(
            events=(ParamChangeEvent(step=step, param_set_idx=1),),
            param_sets=(before, after),
        )

    @staticmethod
    def state_shock(
        step: int,
        state_idx: int,
        eps: float,
        params: SystemParams,
    ) -> StepShockSchedule:
        """Single state perturbation at `step`."""
        return StepShockSchedule(
            events=(StatePerturbEvent(step=step, state_idx=state_idx, eps=eps),),
            param_sets=(params,),
        )

    @staticmethod
    def param_transition(
        before: SystemParams,
        after: SystemParams,
        start_step: int,
        end_step: int,
        n_inserts: int,
    ) -> StepShockSchedule:
        """Compile a parameter stepped-transition window into scheduled param updates."""
        total_transition_steps = end_step - start_step
        if total_transition_steps <= 0:
            raise ValueError(
                f"param transition requires end_step > start_step, got {start_step}..{end_step}"
            )
        if n_inserts <= 0:
            raise ValueError(f"n_inserts must be >= 1, got {n_inserts}")
        # CHG (2026-07-01, ruling 4 applied to the constructor path too): this
        # used to silently clamp n_inserts to the window, quietly compiling a
        # different transition than the caller asked for. One concept, one
        # behavior: raise, matching the generate path and Rust.
        if int(n_inserts) > total_transition_steps:
            raise ValueError(
                f"param transition asks for {n_inserts} scheduled updates, but the "
                f"window steps {start_step}..{end_step} allow only {total_transition_steps}. "
                f"Widen the [start_step, end_step] window or lower n_inserts."
            )

        param_sets = [before]
        events: list[ParamChangeEvent] = []

        n_insert_points = int(n_inserts)
        transition_step_offsets = [
            (insert_idx * total_transition_steps) // n_insert_points
            for insert_idx in range(1, n_insert_points + 1)
        ]

        for transition_step_offset in transition_step_offsets:
            alpha = transition_step_offset / total_transition_steps
            interpolated_params = _interpolate_params(before, after, alpha)
            param_sets.append(interpolated_params)
            transition_step = start_step + transition_step_offset
            events.append(ParamChangeEvent(step=transition_step, param_set_idx=len(param_sets) - 1))
        return StepShockSchedule(
            events=tuple(events),
            param_sets=tuple(param_sets),
        )

    __hash__ = record_hash


class SwitchSpec(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=False,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Two-trajectory switch at a given step.

    Semantically different from StepShockSchedule — not a perturbation
    within one integration, but a cut from one independent run to another.
    """

    switch_step: int  # row index where we cut
    params_b: SystemParams  # system B params
    initial_cond_b: np.ndarray  # system B initial condition
    system_b: str | None = None  # if None, same system as A (same-system switch)

    __hash__ = record_hash
    # For stochastic systems, B gets seed + 1 (handled in _generate_switch)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integrators (inlined — these are only called by dispatchers below)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _params_as_args(p: SystemParams) -> tuple:
    """intent/contract: convert attrs params to positional args in definition order."""
    # CHG: the parameter records are now Structs. Borrow their field values in
    # declaration order, retaining the original non-recursive tuple contract.
    return tuple(getattr(p, name) for name in p.__struct_fields__)


def _increment_scalar_parameter(
    params: SystemParams, event: ScalarParameterIncrementEvent
) -> SystemParams:
    if event.parameter_name not in _transition_field_names(params):
        raise ValueError(
            f"{event.parameter_name!r} is not a movable parameter of {type(params).__name__}"
        )
    current = getattr(params, event.parameter_name)
    if not np.isscalar(current) or not np.isfinite(current):
        raise ValueError("parameter increment requires a finite scalar field")
    updated = float(current) + event.delta
    if not np.isfinite(updated):
        raise ValueError("parameter increment produced a nonfinite value")
    return replace(params, **{event.parameter_name: updated})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integrator label
# The algorithm, where its code lives, and its version. The saved trajectory
# array does not carry this. integrator_label reads it back from the method
# the spec already names, so an old rk4 bank is not rewritten.
# Program: docs/programs/2026-11-02-integrators/
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class IntegratorLabel(
    Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Which stepper produced a trajectory.

    algorithm is the method name. implementation is the code that ran it
    (`serie`, or a Julia package name). version names that code. step is
    `fixed` or `adaptive`. dt is the output spacing. Tolerances are set only
    when the step is adaptive.
    """

    algorithm: str
    implementation: str
    version: str
    step: str
    dt: float | None = None
    absolute_tolerance: float | None = None
    relative_tolerance: float | None = None
    output_grid: str = "one row per requested dt, including the initial state"


def integrator_label(method: str, *, dt: float | None = None) -> IntegratorLabel:
    """The label of a method serie can run today.

    `tsit5_adaptive` names Julia OrdinaryDiffEqTsit5 2.1.3 at tolerances 1e-6.
    The generator runs it for coupled_roessler only.
    """
    if method == "rk4":
        return IntegratorLabel(
            algorithm="rk4",
            implementation="serie",
            version="classical-4-stage",
            step="fixed",
            dt=dt,
        )
    if method == "euler":
        return IntegratorLabel(
            algorithm="euler",
            implementation="serie",
            version="forward-euler",
            step="fixed",
            dt=dt,
        )
    if method == "tsit5_fixed":
        return IntegratorLabel(
            algorithm="tsit5_fixed",
            implementation="serie",
            version="OrdinaryDiffEqTsit5-2.1.3-tableau",
            step="fixed",
            dt=dt,
        )
    if method == "tsit5_adaptive":
        from .julia_tsit5 import (
            ADAPTIVE_ABSOLUTE_TOLERANCE,
            ADAPTIVE_RELATIVE_TOLERANCE,
            PINNED_ORDINARY_DIFFEQ_TSIT5_VERSION,
        )

        if dt is None or not dt > 0:
            raise ValueError("adaptive Tsit5 needs the output dt")
        return julia_tsit5_adaptive_label(
            version=PINNED_ORDINARY_DIFFEQ_TSIT5_VERSION,
            absolute_tolerance=ADAPTIVE_ABSOLUTE_TOLERANCE,
            relative_tolerance=ADAPTIVE_RELATIVE_TOLERANCE,
            dt=dt,
        )
    raise ValueError(f"no integrator label for method {method!r}")


# One table. DISCRETE_RUNNERS is these values, not a second handwritten set.
_DISCRETE_RUNNER = {
    "lg_ssm": "lg_ssm_exact",
    "slds": "slds",
    "seasonal_ar": "seasonal_ar",
    "seasonal_ar_bj": "seasonal_ar_bj",
    "garch": "garch",
}
_ODE_RUNNER = {
    "rk4": "ode_rk4",
    "euler": "ode_euler",
    "tsit5_fixed": "ode_tsit5_fixed",
}
DISCRETE_RUNNERS = frozenset(_DISCRETE_RUNNER.values())
CONTINUOUS_RUNNERS = frozenset((*_ODE_RUNNER.values(), "map", "euler_maruyama"))


def trajectory_runner(system: str, method: str) -> str:
    """The function that walks this system. The label is this name, not a second list.

    ``generate_from_trajectory_spec`` and the discrete and continuous
    generators switch on the same string.
    """
    if method == "tsit5_adaptive":
        if system != "coupled_roessler":
            raise ValueError(
                "adaptive Tsit5 is wired for coupled_roessler. "
                f"{system} stays on its preset stepper"
            )
        return "julia_tsit5_adaptive"
    if system == "ks":
        return "ks_etdrk4"
    if system in _DISCRETE_RUNNER:
        return _DISCRETE_RUNNER[system]
    if system in DISCRETE_SYSTEMS:
        raise ValueError(f"unknown discrete system {system}")
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system}")
    kind = SYSTEMS[system].kind
    if kind == "ode":
        try:
            return _ODE_RUNNER[method]
        except KeyError as error:
            raise ValueError(f"ODE method {method!r} has no runner") from error
    if kind == "sde":
        return "euler_maruyama"
    if kind == "map":
        return "map"
    if kind == "pde":
        return "ks_etdrk4"
    raise ValueError(f"unknown kind {kind} for {system}")


def integrator_label_for_spec(spec: TrajectorySpec) -> IntegratorLabel:
    """The label of the runner that walks this spec."""
    runner = trajectory_runner(spec.system.value, spec.method)
    if runner == "ode_rk4":
        return integrator_label("rk4", dt=spec.dt)
    if runner == "ode_euler":
        return integrator_label("euler", dt=spec.dt)
    if runner == "ode_tsit5_fixed":
        return integrator_label("tsit5_fixed", dt=spec.dt)
    if runner == "julia_tsit5_adaptive":
        return integrator_label("tsit5_adaptive", dt=spec.dt)
    if runner == "euler_maruyama":
        return IntegratorLabel(
            algorithm="euler-maruyama",
            implementation="serie",
            version="euler-maruyama",
            step="fixed",
            dt=spec.dt,
        )
    if runner == "lg_ssm_exact":
        return IntegratorLabel(
            algorithm="exact-discretization",
            implementation="serie",
            version="van-loan",
            step="exact",
            dt=spec.dt,
            output_grid="one row per dt; the transition is the exact linear discretization",
        )
    if runner == "map":
        return IntegratorLabel(
            algorithm="map",
            implementation="serie",
            version="discrete-map",
            step="index",
            dt=spec.dt,
            output_grid="one row per iteration",
        )
    if runner == "ks_etdrk4":
        return IntegratorLabel(
            algorithm="etdrk4",
            implementation="serie",
            version="etdrk4",
            step="fixed",
            dt=spec.dt,
        )
    if runner in {"slds", "seasonal_ar", "seasonal_ar_bj", "garch"}:
        return IntegratorLabel(
            algorithm=runner,
            implementation="serie",
            version="index-step",
            step="index",
            dt=spec.dt,
            output_grid="one row per index; the catalog method field is not the stepper",
        )
    raise ValueError(f"no integrator label for runner {runner}")


def julia_tsit5_adaptive_label(
    *,
    version: str,
    absolute_tolerance: float,
    relative_tolerance: float,
    dt: float,
) -> IntegratorLabel:
    """Name Julia's adaptive Tsit5. This does not integrate anything."""
    if version == "":
        raise ValueError("Julia Tsit5 needs the OrdinaryDiffEqTsit5 version")
    if not absolute_tolerance > 0 or not relative_tolerance > 0:
        raise ValueError("adaptive Tsit5 tolerances must be positive")
    if not dt > 0:
        raise ValueError("output dt must be positive")
    return IntegratorLabel(
        algorithm="tsit5",
        implementation="OrdinaryDiffEqTsit5",
        version=version,
        step="adaptive",
        dt=dt,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        output_grid="saved on the requested dt; internal steps are the solver's",
    )


# Tsitouras 5(4), fixed step. Coefficients are the Float64 tableau in
# OrdinaryDiffEqTsit5 2.1.3, src/tsit_tableaus.jl. The fifth-order update is
# the a7 row. The embedded fourth-order weights are not used: this method
# does not change dt. The adaptive Julia solver is a different recording.
_TSIT5_C1 = 0.161
_TSIT5_C2 = 0.327
_TSIT5_C3 = 0.9
_TSIT5_C4 = 0.9800255409045097
_TSIT5_A21 = 0.161
_TSIT5_A31 = -0.008480655492356989
_TSIT5_A32 = 0.335480655492357
_TSIT5_A41 = 2.8971530571054935
_TSIT5_A42 = -6.359448489975075
_TSIT5_A43 = 4.3622954328695815
_TSIT5_A51 = 5.325864828439257
_TSIT5_A52 = -11.748883564062828
_TSIT5_A53 = 7.4955393428898365
_TSIT5_A54 = -0.09249506636175525
_TSIT5_A61 = 5.86145544294642
_TSIT5_A62 = -12.92096931784711
_TSIT5_A63 = 8.159367898576159
_TSIT5_A64 = -0.071584973281401
_TSIT5_A65 = -0.028269050394068383
_TSIT5_A71 = 0.09646076681806523
_TSIT5_A72 = 0.01
_TSIT5_A73 = 0.4798896504144996
_TSIT5_A74 = 1.379008574103742
_TSIT5_A75 = -3.290069515436081
_TSIT5_A76 = 2.324710524099774


def _tsit5_fixed_step(
    f: Callable,
    state: np.ndarray,
    t: float,
    dt: float,
    args: tuple,
) -> np.ndarray:
    """One fixed step of Tsit5. Stage times follow the tableau, including t + dt."""

    def slope(at_state: np.ndarray, at_time: float) -> np.ndarray:
        return f(at_state, at_time, *args)

    k1 = slope(state, t)
    k2 = slope(state + dt * _TSIT5_A21 * k1, t + _TSIT5_C1 * dt)
    k3 = slope(state + dt * (_TSIT5_A31 * k1 + _TSIT5_A32 * k2), t + _TSIT5_C2 * dt)
    k4 = slope(
        state + dt * (_TSIT5_A41 * k1 + _TSIT5_A42 * k2 + _TSIT5_A43 * k3),
        t + _TSIT5_C3 * dt,
    )
    k5 = slope(
        state + dt * (_TSIT5_A51 * k1 + _TSIT5_A52 * k2 + _TSIT5_A53 * k3 + _TSIT5_A54 * k4),
        t + _TSIT5_C4 * dt,
    )
    k6 = slope(
        state
        + dt
        * (_TSIT5_A61 * k1 + _TSIT5_A62 * k2 + _TSIT5_A63 * k3 + _TSIT5_A64 * k4 + _TSIT5_A65 * k5),
        t + dt,
    )
    return state + dt * (
        _TSIT5_A71 * k1
        + _TSIT5_A72 * k2
        + _TSIT5_A73 * k3
        + _TSIT5_A74 * k4
        + _TSIT5_A75 * k5
        + _TSIT5_A76 * k6
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ODE integrator (Euler / RK4 / fixed-step Tsit5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _run_ode_py(
    f: Callable,
    x0: np.ndarray,
    steps: int,
    dt: float,
    t0: float,
    params: SystemParams,
    method: str,
    time_dependent: bool,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Pure Python ODE integration with optional typed shock schedule."""

    traj = np.empty((steps + 1, x0.size), dtype=np.float64)
    x = x0.copy()
    t = t0
    traj[0] = x
    p = params
    events = schedule.events if schedule is not None else ()
    event_idx = 0

    for i in range(1, steps + 1):
        # Apply all events scheduled for this step.
        while event_idx < len(events) and events[event_idx].step == i:
            e = events[event_idx]
            if isinstance(e, StatePerturbEvent):
                x = x.copy()
                x[e.state_idx] += e.eps
            elif isinstance(e, StateCoupledImpulseEvent):
                if any(index >= x.size for index in e.targets) or any(
                    term.state_idx >= x.size for term in e.terms
                ):
                    raise ValueError("state-coupled impulse coordinate is outside the ODE state")
                if any(term.lag > 0 and i - 1 - term.lag < 0 for term in e.terms):
                    raise ValueError(
                        "state-coupled impulse needs observed history before its event step"
                    )
                values = [
                    term.weight
                    * (
                        x[term.state_idx]
                        if term.lag == 0
                        else traj[i - 1 - term.lag, term.state_idx]
                    )
                    for term in e.terms
                ]
                amount = e.coefficient * (
                    sum(values) if e.operation is StateExpressionOperation.SUM else np.prod(values)
                )
                if not np.isfinite(amount):
                    raise ValueError("state-coupled impulse produced a nonfinite amount")
                x = x.copy()
                x[list(e.targets)] += amount
            elif isinstance(e, ScalarParameterIncrementEvent):
                p = _increment_scalar_parameter(p, e)
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p = cast(SystemParams, schedule.param_sets[e.param_set_idx])
            event_idx += 1

        # Integration step
        if method == "euler":
            k1 = f(x, t, *_params_as_args(p))
            x = x + dt * k1

        elif method == "rk4":
            if time_dependent:
                k1 = f(x, t, *_params_as_args(p))
                k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt, *_params_as_args(p))
                k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt, *_params_as_args(p))
                k4 = f(x + dt * k3, t + dt, *_params_as_args(p))
            else:
                k1 = f(x, t, *_params_as_args(p))
                k2 = f(x + 0.5 * dt * k1, t, *_params_as_args(p))
                k3 = f(x + 0.5 * dt * k2, t, *_params_as_args(p))
                k4 = f(x + dt * k3, t, *_params_as_args(p))
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method == "tsit5_fixed":
            x = _tsit5_fixed_step(f, x, t, dt, _params_as_args(p))
        else:
            raise ValueError(f"Unknown ODE method: {method}")

        t += dt

        if not np.isfinite(x).all():
            raise ValueError(f"Numerical instability at step {i}: state={x}")
        traj[i] = x

    return traj


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Map integrator (discrete iteration)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _run_map_py(
    f: Callable,
    x0: np.ndarray,
    steps: int,
    params: SystemParams,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Pure Python map iteration with optional typed shock schedule."""

    traj = np.empty((steps + 1, x0.size), dtype=np.float64)
    x = x0.copy()
    traj[0] = x
    p = params
    events = schedule.events if schedule is not None else ()
    event_idx = 0

    for i in range(1, steps + 1):
        # Apply all events scheduled for this step.
        while event_idx < len(events) and events[event_idx].step == i:
            e = events[event_idx]
            if isinstance(e, StatePerturbEvent):
                x = x.copy()
                x[e.state_idx] += e.eps
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p = cast(SystemParams, schedule.param_sets[e.param_set_idx])
            event_idx += 1

        x = f(x, *_params_as_args(p))

        if not np.isfinite(x).all():
            raise ValueError(f"Numerical instability at step {i}: state={x}")
        traj[i] = x

    return traj


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SDE integrator (Euler-Maruyama)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _run_sde_py(
    f_drift: Callable,
    x0: np.ndarray,
    steps: int,
    dt: float,
    t0: float,
    params: SystemParams,
    sigma_noise: float,
    seed: int,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Euler-Maruyama SDE integration."""

    rng = np.random.default_rng(seed)
    traj = np.empty((steps + 1, x0.size), dtype=np.float64)
    x = x0.copy()
    t = t0
    traj[0] = x
    p = params
    sigma = sigma_noise
    sqrt_dt = np.sqrt(dt)
    events = schedule.events if schedule is not None else ()
    event_idx = 0

    for i in range(1, steps + 1):
        # Apply all events scheduled for this step.
        while event_idx < len(events) and events[event_idx].step == i:
            e = events[event_idx]
            if isinstance(e, StatePerturbEvent):
                x = x.copy()
                x[e.state_idx] += e.eps
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p = cast(SystemParams, schedule.param_sets[e.param_set_idx])
                sigma = cast(OUParams | DoubleWellParams, p).sigma_noise
            event_idx += 1

        drift = f_drift(x, t, *_params_as_args(p))
        noise = sigma * sqrt_dt * rng.standard_normal(x.shape)
        x = x + drift * dt + noise
        t += dt

        if not np.isfinite(x).all():
            raise ValueError(f"Numerical instability at step {i}: state={x}")
        traj[i] = x

    return traj


def _require_stationary_euler_maruyama_world(trajectory_spec: TrajectorySpec) -> None:
    """Refuse a world whose futures `simulate_euler_maruyama_futures` would not reproduce.

    Correctness guards, each naming the field that would make a re-simulated future
    follow a different law from the recorded one: an SDE kind (`system`), Euler
    stepping (`method`), a drift that does not read absolute time (the catalog's
    `time_dependent`; the re-simulation passes t = 0), and no shock (`shocks`), whose
    parameters would depend on where in the series the origin sits.
    """
    system = SYSTEMS.get(trajectory_spec.system.value)
    if system is None or system.kind != "sde":
        raise ValueError(
            f"system {trajectory_spec.system.value!r} is not a catalog SDE; only an Euler–Maruyama world is re-simulated"
        )
    if trajectory_spec.method != "euler":
        raise ValueError(
            f"method {trajectory_spec.method!r}: the catalog SDE integrator is Euler–Maruyama ('euler')"
        )
    if system.time_dependent:
        raise ValueError(
            f"system {system.name!r} has a time-dependent drift; re-simulation does not carry absolute time"
        )
    if not trajectory_spec.shocks.is_stationary:
        raise ValueError(
            "shocks: a world with a parameter shock has origin-dependent parameters; re-simulation refuses it"
        )


def double_well_euler_escape_radius(trajectory_spec: TrajectorySpec) -> float:
    """r* = sqrt(a + 2/dt): past it the double well's noiseless Euler step moves every value further from zero.

    CHG 2026-10-30 (owner, asked to decide among the recommendations for study 1059: "你来推荐，我明天再看", then
    "意思是你先斩后奏报告，然后我时候再说"; slice 2 of docs/programs/2026-09-30-beyond-nll-probabilistic-dynamics/
    2026-10-06-simulated-true-law). The world's recorded step without its noise is the map

        g(x) = x + dt (a x − x³),   g(x)/x = 1 + a dt − dt x²

    and |g(x)| > |x| on the outside (g(x)/x < −1) exactly when x² > a + 2/dt; there |g(x)| − |x| =
    |x| (dt x² − a dt − 2) grows with |x|, so a value past r* is sent further out at every step and its cube
    overflows float64 within about eight steps. For the catalog world (a = 1.5, dt = 0.5) r* = sqrt(5.5) = 2.3452,
    the threshold of the slice's computed escape rate (2.88e-10 per path and step). Refused for any other world
    (a correctness guard naming the field `system`): the interval is derived for this cubic drift only.
    """
    _require_stationary_euler_maruyama_world(trajectory_spec)
    if trajectory_spec.system.value != "doublewell":
        raise ValueError(
            f"system {trajectory_spec.system.value!r}: the escape radius sqrt(a + 2/dt) is derived for the "
            "double well's Euler step x + dt (a x - x^3) only; no other world's escape is defined"
        )
    params = cast(DoubleWellParams, trajectory_spec.params)
    squared = float(params.a) + 2.0 / float(trajectory_spec.dt)
    if not (np.isfinite(squared) and squared > 0.0):
        raise ValueError(
            f"a + 2/dt = {squared}: the double well's escape radius needs a positive finite a + 2/dt"
        )
    return float(np.sqrt(squared))


def simulate_euler_maruyama_futures(
    trajectory_spec: TrajectorySpec,
    start_states: np.ndarray,
    innovations: np.ndarray,
    *,
    escape_radius: float | None,
) -> np.ndarray:
    """Futures of a catalog SDE world from given exact states, under given standard normal innovations.

    CHG 2026-10-06 (owner, 23:10 EDT: test a forecaster against the true conditional
    law of a stochastic world that can only be sampled, "先接线"). The same step as
    `_run_sde_py`, operation for operation, applied to M paths at once:

        x_{k+1} = x_k + f(x_k, t; θ) dt + σ sqrt(dt) z_{k+1},   z ~ N(0, I_D)

    start_states [M, D] float64     the exact simulator state x_{t₀} of each path
      │ x = start_statesᵀ            [D, M]: the catalog drift reads state[0], ..., so a
      │                                       column is one path (the drift is elementwise)
      │ for k in 0..K−1:  x ← x + f(x) dt + σ sqrt(dt) innovations[k]     [D, M]
      ▼
    futures [M, K, D] float64       futures[m, k] = x_{t₀+k+1} of path m

    innovations [K, D, M] float64 are the z's. With M = 1, x_{t₀} = x0 and
    innovations = default_rng(seed).standard_normal((steps, D, 1)), the result is
    `_run_sde_py`'s trajectory rows 1..steps bit for bit (tests/test_simulated_true_law.py);
    a block draw of shape (K, D, M) from one generator is the same stream as K
    successive draws of shape (D, M). The step is the world's recording step: the
    catalog integrator records every Euler step, so no sub-step exists to reproduce.

    CHG 2026-10-30 (owner: "你来推荐，我明天再看", then "意思是你先斩后奏报告，然后我时候再说"; study 1059):
    `escape_radius` is required and has no default.

        escape_radius None    as before this date, operation for operation: the re-simulation refuses
                              ("Numerical instability at re-simulated step k") as soon as any path holds a
                              value that is not finite
        escape_radius r       a path is escaped from the first step k at which some coordinate has |x| > r;
                              its column holds NaN from step k on, so it never overflows, and every later
                              step of it is NaN. The refusal above still applies to the paths not escaped.
                              futures[m] holds a NaN exactly when path m escaped; the other columns are the
                              numbers escape_radius None computes, bit for bit (the step is elementwise
                              over columns).
    """
    _require_stationary_euler_maruyama_world(trajectory_spec)
    system = SYSTEMS[trajectory_spec.system.value]
    start = np.asarray(start_states)
    noise = np.asarray(innovations)
    if start.dtype != np.float64 or noise.dtype != np.float64:
        raise ValueError(
            "start_states and innovations must be float64, the integrator's own precision"
        )
    if start.ndim != 2 or noise.ndim != 3 or noise.shape[1:] != (start.shape[1], start.shape[0]):
        raise ValueError(
            f"start_states [M, D] and innovations [K, D, M] disagree: {start.shape} and {noise.shape}"
        )
    if system.dim not in (-1, start.shape[1]):
        raise ValueError(
            f"system {system.name!r} has {system.dim} coordinates, not {start.shape[1]}"
        )
    p = trajectory_spec.params
    sigma = cast(OUParams | DoubleWellParams, p).sigma_noise
    dt = trajectory_spec.dt
    sqrt_dt = np.sqrt(dt)
    arguments = _params_as_args(p)
    x = start.T.copy()  # [D, M]
    futures = np.empty((noise.shape[0],) + x.shape, dtype=np.float64)  # [K, D, M]
    t = 0.0
    if escape_radius is None:
        for k in range(noise.shape[0]):
            drift = system.f_py(x, t, *arguments)
            x = x + drift * dt + sigma * sqrt_dt * noise[k]
            t += dt
            if not np.isfinite(x).all():
                raise ValueError(f"Numerical instability at re-simulated step {k + 1}")
            futures[k] = x
        return np.ascontiguousarray(futures.transpose(2, 0, 1))  # [M, K, D]
    # CHG 2026-10-30: the escaped-path branch (docstring). escaped [M] accumulates over the steps; an escaped
    # column is NaN, and NaN stays NaN through the drift and the noise, so its |x| > r is False afterwards and
    # `escaped` keeps it by the |=.
    radius = float(escape_radius)
    if not (np.isfinite(radius) and radius > 0.0):
        raise ValueError(
            f"escape_radius = {escape_radius!r}: an escape radius is a positive finite number"
        )
    escaped = np.zeros(x.shape[1], dtype=bool)  # [M]
    for k in range(noise.shape[0]):
        drift = system.f_py(x, t, *arguments)
        x = x + drift * dt + sigma * sqrt_dt * noise[k]
        t += dt
        escaped |= (np.abs(x) > radius).any(axis=0)
        x[:, escaped] = np.nan
        if not np.isfinite(x[:, ~escaped]).all():
            raise ValueError(
                f"Numerical instability at re-simulated step {k + 1} in a path that had not escaped"
            )
        futures[k] = x
    return np.ascontiguousarray(futures.transpose(2, 0, 1))  # [M, K, D]


def euler_maruyama_path_log_density(
    trajectory_spec: TrajectorySpec, innovations: np.ndarray
) -> np.ndarray:
    """log p(x_{t₀+1}, …, x_{t₀+K} | x_{t₀}) [M] of each re-simulated path, from its own innovations.

    CHG 2026-10-06 (same owner request). Each Euler–Maruyama step of a catalog SDE
    is Gaussian given the previous state, x_{k+1} | x_k ~ N(x_k + f(x_k) dt, q² I_D),
    q = σ sqrt(dt), so the joint density of the K recorded steps is the product of
    the transition densities, and x_{k+1} − x_k − f(x_k) dt = q z_{k+1}:

        log p = Σ_{k,d} [ −½ z_{k+1,d}² − log q − ½ log 2π ]

    This is the exact density of the discrete-time world the series was recorded
    from (the recording step is the Euler step), not of the continuous SDE, whose
    K-step law has no closed form. innovations [K, D, M] float64, the z's that
    `simulate_euler_maruyama_futures` consumed.
    """
    _require_stationary_euler_maruyama_world(trajectory_spec)
    noise = np.asarray(innovations)
    if noise.dtype != np.float64 or noise.ndim != 3:
        raise ValueError("innovations must be float64 [K, D, M]")
    sigma = cast(OUParams | DoubleWellParams, trajectory_spec.params).sigma_noise
    q = float(sigma * np.sqrt(trajectory_spec.dt))
    if not (np.isfinite(q) and q > 0.0):
        raise ValueError(
            f"sigma_noise sqrt(dt) = {q}: a path density needs a positive finite noise scale"
        )
    count = noise.shape[0] * noise.shape[1]
    return -0.5 * np.sum(noise**2, axis=(0, 1)) - count * (np.log(q) + 0.5 * np.log(2.0 * np.pi))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KS PDE (ETDRK4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _ks_etdrk4_coeffs(L: np.ndarray, dt: float):
    """Compute ETDRK4 coefficients (Kassam & Trefethen 2005)."""
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR * LR)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR * LR + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    return E, E2, Q, f1, f2, f3


def _run_ks_etdrk4(
    x0: np.ndarray,
    steps: int,
    dt: float,
    params: KSParams,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """KS PDE integration with ETDRK4."""

    nx, Lx, nu, dealias = params.nx, params.Lx, params.nu, params.dealias

    # Fourier grid
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    k2, k4 = k * k, k * k * k * k

    def build_coeffs(nu_val):
        L = nu_val * k2 - k4
        return _ks_etdrk4_coeffs(L, dt)

    E, E2, Q, f1, f2, f3 = build_coeffs(nu)

    # Dealiasing mask (2/3 rule)
    if dealias:
        kmax = np.max(np.abs(k))
        dealias_mask = (np.abs(k) <= (2.0 / 3.0) * kmax).astype(float)
    else:
        dealias_mask = np.ones(nx)

    def nonlinear_hat(u):
        u2_hat = np.fft.fft(u * u) * dealias_mask
        return -0.5j * k * u2_hat

    # Integration
    u = x0.copy()
    v = np.fft.fft(u)
    traj = np.empty((steps + 1, nx), dtype=np.float64)
    traj[0] = u
    events = schedule.events if schedule is not None else ()
    event_idx = 0

    for i in range(1, steps + 1):
        # Apply all events scheduled for this step.
        while event_idx < len(events) and events[event_idx].step == i:
            e = events[event_idx]
            if isinstance(e, StatePerturbEvent):
                # KS state perturbations shift the whole field.
                u = u + e.eps
                v = np.fft.fft(u)
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p_next = cast(KSParams, schedule.param_sets[e.param_set_idx])
                # CHG (2026-07-02, user-approved guard): the spectral grid k and
                # dealias mask were built from the ORIGINAL Lx/nx/dealias and
                # never rebuilt, so a change to any of them would silently not
                # take effect. Fail loudly instead.
                if (p_next.Lx, p_next.nx, p_next.dealias) != (Lx, nx, dealias):
                    raise ValueError(
                        "KS param transitions may only change nu: Lx/nx/dealias are "
                        f"structural (grid built once). Got Lx={p_next.Lx} nx={p_next.nx} "
                        f"dealias={p_next.dealias}, expected Lx={Lx} nx={nx} dealias={dealias}."
                    )
                E, E2, Q, f1, f2, f3 = build_coeffs(p_next.nu)
            event_idx += 1

        Nv = nonlinear_hat(u)
        a = E2 * v + Q * Nv
        ua = np.fft.ifft(a).real
        Na = nonlinear_hat(ua)

        b = E2 * v + Q * Na
        ub = np.fft.ifft(b).real
        Nb = nonlinear_hat(ub)

        c = E2 * a + Q * (2 * Nb - Nv)
        uc = np.fft.ifft(c).real
        Nc = nonlinear_hat(uc)

        v = E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        u = np.fft.ifft(v).real

        if not np.isfinite(u).all():
            raise ValueError(f"KS instability at step {i}")
        traj[i] = u

    return traj


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Discrete systems (SLDS, Seasonal AR, GARCH)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _run_slds(
    x0: np.ndarray,
    steps: int,
    params: SLDSParams,
    seed: int,
    schedule: StepShockSchedule | None,
    *,
    regimes_out: np.ndarray | None = None,
) -> np.ndarray:
    """Run Switching LDS."""

    # CHG: an optional output retains true modes without changing any RNG draw.
    if regimes_out is not None and (
        regimes_out.shape != (steps + 1,)
        or not np.issubdtype(regimes_out.dtype, np.integer)
        or not regimes_out.flags.writeable
    ):
        raise ValueError("regimes_out must be a writable integer array of shape [steps+1]")
    rng = np.random.default_rng(seed)
    traj = np.empty((steps + 1, 1), dtype=np.float64)
    x = float(x0[0])
    traj[0, 0] = x
    p = params

    # Start regime randomly
    z = 1 if rng.random() < 0.5 else 2
    if regimes_out is not None:
        regimes_out[0] = z
    events = schedule.events if schedule is not None else ()
    event_idx = 0

    for t in range(1, steps + 1):
        # Apply all events scheduled for this step.
        while event_idx < len(events) and events[event_idx].step == t:
            e = events[event_idx]
            if isinstance(e, StatePerturbEvent):
                x += e.eps
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p = cast(SLDSParams, schedule.param_sets[e.param_set_idx])
            event_idx += 1

        # Transition regime
        if z == 1:
            z = 1 if rng.random() < p.p11 else 2
        else:
            z = 2 if rng.random() < p.p22 else 1
        if regimes_out is not None:
            regimes_out[t] = z

        # CHG (user-approved 2026-07-01): dynamics use the NEWLY drawn regime,
        # x_t = A_{z_t}·x_{t-1} + sqrt(Q_{z_t})·η_t, matching the SLDSParams
        # docstring. Previously the pre-transition regime's (A, Q) stepped, so
        # a switch drawn at step t only took effect at t+1. slds /
        # slds_param_shock / slds_switch trajectories differ from all pre-fix
        # runs.
        A, Q = (p.A1, p.Q1) if z == 1 else (p.A2, p.Q2)

        # Evolve
        x = A * x + rng.normal(0.0, np.sqrt(Q))
        traj[t, 0] = x

    return traj


def _run_seasonal_ar(
    x0: np.ndarray,
    steps: int,
    params: SeasonalARParams,
    seed: int,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Run Seasonal AR."""

    rng = np.random.default_rng(seed)
    traj = np.empty((steps + 1, 1), dtype=np.float64)
    y = float(x0[0])
    traj[0, 0] = y
    p = params
    events = schedule.events if schedule is not None else ()
    event_idx = 0

    for t in range(1, steps + 1):
        # Apply all events scheduled for this step.
        while event_idx < len(events) and events[event_idx].step == t:
            e = events[event_idx]
            if isinstance(e, StatePerturbEvent):
                y += e.eps
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p = cast(SeasonalARParams, schedule.param_sets[e.param_set_idx])
            event_idx += 1

        a_t = p.a0 + p.amp_drift * t
        seasonal = a_t * np.sin(2.0 * np.pi * t / p.S)
        y = seasonal + p.phi * y + rng.normal(0.0, p.sigma)
        traj[t, 0] = y

    return traj


def _run_seasonal_ar_bj(
    x0: np.ndarray,
    steps: int,
    params: SeasonalARBJParams,
    seed: int,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Run the Box–Jenkins seasonal AR: y_t = φ y_{t-1} + Φ y_{t-S} + σ ε_t, y_s = 0 for s < 0.

    CHG 2026-10-06 (owner). Written as `_run_seasonal_ar`: row t of the result is
    step t, row 0 is x0, and each step draws one normal of scale σ from the seed's
    generator (one standard normal per step, as `_run_seasonal_ar` draws). The
    values before row 0 are zero, so for t < S the seasonal term is zero.
    """

    rng = np.random.default_rng(seed)
    traj = np.empty((steps + 1, 1), dtype=np.float64)
    y = float(x0[0])
    traj[0, 0] = y
    p = params
    events = schedule.events if schedule is not None else ()
    event_idx = 0

    for t in range(1, steps + 1):
        # Apply all events scheduled for this step.
        while event_idx < len(events) and events[event_idx].step == t:
            e = events[event_idx]
            if isinstance(e, StatePerturbEvent):
                y += e.eps
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p = cast(SeasonalARBJParams, schedule.param_sets[e.param_set_idx])
            event_idx += 1

        seasonal = traj[t - p.S, 0] if t - p.S >= 0 else 0.0
        y = p.phi * y + p.seasonal_phi * seasonal + rng.normal(0.0, p.sigma)
        traj[t, 0] = y

    return traj


def _run_garch(
    x0: np.ndarray,
    steps: int,
    params: GARCHParams,
    seed: int,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Run GARCH(1,1)."""

    # init is fully supplied by x0 (no sigma2_0 logic here)
    r0 = float(x0[0])
    sigma2 = float(x0[1])

    rng = np.random.default_rng(seed)
    traj = np.empty((steps + 1, 2), dtype=np.float64)  # return [r, sigma2]
    traj[0, 0] = r0
    traj[0, 1] = sigma2  # store initial sigma2

    p = params
    events = schedule.events if schedule is not None else ()
    event_idx = 0

    for t in range(1, steps + 1):
        # Apply all events scheduled for this step.
        while event_idx < len(events) and events[event_idx].step == t:
            e = events[event_idx]
            if isinstance(e, StatePerturbEvent):
                traj[t - 1, 0] += e.eps
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p = cast(GARCHParams, schedule.param_sets[e.param_set_idx])
            event_idx += 1

        r_prev = traj[t - 1, 0]  # read r from traj
        sigma2 = p.omega + p.alpha * (r_prev**2) + p.beta * sigma2
        sigma2 = max(sigma2, 0.0)

        r_t = np.sqrt(sigma2) * rng.standard_normal()
        traj[t, 0] = r_t
        traj[t, 1] = sigma2  # store sigma2 state

    return traj  # (T,2)


def _run_lg_ssm(
    x0: np.ndarray,
    steps: int,
    dt: float,
    params: LGSSMParams,
    seed: int,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Run the LG-SSM by exact discretization (roadmap §15.2):

      x_{k+1} = A·x_k + chol(Q)·ε_k,   ε_k ~ N(0, I_D)

    A = exp(F·dt) and Q come from lg_ssm_discretize_exact (Van Loan block
    exponential); this loop performs NO Euler stepping. One
    rng.standard_normal(D) block per step from a single Generator, matching
    _run_sde_py's draw convention.

    A ParamChangeEvent swaps the drift/diffusion params, so the discretized
    (A, Q, chol(Q)) are rebuilt exactly at the event step.
    """
    rng = np.random.default_rng(seed)
    p = params
    dim = 2 * len(p.gammas)
    A, Q = lg_ssm_discretize_exact(p, dt)
    chol_q = np.linalg.cholesky(Q)

    traj = np.empty((steps + 1, dim), dtype=np.float64)
    x = x0.copy()
    traj[0] = x
    events = schedule.events if schedule is not None else ()
    event_idx = 0

    for t in range(1, steps + 1):
        # Apply all events scheduled for this step.
        while event_idx < len(events) and events[event_idx].step == t:
            e = events[event_idx]
            if isinstance(e, StatePerturbEvent):
                x = x.copy()
                x[e.state_idx] += e.eps
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p = cast(LGSSMParams, schedule.param_sets[e.param_set_idx])
                if 2 * len(p.gammas) != dim:
                    raise ValueError(
                        f"lg_ssm param shock must keep the block count: state dim is {dim}, "
                        f"shocked params imply {2 * len(p.gammas)}."
                    )
                A, Q = lg_ssm_discretize_exact(p, dt)
                chol_q = np.linalg.cholesky(Q)
            event_idx += 1

        x = A @ x + chol_q @ rng.standard_normal(dim)
        traj[t] = x

    return traj


# =============================================================================
# Unified Configuration
# =============================================================================


def _generate_continuous_pure(
    *,
    total_steps: int,
    x0: np.ndarray,
    system: str,
    schedule: StepShockSchedule | None,
    params: SystemParams,
    dt: float,
    t0: float,
    method: str,
    seed: int,
) -> np.ndarray:
    """Generate ODE/SDE/Map trajectory with explicit inputs (no self)."""
    spec = SYSTEMS[system]

    if (
        spec.kind != "ode"
        and schedule is not None
        and any(
            isinstance(event, (StateCoupledImpulseEvent, ScalarParameterIncrementEvent))
            for event in schedule.events
        )
    ):
        raise ValueError(
            "exact-step state-coupled impulses and parameter increments require an ODE"
        )

    active_params = schedule.param_sets[0] if schedule else params
    runner = trajectory_runner(system, method)

    if runner in {"ode_rk4", "ode_euler", "ode_tsit5_fixed"}:
        return _run_ode_py(
            f=spec.f_py,
            x0=x0,
            steps=total_steps,
            dt=dt,
            t0=t0,
            params=active_params,
            method=method,
            time_dependent=spec.time_dependent,
            schedule=schedule,
        )
    if runner == "map":
        return _run_map_py(
            f=spec.f_py,
            x0=x0,
            steps=total_steps,
            params=active_params,
            schedule=schedule,
        )
    if runner == "euler_maruyama":
        sigma = cast(OUParams | DoubleWellParams, active_params).sigma_noise
        return _run_sde_py(
            f_drift=spec.f_py,
            x0=x0,
            steps=total_steps,
            dt=dt,
            t0=t0,
            params=active_params,
            sigma_noise=sigma,
            seed=seed,
            schedule=schedule,
        )
    raise ValueError(f"runner {runner} is not a continuous stepper")


def _generate_discrete_pure(
    *,
    total_steps: int,
    x0: np.ndarray,
    system: str,
    params: SystemParams,
    dt: float,
    seed: int,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Generate discrete system trajectory with explicit inputs (no self).

    dt is the real sampling interval for lg_ssm (exact discretization); the
    index-stepped simulators (slds/seasonal_ar/garch) ignore it.
    """
    active_params = schedule.param_sets[0] if schedule else params
    runner = trajectory_runner(system, "euler")

    if runner == "lg_ssm_exact":
        return _run_lg_ssm(
            x0=x0,
            steps=total_steps,
            dt=dt,
            params=cast(LGSSMParams, active_params),
            seed=seed,
            schedule=schedule,
        )
    if runner == "slds":
        return _run_slds(
            x0=x0,
            steps=total_steps,
            params=cast(SLDSParams, active_params),
            seed=seed,
            schedule=schedule,
        )
    if runner == "seasonal_ar":
        return _run_seasonal_ar(
            x0=x0,
            steps=total_steps,
            params=cast(SeasonalARParams, active_params),
            seed=seed,
            schedule=schedule,
        )
    if runner == "seasonal_ar_bj":
        return _run_seasonal_ar_bj(
            x0=x0,
            steps=total_steps,
            params=cast(SeasonalARBJParams, active_params),
            seed=seed,
            schedule=schedule,
        )
    if runner == "garch":
        return _run_garch(
            x0=x0,
            steps=total_steps,
            params=cast(GARCHParams, active_params),
            seed=seed,
            schedule=schedule,
        )
    raise ValueError(f"runner {runner} is not a discrete stepper")


def _generate_ks_pure(
    *,
    total_steps: int,
    x0: np.ndarray,
    dt: float,
    params: SystemParams,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Generate KS PDE trajectory with explicit inputs (no self)."""
    active_params = schedule.param_sets[0] if schedule else params
    return _run_ks_etdrk4(
        x0=x0,
        steps=total_steps,
        dt=dt,
        params=cast(KSParams, active_params),
        schedule=schedule,
    )


def _generate_switch_python_pure(
    *,
    total_steps: int,
    x0: np.ndarray,
    system: str,
    params: SystemParams,
    dt: float,
    method: str,
    t0: float,
    seed: int,
    sw: SwitchSpec,
) -> np.ndarray:
    """Python fallback: two trajectory generations + trajectory switch."""
    traj_a = _generate_trajectory_pure(
        total_steps=total_steps,
        x0=x0,
        system=system,
        params=params,
        dt=dt,
        method=method,
        t0=t0,
        seed=seed,
        schedule=None,
        switch=None,
    ).astype(np.float64)

    system_b = sw.system_b or system
    traj_b = _generate_trajectory_pure(
        total_steps=total_steps,
        x0=sw.initial_cond_b,
        system=system_b,
        params=sw.params_b,
        dt=dt,
        method=method,
        t0=t0,
        seed=seed + 1,
        schedule=None,
        switch=None,
    ).astype(np.float64)

    traj = traj_b.copy()
    traj[: sw.switch_step + 1] = traj_a[: sw.switch_step + 1]
    return traj


def _generate_switch_pure(
    *,
    total_steps: int,
    x0: np.ndarray,
    system: str,
    params: SystemParams,
    dt: float,
    method: str,
    t0: float,
    seed: int,
    sw: SwitchSpec,
) -> np.ndarray:
    """Switch between two independent trajectories at sw.switch_step."""
    return _generate_switch_python_pure(
        total_steps=total_steps,
        x0=x0,
        system=system,
        params=params,
        dt=dt,
        method=method,
        t0=t0,
        seed=seed,
        sw=sw,
    )


def _generate_trajectory_pure(
    *,
    total_steps: int,
    x0: np.ndarray,
    system: str,
    params: SystemParams,
    dt: float,
    method: str,
    t0: float,
    seed: int,
    schedule: StepShockSchedule | None,
    switch: SwitchSpec | None,
) -> np.ndarray:
    if switch is not None:
        return _generate_switch_pure(
            total_steps=total_steps,
            x0=x0,
            system=system,
            params=params,
            dt=dt,
            method=method,
            t0=t0,
            seed=seed,
            sw=switch,
        )

    runner = trajectory_runner(system, method)
    if runner == "ks_etdrk4":
        return _generate_ks_pure(
            total_steps=total_steps,
            x0=x0,
            dt=dt,
            params=params,
            schedule=schedule,
        )
    if runner in DISCRETE_RUNNERS:
        return _generate_discrete_pure(
            total_steps=total_steps,
            x0=x0,
            system=system,
            params=params,
            dt=dt,
            seed=seed,
            schedule=schedule,
        )
    if runner == "julia_tsit5_adaptive":
        raise ValueError("adaptive Tsit5 is one Julia solve in generate_from_trajectory_spec")
    if runner in CONTINUOUS_RUNNERS:
        return _generate_continuous_pure(
            total_steps=total_steps,
            x0=x0,
            system=system,
            schedule=schedule,
            params=params,
            dt=dt,
            t0=t0,
            method=method,
            seed=seed,
        )
    raise ValueError(f"runner {runner} has no generator")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public generation boundary
#
# Everything above is internal step-level integration.
# generate_from_trajectory_spec() is the single entry point:
#   1. Compiles ShockEvents into step-level StepShockSchedule
#   2. Dispatches to the right integrator (ODE/SDE/map/PDE/discrete)
#   3. Applies burn-in truncation on the returned trajectory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_from_trajectory_spec(
    trajectory_spec: TrajectorySpec,
    *,
    seed: int = 0,
    dtype: DTypeLike = np.float32,
    step_schedule: StepShockSchedule | None = None,
) -> np.ndarray:
    """Generate trajectory directly from TrajectorySpec."""
    # CHG: step_schedule is an explicit integer-step declaration kept with the
    # generated data; it cannot be mixed with the source's fraction-timed shocks.
    if seed is None:
        raise ValueError("seed must be provided explicitly (no defaults).")

    system = trajectory_spec.system.value
    all_systems = set(SYSTEMS.keys()) | DISCRETE_SYSTEMS | {"ks"}
    if system not in all_systems:
        raise ValueError(f"Unknown system: {system}. Known: {sorted(all_systems)}")

    if trajectory_spec.steps <= 0:
        raise ValueError("steps must be > 0")
    if trajectory_spec.burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if trajectory_spec.burn_in >= trajectory_spec.steps:
        raise ValueError(
            f"burn_in ({trajectory_spec.burn_in}) must be < steps ({trajectory_spec.steps})"
        )

    params_obj = trajectory_spec.params
    x0 = np.array(trajectory_spec.x0 or [0.1] * trajectory_spec.dim, dtype=np.float64)
    total_steps = trajectory_spec.steps
    usable_steps = trajectory_spec.steps - trajectory_spec.burn_in

    if system == "ks":
        kind = "pde"
        dim = int(cast(KSParams, params_obj).nx)
    elif system in DISCRETE_SYSTEMS:
        kind = "discrete"
        if system == "lg_ssm":
            dim = 2 * len(cast(LGSSMParams, params_obj).gammas)
        else:
            dim = 2 if system == "garch" else 1
    else:
        spec = SYSTEMS[system]
        kind = spec.kind
        dim = spec.dim if spec.dim != -1 else int(x0.size)

    if x0.size != dim:
        raise ValueError(f"initial_cond must have size {dim} for system='{system}'")

    if kind == "ode" and trajectory_spec.method not in {
        "rk4",
        "euler",
        "tsit5_fixed",
        "tsit5_adaptive",
    }:
        raise ValueError("ODE method must be 'rk4', 'euler', 'tsit5_fixed', or 'tsit5_adaptive'")
    if kind == "map" and trajectory_spec.method != "map":
        raise ValueError("Map method must be 'map'")
    if kind == "pde" and trajectory_spec.method != "etdrk4":
        raise ValueError("KS method must be 'etdrk4'")
    if kind in {"ode", "sde", "pde"} and trajectory_spec.dt <= 0.0:
        raise ValueError("dt must be > 0 for ode/sde/pde")

    schedule: StepShockSchedule | None = None
    switch: SwitchSpec | None = None

    if step_schedule is not None:
        if not isinstance(step_schedule, StepShockSchedule):
            raise TypeError("step_schedule requires a StepShockSchedule")
        if not trajectory_spec.shocks.is_stationary:
            raise ValueError(
                "exact-step events and fraction-timed shocks cannot share one trajectory"
            )
        if not step_schedule.param_sets or step_schedule.param_sets[0] != params_obj:
            raise ValueError("exact-step schedule must start with the source's original parameters")
        for event in step_schedule.events:
            if type(event.step) is not int or not 1 <= event.step <= total_steps:
                raise ValueError("exact-step event lies outside the integration steps")
            if isinstance(event, StatePerturbEvent):
                if not 0 <= event.state_idx < dim or not np.isfinite(event.eps):
                    raise ValueError(
                        "exact-step state perturbation needs a valid coordinate and finite amount"
                    )
            elif isinstance(event, StateCoupledImpulseEvent):
                if kind != "ode":
                    raise ValueError("state-coupled impulse requires an ODE")
                if any(index >= dim for index in event.targets) or any(
                    term.state_idx >= dim for term in event.terms
                ):
                    raise ValueError("state-coupled impulse coordinate is outside the source")
                if any(event.step <= term.lag for term in event.terms):
                    raise ValueError("state-coupled impulse requires preceding state history")
            elif isinstance(event, ScalarParameterIncrementEvent):
                if kind != "ode":
                    raise ValueError("parameter increment requires an ODE")
                if event.parameter_name not in _transition_field_names(params_obj):
                    raise ValueError(
                        "parameter increment names a field outside the source's movable parameters"
                    )
        schedule = step_schedule

    # Shock times are interpreted over the usable post-burn_in segment.
    # So t_start=0.35 means "35% into the returned trajectory", not into
    # the raw integration prefix that burn_in later removes.
    if not trajectory_spec.shocks.is_stationary:
        step_events: list[StepShockEvent] = []
        param_sets: list[SystemParams] = [params_obj]
        current_params = params_obj

        for event in trajectory_spec.shocks.events:
            shock_step = trajectory_spec.burn_in + int(event.t_start * usable_steps)
            shock_end_step = trajectory_spec.burn_in + int(event.t_end * usable_steps)

            if event.shocktype == ShockType.ODE_PARAMS:
                if event.params_after is None:
                    raise ValueError("ODE_PARAMS shock requires params_after")
                if not event.is_instantaneous:
                    if not (1 <= shock_step < total_steps):
                        raise ValueError(
                            f"transition start step {shock_step} not in [1, {total_steps - 1}]"
                        )
                    # Ensure a non-empty transition window after float->int conversion.
                    transition_end_step = min(total_steps, max(shock_step + 1, shock_end_step))
                    total_transition_steps = transition_end_step - shock_step
                    # CHG (user-approved 2026-07-01): raise instead of silently
                    # clamping n_inserts to the window (Rust already errors here).
                    if int(event.n_inserts) > total_transition_steps:
                        raise ValueError(
                            f"param transition asks for {int(event.n_inserts)} scheduled updates, "
                            f"but the window steps {shock_step}..{transition_end_step} allow only "
                            f"{total_transition_steps}. Widen the [t_start, t_end] window or "
                            f"lower n_inserts."
                        )
                    n_insert_points = int(event.n_inserts)
                    transition_step_offsets = [
                        (insert_idx * total_transition_steps) // n_insert_points
                        for insert_idx in range(1, n_insert_points + 1)
                    ]
                    for transition_step_offset in transition_step_offsets:
                        alpha = transition_step_offset / total_transition_steps
                        interpolated_params = _interpolate_params(
                            current_params,
                            event.params_after,
                            alpha,
                        )
                        param_sets.append(interpolated_params)
                        step_events.append(
                            ParamChangeEvent(
                                step=shock_step + transition_step_offset,
                                param_set_idx=len(param_sets) - 1,
                            )
                        )
                    current_params = event.params_after
                else:
                    if not (1 <= shock_step <= total_steps):
                        raise ValueError(f"shock step {shock_step} not in [1, {total_steps}]")
                    param_sets.append(event.params_after)
                    step_events.append(
                        ParamChangeEvent(step=shock_step, param_set_idx=len(param_sets) - 1)
                    )
                    current_params = event.params_after
            elif event.shocktype == ShockType.STATE:
                if not event.is_instantaneous:
                    raise ValueError("STATE shock must be instantaneous; use state_perturbation")
                if not (1 <= shock_step <= total_steps):
                    raise ValueError(f"shock step {shock_step} not in [1, {total_steps}]")
                if not (0 <= event.state_idx < dim):
                    raise ValueError(f"state_idx {event.state_idx} out of range [0, {dim})")
                step_events.append(
                    StatePerturbEvent(
                        step=shock_step,
                        state_idx=event.state_idx,
                        eps=event.eps,
                    )
                )
            elif event.shocktype == ShockType.TRAJECTORY_SWITCH:
                if len(trajectory_spec.shocks.events) > 1:
                    raise ValueError(
                        "TRAJECTORY_SWITCH shocks cannot be combined with other events"
                    )
                if not event.is_instantaneous:
                    raise ValueError("TRAJECTORY_SWITCH shock must be instantaneous")
                if not (0 <= shock_step <= total_steps):
                    raise ValueError(f"switch_step {shock_step} not in [0, {total_steps}]")
                if event.ic_b is None:
                    raise ValueError("TRAJECTORY_SWITCH shock requires ic_b")
                ic_b = np.array(event.ic_b, dtype=np.float64)
                if ic_b.size != dim:
                    raise ValueError(
                        f"trajectory switch ic_b must have size {dim}; got {ic_b.size}"
                    )
                p_b = event.params_after if event.params_after is not None else params_obj
                switch = SwitchSpec(
                    switch_step=shock_step,
                    params_b=p_b,
                    initial_cond_b=ic_b,
                    system_b=None,
                )

        if step_events:
            schedule = StepShockSchedule(
                events=tuple(sorted(step_events, key=lambda e: e.step)),
                param_sets=tuple(param_sets),
            )

    if trajectory_runner(system, trajectory_spec.method) == "julia_tsit5_adaptive":
        if schedule is not None or switch is not None:
            raise ValueError(
                "adaptive Tsit5 does not apply shocks; it is one uninterrupted Julia solve"
            )
        from .catalog import CoupledRoesslerParams
        from .julia_tsit5 import adaptive_coupled_roessler

        if not isinstance(params_obj, CoupledRoesslerParams):
            raise ValueError("coupled_roessler adaptive Tsit5 requires CoupledRoesslerParams")
        _version, traj = adaptive_coupled_roessler(
            x0=tuple(float(value) for value in x0),
            params=params_obj,
            dt=float(trajectory_spec.dt),
            steps=total_steps,
        )
    else:
        traj = _generate_trajectory_pure(
            total_steps=total_steps,
            x0=x0,
            system=system,
            params=params_obj,
            dt=trajectory_spec.dt,
            method=trajectory_spec.method,
            t0=0.0,
            seed=seed,
            schedule=schedule,
            switch=switch,
        )

    return traj[trajectory_spec.burn_in :].astype(dtype)
