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
This is consistent with e.g. Scipy integrate.solve_ivp output shape.

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

from typing import Callable, cast

import attr
import numpy as np
from numpy.typing import DTypeLike

from .catalog import (
    DISCRETE_SYSTEMS,
    SYSTEMS,
    DiagonalParams,
    DoubleWellParams,
    GARCHParams,
    KSParams,
    Lorenz96Params,
    OUParams,
    SLDSParams,
    SeasonalARParams,
)
from .specs import ShockType, TrajectorySpec
from attrs import AttrsInstance, astuple


@attr.define(frozen=True, slots=True)
class ParamChangeEvent:
    """Switch to a different parameter set at a specific timestep."""

    step: int
    param_set_idx: int  # index into StepShockSchedule.param_sets


@attr.define(frozen=True, slots=True)
class StatePerturbEvent:
    """Additive state perturbation: x[state_idx] += eps at step."""

    step: int
    state_idx: int
    eps: float


def _interpolate_scalar(before: float, after: float, alpha: float) -> float:
    """Linear interpolation between two scalar values."""
    return before * (1.0 - alpha) + after * alpha


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# parameter stepped-transition values
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _transition_field_names(params: AttrsInstance) -> tuple[str, ...]:
    """Names of the fields that are allowed to move during a param transition.

    Structural fields stay fixed on the original object.
    Examples:
    - Lorenz96 `forcing` can move, but not lattice shape
    - SeasonalAR `S` stays fixed
    - KS `nx` and `dealias` stay fixed
    """
    if isinstance(params, Lorenz96Params):
        return ("forcing",)
    if isinstance(params, SeasonalARParams):
        return ("phi", "sigma", "a0", "amp_drift")
    if isinstance(params, KSParams):
        return ("nu", "Lx")
    if isinstance(params, DiagonalParams):
        return ("eigenvalues",)
    return tuple(field.name for field in attr.fields(type(params)))


def _transition_field_values(params: AttrsInstance) -> tuple[float, ...]:
    """Ordered numeric values that actually participate in a transition."""
    if isinstance(params, DiagonalParams):
        return tuple(float(value) for value in params.eigenvalues)

    return tuple(
        float(getattr(params, field_name)) for field_name in _transition_field_names(params)
    )


def _params_with_transition_field_values(
    params: AttrsInstance,
    transition_values: tuple[float, ...],
) -> AttrsInstance:
    """Rebuild params from transition values while preserving structural fields."""
    if isinstance(params, DiagonalParams):
        return attr.evolve(params, eigenvalues=tuple(float(value) for value in transition_values))

    field_names = _transition_field_names(params)
    if len(field_names) != len(transition_values):
        raise ValueError(
            f"expected {len(field_names)} transition values for {type(params).__name__}, got {len(transition_values)}"
        )

    updates = {
        field_name: float(field_value)
        for field_name, field_value in zip(field_names, transition_values, strict=True)
    }
    return attr.evolve(params, **updates)


def _interpolate_params(before: AttrsInstance, after: AttrsInstance, alpha: float) -> AttrsInstance:
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


StepShockEvent = ParamChangeEvent | StatePerturbEvent


@attr.define(frozen=True, slots=True)
class StepShockSchedule:
    """Compiled shock schedule: events sorted by step + param sets.

    param_sets[0] is always the initial params.
    ParamChangeEvent.param_set_idx indexes into param_sets.
    """

    events: tuple[StepShockEvent, ...]
    param_sets: tuple[attr.AttrsInstance, ...]  # [initial, after_1, after_2, ...]

    def __attrs_post_init__(self):
        steps = [e.step for e in self.events]
        if steps != sorted(steps):  # Validate sorted by step
            raise ValueError("StepShockSchedule events must be sorted by step")

        n = len(self.param_sets)
        for e in self.events:  # Validate param_set_idx in range
            if isinstance(e, ParamChangeEvent) and not (0 <= e.param_set_idx < n):
                raise ValueError(f"param_set_idx {e.param_set_idx} out of range [0, {n})")

    # ── Constructors (high-level → compiled schedule) ──
    @staticmethod
    def no_shock(params: AttrsInstance) -> StepShockSchedule:
        """No perturbation. Just wraps params for uniform API."""
        return StepShockSchedule(events=(), param_sets=(params,))

    @staticmethod
    def param_shock(
        step: int,
        before: AttrsInstance,
        after: AttrsInstance,
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
        params: AttrsInstance,
    ) -> StepShockSchedule:
        """Single state perturbation at `step`."""
        return StepShockSchedule(
            events=(StatePerturbEvent(step=step, state_idx=state_idx, eps=eps),),
            param_sets=(params,),
        )

    @staticmethod
    def param_transition(
        before: AttrsInstance,
        after: AttrsInstance,
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

        param_sets = [before]
        events: list[ParamChangeEvent] = []

        n_insert_points = min(int(n_inserts), total_transition_steps)
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


@attr.define(frozen=True, slots=True)
class SwitchSpec:
    """Two-trajectory switch at a given step.

    Semantically different from StepShockSchedule — not a perturbation
    within one integration, but a cut from one independent run to another.
    """

    switch_step: int  # row index where we cut
    params_b: AttrsInstance  # system B params
    initial_cond_b: np.ndarray  # system B initial condition
    system_b: str | None = None  # if None, same system as A (same-system switch)
    # For stochastic systems, B gets seed + 1 (handled in _generate_switch)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integrators (inlined — these are only called by dispatchers below)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _params_as_args(p: AttrsInstance) -> tuple:
    """intent/contract: convert attrs params to positional args in definition order."""
    return astuple(p, recurse=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ODE integrator (Euler / RK4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _run_ode_py(
    f: Callable,
    x0: np.ndarray,
    steps: int,
    dt: float,
    t0: float,
    params: AttrsInstance,
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
            elif isinstance(e, ParamChangeEvent):
                assert schedule is not None
                p = cast(AttrsInstance, schedule.param_sets[e.param_set_idx])
            event_idx += 1

        # Integration step
        if method == "euler":
            if time_dependent:
                k1 = f(x, t, *_params_as_args(p))
            else:
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
    params: AttrsInstance,
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
                p = cast(AttrsInstance, schedule.param_sets[e.param_set_idx])
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
    params: attr.AttrsInstance,
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
                p = cast(attr.AttrsInstance, schedule.param_sets[e.param_set_idx])
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
) -> np.ndarray:
    """Run Switching LDS."""

    rng = np.random.default_rng(seed)
    traj = np.empty((steps + 1, 1), dtype=np.float64)
    x = float(x0[0])
    traj[0, 0] = x
    p = params

    # Start regime randomly
    z = 1 if rng.random() < 0.5 else 2
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
            A, Q = p.A1, p.Q1
        else:
            z = 2 if rng.random() < p.p22 else 1
            A, Q = p.A2, p.Q2

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


# =============================================================================
# Unified Configuration
# =============================================================================


def _generate_continuous_pure(
    *,
    total_steps: int,
    x0: np.ndarray,
    system: str,
    schedule: StepShockSchedule | None,
    params: AttrsInstance,
    dt: float,
    t0: float,
    method: str,
    seed: int,
) -> np.ndarray:
    """Generate ODE/SDE/Map trajectory with explicit inputs (no self)."""
    spec = SYSTEMS[system]

    active_params = schedule.param_sets[0] if schedule else params

    if spec.kind == "ode":
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
    if spec.kind == "map":
        return _run_map_py(
            f=spec.f_py,
            x0=x0,
            steps=total_steps,
            params=active_params,
            schedule=schedule,
        )
    if spec.kind == "sde":
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
    raise ValueError(f"Unknown kind for {system}: {spec.kind}")


def _generate_discrete_pure(
    *,
    total_steps: int,
    x0: np.ndarray,
    system: str,
    params: AttrsInstance,
    seed: int,
    schedule: StepShockSchedule | None,
) -> np.ndarray:
    """Generate discrete system trajectory with explicit inputs (no self)."""
    active_params = schedule.param_sets[0] if schedule else params

    if system == "slds":
        return _run_slds(
            x0=x0,
            steps=total_steps,
            params=cast(SLDSParams, active_params),
            seed=seed,
            schedule=schedule,
        )
    if system == "seasonal_ar":
        return _run_seasonal_ar(
            x0=x0,
            steps=total_steps,
            params=cast(SeasonalARParams, active_params),
            seed=seed,
            schedule=schedule,
        )
    if system == "garch":
        return _run_garch(
            x0=x0,
            steps=total_steps,
            params=cast(GARCHParams, active_params),
            seed=seed,
            schedule=schedule,
        )
    raise ValueError(f"Unknown discrete system: {system}")


def _generate_ks_pure(
    *,
    total_steps: int,
    x0: np.ndarray,
    dt: float,
    params: AttrsInstance,
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
    params: AttrsInstance,
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
    params: AttrsInstance,
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
    params: AttrsInstance,
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

    if system == "ks":
        return _generate_ks_pure(
            total_steps=total_steps,
            x0=x0,
            dt=dt,
            params=params,
            schedule=schedule,
        )
    if system in DISCRETE_SYSTEMS:
        return _generate_discrete_pure(
            total_steps=total_steps,
            x0=x0,
            system=system,
            params=params,
            seed=seed,
            schedule=schedule,
        )
    if system in SYSTEMS:
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
    raise ValueError(f"Unknown system: {system}")


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
) -> np.ndarray:
    """Generate trajectory directly from TrajectorySpec."""
    if seed is None:
        raise ValueError("seed must be provided explicitly (no defaults).")

    system = trajectory_spec.system.value
    # Historical registry key without underscores; keep enum spelling stable.
    if system == "van_der_pol":
        system = "vanderpol"
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
        dim = 2 if system == "garch" else 1
    else:
        spec = SYSTEMS[system]
        kind = spec.kind
        dim = spec.dim if spec.dim != -1 else int(x0.size)

    if x0.size != dim:
        raise ValueError(f"initial_cond must have size {dim} for system='{system}'")

    if kind == "ode" and trajectory_spec.method not in {"rk4", "euler"}:
        raise ValueError("ODE method must be 'rk4' or 'euler'")
    if kind == "map" and trajectory_spec.method != "map":
        raise ValueError("Map method must be 'map'")
    if kind == "pde" and trajectory_spec.method != "etdrk4":
        raise ValueError("KS method must be 'etdrk4'")
    if kind in {"ode", "sde", "pde"} and trajectory_spec.dt <= 0.0:
        raise ValueError("dt must be > 0 for ode/sde/pde")

    schedule: StepShockSchedule | None = None
    switch: SwitchSpec | None = None

    # Shock times are interpreted over the usable post-burn_in segment.
    # So t_start=0.35 means "35% into the returned trajectory", not into
    # the raw integration prefix that burn_in later removes.
    if not trajectory_spec.shocks.is_stationary:
        step_events: list[StepShockEvent] = []
        param_sets: list[AttrsInstance] = [params_obj]
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
                    n_insert_points = min(int(event.n_inserts), total_transition_steps)
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
