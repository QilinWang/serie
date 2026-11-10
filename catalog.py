"""System catalog — params, equations, presets, grouped by system.

DESIGN: Each system's full definition lives in one visual block.
Param fields have NO defaults for experiment knobs — use the preset functions.
The preset function is the single source of canonical parameter values.

This file is intentionally long. Local proximity beats file-size anxiety.
If params + equations + presets describe the same system, they belong together.

Stability notes, parameter sweep results, and per-system investigation
findings are documented in ``docs/system_presets.md``.
"""

from __future__ import annotations

from msgspec.structs import replace

from msgspec import Struct
from serie.records import StructRecord, record_hash

from enum import StrEnum, auto
from typing import Callable, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .specs import TrajectorySpec


# ═══════════════════════════════════════════════════════════════════════════
# ODESystem enum
# ═══════════════════════════════════════════════════════════════════════════


class ODESystem(StrEnum):
    """Dynamical systems registry. Keys match SYSTEMS in serie.integrators."""

    # Continuous ODEs
    LORENZ = auto()  # 3D atmospheric convection, canonical chaos
    ROSSLER = auto()  # 3D single-scroll, simpler chaos than Lorenz
    HYPER_ROSSLER = auto()  # 4D hyperchaotic extension of Rössler
    CHUA = auto()  # 3D piecewise-linear double-scroll circuit
    DUFFING = auto()  # 2D driven nonlinear oscillator
    VANDERPOL = auto()  # 2D self-sustained limit cycle
    LORENZ96 = auto()  # N-D weather toy model (configurable dim)

    # Discrete maps
    HENON = auto()  # 2D quadratic map, strange attractor
    LOGISTIC = auto()  # 1D map, period-doubling → chaos

    # SDEs (stochastic differential equations)
    OU = auto()  # 1D Ornstein-Uhlenbeck, mean-reverting
    DOUBLEWELL = auto()  # 1D bistable potential + Brownian noise
    LG_SSM = auto()  # D-dim matrix OU (linear Gaussian state-space), EXACT discretization

    # Discrete statistical processes
    SLDS = auto()  # 1D switching linear dynamical system
    SEASONAL_AR = auto()  # 1D seasonal autoregressive
    # CHG 2026-10-06 (owner): Box–Jenkins seasonal AR, stochastic season through lag S.
    SEASONAL_AR_BJ = auto()  # 1D AR with lags 1 and S (no deterministic season)
    GARCH = auto()  # 2D (return + variance) volatility model

    # PDEs
    KS = auto()  # Kuramoto-Sivashinsky, spatiotemporal chaos

    # Designed (ground-truth eigenvalue recovery) — integrators TBD
    LINEAR = auto()  # dx/dt = Ax, prescribed real eigenvalues
    DIAGONAL = auto()  # dx/dt = diag(λ)x, trivial eigenvalue test
    JORDAN_BLOCK = auto()  # dx/dt = Jx, off-diagonal sensitivity
    ROTATION_2D = auto()  # dx/dt = [[0,-ω],[ω,0]]x, pure oscillation

    # Future / referenced in tier 4 catalog
    THOMAS = auto()  # 3D cyclic symmetric attractor
    CHEN = auto()  # 3D Lorenz-family, different topology
    LOTKA_VOLTERRA = auto()  # 2D predator-prey, conservative oscillation
    # CHG 2026-11-02: one member of each dysts family, registered like lorenz.series.
    SPROTT_A = auto()  # 3D quadratic, Sprott's algebraically simplest chaotic flow
    AIZAWA = auto()  # 3D torus-like attractor
    # CHG 2026-11-02: the seven systems named for serie. Equations are the
    # PredefinedDynamicalSystems.jl rule functions. Five use the lorenz.series
    # cut. Coupled Rössler and Hindmarsh–Rose do not; their specs say why.
    HENON_HEILES = auto()  # 4D Hamiltonian, energy conserved, no attractor
    COUPLED_ROESSLER = auto()  # 6D drive-response pair of Rössler oscillators
    LORENZ84 = auto()  # 3D low-order atmospheric circulation
    DOUBLE_PENDULUM = auto()  # 4D angles and angular velocities
    FITZHUGH_NAGUMO = auto()  # 2D neuron membrane and recovery
    HINDMARSH_ROSE = auto()  # 3D neuron burst
    RIKITAKE = auto()  # 3D two-disk dynamo

    @property
    def is_stochastic(self) -> bool:
        """SDEs and discrete stochastic processes produce distinct trajectory draws."""
        return self in _STOCHASTIC_SYSTEMS


_STOCHASTIC_SYSTEMS: frozenset[ODESystem] = frozenset(
    {
        ODESystem.OU,
        ODESystem.DOUBLEWELL,
        ODESystem.LG_SSM,
        ODESystem.SLDS,
        ODESystem.SEASONAL_AR,
        ODESystem.SEASONAL_AR_BJ,  # CHG 2026-10-06
        ODESystem.GARCH,
    }
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lorenz
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LorenzParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Lorenz system (1963).

    The canonical chaotic attractor arising from simplified atmospheric convection.


      dx/dt = σ(y - x)
      dy/dt = x(ρ - z) - y
      dz/dt = xy - βz

    Standard chaotic regime: σ=10, ρ=28, β=8/3
    """

    sigma: float  # Prandtl number (ratio of momentum to thermal diffusivity)
    rho: float  # Rayleigh number (temperature gradient strength)
    beta: float  # Geometric factor related to cell aspect ratio

    __hash__ = record_hash


def _f_lorenz(state: np.ndarray, t: float, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=np.float64)


def lorenz_spec(*, sigma=10.0, rho=28.0, beta=8.0 / 3.0, steps=35999, dt=1e-2, burn_in=0):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.LORENZ,
        params=LorenzParams(sigma=sigma, rho=rho, beta=beta),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=3,
        x0=(1.0, 0.98, 1.1),
    )


def lorenz_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        lorenz_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=LorenzParams(sigma=10.1, rho=28.1, beta=8.1 / 3.0),
                t_start=0.35,
            )
        ),
    )


def lorenz_state_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        lorenz_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.state_perturbation(state_idx=0, eps=0.9, t_start=0.35)
        ),
    )


def lorenz_switch_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        lorenz_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.trajectory_switch(
                ic_b=(1.002, 0.982, 1.102),
                params_after=LorenzParams(sigma=10.0, rho=28.1, beta=8.0 / 3.0),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rossler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RosslerParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Rössler system (1976).

    A simpler chaotic system than Lorenz, with only one nonlinear term.

      dx/dt = -y - z
      dy/dt = x + ay
      dz/dt = b + z(x - c)

    Standard chaotic regime: a=0.2, b=0.2, c=5.7
    """

    a: float  # Linear feedback in y
    b: float  # Offset in z equation
    c: float  # Nonlinear chaos parameter (controls attractor size)

    __hash__ = record_hash


def _f_rossler(state: np.ndarray, t: float, a: float, b: float, c: float) -> np.ndarray:
    x, y, z = state
    return np.array([-y - z, x + a * y, b + z * (x - c)], dtype=np.float64)


def rossler_spec(*, a=0.2, b=0.2, c=5.7, steps=35999, dt=1e-2, burn_in=0):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.ROSSLER,
        params=RosslerParams(a=a, b=b, c=c),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=3,
        x0=(1.0, 1.0, 1.0),
    )


def rossler_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        rossler_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=RosslerParams(a=0.25, b=0.25, c=5.75),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hyper-Rössler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HyperRosslerParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Hyper-Rössler system (4D hyperchaotic).

    Extension of Rössler with an additional dimension w for hyperchaos.

      dx/dt = -y - z
      dy/dt = x + ay + w
      dz/dt = b + xz
      dw/dt = -cz + dw

    Standard hyperchaotic regime: a=0.25, b=3.0, c=0.5, d=0.05
    """

    a: float  # Linear feedback in y
    b: float  # Offset in z
    c: float  # Coupling z→w
    d: float  # Growth term for w

    __hash__ = record_hash


def _f_hyper_rossler(
    state: np.ndarray, t: float, a: float, b: float, c: float, d: float
) -> np.ndarray:
    x, y, z, w = state
    return np.array([-y - z, x + a * y + w, b + x * z, -c * z + d * w], dtype=np.float64)


def hyper_rossler_spec(*, a=0.25, b=3.0, c=0.5, d=0.05, steps=15000, dt=1e-3, burn_in=0):
    # NOTE: the standard Hyper-Rössler is a transient attractor that escapes
    # to infinity around T≈19s. steps=15000 at dt=0.001 gives T=15s, which
    # is comfortably within the bounded regime with a 4s safety margin.
    # The system IS hyperchaotic (two positive Lyapunov exponents).
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.HYPER_ROSSLER,
        params=HyperRosslerParams(a=a, b=b, c=c, d=d),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=4,
        x0=(0.1, 0.1, 0.1, 0.1),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chua
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ChuaParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    h(x) = m0_outer_gt1·x + 0.5·(m1_inner_le1-m0_outer_gt1)·(|x+1| - |x-1|)

    Inner slope (|x| ≤ 1):  m1_inner_le1
    Outer slope (|x| > 1):  m0_outer_gt1

    Standard double-scroll: α=15.6, β=28, m1_inner_le1=-8/7 (inner), m0_outer_gt1=-5/7 (outer)
    """

    alpha: float
    beta: float
    m1_inner_le1: float  # slope for (|x| ≤ 1)
    m0_outer_gt1: float  # slope for (|x| > 1)

    __hash__ = record_hash


def _f_chua(
    state: np.ndarray,
    t: float,
    alpha: float,
    beta: float,
    m1_inner_when_x_le1: float,
    m0_outer_gt1: float,
) -> np.ndarray:
    """
    Chua circuit with piecewise-linear nonlinearity h(x).

    h(x) = m0_outer_gt1 * x
         + 0.5 * (m1_inner_le1 - m0_outer_gt1) * (|x+1| - |x-1|)

    This gives:
      slope = m1_inner_le1 for |x| <= 1
      slope = m0_outer_gt1 for |x| > 1

    ODE:
      dx/dt = alpha * (y - x - h(x))
      dy/dt = x - y + z
      dz/dt = -beta * y
    https://link.springer.com/content/pdf/10.1007/s11071-022-08078-y.pdf
    https://www.physics.smu.edu/~olness/quarknet/chaoscircuit.pdf
    https://acoustique.ec-lyon.fr/chaos/Chua_from-Alligood.pdf

    The circuit consists of an inductor L, capacitors C1
    and C2, a linear resistor R, and a nonlinear element
    with negative conductivity −G, commonly called the
    Chua diode
    # alpha = C2/C1, here alpha = 1 so alpha*-hx and alpha*(y-x)
    # beta=C2R^2/L,
    # gamma=C2RR_L/L set to zero
    """
    x, y, z = state
    # h(x) formulation where m1_inner is slope at 0
    hx = m0_outer_gt1 * x + 0.5 * (m1_inner_when_x_le1 - m0_outer_gt1) * (
        np.abs(x + 1) - np.abs(x - 1)
    )
    return np.array([alpha * (y - x - hx), x - y + z, -beta * y], dtype=np.float64)


def chua_spec(
    *,
    alpha=15.6,
    beta=28.0,
    m1_inner_le1=-8 / 7,
    m0_outer_gt1=-5 / 7,
    steps=35999,
    dt=5e-3,
    burn_in=0,
):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.CHUA,
        params=ChuaParams(
            alpha=alpha, beta=beta, m1_inner_le1=m1_inner_le1, m0_outer_gt1=m0_outer_gt1
        ),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=3,
        x0=(0.1, 0.0, 0.0),
    )


def chua_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        chua_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=ChuaParams(
                    alpha=15.9,
                    beta=28.5,
                    m1_inner_le1=-8.1 / 7.0,
                    m0_outer_gt1=-5.2 / 7.0,
                ),
                t_start=0.35,
            )
        ),
    )


def chua_switch_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        chua_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.trajectory_switch(ic_b=(0.11, 0.01, 0.02), t_start=0.35)
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Duffing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DuffingParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Duffing oscillator (forced, damped).

    Nonlinear oscillator with cubic stiffness and periodic forcing.

      dx/dt = v
      dv/dt = γcos(ωt) - δv - αx - βx³

    TIME-DEPENDENT (requires t in f).
    Standard chaotic regime: α=1, β=-1, δ=0.2, γ=0.65, ω=1.2
    """

    alpha: float  # Linear stiffness
    beta: float  # Cubic stiffness (< 0 for softening spring)
    delta: float  # Damping coefficient
    gamma: float  # Forcing amplitude
    omega: float  # Forcing frequency

    __hash__ = record_hash


def _f_duffing(
    state: np.ndarray,
    t: float,
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    omega: float,
) -> np.ndarray:
    x, v = state
    return np.array(
        [v, gamma * np.cos(omega * t) - delta * v - alpha * x - beta * x**3],
        dtype=np.float64,
    )


def duffing_spec(
    *, alpha=1.0, beta=1.0, delta=0.3, gamma=0.3, omega=1.0, steps=35999, dt=1e-2, burn_in=0
):
    # NOTE: beta=1 (hardening spring) is bounded and stable at any T.
    # beta=-1 (softening spring, the classic chaotic Duffing) diverges
    # around step ~3000 at dt=0.002. Use beta=-1 only with short steps
    # and small dt. The hardening preset is a quasiperiodic forced
    # oscillator — useful for testing forecasters on periodic-ish signals.
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.DUFFING,
        params=DuffingParams(alpha=alpha, beta=beta, delta=delta, gamma=gamma, omega=omega),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=2,
        x0=(0.5, 0.0),
        method="rk4",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Van der Pol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VanderPolParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Van der Pol oscillator (1920).

    Self-sustained oscillator with nonlinear damping.

      dx/dt = y
      dy/dt = μ(1 - x²)y - x

    μ > 0 controls nonlinearity strength. Large μ → relaxation oscillations.
    """

    mu: float  # Nonlinearity parameter

    __hash__ = record_hash


def _f_vanderpol(state: np.ndarray, t: float, mu: float) -> np.ndarray:
    x, y = state
    return np.array([y, mu * (1.0 - x**2) * y - x], dtype=np.float64)


def vanderpol_spec(*, mu=2.0, steps=35999, dt=1e-2, burn_in=1000):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.VANDERPOL,
        params=VanderPolParams(mu=mu),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=2,
        x0=(1.0, 0.0),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lorenz96
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Lorenz96Params(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Lorenz-96 model (1996).

    Simplified atmospheric model on a 1D periodic lattice.

      dx_i/dt = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F

    Variable dimension (typically 40). Chaos when F ≥ 8.
    """

    forcing: float  # External forcing F

    __hash__ = record_hash


def _f_lorenz96(state: np.ndarray, t: float, forcing: float) -> np.ndarray:
    # x_{i+1} -> shift -1
    # x_{i-2} -> shift +2
    # x_{i-1} -> shift +1
    return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + forcing


def lorenz96_spec(*, dim=6, forcing=8.0, steps=55000, dt=7e-3, burn_in=0):
    from .specs import TrajectorySpec

    x0 = tuple([1.01] + [1.0] * (dim - 1))
    return TrajectorySpec(
        system=ODESystem.LORENZ96,
        params=Lorenz96Params(forcing=forcing),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=dim,
        x0=x0,
    )


def lorenz96_switch_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        lorenz96_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.trajectory_switch(
                ic_b=(0.99, 1.02, 1.02, 1.03, 1.01, 1.01),
                params_after=Lorenz96Params(forcing=9.0),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Henon
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HenonParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Hénon map (1976).

    2D discrete-time chaotic map.


      x_{n+1} = 1 - ax_n² + y_n
      y_{n+1} = bx_n

    Standard chaotic regime: a=1.4, b=0.3
    """

    a: float  # Nonlinear coefficient
    b: float  # Linear coefficient (contraction)

    __hash__ = record_hash


def _f_henon(state: np.ndarray, a: float, b: float) -> np.ndarray:
    x, y = state
    x2 = x * x
    return np.array([1 - a * x2 + y, b * x], dtype=np.float64)


def henon_spec(*, a=1.4, b=0.3, steps=35999, burn_in=500):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.HENON,
        params=HenonParams(a=a, b=b),
        dt=1.0,
        steps=steps,
        burn_in=burn_in,
        dim=2,
        method="map",
        x0=(0.1, 0.1),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logistic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LogisticParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Logistic map.

    1D discrete-time map exhibiting period-doubling route to chaos.


      x_{n+1} = rx_n(1 - x_n)

    Chaos when r ∈ [3.57, 4.0]. Full chaos at r ≈ 3.9.
    """

    r: float  # Growth rate

    __hash__ = record_hash


def _f_logistic(state: np.ndarray, r: float) -> np.ndarray:
    x = state[0]
    return np.array([r * x * (1 - x)], dtype=np.float64)


def logistic_spec(*, r=3.9, steps=35999, burn_in=500):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.LOGISTIC,
        params=LogisticParams(r=r),
        dt=1.0,
        steps=steps,
        burn_in=burn_in,
        dim=1,
        method="map",
        x0=(0.5,),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OUParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Ornstein-Uhlenbeck process (SDE).

    Mean-reverting stochastic process.

      dx = θ(μ - x)dt + σdW

    θ: reversion rate, μ: long-term mean, σ: volatility
    REQUIRE: theta >= 0 (mean reversion rate)

    REQUIRE: sigma_noise >= 0
    """

    theta: float  # Mean reversion rate
    mu: float  # Long-term mean
    sigma_noise: float  # Volatility (diffusion coefficient)

    __hash__ = record_hash


def _f_ou_drift(
    state: np.ndarray, t: float, theta: float, mu: float, sigma_noise: float
) -> np.ndarray:
    return np.array([theta * (mu - state[0])], dtype=np.float64)


def ou_spec(*, theta=0.2, mu=0.0, sigma_noise=0.3, steps=25000, dt=0.5, burn_in=0):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.OU,
        params=OUParams(theta=theta, mu=mu, sigma_noise=sigma_noise),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=1,
        method="euler",  # ← NEW
        x0=(0.0,),
    )


def ou_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        ou_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=OUParams(theta=0.2, mu=0.5, sigma_noise=0.3),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DoubleWell
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DoubleWellParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Double-well Langevin dynamics (SDE).

    Particle in bistable potential U(x) = x⁴/4 - (a/2)x².

      dx = (-x³ + ax)dt + σdW

    Minima at x = ±√a. Noise-induced transitions between wells.
    REQUIRE: sigma_noise >= 0

    (optional) NOTE: a > 0 gives a true double-well; a<=0 degenerates
    """

    a: float  # Well separation (minima at ±√a)
    sigma_noise: float  # Noise intensity

    __hash__ = record_hash


def _f_doublewell_drift(state: np.ndarray, t: float, a: float, sigma_noise: float) -> np.ndarray:
    x = state[0]
    return np.array([-(x**3) + a * x], dtype=np.float64)


def doublewell_spec(*, a=1.5, sigma_noise=0.25, steps=25000, dt=0.5, burn_in=0):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.DOUBLEWELL,
        params=DoubleWellParams(a=a, sigma_noise=sigma_noise),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=1,
        method="euler",  # ← NEW
        x0=(0.0,),
    )


def doublewell_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        doublewell_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=DoubleWellParams(a=1.0, sigma_noise=0.35),
                t_start=0.35,
            )
        ),
    )


def doublewell_barrier_collapse_spec():
    """Hard one-axis Double Well shift for the H64 paper expansion.

    The latent SDE is

        dx = (a x - x^3) dt + sigma_noise dW,
        U(x) = x^4 / 4 - a x^2 / 2.

    For ``a > 0`` the wells are at ``±sqrt(a)`` and the barrier height is
    ``a^2 / 4``. At 35% of the trajectory this preset changes only
    ``a: 1.5 -> 0.5`` while keeping ``sigma_noise=0.25``. The barrier height
    therefore drops from 0.5625 to 0.0625, and the dimensionless Kramers-like
    exponent ``2 ΔU / sigma_noise^2`` drops from 18 to 2. This is deliberately
    a latent-dynamics intervention, not an observation-noise intervention:
    the paper titration layer still injects and scales measurement noise
    independently.
    """
    from .specs import ShockSchedule, ShockEvent

    return replace(
        doublewell_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=DoubleWellParams(a=0.5, sigma_noise=0.25),
                t_start=0.35,
            )
        ),
    )


def doublewell_switch_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        doublewell_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.trajectory_switch(
                ic_b=(0.0,),
                params_after=DoubleWellParams(a=1.0, sigma_noise=0.35),
                t_start=0.35,
            )
        ),
    )


def doublewell_diffusion_shock_spec():
    """Owner-authorized test-time noise change, with unchanged potential.

    CHG 2026-09-25: sigma 0.25 -> 0.35 at step23500 of25000 (fraction0.94).
    Preserve Euler dt0.5, a1.5, x0 and burn-in. This intervention changes
    process noise, not measurement noise, and supplies no model input flag.
    """
    from .specs import ShockSchedule, ShockEvent

    return replace(
        doublewell_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=DoubleWellParams(a=1.5, sigma_noise=0.35), t_start=0.94
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SLDSParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Switching Linear Dynamical System (K=2 regimes).

    Markov-switching AR(1):
      z_t ∈ {1,2} with P(z_t=1|z_{t-1}=1) = p11, P(z_t=2|z_{t-1}=2) = p22
      x_t = A_{z_t} x_{t-1} + η_t,  η_t ~ N(0, Q_{z_t})

    REQUIRE: Q1 >= 0, Q2 >= 0

    REQUIRE: 0 <= p11 <= 1, 0 <= p22 <= 1
    """

    A1: float  # AR coefficient in regime 1
    Q1: float  # Noise variance in regime 1
    A2: float  # AR coefficient in regime 2
    Q2: float  # Noise variance in regime 2
    p11: float  # P(stay in regime 1)
    p22: float  # P(stay in regime 2)

    __hash__ = record_hash


def slds_spec(*, A1=0.9, Q1=0.05, A2=0.98, Q2=0.35, p11=0.94, p22=0.95, steps=25000):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.SLDS,
        params=SLDSParams(A1=A1, Q1=Q1, A2=A2, Q2=Q2, p11=p11, p22=p22),
        dt=1.0,
        steps=steps,
        burn_in=0,
        dim=1,
        method="euler",  # ← NEW (ignored for discrete; schema compat)
        x0=(0.0,),
    )  # NOTE Note: dt=1.0 is a dummy


def slds_moe_spec(*, steps=25000):
    # MOE-diagnostic preset: regimes are maximally distinct so a regime-aware
    # model (MOE, HMM-gated, mixture Gaussian) should outperform a single AR.
    #
    # Regime 1: A1=0.3 (fast mean-reversion), Q1=0.01 (quiet)
    # Regime 2: A2=0.99 (near unit root), Q2=1.0 (volatile)
    # p11=p22=0.95 → dwell ≈ 20 steps per regime
    #
    # Estimated oracle-vs-global gap: ~32% (single seed, no error bars).
    # Full methodology, limitations, and parameter sweep tables:
    #   see docs/system_presets.md → "SLDS" section
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.SLDS,
        params=SLDSParams(A1=0.3, Q1=0.01, A2=0.99, Q2=1.0, p11=0.95, p22=0.95),
        dt=1.0,
        steps=steps,
        burn_in=0,
        dim=1,
        method="euler",
        x0=(0.0,),
    )


def slds_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        slds_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=SLDSParams(
                    A1=0.83,
                    Q1=0.50,
                    A2=0.97,
                    Q2=0.30,
                    p11=0.96,
                    p22=0.92,
                ),
                t_start=0.35,
            )
        ),
    )


def slds_switch_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        slds_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.trajectory_switch(
                ic_b=(0.0,),
                params_after=SLDSParams(
                    A1=0.87,
                    Q1=0.07,
                    A2=0.99,
                    Q2=0.45,
                    p11=0.90,
                    p22=0.95,
                ),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SeasonalAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SeasonalARParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Seasonal AR with time-varying amplitude.

      y_t = a(t)·sin(2πt/S) + φy_{t-1} + ε_t,  ε_t ~ N(0, σ²)
      a(t) = a0 + drift·t
    REQUIRE: S > 0, sigma >= 0
    """

    S: int  # Seasonal period
    phi: float  # AR(1) coefficient
    sigma: float  # Noise std
    a0: float  # Initial amplitude
    amp_drift: float  # Amplitude drift per step

    __hash__ = record_hash


def seasonal_ar_spec(*, S=24, phi=0.5, sigma=0.2, a0=1.0, amp_drift=0e-4, steps=25000):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.SEASONAL_AR,
        params=SeasonalARParams(S=S, phi=phi, sigma=sigma, a0=a0, amp_drift=amp_drift),
        dt=1.0,
        steps=steps,
        burn_in=0,
        dim=1,
        method="euler",
        x0=(0.0,),
    )


def seasonal_ar_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        seasonal_ar_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=SeasonalARParams(S=24, phi=0.8, sigma=0.35, a0=1.4, amp_drift=0e-4),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SeasonalAR, Box–Jenkins sense
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SeasonalARBJParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    Seasonal AR in the Box–Jenkins sense: the season is stochastic, carried by a lag of S.

      y_t = φ y_{t-1} + Φ y_{t-S} + σ ε_t,   ε_t ~ N(0, 1),   y_s = 0 for s < 0

    CHG 2026-10-06 (owner: "the Box–Jenkins sense tried as well"). The additive
    form of a seasonal AR(1)×(1)_S: the multiplicative model's cross term −φΦ y_{t−S−1}
    is left out. Box, Jenkins, Reinsel and Ljung, *Time Series Analysis*, chapter 9
    (seasonal models) write the seasonal coefficient Φ. Unlike `SeasonalARParams`
    there is no deterministic sine: the season is whatever the last S values were,
    so its phase and amplitude wander with the innovations. Stationary iff every root
    of 1 − φ z − Φ z^S lies outside the unit circle; |φ| + |Φ| < 1 is sufficient.
    REQUIRE: S >= 2, sigma >= 0
    """

    S: int  # Seasonal lag (period)
    phi: float  # Coefficient of lag 1, φ
    seasonal_phi: float  # Coefficient of lag S, Φ
    sigma: float  # Innovation std

    __hash__ = record_hash


def seasonal_ar_bj_spec(*, S=24, phi=0.5, seasonal_phi=0.4, sigma=0.2, steps=25000):
    from .specs import TrajectorySpec

    # CHG 2026-10-06: φ = 0.5, Φ = 0.4, σ = 0.2 (the owner's example values); |φ| + |Φ| = 0.9 < 1.
    # The 24 roots of 1 − 0.5 z − 0.4 z^24 have smallest modulus 1.008962 (the real root
    # z = 1.008962; next a conjugate pair at 1.011438); every root is outside the unit circle.
    # tests/test_gaussian_conditional_law_datasets.py recomputes them.
    return TrajectorySpec(
        system=ODESystem.SEASONAL_AR_BJ,
        params=SeasonalARBJParams(S=S, phi=phi, seasonal_phi=seasonal_phi, sigma=sigma),
        dt=1.0,
        steps=steps,
        burn_in=0,
        dim=1,
        method="euler",
        x0=(0.0,),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GARCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GARCHParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    GARCH(1,1) model.

      r_t = σ_t ε_t,  ε_t ~ N(0,1)
      σ²_t = ω + α r²_{t-1} + β σ²_{t-1}

    Stationary when α + β < 1.

    REQUIRE: omega > 0, alpha >= 0, beta >= 0

    REQUIRE: alpha + beta < 1 for strict stationarity (else init uses fallback variance)

    NOTE: sigma2_0 sets initial conditional variance sigma^2_0;
    if None and stationary, uses omega/(1-alpha-beta)
    """

    omega: float  # Constant term
    alpha: float  # ARCH coefficient (shock sensitivity)
    beta: float  # GARCH coefficient (persistence)

    __hash__ = record_hash


def garch_spec(*, omega=0.01, alpha=0.06, beta=0.90, steps=25000):
    from .specs import TrajectorySpec

    if (alpha + beta) < 1.0:
        sigma2_0 = omega / max(1e-9, 1.0 - alpha - beta)
    else:
        sigma2_0 = 1.0
    return TrajectorySpec(
        system=ODESystem.GARCH,
        params=GARCHParams(omega=omega, alpha=alpha, beta=beta),
        dt=1.0,
        steps=steps,
        burn_in=0,
        dim=2,
        method="euler",
        x0=(0.0, sigma2_0),
    )


def garch_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        garch_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=GARCHParams(omega=0.03, alpha=0.15, beta=0.70),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LG-SSM (matrix OU / linear Gaussian state-space latent SDE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LGSSMParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    LG-SSM: continuous-time linear Gaussian SDE sampled at a fixed interval dt
    by EXACT discretization (no Euler stepping).

    Roadmap: docs/lab/directions/controlled_model_families_roadmap.md
      §15.2 (matrix Ornstein-Uhlenbeck model),
      §15.3 (damped oscillatory spectral blocks),
      §3.5/§3.6 (controlled non-normality via eigenvector conditioning κ(V)).

    Latent SDE (§15.2, with L = I_D and Q_c = q_c·I_D):

      dx(t) = F x(t) dt + L dW(t)

    Drift construction (§15.3 blocks assembled per §3.5):

      B_c = blockdiag(F_1, …, F_m),  F_i = [[-γ_i, -ω_i], [ω_i, -γ_i]]
      F   = V B_c V⁻¹,               V = diag(σ)·W

    where W is the orthonormal DCT-II matrix and σ_j = kappa_v^{j/(D-1)}, so
    cond₂(V) = kappa_v exactly. kappa_v = 1 gives orthogonal V → normal F;
    kappa_v > 1 gives ill-conditioned V → non-normal F with transient
    amplification at a FIXED spectral radius (§3.5). §3.5 pins only V's
    conditioning, not a specific V; the diag(σ)·DCT-II realization is this
    catalog's deterministic pin so both engines build the identical matrix.

    Exact sampling at interval dt (§15.2):

      x_{k+1} = A x_k + w_k,   w_k ~ N(0, Q)
      A = exp(F·dt),   Q = ∫₀^dt exp(F s)·L Q_c Lᵀ·exp(Fᵀ s) ds   (Van Loan)

    State dimension D = 2·len(gammas). |eig(A)| = exp(-γ_i·dt) is independent
    of kappa_v — this is the "spectral radius held fixed while V's
    conditioning varies" design.

    REQUIRE: len(gammas) == len(omegas) >= 1
    REQUIRE: gammas > 0 (stationarity: Re λ(F) < 0, §15.2)
    REQUIRE: kappa_v >= 1
    REQUIRE: q_c > 0
    """

    gammas: tuple[float, ...]  # Per-block damping rates γ_i (§15.3)
    omegas: tuple[float, ...]  # Per-block angular frequencies ω_i (§15.3)
    kappa_v: float  # Eigenvector conditioning κ(V) (§3.6 menu: 1, 5, 20, 100)
    q_c: float  # Process-noise spectral density: L = I, Q_c = q_c·I (§15.2)

    __hash__ = record_hash


def _lg_ssm_mixing_matrix(dim: int) -> np.ndarray:
    """Orthonormal DCT-II matrix W, shape [dim, dim] — the deterministic
    orthogonal mixing factor of V = diag(σ)·W.

    W[k, n] = a_k·cos(π·(n + 1/2)·k / dim), a_0 = √(1/dim), a_k = √(2/dim).
    Rows are orthonormal, so W Wᵀ = I. Closed-form cosine entries keep the
    Python and Rust constructions bit-comparable without any RNG.
    """
    row = np.arange(dim, dtype=np.float64)[:, None]
    col = np.arange(dim, dtype=np.float64)[None, :]
    w = np.cos(np.pi * (col + 0.5) * row / dim)
    w[0, :] *= np.sqrt(1.0 / dim)
    w[1:, :] *= np.sqrt(2.0 / dim)
    return w


def lg_ssm_drift_matrix(params: LGSSMParams) -> np.ndarray:
    """Build the LG-SSM continuous-time drift F = V B_c V⁻¹, shape [D, D].

    B_c is the §15.3 block-diagonal of damped oscillatory blocks
    [[-γ_i, -ω_i], [ω_i, -γ_i]] (eigenvalues -γ_i ± iω_i); V = diag(σ)·W with
    cond₂(V) = kappa_v (§3.5). V⁻¹ = Wᵀ·diag(1/σ) in closed form, so no
    numerical matrix inversion enters the construction.

    Args:
        params: LGSSMParams. D = 2·len(params.gammas).

    Returns:
        F, float64 array [D, D], eigenvalues -γ_i ± iω_i for every kappa_v.

    Example:
        >>> F = lg_ssm_drift_matrix(lg_ssm_spec().params)
        >>> F.shape  # (8, 8)
    """
    if len(params.gammas) != len(params.omegas) or len(params.gammas) == 0:
        raise ValueError(
            f"lg_ssm requires one omega per gamma with at least one block, got "
            f"{len(params.gammas)} gammas and {len(params.omegas)} omegas."
        )
    if params.kappa_v < 1.0:
        raise ValueError(f"kappa_v is a condition number and must be >= 1, got {params.kappa_v}.")
    dim = 2 * len(params.gammas)
    b = np.zeros((dim, dim), dtype=np.float64)
    for block_idx, (gamma, omega) in enumerate(zip(params.gammas, params.omegas, strict=True)):
        # §15.3: F_i = [[-γ_i, -ω_i], [ω_i, -γ_i]], eigenvalues -γ_i ± iω_i.
        b[2 * block_idx, 2 * block_idx] = -gamma
        b[2 * block_idx, 2 * block_idx + 1] = -omega
        b[2 * block_idx + 1, 2 * block_idx] = omega
        b[2 * block_idx + 1, 2 * block_idx + 1] = -gamma
    w = _lg_ssm_mixing_matrix(dim)
    # σ_j = kappa_v^{j/(D-1)}: geometric spread from 1 to kappa_v, cond₂ = kappa_v.
    sigma = params.kappa_v ** (np.arange(dim, dtype=np.float64) / (dim - 1))
    # F = diag(σ)·W·B_c·Wᵀ·diag(1/σ)  — the §3.5 similarity A = V B V⁻¹ in
    # continuous time. W first rotates B_c into a dense normal matrix, then the
    # diagonal skew couples all directions with ratios up to kappa_v.
    return (sigma[:, None] * w) @ b @ (w.T * (1.0 / sigma)[None, :])


def lg_ssm_discretize_exact(params: LGSSMParams, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exactly discretize the LG-SSM SDE at sampling interval dt → (A, Q).

    Van Loan block-matrix exponential (§15.2, [@vanloan1978computing;
    @sarkka2019applied]): build

      M = [[F, L Q_c Lᵀ], [0, -Fᵀ]] · dt,   Φ = expm(M) = [[Φ₁₁, Φ₁₂], [0, Φ₂₂]]

    then A = Φ₁₁ = exp(F·dt), and Q = Φ₁₂·Aᵀ because Φ₁₂ = Q_dt·Φ₂₂ with
    Φ₂₂ = exp(-Fᵀ·dt) and Φ₂₂⁻¹ = exp(Fᵀ·dt) = Aᵀ — so no matrix inversion.

    Args:
        params: LGSSMParams (defines F and q_c).
        dt: Sampling interval Δ > 0.

    Returns:
        (A, Q): float64 arrays [D, D]. A is the exact transition matrix,
        Q the exact process-noise covariance over one interval.

    Example:
        >>> A, Q = lg_ssm_discretize_exact(lg_ssm_spec().params, dt=1.0)
        >>> np.max(np.abs(np.linalg.eigvals(A)))  # 0.95 for the preset
    """
    try:
        from scipy.linalg import expm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "lg_ssm exact discretization needs scipy (scipy.linalg.expm). "
            "Install scipy in the environment running serie."
        ) from exc

    if dt <= 0.0:
        raise ValueError(f"lg_ssm sampling interval dt must be > 0, got {dt}.")
    if params.q_c <= 0.0:
        raise ValueError(
            f"q_c must be > 0 for a nondegenerate process-noise covariance, got {params.q_c}."
        )
    f = lg_ssm_drift_matrix(params)
    dim = f.shape[0]
    diffusion = params.q_c * np.eye(dim)  # L Q_c Lᵀ with L = I, Q_c = q_c·I
    van_loan_block = np.zeros((2 * dim, 2 * dim), dtype=np.float64)
    van_loan_block[:dim, :dim] = f
    van_loan_block[:dim, dim:] = diffusion
    van_loan_block[dim:, dim:] = -f.T
    phi = expm(van_loan_block * dt)
    a = phi[:dim, :dim]
    q = phi[:dim, dim:] @ a.T
    return a, q


def lg_ssm_spec(
    *,
    periods=(12.0, 24.0, 48.0, 96.0),
    spectral_radius=0.95,
    kappa_v=1.0,
    q_c=None,
    steps=25000,
    dt=1.0,
    burn_in=0,
):
    """Main-benchmark LG-SSM preset: D = 8 latent dims (§2.5 main-benchmark
    state dimension 8–32), one damped oscillatory block per period.

    - periods: cycle lengths in SAMPLING STEPS (§3.6 menu: 12, 24, 48, 96);
      ω_i = 2π/(period_i·dt).
    - spectral_radius: per-step decay |eig(A)| = exp(-γ·dt); equal damping
      across blocks holds ρ(A) fixed while kappa_v varies (§3.7 scenario 1
      uses ρ(A) = 0.95; scenario 4's transient-risk setting is ρ(A) = 0.9
      with non-normal V).
    - kappa_v: eigenvector conditioning κ(V) (§3.6 menu: 1, 5, 20, 100).
      Default 1.0 = normal drift; raise it for controlled non-normality.
    - q_c: process-noise spectral density. Default None resolves to 2γ, which
      standardizes the kappa_v = 1 stationary covariance to the identity
      (baseline per-channel std 1.0, §2.5 "standardize baseline marginal
      scales before applying perturbations").
    """
    from .specs import TrajectorySpec

    gamma = -float(np.log(spectral_radius)) / dt
    resolved_q_c = 2.0 * gamma if q_c is None else q_c
    dim = 2 * len(periods)
    return TrajectorySpec(
        system=ODESystem.LG_SSM,
        params=LGSSMParams(
            gammas=(gamma,) * len(periods),
            omegas=tuple(2.0 * np.pi / (float(period) * dt) for period in periods),
            kappa_v=kappa_v,
            q_c=resolved_q_c,
        ),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=dim,
        method="euler",  # dummy for discrete dispatch (stepping is exact); schema compat like slds
        x0=(0.0,) * dim,
    )


def lg_ssm_small_spec(*, kappa_v=1.0, steps=25000):
    """Small/debug LG-SSM preset: D = 4 (§2.5 small/debug state dimension
    2–4), periods 12 and 24 steps, otherwise identical to lg_ssm_spec."""
    return lg_ssm_spec(periods=(12.0, 24.0), kappa_v=kappa_v, steps=steps)


def lg_ssm_switch_spec(
    *,
    periods=(12.0, 24.0, 48.0, 96.0),
    frequency_multiplier=1.5,
    switch_fraction=0.5,
    spectral_radius=0.95,
    kappa_v=1.0,
    q_c=None,
    steps=25000,
    dt=1.0,
    burn_in=0,
):
    """LG-SSM with one known abrupt frequency-regime switch.

    The system keeps the damping rates, process-noise spectral density, and
    eigenvector conditioning fixed. At ``switch_fraction`` of the trajectory,
    every damped-rotation block changes from angular frequency ``ω_i`` to
    ``frequency_multiplier * ω_i``. Thus the switch changes cycle length while
    preserving the per-step spectral radius and state dimension.

    This is a titration world, not a forecast-failure intervention: the data
    generator itself changes regime, and ``lgssm_switch_truth`` conditions on
    that same known time-varying law.
    """
    from .specs import ShockEvent, ShockSchedule

    if not np.isfinite(frequency_multiplier) or frequency_multiplier <= 0.0:
        raise ValueError(
            f"frequency_multiplier must be finite and > 0, got {frequency_multiplier}."
        )
    if not np.isfinite(switch_fraction) or not 0.0 < switch_fraction < 1.0:
        raise ValueError(f"switch_fraction must lie in (0,1), got {switch_fraction}.")
    base = lg_ssm_spec(
        periods=periods,
        spectral_radius=spectral_radius,
        kappa_v=kappa_v,
        q_c=q_c,
        steps=steps,
        dt=dt,
        burn_in=burn_in,
    )
    if not isinstance(base.params, LGSSMParams):
        raise AssertionError(f"lg_ssm_spec returned unexpected params {type(base.params).__name__}")
    params_after = replace(
        base.params,
        omegas=tuple(frequency_multiplier * omega for omega in base.params.omegas),
    )
    return replace(
        base,
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=params_after,
                t_start=switch_fraction,
                name="lg_ssm_frequency_regime_switch",
            )
        ),
    )


def lg_ssm_regime_switch_spec(
    *,
    periods=(12.0, 24.0, 48.0, 96.0),
    frequency_multiplier=1.5,
    switch_fraction=0.5,
    spectral_radius_before=0.95,
    spectral_radius_after=0.90,
    kappa_v_before=1.0,
    kappa_v_after=5.0,
    q_c_before=None,
    q_c_after=None,
    steps=25000,
    dt=1.0,
    burn_in=0,
):
    """LG-SSM with one known abrupt multi-axis parameter-regime switch.

    This deliberately remains inside one linear-Gaussian model family so the
    complete conditional forecast law remains exactly available through the
    time-varying Kalman recursion in ``lgssm_switch_truth``. It is stronger
    than ``lg_ssm_switch_spec``: at ``switch_fraction`` it simultaneously
    changes

    - cycle lengths through ``ω_after = frequency_multiplier * ω_before``;
    - memory through the per-step spectral radius (default ``0.95 -> 0.90``);
    - cross-channel transient geometry through ``kappa_v`` (default ``1 -> 5``);
    - process-noise spectral density, defaulting after the switch to
      ``q_c_after = 2 * gamma_after``.

    The latent state is continuous across the boundary and the simulator keeps
    consuming the same innovation stream. Therefore this is a piecewise
    parameter-regime switch, not a splice between unrelated dynamical systems.
    Calling that distinction out prevents the stronger world from being
    misreported as a cross-family "system replacement."
    """
    from .specs import ShockEvent, ShockSchedule

    if not np.isfinite(frequency_multiplier) or frequency_multiplier <= 0.0:
        raise ValueError(
            f"frequency_multiplier must be finite and > 0, got {frequency_multiplier}."
        )
    if not np.isfinite(switch_fraction) or not 0.0 < switch_fraction < 1.0:
        raise ValueError(f"switch_fraction must lie in (0,1), got {switch_fraction}.")
    for name, radius in (
        ("spectral_radius_before", spectral_radius_before),
        ("spectral_radius_after", spectral_radius_after),
    ):
        if not np.isfinite(radius) or not 0.0 < radius < 1.0:
            raise ValueError(f"{name} must lie in (0,1), got {radius}.")
    if not np.isfinite(kappa_v_after) or kappa_v_after < 1.0:
        raise ValueError(f"kappa_v_after must be finite and >= 1, got {kappa_v_after}.")

    base = lg_ssm_spec(
        periods=periods,
        spectral_radius=spectral_radius_before,
        kappa_v=kappa_v_before,
        q_c=q_c_before,
        steps=steps,
        dt=dt,
        burn_in=burn_in,
    )
    if not isinstance(base.params, LGSSMParams):
        raise AssertionError(f"lg_ssm_spec returned unexpected params {type(base.params).__name__}")
    gamma_after = -float(np.log(spectral_radius_after)) / dt
    resolved_q_c_after = 2.0 * gamma_after if q_c_after is None else q_c_after
    params_after = LGSSMParams(
        gammas=(gamma_after,) * len(base.params.gammas),
        omegas=tuple(frequency_multiplier * omega for omega in base.params.omegas),
        kappa_v=kappa_v_after,
        q_c=resolved_q_c_after,
    )
    return replace(
        base,
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=params_after,
                t_start=switch_fraction,
                name="lg_ssm_multiaxis_parameter_regime_switch",
            )
        ),
    )


class LGSSMSwitchPredictiveLaw(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=False,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Exact conditional Gaussian law returned by ``lgssm_switch_truth``.

    ``mean_hd`` is timestep-major ``[H,D]`` and ``covariance_pp`` uses flattened
    coordinate ``p=h*D+d``. The filtered state at the final context step is
    included so Monte-Carlo validation can independently simulate the future
    recurrence from the same conditional starting law.
    """

    mean_hd: np.ndarray
    covariance_pp: np.ndarray
    filtered_mean_d: np.ndarray
    filtered_covariance_dd: np.ndarray

    __hash__ = record_hash


def lgssm_switch_truth(
    observations_td: np.ndarray,
    *,
    params_before: LGSSMParams,
    params_after: LGSSMParams,
    dt: float,
    switch_step: int,
    horizon: int,
    observation_covariance_dd: np.ndarray,
) -> LGSSMSwitchPredictiveLaw:
    """Return the exact stacked future law for a known switched LG-SSM.

    The observed history begins at absolute system step 0, where the catalog
    generator has the known deterministic initial state ``x_0=0``. For every
    destination step ``k >= 1``, the transition is ``(A_before,Q_before)`` when
    ``k < switch_step`` and ``(A_after,Q_after)`` otherwise, matching
    ``_run_lg_ssm``: the parameter event is applied before generating the row
    at ``switch_step``.

    With identity observations ``y_k=x_k+v_k`` and ``v_k~N(0,R)``, a
    time-varying Kalman filter gives ``p(x_t | y_0:t)``. The future recursion
    then gives each mean and latent covariance. For future indices ``i<j``,

    ``Cov(x_i,x_j | history) = P_i (A_j ... A_(i+1))^T``.

    ``R`` is added only to the diagonal future-time blocks, because future
    observation noise is independent across steps. No samples, fitting, or
    gradient descent enter this function.
    """
    observations = np.asarray(observations_td, dtype=np.float64)
    observation_covariance = np.asarray(observation_covariance_dd, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[0] < 1:
        raise ValueError(f"observations_td must have shape [T>=1,D], got {observations.shape}.")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if switch_step < 1:
        raise ValueError(f"switch_step must be >= 1, got {switch_step}.")
    dim = observations.shape[1]
    if observation_covariance.shape != (dim, dim):
        raise ValueError(
            f"observation_covariance_dd must have shape {(dim, dim)}, got {observation_covariance.shape}."
        )
    if (
        not np.isfinite(observations).all()
        or not np.isfinite(observation_covariance).all()
        or not np.allclose(observation_covariance, observation_covariance.T)
    ):
        raise ValueError("observations and symmetric observation covariance must be finite.")

    before_transition, before_process_covariance = lg_ssm_discretize_exact(params_before, dt)
    after_transition, after_process_covariance = lg_ssm_discretize_exact(params_after, dt)
    expected = (dim, dim)
    for name, matrix in {
        "before_transition": before_transition,
        "before_process_covariance": before_process_covariance,
        "after_transition": after_transition,
        "after_process_covariance": after_process_covariance,
    }.items():
        if matrix.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {matrix.shape}.")

    def matrices_for_destination_step(destination_step: int) -> tuple[np.ndarray, np.ndarray]:
        if destination_step < switch_step:
            return before_transition, before_process_covariance
        return after_transition, after_process_covariance

    identity = np.eye(dim)
    filtered_mean = np.zeros(dim, dtype=np.float64)
    filtered_covariance = np.zeros((dim, dim), dtype=np.float64)
    for step_idx, observation in enumerate(observations):
        if step_idx > 0:
            transition, process_covariance = matrices_for_destination_step(step_idx)
            filtered_mean = transition @ filtered_mean
            filtered_covariance = (
                transition @ filtered_covariance @ transition.T + process_covariance
            )
        innovation_covariance = filtered_covariance + observation_covariance
        kalman_gain = np.linalg.solve(innovation_covariance, filtered_covariance.T).T
        filtered_mean = filtered_mean + kalman_gain @ (observation - filtered_mean)
        residual_map = identity - kalman_gain
        filtered_covariance = (
            residual_map @ filtered_covariance @ residual_map.T
            + kalman_gain @ observation_covariance @ kalman_gain.T
        )
        filtered_covariance = 0.5 * (filtered_covariance + filtered_covariance.T)

    future_means: list[np.ndarray] = []
    future_covariances: list[np.ndarray] = []
    future_transitions: list[np.ndarray] = []
    predicted_mean = filtered_mean.copy()
    predicted_covariance = filtered_covariance.copy()
    last_context_step = observations.shape[0] - 1
    for horizon_idx in range(horizon):
        destination_step = last_context_step + horizon_idx + 1
        transition, process_covariance = matrices_for_destination_step(destination_step)
        predicted_mean = transition @ predicted_mean
        predicted_covariance = transition @ predicted_covariance @ transition.T + process_covariance
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
        future_means.append(predicted_mean.copy())
        future_covariances.append(predicted_covariance.copy())
        future_transitions.append(transition)

    joint_covariance = np.empty((horizon * dim, horizon * dim), dtype=np.float64)
    for left_idx, left_covariance in enumerate(future_covariances):
        left_slice = slice(left_idx * dim, (left_idx + 1) * dim)
        propagator = np.eye(dim)
        for right_idx in range(left_idx, horizon):
            right_slice = slice(right_idx * dim, (right_idx + 1) * dim)
            if right_idx > left_idx:
                propagator = future_transitions[right_idx] @ propagator
            cross_covariance = left_covariance @ propagator.T
            if right_idx == left_idx:
                cross_covariance = cross_covariance + observation_covariance
            joint_covariance[left_slice, right_slice] = cross_covariance
            joint_covariance[right_slice, left_slice] = cross_covariance.T

    return LGSSMSwitchPredictiveLaw(
        mean_hd=np.stack(future_means, axis=0),
        covariance_pp=0.5 * (joint_covariance + joint_covariance.T),
        filtered_mean_d=filtered_mean,
        filtered_covariance_dd=filtered_covariance,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class KSParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """
    1D Kuramoto-Sivashinsky PDE (1D).

    Spatiotemporal chaos. Integrated with ETDRK4 (Kassam & Trefethen 2005).

    u_t = -u·u_x - ν·u_xx - u_xxxx,  x ∈ [0, Lx], periodic
    Linear part in Fourier: L(k) = ν k^2 - k^4
    Nonlinear part: N(u) = - 1/2 ∂x(u^2)  (computed pseudospectrally)
    Time stepping: ETDRK4 (Kassam & Trefethen 2005)

    REQUIRE: nx >= 2 and typically power-of-2 (recommendation)

    NOTE: dt stability depends on params/resolution (hard to validate universally)
    """

    nu: float  # Diffusion coefficient (on -u_xx term)
    Lx: float  # Domain length
    nx: int = 128  # Number of spatial points (power of 2 recommended)
    dealias: bool = True  # Whether to apply 2/3-rule dealiasing

    __hash__ = record_hash


def ks_spec(*, nx=64, Lx=22.0, nu=1.0, steps=27500, dt=1e-2, burn_in=2500):
    from .specs import TrajectorySpec

    # CHG (2026-07-01, user-approved scientific fix): x0 was zeros(nx), which
    # is a fixed point of the KS flow — ETDRK4 kept the trajectory identically
    # zero, so this preset generated a degenerate all-zero series. Seed field
    # is now the standard Kassam–Trefethen (2005) initial condition
    # generalized to domain length Lx:
    #     u0(x) = cos(2πx/Lx) · (1 + sin(2πx/Lx)),   x_j = j·Lx/nx
    # (reduces exactly to their cos(x/16)(1+sin(x/16)) at Lx = 32π).
    # burn_in=2500 (t = 25) discards the smooth-IC transient into the chaotic
    # attractor; steps grew by the same 2500 so the usable output length stays
    # 25001 rows (t ≈ 250). Shock timing is unaffected: ShockEvent t_start is
    # relative to the post-burn-in trajectory (integrators.py shock_step).
    grid_x = np.arange(nx, dtype=np.float64) * (Lx / nx)
    phase = 2.0 * np.pi * grid_x / Lx
    seed_field = np.cos(phase) * (1.0 + np.sin(phase))

    return TrajectorySpec(
        system=ODESystem.KS,
        params=KSParams(nu=nu, Lx=Lx, nx=nx, dealias=True),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=nx,
        method="etdrk4",
        x0=tuple(float(v) for v in seed_field),
    )


def ks_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return replace(
        ks_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=KSParams(nx=64, Lx=22.0, nu=0.80, dealias=True),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Diagonal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DiagonalParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=False,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """dx/dt = diag(eigenvalues) · x — trivial eigenvalue recovery test.
    Ground truth: learned discrete eigenvalues should be exp(λ_i · dt)."""

    eigenvalues: tuple[float, ...] = (-0.5, -1.0, -2.0)

    __hash__ = record_hash


def _f_diagonal(state: np.ndarray, t: float, eigenvalues: tuple[float, ...]) -> np.ndarray:
    return np.array(eigenvalues, dtype=np.float64) * state


def diagonal_spec(
    *,
    eigenvalues: tuple[float, ...] = (-0.5, -1.0, -2.0),
    steps: int = 10000,
    dt: float = 0.01,
    burn_in: int = 0,
):
    """dx/dt = diag(λ)x — trivial eigenvalue test.
    Ground truth: discrete propagator eigenvalues = exp(λ_i · dt).

    NOTE: Output shape is (steps - burn_in + 1, dim), not (steps, dim).
    """
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.DIAGONAL,
        params=DiagonalParams(eigenvalues=eigenvalues),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=len(eigenvalues),
        x0=tuple([1.0] * len(eigenvalues)),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprott A, Aizawa, Thomas
# One member of each family in dysts (Gilpin), recorded like lorenz.series:
# RK4, dt 0.01, 35999 steps, burn-in 0, so a long window of 36000 rows exists.
# The initial state and the coefficients are dysts 0.96's published values, not
# Lorenz's (1, 0.98, 1.1). dysts integrates at its own much smaller dt; this
# recording step is Fern's, the same one lorenz.series uses.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SprottAParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Sprott A has no coefficients. The struct exists so the parameter union stays uniform."""

    __hash__ = record_hash


def _f_sprott_a(state: np.ndarray, t: float) -> np.ndarray:
    """Sprott A (Sprott 1994), the same right-hand side as dysts.flows.SprottA.

    dx/dt = y
    dy/dt = -x + y z
    dz/dt = 1 - y^2
    """
    x, y, z = state
    return np.array([y, -x + y * z, 1.0 - y * y], dtype=np.float64)


def sprott_a_spec(*, steps=35999, dt=1e-2, burn_in=0):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.SPROTT_A,
        params=SprottAParams(),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=3,
        x0=(1.54131864, 1.43079727, -1.52555412),
    )


class AizawaParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Aizawa (1982), coefficients from dysts 0.96.

    dx/dt = (z - b) x - d y
    dy/dt = d x + (z - b) y
    dz/dt = c + a z - z^3/3 - (x^2 + y^2) (1 + e z) + f z x^3
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    __hash__ = record_hash


def _f_aizawa(
    state: np.ndarray, t: float, a: float, b: float, c: float, d: float, e: float, f: float
) -> np.ndarray:
    x, y, z = state
    return np.array(
        [
            (z - b) * x - d * y,
            d * x + (z - b) * y,
            c + a * z - (z**3) / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * x**3,
        ],
        dtype=np.float64,
    )


def aizawa_spec(
    *,
    a=0.95,
    b=0.7,
    c=0.6,
    d=3.5,
    e=0.25,
    f=0.1,
    steps=35999,
    dt=1e-2,
    burn_in=0,
):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.AIZAWA,
        params=AizawaParams(a=a, b=b, c=c, d=d, e=e, f=f),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=3,
        x0=(-0.78450179, -0.62887672, -0.17620268),
    )


class ThomasParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Thomas's cyclically symmetric system, dysts.flows.Thomas (two coefficients).

    dx/dt = -a x + b sin(y)
    dy/dt = -a y + b sin(z)
    dz/dt = -a z + b sin(x)

    dysts 0.96 uses a=1.85, b=10. The one-parameter labyrinth form sin(y) - b x
    is the special case a=b, b_there = 1, and is not this recording.
    """

    a: float
    b: float

    __hash__ = record_hash


def _f_thomas(state: np.ndarray, t: float, a: float, b: float) -> np.ndarray:
    x, y, z = state
    return np.array(
        [-a * x + b * np.sin(y), -a * y + b * np.sin(z), -a * z + b * np.sin(x)],
        dtype=np.float64,
    )


def thomas_spec(*, a=1.85, b=10.0, steps=35999, dt=1e-2, burn_in=0):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.THOMAS,
        params=ThomasParams(a=a, b=b),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=3,
        x0=(2.199442, 2.3634225, 3.220197),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hénon–Heiles, coupled Rössler, Lorenz-84, double pendulum,
# FitzHugh–Nagumo, Hindmarsh–Rose, Rikitake.
# Equations and default numbers are the rule functions in
# PredefinedDynamicalSystems.jl (the code, where its math comment and its
# code disagree). Julia is not called. Five use the lorenz.series cut:
# RK4, dt 0.01, 35999 steps, burn-in 0, 36000 rows. Coupled Rössler and
# Hindmarsh–Rose do not; their spec docstrings say why.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class HenonHeilesParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Hénon–Heiles has no coefficients. The struct keeps the parameter union uniform."""

    __hash__ = record_hash


def _f_henon_heiles(state: np.ndarray, t: float) -> np.ndarray:
    """Hénon–Heiles (1964). State is (x, y, p_x, p_y).

    dx/dt = p_x
    dy/dt = p_y
    dp_x/dt = -x - 2 x y
    dp_y/dt = -y - (x² - y²)

    Energy E = ½(p_x² + p_y²) + ½(x² + y²) + x² y - y³/3 is constant for the
    true flow. There is no attractor: an orbit stays on its energy surface.
    """
    x, y, p_x, p_y = state
    return np.array(
        [p_x, p_y, -x - 2.0 * x * y, -y - (x * x - y * y)],
        dtype=np.float64,
    )


def henon_heiles_spec(*, steps=35999, dt=1e-2, burn_in=0):
    """The published initial state (0, -0.25, 0.42081, 0) has energy 1/8."""
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.HENON_HEILES,
        params=HenonHeilesParams(),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=4,
        x0=(0.0, -0.25, 0.42081, 0.0),
    )


class CoupledRoesslerParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Two Rössler oscillators. k_2 = 0 means oscillator 2 drives oscillator 1 and is not driven back.

    State order is (x1, y1, z1, x2, y2, z2).

    dx1/dt = -ω1 y1 - z1
    dy1/dt = ω1 x1 + a y1 + k1 (y2 - y1)
    dz1/dt = b + z1 (x1 - c)
    dx2/dt = -ω2 y2 - z2
    dy2/dt = ω2 x2 + a y2 + k2 (y1 - y2)
    dz2/dt = b + z2 (x2 - c)
    """

    omega_1: float
    omega_2: float
    a: float
    b: float
    c: float
    k_1: float
    k_2: float

    __hash__ = record_hash


def _f_coupled_roessler(
    state: np.ndarray,
    t: float,
    omega_1: float,
    omega_2: float,
    a: float,
    b: float,
    c: float,
    k_1: float,
    k_2: float,
) -> np.ndarray:
    x1, y1, z1, x2, y2, z2 = state
    return np.array(
        [
            -omega_1 * y1 - z1,
            omega_1 * x1 + a * y1 + k_1 * (y2 - y1),
            b + z1 * (x1 - c),
            -omega_2 * y2 - z2,
            omega_2 * x2 + a * y2 + k_2 * (y1 - y2),
            b + z2 * (x2 - c),
        ],
        dtype=np.float64,
    )


def coupled_roessler_spec(
    *,
    omega_1=0.18,
    omega_2=0.22,
    a=0.2,
    b=0.2,
    c=5.7,
    k_1=0.115,
    k_2=0.0,
    steps=100_000,
    dt=1e-3,
    burn_in=0,
):
    """Drive-response Rössler pair, recorded through t = 100.

    The published frequencies 0.18 and 0.22 are about five times slower than
    the ordinary Rössler, whose rotation is not scaled. The z-spike stays just
    as fast, so it is much sharper. RK4 at the lorenz.series step dt = 0.01
    overflows before t = 100. At dt = 0.001 the same initial state stays finite
    through t = 100. The largest component is y, about −9.15e5; z peaks near
    2.3e5. dt = 0.0005 and dt = 0.0002 give that same largest component.
    Past about t = 118, RK4 overflows on a later spike
    even at dt = 1e-4. This preset therefore stops at t = 100 (100001 rows),
    not at the 36000-row, dt = 0.01 cut the other systems use.
    """
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.COUPLED_ROESSLER,
        params=CoupledRoesslerParams(
            omega_1=omega_1, omega_2=omega_2, a=a, b=b, c=c, k_1=k_1, k_2=k_2
        ),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=6,
        x0=(1.0, -2.0, 0.0, 0.11, 0.2, 0.1),
    )


class Lorenz84Params(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Lorenz's 1984 low-order atmospheric circulation.

    dx/dt = -y² - z² - a x + a F
    dy/dt = x y - y - b x z + G
    dz/dt = b x y + x z - z

    F is the symmetric heating contrast, G the longitude heating contrast.
    At the published defaults the flow has four coexisting attractors.
    One recorded series uses one initial state and therefore lands in one of them.
    """

    f: float
    g: float
    a: float
    b: float

    __hash__ = record_hash


def _f_lorenz84(state: np.ndarray, t: float, f: float, g: float, a: float, b: float) -> np.ndarray:
    x, y, z = state
    return np.array(
        [
            -y * y - z * z - a * x + a * f,
            x * y - y - b * x * z + g,
            b * x * y + x * z - z,
        ],
        dtype=np.float64,
    )


def lorenz84_spec(*, f=6.846, g=1.287, a=0.25, b=4.0, steps=35999, dt=1e-2, burn_in=0):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.LORENZ84,
        params=Lorenz84Params(f=f, g=g, a=a, b=b),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=3,
        x0=(0.1, 0.1, 0.1),
    )


class DoublePendulumParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Planar double pendulum. State is (θ₁, ω₁, θ₂, ω₂), angles from hanging straight down.

    gravity, length_1, length_2, mass_1, mass_2 are G, L1, L2, M1, M2.
    φ = θ₂ - θ₁ and Δ = (M₁ + M₂) - M₂ cos²φ.
    The acceleration formulas are doublependulum_rule in PredefinedDynamicalSystems.jl.
    """

    gravity: float
    length_1: float
    length_2: float
    mass_1: float
    mass_2: float

    __hash__ = record_hash


def _f_double_pendulum(
    state: np.ndarray,
    t: float,
    gravity: float,
    length_1: float,
    length_2: float,
    mass_1: float,
    mass_2: float,
) -> np.ndarray:
    theta_1, omega_1, theta_2, omega_2 = state
    phi = theta_2 - theta_1
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta_1 = np.sin(theta_1)
    sin_theta_2 = np.sin(theta_2)
    delta = (mass_1 + mass_2) - mass_2 * cos_phi * cos_phi
    d_theta_1 = omega_1
    d_omega_1 = (
        mass_2
        * (
            cos_phi * (length_1 * omega_1 * omega_1 * sin_phi + gravity * sin_theta_2)
            + length_2 * omega_2 * omega_2 * sin_phi
        )
        - (mass_1 + mass_2) * gravity * sin_theta_1
    ) / (length_1 * delta)
    d_theta_2 = omega_2
    d_omega_2 = (
        -mass_2 * length_2 * omega_2 * omega_2 * sin_phi * cos_phi
        + (mass_1 + mass_2)
        * (gravity * (sin_theta_1 * cos_phi - sin_theta_2) - length_1 * omega_1 * omega_1 * sin_phi)
    ) / (length_2 * delta)
    return np.array([d_theta_1, d_omega_1, d_theta_2, d_omega_2], dtype=np.float64)


def double_pendulum_spec(
    *,
    gravity=10.0,
    length_1=1.0,
    length_2=1.0,
    mass_1=1.0,
    mass_2=1.0,
    steps=35999,
    dt=1e-2,
    burn_in=0,
):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.DOUBLE_PENDULUM,
        params=DoublePendulumParams(
            gravity=gravity,
            length_1=length_1,
            length_2=length_2,
            mass_1=mass_1,
            mass_2=mass_2,
        ),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=4,
        x0=(float(np.pi) / 2.0, 0.0, 0.0, 0.5),
    )


class FitzHughNagumoParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """FitzHugh–Nagumo neuron. State is (v, w): membrane and recovery.

    dv/dt = a v (v - b) (1 - v) - w + I
    dw/dt = ε (v - w)

    The published default applied current I is 0. From (0.5, 0.5) the orbit
    relaxes to rest. It is not a periodic spike train.
    """

    a: float
    b: float
    epsilon: float
    applied_current: float

    __hash__ = record_hash


def _f_fitzhugh_nagumo(
    state: np.ndarray, t: float, a: float, b: float, epsilon: float, applied_current: float
) -> np.ndarray:
    membrane, recovery = state
    return np.array(
        [
            a * membrane * (membrane - b) * (1.0 - membrane) - recovery + applied_current,
            epsilon * (membrane - recovery),
        ],
        dtype=np.float64,
    )


def fitzhugh_nagumo_spec(
    *, a=3.0, b=0.2, epsilon=0.01, applied_current=0.0, steps=35999, dt=1e-2, burn_in=0
):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.FITZHUGH_NAGUMO,
        params=FitzHughNagumoParams(a=a, b=b, epsilon=epsilon, applied_current=applied_current),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=2,
        x0=(0.5, 0.5),
    )


class HindmarshRoseParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Hindmarsh–Rose neuron. State is (x, y, z): membrane, fast ionic current, slow adaptation.

    dx/dt = y - a x³ + b x² - z + I
    dy/dt = c - d x² - y
    dz/dt = r (s (x - x_rest) - z)

    The published default applied current I = 2 gives periodic bursting, not a strange attractor.
    """

    a: float
    b: float
    c: float
    d: float
    r: float
    s: float
    x_rest: float
    applied_current: float

    __hash__ = record_hash


def _f_hindmarsh_rose(
    state: np.ndarray,
    t: float,
    a: float,
    b: float,
    c: float,
    d: float,
    r: float,
    s: float,
    x_rest: float,
    applied_current: float,
) -> np.ndarray:
    membrane, fast, adaptation = state
    return np.array(
        [
            fast - a * membrane**3 + b * membrane * membrane - adaptation + applied_current,
            c - d * membrane * membrane - fast,
            r * (s * (membrane - x_rest) - adaptation),
        ],
        dtype=np.float64,
    )


def hindmarsh_rose_spec(
    *,
    a=1.0,
    b=3.0,
    c=1.0,
    d=5.0,
    r=0.001,
    s=4.0,
    x_rest=-1.6,
    applied_current=2.0,
    steps=200_000,
    dt=1e-2,
    burn_in=0,
):
    """Hindmarsh–Rose at the published current I = 2, recorded through t = 2000.

    The adaptation rate r = 0.001 is a thousand times slower than the spike.
    A lorenz.series-length recording (t = 360) ends inside the first run of
    spikes: the membrane crosses 0 about every 4 to 10 time units and the
    silence of about 300 time units has not happened yet. Through t = 2000
    that silence repeats (about 301 time units, three times in this window).
    dt stays 0.01. The bank has 200001 rows, not 36000.
    """
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.HINDMARSH_ROSE,
        params=HindmarshRoseParams(
            a=a,
            b=b,
            c=c,
            d=d,
            r=r,
            s=s,
            x_rest=x_rest,
            applied_current=applied_current,
        ),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=3,
        x0=(-1.0, 0.0, 0.0),
    )


class RikitakeParams(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=True,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Rikitake two-disk dynamo, a toy of geomagnetic reversals.

    dx/dt = -μ x + y z
    dy/dt = -μ y + x (z - α)
    dz/dt = 1 - x y
    """

    mu: float
    alpha: float

    __hash__ = record_hash


def _f_rikitake(state: np.ndarray, t: float, mu: float, alpha: float) -> np.ndarray:
    x, y, z = state
    return np.array(
        [-mu * x + y * z, -mu * y + x * (z - alpha), 1.0 - x * y],
        dtype=np.float64,
    )


def rikitake_spec(*, mu=1.0, alpha=1.0, steps=35999, dt=1e-2, burn_in=0):
    from .specs import TrajectorySpec

    return TrajectorySpec(
        system=ODESystem.RIKITAKE,
        params=RikitakeParams(mu=mu, alpha=alpha),
        dt=dt,
        steps=steps,
        burn_in=burn_in,
        dim=3,
        x0=(1.0, 0.0, 0.6),
    )


# ═══════════════════════════════════════════════════════════════════════════
# SystemParams union — what TrajectorySpec.params accepts
# ═══════════════════════════════════════════════════════════════════════════

SystemParams = (
    LorenzParams
    | RosslerParams
    | HyperRosslerParams
    | ChuaParams
    | DuffingParams
    | VanderPolParams
    | Lorenz96Params
    | HenonParams
    | LogisticParams
    | OUParams
    | DoubleWellParams
    | LGSSMParams
    | SLDSParams
    | SeasonalARParams
    | SeasonalARBJParams
    | GARCHParams
    | KSParams
    | DiagonalParams
    | SprottAParams
    | AizawaParams
    | ThomasParams
    | HenonHeilesParams
    | CoupledRoesslerParams
    | Lorenz84Params
    | DoublePendulumParams
    | FitzHughNagumoParams
    | HindmarshRoseParams
    | RikitakeParams
)


# ═══════════════════════════════════════════════════════════════════════════
# System registry
# ═══════════════════════════════════════════════════════════════════════════


class SystemSpec(
    StructRecord,
    Struct,
    frozen=True,
    kw_only=False,
    eq=False,
    weakref=True,
    forbid_unknown_fields=True,
):
    """Specification for a dynamical system (structure only, no defaults)."""

    name: str
    dim: int  # -1 for variable dimension
    param_type: type[SystemParams]  # AttrsInstance type for params
    kind: Literal["ode", "sde", "map", "pde", "discrete"]
    time_dependent: bool
    f_py: Callable  # Python vector field

    __hash__ = record_hash


SYSTEMS: dict[str, SystemSpec] = {
    "lorenz": SystemSpec(
        name="lorenz",
        dim=3,
        param_type=LorenzParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_lorenz,
    ),
    "rossler": SystemSpec(
        name="rossler",
        dim=3,
        param_type=RosslerParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_rossler,
    ),
    "hyper_rossler": SystemSpec(
        name="hyper_rossler",
        dim=4,
        param_type=HyperRosslerParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_hyper_rossler,
    ),
    "chua": SystemSpec(
        name="chua",
        dim=3,
        param_type=ChuaParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_chua,
    ),
    "duffing": SystemSpec(
        name="duffing",
        dim=2,
        param_type=DuffingParams,
        kind="ode",
        time_dependent=True,
        f_py=_f_duffing,
    ),
    "vanderpol": SystemSpec(
        name="vanderpol",
        dim=2,
        param_type=VanderPolParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_vanderpol,
    ),
    "lorenz96": SystemSpec(
        name="lorenz96",
        dim=-1,
        param_type=Lorenz96Params,
        kind="ode",
        time_dependent=False,
        f_py=_f_lorenz96,
    ),
    "henon": SystemSpec(
        name="henon",
        dim=2,
        param_type=HenonParams,
        kind="map",
        time_dependent=False,
        f_py=_f_henon,
    ),
    "logistic": SystemSpec(
        name="logistic",
        dim=1,
        param_type=LogisticParams,
        kind="map",
        time_dependent=False,
        f_py=_f_logistic,
    ),
    "ou": SystemSpec(
        name="ou",
        dim=1,
        param_type=OUParams,
        kind="sde",
        time_dependent=False,
        f_py=_f_ou_drift,
    ),
    "doublewell": SystemSpec(
        name="doublewell",
        dim=1,
        param_type=DoubleWellParams,
        kind="sde",
        time_dependent=False,
        f_py=_f_doublewell_drift,
    ),
    "diagonal": SystemSpec(
        name="diagonal",
        dim=-1,
        param_type=DiagonalParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_diagonal,
    ),
    "sprott_a": SystemSpec(
        name="sprott_a",
        dim=3,
        param_type=SprottAParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_sprott_a,
    ),
    "aizawa": SystemSpec(
        name="aizawa",
        dim=3,
        param_type=AizawaParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_aizawa,
    ),
    "thomas": SystemSpec(
        name="thomas",
        dim=3,
        param_type=ThomasParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_thomas,
    ),
    "henon_heiles": SystemSpec(
        name="henon_heiles",
        dim=4,
        param_type=HenonHeilesParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_henon_heiles,
    ),
    "coupled_roessler": SystemSpec(
        name="coupled_roessler",
        dim=6,
        param_type=CoupledRoesslerParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_coupled_roessler,
    ),
    "lorenz84": SystemSpec(
        name="lorenz84",
        dim=3,
        param_type=Lorenz84Params,
        kind="ode",
        time_dependent=False,
        f_py=_f_lorenz84,
    ),
    "double_pendulum": SystemSpec(
        name="double_pendulum",
        dim=4,
        param_type=DoublePendulumParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_double_pendulum,
    ),
    "fitzhugh_nagumo": SystemSpec(
        name="fitzhugh_nagumo",
        dim=2,
        param_type=FitzHughNagumoParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_fitzhugh_nagumo,
    ),
    "hindmarsh_rose": SystemSpec(
        name="hindmarsh_rose",
        dim=3,
        param_type=HindmarshRoseParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_hindmarsh_rose,
    ),
    "rikitake": SystemSpec(
        name="rikitake",
        dim=3,
        param_type=RikitakeParams,
        kind="ode",
        time_dependent=False,
        f_py=_f_rikitake,
    ),
}


# lg_ssm is a continuous-time SDE, but its generator is an exact stepwise
# recurrence (A, Q from the Van Loan discretization), so it dispatches with
# the discrete statistical simulators rather than the Euler-Maruyama path.
# CHG 2026-10-06: seasonal_ar_bj, the Box–Jenkins seasonal AR, steps by index like seasonal_ar.
DISCRETE_SYSTEMS = {"slds", "seasonal_ar", "seasonal_ar_bj", "garch", "lg_ssm"}

# Named initial-condition sources are Recipe cells, not new dynamical systems.
# Their values were fixed before the corresponding campaigns ran; keep the
# coordinates as data and reuse the ordinary typed Lorenz/Rössler specs.
_ROSSLER_HOLDOUT_INITIAL_CONDITIONS = (
    (0.99, 0.99, 0.99),
    (0.99, 0.99, 1.01),
    (0.99, 1.01, 0.99),
    (0.99, 1.01, 1.01),
    (1.01, 0.99, 0.99),
    (1.01, 0.99, 1.01),
    (1.01, 1.01, 0.99),
    (1.01, 1.01, 1.01),
)

_ROSSLER_RANDOM_INITIAL_CONDITIONS = (
    (0.9913589285360874, 0.9964693520999365, 0.9969714152588378),
    (1.0023712242555267, 1.0032361788447315, 1.0086349895198503),
    (0.9983109620583542, 0.997244910963987, 1.0059314342072596),
    (0.9988722008162245, 1.0087166901961397, 1.001766576110946),
    (0.9941950653220915, 1.007970007119063, 1.007992588454839),
    (0.9952357682610365, 0.9965152137573037, 1.0033643866830637),
    (0.9980755851763866, 0.9983412542999786, 0.997459721253305),
    (0.9981867674889127, 0.997156698119659, 0.9909477205765774),
    (0.9986994139947585, 0.996426547188648, 1.0092897303973374),
    (1.007009346643334, 1.0065884767654425, 0.9977558143517135),
    (0.9978908150829802, 1.006323109514108, 0.992399975600262),
    (0.9970080933479296, 1.0078391629459762, 1.008301687477164),
    (1.0054512181579742, 1.0068357763976359, 1.0056624446933773),
    (1.0009111090038931, 0.9969723363603481, 0.9959078707035146),
    (0.9937795393984853, 0.9986275958243241, 0.9904306142562959),
    (1.0037225046260485, 0.9921321284829052, 1.0048187731345346),
)

_LORENZ_RANDOM_INITIAL_CONDITIONS = (
    (0.42436157047691725, 0.13387029225556435, 1.8456599328215408),
    (0.26214450641798437, 1.445845993322646, 1.8512344128298253),
    (0.3868083991867488, 0.4809231301244763, 0.7440743324204397),
    (1.734446312565029, 0.1262599206497519, 1.5526288169842803),
    (1.5092157959428136, 1.962173083638532, 0.6084641074126758),
    (0.6618332106000988, 1.4100706642466636, 0.909404702879385),
    (0.9237372122111931, 1.168850602090294, 0.3221176360323419),
    (1.5753405621002174, 1.5769144882314774, 1.6167860936846463),
    (0.03531552372176616, 0.19456815254927284, 1.9810217144318145),
    (0.7044626387284858, 1.6134342444494865, 0.9238313102943174),
    (1.099633234083752, 1.5874496984691822, 0.6935227134624633),
    (1.401623215746442, 0.6275329702727068, 1.0929224527354438),
    (0.19539487959717783, 0.9722462694091378, 0.30254853039734897),
    (1.294723898745695, 0.039347671617752766, 0.7056032832591432),
    (1.3462130347533259, 1.3484673318779647, 1.9185375537795222),
    (0.07975996295866583, 1.3642066367795043, 1.7588223539298788),
)

# String name -> preset function (for build_dgp_bank("lorenz") convenience)
_ALL_PRESETS = {
    "lorenz": lorenz_spec,
    "lorenz_param_shock": lorenz_param_shock_spec,
    "lorenz_state_shock": lorenz_state_shock_spec,
    "lorenz_switch": lorenz_switch_spec,
    "rossler": rossler_spec,
    "rossler_param_shock": rossler_param_shock_spec,
    "hyper_rossler": hyper_rossler_spec,
    "chua": chua_spec,
    "chua_param_shock": chua_param_shock_spec,
    "chua_switch": chua_switch_spec,
    "duffing": duffing_spec,
    "vanderpol": vanderpol_spec,
    "lorenz96": lorenz96_spec,
    "lorenz96_switch": lorenz96_switch_spec,
    "henon": henon_spec,
    "logistic": logistic_spec,
    "ou": ou_spec,
    "ou_param_shock": ou_param_shock_spec,
    "doublewell": doublewell_spec,
    "doublewell_barrier_collapse": doublewell_barrier_collapse_spec,
    "doublewell_param_shock": doublewell_param_shock_spec,
    "doublewell_switch": doublewell_switch_spec,
    "doublewell_diffusion_shock": doublewell_diffusion_shock_spec,
    "lg_ssm": lg_ssm_spec,
    "lg_ssm_small": lg_ssm_small_spec,
    "lg_ssm_switch": lg_ssm_switch_spec,
    "lg_ssm_regime_switch": lg_ssm_regime_switch_spec,
    "slds": slds_spec,
    "slds_moe": slds_moe_spec,
    "slds_param_shock": slds_param_shock_spec,
    "slds_switch": slds_switch_spec,
    "seasonal_ar": seasonal_ar_spec,
    "seasonal_ar_param_shock": seasonal_ar_param_shock_spec,
    "seasonal_ar_bj": seasonal_ar_bj_spec,  # CHG 2026-10-06
    "garch": garch_spec,
    "garch_param_shock": garch_param_shock_spec,
    "ks": ks_spec,
    "ks_param_shock": ks_param_shock_spec,
    "diagonal": diagonal_spec,
    "sprott_a": sprott_a_spec,
    "aizawa": aizawa_spec,
    "thomas": thomas_spec,
    "henon_heiles": henon_heiles_spec,
    "coupled_roessler": coupled_roessler_spec,
    "lorenz84": lorenz84_spec,
    "double_pendulum": double_pendulum_spec,
    "fitzhugh_nagumo": fitzhugh_nagumo_spec,
    "hindmarsh_rose": hindmarsh_rose_spec,
    "rikitake": rikitake_spec,
}

_NAMED_INITIAL_CONDITION_PRESETS: dict[
    str,
    tuple[Callable[[], "TrajectorySpec"], tuple[float, ...]],
] = {
    **{
        f"rossler_ic_holdout_{ordinal}": (rossler_spec, initial_condition)
        for ordinal, initial_condition in enumerate(_ROSSLER_HOLDOUT_INITIAL_CONDITIONS)
    },
    **{
        f"rossler_ic_random_{ordinal}": (rossler_spec, initial_condition)
        for ordinal, initial_condition in enumerate(_ROSSLER_RANDOM_INITIAL_CONDITIONS)
    },
    **{
        f"lorenz_ic_{ordinal}": (lorenz_spec, initial_condition)
        for ordinal, initial_condition in enumerate(_LORENZ_RANDOM_INITIAL_CONDITIONS)
    },
}


def _named_initial_condition_spec(name: str) -> "TrajectorySpec":
    base_factory, initial_condition = _NAMED_INITIAL_CONDITION_PRESETS[name]
    return replace(base_factory(), x0=initial_condition)


def list_sources() -> tuple[str, ...]:
    """Return built-in synthetic source names accepted by `build_dgp_bank(name)`."""
    return tuple(sorted(_ALL_PRESETS.keys() | _NAMED_INITIAL_CONDITION_PRESETS.keys()))


def source_preset(name: str) -> Callable[[], "TrajectorySpec"]:
    """Return the built-in synthetic source factory for `name`."""
    key = name.lower().replace("-", "_")
    if key in _ALL_PRESETS:
        return _ALL_PRESETS[key]
    if key in _NAMED_INITIAL_CONDITION_PRESETS:

        def named_initial_condition_preset() -> "TrajectorySpec":
            return _named_initial_condition_spec(key)

        return named_initial_condition_preset
    valid = ", ".join(list_sources())
    raise ValueError(f"Unknown system name {name!r}. Valid names: {valid}.")
