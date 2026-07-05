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

from enum import StrEnum, auto
from typing import Callable, Literal, TYPE_CHECKING

import attr
import numpy as np
from attrs import AttrsInstance, define, evolve

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
        ODESystem.GARCH,
    }
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lorenz
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, kw_only=True)
class LorenzParams:
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

    return evolve(
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

    return evolve(
        lorenz_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.state_perturbation(state_idx=0, eps=0.9, t_start=0.35)
        ),
    )


def lorenz_switch_spec():
    from .specs import ShockSchedule, ShockEvent

    return evolve(
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


@define(frozen=True, kw_only=True)
class RosslerParams:
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

    return evolve(
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


@define(frozen=True, kw_only=True)
class HyperRosslerParams:
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


@define(frozen=True, kw_only=True)
class ChuaParams:
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

    return evolve(
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

    return evolve(
        chua_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.trajectory_switch(ic_b=(0.11, 0.01, 0.02), t_start=0.35)
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Duffing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, kw_only=True)
class DuffingParams:
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


@define(frozen=True, kw_only=True)
class VanderPolParams:
    """
    Van der Pol oscillator (1920).

    Self-sustained oscillator with nonlinear damping.

      dx/dt = y
      dy/dt = μ(1 - x²)y - x

    μ > 0 controls nonlinearity strength. Large μ → relaxation oscillations.
    """

    mu: float  # Nonlinearity parameter


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


@define(frozen=True, kw_only=True)
class Lorenz96Params:
    """
    Lorenz-96 model (1996).

    Simplified atmospheric model on a 1D periodic lattice.

      dx_i/dt = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F

    Variable dimension (typically 40). Chaos when F ≥ 8.
    """

    forcing: float  # External forcing F


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

    return evolve(
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


@define(frozen=True, kw_only=True)
class HenonParams:
    """
    Hénon map (1976).

    2D discrete-time chaotic map.


      x_{n+1} = 1 - ax_n² + y_n
      y_{n+1} = bx_n

    Standard chaotic regime: a=1.4, b=0.3
    """

    a: float  # Nonlinear coefficient
    b: float  # Linear coefficient (contraction)


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


@define(frozen=True, kw_only=True)
class LogisticParams:
    """
    Logistic map.

    1D discrete-time map exhibiting period-doubling route to chaos.


      x_{n+1} = rx_n(1 - x_n)

    Chaos when r ∈ [3.57, 4.0]. Full chaos at r ≈ 3.9.
    """

    r: float  # Growth rate


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


@define(frozen=True, kw_only=True)
class OUParams:
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

    return evolve(
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


@define(frozen=True, kw_only=True)
class DoubleWellParams:
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

    return evolve(
        doublewell_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=DoubleWellParams(a=1.0, sigma_noise=0.35),
                t_start=0.35,
            )
        ),
    )


def doublewell_switch_spec():
    from .specs import ShockSchedule, ShockEvent

    return evolve(
        doublewell_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.trajectory_switch(
                ic_b=(0.0,),
                params_after=DoubleWellParams(a=1.0, sigma_noise=0.35),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, kw_only=True)
class SLDSParams:
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
        dt=1.0, steps=steps, burn_in=0, dim=1,
        method="euler",
        x0=(0.0,),
    )


def slds_param_shock_spec():
    from .specs import ShockSchedule, ShockEvent

    return evolve(
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

    return evolve(
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


@define(frozen=True, kw_only=True)
class SeasonalARParams:
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

    return evolve(
        seasonal_ar_spec(),
        shocks=ShockSchedule.single(
            ShockEvent.params_step_change(
                params_after=SeasonalARParams(S=24, phi=0.8, sigma=0.35, a0=1.4, amp_drift=0e-4),
                t_start=0.35,
            )
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GARCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, kw_only=True)
class GARCHParams:
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

    return evolve(
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


@define(frozen=True, kw_only=True)
class LGSSMParams:
    """
    LG-SSM: continuous-time linear Gaussian SDE sampled at a fixed interval dt
    by EXACT discretization (no Euler stepping).

    Roadmap: docs/new_titration/controlled_model_families_roadmap.md
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
        raise ValueError(
            f"kappa_v is a condition number and must be >= 1, got {params.kappa_v}."
        )
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@define(frozen=True, kw_only=True)
class KSParams:
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

    return evolve(
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


@define(frozen=True, slots=True)
class DiagonalParams:
    """dx/dt = diag(eigenvalues) · x — trivial eigenvalue recovery test.
    Ground truth: learned discrete eigenvalues should be exp(λ_i · dt)."""

    eigenvalues: tuple[float, ...] = (-0.5, -1.0, -2.0)


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
    | GARCHParams
    | KSParams
    | DiagonalParams
)


# ═══════════════════════════════════════════════════════════════════════════
# System registry
# ═══════════════════════════════════════════════════════════════════════════


@attr.define(frozen=True)
class SystemSpec:
    """Specification for a dynamical system (structure only, no defaults)."""

    name: str
    dim: int  # -1 for variable dimension
    param_type: type[AttrsInstance]  # AttrsInstance type for params
    kind: Literal["ode", "sde", "map", "pde", "discrete"]
    time_dependent: bool
    f_py: Callable  # Python vector field


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
}


# lg_ssm is a continuous-time SDE, but its generator is an exact stepwise
# recurrence (A, Q from the Van Loan discretization), so it dispatches with
# the discrete statistical simulators rather than the Euler-Maruyama path.
DISCRETE_SYSTEMS = {"slds", "seasonal_ar", "garch", "lg_ssm"}

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
    "doublewell_param_shock": doublewell_param_shock_spec,
    "doublewell_switch": doublewell_switch_spec,
    "lg_ssm": lg_ssm_spec,
    "lg_ssm_small": lg_ssm_small_spec,
    "slds": slds_spec,
    "slds_moe": slds_moe_spec,
    "slds_param_shock": slds_param_shock_spec,
    "slds_switch": slds_switch_spec,
    "seasonal_ar": seasonal_ar_spec,
    "seasonal_ar_param_shock": seasonal_ar_param_shock_spec,
    "garch": garch_spec,
    "garch_param_shock": garch_param_shock_spec,
    "ks": ks_spec,
    "ks_param_shock": ks_param_shock_spec,
    "diagonal": diagonal_spec,
}


def list_sources() -> tuple[str, ...]:
    """Return built-in synthetic source names accepted by `build_dgp_bank(name)`."""
    return tuple(sorted(_ALL_PRESETS))


def source_preset(name: str) -> Callable[[], "TrajectorySpec"]:
    """Return the built-in synthetic source factory for `name`."""
    key = name.lower().replace("-", "_")
    if key not in _ALL_PRESETS:
        valid = ", ".join(list_sources())
        raise ValueError(f"Unknown system name {name!r}. Valid names: {valid}.")
    return _ALL_PRESETS[key]
