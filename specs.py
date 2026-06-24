"""
Architecture:
    TrajectorySpec — which dynamical system + base parameters
    ShockEvent    — one designed perturbation
    ShockSchedule — a stationary or time-ordered perturbation schedule
    TitrationPlan — the experimental grid (noise levels + draw counts)
    DGPBank — clean trajectories plus reusable noise realizations

Design principles:
    1. Protocol specs are frozen+slots — no field reassignment after construction
    2. Presets as classmethods — TrajectorySpec.lorenz_stationary(), etc.
    3. Shocks compose via tuple — order = timeline, validated at construction
    4. Seed axes: trajectory root/child seeds, noise root/child seeds, model seed
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal
import numpy as np
from attrs import (
    define, 
    field, 
    Factory, 
    evolve,       
    )
from attrs.validators import ge, gt, le, instance_of
# System parameter classes, SystemParams union, ODESystem enum → catalog.py
from .catalog import (  # noqa: E402
    LorenzParams,
    RosslerParams,
    HyperRosslerParams,
    ChuaParams,
    DuffingParams,
    VanderPolParams,
    Lorenz96Params,
    HenonParams,
    LogisticParams,
    OUParams,
    DoubleWellParams,
    SLDSParams,
    SeasonalARParams,
    GARCHParams,
    KSParams,
    DiagonalParams,
    SystemParams,
    ODESystem,
)


def spawn_draw_seeds(seed: int, n: int) -> tuple[int, ...]:
    """Spawn independent per-draw seeds from one root entropy value."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    sequence = np.random.SeedSequence(int(seed))
    return tuple(int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(n))

_ODE_METHOD_SYSTEMS = frozenset(
    {
        ODESystem.LORENZ,
        ODESystem.ROSSLER,
        ODESystem.HYPER_ROSSLER,
        ODESystem.CHUA,
        ODESystem.DUFFING,
        ODESystem.VAN_DER_POL,
        ODESystem.LORENZ96,
        ODESystem.DIAGONAL,
    }
)
_MAP_METHOD_SYSTEMS = frozenset({ODESystem.HENON, ODESystem.LOGISTIC})
_EULER_METHOD_SYSTEMS = frozenset(
    {
        ODESystem.OU,
        ODESystem.DOUBLEWELL,
        ODESystem.SLDS,
        ODESystem.SEASONAL_AR,
        ODESystem.GARCH,
    }
)

# region TrajectorySpec

@define(frozen=True, slots=True, kw_only=True)
class TrajectorySpec:
    """   
    NOTE —  load() returns series of shape (steps - burn_in + 1, dim).
        The +1 is the initial condition x(t0). This is universal across
        all system types (ODE, SDE, discrete, PDE). For steps=35999,
        burn_in=0, Lorenz gives 36000 rows.
    """ 
    system: ODESystem
    params: SystemParams
    dt: float
    steps: int
    dim: int
    burn_in: int
    method: str = "rk4"                    # integration method, set per-system in classmethods
    x0: tuple[float, ...] | None = None    # None → bridge picks default
    shocks: "ShockSchedule" = field(factory=lambda: ShockSchedule.none())

    def __attrs_post_init__(self) -> None:
        if self.system in _ODE_METHOD_SYSTEMS and self.method not in {"rk4", "euler"}:
            raise ValueError(
                f"{self.system.value} expects method 'rk4' or 'euler', got {self.method!r}"
            )
        if self.system in _MAP_METHOD_SYSTEMS and self.method != "map":
            raise ValueError(f"{self.system.value} expects method 'map', got {self.method!r}")
        if self.system in _EULER_METHOD_SYSTEMS and self.method != "euler":
            raise ValueError(f"{self.system.value} expects method 'euler', got {self.method!r}")
        if self.system == ODESystem.KS and self.method != "etdrk4":
            raise ValueError(f"{self.system.value} expects method 'etdrk4', got {self.method!r}")
        
    @property
    def T(self) -> float:
        """Total physical time = steps * dt."""
        return self.steps * self.dt

    @property
    def n_usable_steps(self) -> int: 
        return self.steps - self.burn_in 
    
    @property
    def tag(self) -> str:
        """Stable identifier including shock type (if any).
        Without this, variants like lorenz_switch / lorenz_param_shock 
        """
        base = self.system.value
        shocks = self.shocks
        if shocks is None or shocks.is_stationary or shocks.n_shocks == 0:
            return base

        if shocks.n_shocks > 1:
            return f"{base}_{shocks.n_shocks}shocks"
        event = shocks.events[0]
        target_name = event.shocktype.name
        if target_name == "SPLICE":
            return f"{base}_switch"
        if target_name == "ODE_PARAMS":
            return f"{base}_param_shock"
        if target_name == "STATE":
            return f"{base}_state_shock"
        return f"{base}_shock"
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # stage boundary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def load(
        self,
        seed: int = 0,
    ) -> "DataBundle":
        """Load one clean trajectory into a DataBundle.

        NOTE: Output series has shape (steps - burn_in + 1, dim).
        The +1 is the initial condition at t=0. For steps=1000, burn_in=0,
        you get 1001 rows, not 1000.

        Factory default for development/presets. DGPBank handles
        ensemble expansion (multiple seeds, noise layers) on top.
        """
        from .pipeline import DataBundle, StageLoadSynthetic
        db = DataBundle()
        return db | StageLoadSynthetic(
            trajectory_spec=self,
            seed=seed,
        )

    # ══════════════════════════════════════════════════════════════════
    # Presets — thin forwarding to catalog.py (real definitions there)
    # ══════════════════════════════════════════════════════════════════

    @classmethod
    def lorenz(cls, **kw) -> "TrajectorySpec":
        from .catalog import lorenz_spec
        return lorenz_spec(**kw)

    @classmethod
    def lorenz_param_shock(cls) -> "TrajectorySpec":
        from .catalog import lorenz_param_shock_spec
        return lorenz_param_shock_spec()

    @classmethod
    def lorenz_state_shock(cls) -> "TrajectorySpec":
        from .catalog import lorenz_state_shock_spec
        return lorenz_state_shock_spec()

    @classmethod
    def lorenz_switch(cls) -> "TrajectorySpec":
        from .catalog import lorenz_switch_spec
        return lorenz_switch_spec()

    @classmethod
    def rossler(cls, **kw) -> "TrajectorySpec":
        from .catalog import rossler_spec
        return rossler_spec(**kw)

    @classmethod
    def rossler_param_shock(cls) -> "TrajectorySpec":
        from .catalog import rossler_param_shock_spec
        return rossler_param_shock_spec()

    @classmethod
    def chua(cls, **kw) -> "TrajectorySpec":
        from .catalog import chua_spec
        return chua_spec(**kw)

    @classmethod
    def chua_param_shock(cls) -> "TrajectorySpec":
        from .catalog import chua_param_shock_spec
        return chua_param_shock_spec()

    @classmethod
    def chua_switch(cls) -> "TrajectorySpec":
        from .catalog import chua_switch_spec
        return chua_switch_spec()

    @classmethod
    def lorenz96(cls, **kw) -> "TrajectorySpec":
        from .catalog import lorenz96_spec
        return lorenz96_spec(**kw)

    @classmethod
    def lorenz96_switch(cls) -> "TrajectorySpec":
        from .catalog import lorenz96_switch_spec
        return lorenz96_switch_spec()

    @classmethod
    def ou(cls, **kw) -> "TrajectorySpec":
        from .catalog import ou_spec
        return ou_spec(**kw)

    @classmethod
    def ou_param_shock(cls) -> "TrajectorySpec":
        from .catalog import ou_param_shock_spec
        return ou_param_shock_spec()

    @classmethod
    def doublewell(cls, **kw) -> "TrajectorySpec":
        from .catalog import doublewell_spec
        return doublewell_spec(**kw)

    @classmethod
    def doublewell_param_shock(cls) -> "TrajectorySpec":
        from .catalog import doublewell_param_shock_spec
        return doublewell_param_shock_spec()

    @classmethod
    def doublewell_switch(cls) -> "TrajectorySpec":
        from .catalog import doublewell_switch_spec
        return doublewell_switch_spec()

    @classmethod
    def slds(cls, **kw) -> "TrajectorySpec":
        from .catalog import slds_spec
        return slds_spec(**kw)

    @classmethod
    def slds_param_shock(cls) -> "TrajectorySpec":
        from .catalog import slds_param_shock_spec
        return slds_param_shock_spec()

    @classmethod
    def slds_switch(cls) -> "TrajectorySpec":
        from .catalog import slds_switch_spec
        return slds_switch_spec()

    @classmethod
    def seasonal_ar(cls, **kw) -> "TrajectorySpec":
        from .catalog import seasonal_ar_spec
        return seasonal_ar_spec(**kw)

    @classmethod
    def seasonal_ar_param_shock(cls) -> "TrajectorySpec":
        from .catalog import seasonal_ar_param_shock_spec
        return seasonal_ar_param_shock_spec()

    @classmethod
    def garch(cls, **kw) -> "TrajectorySpec":
        from .catalog import garch_spec
        return garch_spec(**kw)

    @classmethod
    def garch_param_shock(cls) -> "TrajectorySpec":
        from .catalog import garch_param_shock_spec
        return garch_param_shock_spec()

    @classmethod
    def ks(cls, **kw) -> "TrajectorySpec":
        from .catalog import ks_spec
        return ks_spec(**kw)

    @classmethod
    def ks_param_shock(cls) -> "TrajectorySpec":
        from .catalog import ks_param_shock_spec
        return ks_param_shock_spec()

    @classmethod
    def diagonal(cls, **kw) -> "TrajectorySpec":
        from .catalog import diagonal_spec
        return diagonal_spec(**kw)
     
 
class ShockType(StrEnum):
    """What gets perturbed."""
    ODE_PARAMS = auto()     # wholesale replacement of ALL params
    STATE = auto()          # additive state perturbation x[idx] += eps
    SPLICE = auto()         # same-system trajectory splice (different IC/params)


# ═══════════════════════════════════════════════════════════════════════════
# Core Specs
# ═══════════════════════════════════════════════════════════════════════════


@define(frozen=True, slots=True, kw_only=True)
class ShockEvent:
    """A single designed perturbation on the timeline. 
    Consumer: SyntheticDataGenerator (applies shock during integration). 
    Times are in FRACTION of usable T (0.0 = start of usable data, 1.0 = end).
    This makes shocks independent of absolute T and transient settings.

    For parameter transitions:
      - t_start == t_end and n_inserts == 1 means one abrupt replacement
      - t_start < t_end and n_inserts == k means k scheduled updates over the window
    """

    name: str  # human-readable: "rho_transition_28_to_35"
    shocktype: ShockType 

    # ── Timing ──
    t_start: float = field(validator=[ge(0.0), le(1.0)])  # type: ignore
    t_end: float = field(validator=[ge(0.0), le(1.0)])  # type: ignore
    n_inserts: int = field(default=1, validator=ge(1))  # type: ignore

    # ── For ODE_PARAMS target (wholesale replacement) ──
    params_after: SystemParams | None = None

    # ── For STATE target ──
    state_idx: int = 0
    eps: float = 0.0

    # ── For SPLICE target (same-system, different IC) ──
    ic_b: tuple[float, ...] | None = None
    # params_after reused for splice (B-trajectory params)

    def __attrs_post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError(
                f"ShockEvent '{self.name}' has t_end={self.t_end} before t_start={self.t_start}"
            )

        if self.shocktype in {ShockType.STATE, ShockType.SPLICE}:
            if self.t_end != self.t_start:
                raise ValueError(
                    f"{self.shocktype.name} shock '{self.name}' must be instantaneous: "
                    f"expected t_end == t_start, got {self.t_end} vs {self.t_start}"
                )
            if self.n_inserts != 1:
                raise ValueError(
                    f"{self.shocktype.name} shock '{self.name}' must use n_inserts=1, got {self.n_inserts}"
                )

        if self.shocktype == ShockType.ODE_PARAMS and self.t_end == self.t_start and self.n_inserts != 1:
            raise ValueError(
                f"ODE_PARAMS shock '{self.name}' uses a zero-width transition window, "
                f"so n_inserts must be 1, got {self.n_inserts}"
            )

    @property
    def is_instantaneous(self) -> bool:
        return self.t_end == self.t_start and self.n_inserts == 1

    # ── Presets ──────────────────────────────────────────────────────
    @classmethod
    def params_replace(
        cls, params_after: SystemParams, t_start: float, name: str = ""
    ) -> ShockEvent:
        """Wholesale parameter replacement at t_start. All params change at once."""
        return cls(
            name=name or f"params_replace_at_{t_start}",
            shocktype=ShockType.ODE_PARAMS, 
            t_start=t_start,
            t_end=t_start,
            n_inserts=1,
            params_after=params_after,
        )

    @classmethod
    def params_transition(
        cls,
        params_after: SystemParams,
        t_start: float,
        t_end: float,
        n_inserts: int,
        name: str = "",
    ) -> ShockEvent:
        """Parameter transition from t_start to t_end with explicit update count."""
        return cls(
            name=name or f"params_transition_at_{t_start}_to_{t_end}",
            shocktype=ShockType.ODE_PARAMS,
            t_start=t_start,
            t_end=t_end,
            n_inserts=n_inserts,
            params_after=params_after,
        )
    
    @classmethod
    def state_kick(
        cls, state_idx: int, eps: float, t_start: float, name: str = ""
    ) -> ShockEvent:
        """Additive state perturbation: x[state_idx] += eps at t_start."""
        return cls(
            name=name or f"state_kick_idx{state_idx}_eps{eps}",
            shocktype=ShockType.STATE, 
            t_start=t_start,
            t_end=t_start,
            n_inserts=1,
            state_idx=state_idx,
            eps=eps,
        )
        
    @classmethod
    def trajectory_splice(
        cls,
        ic_b: tuple[float, ...],
        t_start: float,
        params_after: SystemParams | None = None,
        name: str = "",
    ) -> ShockEvent:
        """Same-system trajectory splice at t_start.

        Runs two independent trajectories (A with TrajectorySpec.x0/params,
        B with ic_b/params_after) and splices at t_start.
        If params_after is None, B uses the same params as A.
        """
        return cls(
            name=name or f"splice_at_{t_start}",
            shocktype=ShockType.SPLICE, 
            t_start=t_start,
            t_end=t_start,
            n_inserts=1,
            ic_b=ic_b,
            params_after=params_after,
        )


@define(frozen=True, slots=True, kw_only=True)
class ShockSchedule:
    """Time-ordered shock schedule. Empty means stationary baseline.

    Multiple ODE parameter and state shocks are supported. Splice remains a
    single-event schedule because it cuts between independently generated
    trajectories rather than perturbing one integration in place.
    """

    events: tuple[ShockEvent, ...] = ()

    def __attrs_post_init__(self) -> None:
        if len(self.events) > 1 and any(e.shocktype == ShockType.SPLICE for e in self.events):
            raise ValueError("SPLICE shocks cannot be combined with other events")
        for i in range(len(self.events) - 1):
            if self.events[i].t_start > self.events[i + 1].t_start:
                raise ValueError(
                    f"ShockSchedule: events must be time-ordered. "
                    f"Event '{self.events[i].name}' at t={self.events[i].t_start} "
                    f"comes before '{self.events[i + 1].name}' at t={self.events[i + 1].t_start}"
                )

    @property
    def n_shocks(self) -> int:
        return len(self.events)

    @property
    def is_stationary(self) -> bool:
        return len(self.events) == 0

    @property
    def shock_times(self) -> tuple[float, ...]:
        """All shock start times (for plotting / annotation)."""
        return tuple(e.t_start for e in self.events)

    # ── Presets ──────────────────────────────────────────────────────

    @classmethod
    def none(cls) -> ShockSchedule:
        return cls()

    @classmethod
    def single(cls, event: ShockEvent) -> ShockSchedule:
        return cls(events=(event,))

    @classmethod
    def multiple(cls, *events: ShockEvent) -> ShockSchedule:
        return cls(events=events)

@define(frozen=True, slots=True, kw_only=True)
class TitrationCell:
    """Identity of one cell in a titration grid. For logging and DB registration."""

    source_name: str
    noise_level: float        # noise amplitude for this cell
    trajectory_seed_base: int        # root entropy for trajectory draw spawning
    noise_seed_base: int      # root entropy for noise draw spawning
    trajectory_draw_seed: int         # child seed used to generate this trajectory
    noise_draw_seed: int       # child seed used to generate this noise draw
    trajectory_draw_ordinal: int       # spawn ordinal and array slot (0..n-1)
    noise_draw_ordinal: int            # spawn ordinal and array slot (0..n-1)
    T: int
    D: int

    @property
    def tag(self) -> str:
        level = f"{self.noise_level:g}"  # compact: 0.1, 0.001, 1e-05
        return f"{self.source_name}_traj{self.trajectory_draw_ordinal}_sig{level}_noise{self.noise_draw_ordinal}"

# ═══════════════════════════════════════════════════════════════════════════
# DGPBank — factored clean trajectories plus reusable noise draws
# ═══════════════════════════════════════════════════════════════════════════

@define(frozen=True, slots=True, kw_only=True)
class DGPBank:
    """Frozen draw bank for one data-generating process.

    noisy = clean[trajectory_draw_ordinal] + noise_level * noise_draws[noise_draw_ordinal]

    `TitrationPlan` decides which `(trajectory_draw_ordinal, noise_draw_ordinal)` pairs are benchmark
    cells: deterministic sources normally use one clean path with many
    noise draws; stochastic sources normally pair trajectory and noise
    draws by spawn ordinal.

    Built-in noise draws are unit-variance by convention. Custom draws may
    carry any known scale; in that case `noise_level` is a raw multiplier.
    """
    name: str
    trajectory_seed_base: int
    noise_seed_base: int
    trajectory_draw_seeds: tuple[int, ...]
    noise_draw_seeds: tuple[int, ...]
    clean: np.ndarray                    # [N_trajectory, T, D] — frozen trajectory draws
    noise_draws: np.ndarray              # [N_noise, T, D] — frozen noise draws
    feature_names: tuple[str, ...]
    # True for stochastic generators whose trajectory draws are distinct
    # future trajectory draws, not repeated deterministic clean paths.
    is_stochastic: bool = False
    # Which noise law the draws follow: "gaussian", "laplace", or "custom".
    # Provenance for downstream statistical tests — the protocol requires the
    # injected law to be exactly known, not necessarily Gaussian.
    noise_kind: str = "gaussian"

    def __attrs_post_init__(self) -> None:
        if self.clean.ndim != 3:
            raise ValueError(f"clean must have shape [N_trajectory, T, D], got {self.clean.shape}")
        if self.noise_draws.ndim != 3:
            raise ValueError(
                "noise_draws must have shape [N_noise, T, D], "
                f"got {self.noise_draws.shape}"
            )
        if self.clean.shape[1:] != self.noise_draws.shape[1:]:
            raise ValueError(
                "clean and noise_draws must agree on [T, D], "
                f"got {self.clean.shape[1:]} vs {self.noise_draws.shape[1:]}"
            )
        if len(self.trajectory_draw_seeds) != self.clean.shape[0]:
            raise ValueError(
                "trajectory_draw_seeds length "
                f"{len(self.trajectory_draw_seeds)} must match N_trajectory={self.clean.shape[0]}"
            )
        if len(self.noise_draw_seeds) != self.noise_draws.shape[0]:
            raise ValueError(
                "noise_draw_seeds length "
                f"{len(self.noise_draw_seeds)} must match "
                f"N_noise={self.noise_draws.shape[0]}"
            )
        expected_trajectory_seeds = spawn_draw_seeds(
            self.trajectory_seed_base,
            len(self.trajectory_draw_seeds),
        )
        if self.trajectory_draw_seeds != expected_trajectory_seeds:
            raise ValueError(
                "trajectory_draw_seeds must equal spawn_draw_seeds("
                "trajectory_seed_base, n_trajectory_draws); "
                "the draw ordinal is part of the reproducible seed recipe"
            )
        expected_noise_seeds = spawn_draw_seeds(
            self.noise_seed_base,
            len(self.noise_draw_seeds),
        )
        if self.noise_draw_seeds != expected_noise_seeds:
            raise ValueError(
                "noise_draw_seeds must equal spawn_draw_seeds("
                "noise_seed_base, n_noise_draws); "
                "the draw ordinal is part of the reproducible seed recipe"
            )

    def _check_cell(self, trajectory_draw_ordinal: int, noise_draw_ordinal: int) -> None:
        if not 0 <= trajectory_draw_ordinal < self.n_trajectory_draws:
            raise IndexError(f"trajectory_draw_ordinal={trajectory_draw_ordinal} outside [0, {self.n_trajectory_draws})")
        if not 0 <= noise_draw_ordinal < self.n_noise_draws:
            raise IndexError(f"noise_draw_ordinal={noise_draw_ordinal} outside [0, {self.n_noise_draws})")

    # ── Core noise injection ──
     
    def cell(self, noise_level: float, trajectory_draw_ordinal: int = 0, noise_draw_ordinal: int = 0) -> TitrationCell:
        """Return the provenance identity for one physical titration cell."""
        self._check_cell(trajectory_draw_ordinal, noise_draw_ordinal)
        return TitrationCell(
            source_name=self.name,
            noise_level=noise_level,
            trajectory_seed_base=self.trajectory_seed_base,
            noise_seed_base=self.noise_seed_base,
            trajectory_draw_seed=self.trajectory_draw_seeds[trajectory_draw_ordinal],
            noise_draw_seed=self.noise_draw_seeds[noise_draw_ordinal],
            trajectory_draw_ordinal=trajectory_draw_ordinal,
            noise_draw_ordinal=noise_draw_ordinal,
            T=self.T,
            D=self.D,
        )
    
    def build_observed_series(
        self,
        noise_level: float,
        trajectory_draw_ordinal: int = 0,
        noise_draw_ordinal: int = 0,
    ) -> np.ndarray:
        """Return one observed `[T, D]` series for a cell.

        DGP-bank path:
            clean + noise_level * noise_draw
        """
        self._check_cell(trajectory_draw_ordinal, noise_draw_ordinal)
        if noise_level == 0.0:
            return self.clean[trajectory_draw_ordinal]
        return self.clean[trajectory_draw_ordinal] + noise_level * self.noise_draws[noise_draw_ordinal]
    # ── Shape accessors ──

    @property
    def T(self) -> int:
        return self.clean.shape[1]

    @property
    def D(self) -> int:
        return self.clean.shape[2]

    @property
    def n_trajectory_draws(self) -> int:
        return self.clean.shape[0]

    @property
    def n_noise_draws(self) -> int:
        return self.noise_draws.shape[0]
