"""
Architecture:
    TrajectorySpec — which dynamical system + base parameters
    ShockEvent    — one designed perturbation
    ShockSchedule — a stationary or time-ordered perturbation schedule
    TitrationPlan — the experimental grid (noise levels + draw counts)
    DGPBank — clean trajectories plus reusable noise realizations

Design principles:
    1. Protocol specs are frozen+slots — no field reassignment after construction
    2. Presets as classmethods — TrajectorySpec.lorenz(), TrajectorySpec.ou(), etc.
    3. Shocks compose via tuple — order = timeline, validated at construction
    4. Seed axes: trajectory root/child seeds, noise root/child seeds, model seed
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING
import numpy as np
from attrs import (
    define, 
    field,       
    )
from attrs.validators import ge, le

if TYPE_CHECKING:
    from .pipeline import DataBundle

# System parameter classes, SystemParams union, ODESystem enum → catalog.py
from .catalog import (  # noqa: E402
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
    """Specification for generating one synthetic dynamical system trajectory.
    用于生成一条合成动力学系统轨迹的规格说明。

    Defines which system to simulate (Lorenz, OU, GARCH, etc.), its parameters,
    integration method, time step, and optional interventions (shocks).
    定义要模拟的系统（Lorenz、OU、GARCH 等）、其参数、积分方法、时间步长
    以及可选的干预（冲击）。

    Use preset classmethods for common configurations:
    使用预设类方法获取常见配置：

        TrajectorySpec.lorenz()              # standard Lorenz attractor
        TrajectorySpec.lorenz_param_shock()   # Lorenz with a parameter shift at 35%
        TrajectorySpec.ou(steps=5000)         # Ornstein-Uhlenbeck process

    NOTE: Output series has shape (steps - burn_in + 1, dim).
    注意：输出序列的形状为 (steps - burn_in + 1, dim)。
    The +1 is the initial condition x(t0). For steps=35999, burn_in=0,
    Lorenz gives 36000 rows.
    +1 是初始条件 x(t0)。对于 steps=35999、burn_in=0，Lorenz 给出 36000 行。
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
        if target_name == "TRAJECTORY_SWITCH":
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
        """3D Lorenz attractor (deterministic chaos). 3维Lorenz吸引子（确定性混沌）。"""
        from .catalog import lorenz_spec
        return lorenz_spec(**kw)

    @classmethod
    def lorenz_param_shock(cls) -> "TrajectorySpec":
        """Lorenz with an abrupt parameter shift at 35%. Lorenz系统在35%处突然参数偏移。"""
        from .catalog import lorenz_param_shock_spec
        return lorenz_param_shock_spec()

    @classmethod
    def lorenz_state_shock(cls) -> "TrajectorySpec":
        """Lorenz with an additive state perturbation at 35%. Lorenz系统在35%处叠加状态扰动。"""
        from .catalog import lorenz_state_shock_spec
        return lorenz_state_shock_spec()

    @classmethod
    def lorenz_switch(cls) -> "TrajectorySpec":
        """Lorenz with a trajectory switch at 35% (different IC/params). Lorenz系统在35%处轨迹切换（不同初条件/参数）。"""
        from .catalog import lorenz_switch_spec
        return lorenz_switch_spec()

    @classmethod
    def rossler(cls, **kw) -> "TrajectorySpec":
        """3D Rössler attractor (simpler single-scroll chaos). 3维Rössler吸引子（更简单的单涡卷混沌）。"""
        from .catalog import rossler_spec
        return rossler_spec(**kw)

    @classmethod
    def rossler_param_shock(cls) -> "TrajectorySpec":
        """Rössler with an abrupt parameter shift at 35%. Rössler系统在35%处突然参数偏移。"""
        from .catalog import rossler_param_shock_spec
        return rossler_param_shock_spec()

    @classmethod
    def chua(cls, **kw) -> "TrajectorySpec":
        """3D Chua circuit double-scroll attractor. 3维蔡氏电路双涡卷吸引子。"""
        from .catalog import chua_spec
        return chua_spec(**kw)

    @classmethod
    def chua_param_shock(cls) -> "TrajectorySpec":
        """Chua with an abrupt parameter shift at 35%. 蔡氏电路在35%处突然参数偏移。"""
        from .catalog import chua_param_shock_spec
        return chua_param_shock_spec()

    @classmethod
    def chua_switch(cls) -> "TrajectorySpec":
        """Chua with a trajectory switch at 35%. 蔡氏电路在35%处轨迹切换。"""
        from .catalog import chua_switch_spec
        return chua_switch_spec()

    @classmethod
    def lorenz96(cls, **kw) -> "TrajectorySpec":
        """N-D Lorenz-96 weather model (default dim=6). N维Lorenz-96气象模型（默认dim=6）。"""
        from .catalog import lorenz96_spec
        return lorenz96_spec(**kw)

    @classmethod
    def lorenz96_switch(cls) -> "TrajectorySpec":
        """Lorenz-96 with a trajectory switch at 35%. Lorenz-96在35%处轨迹切换。"""
        from .catalog import lorenz96_switch_spec
        return lorenz96_switch_spec()

    @classmethod
    def ou(cls, **kw) -> "TrajectorySpec":
        """1D Ornstein-Uhlenbeck mean-reverting SDE (stochastic). 1维OU均值回归随机微分方程（随机）。"""
        from .catalog import ou_spec
        return ou_spec(**kw)

    @classmethod
    def ou_param_shock(cls) -> "TrajectorySpec":
        """OU with a mean-level shift at 35%. OU过程在35%处均值水平偏移。"""
        from .catalog import ou_param_shock_spec
        return ou_param_shock_spec()

    @classmethod
    def doublewell(cls, **kw) -> "TrajectorySpec":
        """1D bistable double-well SDE (stochastic). 1维双稳态双势阱随机微分方程（随机）。"""
        from .catalog import doublewell_spec
        return doublewell_spec(**kw)

    @classmethod
    def doublewell_param_shock(cls) -> "TrajectorySpec":
        """Double-well with a parameter shift at 35%. 双势阱在35%处参数偏移。"""
        from .catalog import doublewell_param_shock_spec
        return doublewell_param_shock_spec()

    @classmethod
    def doublewell_switch(cls) -> "TrajectorySpec":
        """Double-well with a trajectory switch at 35%. 双势阱在35%处轨迹切换。"""
        from .catalog import doublewell_switch_spec
        return doublewell_switch_spec()

    @classmethod
    def slds(cls, **kw) -> "TrajectorySpec":
        """1D switching linear dynamical system with 2 regimes (stochastic). 1维2体制切换线性动力系统（随机）。"""
        from .catalog import slds_spec
        return slds_spec(**kw)

    @classmethod
    def slds_param_shock(cls) -> "TrajectorySpec":
        """SLDS with a parameter shift at 35%. SLDS在35%处参数偏移。"""
        from .catalog import slds_param_shock_spec
        return slds_param_shock_spec()

    @classmethod
    def slds_switch(cls) -> "TrajectorySpec":
        """SLDS with a trajectory switch at 35%. SLDS在35%处轨迹切换。"""
        from .catalog import slds_switch_spec
        return slds_switch_spec()

    @classmethod
    def seasonal_ar(cls, **kw) -> "TrajectorySpec":
        """1D seasonal autoregressive process (stochastic). 1维季节性自回归过程（随机）。"""
        from .catalog import seasonal_ar_spec
        return seasonal_ar_spec(**kw)

    @classmethod
    def seasonal_ar_param_shock(cls) -> "TrajectorySpec":
        """Seasonal AR with a parameter shift at 35%. 季节性AR在35%处参数偏移。"""
        from .catalog import seasonal_ar_param_shock_spec
        return seasonal_ar_param_shock_spec()

    @classmethod
    def garch(cls, **kw) -> "TrajectorySpec":
        """2D GARCH(1,1) volatility model (stochastic). 2维GARCH(1,1)波动率模型（随机）。"""
        from .catalog import garch_spec
        return garch_spec(**kw)

    @classmethod
    def garch_param_shock(cls) -> "TrajectorySpec":
        """GARCH with a parameter shift at 35%. GARCH在35%处参数偏移。"""
        from .catalog import garch_param_shock_spec
        return garch_param_shock_spec()

    @classmethod
    def ks(cls, **kw) -> "TrajectorySpec":
        """Kuramoto-Sivashinsky PDE (spatiotemporal chaos, default 64 modes). KS偏微分方程（时空混沌，默认64模态）。"""
        from .catalog import ks_spec
        return ks_spec(**kw)

    @classmethod
    def ks_param_shock(cls) -> "TrajectorySpec":
        """KS with a diffusion parameter shift at 35%. KS方程在35%处扩散参数偏移。"""
        from .catalog import ks_param_shock_spec
        return ks_param_shock_spec()

    @classmethod
    def diagonal(cls, **kw) -> "TrajectorySpec":
        """Linear diagonal system dx/dt = diag(λ)x — eigenvalue recovery test. 线性对角系统dx/dt=diag(λ)x——特征值恢复测试。"""
        from .catalog import diagonal_spec
        return diagonal_spec(**kw)
     
 
class ShockType(StrEnum):
    """What kind of perturbation is applied. 施加哪种类型的扰动。"""
    ODE_PARAMS = auto()     # wholesale replacement of ALL params / 批量替换所有参数
    STATE = auto()          # additive state perturbation x[idx] += eps / 叠加状态扰动
    TRAJECTORY_SWITCH = auto()         # same-system trajectory switch (different IC/params) / 同系统轨迹切换（不同初条件/参数）


# ═══════════════════════════════════════════════════════════════════════════
# Core Specs
# ═══════════════════════════════════════════════════════════════════════════


@define(frozen=True, slots=True, kw_only=True)
class ShockEvent:
    """One designed perturbation (intervention) on the trajectory timeline.
    轨迹时间线上的一次设计性扰动（干预）。

    Applied during integration to test how a forecaster handles regime shifts,
    state perturbations, or trajectory switches.
    在积分过程中应用，用于测试预测器如何处理体制转换、状态扰动或轨迹切换。

    Times are FRACTIONS of the usable trajectory (0.0 = start, 1.0 = end),
    so shocks are independent of absolute T and burn-in settings.
    时间是可用轨迹的分数（0.0=起点，1.0=终点），因此冲击与绝对T和burn_in设置无关。

    For parameter stepped transitions:
    参数分步过渡：
      - t_start == t_end and n_inserts == 1: abrupt replacement (突然替换)
      - t_start < t_end and n_inserts == k: k scheduled updates over window (窗口内k次调度更新)

    Use presets: ShockEvent.params_step_change(), .state_perturbation(), .trajectory_switch()
    使用预设：ShockEvent.params_step_change()、.state_perturbation()、.trajectory_switch()
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

    # ── For TRAJECTORY_SWITCH target (same-system, different IC) ──
    ic_b: tuple[float, ...] | None = None
    # params_after reused for the B-trajectory params

    def __attrs_post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError(
                f"ShockEvent '{self.name}' has t_end={self.t_end} before t_start={self.t_start}"
            )

        if self.shocktype in {ShockType.STATE, ShockType.TRAJECTORY_SWITCH}:
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
    def params_step_change(
        cls, params_after: SystemParams, t_start: float, name: str = ""
    ) -> ShockEvent:
        """Parameter step change at t_start. All params change at once."""
        return cls(
            name=name or f"params_step_change_at_{t_start}",
            shocktype=ShockType.ODE_PARAMS, 
            t_start=t_start,
            t_end=t_start,
            n_inserts=1,
            params_after=params_after,
        )

    @classmethod
    def params_stepped_transition(
        cls,
        params_after: SystemParams,
        t_start: float,
        t_end: float,
        n_inserts: int,
        name: str = "",
    ) -> ShockEvent:
        """Parameter stepped transition from t_start to t_end with explicit update count."""
        return cls(
            name=name or f"params_stepped_transition_at_{t_start}_to_{t_end}",
            shocktype=ShockType.ODE_PARAMS,
            t_start=t_start,
            t_end=t_end,
            n_inserts=n_inserts,
            params_after=params_after,
        )
    
    @classmethod
    def state_perturbation(
        cls, state_idx: int, eps: float, t_start: float, name: str = ""
    ) -> ShockEvent:
        """Additive state perturbation: x[state_idx] += eps at t_start."""
        return cls(
            name=name or f"state_perturbation_idx{state_idx}_eps{eps}",
            shocktype=ShockType.STATE, 
            t_start=t_start,
            t_end=t_start,
            n_inserts=1,
            state_idx=state_idx,
            eps=eps,
        )
        
    @classmethod
    def trajectory_switch(
        cls,
        ic_b: tuple[float, ...],
        t_start: float,
        params_after: SystemParams | None = None,
        name: str = "",
    ) -> ShockEvent:
        """Same-system trajectory switch at t_start.

        Runs two independent trajectories (A with TrajectorySpec.x0/params,
        B with ic_b/params_after) and switches from A to B at t_start.
        If params_after is None, B uses the same params as A.
        """
        return cls(
            name=name or f"trajectory_switch_at_{t_start}",
            shocktype=ShockType.TRAJECTORY_SWITCH, 
            t_start=t_start,
            t_end=t_start,
            n_inserts=1,
            ic_b=ic_b,
            params_after=params_after,
        )


@define(frozen=True, slots=True, kw_only=True)
class ShockSchedule:
    """Time-ordered sequence of interventions on a trajectory. Empty = no shocks.
    轨迹上的时间排序干预序列。空 = 无冲击（静态基线）。

    Multiple parameter and state shocks are supported. Trajectory switch remains
    single-event because it cuts between independently generated trajectories
    rather than perturbing one integration in place.
    支持多个参数和状态冲击。轨迹切换保持为单一事件，因为它是在独立生成的
    轨迹之间切换，而非在原地扰动一次积分。

    Use presets: ShockSchedule.none(), .single(event), .multiple(e1, e2, ...)
    使用预设：ShockSchedule.none()、.single(event)、.multiple(e1, e2, ...)
    """

    events: tuple[ShockEvent, ...] = ()

    def __attrs_post_init__(self) -> None:
        if len(self.events) > 1 and any(e.shocktype == ShockType.TRAJECTORY_SWITCH for e in self.events):
            raise ValueError("TRAJECTORY_SWITCH shocks cannot be combined with other events")
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
    """One point in a noise-dose sweep: a specific (noise_level, trajectory_draw,
    noise_draw) combination, with full seed provenance for reproducibility.
    噪声剂量扫描中的一个点：特定的（噪声水平、轨迹抽取、噪声抽取）组合，
    附带完整的种子溯源信息以确保可复现性。

    Created by ``DGPBank.cell()`` or yielded by ``TitrationPlan.iter_cells()``.
    由 ``DGPBank.cell()`` 创建或由 ``TitrationPlan.iter_cells()`` 产出。
    """

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
    """Pre-computed clean trajectories and frozen noise arrays for one
    data-generating process (DGP). Build once, then inject noise many times.
    一个数据生成过程（DGP）的预计算干净轨迹和冻结噪声数组。构建一次，然后多次注入噪声。

    The core equation is:
    核心方程为：

        observed = clean[trajectory_draw] + sigma * noise_draws[noise_draw]

    Because the noise arrays are frozen inside the bank, every sigma level
    reuses the SAME underlying random draw (common random numbers), so
    differences across sigma levels are caused by noise dose, not luck.
    因为噪声数组在bank内部是冻结的，每个sigma水平重用相同的底层随机抽取
    （公共随机数），所以不同sigma水平之间的差异是由噪声剂量造成的，而非运气。

    Created by ``build_dgp_bank()`` or ``build_empirical_dgp_bank()``.
    Use ``inject_noise(bank, sigma=...)`` to get train/val/test splits.
    由 ``build_dgp_bank()`` 或 ``build_empirical_dgp_bank()`` 创建。
    使用 ``inject_noise(bank, sigma=...)`` 获取训练/验证/测试分割。
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
