# System Presets — Stability Notes and Parameter Choices

Every system in `_ALL_PRESETS` has a default configuration that is verified to
produce finite, bounded trajectories at the preset's `steps` and `dt`. This
document records the investigation behind each choice so future contributors
know what was tested and what breaks.

DGP = Data Generating Process (数据生成过程).

## Methodology note

Stability checks ("Verified: all finite") were done by running standalone
Python simulations and checking `np.all(np.isfinite(...))` on the output.
Quantitative gap measurements (SLDS section) used standalone simulations with
a single seed (seed=42) — they are **point estimates without error bars**.
None of these numbers have been validated through `serie`'s actual pipeline
(`build_dgp_bank` → `inject_noise`). The qualitative conclusions (which
presets are stable, which regime configurations have large gaps) are likely
correct, but specific percentages should be treated as approximate.

## Quick Reference

| Preset name | Dim | Steps | dt | T (s) | Category | Bounded? |
|-------------|-----|-------|----|-------|----------|----------|
| `lorenz` | 3 | 35999 | 0.01 | 360 | ODE (chaos) | Yes — strange attractor |
| `rossler` | 3 | 35999 | 0.01 | 360 | ODE (chaos) | Yes — single-scroll |
| `hyper_rossler` | 4 | 15000 | 0.001 | 15 | ODE (hyperchaos) | **Transient** — escapes at T≈19s |
| `chua` | 3 | 35999 | 0.005 | 180 | ODE (chaos) | Yes — double-scroll circuit |
| `duffing` | 2 | 35999 | 0.01 | 360 | ODE (forced osc.) | Yes — hardening spring (β=1) |
| `vanderpol` | 2 | 35999 | 0.01 | 360 | ODE (limit cycle) | Yes — self-sustained oscillation |
| `lorenz96` | 6 | 55000 | 0.007 | 385 | ODE (weather toy) | Yes |
| `henon` | 2 | 35999 | — | 35999 | Map (chaos) | Yes — \|x\| < 1.3 |
| `logistic` | 1 | 35999 | — | 35999 | Map (chaos) | Yes — x ∈ [0, 1] |
| `ou` | 1 | 25000 | 0.5 | 12500 | SDE (mean-rev.) | Yes — stationary |
| `doublewell` | 1 | 25000 | 0.5 | 12500 | SDE (bistable) | Yes — potential wells |
| `slds` | 1 | 25000 | 1.0 | 25000 | Discrete (switching) | Yes |
| `seasonal_ar` | 1 | 25000 | 1.0 | 25000 | Discrete (seasonal) | Yes |
| `garch` | 2 | 25000 | 1.0 | 25000 | Discrete (volatility) | Yes — stationary when α+β<1 |
| `ks` | 64 | 25000 | 0.01 | 250 | PDE (spatiotemporal) | Yes |
| `diagonal` | 3 | 10000 | 0.01 | 100 | ODE (linear) | Yes — exponential decay |

---

## Per-System Investigation Notes

### Lorenz (3D)

Standard parameters: σ=10, ρ=28, β=8/3. Canonical chaotic attractor.
No stability issues at any practical T or dt. The butterfly attractor is
globally bounded.

Verified: `steps=35999, dt=0.01` → 36000 rows, all finite, max\|x\| ≈ 48.

### Rössler (3D)

Standard parameters: a=0.2, b=0.2, c=5.7. Single-scroll attractor.
The key stabilizing term is `dz/dt = b + z(x - c)`: when `c` is much larger
than `x`, the `z(x-c)` term acts as a restoring force. This is why Rössler
is bounded but Hyper-Rössler (which uses `x*z` without the `-c` term) is not.

Verified: `steps=35999, dt=0.01` → all finite.

### Hyper-Rössler (4D) ⚠️

Standard parameters: a=0.25, b=3.0, c=0.5, d=0.05. Two positive Lyapunov
exponents (hyperchaotic).

**This is a transient attractor.** The system escapes to infinity around
T≈19s regardless of dt or parameter tuning. The root cause is the `dz/dt =
b + x*z` term — unlike the 3D Rössler's `z*(x-c)`, there is no restoring
`-c*z` term, so `x` and `z` feed each other without bound.

Investigation results (dt=0.001):

| Steps | T (s) | max\|state\| | Status |
|-------|-------|-------------|--------|
| 10000 | 10 | 2,821 | OK |
| 15000 | 15 | 203,651 | OK (preset choice) |
| 18000 | 18 | 2,679,101 | OK but growing fast |
| 19423 | 19.4 | ∞ | Blows up |

Parameter sweeps tried (all blow up at the same physical time):
- `d=-0.05` (damped w): still blows up at step ~20k
- `d=0.005` (weaker w growth): still blows up at step ~20k
- `b=0.2` (smaller z offset): still blows up at step ~22k
- `c=2.0, c=5.0` (stronger w restoring): blows up *faster*
- `dt=0.0005` (finer stepping): blows up at step ~42k (same T≈21s)

**Conclusion:** the blowup is physical (the attractor IS transient), not
numerical. The preset uses `steps=15000, dt=0.001` (T=15s) with a 4s safety
margin. This gives 15001 rows of genuinely hyperchaotic data.

Chaos verified: nearby ICs (Δw=0.01) diverge by 50 at T=11s.

### Chua (3D)

Standard parameters: α=15.6, β=28, m1=-8/7, m0=-5/7. Double-scroll circuit.
Bounded by the piecewise-linear nonlinearity. Uses smaller dt=0.005 than
Lorenz because the double-scroll dynamics are stiffer.

Verified: `steps=35999, dt=0.005` → all finite.

### Duffing (2D) ⚠️

**The default preset uses the hardening spring (β=1), NOT the classic
softening spring (β=-1).**

Investigation:

| β | Steps | dt | Status |
|---|-------|----|--------|
| -1 (softening) | 2966 | 0.002 | **Blows up** — x³ term diverges |
| -1 (softening) | 596 | 0.01 | **Blows up** faster with larger dt |
| +1 (hardening) | 35000 | 0.01 | Stable, bounded, max\|x\| = 0.7 |

The softening spring (β < 0) is the classic chaotic Duffing, but it has
unbounded orbits: the cubic term `-βx³ = +|β|x³` pushes x away from the
origin. This diverges in finite time for the standard parameters.

The hardening spring (β > 0) produces a bounded quasiperiodic forced
oscillator. Not chaotic, but useful for testing forecasters on periodic-ish
signals with known forcing frequency.

**If you want chaotic Duffing**, use very short runs:
```python
TrajectorySpec.duffing(beta=-1.0, gamma=0.65, omega=1.2, delta=0.2,
                       steps=2000, dt=0.002)
```

### Van der Pol (2D)

Standard parameters: μ=2.0. Self-sustained limit-cycle oscillator.
Always bounded — the nonlinear damping `μ(1-x²)y` dissipates energy when
x > 1 and amplifies when x < 1, creating a stable limit cycle.

Verified: `steps=35999, dt=0.01` → all finite, max\|x\| = 3.8.

Uses `burn_in=1000` to skip the transient approach to the limit cycle.

### Lorenz96 (6D)

Standard parameters: F=8 (chaotic regime). N-dimensional weather toy model
on a periodic lattice. Always bounded for F ≈ 8 because the quadratic
advection term conserves energy.

### Hénon (2D map)

Standard parameters: a=1.4, b=0.3. Classic strange attractor.
Bounded: all points stay within \|x\| < 1.3, \|y\| < 0.4.

Verified: `steps=35999` → all finite, max\|x\| = 1.3.

Uses `burn_in=500` to skip the transient.

### Logistic (1D map)

Standard parameters: r=3.9 (full chaos). Period-doubling route to chaos.
Always bounded: x ∈ [0, 1] for x₀ ∈ (0, 1) when r ∈ [0, 4].

Verified: `steps=35999` → all finite, all values in [0, 1].

Uses `burn_in=500` to skip the transient.

### OU (1D SDE)

Standard parameters: θ=0.2, μ=0.0, σ_noise=0.3. Mean-reverting.
Stationary distribution: N(μ, σ²/(2θ)). Always bounded in expectation.

Euler-Maruyama discretization: AR(1) with a = 1-θdt, b = σ√dt.
Stationary variance verified against the closed-form formula in tests.

### DoubleWell (1D SDE)

Potential U(x) = x⁴/4 - (a/2)x². Minima at x = ±√a.
Noise-induced transitions between wells. Bounded by the quartic potential.

### SLDS (1D discrete)

2-regime Markov-switching AR(1). Bounded when \|A₁\|,\|A₂\| < 1.

SLDS 是一个切换线性动态系统。在每个时间步，一个隐藏的马尔可夫链选择两个
AR(1) 中的哪一个生成下一个观测值。参数 p11/p22 控制停留概率（在当前状态下
停留的概率），决定了平均停留时间 dwell = 1/(1-p)。

The data generation process at each step:
```
regime[t] ~ Markov(transition_matrix)     # hidden state picks which AR
x[t+1]   = A[regime[t]] * x[t]           # autoregressive coefficient for that regime
         + sqrt(Q[regime[t]]) * noise[t]  # noise variance for that regime
```

**Default preset** (`slds`): A1=0.9, Q1=0.05, A2=0.98, Q2=0.35, p11=0.94,
p22=0.95 (dwell ≈ 17/20 steps). The two regimes are close enough that a
single AR(1) model captures 98% of the variance — the oracle-vs-global MSE
gap is only 1.7%. This preset tests basic stochastic forecasting, not
regime detection.

**MOE preset** (`slds_moe`): A1=0.3, Q1=0.01, A2=0.99, Q2=1.0, p11=0.95,
p22=0.95. The two regimes are maximally distinct:
- Regime 1 (A=0.3, Q=0.01): fast mean-reversion, quiet — signal decays
  quickly, low noise, so this regime produces small, stable values.
- Regime 2 (A=0.99, Q=1.0): near-unit-root, volatile — signal barely
  decays, high noise, so this regime produces large, wandering values.

#### Measurement methodology

The "oracle-vs-global gap" measures how much a model gains from knowing the
true regime at each timestep. This quantifies the *value of regime detection*
for the SLDS preset.

**How the gap is computed:**

1. Generate a trajectory of N=25000 steps from the SLDS with known regime
   labels (the hidden Markov chain is part of the simulation, so ground-truth
   labels are available).

2. **Global model**: fit a single AR(1) coefficient `a_global` and variance
   `q_global` to the entire sequence, ignoring regime labels. One-step-ahead
   prediction: `x̂[t+1] = a_global * x[t]`. Compute MSE over all steps.

3. **Oracle model**: fit separate AR(1) coefficients `a_1, a_2` and variances
   `q_1, q_2` to the steps belonging to each regime (using ground-truth
   labels). One-step-ahead prediction: `x̂[t+1] = a[regime[t]] * x[t]`.
   Compute MSE over all steps.

4. **Gap** = `1 - oracle_mse / global_mse`. A gap of 32% means the oracle
   has 32% lower MSE than the global model.

**What the oracle represents:** The oracle is NOT a realistic model — no
real forecaster has access to ground-truth regime labels. It is an upper
bound on how much regime-aware modeling (MOE, HMM, gating network) could
help. If the oracle gap is 2%, there is no point building a regime-detection
model. If it is 32%, there is substantial room for a regime-aware model to
improve over a single-regime baseline.

实际中没有模型能达到 oracle 的性能，因为真实模型必须从观测值推断状态，
不可能完美。oracle gap 是一个上界，回答的问题是"值不值得做状态检测"。

**Limitations of these numbers:**

- **Single seed, no error bars.** Each gap percentage is a point estimate
  from one run of 25000 steps with seed=42. No variance across seeds was
  computed. The true gap could differ by ±3-5 percentage points.
- **One-step-ahead only.** The gap is measured at prediction horizon h=1.
  At longer horizons, the gap could be larger (regime identity matters more
  for multi-step forecasts) or smaller (both models revert toward the
  stationary mean).
- **Standalone simulation.** The numbers were computed from a standalone
  Python SLDS simulation, not validated through `serie`'s actual integrator
  pipeline (`build_dgp_bank` → `inject_noise`). The integrator should
  produce equivalent results, but this has not been verified.
- **Fitted AR(1) only.** The global model is a simple linear fit. A more
  flexible single-regime model (e.g., RNN, transformer) might narrow the
  gap without explicit regime detection.

#### Table 1: Gap vs switching speed

Fixed regimes (A1=0.3, Q1=0.01, A2=0.99, Q2=1.0), varying transition
probability p (where p11=p22=p):

| p (p11=p22) | Dwell (steps) | Gap | Interpretation |
|-------------|--------------|-----|----------------|
| 0.5 | 2 | ~19% | Switching every ~2 steps. Too fast to detect — by the time you identify regime 1, you're already in regime 2. The process looks like a single noisy mixture. |
| 0.8 | 5 | ~29% | Moderate switching. Some regime structure is visible but hard to exploit in real-time. |
| 0.9 | 10 | ~31% | Regimes last ~10 steps. Enough to detect and adapt, but the model has limited data per regime stretch. |
| **0.95** | **20** | **~32%** | **Chosen for preset.** Each regime lasts ~20 steps — long enough that a gating network or HMM can detect the switch and adapt within a few steps, short enough that a single AR(1) cannot locally track the dynamics. |
| 0.99 | 100 | ~19% | Regimes last ~100 steps. A single adaptive model (or one with long context) can effectively track the local dynamics without needing explicit regime detection — it has time to "settle in." |
| 0.995 | 200 | ~15% | Very slow switching. Each stretch is so long that even a simple model adapts. The switching itself is rare enough to be a minor perturbation. |

注意：所有百分比都是单次运行的点估计值（seed=42，N=25000），未报告置信区间。
真实值可能有 ±3-5 个百分点的偏差。

**Pattern:** the gap peaks at intermediate dwell times. Too fast → mixture
noise hides the structure. Too slow → a single model can locally adapt.
The sweet spot is where regime structure is learnable but not trivially
trackable.

#### Table 2: Gap vs regime separation

Fixed switching speed (p11=p22=0.95, dwell≈20), varying how different the
two regimes are:

| A1 | Q1 | A2 | Q2 | Gap | Interpretation |
|----|-----|-----|-----|-----|----------------|
| 0.9 | 0.05 | 0.98 | 0.35 | ~2% | **Default preset.** Both regimes are slow-decaying and quiet. A single AR(1) with a≈0.94 fits both almost equally well. No regime detection needed. |
| 0.5 | 0.05 | 0.98 | 0.35 | ~18% | Regime 1 now decays faster (A=0.5 vs 0.9), creating a moderate gap. |
| 0.3 | 0.01 | 0.99 | 1.0 | ~32% | **MOE preset.** Regime 1 is fast/quiet (A=0.3, Q=0.01), regime 2 is slow/loud (A=0.99, Q=1.0). A single AR(1) must compromise between two very different dynamics. |
| 0.0 | 0.01 | 0.99 | 2.0 | ~47% | Maximally distinct: regime 1 is iid noise (A=0), regime 2 is near-random-walk with high variance. Not chosen for preset because A1=0.0 makes regime 1 degenerate (no temporal structure to learn). |

**Why A1=0.3, not 0.0:** At A1=0.0, regime 1 is just iid white noise — there
is no autoregressive structure for a model to learn. At A1=0.3, regime 1 is
still a recognizable mean-reverting process with short memory (≈3 steps to
decay to 3% of initial value). This makes regime 1 a meaningful forecasting
target, not just noise.

**Why Q2=1.0, not 2.0:** Q2=2.0 produces a 47% gap but regime 2 dominates
the signal variance so heavily that the task becomes "detect high-variance
periods" rather than "detect regime-specific dynamics." Q2=1.0 keeps the
two regimes distinguishable without trivializing the detection problem.

### Seasonal AR (1D discrete)

AR(1) with sinusoidal forcing. Bounded when \|φ\| < 1.

### GARCH (2D discrete)

Returns + conditional variance. Stationary when α + β < 1.
Default: ω=0.01, α=0.06, β=0.90 → α+β = 0.96 < 1.

### KS (64D PDE)

Kuramoto-Sivashinsky. Integrated with ETDRK4 (Kassam & Trefethen 2005).
Spatiotemporal chaos on a periodic domain. The 2/3 dealiasing rule prevents
energy buildup at high wavenumbers.

### Diagonal (3D linear)

dx/dt = diag(λ)x. Ground-truth test for eigenvalue recovery.
Exact solution: x_i(t) = x_i(0) exp(λ_i t). Default eigenvalues are all
negative → exponential decay → always bounded.

---

## Systems NOT in `_ALL_PRESETS`

These exist in the `ODESystem` enum and `SYSTEMS` registry but do not have
a preset function. They are either not yet implemented or require
user-supplied parameters:

- `LINEAR`, `JORDAN_BLOCK`, `ROTATION_2D` — designed eigenvalue-recovery
  tests, not yet wired into the integrator
- `THOMAS`, `CHEN`, `LOTKA_VOLTERRA` — referenced in the enum for future use
