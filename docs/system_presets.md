# System Presets — Stability Notes and Parameter Choices

Every system in `_ALL_PRESETS` has a default configuration that is verified to
produce finite, bounded trajectories at the preset's `steps` and `dt`. This
document records the investigation behind each choice so future contributors
know what was tested and what breaks.

DGP = Data Generating Process (数据生成过程).

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
