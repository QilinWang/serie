[← 返回 README](../README.zh-CN.md)

## Python API

### 构建一次，多次注入

```python
from serie import build_dgp_bank, inject_noise

bank = build_dgp_bank("lorenz", seed=42, n_noise_draws=8)

for sigma in [0.0, 0.1, 0.25, 0.5]:
    split = inject_noise(bank, sigma=sigma)
    print(sigma, split.values.shape)
```

### 使用真实干净数据

```python
from serie import build_empirical_dgp_bank, inject_noise

bank = build_empirical_dgp_bank(values, name="wearable_gait")
split = inject_noise(bank, sigma=1.0, sigma_scale="train_std")
```

`values` 应该是 `[T, D]` 形状的干净数组。SERIE 会在它上面构建冻结噪声 draw，并按你指定的 `sigma` 生成观测序列。

### 自定义噪声分布

```python
def laplace_noise(rng, shape):
    return rng.laplace(0.0, 1.0 / 2**0.5, size=shape)

bank = build_dgp_bank("lorenz", noise_kind=laplace_noise)
split = inject_noise(bank, sigma=0.2)
```

SERIE 内置噪声默认是 unit variance，这样 `sigma` 可以直接理解为噪声标准差倍率。自定义噪声不强制必须 unit variance；真正要求是你知道并声明噪声 law。只是如果你想公平比较不同 law 的“形状”而不是“能量”，unit variance 是最清楚的约定。

### 干预

```python
from attrs import evolve
from serie import ShockEvent, ShockSchedule, TrajectorySpec, build_dgp_bank

base = TrajectorySpec.lorenz()
source = evolve(
    base,
    shocks=ShockSchedule.single(
        ShockEvent.params_step_change(
            params_after=evolve(base.params, rho=35.0),
            t_start=0.60,
        )
    ),
)
bank = build_dgp_bank(source, seed=42)
```

SERIE 支持多个按时间排序的参数干预和状态干预；例如可以先做
`params_stepped_transition(...)`，再做 `state_perturbation(...)`。`trajectory_switch(...)`
仍然保持单事件，因为 trajectory switch 是在两条独立生成的轨迹之间切换，不是在同一条积分轨迹里原地扰动。

### 种子和 draw ordinal

SERIE 使用 `np.random.SeedSequence(...).spawn(...)` 生成独立 child seeds。不要把 `seed + i` 当成独立随机流。

需要区分两件事：

- `*_seed_base`：你传入的根种子，用来重新生成整个 draw bank；
- `*_draw_ordinal`：第几个 spawn/draw，也就是重建某个 draw 所需的 ordinal；
- `*_draw_seed`：由根种子和 ordinal 派生出的 child seed，方便审计和对照。

因此 `noise_draw_ordinal=3` 的意思不是“随机数种子是 3”，而是“使用这个 bank 中第 3 个冻结噪声 realization”。它对应的真实 child seed 会记录在 metadata 里。

