# SERIE

**构造受控的时间序列数据，研究预测模型面对机制变化和观测噪声时的表现。**

[English README](README.md) · Rust 引擎：[**serie-rs**](https://crates.io/crates/serie-rs)（[源码](https://github.com/qilinwang/serie-rs)）

你已经训练好了一个时间序列预测模型，验证集指标也不错。接下来可以问：**输入条件变化时，它的预测表现会怎样？**

桥梁通车前要做压力测试，芯片出厂前要过可靠性测试。预测模型也一样。如果它要参与交易、调度电力、监测设备故障，单靠一个历史 benchmark 分数是不够的。我们需要知道它在什么条件下还可靠，到了哪里会开始失效。

SERIE 用来构造受控的测试数据：你可以指定合成系统、干预和观测噪声，分别查看生成的干净轨迹和加入噪声后的观测。预测、指标计算和统计结论由使用这些数据的评估负责；误差也可能包含数据本身的随机性。

合成场景体现的是我们对系统可能遭遇的变化所作的假设。机制变化和观测噪声可以单独或组合测试，但有限场景不能覆盖所有现实情况，也不能证明部署安全。评估可以回答：

> 在我关心的条件下，这个模型能承受什么，不能承受什么？别人能不能用同样的种子和协议重新验证？

## SERIE 能做什么

### 设计你真正担心的机制变化

SERIE 不只提供一个固定的冲击菜单。你可以自己声明干预：

- 哪个参数发生漂移；
- 状态是否被突然踢了一下；
- 是否从某个时刻开始拼接到另一个机制；
- 这个变化发生在序列的什么位置。

这让领域知识进入测试过程。例如，可以把某种波动率冲击、执行器故障或负荷突变写成假设场景，再报告模型在该场景和所选评估条件下的表现。

### 调高你自己的噪声

真实观测噪声很少是完美高斯噪声。传感器可能有重尾异常值，通信链路可能出现突发噪声，某些测量误差还有实际物理单位。

SERIE 允许你提供自己的噪声分布，只要它是一个能生成数组的采样函数。然后你可以用同一批冻结的噪声 realization，逐步调高 `sigma`，观察模型误差从哪里开始恶化。

对于同一个 bank、轨迹和噪声 draw，改变 `sigma` 会复用同一份噪声，减少比较中由重新抽样带来的变化。实验的其余条件、重复次数和统计分析仍需另行控制。

### 测模型，而不是和数据纠缠

在真实数据上，模型出错时很难判断原因：是模型没学到？还是数据本身就含混？还是机制已经变了？

SERIE 的合成系统，如 Lorenz、Ornstein-Uhlenbeck、GARCH 等，有明确的数据生成过程。可以分别查看生成轨迹、注入的噪声与所设干预，并在报告误差时说明这些条件；这并不意味着误差完全来自模型。

### 公平比较模型和噪声剂量

SERIE 使用固定的噪声库，也就是 simulation 里常说的 **common random numbers**。同一个 DGP bank 中的噪声 draw 会在不同 `sigma` 下重复使用。

所以：

```text
sigma = 0.1: clean + 0.1 * same_noise
sigma = 0.5: clean + 0.5 * same_noise
```

这让噪声曲线更干净，也让模型之间的比较更公平。

### 检查不确定性是不是诚实

预测模型不只输出点预测，还经常输出区间、分布或置信度。一个“很自信但错得离谱”的模型比普通错误更危险。

如果模型提供了适当的预测区间或分布，可以在独立的评估中检查：

- coverage：90% 区间是否真的大约覆盖 90% 的真值；
- PIT：其分布是否符合所用校准检验的预期；
- NLL：实际结果在预测分布下的负对数似然。

这对概率预测、风险控制和下游决策尤其重要。

### 也可以带自己的干净数据

SERIE 不是 synthetic-only。你可以把一段已经清洗好的真实记录作为 `[T, D]` 数组交给它，比如机器人传感器、步态轨迹、生产环境里的某条干净序列。

SERIE 可以把它当成经验 DGP bank：以记录作为参考，再注入受控观测噪声、切分 train/validation/test。真实记录本身可能含有测量误差，不应自动视为底层物理真值。

## 五分钟理解 SERIE

SERIE 的核心节奏是：

```text
source  ──build_dgp_bank()──▶  DGP bank  ──inject_noise()──▶  split
系统          构建一次              多次注入噪声             训练/验证/测试数据
```

三个对象：

- **source**：你要测试的系统。可以是 `"lorenz"` 这样的内置名字，也可以是你自己配置的 `TrajectorySpec`，还可以是真实数组。
- **DGP bank**：一次性构建出来的数据生成库。里面有干净轨迹、冻结噪声 draw、种子信息和协议版本。
- **split**：某个具体 `sigma` 下的 train/validation/test 切分结果。

两个动作：

- **`build_dgp_bank(...)`**：昂贵操作，只做一次。
- **`inject_noise(...)`**：便宜操作，可以对同一个 bank 反复调用不同的 `sigma`。

最小 Python 例子：

```python
from serie import PartitionSpec, build_dgp_bank, inject_noise

bank = build_dgp_bank("lorenz", seed=42)

split = inject_noise(
    bank,
    sigma=0.1,
    partition=PartitionSpec.p70_20_10(),
)

print(split.train_values.shape)
print(split.train_stats.std)
```

`split.train_stats` 只来自训练段，不会偷看 validation/test。这一点对 `sigma_scale="train_std"` 尤其重要，因为噪声按训练段的标准差缩放，不能用全序列统计量泄漏未来信息。

## 为什么要先 build，再 inject？

如果你每个 `sigma` 都重新生成一次噪声，那么误差曲线会混入随机抽样本身的波动。

SERIE 的做法是：

```python
bank = build_dgp_bank("lorenz", seed=42)

clean = inject_noise(bank, sigma=0.0)
mild = inject_noise(bank, sigma=0.1)
harsh = inject_noise(bank, sigma=0.5)
```

`mild` 和 `harsh` 复用了相同的轨迹和噪声 draw，观测数据中的区别是噪声倍率。这个配对控制了一项随机变化；要评价模型的典型表现，还需要独立重复，并考虑评估中的其他随机因素。

对于 deterministic 系统，一个干净轨迹配多组观测噪声通常就够了。对于 stochastic 系统，SERIE 默认把 trajectory draw 和 noise draw 按 ordinal 配对：`(0, 0)`, `(1, 1)`, `(2, 2)`。每一对代表一个联合未来 realization，这是做预测分布评估时更自然的复制单位。

## 安装

从 PyPI 安装：

```bash
python -m pip install serie
```

如果你使用 `uv`：

```bash
uv pip install serie
```

从源码开发：

```bash
git clone https://github.com/QilinWang/serie.git
cd serie
python -m pip install -e ".[dev,diagnostics]"
```

## 文档

详细参考文档在 [`docs/`](docs)：

- [命令行 API](docs/command_line.zh-CN.md) —— 每个 `serie` 命令、选项和示例。
- [Python API](docs/python_api.zh-CN.md) —— 可编程接口：`build_dgp_bank`、`inject_noise`、干预、种子溯源、draw design，以及稳定的导入列表。
- [系统预设](docs/system_presets.md) —— 每个内置 source 的稳定性说明和参数选择。

## 发布边界

SERIE 的 PyPI wheel 保持为数据层和轻量查看器：

- Python API；
- CLI；
- plotting；
- `serie demo` 所需的小型 `demo_dist/` viewer template。

它不打包：

- 大型已知数据集；
- 生成好的 HTML demo；
- `node_modules`；
- dashboard；
- 模型 checkpoint；
- 实验结果数据库；
- run folders。

生成的 Lorenz/Rossler/Chua HTML demo 应该作为 GitHub Release assets 或静态网站发布，而不是进入 wheel。

## 已知数据集

默认 wheel 不包含大型真实数据集。外部数据默认放在：

```bash
serie show-data-dir --format json
```

打印出来的位置，或者通过 `SERIE_KNOWN_ROOT` 指定。

这保持了包本身的干净，也避免让用户在安装一个库时被迫下载几百 MB 数据。

## Rust 引擎 (serie-rs)

SERIE 有一个 Rust 姊妹包 [**serie-rs**](https://crates.io/crates/serie-rs)
（[源码](https://github.com/qilinwang/serie-rs)）。两者是同一套协议下的姊妹实现，而不是
简单的移植：本 Python 包是面向用户、有文档的 benchmark 表面（CLI、demo、论文引用的术语
体系），而 `serie-rs` 是数据层引擎 —— 相同的阶段流水线（`load → clean → inject_noise →
partition → ready`）和相同的系统家族，用 Rust 实现。它还能读取本包写出的 grouped
`dgp_bank.safetensors`（`serie generate --array-format safetensors`），所以你可以在
Python 里构建一次 bank，再从 Rust 里消费。（`serie` 在 crates.io 上已被占用，因此用了
`-rs` 后缀。）

## 测试

从源码目录运行：

```bash
pytest
```

或用 `uv`：

```bash
uv run --no-project --with-editable ".[dev,diagnostics]" pytest
```

测试覆盖的重点不是普通 smoke test，而是 SERIE 的实验承诺：

- `sigma=0` 返回干净数据；
- 同一个 bank 下不同 `sigma` 复用同一批噪声 realization；
- train/validation/test 统计不泄漏；
- stochastic source 默认使用 paired trajectory/noise draws；
- 导出 manifest 能回读；
- CLI payload 和 Python API 保持一致。

## 引用

SERIE 的思想来自两条线：

- controlled noise titration：用可控噪声扫描模型分辨率和鲁棒性边界；
- interventionist shocks：用明确声明的机制变化测试预测模型在分布变化下的行为。

如果你在论文或项目中使用 SERIE，请同时报告 source、seed、protocol version、`sigma`、`sigma_scale`、draw design 和干预定义。这样别人才能复现你测试的具体条件。
