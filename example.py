"""One-file SERIE example — start here if you're new.
SERIE 单文件示例——新用户请从这里开始。

Run from the study checkout:
从源码检出运行：

    python serie/example.py

Run after installing the package:
安装包后运行：

    python -m serie.example
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    # Let `python serie/example.py` work from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serie import (  # noqa: E402
    DrawDesign,
    PartitionSpec,
    TitrationPlan,
    TrajectorySpec,
    build_dgp_bank,
    inject_noise,
    iter_cells,
)


def describe_split(label, split) -> None:
    """Print the parts a model usually consumes."""
    print(f"\n{label}")
    print(f"  source: {split.source_id}")
    print(f"  train/val/test: {split.train_values.shape} / {split.val_values.shape} / {split.test_values.shape}")
    print(f"  sigma: {split.metadata['sigma']} ({split.metadata['sigma_scale']})")
    print(
        "  draw ordinals: "
        f"trajectory={split.metadata['trajectory_draw_ordinal']}, "
        f"noise={split.metadata['noise_draw_ordinal']}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: The simplest path — build once, inject noise many times
# 示例 1：最简单的路径——构建一次，多次注入噪声
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def noise_sweep_example() -> None:
    """Sweep noise levels on a Lorenz attractor — the core SERIE workflow.
    在 Lorenz 吸引子上扫描噪声水平——SERIE 的核心工作流。"""
    print("== noise sweep on Lorenz ==")

    # Build the DGP (Data Generating Process) bank ONCE.
    # 构建 DGP（数据生成过程）bank，只需一次。
    bank = build_dgp_bank("lorenz", seed=42)

    # Sweep noise levels. Every sigma reuses the SAME frozen noise draw
    # (common random numbers), so differences are caused by dose, not luck.
    # 扫描噪声水平。每个 sigma 重用相同的冻结噪声抽取（公共随机数），
    # 因此差异是由剂量引起的，而非运气。
    partition = PartitionSpec.p70_20_10()
    for sigma in [0.0, 0.1, 0.25, 0.5, 1.0]:
        split = inject_noise(bank, sigma=sigma, partition=partition)
        train_std = split.train_stats.std
        print(f"  sigma={sigma:<5}  train shape={split.train_values.shape}  "
              f"train std (ch0)={train_std[0]:.3f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Deterministic source with multiple noise draws
# 示例 2：确定性数据源，多个噪声抽取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def deterministic_lorenz_example() -> None:
    """Deterministic system: one clean trajectory, many observation-noise draws.
    确定性系统：一条干净轨迹，多个观测噪声抽取。"""
    print("\n== deterministic Lorenz with multiple noise draws ==")
    spec = TrajectorySpec.lorenz(steps=1999)
    bank = build_dgp_bank(
        spec,
        seed=42,
        noise_seed_base=3000,
        n_trajectory_draws=1,
        n_noise_draws=3,
    )
    partition = PartitionSpec.p70_20_10()

    clean = inject_noise(bank, sigma=0.0, partition=partition, noise_draw_ordinal=0)
    noisy = inject_noise(bank, sigma=0.25, partition=partition, noise_draw_ordinal=1)

    describe_split("clean split", clean)
    describe_split("same trajectory, different frozen noise draw", noisy)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3 (advanced): Stochastic source with paired draws
# 示例 3（进阶）：随机数据源，配对抽取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stochastic_ou_example() -> None:
    """Stochastic system: trajectory and observation-noise draws are paired.
    随机系统：轨迹和观测噪声抽取是配对的。

    For stochastic sources (OU, SLDS, GARCH, etc.), each trajectory draw IS
    a different future — so trajectory draw r pairs with noise draw r.
    This is the statistically correct design for predictive-distribution tests.
    对于随机数据源（OU、SLDS、GARCH 等），每个轨迹抽取是一个不同的未来——
    因此轨迹抽取 r 与噪声抽取 r 配对。这是预测分布检验的统计学正确设计。
    """
    print("\n== stochastic OU with paired draws ==")
    spec = TrajectorySpec.ou(steps=999)
    bank = build_dgp_bank(
        spec,
        seed=11,
        noise_seed_base=4000,
        n_trajectory_draws=3,
        n_noise_draws=3,
    )
    partition = PartitionSpec.p70_20_10()
    plan = TitrationPlan(
        noise_levels=(0.0, 0.5),
        n_trajectory_draws=3,
        n_noise_draws=3,
        draw_design=DrawDesign.AUTO,
    )

    print("cells selected by AUTO for a stochastic source:")
    for cell in iter_cells(bank, plan):
        print(
            "  "
            f"sigma={cell.noise_level:<3} "
            f"trajectory_ordinal={cell.trajectory_draw_ordinal} "
            f"noise_ordinal={cell.noise_draw_ordinal}"
        )

    cell = next(iter_cells(bank, plan))
    split = inject_noise(
        bank,
        sigma=cell.noise_level,
        partition=partition,
        trajectory_draw_ordinal=cell.trajectory_draw_ordinal,
        noise_draw_ordinal=cell.noise_draw_ordinal,
    )
    describe_split("first paired stochastic cell", split)


def main() -> None:
    noise_sweep_example()
    deterministic_lorenz_example()
    stochastic_ou_example()


if __name__ == "__main__":
    main()
