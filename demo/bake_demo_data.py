"""Bake SERIE attractor data for the static web demo.

The browser demo is intentionally not a SERIE runner. This script calls SERIE
once, writes plain JSON, and the frontend only renders:

    noisy = clean + sigma * draw

Output contract:
    public/attractor_data.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from serie import TrajectorySpec, build_dgp_bank

DemoSystem = Literal["lorenz", "rossler", "chua"]


def _demo_spec(system: str) -> TrajectorySpec:
    if system == "lorenz":
        return TrajectorySpec.lorenz()
    if system == "rossler":
        return TrajectorySpec.rossler()
    if system == "chua":
        return TrajectorySpec.chua()
    raise ValueError("demo system must be one of: lorenz, rossler, chua")


def bake(
    system: DemoSystem = "lorenz",
    *,
    seed: int = 42,
    n_points: int = 4000,
    n_draws: int = 30,
    out_path: str | Path | None = None,
) -> dict[str, object]:
    """Generate one clean attractor and a frozen unit-noise draw bank."""
    spec = _demo_spec(system)
    bank = build_dgp_bank(spec, seed=seed, n_noise_draws=n_draws)

    clean = np.asarray(bank.clean[0], dtype=np.float32)
    draws = np.asarray(bank.noise_draws, dtype=np.float32)

    if clean.shape[1] < 3:
        raise ValueError(f"{system!r} has D={clean.shape[1]}; the demo requires a 3D source")

    if clean.shape[0] > n_points:
        idx = np.linspace(0, clean.shape[0] - 1, n_points).astype(np.int64)
        clean = clean[idx]
        draws = draws[:, idx, :]

    clean = np.ascontiguousarray(clean[:, :3], dtype=np.float32)
    draws = np.ascontiguousarray(draws[:, :, :3], dtype=np.float32)

    payload: dict[str, object] = {
        "system": system,
        "seed": int(seed),
        "feature_names": list(bank.feature_names[:3]),
        "dt": float(spec.dt),
        "n_points": int(clean.shape[0]),
        "clean": clean.tolist(),
        "draws": draws.tolist(),
    }

    output = (
        Path(out_path)
        if out_path is not None
        else Path(__file__).resolve().parent / "public" / "attractor_data.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"baked {clean.shape[0]} points x {draws.shape[0]} draws -> {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("lorenz", "rossler", "chua"), default="lorenz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-points", type=int, default=4000)
    parser.add_argument("--n-draws", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    bake(
        args.system,
        seed=args.seed,
        n_points=args.n_points,
        n_draws=args.n_draws,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
