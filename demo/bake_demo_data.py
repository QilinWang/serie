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
import numpy as np

from serie import build_dgp_bank
from serie.catalog import source_preset

DEMO_SHAPE_SYSTEMS = ("lorenz", "rossler", "chua")
DEMO_CANDIDATE_SYSTEMS = (*DEMO_SHAPE_SYSTEMS, "garch")


def _project_arrays(
    clean: np.ndarray,
    draws: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    T, D = clean.shape
    if D >= 3:
        return (
            np.ascontiguousarray(clean[:, :3], dtype=np.float32),
            np.ascontiguousarray(draws[:, :, :3], dtype=np.float32),
            list(feature_names[:3]),
            "first_three_channels" if D > 3 else "native_3d",
        )
    if D == 2:
        zeros = np.zeros((T, 1), dtype=np.float32)
        draw_zeros = np.zeros((draws.shape[0], T, 1), dtype=np.float32)
        return (
            np.ascontiguousarray(np.concatenate([clean, zeros], axis=1), dtype=np.float32),
            np.ascontiguousarray(np.concatenate([draws, draw_zeros], axis=2), dtype=np.float32),
            [*feature_names, "zero"],
            "2d_padded_z0",
        )
    if D == 1:
        time_axis = np.linspace(-1.0, 1.0, T, dtype=np.float32)[:, None]
        zeros = np.zeros((T, 1), dtype=np.float32)
        draw_zeros = np.zeros((draws.shape[0], T, 1), dtype=np.float32)
        return (
            np.ascontiguousarray(
                np.concatenate([time_axis, clean, zeros], axis=1), dtype=np.float32
            ),
            np.ascontiguousarray(
                np.concatenate([draw_zeros, draws, draw_zeros], axis=2), dtype=np.float32
            ),
            ["time", feature_names[0], "zero"],
            "1d_time_value_z0",
        )
    raise ValueError(f"demo system has invalid D={D}")


def bake(
    system: str = "shapes",
    *,
    seed: int = 42,
    n_points: int = 4000,
    n_draws: int = 30,
    out_path: str | Path | None = None,
) -> dict[str, object]:
    """Generate clean attractors and frozen unit-noise draw banks."""
    systems = DEMO_SHAPE_SYSTEMS if system == "shapes" else (system,)
    payload_systems = [
        _bake_one(system_name, seed=seed, n_points=n_points, n_draws=n_draws)
        for system_name in systems
    ]
    payload: dict[str, object]
    if len(payload_systems) == 1:
        payload = payload_systems[0]
    else:
        payload = {
            "kind": "serie_demo_bundle",
            "default_system": payload_systems[0]["system"],
            "systems": payload_systems,
        }

    output = (
        Path(out_path)
        if out_path is not None
        else Path(__file__).resolve().parent / "public" / "attractor_data.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    names = ", ".join(str(item["system"]) for item in payload_systems)
    print(f"baked {names} -> {output}")
    return payload


def _bake_one(
    system: str,
    *,
    seed: int,
    n_points: int,
    n_draws: int,
) -> dict[str, object]:
    """Generate one clean attractor and a frozen unit-noise draw bank."""
    if system not in DEMO_CANDIDATE_SYSTEMS:
        valid = ", ".join(("shapes", *DEMO_CANDIDATE_SYSTEMS))
        raise ValueError(f"demo system must be one of: {valid}")

    spec = source_preset(system)()
    bank = build_dgp_bank(spec, seed=seed, n_noise_draws=n_draws)

    clean = np.asarray(bank.clean[0], dtype=np.float32)
    draws = np.asarray(bank.noise_draws, dtype=np.float32)

    if clean.shape[0] > n_points:
        idx = np.linspace(0, clean.shape[0] - 1, n_points).astype(np.int64)
        clean = clean[idx]
        draws = draws[:, idx, :]

    clean, draws, feature_names, projection = _project_arrays(clean, draws, bank.feature_names)

    return {
        "system": system,
        "seed": int(seed),
        "source_dim": int(bank.D),
        "projection": projection,
        "feature_names": feature_names,
        "dt": float(spec.dt),
        "n_points": int(clean.shape[0]),
        "shape": {
            "clean": [int(clean.shape[0]), int(clean.shape[1])],
            "draws": [int(draws.shape[0]), int(draws.shape[1]), int(draws.shape[2])],
        },
        "clean": clean.tolist(),
        "draws": draws.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("shapes", *DEMO_CANDIDATE_SYSTEMS), default="shapes")
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
