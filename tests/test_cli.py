"""CLI integration tests — verify every subcommand produces parseable output.

These tests call ``cli.main([...])`` directly (no subprocess) and parse stdout.
They are the contract between the CLI surface and agents/scripts that consume
SERIE's JSON output.  If a JSON key disappears or a command silently changes
its output shape, these tests catch it.

Coverage:
    discovery  — list-cli-commands, list-data-ids, show-data-dir
    generate   — metadata-only, with arrays+plots, with shock, schema stability,
                 train_std artifact rejection
    walkthrough — full end-to-end audit with file existence checks

Each test is named ``test_cli_<subcommand>_<what it checks>``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from serie import cli


def _generated_artifact_dir(payload) -> Path:
    return Path(payload["artifacts"]["out_dir"])


def test_cli_commands_json_is_agent_readable(capsys):
    cli.main(["list-cli-commands", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    commands = {row["command"]: row for row in payload["commands"]}
    assert "serie --help" == payload["help"]["all_commands"]
    assert "serie <command> --help" == payload["help"]["one_command"]
    assert "list-cli-commands" in commands
    assert "list-data-ids" in commands
    assert "show-data-dir" in commands
    assert commands["generate"]["group"] == "02 generate"
    assert "sigma" in commands["generate"]["example"]
    assert "shock" in commands["generate"]["example"]
    assert commands["demo"]["group"] == "03 visualize"
    assert "inspect" not in commands
    assert "inject-noise" not in commands
    assert "export" not in commands
    assert "plot" not in commands


def test_cli_commands_text_groups_commands(capsys):
    cli.main(["list-cli-commands"])
    out = capsys.readouterr().out
    assert "[01 discovery]" in out
    assert "list-cli-commands: List SERIE CLI commands with examples." in out
    assert "[02 generate]" in out
    assert "generate: Generate one SERIE cell" in out
    assert "Help:" in out


def test_cli_list_json_is_agent_readable(capsys):
    cli.main(["list-data-ids", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert "lorenz" in payload["synthetic"]
    assert "lorenz_param_shock" in payload["synthetic"]
    assert "vanderpol" in payload["synthetic"]
    assert "henon" in payload["synthetic"]
    assert "logistic" in payload["synthetic"]
    assert "ou" in payload["synthetic"]
    assert "etth1" in payload["known"]


def test_cli_data_dir_uses_environment(monkeypatch, tmp_path, capsys):
    root = tmp_path / "known"
    monkeypatch.setenv("SERIE_KNOWN_ROOT", str(root))

    cli.main(["show-data-dir", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"data_dir": str(root), "exists": False}


def test_cli_rejects_unknown_source():
    with pytest.raises(SystemExit, match="Unknown system name"):
        cli.main(["generate", "not_a_source"])


def test_cli_generate_train_std_json_is_parseable(capsys):
    cli.main(
        [
            "generate",
            "ou",
            "--sigma",
            "1.0",
            "--sigma-scale",
            "train_std",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["sigma"] == 1.0
    assert payload["sigma_scale"] == "train_std"
    assert payload["shape"]["train"]


def test_cli_generate_json_schema_is_stable(capsys):
    cli.main(["generate", "ou", "--sigma", "0.1", "--array-format", "none"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"] == "ou"
    assert payload["shock"] == "none"
    assert payload["sigma"] == 0.1
    assert payload["cell_id"].startswith("ou_")
    assert len(payload["cell_hash"]) == 16
    assert payload["artifacts"] is None
    assert sorted(payload) == [
        "artifacts",
        "cell_hash",
        "cell_id",
        "cell_tag",
        "dgp_bank",
        "name",
        "noise_kind",
        "partition",
        "shape",
        "shock",
        "sigma",
        "sigma_scale",
        "source",
        "train_stats",
    ]
    assert sorted(payload["dgp_bank"]) == [
        "D",
        "T",
        "feature_names",
        "is_stochastic",
        "n_noise_draws",
        "n_trajectory_draws",
        "name",
        "noise_draw_seeds",
        "noise_kind",
        "noise_seed_base",
        "trajectory_draw_seeds",
        "trajectory_seed_base",
    ]
    assert payload["dgp_bank"]["feature_names"] == ["f0"]
    assert payload["dgp_bank"]["n_trajectory_draws"] == 1
    assert payload["dgp_bank"]["n_noise_draws"] == 1
    assert sorted(payload["shape"]) == ["full", "test", "train", "val"]
    assert sorted(payload["train_stats"]) == ["mean_head", "std_head"]
    assert payload["shape"]["full"] == [25001, 1]
    assert payload["shape"]["train"] == [17500, 1]
    assert payload["shape"]["val"] == [5000, 1]
    assert payload["shape"]["test"] == [2501, 1]


def test_cli_generate_writes_arrays_and_plot_for_shock(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "generated"
    cli.main(
        [
            "generate",
            "lorenz",
            "--shock",
            "param",
            "--sigma",
            "0.1",
            "--out",
            str(out),
            "--array-format",
            "csv",
            "--plot",
            "timeseries",
            "--max-points",
            "200",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    artifact_dir = _generated_artifact_dir(payload)
    assert payload["source"] == "lorenz_param_shock"
    assert payload["name"] == "lorenz_param_shock"
    assert payload["shock"] == "param"
    assert payload["artifacts"]["requested_out_dir"] == str(out)
    assert artifact_dir.parent == out
    assert (artifact_dir / "serie_cell_identity.json").exists()
    assert payload["artifacts"]["arrays"]["array_format"] == "csv"
    assert payload["artifacts"]["plots"]["timeseries"]
    assert (artifact_dir / "clean.csv").exists()
    assert (artifact_dir / "noise_draws.csv").exists()
    assert (artifact_dir / "manifest.json").exists()
    assert (artifact_dir / "lorenz_param_shock_timeseries.png").exists()


def test_cli_generate_rejects_train_std_artifact_writes(tmp_path):
    with pytest.raises(
        SystemExit, match="File exports and plots currently require --sigma-scale unit"
    ):
        cli.main(
            [
                "generate",
                "ou",
                "--sigma",
                "1.0",
                "--sigma-scale",
                "train_std",
                "--out",
                str(tmp_path / "cell"),
                "--array-format",
                "csv",
            ]
        )


@pytest.mark.parametrize(
    ("source", "projection"),
    [
        ("henon", "2d_padded_z0"),
        ("logistic", "1d_time_value_z0"),
    ],
)
def test_cli_demo_writes_self_contained_html_and_optional_json(
    source, projection, tmp_path, capsys
):
    out = tmp_path / f"{source}.html"
    data_out = tmp_path / f"{source}.json"
    cli.main(
        [
            "demo",
            source,
            "--out-html",
            str(out),
            "--data-out",
            str(data_out),
            "--n-points",
            "80",
            "--n-draws",
            "1",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == source
    assert payload["out"] == str(out)
    assert payload["data_out"] == str(data_out)
    assert out.exists()
    assert data_out.exists()
    assert "window.__SERIE_DEMO_DATA__" in out.read_text(encoding="utf-8")

    data = json.loads(data_out.read_text(encoding="utf-8"))
    assert data["system"] == source
    assert data["projection"] == projection
    assert data["n_points"] == 80
    assert len(data["clean"]) == 80
    assert len(data["draws"]) == 1
    assert data["shape"] == {"clean": [80, 3], "draws": [1, 80, 3]}


def test_cli_demo_shapes_writes_dropdown_bundle(tmp_path, capsys):
    out = tmp_path / "shapes.html"
    data_out = tmp_path / "shapes.json"
    cli.main(
        [
            "demo",
            "shapes",
            "--out-html",
            str(out),
            "--data-out",
            str(data_out),
            "--n-points",
            "80",
            "--n-draws",
            "1",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "lorenz"
    assert payload["sources"] == ["lorenz", "rossler", "chua"]
    assert payload["shapes"] == {
        "lorenz": {"clean": [80, 3], "draws": [1, 80, 3]},
        "rossler": {"clean": [80, 3], "draws": [1, 80, 3]},
        "chua": {"clean": [80, 3], "draws": [1, 80, 3]},
    }
    assert out.exists()
    assert data_out.exists()
    assert "window.__SERIE_DEMO_DATA__" in out.read_text(encoding="utf-8")

    data = json.loads(data_out.read_text(encoding="utf-8"))
    assert data["kind"] == "serie_demo_bundle"
    assert data["default_system"] == "lorenz"
    assert [item["system"] for item in data["systems"]] == ["lorenz", "rossler", "chua"]
    assert all(item["projection"] == "native_3d" for item in data["systems"])


def test_cli_demo_converts_local_csv_to_viewer_json(tmp_path, capsys):
    csv_path = tmp_path / "recording.csv"
    csv_path.write_text(
        "t,x,y\n0,0.0,1.0\n1,0.5,0.7\n2,1.0,0.2\n3,0.5,-0.4\n4,0.0,-0.8\n",
        encoding="utf-8",
    )
    out = tmp_path / "recording.html"
    data_out = tmp_path / "recording.json"
    cli.main(
        [
            "demo",
            "--input",
            str(csv_path),
            "--input-name",
            "wearable_gait",
            "--out-html",
            str(out),
            "--data-out",
            str(data_out),
            "--n-points",
            "5",
            "--n-draws",
            "1",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "wearable_gait"
    assert out.exists()
    assert data_out.exists()

    data = json.loads(data_out.read_text(encoding="utf-8"))
    assert data["system"] == "wearable_gait"
    assert data["source_dim"] == 2
    assert data["projection"] == "2d_padded_z0"
    assert data["feature_names"] == ["x", "y", "zero"]


def test_cli_demo_out_dir_writes_html_ipc_arrays_and_static_plots(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out_dir = tmp_path / "henon_demo"
    cli.main(
        [
            "demo",
            "henon",
            "--out-dir",
            str(out_dir),
            "--n-points",
            "80",
            "--n-draws",
            "1",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["out_dir"] == str(out_dir)
    assert payload["html"] == str(out_dir / "index.html")
    assert payload["data"] is None
    assert payload["data_out"] is None
    assert payload["arrays"]["array_format"] == "ipc"
    assert sorted(payload["plots"]) == ["attractor", "timeseries"]
    assert (out_dir / "index.html").exists()
    assert not (out_dir / "demo_data.json").exists()
    assert (out_dir / "arrays" / "clean.ipc").exists()
    assert (out_dir / "arrays" / "noise_draws.ipc").exists()
    assert (out_dir / "arrays" / "manifest.json").exists()
    assert (out_dir / "static" / "henon_timeseries.png").exists()
    assert (out_dir / "static" / "henon_attractor.png").exists()


def test_cli_generate_writes_all_array_formats_and_manifest(tmp_path, capsys):
    out = tmp_path / "export"
    cli.main(
        [
            "generate",
            "ou",
            "--out",
            str(out),
            "--sigma",
            "0.25",
            "--array-format",
            "all",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    artifact_dir = _generated_artifact_dir(payload)
    assert payload["source"] == "ou"
    assert payload["artifacts"]["arrays"]["array_format"] == "all"
    assert payload["artifacts"]["arrays"]["n_noisy"] == 1
    assert artifact_dir.parent == out
    assert (artifact_dir / "serie_cell_identity.json").exists()
    assert (artifact_dir / "clean.npy").exists()
    assert (artifact_dir / "clean.csv").exists()
    assert (artifact_dir / "clean.ipc").exists()
    assert (artifact_dir / "clean.parquet").exists()
    assert (artifact_dir / "dgp_bank.safetensors").exists()
    assert (artifact_dir / "noise_draws.npy").exists()
    assert (artifact_dir / "noise_draws.csv").exists()
    assert (artifact_dir / "noise_draws.ipc").exists()
    assert (artifact_dir / "noise_draws.parquet").exists()
    assert (artifact_dir / "manifest.json").exists()


def test_cli_generate_out_is_identity_named_and_idempotent(tmp_path, capsys):
    out = tmp_path / "export"
    args = [
        "generate",
        "ou",
        "--out",
        str(out),
        "--sigma",
        "0.25",
        "--array-format",
        "csv",
    ]

    cli.main(args)
    first = json.loads(capsys.readouterr().out)
    cli.main(args)
    second = json.loads(capsys.readouterr().out)

    assert first["cell_id"] == second["cell_id"]
    assert first["cell_hash"] == second["cell_hash"]
    assert first["artifacts"]["out_dir"] == second["artifacts"]["out_dir"]
    artifact_dir = _generated_artifact_dir(second)
    assert artifact_dir.parent == out
    assert (artifact_dir / "serie_cell_identity.json").exists()
    assert (artifact_dir / "clean.csv").exists()


def test_cli_generate_plot_both_writes_timeseries_and_attractor(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "cell"
    cli.main(
        [
            "generate",
            "lorenz",
            "--out",
            str(out),
            "--plot",
            "both",
            "--sigma",
            "0.1",
            "--max-points",
            "200",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    artifact_dir = _generated_artifact_dir(payload)
    assert payload["source"] == "lorenz"
    assert sorted(payload["artifacts"]["plots"]) == ["attractor", "timeseries"]
    assert (artifact_dir / "lorenz_timeseries.png").exists()
    assert (artifact_dir / "lorenz_attractor.png").exists()


def test_cli_generate_plot_intervention_source_writes_timeseries(
    tmp_path, capsys
):
    pytest.importorskip("matplotlib")
    out = tmp_path / "cell"
    cli.main(
        [
            "generate",
            "lorenz",
            "--shock",
            "param",
            "--out",
            str(out),
            "--plot",
            "timeseries",
            "--sigma",
            "0.1",
            "--max-points",
            "200",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    artifact_dir = _generated_artifact_dir(payload)
    assert payload["source"] == "lorenz_param_shock"
    assert sorted(payload["artifacts"]["plots"]) == ["timeseries"]
    assert (artifact_dir / "lorenz_param_shock_timeseries.png").exists()


@pytest.mark.parametrize(("shock", "has_reference"), [("none", False), ("param", True)])
def test_cli_plot_routes_unshocked_reference_only_for_interventions(
    shock, has_reference, tmp_path, capsys, monkeypatch
):
    import msgspec
    import numpy as np
    from serie import build_empirical_dgp_bank, plots

    class SyntheticSchedule(msgspec.Struct):
        is_stationary: bool

    class SyntheticSource(msgspec.Struct):
        tag: str
        shocks: SyntheticSchedule
        shape_label: str

    def resolve(source, *, shock):
        return "synthetic", SyntheticSource(
            tag="synthetic", shocks=SyntheticSchedule(shock == "none"), shape_label="three_channels"
        )

    values = np.arange(60, dtype=np.float32).reshape(20, 3)
    observed_bank = build_empirical_dgp_bank(values, name="observed")
    reference_bank = build_empirical_dgp_bank(values + 1, name="reference")
    calls = []
    seen = []

    def build(source, **kwargs):
        calls.append((source, kwargs))
        return observed_bank if len(calls) == 1 else reference_bank

    def plot(bank, *, reference_bank, out, **kwargs):
        seen.append((bank, reference_bank))
        return {"timeseries": out / "synthetic.png"}

    monkeypatch.setattr(cli, "_resolve_generate_source", resolve)
    monkeypatch.setattr(cli, "build_dgp_bank", build)
    monkeypatch.setattr(plots, "plot_dgp_bank_saves", plot)
    cli.main(["generate", "synthetic", "--shock", shock, "--out", str(tmp_path),
              "--plot", "timeseries"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["shock"] == shock
    assert len(calls) == (2 if has_reference else 1)
    assert seen == [(observed_bank, reference_bank if has_reference else None)]
    assert calls[0][0].shocks.is_stationary is (not has_reference)
    if has_reference:
        assert calls[1][0].shocks.is_stationary
        assert calls[0][1] == calls[1][1]
        original = msgspec.to_builtins(calls[0][0])
        reference = msgspec.to_builtins(calls[1][0])
        original.pop("shocks")
        reference.pop("shocks")
        assert original == reference


def test_cli_generate_plot_both_skips_attractor_for_one_channel_source(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "cell"
    cli.main(
        [
            "generate",
            "ou",
            "--out",
            str(out),
            "--plot",
            "both",
            "--max-points",
            "200",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    artifact_dir = _generated_artifact_dir(payload)
    assert payload["source"] == "ou"
    assert sorted(payload["artifacts"]["plots"]) == ["timeseries"]
    assert (artifact_dir / "ou_timeseries.png").exists()
    assert not (artifact_dir / "ou_attractor.png").exists()


def test_cli_generate_plot_attractor_rejects_one_channel_source(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="requires at least two channels"):
        cli.main(
            [
                "generate",
                "ou",
                "--out",
                str(tmp_path / "cell"),
                "--plot",
                "attractor",
            ]
        )


def test_cli_walkthrough_writes_arrays_plots_and_file_checks(tmp_path, capsys):
    out = tmp_path / "walkthrough"
    cli.main(
        [
            "walkthrough",
            "--out",
            str(out),
            "--kind",
            "timeseries",
            "--max-points",
            "120",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["sources"] == {
        "regular": "lorenz",
        "param_shift": "lorenz_param_shock",
    }
    assert payload["array_format"] == "all"
    assert sorted(payload["plots"]) == [
        "param_shift",
        "param_shift_noise",
        "regular",
        "regular_noise",
    ]
    assert all(record["exists"] for record in payload["files"])
    assert (out / "arrays" / "lorenz" / "clean.npy").exists()
    assert (out / "arrays" / "lorenz" / "clean.csv").exists()
    assert (out / "arrays" / "lorenz" / "clean.ipc").exists()
    assert (out / "arrays" / "lorenz" / "clean.parquet").exists()
    assert (out / "arrays" / "lorenz" / "dgp_bank.safetensors").exists()
    assert (out / "arrays" / "lorenz_param_shock" / "manifest.json").exists()
    assert (out / "arrays" / "lorenz_param_shock" / "noise_draws.csv").exists()
    assert (out / "arrays" / "lorenz_param_shock" / "noise_draws.ipc").exists()
    assert (out / "arrays" / "lorenz_param_shock" / "noise_draws.parquet").exists()
    assert (out / "plots" / "regular.png").exists()
    assert (out / "plots" / "param_shift_noise.png").exists()
