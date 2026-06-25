"""CLI integration tests — verify every subcommand produces parseable output.

These tests call ``cli.main([...])`` directly (no subprocess) and parse stdout.
They are the contract between the CLI surface and agents/scripts that consume
SERIE's JSON output.  If a JSON key disappears or a command silently changes
its output shape, these tests catch it.

Coverage:
    discovery  — commands, list, data-dir, inspect
    generate   — metadata-only, with arrays+plots, with shock, schema stability,
                 train_std artifact rejection
    legacy     — inject-noise (unit + train_std), export, plot (timeseries,
                 attractor, both, 1-channel fallback)
    walkthrough — full end-to-end audit with file existence checks

Each test is named ``test_cli_<subcommand>_<what it checks>``.
"""

from __future__ import annotations

import json

import pytest

from serie import cli


def test_cli_commands_json_is_agent_readable(capsys):
    cli.main(["commands", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    commands = {row["command"]: row for row in payload["commands"]}
    assert "serie --help" == payload["help"]["all_commands"]
    assert "serie <command> --help" == payload["help"]["one_command"]
    assert commands["generate"]["group"] == "02 generate"
    assert "sigma" in commands["generate"]["example"]
    assert "shock" in commands["generate"]["example"]
    assert commands["demo"]["group"] == "03 visualize"


def test_cli_commands_text_groups_commands(capsys):
    cli.main(["commands"])
    out = capsys.readouterr().out
    assert "[01 discovery]" in out
    assert "commands: List SERIE CLI commands with examples." in out
    assert "[02 generate]" in out
    assert "generate: Generate one SERIE cell" in out
    assert "Help:" in out


def test_cli_list_json_is_agent_readable(capsys):
    cli.main(["list", "--format", "json"])
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

    cli.main(["data-dir", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"data_dir": str(root), "exists": False}


def test_cli_rejects_unknown_source():
    with pytest.raises(SystemExit, match="unknown source"):
        cli.main(["inspect", "not_a_source"])


def test_cli_inject_noise_json_is_parseable(capsys):
    cli.main(["inject-noise", "ou", "--sigma", "0.1", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "ou"
    assert payload["sigma"] == 0.1
    assert payload["sigma_scale"] == "unit"
    assert payload["shape"]["train"]


def test_cli_inject_noise_train_std_json_is_parseable(capsys):
    cli.main(
        [
            "inject-noise",
            "ou",
            "--sigma",
            "1.0",
            "--sigma-scale",
            "train_std",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["sigma"] == 1.0
    assert payload["sigma_scale"] == "train_std"


def test_cli_generate_prints_one_cell_json(capsys):
    cli.main(["generate", "ou", "--sigma", "0.1", "--array-format", "none", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "ou"
    assert payload["shock"] == "none"
    assert payload["sigma"] == 0.1
    assert payload["artifacts"] is None


def test_cli_generate_json_schema_is_stable(capsys):
    cli.main(["generate", "ou", "--sigma", "0.1", "--array-format", "none", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)

    assert sorted(payload) == [
        "artifacts",
        "cell_tag",
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
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "lorenz_param_shock"
    assert payload["name"] == "lorenz_param_shock"
    assert payload["shock"] == "param"
    assert payload["artifacts"]["arrays"]["array_format"] == "csv"
    assert payload["artifacts"]["plots"]["timeseries"]
    assert (out / "clean.csv").exists()
    assert (out / "noise_draws.csv").exists()
    assert (out / "manifest.json").exists()
    assert (out / "lorenz_param_shock_timeseries.png").exists()


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
            "--out",
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
            "--out",
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


def test_cli_demo_out_dir_writes_html_json_and_static_plots(tmp_path, capsys):
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
    assert payload["data"] == str(out_dir / "demo_data.json")
    assert sorted(payload["plots"]) == ["attractor", "timeseries"]
    assert (out_dir / "index.html").exists()
    assert (out_dir / "demo_data.json").exists()
    assert (out_dir / "static" / "henon_timeseries.png").exists()
    assert (out_dir / "static" / "henon_attractor.png").exists()


def test_cli_export_writes_npys_and_manifest(tmp_path, capsys):
    out = tmp_path / "export"
    cli.main(
        [
            "export",
            "ou",
            "--out",
            str(out),
            "--sigma",
            "0.25",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "ou"
    assert payload["array_format"] == "all"
    assert payload["n_noisy"] == 1
    assert (out / "clean.npy").exists()
    assert (out / "clean.csv").exists()
    assert (out / "clean.ipc").exists()
    assert (out / "clean.parquet").exists()
    assert (out / "noise_draws.npy").exists()
    assert (out / "noise_draws.csv").exists()
    assert (out / "noise_draws.ipc").exists()
    assert (out / "noise_draws.parquet").exists()
    assert (out / "manifest.json").exists()


def test_cli_plot_both_writes_timeseries_and_attractor(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "lorenz.png"
    cli.main(
        [
            "plot",
            "lorenz",
            "--out",
            str(out),
            "--kind",
            "both",
            "--sigma",
            "0.1",
            "--max-points",
            "200",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "lorenz"
    assert sorted(payload["files"]) == ["attractor", "timeseries"]
    assert (tmp_path / "lorenz_timeseries.png").exists()
    assert (tmp_path / "lorenz_attractor.png").exists()


def test_cli_plot_intervention_source_writes_timeseries_with_reference_path(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "lorenz_param_shock.png"
    cli.main(
        [
            "plot",
            "lorenz_param_shock",
            "--out",
            str(out),
            "--kind",
            "timeseries",
            "--sigma",
            "0.1",
            "--max-points",
            "200",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "lorenz_param_shock"
    assert sorted(payload["files"]) == ["timeseries"]
    assert out.exists()


def test_cli_plot_both_skips_attractor_for_one_channel_source(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "ou.png"
    cli.main(
        [
            "plot",
            "ou",
            "--out",
            str(out),
            "--kind",
            "both",
            "--max-points",
            "200",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "ou"
    assert sorted(payload["files"]) == ["timeseries"]
    assert (tmp_path / "ou_timeseries.png").exists()
    assert not (tmp_path / "ou_attractor.png").exists()


def test_cli_plot_attractor_rejects_one_channel_source(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="requires at least two channels"):
        cli.main(
            [
                "plot",
                "ou",
                "--out",
                str(tmp_path / "ou.png"),
                "--kind",
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
    assert (out / "arrays" / "lorenz_param_shock" / "manifest.json").exists()
    assert (out / "arrays" / "lorenz_param_shock" / "noise_draws.csv").exists()
    assert (out / "arrays" / "lorenz_param_shock" / "noise_draws.ipc").exists()
    assert (out / "arrays" / "lorenz_param_shock" / "noise_draws.parquet").exists()
    assert (out / "plots" / "regular.png").exists()
    assert (out / "plots" / "param_shift_noise.png").exists()
