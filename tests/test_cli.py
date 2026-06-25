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
    cli.main([
        "inject-noise",
        "ou",
        "--sigma",
        "1.0",
        "--sigma-scale",
        "train_std",
        "--format",
        "json",
    ])
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
    cli.main([
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
    ])
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
    with pytest.raises(SystemExit, match="File exports and plots currently require --sigma-scale unit"):
        cli.main([
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
        ])


def test_cli_export_writes_npys_and_manifest(tmp_path, capsys):
    out = tmp_path / "export"
    cli.main([
        "export",
        "ou",
        "--out",
        str(out),
        "--sigma",
        "0.25",
        "--format",
        "json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "ou"
    assert payload["array_format"] == "both"
    assert payload["n_noisy"] == 1
    assert (out / "clean.npy").exists()
    assert (out / "clean.csv").exists()
    assert (out / "noise_draws.npy").exists()
    assert (out / "noise_draws.csv").exists()
    assert (out / "manifest.json").exists()


def test_cli_plot_both_writes_timeseries_and_attractor(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "lorenz.png"
    cli.main([
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
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "lorenz"
    assert sorted(payload["files"]) == ["attractor", "timeseries"]
    assert (tmp_path / "lorenz_timeseries.png").exists()
    assert (tmp_path / "lorenz_attractor.png").exists()


def test_cli_plot_intervention_source_writes_timeseries_with_reference_path(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "lorenz_param_shock.png"
    cli.main([
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
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "lorenz_param_shock"
    assert sorted(payload["files"]) == ["timeseries"]
    assert out.exists()


def test_cli_plot_both_skips_attractor_for_one_channel_source(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "ou.png"
    cli.main([
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
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "ou"
    assert sorted(payload["files"]) == ["timeseries"]
    assert (tmp_path / "ou_timeseries.png").exists()
    assert not (tmp_path / "ou_attractor.png").exists()


def test_cli_plot_attractor_rejects_one_channel_source(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="requires at least two channels"):
        cli.main([
            "plot",
            "ou",
            "--out",
            str(tmp_path / "ou.png"),
            "--kind",
            "attractor",
        ])


def test_cli_walkthrough_writes_arrays_plots_and_file_checks(tmp_path, capsys):
    out = tmp_path / "walkthrough"
    cli.main([
        "walkthrough",
        "--out",
        str(out),
        "--kind",
        "timeseries",
        "--max-points",
        "120",
        "--format",
        "json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["sources"] == {
        "regular": "lorenz",
        "param_shift": "lorenz_param_shock",
    }
    assert payload["array_format"] == "both"
    assert sorted(payload["plots"]) == [
        "param_shift",
        "param_shift_noise",
        "regular",
        "regular_noise",
    ]
    assert all(record["exists"] for record in payload["files"])
    assert (out / "arrays" / "lorenz" / "clean.npy").exists()
    assert (out / "arrays" / "lorenz" / "clean.csv").exists()
    assert (out / "arrays" / "lorenz_param_shock" / "manifest.json").exists()
    assert (out / "arrays" / "lorenz_param_shock" / "noise_draws.csv").exists()
    assert (out / "plots" / "regular.png").exists()
    assert (out / "plots" / "param_shift_noise.png").exists()
