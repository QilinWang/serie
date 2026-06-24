from __future__ import annotations

import pytest

from serie import KnownMeta, load_data


def test_known_meta_presets_use_external_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SERIE_KNOWN_ROOT", str(tmp_path))

    meta = KnownMeta.etth1()

    assert meta.path == tmp_path / "ett" / "ETTh1.ipc"


def test_known_meta_missing_preset_data_has_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SERIE_KNOWN_ROOT", str(tmp_path))

    with pytest.raises(FileNotFoundError) as exc:
        load_data(KnownMeta.etth1())

    message = str(exc.value)
    assert "not bundled with the SERIE wheel" in message
    assert "SERIE_KNOWN_ROOT" in message
    assert "KnownMeta.custom" in message
    assert "https://github.com/zhouhaoyi/ETDataset" in message
