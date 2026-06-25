"""Known-data boundary tests — verify error messages when datasets are missing.

The SERIE wheel does not bundle large known datasets (ETT, HuGaDB).  These
tests verify that when a user tries to load a known dataset that isn't staged
locally, the error message:

    1. says the dataset is not bundled with the wheel,
    2. tells them to set SERIE_KNOWN_ROOT or use KnownMeta.custom(path=...),
    3. includes the upstream source URL so they can find the data.

These tests use monkeypatch to point SERIE_KNOWN_ROOT at an empty temp
directory, so they never need real dataset files.
"""

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
