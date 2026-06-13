from __future__ import annotations

from pathlib import Path

from beewings.core.paths import app_base_dir, resolve_checkpoint


def test_app_base_dir_is_repo_root_from_source():
    base = app_base_dir()
    # repo root contains the beewings package and pyproject.toml
    assert (base / "beewings").is_dir()
    assert (base / "pyproject.toml").is_file()


def test_resolve_absolute_path_unchanged(tmp_path):
    f = tmp_path / "m.pt"
    f.write_bytes(b"x")
    assert resolve_checkpoint(f) == f


def test_resolve_existing_relative(tmp_path, monkeypatch):
    (tmp_path / "checkpoints").mkdir()
    f = tmp_path / "checkpoints" / "a.pt"
    f.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert resolve_checkpoint("checkpoints/a.pt") == Path("checkpoints/a.pt")


def test_resolve_missing_falls_back_to_app_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)            # nothing here
    got = resolve_checkpoint("checkpoints/none.pt")
    assert got == app_base_dir() / "checkpoints" / "none.pt"
