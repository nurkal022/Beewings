"""Regression tests for crash-hardening: malformed inputs must degrade
gracefully (return None / fall back / no-op), never raise into the GUI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from beewings.app.recents import load_recents, save_recents
from beewings.core.profiles import DEFAULT_PROFILE, PROFILES, get_profile
from beewings.core.schema import load_annotation
from beewings.pipeline.project import CropProject


def test_get_profile_unknown_falls_back_to_default():
    assert get_profile("does-not-exist") is PROFILES[DEFAULT_PROFILE]


def test_get_profile_legacy_name_still_resolves():
    assert get_profile("12-point") is PROFILES["Алпатов 12 точек"]


def test_load_annotation_corrupt_returns_none(tmp_path: Path):
    bad = tmp_path / "ann.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")
    assert load_annotation(bad) is None


def test_load_annotation_missing_returns_none(tmp_path: Path):
    assert load_annotation(tmp_path / "nope.json") is None


def test_project_load_corrupt_json_raises_friendly_valueerror(tmp_path: Path):
    (tmp_path / "project.json").write_text("{ broken", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        CropProject.load(tmp_path)
    assert "повреждён" in str(ei.value)


def test_project_load_tolerates_missing_optional_fields(tmp_path: Path):
    (tmp_path / "project.json").write_text(
        json.dumps({"scans": [{"path": str(tmp_path / "a.jpg")}]}), encoding="utf-8")
    proj = CropProject.load(tmp_path)  # no scans_root / settings / stage
    assert [Path(s.path).name for s in proj.scans] == ["a.jpg"]


def test_project_load_skips_malformed_scan_entries(tmp_path: Path):
    (tmp_path / "project.json").write_text(json.dumps({
        "scans_root": str(tmp_path),
        "scans": [
            {"path": str(tmp_path / "good.jpg"), "wing_boxes": [[1, 2, 3, 4]]},
            {"no_path": True},                 # missing path -> skipped
            "not-a-dict",                      # wrong type -> skipped
            {"path": str(tmp_path / "b.jpg"), "wing_boxes": [["x"]]},  # bad box -> dropped
        ],
    }), encoding="utf-8")
    proj = CropProject.load(tmp_path)
    names = [Path(s.path).name for s in proj.scans]
    assert names == ["good.jpg", "b.jpg"]
    assert proj.scans[1].wing_boxes == []  # malformed box silently dropped


def test_save_recents_never_raises_on_unwritable_path(tmp_path: Path):
    # A file standing where the parent directory should be makes mkdir/write fail.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    store = blocker / "recents.json"
    save_recents([{"path": "/x"}], store)  # must not raise


def test_load_recents_non_dict_json_returns_empty(tmp_path: Path):
    store = tmp_path / "recents.json"
    store.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_recents(store) == []
