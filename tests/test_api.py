"""Tests for the BeeWings HTTP API (beewings.api).

Endpoints that need a trained checkpoint (/detect) are skipped automatically
when the .pt files are absent, so the suite runs anywhere.
"""
from __future__ import annotations

import json

import cv2
import pytest
from fastapi.testclient import TestClient

from beewings.api.app import app
from beewings.api.registry import checkpoints_dir

client = TestClient(app)

_ALPATOV_CKPT = checkpoints_dir() / "alpatov12.pt"
_TOFILSKI_CKPT = checkpoints_dir() / "tofilski19.pt"


# --- meta -----------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "loaded_models" in body


def test_models():
    r = client.get("/models")
    assert r.status_code == 200
    methods = r.json()["methodologies"]
    assert "alpatov" in methods and "tofilski" in methods


# --- indices --------------------------------------------------------------

def test_indices_synthetic():
    # 12 points on a unit-ish layout so CI denominators are non-zero.
    pts = [{"id": i, "x": float(i * 10), "y": float((i % 3) * 7 + 5)} for i in range(1, 13)]
    r = client.post("/indices", json={"landmarks": pts})
    assert r.status_code == 200
    body = r.json()
    names = {iv["name"] for iv in body["indices"]}
    assert any(n.startswith("CI (Алпатов") for n in names)
    ci = next(iv["value"] for iv in body["indices"] if iv["name"].startswith("CI (Алпатов"))
    assert ci is not None


# --- segment (inline, hermetic via synthetic scan) ------------------------

def test_segment_inline_upload(scan_file):
    path, meta = scan_file
    with open(path, "rb") as fh:
        r = client.post("/segment", files={"scan": ("scan.jpg", fh, "image/jpeg")},
                        data={"output": "inline"})
    assert r.status_code == 200
    body = r.json()
    assert body["output"] == "inline"
    assert body["n_wings"] > 0
    # every box carries an inline crop
    assert all(b["crop_b64"] for b in body["boxes"])


def test_segment_files_path(scan_file, tmp_path, monkeypatch):
    path, meta = scan_file
    out = tmp_path / "crops"
    # Confine server-side paths to tmp_path for this test.
    monkeypatch.setenv("BEEWINGS_DATA_ROOT", str(tmp_path))
    r = client.post("/segment", json={
        "scan_path": str(path), "output": "files", "out_dir": str(out),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["n_wings"] > 0
    assert len(body["files"]) == body["n_wings"]
    assert out.exists()


# --- export ---------------------------------------------------------------

def _wings_payload():
    lm = [{"id": i, "x": float(i * 12), "y": float((i % 4) * 9 + 3)} for i in range(1, 13)]
    return [{"wing": "crop_0.jpg", "landmarks": lm},
            {"wing": "crop_1.jpg", "landmarks": lm}]


def test_export_json_download():
    r = client.post("/export?format=json", json={
        "methodology": "alpatov", "wings": _wings_payload(), "output": "download",
    })
    assert r.status_code == 200
    payload = json.loads(r.content)
    assert payload["methodology"] == "alpatov"
    assert payload["n_wings"] == 2
    assert "index_means" in payload


def test_export_files(tmp_path, monkeypatch):
    monkeypatch.setenv("BEEWINGS_DATA_ROOT", str(tmp_path))
    out = tmp_path / "export"
    r = client.post("/export?format=all", json={
        "methodology": "alpatov", "wings": _wings_payload(),
        "output": "files", "out_dir": str(out),
    })
    assert r.status_code == 200
    files = r.json()["files"]
    assert {"tps", "xlsx", "json"} <= set(files)
    for p in files.values():
        assert (tmp_path in __import__("pathlib").Path(p).parents)


def test_export_all_download_zip():
    r = client.post("/export?format=all", json={
        "methodology": "tofilski", "wings": [
            {"wing": "w.jpg", "landmarks": [{"id": i, "x": float(i), "y": float(i)}
                                             for i in range(1, 20)]}],
        "output": "download",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"


# --- detect (needs checkpoint) --------------------------------------------

@pytest.mark.skipif(not _ALPATOV_CKPT.exists(), reason="alpatov12.pt not present")
def test_detect_upload_alpatov():
    img = sorted((__import__("pathlib").Path("demo_wings")).glob("*.jpg"))
    if not img:
        pytest.skip("no demo wings")
    with open(img[0], "rb") as fh:
        r = client.post("/detect", files={"image": ("w.jpg", fh, "image/jpeg")},
                        data={"methodology": "alpatov", "tta": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_points"] == 12
    assert len(body["landmarks"]) > 0
    assert body["indices"] is not None  # indices auto-computed for Alpatov-12


@pytest.mark.skipif(not _ALPATOV_CKPT.exists(), reason="alpatov12.pt not present")
def test_detect_path_json(monkeypatch):
    from pathlib import Path
    imgs = sorted(Path("demo_wings").glob("*.jpg"))
    if not imgs:
        pytest.skip("no demo wings")
    monkeypatch.setenv("BEEWINGS_DATA_ROOT", str(Path("demo_wings").resolve()))
    r = client.post("/detect", json={
        "image_path": str(imgs[0].resolve()), "methodology": "alpatov",
    })
    assert r.status_code == 200
    assert r.json()["n_points"] == 12
