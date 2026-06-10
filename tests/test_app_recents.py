from __future__ import annotations

from beewings.app.recents import add_recent, remove_recent, load_recents, save_recents


def test_add_and_load_roundtrip(tmp_path):
    store = tmp_path / "recents.json"
    items = add_recent([], "/a/Клат", {"n_splits": 3}, "2026-06-10T10:00:00")
    save_recents(items, store)
    loaded = load_recents(store)
    assert loaded[0]["path"] == "/a/Клат"
    assert loaded[0]["name"] == "Клат"
    assert loaded[0]["n_splits"] == 3
    assert loaded[0]["opened_at"] == "2026-06-10T10:00:00"


def test_dedup_moves_to_front(tmp_path):
    items = []
    items = add_recent(items, "/a/x", {}, "2026-06-10T10:00:00")
    items = add_recent(items, "/a/y", {}, "2026-06-10T11:00:00")
    items = add_recent(items, "/a/x", {"n_splits": 9}, "2026-06-10T12:00:00")
    assert [i["path"] for i in items] == ["/a/x", "/a/y"]
    assert items[0]["n_splits"] == 9


def test_truncates_to_20(tmp_path):
    items = []
    for i in range(25):
        items = add_recent(items, f"/p/{i}", {}, f"2026-06-10T{i:02d}:00:00")
    assert len(items) == 20
    assert items[0]["path"] == "/p/24"


def test_remove(tmp_path):
    items = add_recent([], "/a/x", {}, "t")
    items = remove_recent(items, "/a/x")
    assert items == []


def test_load_missing_file_returns_empty(tmp_path):
    assert load_recents(tmp_path / "nope.json") == []
