"""Recent-projects history (Qt-free, JSON-backed)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

MAX_ITEMS = 20


def default_store() -> Path:
    return Path.home() / ".beewings" / "recents.json"


def load_recents(store: Path) -> List[Dict]:
    store = Path(store)
    if not store.exists():
        return []
    try:
        data = json.loads(store.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        return [i for i in data.get("items", []) if isinstance(i, dict)]
    except (json.JSONDecodeError, OSError):
        return []


def save_recents(items: List[Dict], store: Path) -> None:
    store = Path(store)
    try:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps({"version": 1, "items": items},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        # A non-writable home / locked file must not break opening a project.
        pass


def add_recent(items: List[Dict], path: str, stats: Dict, opened_at: str) -> List[Dict]:
    """Return a new list with `path` at the front (deduped), stats+time merged."""
    entry = {"path": str(path), "name": Path(path).name, "opened_at": opened_at}
    entry.update(stats)
    rest = [i for i in items if i.get("path") != str(path)]
    return [entry] + rest[: MAX_ITEMS - 1]


def remove_recent(items: List[Dict], path: str) -> List[Dict]:
    return [i for i in items if i.get("path") != str(path)]
