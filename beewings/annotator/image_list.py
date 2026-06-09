"""Left-side panel listing all images in the project folder with progress."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget, QLabel

from ..core.profiles import Profile
from ..core.schema import annotation_path, load_annotation


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class ImageList(QWidget):
    image_selected = pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._title = QLabel("Папка не выбрана")
        self._title.setStyleSheet("QLabel { font-weight: bold; padding: 4px; }")
        layout.addWidget(self._title)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_changed)
        layout.addWidget(self.list)
        self._folder: Optional[Path] = None
        self._paths: List[Path] = []

    def load_folder(self, folder: Path, profile: Profile) -> None:
        self._folder = folder
        self._paths = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
        )
        self._title.setText(f"{folder.name} (изображений: {len(self._paths)})")
        self.list.clear()
        for p in self._paths:
            self.list.addItem(self._make_item(p, profile))
        if self._paths:
            self.list.setCurrentRow(0)

    def refresh_progress(self, profile: Profile) -> None:
        for i, p in enumerate(self._paths):
            item = self.list.item(i)
            item.setText(self._format_label(p, profile))

    def select_offset(self, delta: int) -> None:
        row = self.list.currentRow()
        new_row = max(0, min(self.list.count() - 1, row + delta))
        if new_row != row:
            self.list.setCurrentRow(new_row)

    def current_path(self) -> Optional[Path]:
        row = self.list.currentRow()
        if row < 0 or row >= len(self._paths):
            return None
        return self._paths[row]

    def previous_path(self) -> Optional[Path]:
        row = self.list.currentRow()
        if row <= 0:
            return None
        return self._paths[row - 1]

    def all_paths(self) -> List[Path]:
        return list(self._paths)

    # ---- internals ----------------------------------------------------------

    def _make_item(self, p: Path, profile: Profile) -> QListWidgetItem:
        item = QListWidgetItem(self._format_label(p, profile))
        item.setData(Qt.ItemDataRole.UserRole, str(p))
        return item

    def _format_label(self, p: Path, profile: Profile) -> str:
        if self._folder is None:
            return p.name
        ann = load_annotation(annotation_path(p, self._folder))
        if ann is None:
            return f"○  {p.name}  (0/{len(profile.ids)})"
        done, total = ann.progress(profile.ids)
        mark = "✓" if done == total else "●"
        return f"{mark}  {p.name}  ({done}/{total})"

    def _on_changed(self, current: QListWidgetItem, _previous):
        if current is None:
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        self.image_selected.emit(path)
