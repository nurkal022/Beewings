"""Stage 1: choose the folder of scans."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (QCheckBox, QFileDialog, QLabel, QListWidget,
                             QPushButton, QVBoxLayout, QWidget)

from ..project import CropProject

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class SelectPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Этап 1. Выберите папку со сканами:"))
        self.recursive = QCheckBox("Искать во вложенных папках")
        lay.addWidget(self.recursive)
        btn = QPushButton("Выбрать папку…")
        btn.clicked.connect(self._choose)
        lay.addWidget(btn)
        self.list = QListWidget()
        lay.addWidget(self.list)

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Папка со сканами")
        if not folder:
            return
        root = Path(folder)
        globber = root.rglob if self.recursive.isChecked() else root.glob
        skip = ("_crop_", "_label", "_debug")
        scans = sorted(p for p in globber("*")
                       if p.suffix.lower() in IMAGE_EXTS
                       and not any(t in p.stem for t in skip))
        self.list.clear()
        self.list.addItems([str(p) for p in scans])
        self.ctx["scan_paths"] = scans
        self.ctx["scans_root"] = str(root)

    def commit(self) -> bool:
        scans = self.ctx.get("scan_paths") or []
        if not scans:
            return False
        proj: CropProject = self.ctx["project"]
        from ..project import ScanEntry
        proj.scans = [ScanEntry(path=str(p)) for p in scans]
        proj.scans_root = self.ctx["scans_root"]
        proj.stage = "crop"
        proj.save()
        return True
