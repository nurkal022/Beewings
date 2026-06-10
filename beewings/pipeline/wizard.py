"""PipelineWizard: a 4-stage stepper over a CropProject."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
                             QStackedWidget, QVBoxLayout, QWidget)

from .pages.crop_page import CropPage
from .pages.export_page import ExportPage
from .pages.landmark_page import LandmarkPage
from .pages.select_page import SelectPage
from .project import CropProject

STAGES = ["1. Сканы", "2. Нарезка и правка", "3. Точки", "4. Экспорт"]


class PipelineWizard(QWidget):
    def __init__(self, project_root: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BeeWings — Конвейер нарезки")
        self.resize(1100, 720)
        root = Path(project_root) if project_root else Path.cwd() / "beewings_project"
        if (root / "project.json").exists():
            self.project = CropProject.load(root)
        else:
            self.project = CropProject.create(root, scans_root="", scan_paths=[])
            self.project.save()

        self.ctx = {"project": self.project, "open_in_annotator": None}

        outer = QVBoxLayout(self)
        self.step_label = QLabel()
        outer.addWidget(self.step_label)
        self.stack = QStackedWidget()
        self.pages = [SelectPage(self.ctx), CropPage(self.ctx),
                      LandmarkPage(self.ctx), ExportPage(self.ctx)]
        for pg in self.pages:
            self.stack.addWidget(pg)
        outer.addWidget(self.stack, 1)

        nav = QHBoxLayout()
        self.back_btn = QPushButton("◀ Назад")
        self.back_btn.clicked.connect(self._back)
        self.next_btn = QPushButton("Далее ▶")
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.back_btn)
        nav.addStretch(1)
        nav.addWidget(self.next_btn)
        outer.addLayout(nav)
        self._update_header()

    def _update_header(self) -> None:
        i = self.stack.currentIndex()
        self.step_label.setText("   →   ".join(
            (f"[{s}]" if n == i else s) for n, s in enumerate(STAGES)))
        self.back_btn.setEnabled(i > 0)
        self.next_btn.setText("Готово" if i == len(self.pages) - 1 else "Далее ▶")

    def _enter_current(self) -> None:
        pg = self.pages[self.stack.currentIndex()]
        if hasattr(pg, "enter"):
            pg.enter()

    def _next(self) -> None:
        pg = self.pages[self.stack.currentIndex()]
        if hasattr(pg, "commit") and not pg.commit():
            return
        if self.stack.currentIndex() < len(self.pages) - 1:
            self.stack.setCurrentIndex(self.stack.currentIndex() + 1)
            self._enter_current()
            self._update_header()

    def _back(self) -> None:
        if self.stack.currentIndex() > 0:
            self.stack.setCurrentIndex(self.stack.currentIndex() - 1)
            self._enter_current()
            self._update_header()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    w = PipelineWizard()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
