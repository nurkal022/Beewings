"""Project window: top tabs over a CropProject (Нарезка / Точки / Экспорт)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtWidgets import (QHBoxLayout, QListWidget, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QStatusBar, QTabWidget,
                             QVBoxLayout, QWidget)

from ..annotator.annotator_widget import AnnotatorWidget
from ..pipeline.pages.crop_page import CropPage
from ..pipeline.pages.export_page import ExportPage
from ..pipeline.project import CropProject
from ..pipeline.workers import LandmarkWorker


class LandmarkTab(QWidget):
    """Pick a split, run ML on its crops, edit points in the embedded annotator."""

    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker = None
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.split_list = QListWidget()
        self.split_list.currentRowChanged.connect(self._show_split)
        left.addWidget(self.split_list)
        self.run_btn = QPushButton("\U0001f9e0 Расставить точки (ML)")
        self.run_btn.clicked.connect(self._run)
        left.addWidget(self.run_btn)
        self.progress = QProgressBar()
        left.addWidget(self.progress)
        root.addLayout(left, 0)
        self.annot = AnnotatorWidget()
        root.addWidget(self.annot, 1)

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        self.split_list.clear()
        self.split_list.addItems([Path(s.path).name for s in proj.scans])
        if proj.scans:
            self.split_list.setCurrentRow(0)

    def _show_split(self, row: int) -> None:
        proj: CropProject = self.ctx["project"]
        if 0 <= row < len(proj.scans):
            self._row = row
            cdir = proj.crops_dir(proj.scans[row])
            if cdir.exists():
                self.annot.load_folder(cdir)

    def _run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        proj: CropProject = self.ctx["project"]
        self.progress.setRange(0, 0)
        self.run_btn.setEnabled(False)
        self._worker = LandmarkWorker(proj)
        self._worker.progress.connect(
            lambda i, t, name: self.progress.setFormat(name))
        self._worker.failed.connect(
            lambda path, err: QMessageBox.warning(self, "Ошибка разметки", f"{path}\n{err}"))
        self._worker.finished_ok.connect(self._done)
        self._worker.start()

    def _done(self) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.run_btn.setEnabled(True)
        self._show_split(getattr(self, "_row", 0))


class ProjectWindow(QMainWindow):
    def __init__(self, project: CropProject, on_home: Callable[[], None], parent=None):
        super().__init__(parent)
        self.project = project
        self._on_home = on_home
        self.setWindowTitle(f"BeeWings — {Path(project.root).name}")
        self.resize(1500, 920)
        self.ctx = {"project": project}

        central = QWidget()
        outer = QVBoxLayout(central)
        bar = QHBoxLayout()
        home_btn = QPushButton("← Дом")
        home_btn.setObjectName("ghost")
        home_btn.clicked.connect(self._go_home)
        bar.addWidget(home_btn)
        bar.addStretch(1)
        outer.addLayout(bar)

        self.tabs = QTabWidget()
        self.crop_tab = CropPage(self.ctx)
        self.landmark_tab = LandmarkTab(self.ctx)
        self.export_tab = ExportPage(self.ctx)
        self.tabs.addTab(self.crop_tab, "Нарезка")
        self.tabs.addTab(self.landmark_tab, "Точки")
        self.tabs.addTab(self.export_tab, "Экспорт")
        self.tabs.currentChanged.connect(self._on_tab)
        outer.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))
        self.landmark_tab.annot.status.connect(self.statusBar().showMessage)
        self._on_tab(0)

    def _on_tab(self, idx: int) -> None:
        w = self.tabs.widget(idx)
        if hasattr(w, "enter"):
            w.enter()

    def _go_home(self) -> None:
        self.project.save()
        self._on_home()
        self.close()
