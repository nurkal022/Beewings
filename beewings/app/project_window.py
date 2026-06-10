"""Project window: top tabs over a CropProject (Нарезка / Точки / Экспорт)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QHBoxLayout, QMainWindow, QMessageBox, QProgressBar,
                             QPushButton, QStatusBar, QTabWidget, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core.profiles import get_profile
from ..core.schema import annotation_path, load_annotation

from ..annotator.annotator_widget import AnnotatorWidget
from ..pipeline.pages.crop_page import CropPage
from ..pipeline.pages.export_page import ExportPage
from ..pipeline.project import CropProject
from ..pipeline.workers import LandmarkWorker


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_ROLE = Qt.ItemDataRole.UserRole


def _crop_files(cdir):
    skip = ("_label", "_debug")
    return sorted(p for p in cdir.glob("*")
                  if p.suffix.lower() in _IMAGE_EXTS and not any(t in p.stem for t in skip))


class LandmarkTab(QWidget):
    """Tree (scan -> wings) driving the embedded annotator; ML over all crops."""

    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker = None
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemExpanded.connect(self._populate)
        self.tree.itemClicked.connect(self._on_item_clicked)
        left.addWidget(self.tree)
        self.run_btn = QPushButton("\U0001f9e0 Расставить точки (ML)")
        self.run_btn.clicked.connect(self._run)
        left.addWidget(self.run_btn)
        self.progress = QProgressBar()
        left.addWidget(self.progress)
        root.addLayout(left, 0)
        self.annot = AnnotatorWidget()
        self.annot.set_browser_visible(False)
        root.addWidget(self.annot, 1)

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        self.tree.clear()
        for i, scan in enumerate(proj.scans):
            item = QTreeWidgetItem([Path(scan.path).name])
            item.setData(0, _ROLE, ("scan", i))
            QTreeWidgetItem(item, ["…"])  # placeholder so it is expandable
            self.tree.addTopLevelItem(item)

    def _populate(self, parent: "QTreeWidgetItem") -> None:
        kind, idx = parent.data(0, _ROLE)
        if kind != "scan":
            return
        if parent.childCount() == 1 and parent.child(0).data(0, _ROLE) is None:
            parent.takeChildren()  # drop placeholder
        else:
            return  # already populated
        proj: CropProject = self.ctx["project"]
        cdir = proj.crops_dir(proj.scans[idx])
        crops = _crop_files(cdir) if cdir.exists() else []
        if not crops:
            child = QTreeWidgetItem(parent, ["(не нарезано)"])
            child.setDisabled(True)
            child.setData(0, _ROLE, ("empty",))
            return
        prof = get_profile(proj.settings.profile)
        for cp in crops:
            child = QTreeWidgetItem(parent, [self._wing_label(cp, cdir, prof)])
            child.setData(0, _ROLE, ("wing", idx, str(cp)))

    def _wing_label(self, cp, cdir, prof) -> str:
        ann = load_annotation(annotation_path(cp, cdir))
        if ann is None:
            return f"○  {cp.name}"
        done, total = ann.progress(prof.ids)
        mark = "✓" if done == total and total else "●"
        return f"{mark}  {cp.name}"

    def _on_item_clicked(self, item: "QTreeWidgetItem", _col: int) -> None:
        data = item.data(0, _ROLE)
        if not data:
            return
        if data[0] == "scan":
            item.setExpanded(True)
            if item.childCount() and item.child(0).data(0, _ROLE):
                self._open_wing(item.child(0))
        elif data[0] == "wing":
            self._open_wing(item)

    def _open_wing(self, item: "QTreeWidgetItem") -> None:
        data = item.data(0, _ROLE)
        if not data or data[0] != "wing":
            return
        _, idx, crop = data
        proj: CropProject = self.ctx["project"]
        cdir = proj.crops_dir(proj.scans[idx])
        self.annot.show_image(cdir, Path(crop))

    def _run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        proj: CropProject = self.ctx["project"]
        self.progress.setRange(0, 0)
        self.run_btn.setEnabled(False)
        self._worker = LandmarkWorker(proj)
        self._worker.progress.connect(lambda i, t, name: self.progress.setFormat(name))
        self._worker.failed.connect(
            lambda path, err: QMessageBox.warning(self, "Ошибка разметки", f"{path}\n{err}"))
        self._worker.finished_ok.connect(self._done)
        self._worker.start()

    def _done(self) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.run_btn.setEnabled(True)
        self.enter()  # rebuild tree (refresh progress marks)


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
