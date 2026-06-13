"""Project window: top tabs over a CropProject (Нарезка / Точки / Экспорт)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6 import sip
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QHBoxLayout, QMainWindow, QMessageBox, QProgressBar,
                             QPushButton, QStatusBar, QTabWidget, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core.profiles import (DEFAULT_PROFILE_PER_METHODOLOGY, METHODOLOGIES,
                             get_profile)
from ..core.schema import annotation_path, load_annotation

from ..annotator.annotator_widget import AnnotatorWidget
from ..pipeline.pages.crop_page import CropPage
from ..pipeline.pages.export_page import ExportPage
from ..pipeline.project import CropProject
from ..pipeline.workers import CropWorker, LandmarkWorker


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
        self._crop_worker = None
        root = QHBoxLayout(self)
        left = QVBoxLayout()

        # Methodology selector: drives which methodology's points are shown,
        # edited, and produced by ML. Both are stored separately on disk.
        meth_row = QHBoxLayout()
        self.meth_buttons: dict[str, QPushButton] = {}
        for mid, m in METHODOLOGIES.items():
            b = QPushButton(m.display_name)
            b.setCheckable(True)
            b.clicked.connect(lambda _c=False, x=mid: self._on_methodology(x))
            meth_row.addWidget(b)
            self.meth_buttons[mid] = b
        left.addLayout(meth_row)

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

    # ---- methodology -------------------------------------------------------

    def _active_profile(self):
        return get_profile(self.ctx["project"].settings.profile)

    def _sync_meth_buttons(self) -> None:
        active = self._active_profile().methodology_id
        for mid, b in self.meth_buttons.items():
            b.setChecked(mid == active)

    def _on_methodology(self, mid: str) -> None:
        proj: CropProject = self.ctx["project"]
        prof = get_profile(DEFAULT_PROFILE_PER_METHODOLOGY[mid])
        if prof.name == proj.settings.profile:
            self._sync_meth_buttons()
            return
        proj.settings.profile = prof.name
        proj.settings.checkpoint = f"checkpoints/{prof.checkpoint_name}"
        proj.save()
        self.annot.set_active_profile(prof.name)
        self._sync_meth_buttons()
        self._build_tree()

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        self._sync_meth_buttons()
        self.annot.set_active_profile(self._active_profile().name)
        # Crops are written to disk by recrop(), not by auto-detection. If the
        # operator auto-detected (and edited) boxes but never pressed
        # "Пересоздать кропы", the crops are missing and this stage would show
        # "(не нарезано)". Write any missing crops first, then build the tree.
        needs_crop = any(s.wing_boxes and not s.cropped for s in proj.scans)
        if needs_crop and (self._crop_worker is None
                           or not self._crop_worker.isRunning()):
            self._recrop_then_build()
        else:
            self._build_tree()

    def _recrop_then_build(self) -> None:
        proj: CropProject = self.ctx["project"]
        self.tree.clear()
        self.progress.setRange(0, len(proj.scans))
        self.run_btn.setEnabled(False)
        self._crop_worker = CropWorker(proj, do_autodetect=False, do_recrop=True)
        self._crop_worker.progress.connect(
            lambda i, t, n: None if sip.isdeleted(self) else self.progress.setValue(i))
        self._crop_worker.failed.connect(
            lambda path, err: None if sip.isdeleted(self)
            else QMessageBox.warning(self, "Ошибка нарезки", f"{path}\n{err}"))
        self._crop_worker.finished_ok.connect(
            lambda: None if sip.isdeleted(self) else self._on_recrop_done())
        self._crop_worker.start()

    def _on_recrop_done(self) -> None:
        self.run_btn.setEnabled(True)
        self.progress.reset()
        self._build_tree()

    def _build_tree(self) -> None:
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
        for cp in crops:
            child = QTreeWidgetItem(parent, [self._wing_label(cp, cdir)])
            child.setData(0, _ROLE, ("wing", idx, str(cp)))

    def _meth_mark(self, cp, cdir, prof) -> str:
        """✓ all points, ● some, ○ none — for one methodology."""
        ann = load_annotation(annotation_path(cp, cdir, prof.methodology_id))
        if ann is None:
            return "○"
        done, total = ann.progress(prof.ids)
        return "✓" if done == total and total else "●"

    def _wing_label(self, cp, cdir) -> str:
        # Show both methodologies' status so it is always clear which is done.
        marks = []
        for mid, m in METHODOLOGIES.items():
            prof = get_profile(DEFAULT_PROFILE_PER_METHODOLOGY[mid])
            marks.append(f"{m.display_name[0]}:{self._meth_mark(cp, cdir, prof)}")
        return f"{cp.name}   {'  '.join(marks)}"

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
        self._worker.progress.connect(
            lambda i, t, name: None if sip.isdeleted(self) else self.progress.setFormat(name))
        self._worker.failed.connect(
            lambda path, err: None if sip.isdeleted(self)
            else QMessageBox.warning(self, "Ошибка разметки", f"{path}\n{err}"))
        self._worker.finished_ok.connect(self._done)
        self._worker.start()

    def stop_worker(self) -> None:
        """Interrupt and wait for background work so its signals can't reach a
        widget that is about to be destroyed."""
        for w in (self._worker, self._crop_worker):
            if w is not None and not sip.isdeleted(w) and w.isRunning():
                w.requestInterruption()
                w.wait(5000)
        mw = getattr(self.annot, "_ml_worker", None)
        if mw is not None and not sip.isdeleted(mw) and mw.isRunning():
            mw.requestInterruption()
            mw.wait(5000)

    def _done(self) -> None:
        if sip.isdeleted(self):
            return
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.run_btn.setEnabled(True)
        self._build_tree()      # refresh progress marks
        self._open_first_crop()  # land on the first crop with its new points

    def _open_first_crop(self) -> None:
        if self.tree.topLevelItemCount() == 0:
            return
        scan = self.tree.topLevelItem(0)
        self.tree.expandItem(scan)      # triggers _populate
        if scan.childCount() and scan.child(0).data(0, _ROLE):
            self.tree.setCurrentItem(scan.child(0))
            self._open_wing(scan.child(0))


class ProjectWindow(QMainWindow):
    def __init__(self, project: CropProject, on_home: Callable[[], None], parent=None):
        super().__init__(parent)
        self.project = project
        self._on_home = on_home
        self._disposing = False
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
        # Let pages drive navigation (e.g. crop page "done" -> Точки).
        self.ctx["goto_tab"] = self.tabs.setCurrentIndex
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

    def _stop_workers(self) -> None:
        for tab in (self.crop_tab, self.landmark_tab):
            stop = getattr(tab, "stop_worker", None)
            if stop is None:
                continue
            try:
                stop()
            except Exception:
                pass

    def dispose(self) -> None:
        """Called by the controller when this window is being replaced: stop
        threads first (so stale signals can't hit deleted widgets), then delete."""
        self._disposing = True
        self._stop_workers()
        self.close()
        self.deleteLater()

    def _save_quietly(self) -> None:
        try:
            self.project.save()
        except Exception:
            pass

    def _go_home(self) -> None:
        self._disposing = True
        self._stop_workers()
        self._save_quietly()
        self._on_home()
        self.close()

    def closeEvent(self, event) -> None:
        # Closing via the window's X (not the "← Дом" button): tidy up the same
        # way and fall back to the Home screen instead of leaving no window.
        self._stop_workers()
        self._save_quietly()
        if not self._disposing:
            self._disposing = True
            self._on_home()
        event.accept()
