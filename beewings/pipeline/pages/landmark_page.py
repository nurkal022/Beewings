"""Stage 3: run ML landmarks, then open crops in the existing annotator."""
from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QVBoxLayout, QWidget)

from ...core.profiles import PROFILES
from ..project import CropProject
from ..workers import LandmarkWorker


class LandmarkPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker = None
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Этап 3. Расстановка точек:"))
        self.profile = QComboBox()
        self.profile.addItems(list(PROFILES.keys()))
        lay.addWidget(self.profile)
        self.run_btn = QPushButton("Расставить точки")
        self.run_btn.clicked.connect(self._run)
        lay.addWidget(self.run_btn)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.open_btn = QPushButton("Открыть в редакторе точек")
        self.open_btn.clicked.connect(self._open_editor)
        lay.addWidget(self.open_btn)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        lay.addWidget(self.cancel_btn)

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        idx = self.profile.findText(proj.settings.profile)
        if idx >= 0:
            self.profile.setCurrentIndex(idx)

    def _run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        proj: CropProject = self.ctx["project"]
        proj.settings.profile = self.profile.currentText()
        prof = PROFILES[proj.settings.profile]
        proj.settings.checkpoint = f"checkpoints/{prof.checkpoint_name}"
        proj.save()
        self.progress.setRange(0, 0)  # busy
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._worker = LandmarkWorker(proj)
        self._worker.failed.connect(
            lambda path, err: QMessageBox.warning(self, "Ошибка разметки", f"{path}\n{err}"))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.start()

    def _on_done(self) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()

    def _open_editor(self) -> None:
        proj: CropProject = self.ctx["project"]
        cb = self.ctx.get("open_in_annotator")
        if cb:
            cb(proj.root / "crops")

    def commit(self) -> bool:
        proj: CropProject = self.ctx["project"]
        proj.stage = "export"
        proj.save()
        return True
