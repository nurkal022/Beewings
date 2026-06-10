"""Stage 3: run ML landmarks, then open crops in the existing annotator."""
from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QLabel, QProgressBar, QPushButton,
                             QVBoxLayout, QWidget)

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

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        idx = self.profile.findText(proj.settings.profile)
        if idx >= 0:
            self.profile.setCurrentIndex(idx)

    def _run(self) -> None:
        proj: CropProject = self.ctx["project"]
        proj.settings.profile = self.profile.currentText()
        prof = PROFILES[proj.settings.profile]
        proj.settings.checkpoint = f"checkpoints/{prof.checkpoint_name}"
        proj.save()
        self.progress.setRange(0, 0)  # busy
        self._worker = LandmarkWorker(proj)
        self._worker.finished_ok.connect(lambda: self.progress.setRange(0, 1) or
                                         self.progress.setValue(1))
        self._worker.start()

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
