"""QThread workers for the heavy pipeline steps (cropping, landmarks)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from .landmarks import run_landmarks
from .project import CropProject, auto_detect, recrop


class CropWorker(QThread):
    """Auto-detect boxes and/or recrop every scan in the project."""

    progress = pyqtSignal(int, int, str)   # done, total, name
    failed = pyqtSignal(str, str)          # scan path, error
    finished_ok = pyqtSignal()

    def __init__(self, project: CropProject, do_autodetect: bool, do_recrop: bool,
                 parent=None):
        super().__init__(parent)
        self.project = project
        self.do_autodetect = do_autodetect
        self.do_recrop = do_recrop

    def run(self) -> None:
        total = len(self.project.scans)
        for i, entry in enumerate(self.project.scans, start=1):
            if self.isInterruptionRequested():
                break
            try:
                if self.do_autodetect:
                    auto_detect(entry, self.project.settings)
                if self.do_recrop:
                    recrop(self.project, entry)
            except Exception as exc:  # keep going on the rest
                self.failed.emit(entry.path, f"{type(exc).__name__}: {exc}")
            self.progress.emit(i, total, Path(entry.path).name)
        self.project.save()
        self.finished_ok.emit()


class LandmarkWorker(QThread):
    """Run ML landmarks over every cropped scan's crops folder."""

    progress = pyqtSignal(int, int, str)
    failed = pyqtSignal(str, str)
    finished_ok = pyqtSignal()

    def __init__(self, project: CropProject, parent=None):
        super().__init__(parent)
        self.project = project

    def run(self) -> None:
        cropped = [e for e in self.project.scans if e.cropped]
        for entry in cropped:
            if self.isInterruptionRequested():
                break
            cdir = self.project.crops_dir(entry)
            try:
                run_landmarks(
                    cdir,
                    profile_name=self.project.settings.profile,
                    checkpoint=Path(self.project.settings.checkpoint),
                    # CPU: this is a background QThread; GPU (MPS/Metal) here
                    # contends with the main thread's canvas rendering and
                    # crashes natively on macOS.
                    device="cpu",
                    progress_cb=lambda i, t, name, e=entry: self.progress.emit(
                        i, t, f"{Path(e.path).stem}: {name}"),
                    should_cancel=self.isInterruptionRequested,
                )
            except Exception as exc:
                self.failed.emit(entry.path, f"{type(exc).__name__}: {exc}")
        self.finished_ok.emit()
