"""Standalone annotator window: a thin QMainWindow wrapper over AnnotatorWidget."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMainWindow, QStatusBar

from .annotator_widget import AnnotatorWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BeeWings — разметка крыльев")
        self.resize(1500, 900)
        self.annot = AnnotatorWidget(self)
        self.setCentralWidget(self.annot)
        self.setStatusBar(QStatusBar(self))
        self.annot.status.connect(self.statusBar().showMessage)
        self._build_menu()

    def load_folder(self, folder) -> None:
        self.annot.load_folder(Path(folder))

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&Файл")
        open_act = QAction("Открыть папку…", self)
        open_act.triggered.connect(self.annot.open_folder_dialog)
        file_menu.addAction(open_act)
        for label, slot in (("Экспорт CSV…", self.annot._export_csv),
                            ("Экспорт TPS…", self.annot._export_tps),
                            ("Экспорт COCO…", self.annot._export_coco),
                            ("Экспорт .dw.png…", self.annot._export_dw)):
            act = QAction(label, self)
            act.triggered.connect(slot)
            file_menu.addAction(act)
        quit_act = QAction("Выход", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)
        view_menu = self.menuBar().addMenu("&Вид")
        view_menu.addAction(self.annot._toggle_left_act)
        view_menu.addAction(self.annot._toggle_right_act)
        view_menu.addAction(self.annot._hide_both_act)

    def closeEvent(self, event) -> None:
        self.annot.save_current()
        event.accept()
