"""Stage 4: export the structured protocol (clean, card-based layout)."""
from __future__ import annotations

from pathlib import Path

from PyQt6 import sip
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from ..export import export_per_scan
from ..project import CropProject


class ExportPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        row = QHBoxLayout()
        row.addStretch(1)

        card = QFrame()
        card.setObjectName("card")
        card.setMaximumWidth(560)
        card.setMinimumWidth(440)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(22, 20, 22, 20)
        cl.setSpacing(14)

        title = QLabel("Экспорт протокола")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        cl.addWidget(title)

        self.info = QLabel("")
        self.info.setStyleSheet("color: #667;")
        cl.addWidget(self.info)

        contents = QLabel(
            "На каждый скан создаётся папка рядом со сканами:\n"
            "• сам скан  • crops/  • TPS  • Excel  • JSON (с метаданными)")
        contents.setWordWrap(True)
        contents.setStyleSheet("color: #99a; font-size: 12px;")
        cl.addWidget(contents)

        self.dest = QLabel("")
        self.dest.setStyleSheet("color: #99a; font-size: 11px;")
        cl.addWidget(self.dest)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.export_btn = QPushButton("Экспортировать")
        self.export_btn.setMinimumWidth(200)
        self.export_btn.clicked.connect(self._export)
        btn_row.addWidget(self.export_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setStyleSheet("color: #1a8a3a; font-weight: bold;")
        self.result.setVisible(False)
        cl.addWidget(self.result)

        row.addWidget(card)
        row.addStretch(1)
        root.addLayout(row)
        root.addStretch(1)

    def enter(self) -> None:
        if sip.isdeleted(self):
            return
        proj: CropProject = self.ctx["project"]
        prog = proj.progress()
        self.info.setText(
            f"Папка проекта: {Path(proj.root).name}  ·  "
            f"крыльев с точками: {prog['n_landmarked']} из {prog['n_splits']} сканов")
        self.dest.setText(f"Назначение:  {proj.root}  (папка на каждый скан)")
        self.result.setVisible(False)

    def _export(self) -> None:
        if sip.isdeleted(self):
            return
        proj: CropProject = self.ctx["project"]
        stats = export_per_scan(proj)
        if stats["n_scans"] == 0:
            self._show_result(
                "Нет размеченных крыльев — сначала расставьте точки.", ok=False)
            return
        self._show_result(
            f"✓ Готово: {stats['n_wings']} крыльев из {stats['n_scans']} "
            f"сканов → папки в {proj.root}", ok=True)

    def _show_result(self, text: str, ok: bool) -> None:
        # The slot may outlive its widgets (e.g. project window reopened); touching
        # a deleted QLabel raises RuntimeError on Windows / aborts on macOS.
        if sip.isdeleted(self) or sip.isdeleted(self.result):
            return
        color = "#1a8a3a" if ok else "#c0392b"
        self.result.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.result.setText(text)
        self.result.setVisible(True)

    def commit(self) -> bool:
        return True
