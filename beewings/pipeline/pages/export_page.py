"""Stage 4: export the structured protocol (clean, card-based layout)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QFrame, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                             QWidget)

from ..export import export_protocol
from ..project import CropProject

# (option-key, title, short description)
_OPTIONS = [
    ("folders", "Структура папок", "крылья + файлы точек по сканам"),
    ("summary", "Сводная таблица (CSV)", "координаты всех точек + индексы"),
    ("tps", "TPS для морфометрии", "совместимо с IdentiFly / MorphoJ"),
    ("report", "Сводный отчёт (JSON)", "сводка по партии и индексам"),
]


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

        box = QGroupBox("Что включить")
        grid = QGridLayout(box)
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(10)
        self.checks: dict[str, QCheckBox] = {}
        for r, (key, label, desc) in enumerate(_OPTIONS):
            cb = QCheckBox(label)
            cb.setChecked(True)
            d = QLabel(desc)
            d.setStyleSheet("color: #99a; font-size: 11px;")
            grid.addWidget(cb, r, 0)
            grid.addWidget(d, r, 1)
            self.checks[key] = cb
        grid.setColumnStretch(1, 1)
        cl.addWidget(box)

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

        # convenient aliases for callers/tests
        self.cb_folders = self.checks["folders"]
        self.cb_summary = self.checks["summary"]
        self.cb_tps = self.checks["tps"]
        self.cb_report = self.checks["report"]

    def enter(self) -> None:
        proj: CropProject = self.ctx["project"]
        prog = proj.progress()
        self.info.setText(
            f"Папка проекта: {Path(proj.root).name}  ·  "
            f"крыльев с точками: {prog['n_landmarked']} из {prog['n_splits']} сканов")
        self.dest.setText(f"Назначение:  {proj.root / 'export'}")
        self.result.setVisible(False)

    def _export(self) -> None:
        proj: CropProject = self.ctx["project"]
        include = {key for key, cb in self.checks.items() if cb.isChecked()}
        if not include:
            self.result.setStyleSheet("color: #c0392b; font-weight: bold;")
            self.result.setText("Выберите хотя бы один формат для экспорта.")
            self.result.setVisible(True)
            return
        stats = export_protocol(proj, proj.root / "export", include)
        self.result.setStyleSheet("color: #1a8a3a; font-weight: bold;")
        self.result.setText(
            f"✓ Готово: {stats['n_wings']} крыльев из {stats['n_scans']} "
            f"сканов → {proj.root / 'export'}")
        self.result.setVisible(True)

    def commit(self) -> bool:
        return True
