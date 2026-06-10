"""Stage 4: export the structured protocol."""
from __future__ import annotations

from PyQt6.QtWidgets import (QCheckBox, QLabel, QMessageBox, QPushButton,
                             QVBoxLayout, QWidget)

from ..export import export_protocol
from ..project import CropProject


class ExportPage(QWidget):
    def __init__(self, ctx: dict, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Этап 4. Экспорт протокола:"))
        self.cb_folders = QCheckBox("Структура папок (крылья + точки)")
        self.cb_summary = QCheckBox("Сводная таблица (CSV + индексы)")
        self.cb_tps = QCheckBox("TPS для морфометрии")
        self.cb_report = QCheckBox("Сводный отчёт")
        for cb in (self.cb_folders, self.cb_summary, self.cb_tps, self.cb_report):
            cb.setChecked(True)
            lay.addWidget(cb)
        btn = QPushButton("Экспортировать")
        btn.clicked.connect(self._export)
        lay.addWidget(btn)

    def enter(self) -> None:
        pass

    def _export(self) -> None:
        proj: CropProject = self.ctx["project"]
        include = set()
        if self.cb_folders.isChecked(): include.add("folders")
        if self.cb_summary.isChecked(): include.add("summary")
        if self.cb_tps.isChecked(): include.add("tps")
        if self.cb_report.isChecked(): include.add("report")
        stats = export_protocol(proj, proj.root / "export", include)
        QMessageBox.information(self, "Готово",
                               f"Экспортировано: {stats['n_wings']} крыльев "
                               f"из {stats['n_scans']} сканов в {proj.root / 'export'}")

    def commit(self) -> bool:
        return True
