"""Home screen: recent projects + open-folder entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)


class HomeScreen(QWidget):
    def __init__(self, recents: List[Dict],
                 on_open: Callable[[str], None], parent=None):
        super().__init__(parent)
        self.setObjectName("Home")
        self._on_open = on_open
        self._cards = 0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 32, 40, 32)
        outer.setSpacing(16)

        title = QLabel("BeeWings")
        title.setStyleSheet("font-size: 26px; font-weight: bold;")
        outer.addWidget(title)

        open_btn = QPushButton("\U0001f4c2  Открыть папку со сканами…")
        open_btn.clicked.connect(self._choose_folder)
        outer.addWidget(open_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        outer.addWidget(QLabel("Недавние проекты:"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        self._list = QVBoxLayout(host)
        self._list.setSpacing(10)
        self._list.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll, 1)

        if not recents:
            self._list.insertWidget(0, QLabel("Пока пусто — откройте папку, чтобы начать."))
        for item in recents:
            self._list.insertWidget(self._list.count() - 1, self._make_card(item))

    def cards_count(self) -> int:
        return self._cards

    def _emit_open(self, path: str) -> None:
        self._on_open(path)

    def _make_card(self, item: Dict) -> QFrame:
        self._cards += 1
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        name = QLabel(f"<b>{item.get('name','?')}</b>")
        prog = QLabel(f"нарезано {item.get('n_cropped',0)}/{item.get('n_splits',0)} · "
                      f"точки {item.get('n_landmarked',0)}")
        prog.setStyleSheet("color: #667;")
        path = QLabel(item.get("path", ""))
        path.setStyleSheet("color: #99a; font-size: 11px;")
        lay.addWidget(name)
        lay.addWidget(prog)
        lay.addWidget(path)
        row = QHBoxLayout()
        open_btn = QPushButton("Открыть")
        open_btn.clicked.connect(lambda _=False, p=item.get("path"): self._emit_open(p))
        row.addStretch(1)
        row.addWidget(open_btn)
        lay.addLayout(row)
        return card

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Папка проекта со сканами")
        if folder:
            self._on_open(folder)
