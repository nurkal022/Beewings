"""Right-side panel: landmark list, progress, metrics, enhancement controls."""
from __future__ import annotations

from typing import Callable, Optional

from typing import Dict

from PyQt6 import sip
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.metrics import compute_metrics
from ..core.profiles import (
    METHODOLOGIES, PROFILES, Profile, profiles_for_methodology,
)
from ..core.schema import WingAnnotation
from .enhancements import EnhanceParams


def _color_swatch(color_hex: str, size: int = 14) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(color_hex))
    return QIcon(pm)


class SidePanel(QWidget):
    landmark_selected = pyqtSignal(int)
    profile_changed = pyqtSignal(str)
    methodology_changed = pyqtSignal(str)   # emits methodology_id
    enhancement_changed = pyqtSignal(object)  # EnhanceParams
    template_toggled = pyqtSignal(bool)
    fit_view_requested = pyqtSignal()
    zoom_preset_requested = pyqtSignal(float)   # absolute zoom, e.g. 1.0, 2.0, 4.0
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    zoom_to_current_requested = pyqtSignal()
    crosshair_toggled = pyqtSignal(bool)
    delete_landmark_requested = pyqtSignal(int)
    toggle_uncertain_requested = pyqtSignal(int)
    toggle_skipped_requested = pyqtSignal(int)
    zoom_to_landmark_requested = pyqtSignal(int)
    auto_detect_requested = pyqtSignal()
    ml_detect_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll)
        content = QWidget()
        self._scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- methodology selector (the BIG, primary control)
        method_box = QGroupBox("Методика")
        method_layout = QVBoxLayout(method_box)
        method_layout.setContentsMargins(6, 8, 6, 6)
        method_layout.setSpacing(4)

        self.methodology_buttons: Dict[str, QPushButton] = {}
        radio_row = QHBoxLayout()
        radio_row.setSpacing(4)
        for mid, meth in METHODOLOGIES.items():
            btn = QPushButton(meth.display_name)
            btn.setCheckable(True)
            btn.setToolTip(f"{meth.description}\n\n{meth.reference}")
            btn.setStyleSheet(
                f"QPushButton {{ padding: 8px; font-weight: bold; }}"
                f"QPushButton:checked {{ background: {meth.accent_color}; color: white; }}"
            )
            btn.clicked.connect(lambda _checked=False, m=mid: self._on_methodology_clicked(m))
            self.methodology_buttons[mid] = btn
            radio_row.addWidget(btn)
        method_layout.addLayout(radio_row)

        self.methodology_info = QLabel("")
        self.methodology_info.setWordWrap(True)
        self.methodology_info.setMinimumHeight(46)
        self.methodology_info.setStyleSheet(
            "QLabel { color: #aaa; font-size: 10px; padding: 4px 2px 0 2px; }"
        )
        method_layout.addWidget(self.methodology_info)

        # Profile (sub-variant within the chosen methodology)
        prof_row = QHBoxLayout()
        prof_row.addWidget(QLabel("Подсхема:"))
        self.profile_combo = QComboBox()
        # Populated dynamically by set_methodology()
        self.profile_combo.currentTextChanged.connect(self._on_profile_combo_changed)
        prof_row.addWidget(self.profile_combo)
        method_layout.addLayout(prof_row)

        layout.addWidget(method_box)

        # --- progress
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m точек")
        layout.addWidget(self.progress)

        # --- auto detector
        self.ml_btn = QPushButton("🧠  ML авто-определить точки")
        self.ml_btn.setToolTip(
            "Запустить ML-модель (UNet heatmap regression). "
            "Точность ~2-3 px, работает на любой картинке. "
            "Требует файл checkpoint (.pt) — настраивается через переменную "
            "BEEWINGS_ML_CHECKPOINT или диалог Файл → Настройки ML."
        )
        self.ml_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; background: #2d4a8a; color: white; }")
        self.ml_btn.clicked.connect(self.ml_detect_requested.emit)
        layout.addWidget(self.ml_btn)

        self.auto_btn = QPushButton("⚡  Классический детектор")
        self.auto_btn.setToolTip(
            "Классическая компьютерная зрение (skeletonize + mean shape). "
            "Не требует обученной модели, работает на любом железе. "
            "Точность хуже ML, но не нужны GPU/checkpoint."
        )
        self.auto_btn.setStyleSheet("QPushButton { padding: 6px; background: #2d5a3d; color: white; }")
        self.auto_btn.clicked.connect(self.auto_detect_requested.emit)
        layout.addWidget(self.auto_btn)

        # --- landmark list
        self.list = QListWidget()
        self.list.itemClicked.connect(self._on_list_clicked)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._on_list_context_menu)
        self.list.setMinimumHeight(220)
        layout.addWidget(self.list, stretch=2)

        # --- view controls
        view_box = QGroupBox("Вид")
        view_layout = QVBoxLayout(view_box)

        # Zoom controls
        zoom_label_row = QHBoxLayout()
        zoom_label_row.addWidget(QLabel("Масштаб:"))
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("QLabel { font-weight: bold; color: #6bd76b; }")
        zoom_label_row.addWidget(self.zoom_label)
        zoom_label_row.addStretch(1)
        view_layout.addLayout(zoom_label_row)

        zoom_row = QHBoxLayout()
        for label, factor in (("Fit", 0.0), ("100%", 1.0), ("200%", 2.0), ("400%", 4.0), ("800%", 8.0)):
            btn = QPushButton(label)
            btn.setMaximumWidth(48)
            if factor == 0.0:
                btn.clicked.connect(self.fit_view_requested.emit)
            else:
                btn.clicked.connect(lambda _=False, f=factor: self.zoom_preset_requested.emit(f))
            zoom_row.addWidget(btn)
        view_layout.addLayout(zoom_row)

        zoom_btn_row = QHBoxLayout()
        zoom_out = QPushButton("− (Cmd−)")
        zoom_out.clicked.connect(self.zoom_out_requested.emit)
        zoom_in = QPushButton("+ (Cmd+)")
        zoom_in.clicked.connect(self.zoom_in_requested.emit)
        focus_pt = QPushButton("К текущей точке (Z)")
        focus_pt.clicked.connect(self.zoom_to_current_requested.emit)
        zoom_btn_row.addWidget(zoom_out)
        zoom_btn_row.addWidget(zoom_in)
        view_layout.addLayout(zoom_btn_row)
        view_layout.addWidget(focus_pt)

        self.crosshair_chk = QCheckBox("Прицел под курсором")
        self.crosshair_chk.setChecked(True)
        self.crosshair_chk.toggled.connect(self.crosshair_toggled.emit)
        view_layout.addWidget(self.crosshair_chk)

        self.template_chk = QCheckBox("Показывать предыдущее крыло как шаблон (T)")
        self.template_chk.toggled.connect(self.template_toggled.emit)
        view_layout.addWidget(self.template_chk)
        fit_btn = QPushButton("Сбросить масштаб (R)")
        fit_btn.clicked.connect(self.fit_view_requested.emit)
        view_layout.addWidget(fit_btn)
        layout.addWidget(view_box)

        # --- enhancement controls
        enh_box = QGroupBox("Улучшение изображения (только отображение)")
        enh_form = QFormLayout(enh_box)
        self.contrast = QDoubleSpinBox(); self.contrast.setRange(0.2, 3.0); self.contrast.setSingleStep(0.1); self.contrast.setValue(1.0)
        self.brightness = QSlider(Qt.Orientation.Horizontal); self.brightness.setRange(-100, 100); self.brightness.setValue(0)
        self.gamma = QDoubleSpinBox(); self.gamma.setRange(0.2, 3.0); self.gamma.setSingleStep(0.1); self.gamma.setValue(1.0)
        self.clahe = QCheckBox("CLAHE (адаптивное выравнивание гистограммы)")
        self.invert = QCheckBox("Инверсия")
        self.flip = QCheckBox("Зеркальное отражение по горизонтали (F)")
        for w in (self.contrast, self.gamma):
            w.valueChanged.connect(self._emit_enh)
        self.brightness.valueChanged.connect(self._emit_enh)
        for w in (self.clahe, self.invert, self.flip):
            w.toggled.connect(self._emit_enh)
        enh_form.addRow("Контраст", self.contrast)
        enh_form.addRow("Яркость", self.brightness)
        enh_form.addRow("Гамма", self.gamma)
        enh_form.addRow(self.clahe)
        enh_form.addRow(self.invert)
        enh_form.addRow(self.flip)
        layout.addWidget(enh_box)

        # --- metrics
        self.metrics_label = QLabel("Метрики: —")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("QLabel { background: #222; color: #ddd; padding: 6px; }")
        layout.addWidget(self.metrics_label)

        layout.addStretch(1)

        self._profile: Optional[Profile] = None
        self._current_id: int = 1

    # ---- helpers ------------------------------------------------------------

    def set_methodology(self, methodology_id: str, default_profile_name: str = "") -> None:
        """Update top radio + repopulate the profile dropdown for this methodology."""
        for mid, btn in self.methodology_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(mid == methodology_id)
            btn.blockSignals(False)
        meth = METHODOLOGIES[methodology_id]
        self.methodology_info.setText(f"{meth.description}\n[{meth.reference}]")
        # Repopulate profile combo
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for p in profiles_for_methodology(methodology_id):
            self.profile_combo.addItem(p.name)
        if default_profile_name:
            self.profile_combo.setCurrentText(default_profile_name)
        self.profile_combo.blockSignals(False)

    def set_profile(self, profile: Profile, ann: WingAnnotation, current_id: int) -> None:
        if sip.isdeleted(self):
            return
        self._profile = profile
        self._current_id = current_id
        # avoid feedback loop
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentText(profile.name)
        self.profile_combo.blockSignals(False)
        # Reflect methodology badge state
        self.set_methodology(profile.methodology_id, default_profile_name=profile.name)
        self._rebuild_list(ann)
        self._update_progress(ann)
        self._update_metrics(ann)

    def refresh(self, ann: WingAnnotation) -> None:
        if sip.isdeleted(self):
            return
        self._rebuild_list(ann)
        self._update_progress(ann)
        self._update_metrics(ann)

    def set_current(self, lm_id: int) -> None:
        self._current_id = lm_id
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == lm_id:
                self.list.setCurrentItem(item)
                break

    def get_enhancement(self) -> EnhanceParams:
        return EnhanceParams(
            contrast=self.contrast.value(),
            brightness=self.brightness.value(),
            gamma=self.gamma.value(),
            clahe=self.clahe.isChecked(),
            invert=self.invert.isChecked(),
            flip_h=self.flip.isChecked(),
        )

    def set_zoom_display(self, zoom: float) -> None:
        self.zoom_label.setText(f"{zoom * 100:.0f}%")

    def set_flip(self, flipped: bool) -> None:
        self.flip.blockSignals(True)
        self.flip.setChecked(flipped)
        self.flip.blockSignals(False)
        self._emit_enh()

    # ---- internals ----------------------------------------------------------

    def _on_methodology_clicked(self, methodology_id: str) -> None:
        # Visually lock to single selected button regardless of click toggle.
        for mid, btn in self.methodology_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(mid == methodology_id)
            btn.blockSignals(False)
        self.methodology_changed.emit(methodology_id)

    def _on_profile_combo_changed(self, name: str) -> None:
        if name:
            self.profile_changed.emit(name)

    def _emit_enh(self) -> None:
        self.enhancement_changed.emit(self.get_enhancement())

    def _on_list_clicked(self, item: QListWidgetItem) -> None:
        lm_id = item.data(Qt.ItemDataRole.UserRole)
        if lm_id is not None:
            self.landmark_selected.emit(int(lm_id))

    def _on_list_context_menu(self, pos: QPoint) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        lm_id = int(item.data(Qt.ItemDataRole.UserRole))
        menu = QMenu(self.list)

        act_select = QAction(f"Выбрать L{lm_id}", menu)
        act_select.triggered.connect(lambda: self.landmark_selected.emit(lm_id))
        menu.addAction(act_select)

        act_zoom = QAction(f"Приблизиться к L{lm_id}", menu)
        act_zoom.triggered.connect(lambda: self.zoom_to_landmark_requested.emit(lm_id))
        menu.addAction(act_zoom)

        menu.addSeparator()

        act_unc = QAction("Пометить как неуверенно (U)", menu)
        act_unc.triggered.connect(lambda: self.toggle_uncertain_requested.emit(lm_id))
        menu.addAction(act_unc)

        act_skip = QAction("Пометить как пропущенную (S)", menu)
        act_skip.triggered.connect(lambda: self.toggle_skipped_requested.emit(lm_id))
        menu.addAction(act_skip)

        menu.addSeparator()

        act_del = QAction(f"Удалить точку L{lm_id}", menu)
        act_del.triggered.connect(lambda: self.delete_landmark_requested.emit(lm_id))
        menu.addAction(act_del)

        menu.exec(self.list.viewport().mapToGlobal(pos))

    def _rebuild_list(self, ann: WingAnnotation) -> None:
        self.list.clear()
        if self._profile is None:
            return
        for spec in self._profile.landmarks:
            lm = ann.get_landmark(spec.id)
            mark = "○"
            extra = ""
            if lm is not None:
                if lm.skipped:
                    mark = "—"
                    extra = " [пропущена]"
                else:
                    mark = "●"
                    if lm.uncertain:
                        extra = " [неуверенно]"
            text = f"{mark}  {spec.id:>2}  {spec.label}{extra}"
            item = QListWidgetItem(_color_swatch(spec.color), text)
            item.setData(Qt.ItemDataRole.UserRole, spec.id)
            if spec.id == self._current_id:
                item.setBackground(QColor("#3a5a3a"))
            self.list.addItem(item)

    def _update_progress(self, ann: WingAnnotation) -> None:
        if self._profile is None:
            return
        done, total = ann.progress(self._profile.ids)
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _update_metrics(self, ann: WingAnnotation) -> None:
        m = compute_metrics(ann)
        lines = []
        for k, v in m.items():
            lines.append(f"{k}: {v:.3f}" if v is not None else f"{k}: —")
        self.metrics_label.setText("Метрики:\n" + "\n".join(lines))
