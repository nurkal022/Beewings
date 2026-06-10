"""Main application window: wires together canvas, panels, shortcuts, and IO."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.io_coco import export_coco
from ..core.io_csv import export_csv
from ..core.io_tps import export_tps
from ..core.profiles import DEFAULT_PROFILE, Profile, get_profile
from ..core.schema import (
    Landmark,
    WingAnnotation,
    annotation_path,
    load_annotation,
    save_annotation,
)
from .canvas import WingCanvas
from .enhancements import EnhanceParams
from .image_list import ImageList
from .magnifier import Magnifier
from .side_panel import SidePanel


class UndoStack:
    """Per-image undo/redo for landmark operations.

    Each entry is a full snapshot of the WingAnnotation's landmarks list.
    Cheap because there are at most 19 of them.
    """
    def __init__(self):
        self._stack: List[List[Landmark]] = []
        self._redo: List[List[Landmark]] = []

    def reset(self, landmarks: List[Landmark]) -> None:
        self._stack = [self._snap(landmarks)]
        self._redo.clear()

    def push(self, landmarks: List[Landmark]) -> None:
        self._stack.append(self._snap(landmarks))
        self._redo.clear()

    def undo(self) -> Optional[List[Landmark]]:
        if len(self._stack) < 2:
            return None
        cur = self._stack.pop()
        self._redo.append(cur)
        return [Landmark.model_validate(lm.model_dump()) for lm in self._stack[-1]]

    def redo(self) -> Optional[List[Landmark]]:
        if not self._redo:
            return None
        item = self._redo.pop()
        self._stack.append(item)
        return [Landmark.model_validate(lm.model_dump()) for lm in item]

    @staticmethod
    def _snap(landmarks: List[Landmark]) -> List[Landmark]:
        return [Landmark.model_validate(lm.model_dump()) for lm in landmarks]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BeeWings — разметка крыльев")
        self.resize(1500, 900)

        self._project_dir: Optional[Path] = None
        self._profile: Profile = get_profile(DEFAULT_PROFILE)
        self._current_image_path: Optional[Path] = None
        self._current_ann: Optional[WingAnnotation] = None
        self._current_lm_id: int = 1
        self._undo = UndoStack()
        self._annotator_name = os.environ.get("USER", "anon")
        self._template_enabled = False
        self._cursor_xy: Optional[tuple] = None
        self._ml_model = None
        self._ml_ckpt_path: Optional[str] = None

        # ---- widgets
        self.image_list = ImageList()
        self.canvas = WingCanvas()
        self.side = SidePanel()
        self.magnifier = Magnifier(self)
        self.magnifier.hide()

        # central layout: [image list | canvas | side panel]
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self.image_list)
        self._splitter.addWidget(self.canvas)
        self._splitter.addWidget(self.side)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([240, 900, 360])
        self._splitter.setChildrenCollapsible(True)
        # Remember sizes before collapse so we can restore them on toggle.
        self._left_last_size = 240
        self._right_last_size = 360
        self.setCentralWidget(self._splitter)

        self.setStatusBar(QStatusBar(self))

        self._build_menu()
        self._build_toolbar()
        self._wire_signals()
        self._install_shortcuts()

        # initial profile combo state
        self.side.profile_combo.setCurrentText(self._profile.name)

    # ---- menu ---------------------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&Файл")

        open_act = QAction("Открыть папку…", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_act)

        pipeline_act = QAction("🧩 Конвейер нарезки…", self)
        pipeline_act.triggered.connect(self._on_open_pipeline)
        file_menu.addAction(pipeline_act)

        file_menu.addSeparator()

        for fmt, handler in (
            ("Экспорт в CSV…", self._export_csv),
            ("Экспорт в TPS…", self._export_tps),
            ("Экспорт в COCO keypoints…", self._export_coco),
        ):
            act = QAction(fmt, self)
            act.triggered.connect(handler)
            file_menu.addAction(act)

        file_menu.addSeparator()
        quit_act = QAction("Выход", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = self.menuBar().addMenu("&Вид")
        self._toggle_left_act = QAction("Левая панель (список изображений)", self)
        self._toggle_left_act.setCheckable(True)
        self._toggle_left_act.setChecked(True)
        self._toggle_left_act.setShortcut(QKeySequence("Ctrl+\\"))
        self._toggle_left_act.toggled.connect(self._toggle_left_panel)
        view_menu.addAction(self._toggle_left_act)

        self._toggle_right_act = QAction("Правая панель (точки и настройки)", self)
        self._toggle_right_act.setCheckable(True)
        self._toggle_right_act.setChecked(True)
        self._toggle_right_act.setShortcut(QKeySequence("Ctrl+]"))
        self._toggle_right_act.toggled.connect(self._toggle_right_panel)
        view_menu.addAction(self._toggle_right_act)

        view_menu.addSeparator()
        self._hide_both_act = QAction("Скрыть обе панели (максимум канваса)", self)
        self._hide_both_act.setShortcut(QKeySequence("F11"))
        self._hide_both_act.triggered.connect(self._toggle_both_panels)
        view_menu.addAction(self._hide_both_act)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Панели", self)
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        # Left-panel toggle: arrow points "in" (collapse) when visible, "out" when hidden.
        self._left_btn = QToolButton()
        self._left_btn.setCheckable(True)
        self._left_btn.setChecked(True)
        self._left_btn.setToolTip("Свернуть/развернуть список изображений  (Ctrl+\\)")
        self._left_btn.toggled.connect(self._toggle_left_act.setChecked)
        self._toggle_left_act.toggled.connect(self._sync_left_btn)
        self._sync_left_btn(True)
        tb.addWidget(self._left_btn)

        tb.addSeparator()

        # Center fullscreen toggle.
        self._full_btn = QToolButton()
        self._full_btn.setText("⛶  Скрыть обе панели  (F11)")
        self._full_btn.setToolTip("Скрыть обе боковые панели для максимума канваса")
        self._full_btn.clicked.connect(self._toggle_both_panels)
        tb.addWidget(self._full_btn)

        tb.addSeparator()

        # Right-panel toggle.
        self._right_btn = QToolButton()
        self._right_btn.setCheckable(True)
        self._right_btn.setChecked(True)
        self._right_btn.setToolTip("Свернуть/развернуть панель точек  (Ctrl+])")
        self._right_btn.toggled.connect(self._toggle_right_act.setChecked)
        self._toggle_right_act.toggled.connect(self._sync_right_btn)
        self._sync_right_btn(True)
        tb.addWidget(self._right_btn)

    def _sync_left_btn(self, visible: bool) -> None:
        self._left_btn.blockSignals(True)
        self._left_btn.setChecked(visible)
        self._left_btn.setText("◀  Список изображений" if visible else "▶  Список изображений")
        self._left_btn.blockSignals(False)

    def _sync_right_btn(self, visible: bool) -> None:
        self._right_btn.blockSignals(True)
        self._right_btn.setChecked(visible)
        self._right_btn.setText("Точки и настройки  ▶" if visible else "Точки и настройки  ◀")
        self._right_btn.blockSignals(False)

    # ---- signal wiring ------------------------------------------------------

    def _wire_signals(self) -> None:
        self.image_list.image_selected.connect(self._on_image_selected)
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.landmark_clicked.connect(self._on_landmark_clicked)
        self.canvas.cursor_moved.connect(self._on_cursor_moved)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.side.landmark_selected.connect(self._set_current_id)
        self.side.profile_changed.connect(self._on_profile_changed)
        self.side.methodology_changed.connect(self._on_methodology_changed)
        self.side.enhancement_changed.connect(self._on_enhancement_changed)
        self.side.template_toggled.connect(self._on_template_toggled)
        self.side.fit_view_requested.connect(self.canvas.fit_view)
        self.side.zoom_in_requested.connect(lambda: self.canvas.zoom_by(1.25))
        self.side.zoom_out_requested.connect(lambda: self.canvas.zoom_by(1 / 1.25))
        self.side.zoom_preset_requested.connect(lambda f: self.canvas.set_zoom(f))
        self.side.zoom_to_current_requested.connect(self._zoom_to_current)
        self.side.crosshair_toggled.connect(self.canvas.set_crosshair_enabled)
        self.side.delete_landmark_requested.connect(self._delete_landmark_by_id)
        self.side.toggle_uncertain_requested.connect(self._toggle_uncertain_id)
        self.side.toggle_skipped_requested.connect(self._toggle_skipped_id)
        self.side.zoom_to_landmark_requested.connect(lambda lid: self.canvas.zoom_to_landmark(lid, factor=8.0))
        self.side.auto_detect_requested.connect(self._run_auto_detect)
        self.side.ml_detect_requested.connect(self._run_ml_detect)
        self.canvas.landmark_context_menu.connect(self._show_landmark_context_menu)

    def _install_shortcuts(self) -> None:
        # number keys 1..9 -> select landmark id
        for n in range(1, 10):
            QShortcut(QKeySequence(str(n)), self, activated=lambda nn=n: self._set_current_id(nn))
        # 0 -> id 10
        QShortcut(QKeySequence("0"), self, activated=lambda: self._set_current_id(10))
        # Shift+1..9 -> id 11..19
        for n in range(1, 10):
            QShortcut(QKeySequence(f"Shift+{n}"), self, activated=lambda nn=n: self._set_current_id(10 + nn))

        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self._step_current_id(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._step_current_id(+1))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, activated=lambda: self.image_list.select_offset(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, activated=lambda: self.image_list.select_offset(+1))

        QShortcut(QKeySequence("Backspace"), self, activated=self._delete_current)
        QShortcut(QKeySequence("Delete"), self, activated=self._delete_current)
        QShortcut(QKeySequence("U"), self, activated=self._toggle_uncertain)
        QShortcut(QKeySequence("S"), self, activated=self._toggle_skipped)

        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo_action)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._redo_action)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo_action)

        QShortcut(QKeySequence("R"), self, activated=self.canvas.fit_view)
        QShortcut(QKeySequence("T"), self, activated=lambda: self.side.template_chk.toggle())
        QShortcut(QKeySequence("F"), self, activated=lambda: self.side.flip.toggle())

        # Zoom shortcuts — both QKeySequence.StandardKey and explicit variants.
        QShortcut(QKeySequence.StandardKey.ZoomIn, self, activated=lambda: self.canvas.zoom_by(1.25))
        QShortcut(QKeySequence.StandardKey.ZoomOut, self, activated=lambda: self.canvas.zoom_by(1 / 1.25))
        QShortcut(QKeySequence("Ctrl++"), self, activated=lambda: self.canvas.zoom_by(1.25))
        QShortcut(QKeySequence("Ctrl+="), self, activated=lambda: self.canvas.zoom_by(1.25))
        QShortcut(QKeySequence("Ctrl+-"), self, activated=lambda: self.canvas.zoom_by(1 / 1.25))
        QShortcut(QKeySequence("Meta++"), self, activated=lambda: self.canvas.zoom_by(1.25))
        QShortcut(QKeySequence("Meta+="), self, activated=lambda: self.canvas.zoom_by(1.25))
        QShortcut(QKeySequence("Meta+-"), self, activated=lambda: self.canvas.zoom_by(1 / 1.25))
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.canvas.fit_view)
        QShortcut(QKeySequence("Meta+0"), self, activated=self.canvas.fit_view)
        QShortcut(QKeySequence("Ctrl+1"), self, activated=lambda: self.canvas.set_zoom(1.0))
        QShortcut(QKeySequence("Meta+1"), self, activated=lambda: self.canvas.set_zoom(1.0))
        QShortcut(QKeySequence("Z"), self, activated=self._zoom_to_current)

        # Sub-pixel nudge of the currently selected landmark (Alt+arrows = 1px,
        # Alt+Shift+arrows = 0.1px). Lets the annotator make microscopic
        # corrections without dragging when zoomed in deeply.
        QShortcut(QKeySequence("Alt+Left"),  self, activated=lambda: self._nudge_current(-1, 0, fine=False))
        QShortcut(QKeySequence("Alt+Right"), self, activated=lambda: self._nudge_current(+1, 0, fine=False))
        QShortcut(QKeySequence("Alt+Up"),    self, activated=lambda: self._nudge_current(0, -1, fine=False))
        QShortcut(QKeySequence("Alt+Down"),  self, activated=lambda: self._nudge_current(0, +1, fine=False))
        QShortcut(QKeySequence("Alt+Shift+Left"),  self, activated=lambda: self._nudge_current(-1, 0, fine=True))
        QShortcut(QKeySequence("Alt+Shift+Right"), self, activated=lambda: self._nudge_current(+1, 0, fine=True))
        QShortcut(QKeySequence("Alt+Shift+Up"),    self, activated=lambda: self._nudge_current(0, -1, fine=True))
        QShortcut(QKeySequence("Alt+Shift+Down"),  self, activated=lambda: self._nudge_current(0, +1, fine=True))

    # ---- folder / image loading --------------------------------------------

    def _on_open_pipeline(self) -> None:
        from ..pipeline.wizard import PipelineWizard
        self._pipeline = PipelineWizard()
        # Let the wizard hand a finished crops folder back to this annotator.
        self._pipeline.ctx["open_in_annotator"] = self._open_crops_from_pipeline
        self._pipeline.show()

    def _open_crops_from_pipeline(self, crops_root) -> None:
        from pathlib import Path
        crops_root = Path(crops_root)
        # crops_root contains one subfolder per scan; open the first that exists,
        # or the crops_root itself if it directly holds images.
        subdirs = [d for d in crops_root.glob("*") if d.is_dir()]
        target = subdirs[0] if subdirs else crops_root
        self.load_folder(target)
        if hasattr(self, "_pipeline"):
            self._pipeline.raise_()

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder:
            return
        self.load_folder(Path(folder))

    def load_folder(self, folder: Path) -> None:
        # Persist any pending work in the OLD project before switching, then
        # forget the old image so the next image-selected event doesn't try
        # to save under the new project_dir.
        self._save_current()
        self._current_ann = None
        self._current_image_path = None
        self._project_dir = folder
        # Migrate legacy annotations (pre-methodology-split layout) once.
        try:
            from ..core.schema import migrate_legacy_annotations
            report = migrate_legacy_annotations(folder)
            if report["moved"]:
                self.statusBar().showMessage(
                    f"Старые аннотации перемещены в папки методик: "
                    f"перемещено {report['moved']}"
                )
        except Exception:
            pass
        self.image_list.load_folder(folder, self._profile)
        self.statusBar().showMessage(f"Открыта папка: {folder}")

    def _on_image_selected(self, path: Path) -> None:
        self._save_current()
        self._load_image(path)

    def _load_image(self, path: Path) -> None:
        if self._project_dir is None:
            return
        # robust unicode-safe imread
        with path.open("rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Ошибка загрузки", f"Не удалось прочитать файл:\n{path}")
            return
        self._current_image_path = path
        h, w = img.shape[:2]

        ann_path = annotation_path(path, self._project_dir, self._profile.methodology_id)
        ann = load_annotation(ann_path)
        if ann is None:
            ann = WingAnnotation(
                image=path.name,
                image_size=(w, h),
                profile=self._profile.name,
                annotator=self._annotator_name,
            )
        else:
            ann.image_size = (w, h)
            if ann.profile != self._profile.name:
                ann.profile = self._profile.name

        self._current_ann = ann
        self._undo.reset(ann.landmarks)

        # restore flip from annotation
        self.side.set_flip(ann.flipped)

        self.canvas.load_image(img)
        self.canvas.set_enhancement(self.side.get_enhancement())
        self.canvas.set_annotation(ann, self._profile, on_moved=self._on_landmark_dragged)

        # start at first unset landmark
        first_unset = next(
            (i for i in self._profile.ids
             if (lm := ann.get_landmark(i)) is None or lm.skipped),
            self._profile.ids[0],
        )
        self._set_current_id(first_unset)

        self.side.set_profile(self._profile, ann, self._current_lm_id)
        self._apply_template()
        self.statusBar().showMessage(f"{path.name}  ({w}×{h} пикс.)")

    def _save_current(self) -> None:
        if self._current_ann is None or self._current_image_path is None or self._project_dir is None:
            return
        self._current_ann.flipped = self.side.flip.isChecked()
        save_annotation(self._current_ann, annotation_path(self._current_image_path, self._project_dir, self._profile.methodology_id))
        self.image_list.refresh_progress(self._profile)

    # ---- canvas interactions -----------------------------------------------

    def _on_canvas_clicked(self, x: float, y: float) -> None:
        if self._current_ann is None:
            return
        # Account for horizontal flip: stored coords stay in original image space.
        x = self._unflip_x(x)
        self._undo.push(self._current_ann.landmarks)
        self._current_ann.set_landmark(self._current_lm_id, x, y)
        self._after_change(advance=True)

    def _on_landmark_clicked(self, lm_id: int) -> None:
        self._set_current_id(lm_id)

    def _on_landmark_dragged(self, lm_id: int, x: float, y: float) -> None:
        if self._current_ann is None:
            return
        lm = self._current_ann.get_landmark(lm_id)
        if lm is None:
            return
        if abs(lm.x - self._unflip_x(x)) < 0.01 and abs(lm.y - y) < 0.01:
            return
        self._undo.push(self._current_ann.landmarks)
        self._current_ann.set_landmark(lm_id, self._unflip_x(x), y)
        self._after_change(advance=False, rebuild=False)

    def _on_cursor_moved(self, x: float, y: float) -> None:
        img = self.canvas.get_enhanced_image()
        if img is None:
            self.magnifier.hide()
            return
        self.magnifier.update_view(img, x, y)
        # position magnifier in top-right corner of canvas viewport
        if not self.magnifier.isVisible():
            self.magnifier.show()
        canvas_top_right = self.canvas.mapToGlobal(QPoint(self.canvas.viewport().width() - self.magnifier.width() - 12, 12))
        win_pos = self.mapFromGlobal(canvas_top_right)
        self.magnifier.move(win_pos)
        self._update_status()
        self._cursor_xy = (x, y)

    def _on_zoom_changed(self, zoom: float) -> None:
        self.side.set_zoom_display(zoom)
        self._update_status()

    def _zoom_to_current(self) -> None:
        if self._current_ann is None:
            return
        lm = self._current_ann.get_landmark(self._current_lm_id)
        if lm is None or lm.skipped or lm.x < 0:
            # No placed landmark yet — zoom to image center at 4x.
            self.canvas.set_zoom(4.0)
            return
        self.canvas.zoom_to_landmark(self._current_lm_id, factor=8.0)

    def _nudge_current(self, dx: int, dy: int, fine: bool) -> None:
        if self._current_ann is None:
            return
        lm = self._current_ann.get_landmark(self._current_lm_id)
        if lm is None or lm.skipped or lm.x < 0:
            return
        step = 0.1 if fine else 1.0
        new_x = max(0.0, min(self._current_ann.image_size[0] - 1, lm.x + dx * step))
        new_y = max(0.0, min(self._current_ann.image_size[1] - 1, lm.y + dy * step))
        if abs(new_x - lm.x) < 1e-6 and abs(new_y - lm.y) < 1e-6:
            return
        self._undo.push(self._current_ann.landmarks)
        self._current_ann.set_landmark(self._current_lm_id, new_x, new_y)
        self._after_change(advance=False)

    def _update_status(self) -> None:
        z = self.canvas.current_zoom()
        msg = f"Масштаб: {z * 100:.0f}%"
        if hasattr(self, "_cursor_xy") and self._cursor_xy is not None:
            x, y = self._cursor_xy
            if self._current_ann is not None:
                w, h = self._current_ann.image_size
                if 0 <= x < w and 0 <= y < h:
                    msg += f"   ·   X={x:.1f}  Y={y:.1f}"
        if self._current_image_path is not None:
            msg = f"{self._current_image_path.name}   ·   {msg}   ·   точка L{self._current_lm_id}"
        self.statusBar().showMessage(msg)

    # ---- current id management ---------------------------------------------

    def _set_current_id(self, lm_id: int) -> None:
        if lm_id not in self._profile.ids:
            return
        self._current_lm_id = lm_id
        self.canvas.set_current_id(lm_id)
        self.side.set_current(lm_id)
        if self._current_ann is not None:
            self.side.refresh(self._current_ann)
        self._update_status()

    def _step_current_id(self, delta: int) -> None:
        ids = self._profile.ids
        if not ids:
            return
        try:
            idx = ids.index(self._current_lm_id)
        except ValueError:
            idx = 0
        new = ids[(idx + delta) % len(ids)]
        self._set_current_id(new)

    def _advance_to_next_unset(self) -> None:
        if self._current_ann is None:
            return
        ids = self._profile.ids
        try:
            idx = ids.index(self._current_lm_id)
        except ValueError:
            idx = 0
        for off in range(1, len(ids) + 1):
            cand = ids[(idx + off) % len(ids)]
            lm = self._current_ann.get_landmark(cand)
            if lm is None or lm.skipped or (lm.x < 0 and lm.y < 0):
                self._set_current_id(cand)
                return
        # all set — stay on current
        self._step_current_id(+1)

    # ---- key actions --------------------------------------------------------

    def _delete_current(self) -> None:
        if self._current_ann is None:
            return
        self._undo.push(self._current_ann.landmarks)
        self._current_ann.remove_landmark(self._current_lm_id)
        self._after_change(advance=False)

    def _toggle_uncertain(self) -> None:
        if self._current_ann is None:
            return
        self._undo.push(self._current_ann.landmarks)
        self._current_ann.toggle_uncertain(self._current_lm_id)
        self._after_change(advance=False)

    def _toggle_skipped(self) -> None:
        if self._current_ann is None:
            return
        self._undo.push(self._current_ann.landmarks)
        lm = self._current_ann.get_landmark(self._current_lm_id)
        currently_skipped = lm is not None and lm.skipped
        self._current_ann.mark_skipped(self._current_lm_id, not currently_skipped)
        self._after_change(advance=not currently_skipped)

    def _undo_action(self) -> None:
        if self._current_ann is None:
            return
        prev = self._undo.undo()
        if prev is None:
            return
        self._current_ann.landmarks = prev
        self._after_change(advance=False, push_undo=False)

    def _redo_action(self) -> None:
        if self._current_ann is None:
            return
        nxt = self._undo.redo()
        if nxt is None:
            return
        self._current_ann.landmarks = nxt
        self._after_change(advance=False, push_undo=False)

    # ---- side panel -> state -----------------------------------------------

    def _on_profile_changed(self, name: str) -> None:
        new_profile = get_profile(name)
        # If we're changing methodology, save current to its own folder first,
        # then load the OTHER methodology's annotation for this image (or empty).
        prev_methodology = self._profile.methodology_id if self._profile else None
        self._save_current()                 # save under the old methodology
        self._profile = new_profile
        # Cached ML model becomes invalid since checkpoint differs.
        if prev_methodology != new_profile.methodology_id:
            self._ml_model = None
            self._ml_ckpt_path = None
        # Reload the current image so we pick up the right per-methodology
        # annotation file (or start fresh in this methodology).
        if self._current_image_path is not None:
            self._load_image(self._current_image_path)
        elif self._current_ann is not None:
            self._current_ann.profile = name
        self.side.set_methodology(new_profile.methodology_id, new_profile.name)
        self.side.set_profile(new_profile, self._current_ann or WingAnnotation(
            image="", image_size=(0, 0), profile=name), self._current_lm_id)
        if self._project_dir is not None:
            self.image_list.refresh_progress(self._profile)
        self.statusBar().showMessage(
            f"Методика: {new_profile.methodology.display_name} · "
            f"Подсхема: {new_profile.name}"
        )

    def _on_methodology_changed(self, methodology_id: str) -> None:
        """User clicked the methodology radio button at the top of the side panel."""
        from ..core.profiles import DEFAULT_PROFILE_PER_METHODOLOGY
        target_profile = DEFAULT_PROFILE_PER_METHODOLOGY.get(methodology_id)
        if target_profile and target_profile != self._profile.name:
            self._on_profile_changed(target_profile)

    def _on_enhancement_changed(self, params: EnhanceParams) -> None:
        self.canvas.set_enhancement(params)
        if self._current_ann is not None:
            self._current_ann.flipped = params.flip_h

    def _on_template_toggled(self, enabled: bool) -> None:
        self._template_enabled = enabled
        self._apply_template()

    def _apply_template(self) -> None:
        if not self._template_enabled or self._project_dir is None:
            self.canvas.clear_template()
            return
        prev = self.image_list.previous_path()
        if prev is None:
            self.canvas.clear_template()
            return
        ann = load_annotation(annotation_path(prev, self._project_dir, self._profile.methodology_id))
        self.canvas.set_template(ann.landmarks if ann else [])

    # ---- common change handler ---------------------------------------------

    def _after_change(self, advance: bool, rebuild: bool = True, push_undo: bool = True) -> None:
        if self._current_ann is None:
            return
        if rebuild:
            self.canvas.refresh_landmarks()
            self.canvas.set_current_id(self._current_lm_id)
        self.side.refresh(self._current_ann)
        self._save_current()
        if advance:
            self._advance_to_next_unset()

    def _unflip_x(self, x: float) -> float:
        # We do not actually flip the underlying image data — flip is a display
        # transform applied in enhancements.apply(). However QGraphicsView's
        # scene coordinates are the *displayed* pixmap, so when the image is
        # flipped horizontally, clicks come in flipped too. Mirror them back
        # so stored coordinates remain in the original image frame.
        if self._current_ann is None or not self.side.flip.isChecked():
            return x
        w = self._current_ann.image_size[0]
        return w - x

    # ---- exports ------------------------------------------------------------

    def _collect_all_annotations(self) -> List[WingAnnotation]:
        if self._project_dir is None:
            return []
        out = []
        for p in self.image_list.all_paths():
            ann = load_annotation(annotation_path(p, self._project_dir, self._profile.methodology_id))
            if ann is not None:
                out.append(ann)
        return out

    def _export_csv(self) -> None:
        self._save_current()
        out, _ = QFileDialog.getSaveFileName(self, "Экспорт в CSV", "landmarks.csv", "CSV (*.csv)")
        if not out:
            return
        export_csv(self._collect_all_annotations(), Path(out))
        self.statusBar().showMessage(f"Сохранено: {out}")

    def _export_tps(self) -> None:
        self._save_current()
        out, _ = QFileDialog.getSaveFileName(self, "Экспорт в TPS", "landmarks.tps", "TPS (*.tps)")
        if not out:
            return
        export_tps(self._collect_all_annotations(), Path(out))
        self.statusBar().showMessage(f"Сохранено: {out}")

    def _export_coco(self) -> None:
        self._save_current()
        if self._project_dir is None:
            return
        out, _ = QFileDialog.getSaveFileName(self, "Экспорт в COCO", "landmarks_coco.json", "JSON (*.json)")
        if not out:
            return
        export_coco(self._collect_all_annotations(), Path(out), self._project_dir, self._profile.name)
        self.statusBar().showMessage(f"Сохранено: {out}")

    def closeEvent(self, event) -> None:
        self._save_current()
        super().closeEvent(event)

    # ---- panel collapse / per-landmark actions ------------------------------

    def _toggle_left_panel(self, visible: bool) -> None:
        sizes = self._splitter.sizes()
        if visible:
            sizes[0] = max(self._left_last_size, 160)
        else:
            if sizes[0] > 0:
                self._left_last_size = sizes[0]
            sizes[0] = 0
        self._splitter.setSizes(sizes)

    def _toggle_right_panel(self, visible: bool) -> None:
        sizes = self._splitter.sizes()
        if visible:
            sizes[2] = max(self._right_last_size, 200)
        else:
            if sizes[2] > 0:
                self._right_last_size = sizes[2]
            sizes[2] = 0
        self._splitter.setSizes(sizes)

    def _toggle_both_panels(self) -> None:
        both_hidden = not (self._toggle_left_act.isChecked() or self._toggle_right_act.isChecked())
        new_state = both_hidden  # if both hidden -> show both; else hide both
        self._toggle_left_act.setChecked(new_state)
        self._toggle_right_act.setChecked(new_state)

    def _delete_landmark_by_id(self, lm_id: int) -> None:
        if self._current_ann is None:
            return
        if self._current_ann.get_landmark(lm_id) is None:
            return
        self._undo.push(self._current_ann.landmarks)
        self._current_ann.remove_landmark(lm_id)
        if lm_id == self._current_lm_id:
            self.canvas.refresh_landmarks()
            self.canvas.set_current_id(self._current_lm_id)
        self._after_change(advance=False)

    def _toggle_uncertain_id(self, lm_id: int) -> None:
        if self._current_ann is None:
            return
        self._undo.push(self._current_ann.landmarks)
        self._current_ann.toggle_uncertain(lm_id)
        self._after_change(advance=False)

    def _toggle_skipped_id(self, lm_id: int) -> None:
        if self._current_ann is None:
            return
        self._undo.push(self._current_ann.landmarks)
        lm = self._current_ann.get_landmark(lm_id)
        self._current_ann.mark_skipped(lm_id, not (lm is not None and lm.skipped))
        self._after_change(advance=False)

    def _run_ml_detect(self) -> None:
        """Run trained UNet ML detector on the current image."""
        if self._current_ann is None or self._current_image_path is None or self._project_dir is None:
            return
        import os
        # Each profile uses its OWN checkpoint — Alpatov and Tofilski are
        # separate methodologies, never share weights.
        ckpt_filename = self._profile.checkpoint_name or "tofilski19.pt"

        env_ckpt = os.environ.get("BEEWINGS_ML_CHECKPOINT", "").strip()
        ckpt_candidates: list[Path] = []
        if env_ckpt:
            ckpt_candidates.append(Path(env_ckpt))
        ckpt_candidates.extend([
            self._project_dir / "checkpoints" / ckpt_filename,
            Path("checkpoints") / ckpt_filename,
            # back-compat: legacy default name + old training dir
            Path("checkpoints/best.pt") if ckpt_filename == "tofilski19.pt" else Path(""),
            Path("runs/unet19_v1/best.pt") if ckpt_filename == "tofilski19.pt" else Path(""),
            Path("runs/unet12_v1/best.pt") if ckpt_filename == "alpatov12.pt" else Path(""),
        ])
        ckpt = next((p for p in ckpt_candidates if str(p) and p.is_file()), None)
        if ckpt is None:
            QMessageBox.warning(
                self, "Модель не найдена",
                f"Для профиля «{self._profile.name}» нужен checkpoint "
                f"{ckpt_filename}.\n\n"
                f"Положи файл в один из путей:\n"
                f"  • checkpoints/{ckpt_filename}\n"
                f"  • <папка_проекта>/checkpoints/{ckpt_filename}\n"
                f"  • или укажи через BEEWINGS_ML_CHECKPOINT env var\n\n"
                f"Алпатов и Тофильский — разные методики, у каждой своя модель."
            )
            return

        # Sanity check: loaded model must match profile point count.
        # We discover this by peeking at the checkpoint config.
        try:
            import torch
            peek = torch.load(ckpt, map_location="cpu", weights_only=False)
            ckpt_n = peek["config"]["n_points"]
            expected_n = len(self._profile.ids)
            # 8-point Alpatov is served by the 12-point model; we'll just take first 8.
            if ckpt_n != expected_n and not (
                self._profile.methodology_id == "alpatov" and ckpt_n == 12 and expected_n == 8
            ):
                QMessageBox.critical(
                    self, "Несовпадение схемы",
                    f"Checkpoint {ckpt.name} обучен на {ckpt_n} точек, "
                    f"а профиль «{self._profile.name}» ожидает {expected_n}. "
                    f"Это разные методики — нельзя смешивать."
                )
                return
        except Exception:
            pass  # if peek fails, let load_ml below produce the real error
        try:
            from ..ml.inference import load as load_ml, predict as ml_predict
        except ImportError as ex:
            QMessageBox.critical(self, "ML не установлен",
                                 f"Не установлены ML-зависимости:\n{ex}\n\n"
                                 f"Запусти: pip install torch torchvision albumentations")
            return

        try:
            self.statusBar().showMessage(f"Загружаю модель из {ckpt} ...")
            self.repaint()
            if getattr(self, "_ml_model", None) is None or self._ml_ckpt_path != str(ckpt):
                self._ml_model = load_ml(ckpt, device="auto")
                self._ml_ckpt_path = str(ckpt)
            with self._current_image_path.open("rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            self.statusBar().showMessage("Применяю модель (TTA flip) …")
            self.repaint()
            pred, conf = ml_predict(self._ml_model, bgr, tta=True,
                                    return_confidence=True)
            self._last_ml_confidences = conf
        except Exception as ex:
            QMessageBox.critical(self, "Ошибка ML-детектора", f"{type(ex).__name__}: {ex}")
            return

        # Restrict to landmarks belonging to the current profile (e.g. the
        # 8-point Alpatov subset of the 12-point model output).
        active_ids = set(self._profile.ids)
        self._undo.push(self._current_ann.landmarks)
        applied = 0
        low_conf_ids: list = []
        for lid, (x, y) in pred.items():
            if lid not in active_ids:
                continue
            self._current_ann.set_landmark(lid, x, y)
            # Mark low-confidence landmarks as uncertain — flags for manual review.
            if conf.get(lid, 1.0) < 0.45:
                lm = self._current_ann.get_landmark(lid)
                if lm is not None:
                    lm.uncertain = True
                low_conf_ids.append(lid)
            applied += 1
        self.canvas.refresh_landmarks()
        self.canvas.set_current_id(self._current_lm_id)
        self.side.refresh(self._current_ann)
        self._save_current()

        active_conf = [conf[lid] for lid in active_ids if lid in conf]
        mean_conf = sum(active_conf) / max(1, len(active_conf))
        msg = (f"ML: {applied}/{len(active_ids)} точек, "
               f"средняя уверенность {mean_conf:.2f}")
        if low_conf_ids:
            msg += f" — проверь точки {low_conf_ids}"
        self.statusBar().showMessage(msg)

    def _run_auto_detect(self) -> None:
        """Run the classical detector on the current image and fill landmarks.

        Looks for a trained shape model at <project_dir>/shape_model.json or
        falls back to ./shape_model.json. If neither exists, falls back to
        training one from <project_dir>/landmarks.csv (the workflow if the
        user annotated some specimens first and exported a CSV).
        """
        if self._current_ann is None or self._current_image_path is None or self._project_dir is None:
            return
        from ..detector.pipeline import Detector
        from ..detector.shape_model import ShapeModel, build_from_csv

        # Only look inside the currently open project — never fall back to the
        # CWD, because a stale shape_model.json or landmarks.csv there can
        # reference images outside this project and yield zero specimens.
        model_path = self._project_dir / "shape_model.json"
        csv_path = self._project_dir / "landmarks.csv"
        try:
            if model_path.exists():
                model = ShapeModel.load(model_path)
            elif csv_path.exists():
                self.statusBar().showMessage("Обучаю shape-модель из landmarks.csv …")
                self.repaint()
                model = build_from_csv(csv_path, self._project_dir)
                if model.n_specimens == 0:
                    QMessageBox.warning(
                        self, "Нет образцов",
                        "В landmarks.csv нет ни одной строки, ссылающейся на изображение в этой папке.\n\n"
                        "Используй кнопку «🧠 ML авто-определить точки» — она работает без локальной "
                        "обучающей выборки."
                    )
                    return
                model.save(self._project_dir / "shape_model.json")
            else:
                QMessageBox.information(
                    self, "Классический детектор недоступен",
                    "Для классического детектора нужны размеченные эталоны "
                    "в этой папке (shape_model.json или landmarks.csv).\n\n"
                    "Используй кнопку «🧠 ML авто-определить точки» — она "
                    "работает без локальной обучающей выборки."
                )
                return

            det = Detector(model, gate_radius=80.0)
            self.statusBar().showMessage("Запускаю детектор на текущем крыле …")
            self.repaint()
            # Use the original image (without flip): detector operates on raw pixels.
            with self._current_image_path.open("rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            result = det.detect(bgr, self._current_image_path)
        except Exception as ex:  # surface any pipeline error nicely
            QMessageBox.critical(self, "Ошибка детектора", f"{type(ex).__name__}: {ex}")
            return

        # Save undo snapshot and overwrite landmarks with detector output.
        self._undo.push(self._current_ann.landmarks)
        for lid, (x, y) in result.predicted.items():
            self._current_ann.set_landmark(lid, x, y)
        self.canvas.refresh_landmarks()
        self.canvas.set_current_id(self._current_lm_id)
        self.side.refresh(self._current_ann)
        self._save_current()
        self.statusBar().showMessage(
            f"Детектор: {len(result.predicted)}/{len(model.landmark_ids)} точек расставлено. "
            f"Проверь и подправь вручную."
        )

    def _show_landmark_context_menu(self, lm_id: int, global_pos) -> None:
        menu = QMenu(self)
        menu.addAction(f"Выбрать L{lm_id}", lambda: self._set_current_id(lm_id))
        menu.addAction(f"Приблизиться к L{lm_id}", lambda: self.canvas.zoom_to_landmark(lm_id, factor=8.0))
        menu.addSeparator()
        menu.addAction("Пометить как неуверенно (U)", lambda: self._toggle_uncertain_id(lm_id))
        menu.addAction("Пометить как пропущенную (S)", lambda: self._toggle_skipped_id(lm_id))
        menu.addSeparator()
        menu.addAction(f"Удалить точку L{lm_id}", lambda: self._delete_landmark_by_id(lm_id))
        menu.exec(global_pos)
