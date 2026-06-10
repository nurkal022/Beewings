"""Canvas: image + landmarks + zoom/pan + drag + magnifier.

Coordinate system: scene units == original image pixel coordinates.
Landmarks are stored in the WingAnnotation in those same coordinates,
so flipping/zoom only affects display, not stored data.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from ..core.profiles import Profile
from ..core.schema import Landmark, WingAnnotation
from .enhancements import EnhanceParams, apply as apply_enh


def ndarray_to_qimage(arr: np.ndarray) -> QImage:
    if arr.ndim == 2:
        h, w = arr.shape
        return QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()
    h, w, c = arr.shape
    if c == 3:
        rgb = arr[..., ::-1].copy()  # BGR -> RGB
        return QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    if c == 4:
        rgba = arr[..., [2, 1, 0, 3]].copy()
        return QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    raise ValueError(f"Unsupported array shape: {arr.shape}")


class LandmarkItem(QGraphicsEllipseItem):
    """A single landmark dot on the canvas.

    Its center sits at scene coords (lm.x, lm.y). The on-screen radius stays
    visually constant regardless of zoom (achieved via ItemIgnoresTransformations).
    """
    RADIUS = 6.0

    def __init__(self, lm: Landmark, color: QColor, is_current: bool, on_moved: Callable[[int, float, float], None]):
        super().__init__(-self.RADIUS, -self.RADIUS, 2 * self.RADIUS, 2 * self.RADIUS)
        self.lm_id = lm.id
        self._on_moved = on_moved
        self.setPos(lm.x, lm.y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setZValue(10)
        self.setAcceptHoverEvents(True)
        self.update_style(color, is_current, lm.uncertain, lm.skipped)
        # Label text floats just above the dot.
        self._label = QGraphicsSimpleTextItem(str(lm.id), self)
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        self._label.setFont(f)
        self._label.setBrush(QBrush(QColor("white")))
        self._label.setPos(self.RADIUS + 1, -self.RADIUS - 12)
        self._label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)

    def update_style(self, color: QColor, is_current: bool, uncertain: bool, skipped: bool):
        pen = QPen(QColor("black"))
        pen.setWidthF(1.5)
        if is_current:
            pen = QPen(QColor("white"))
            pen.setWidthF(2.5)
        self.setPen(pen)
        fill = QColor(color)
        if uncertain:
            fill.setAlpha(120)
        if skipped:
            fill = QColor("#888")
        self.setBrush(QBrush(fill))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            p = self.scenePos()
            self._on_moved(self.lm_id, p.x(), p.y())
        return super().itemChange(change, value)


class WingCanvas(QGraphicsView):
    landmark_clicked = pyqtSignal(int)         # user clicked a landmark dot
    canvas_clicked = pyqtSignal(float, float)  # user clicked empty area at (x, y)
    cursor_moved = pyqtSignal(float, float)    # for magnifier (scene coords)
    zoom_changed = pyqtSignal(float)           # emitted with new zoom factor (1.0 = 100%)
    landmark_context_menu = pyqtSignal(int, QPoint)  # right-click on a landmark (id, global pos)

    # Zoom limits in "scene-to-view" scale. 0.05 => see 5% of pixel size,
    # 64x => single pixel becomes a 64-pixel block (enough for sub-pixel work).
    MIN_ZOOM = 0.05
    MAX_ZOOM = 64.0

    def __init__(self, parent=None):
        super().__init__(parent)
        # Parentless, Python-owned scene: its lifetime is tied to this canvas's
        # Python object, not to the view's Qt parent chain. Parenting it to the
        # view (QGraphicsScene(self)) let Qt delete the scene during embedded
        # window/tab teardown while a usable Python reference survived, which
        # then crashed on the next scene access (deleted-QGraphicsScene abort).
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor("#1a1a1a")))
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        # Snappy rendering when zoomed in: don't smooth interpolation past 4x,
        # show actual pixels so the annotator can hit a vein crossing precisely.
        self._smooth_until = 3.0

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._raw_image: Optional[np.ndarray] = None
        self._enhance = EnhanceParams()
        self._landmark_items: Dict[int, LandmarkItem] = {}
        self._template_items: list = []
        self._profile: Optional[Profile] = None
        self._current_id: int = 1
        self._ann: Optional[WingAnnotation] = None
        self._on_landmark_moved: Optional[Callable[[int, float, float], None]] = None
        self._space_pan = False
        # Middle-button / spacebar drag panning state
        self._panning = False
        self._pan_last: Optional[QPoint] = None
        # Crosshair following the cursor for precise targeting.
        self._show_crosshair = True
        self._cursor_scene_pos: Optional[QPointF] = None

    # ---- public API ---------------------------------------------------------

    def load_image(self, img_bgr: np.ndarray) -> None:
        self._raw_image = img_bgr
        self._refresh_pixmap()
        if self._pixmap_item is not None:
            self.setSceneRect(self._pixmap_item.boundingRect())
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._after_zoom()

    def set_enhancement(self, params: EnhanceParams) -> None:
        self._enhance = params
        self._refresh_pixmap()

    def set_annotation(self, ann: WingAnnotation, profile: Profile,
                       on_moved: Callable[[int, float, float], None]) -> None:
        self._ann = ann
        self._profile = profile
        self._on_landmark_moved = on_moved
        self._rebuild_landmark_items()

    def set_current_id(self, lm_id: int) -> None:
        self._current_id = lm_id
        self._restyle_items()

    def _scene_alive(self) -> bool:
        """Defense-in-depth: never touch a scene whose C++ object is gone."""
        return self._scene is not None and not sip.isdeleted(self._scene)

    def refresh_landmarks(self) -> None:
        if not self._scene_alive():
            return
        self._rebuild_landmark_items()

    def set_template(self, template_landmarks: list) -> None:
        if not self._scene_alive():
            return
        for it in self._template_items:
            self._scene.removeItem(it)
        self._template_items.clear()
        for lm in template_landmarks:
            if lm.skipped or lm.x < 0:
                continue
            it = QGraphicsEllipseItem(-5, -5, 10, 10)
            it.setPos(lm.x, lm.y)
            it.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            it.setBrush(QBrush(QColor(0, 0, 0, 0)))
            pen = QPen(QColor(255, 255, 255, 140))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidthF(1.2)
            it.setPen(pen)
            it.setZValue(5)
            self._scene.addItem(it)
            self._template_items.append(it)

    def clear_template(self) -> None:
        if not self._scene_alive():
            return
        for it in self._template_items:
            self._scene.removeItem(it)
        self._template_items.clear()

    def fit_view(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._after_zoom()

    def current_zoom(self) -> float:
        """Return current scale factor (1.0 = 100% = one image pixel per screen pixel)."""
        return float(self.transform().m11())

    def set_zoom(self, factor: float, anchor: Optional[QPointF] = None) -> None:
        """Set absolute zoom factor, optionally anchored to a scene point."""
        factor = max(self.MIN_ZOOM, min(self.MAX_ZOOM, factor))
        cur = self.current_zoom()
        if cur <= 0:
            return
        if anchor is None:
            # anchor at viewport center to keep current center visible
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.scale(factor / cur, factor / cur)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            # Temporarily set anchor under a chosen scene point.
            view_pt = self.mapFromScene(anchor)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            self.scale(factor / cur, factor / cur)
            new_view_pt = self.mapFromScene(anchor)
            delta = new_view_pt - view_pt
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() + delta.x())
            vbar.setValue(vbar.value() + delta.y())
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._after_zoom()

    def zoom_by(self, factor: float) -> None:
        """Multiplicative zoom anchored at viewport center."""
        self.set_zoom(self.current_zoom() * factor, anchor=None)

    def zoom_to_landmark(self, lm_id: int, factor: float = 8.0) -> None:
        if self._ann is None:
            return
        lm = self._ann.get_landmark(lm_id)
        if lm is None or lm.skipped or lm.x < 0:
            return
        self.set_zoom(factor, anchor=None)
        self.centerOn(lm.x, lm.y)
        self._after_zoom()

    def set_crosshair_enabled(self, enabled: bool) -> None:
        self._show_crosshair = enabled
        self.viewport().update()

    def _after_zoom(self) -> None:
        z = self.current_zoom()
        # At deep zoom switch off bilinear smoothing so single pixels stay sharp.
        if self._pixmap_item is not None:
            self._pixmap_item.setTransformationMode(
                Qt.TransformationMode.SmoothTransformation
                if z < self._smooth_until
                else Qt.TransformationMode.FastTransformation
            )
        self.zoom_changed.emit(z)

    # ---- internal -----------------------------------------------------------

    def _refresh_pixmap(self) -> None:
        if self._raw_image is None or not self._scene_alive():
            return
        shown = apply_enh(self._raw_image, self._enhance)
        qimg = ndarray_to_qimage(shown)
        pm = QPixmap.fromImage(qimg)
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pm)
            self._pixmap_item.setZValue(0)
        else:
            self._pixmap_item.setPixmap(pm)

    def _rebuild_landmark_items(self) -> None:
        if not self._scene_alive():
            return
        for it in self._landmark_items.values():
            self._scene.removeItem(it)
        self._landmark_items.clear()
        if self._ann is None or self._profile is None:
            return
        for lm in self._ann.landmarks:
            try:
                spec = self._profile.get(lm.id)
            except KeyError:
                continue
            color = QColor(spec.color)
            it = LandmarkItem(
                lm, color,
                is_current=(lm.id == self._current_id),
                on_moved=self._on_landmark_moved or (lambda *_: None),
            )
            self._scene.addItem(it)
            self._landmark_items[lm.id] = it

    def _restyle_items(self) -> None:
        if self._ann is None or self._profile is None:
            return
        for lm_id, item in self._landmark_items.items():
            lm = self._ann.get_landmark(lm_id)
            if lm is None:
                continue
            spec = self._profile.get(lm_id)
            item.update_style(QColor(spec.color), lm_id == self._current_id, lm.uncertain, lm.skipped)

    # ---- events -------------------------------------------------------------

    def wheelEvent(self, ev: QWheelEvent) -> None:
        # Ctrl/Cmd + wheel => zoom; plain wheel => pan (natural on trackpads).
        mods = ev.modifiers()
        zoom_keys = (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        if mods & zoom_keys:
            delta = ev.angleDelta().y()
            if delta == 0:
                return
            # Smooth logarithmic zoom. 120 == one mouse notch -> ~1.2x.
            factor = 1.0015 ** delta
            target = self.current_zoom() * factor
            target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, target))
            # Anchor zoom to the scene point under the cursor.
            scene_anchor = self.mapToScene(ev.position().toPoint())
            self.set_zoom(target, anchor=scene_anchor)
            ev.accept()
            return
        # Pan: trackpads send pixelDelta, mice send angleDelta (no pixelDelta).
        px = ev.pixelDelta()
        if not px.isNull():
            dx, dy = px.x(), px.y()
        else:
            ang = ev.angleDelta()
            dx, dy = ang.x() // 2, ang.y() // 2
        if dx or dy:
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - dx)
            vbar.setValue(vbar.value() - dy)
        ev.accept()

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._space_pan = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._space_pan = False
            if not self._panning:
                self.viewport().unsetCursor()
        else:
            super().keyReleaseEvent(ev)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        # Middle button -> start panning. Also: Space+left, or Left when no
        # image hit (panning through empty area is fine).
        if ev.button() == Qt.MouseButton.MiddleButton or (
            ev.button() == Qt.MouseButton.LeftButton and self._space_pan
        ):
            self._panning = True
            self._pan_last = ev.pos()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return

        # Right click on a landmark -> emit context menu request.
        if ev.button() == Qt.MouseButton.RightButton:
            item = self.itemAt(ev.pos())
            target = None
            if isinstance(item, LandmarkItem):
                target = item
            elif isinstance(item, QGraphicsSimpleTextItem) and isinstance(item.parentItem(), LandmarkItem):
                target = item.parentItem()
            if target is not None:
                self.landmark_context_menu.emit(target.lm_id, ev.globalPosition().toPoint())
                ev.accept()
                return

        if ev.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(ev.pos())
            item = self.itemAt(ev.pos())
            if isinstance(item, LandmarkItem):
                self.landmark_clicked.emit(item.lm_id)
                super().mousePressEvent(ev)  # allow drag
                return
            if isinstance(item, QGraphicsSimpleTextItem) and isinstance(item.parentItem(), LandmarkItem):
                self.landmark_clicked.emit(item.parentItem().lm_id)
                return
            if self._pixmap_item is not None and self._pixmap_item.contains(self._pixmap_item.mapFromScene(scene_pos)):
                self.canvas_clicked.emit(scene_pos.x(), scene_pos.y())
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._panning and self._pan_last is not None:
            delta = ev.pos() - self._pan_last
            self._pan_last = ev.pos()
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            ev.accept()
            return
        scene_pos = self.mapToScene(ev.pos())
        self._cursor_scene_pos = scene_pos
        if self._show_crosshair:
            self.viewport().update()
        self.cursor_moved.emit(scene_pos.x(), scene_pos.y())
        super().mouseMoveEvent(ev)

    def leaveEvent(self, ev) -> None:
        self._cursor_scene_pos = None
        if self._show_crosshair:
            self.viewport().update()
        super().leaveEvent(ev)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        if not self._show_crosshair or self._cursor_scene_pos is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        # Draw in scene coords but with a pen whose width is constant in view px.
        pen = QPen(QColor(0, 255, 120, 180))
        pen.setCosmetic(True)
        pen.setWidth(1)
        painter.setPen(pen)
        x, y = self._cursor_scene_pos.x(), self._cursor_scene_pos.y()
        # Long crosshair across the visible scene rectangle.
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        painter.restore()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if self._panning and ev.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._panning = False
            self._pan_last = None
            if self._space_pan:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.viewport().unsetCursor()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def get_raw_image(self) -> Optional[np.ndarray]:
        return self._raw_image

    def get_enhanced_image(self) -> Optional[np.ndarray]:
        if self._raw_image is None:
            return None
        return apply_enh(self._raw_image, self._enhance)
