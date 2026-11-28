"""Interactive Mask & Decensor editor window for DeepCreamPy."""

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
  QColor,
  QImage,
  QKeySequence,
  QPainter,
  QPen,
  QPixmap,
  QShortcut,
)
from PySide6.QtWidgets import (
  QApplication,
  QButtonGroup,
  QCheckBox,
  QComboBox,
  QFileDialog,
  QFrame,
  QGridLayout,
  QGroupBox,
  QHBoxLayout,
  QLabel,
  QMessageBox,
  QProgressBar,
  QPushButton,
  QRadioButton,
  QScrollArea,
  QSizePolicy,
  QSlider,
  QToolButton,
  QVBoxLayout,
  QWidget,
)

from anima_ui import AnimaSettingsPanel
from i18n import t
from mask_detect import (
  DetectionParams,
  apply_mask_to_image,
  clamp_rect,
  detect_censor_in_rect,
)
from model import (
  anima_availability,
  anima_radio_tooltip,
  lama_radio_tooltip,
  lama_weights_available,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_UNDO_STATES = 40

TOOL_AUTO_DETECT = "auto_detect"
TOOL_RECT_FILL = "rect_fill"
TOOL_BRUSH = "brush"
TOOL_ERASER = "eraser"


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
  """Convert a PIL Image to an owned QPixmap."""
  rgba = image.convert("RGBA")
  data = rgba.tobytes("raw", "RGBA")
  qimage = QImage(
    data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888
  ).copy()
  return QPixmap.fromImage(qimage)


def qimage_to_pil(qimage: QImage) -> Image.Image:
  """Convert a non-null QImage to a PIL Image (RGB if opaque, else RGBA)."""
  converted = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
  width, height = converted.width(), converted.height()
  raw = bytes(converted.constBits())
  arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, converted.bytesPerLine()))
  rgba = arr[:, : width * 4].reshape((height, width, 4)).copy()
  if (rgba[..., 3] == 255).all():
    return Image.fromarray(rgba[..., :3], mode="RGB")
  return Image.fromarray(rgba, mode="RGBA")


def _extract_dropped_image_path(mime_data) -> Path | None:
  if not mime_data.hasUrls():
    return None
  for url in mime_data.urls():
    if url.isLocalFile():
      path = Path(url.toLocalFile())
      if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return path
  return None


class MaskCanvas(QWidget):
  """Zoomable, pannable image canvas with mask painting and live auto-detect preview."""

  file_dropped = Signal(str)
  zoom_changed = Signal(float)
  mask_changed = Signal()
  pending_changed = Signal(bool)
  auto_detect_requested = Signal(tuple)  # (left, top, right, bottom)
  commit_pending_requested = Signal()

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setAcceptDrops(True)
    self.setMouseTracking(True)
    self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    self.setMinimumSize(420, 360)

    self.base_pixmap = QPixmap()
    self.result_pixmap = QPixmap()
    self.image_width = 0
    self.image_height = 0

    self.mask: np.ndarray | None = None
    self._overlay_rgba: np.ndarray | None = None
    self._overlay_qimage = QImage()
    self.mask_color = (0, 255, 0)
    self.mask_opacity = 0.55

    self.pending_rect: tuple[int, int, int, int] | None = None
    self.pending_mask: np.ndarray | None = None
    self._pending_rgba: np.ndarray | None = None
    self._pending_qimage = QImage()

    self.tool = TOOL_AUTO_DETECT
    self.brush_size = 20
    self.view_mode = "edit"  # "edit" or "result"
    self.comparing_before = False

    self.zoom = 1.0
    self.pan = QPointF(0.0, 0.0)
    self._space_pressed = False
    self._panning = False
    self._pan_last_pos = QPointF()

    self._dragging_rect = False
    self._rect_start = QPointF()
    self._rect_end = QPointF()

    self._painting = False
    self._paint_last_img_pos: QPointF | None = None
    self._stroke_before: bytes | None = None
    self._stroke_modified = False

    self._hover_pos: QPointF | None = None

    self._undo_stack: list[bytes] = []
    self._redo_stack: list[bytes] = []

  def has_image(self) -> bool:
    return not self.base_pixmap.isNull()

  def set_mask_color(self, rgb: tuple[int, int, int]):
    self.mask_color = tuple(int(c) for c in rgb)
    self._rebuild_overlay()
    self.update()

  def set_mask_opacity(self, opacity: float):
    self.mask_opacity = max(0.05, min(1.0, float(opacity)))
    self.update()

  def set_tool(self, tool: str):
    if self.tool != tool:
      self.tool = tool
      self._update_cursor()
      self.update()

  def set_brush_size(self, size: int):
    self.brush_size = max(1, int(size))
    self.update()

  def set_view_mode(self, mode: str):
    if mode in ("edit", "result"):
      self.view_mode = mode
      self._update_cursor()
      self.update()

  def set_comparing_before(self, comparing: bool):
    self.comparing_before = comparing
    self.update()

  def set_result_pixmap(self, pixmap: QPixmap):
    self.result_pixmap = pixmap
    self.update()

  def load_image(self, image: Image.Image):
    self.base_pixmap = pil_to_qpixmap(image)
    self.result_pixmap = QPixmap()
    self.image_width = image.width
    self.image_height = image.height
    self.mask = np.zeros((self.image_height, self.image_width), dtype=bool)
    self._overlay_rgba = np.zeros(
      (self.image_height, self.image_width, 4), dtype=np.uint8
    )
    self._overlay_qimage = QImage(
      self._overlay_rgba.data,
      self.image_width,
      self.image_height,
      self.image_width * 4,
      QImage.Format.Format_RGBA8888,
    )
    self.clear_pending(emit_signal=False)
    self._undo_stack.clear()
    self._redo_stack.clear()
    self.view_mode = "edit"
    self.comparing_before = False
    self.fit_to_view()
    self.pending_changed.emit(False)
    self.mask_changed.emit()

  def _pack_mask(self, mask: np.ndarray) -> bytes:
    return np.packbits(mask.ravel()).tobytes()

  def _unpack_mask(self, packed: bytes) -> np.ndarray:
    total = self.image_width * self.image_height
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:total]
    return bits.astype(bool).reshape((self.image_height, self.image_width))

  def _push_undo_snapshot(self, packed_before: bytes):
    self._undo_stack.append(packed_before)
    if len(self._undo_stack) > MAX_UNDO_STATES:
      self._undo_stack.pop(0)
    self._redo_stack.clear()

  def _rebuild_overlay(self):
    if self.mask is None or self._overlay_rgba is None:
      return
    self._overlay_rgba.fill(0)
    self._overlay_rgba[self.mask] = (*self.mask_color, 255)

  def _sync_overlay_region(self, x0: int, y0: int, x1: int, y1: int):
    if self.mask is None or self._overlay_rgba is None:
      return
    sub_mask = self.mask[y0:y1, x0:x1]
    sub_rgba = self._overlay_rgba[y0:y1, x0:x1]
    sub_rgba.fill(0)
    sub_rgba[sub_mask] = (*self.mask_color, 255)

  def set_pending_detection(
    self, rect: tuple[int, int, int, int], roi_mask: np.ndarray
  ):
    left, top, right, bottom = rect
    if left >= right or top >= bottom or roi_mask.size == 0:
      self.clear_pending()
      return
    self.pending_rect = (left, top, right, bottom)
    self.pending_mask = roi_mask.copy()
    h, w = self.pending_mask.shape
    self._pending_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    self._pending_rgba[self.pending_mask] = (40, 255, 160, 255)
    self._pending_qimage = QImage(
      self._pending_rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888
    )
    self.pending_changed.emit(True)
    self.update()

  def has_pending(self) -> bool:
    return self.pending_rect is not None and self.pending_mask is not None

  def commit_pending(self) -> bool:
    if not self.has_pending() or self.mask is None:
      return False
    left, top, right, bottom = self.pending_rect
    roi = self.pending_mask
    before = self._pack_mask(self.mask)
    target = self.mask[top:bottom, left:right]
    changed = bool(np.any(roi & ~target))
    if changed:
      target |= roi
      self._sync_overlay_region(left, top, right, bottom)
      self._push_undo_snapshot(before)
    self.clear_pending(emit_signal=True)
    if changed:
      self.mask_changed.emit()
    return changed

  def clear_pending(self, *, emit_signal: bool = True):
    had_pending = self.pending_rect is not None
    self.pending_rect = None
    self.pending_mask = None
    self._pending_rgba = None
    self._pending_qimage = QImage()
    if emit_signal and had_pending:
      self.pending_changed.emit(False)
    self.update()

  def clear_mask(self):
    if self.mask is None:
      return
    self.clear_pending()
    if not self.mask.any():
      return
    self._push_undo_snapshot(self._pack_mask(self.mask))
    self.mask.fill(False)
    self._rebuild_overlay()
    self.mask_changed.emit()
    self.update()

  def can_undo(self) -> bool:
    return self.has_pending() or bool(self._undo_stack)

  def can_redo(self) -> bool:
    return bool(self._redo_stack)

  def undo(self):
    if self.has_pending():
      self.clear_pending()
      self.mask_changed.emit()
      return
    if not self._undo_stack or self.mask is None:
      return
    self._redo_stack.append(self._pack_mask(self.mask))
    self.mask = self._unpack_mask(self._undo_stack.pop())
    self._rebuild_overlay()
    self.mask_changed.emit()
    self.update()

  def redo(self):
    if not self._redo_stack or self.mask is None:
      return
    self.clear_pending()
    self._undo_stack.append(self._pack_mask(self.mask))
    self.mask = self._unpack_mask(self._redo_stack.pop())
    self._rebuild_overlay()
    self.mask_changed.emit()
    self.update()

  def has_any_mask(self) -> bool:
    return (self.mask is not None and bool(self.mask.any())) or (
      self.pending_mask is not None and bool(self.pending_mask.any())
    )

  def fit_to_view(self):
    if not self.has_image():
      return
    margin = 16
    avail_w = max(1, self.width() - 2 * margin)
    avail_h = max(1, self.height() - 2 * margin)
    self.zoom = min(avail_w / self.image_width, avail_h / self.image_height, 1.0)
    if self.image_width < avail_w and self.image_height < avail_h:
      self.zoom = min(avail_w / self.image_width, avail_h / self.image_height)
    self.zoom = max(0.05, min(20.0, self.zoom))
    disp_w = self.image_width * self.zoom
    disp_h = self.image_height * self.zoom
    self.pan = QPointF((self.width() - disp_w) / 2.0, (self.height() - disp_h) / 2.0)
    self.zoom_changed.emit(self.zoom)
    self.update()

  def set_zoom_around(self, new_zoom: float, anchor: QPointF | None = None):
    if not self.has_image():
      return
    new_zoom = max(0.05, min(20.0, new_zoom))
    if anchor is None:
      anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
    img_pt = self.widget_to_image(anchor, clamp=False)
    self.zoom = new_zoom
    self.pan = QPointF(
      anchor.x() - img_pt.x() * self.zoom,
      anchor.y() - img_pt.y() * self.zoom,
    )
    self.zoom_changed.emit(self.zoom)
    self.update()

  def widget_to_image(self, pos: QPointF, *, clamp: bool = True) -> QPointF:
    x = (pos.x() - self.pan.x()) / self.zoom
    y = (pos.y() - self.pan.y()) / self.zoom
    if clamp and self.has_image():
      x = min(max(x, 0.0), float(self.image_width))
      y = min(max(y, 0.0), float(self.image_height))
    return QPointF(x, y)

  def _paint_capsule(self, p0: QPointF, p1: QPointF, value: bool):
    if self.mask is None:
      return
    radius = max(0.5, self.brush_size / 2.0)
    x0 = max(0, int(np.floor(min(p0.x(), p1.x()) - radius - 1)))
    y0 = max(0, int(np.floor(min(p0.y(), p1.y()) - radius - 1)))
    x1 = min(self.image_width, int(np.ceil(max(p0.x(), p1.x()) + radius + 1)))
    y1 = min(self.image_height, int(np.ceil(max(p0.y(), p1.y()) + radius + 1)))
    if x0 >= x1 or y0 >= y1:
      return

    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
    ys = np.arange(y0, y1, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    dx = float(p1.x() - p0.x())
    dy = float(p1.y() - p0.y())
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-6:
      dist_sq = (grid_x - float(p0.x())) ** 2 + (grid_y - float(p0.y())) ** 2
    else:
      t = ((grid_x - float(p0.x())) * dx + (grid_y - float(p0.y())) * dy) / len_sq
      t = np.clip(t, 0.0, 1.0)
      proj_x = float(p0.x()) + t * dx
      proj_y = float(p0.y()) + t * dy
      dist_sq = (grid_x - proj_x) ** 2 + (grid_y - proj_y) ** 2

    hit = dist_sq <= (radius * radius)
    sub = self.mask[y0:y1, x0:x1]
    if np.any(sub[hit] != value):
      sub[hit] = value
      self._sync_overlay_region(x0, y0, x1, y1)
      self._stroke_modified = True

  def _apply_rect_fill(self, rect: tuple[int, int, int, int], value: bool):
    if self.mask is None:
      return
    left, top, right, bottom = clamp_rect(rect, self.image_width, self.image_height)
    if left >= right or top >= bottom:
      return
    sub = self.mask[top:bottom, left:right]
    if not np.any(sub != value):
      return
    self._push_undo_snapshot(self._pack_mask(self.mask))
    sub[:, :] = value
    self._sync_overlay_region(left, top, right, bottom)
    self.mask_changed.emit()
    self.update()

  def _update_cursor(self):
    if self._panning:
      self.setCursor(Qt.CursorShape.ClosedHandCursor)
    elif self._space_pressed:
      self.setCursor(Qt.CursorShape.OpenHandCursor)
    elif not self.has_image() or self.view_mode == "result":
      self.setCursor(Qt.CursorShape.ArrowCursor)
    elif self.tool in (TOOL_AUTO_DETECT, TOOL_RECT_FILL):
      self.setCursor(Qt.CursorShape.CrossCursor)
    else:
      self.setCursor(Qt.CursorShape.BlankCursor)

  def resizeEvent(self, event):
    super().resizeEvent(event)
    if self.has_image() and event.oldSize().width() <= 0:
      self.fit_to_view()

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.fillRect(self.rect(), QColor("#1f1f22"))

    if not self.has_image():
      pen = QPen(QColor("#666a73"), 2, Qt.PenStyle.DashLine)
      painter.setPen(pen)
      margin = 28
      box = self.rect().adjusted(margin, margin, -margin, -margin)
      painter.drawRoundedRect(box, 12, 12)
      painter.setPen(QColor("#cfd3dc"))
      painter.drawText(
        box,
        Qt.AlignmentFlag.AlignCenter,
        t("mask_editor.canvas_drop_hint"),
      )
      return

    target_rect = QRectF(
      self.pan.x(),
      self.pan.y(),
      self.image_width * self.zoom,
      self.image_height * self.zoom,
    )

    if self.zoom < 2.0:
      painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    show_result = (
      self.view_mode == "result"
      and not self.comparing_before
      and not self.result_pixmap.isNull()
    )
    pixmap = self.result_pixmap if show_result else self.base_pixmap
    painter.drawPixmap(target_rect, pixmap, QRectF(pixmap.rect()))

    if self.view_mode == "edit":
      painter.save()
      painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
      painter.setOpacity(self.mask_opacity)
      if not self._overlay_qimage.isNull():
        painter.drawImage(target_rect, self._overlay_qimage)
      if self.pending_rect is not None and not self._pending_qimage.isNull():
        left, top, right, bottom = self.pending_rect
        pending_target = QRectF(
          self.pan.x() + left * self.zoom,
          self.pan.y() + top * self.zoom,
          (right - left) * self.zoom,
          (bottom - top) * self.zoom,
        )
        painter.drawImage(pending_target, self._pending_qimage)
      painter.restore()

      if self.pending_rect is not None:
        left, top, right, bottom = self.pending_rect
        box = QRectF(
          self.pan.x() + left * self.zoom,
          self.pan.y() + top * self.zoom,
          (right - left) * self.zoom,
          (bottom - top) * self.zoom,
        )
        pen = QPen(QColor("#40ffd0"), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(box)

      if self._dragging_rect:
        rect = QRectF(self._rect_start, self._rect_end).normalized()
        screen_rect = QRectF(
          self.pan.x() + rect.left() * self.zoom,
          self.pan.y() + rect.top() * self.zoom,
          rect.width() * self.zoom,
          rect.height() * self.zoom,
        )
        color = "#ffcc40" if self.tool == TOOL_AUTO_DETECT else "#40ff40"
        pen = QPen(QColor(color), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(screen_rect)

      if (
        self._hover_pos is not None
        and not self._panning
        and not self._space_pressed
        and self.tool in (TOOL_BRUSH, TOOL_ERASER)
      ):
        radius_px = max(2.0, (self.brush_size * self.zoom) / 2.0)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(
          QColor("#00ff66") if self.tool == TOOL_BRUSH else QColor("#ff6666"),
          1.5,
        )
        painter.setPen(pen)
        painter.drawEllipse(self._hover_pos, radius_px, radius_px)

    if self.view_mode == "result":
      badge = (
        t("mask_editor.badge_before")
        if self.comparing_before
        else t("mask_editor.badge_after")
      )
      badge_bg = QColor(30, 30, 35, 190)
      badge_width = max(
        140.0, float(painter.fontMetrics().horizontalAdvance(badge)) + 24.0
      )
      badge_rect = QRectF(12, 12, badge_width, 26)
      painter.fillRect(badge_rect, badge_bg)
      painter.setPen(QColor("#ffffff"))
      painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge)

  def wheelEvent(self, event):
    if not self.has_image():
      return
    delta = event.angleDelta().y()
    if delta == 0:
      return
    factor = 1.18 if delta > 0 else (1.0 / 1.18)
    self.set_zoom_around(self.zoom * factor, event.position())
    event.accept()

  def mousePressEvent(self, event):
    if not self.has_image():
      return
    if event.button() == Qt.MouseButton.MiddleButton or (
      event.button() == Qt.MouseButton.LeftButton and self._space_pressed
    ):
      self._panning = True
      self._pan_last_pos = event.position()
      self._update_cursor()
      event.accept()
      return

    if self.view_mode != "edit":
      return

    if event.button() == Qt.MouseButton.LeftButton:
      if self.has_pending():
        self.commit_pending_requested.emit()
      img_pos = self.widget_to_image(event.position(), clamp=True)
      if self.tool in (TOOL_AUTO_DETECT, TOOL_RECT_FILL):
        self._dragging_rect = True
        self._rect_start = img_pos
        self._rect_end = img_pos
        self.update()
      elif self.tool in (TOOL_BRUSH, TOOL_ERASER):
        self._painting = True
        self._stroke_before = self._pack_mask(self.mask)
        self._stroke_modified = False
        self._paint_last_img_pos = img_pos
        self._paint_capsule(img_pos, img_pos, self.tool == TOOL_BRUSH)
        self.update()
      event.accept()
    elif event.button() == Qt.MouseButton.RightButton and self.tool == TOOL_RECT_FILL:
      if self.has_pending():
        self.commit_pending_requested.emit()
      img_pos = self.widget_to_image(event.position(), clamp=True)
      self._dragging_rect = True
      self._rect_start = img_pos
      self._rect_end = img_pos
      self._right_rect_erase = True
      self.update()
      event.accept()

  def mouseMoveEvent(self, event):
    self._hover_pos = event.position()
    if self._panning:
      delta = event.position() - self._pan_last_pos
      self._pan_last_pos = event.position()
      self.pan += delta
      self.update()
      event.accept()
      return

    if self._dragging_rect:
      self._rect_end = self.widget_to_image(event.position(), clamp=True)
      self.update()
      event.accept()
      return

    if self._painting and self._paint_last_img_pos is not None:
      img_pos = self.widget_to_image(event.position(), clamp=True)
      self._paint_capsule(self._paint_last_img_pos, img_pos, self.tool == TOOL_BRUSH)
      self._paint_last_img_pos = img_pos
      self.update()
      event.accept()
      return

    if self.view_mode == "edit" and self.tool in (TOOL_BRUSH, TOOL_ERASER):
      self.update()

  def mouseReleaseEvent(self, event):
    if self._panning and event.button() in (
      Qt.MouseButton.MiddleButton,
      Qt.MouseButton.LeftButton,
    ):
      self._panning = False
      self._update_cursor()
      event.accept()
      return

    if self._dragging_rect and event.button() in (
      Qt.MouseButton.LeftButton,
      Qt.MouseButton.RightButton,
    ):
      self._dragging_rect = False
      erase_rect = getattr(self, "_right_rect_erase", False)
      self._right_rect_erase = False
      end_pos = self.widget_to_image(event.position(), clamp=True)
      box = QRectF(self._rect_start, end_pos).normalized()
      rect = (
        int(np.floor(box.left())),
        int(np.floor(box.top())),
        int(np.ceil(box.right())),
        int(np.ceil(box.bottom())),
      )
      left, top, right, bottom = clamp_rect(rect, self.image_width, self.image_height)
      if left < right and top < bottom:
        if self.tool == TOOL_AUTO_DETECT and not erase_rect:
          self.auto_detect_requested.emit((left, top, right, bottom))
        elif self.tool == TOOL_RECT_FILL:
          self._apply_rect_fill((left, top, right, bottom), not erase_rect)
      self.update()
      event.accept()
      return

    if self._painting and event.button() == Qt.MouseButton.LeftButton:
      self._painting = False
      self._paint_last_img_pos = None
      if self._stroke_modified and self._stroke_before is not None:
        self._push_undo_snapshot(self._stroke_before)
        self.mask_changed.emit()
      self._stroke_before = None
      self._stroke_modified = False
      event.accept()

  def leaveEvent(self, event):
    self._hover_pos = None
    self.update()
    super().leaveEvent(event)

  def keyPressEvent(self, event):
    if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
      self._space_pressed = True
      self._update_cursor()
      event.accept()
      return
    super().keyPressEvent(event)

  def keyReleaseEvent(self, event):
    if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
      self._space_pressed = False
      self._update_cursor()
      event.accept()
      return
    super().keyReleaseEvent(event)

  def dragEnterEvent(self, event):
    if _extract_dropped_image_path(event.mimeData()) is not None:
      event.acceptProposedAction()
    else:
      event.ignore()

  def dropEvent(self, event):
    path = _extract_dropped_image_path(event.mimeData())
    if path is not None:
      self.file_dropped.emit(str(path))
      event.acceptProposedAction()


class MaskEditorWindow(QWidget):
  """Separate interactive window for masking images and running decensoring."""

  settings_changed = Signal(dict)
  decensor_requested = Signal(object, object, str)  # ori_img, colored_img, output_stem
  cancel_requested = Signal()

  def __init__(self, initial_settings: dict, mask_color=(0, 255, 0), parent=None):
    super().__init__(parent)
    self.setAcceptDrops(True)
    self.resize(1220, 780)

    self._syncing_settings = False
    self._busy = False
    self._busy_reason = ""
    self._busy_reason_key = ""
    self._can_cancel = False
    self._cancelling = False
    self._status_is_initial = True

    self.original_image: Image.Image | None = None
    self.rgb_array: np.ndarray | None = None
    self.source_path: Path | None = None
    self.output_stem: str = ""
    self.result_images: list[Image.Image] = []
    self.saved_paths: list[str] = []
    self.mask_color = tuple(int(c) for c in mask_color)

    self._build_ui()
    self.canvas.set_mask_color(self.mask_color)
    self.apply_synced_settings(initial_settings, auto_switch_tool=False)
    self._apply_tool_for_mode(initial_settings.get("is_mosaic", False))
    self._setup_shortcuts()
    self.retranslate_ui()

  def _build_ui(self):
    root = QVBoxLayout(self)
    root.setSpacing(6)
    root.setContentsMargins(8, 8, 8, 8)

    # Top toolbar
    toolbar = QHBoxLayout()
    toolbar.setSpacing(6)

    self.openButton = QPushButton()
    self.openButton.clicked.connect(self.open_image_dialog)
    self.pasteButton = QPushButton()
    self.pasteButton.clicked.connect(self.paste_from_clipboard)
    toolbar.addWidget(self.openButton)
    toolbar.addWidget(self.pasteButton)

    toolbar.addWidget(self._vsep())

    self.toolGroup = QButtonGroup(self)
    self.toolGroup.setExclusive(True)
    self.toolButtons: dict[str, QToolButton] = {}
    for tool_id in (
      TOOL_AUTO_DETECT,
      TOOL_RECT_FILL,
      TOOL_BRUSH,
      TOOL_ERASER,
    ):
      btn = QToolButton()
      btn.setCheckable(True)
      btn.clicked.connect(lambda checked, t_id=tool_id: self._select_tool(t_id))
      self.toolGroup.addButton(btn)
      self.toolButtons[tool_id] = btn
      toolbar.addWidget(btn)
    self.toolButtons[TOOL_AUTO_DETECT].setChecked(True)

    toolbar.addWidget(self._vsep())

    self.undoButton = QPushButton()
    self.undoButton.clicked.connect(self._undo)
    self.redoButton = QPushButton()
    self.redoButton.clicked.connect(self._redo)
    self.clearMaskButton = QPushButton()
    self.clearMaskButton.clicked.connect(self._clear_mask)
    toolbar.addWidget(self.undoButton)
    toolbar.addWidget(self.redoButton)
    toolbar.addWidget(self.clearMaskButton)

    toolbar.addWidget(self._vsep())

    self.zoomOutButton = QToolButton()
    self.zoomOutButton.setText("-")
    self.zoomOutButton.clicked.connect(
      lambda: self.canvas.set_zoom_around(self.canvas.zoom / 1.25)
    )
    self.fitButton = QToolButton()
    self.fitButton.clicked.connect(lambda: self.canvas.fit_to_view())
    self.zoomInButton = QToolButton()
    self.zoomInButton.setText("+")
    self.zoomInButton.clicked.connect(
      lambda: self.canvas.set_zoom_around(self.canvas.zoom * 1.25)
    )
    self.zoomLabel = QLabel("100%")
    self.zoomLabel.setFixedWidth(48)
    self.zoomLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    toolbar.addWidget(self.zoomOutButton)
    toolbar.addWidget(self.fitButton)
    toolbar.addWidget(self.zoomInButton)
    toolbar.addWidget(self.zoomLabel)

    toolbar.addStretch(1)

    # View toggle & Before/After compare
    self.editViewButton = QToolButton()
    self.editViewButton.setCheckable(True)
    self.editViewButton.setChecked(True)
    self.resultViewButton = QToolButton()
    self.resultViewButton.setCheckable(True)
    self.viewGroup = QButtonGroup(self)
    self.viewGroup.setExclusive(True)
    self.viewGroup.addButton(self.editViewButton)
    self.viewGroup.addButton(self.resultViewButton)
    self.editViewButton.clicked.connect(lambda: self._set_view_mode("edit"))
    self.resultViewButton.clicked.connect(lambda: self._set_view_mode("result"))

    self.compareButton = QPushButton()
    self.compareButton.pressed.connect(lambda: self.canvas.set_comparing_before(True))
    self.compareButton.released.connect(lambda: self.canvas.set_comparing_before(False))

    self.variationCombo = QComboBox()
    self.variationCombo.setMinimumWidth(110)
    self.variationCombo.currentIndexChanged.connect(self._select_variation)

    toolbar.addWidget(self.editViewButton)
    toolbar.addWidget(self.resultViewButton)
    toolbar.addWidget(self.compareButton)
    toolbar.addWidget(self.variationCombo)

    root.addLayout(toolbar)

    # Main content: Canvas + Sidebar
    body = QHBoxLayout()
    body.setSpacing(8)

    self.canvas = MaskCanvas(self)
    self.canvas.file_dropped.connect(self.load_image_from_path)
    self.canvas.zoom_changed.connect(self._on_zoom_changed)
    self.canvas.mask_changed.connect(self._update_ui_state)
    self.canvas.pending_changed.connect(self._on_pending_changed)
    self.canvas.auto_detect_requested.connect(self._run_auto_detect)
    self.canvas.commit_pending_requested.connect(self._commit_pending)
    body.addWidget(self.canvas, 1)

    sidebar_container = QWidget()
    sidebar = QVBoxLayout(sidebar_container)
    sidebar.setContentsMargins(4, 4, 4, 4)
    sidebar.setSpacing(8)

    # 1. Auto Detect group
    self.detectGroup = QGroupBox()
    detect_layout = QVBoxLayout(self.detectGroup)
    detect_layout.setSpacing(6)

    mode_row = QHBoxLayout()
    self.detectBlackRadio = QRadioButton()
    self.detectWhiteRadio = QRadioButton()
    self.detectBothRadio = QRadioButton()
    self.detectBlackRadio.setChecked(True)
    for radio in (
      self.detectBlackRadio,
      self.detectWhiteRadio,
      self.detectBothRadio,
    ):
      radio.toggled.connect(self._on_detect_params_changed)
      mode_row.addWidget(radio)
    detect_layout.addLayout(mode_row)

    sliders_grid = QGridLayout()
    sliders_grid.setSpacing(4)
    self.blackThreshLabel, self.blackThreshSlider, self.blackThreshValue = (
      self._add_slider_row(
        sliders_grid, 0, "", 0, 160, 50, self._on_detect_params_changed
      )
    )
    self.whiteThreshLabel, self.whiteThreshSlider, self.whiteThreshValue = (
      self._add_slider_row(
        sliders_grid, 1, "", 120, 255, 220, self._on_detect_params_changed
      )
    )
    self.maxSatLabel, self.maxSatSlider, self.maxSatValue = self._add_slider_row(
      sliders_grid, 2, "", 0, 120, 35, self._on_detect_params_changed
    )
    self.minThickLabel, self.minThickSlider, self.minThickValue = self._add_slider_row(
      sliders_grid, 3, "", 0, 12, 3, self._on_detect_params_changed
    )
    self.dilationLabel, self.dilationSlider, self.dilationValue = self._add_slider_row(
      sliders_grid, 4, "", 0, 10, 2, self._on_detect_params_changed
    )
    detect_layout.addLayout(sliders_grid)

    pending_row = QHBoxLayout()
    self.applyPendingButton = QPushButton()
    self.applyPendingButton.clicked.connect(self._commit_pending)
    self.cancelPendingButton = QPushButton()
    self.cancelPendingButton.clicked.connect(self._cancel_pending)
    pending_row.addWidget(self.applyPendingButton)
    pending_row.addWidget(self.cancelPendingButton)
    detect_layout.addLayout(pending_row)

    sidebar.addWidget(self.detectGroup)

    # 2. Brush & Display group
    self.displayGroup = QGroupBox()
    display_grid = QGridLayout(self.displayGroup)
    display_grid.setSpacing(4)
    self.brushSizeLabel, self.brushSizeSlider, self.brushSizeValue = (
      self._add_slider_row(display_grid, 0, "", 1, 120, 20, self._on_brush_size_changed)
    )
    self.opacityLabel, self.opacitySlider, self.opacityValue = self._add_slider_row(
      display_grid, 1, "", 10, 100, 55, self._on_opacity_changed, suffix="%"
    )
    sidebar.addWidget(self.displayGroup)

    # 3. Decensor Settings group (two-way synced with MainWindow)
    self.settingsGroup = QGroupBox()
    settings_layout = QVBoxLayout(self.settingsGroup)
    settings_layout.setSpacing(6)

    censor_row = QHBoxLayout()
    self.censorTypeLabel = QLabel()
    censor_row.addWidget(self.censorTypeLabel)
    self.barRadio = QRadioButton()
    self.mosaicRadio = QRadioButton()
    self.censorBtnGroup = QButtonGroup(self)
    self.censorBtnGroup.addButton(self.barRadio)
    self.censorBtnGroup.addButton(self.mosaicRadio)
    self.barRadio.toggled.connect(self._on_decensor_settings_ui_changed)
    censor_row.addWidget(self.barRadio)
    censor_row.addWidget(self.mosaicRadio)
    censor_row.addStretch(1)
    settings_layout.addLayout(censor_row)

    model_row = QHBoxLayout()
    self.modelLabel = QLabel()
    model_row.addWidget(self.modelLabel)
    self.pepsiRadio = QRadioButton("PEPSI")
    self.lamaRadio = QRadioButton("LaMa")
    self.animaRadio = QRadioButton("Anima")
    self.modelBtnGroup = QButtonGroup(self)
    self.modelBtnGroup.addButton(self.pepsiRadio)
    self.modelBtnGroup.addButton(self.lamaRadio)
    self.modelBtnGroup.addButton(self.animaRadio)
    self.pepsiRadio.toggled.connect(self._on_decensor_settings_ui_changed)
    self.lamaRadio.toggled.connect(self._on_decensor_settings_ui_changed)
    self.animaRadio.toggled.connect(self._on_decensor_settings_ui_changed)
    model_row.addWidget(self.pepsiRadio)
    model_row.addWidget(self.lamaRadio)
    model_row.addWidget(self.animaRadio)
    model_row.addStretch(1)
    settings_layout.addLayout(model_row)

    tta_var_row = QHBoxLayout()
    self.ttaCheckbox = QCheckBox()
    self.ttaCheckbox.toggled.connect(self._on_decensor_settings_ui_changed)
    tta_var_row.addWidget(self.ttaCheckbox)
    tta_var_row.addStretch(1)
    self.variationsLabel = QLabel()
    tta_var_row.addWidget(self.variationsLabel)
    self.variationsCombo = QComboBox()
    self.variationsCombo.addItems(["1", "2", "4"])
    self.variationsCombo.currentIndexChanged.connect(
      self._on_decensor_settings_ui_changed
    )
    tta_var_row.addWidget(self.variationsCombo)
    settings_layout.addLayout(tta_var_row)

    sidebar.addWidget(self.settingsGroup)

    self.animaSettingsPanel = AnimaSettingsPanel(compact=True)
    self.animaSettingsPanel.setVisible(False)
    self.animaSettingsPanel.settings_changed.connect(
      self._on_decensor_settings_ui_changed
    )
    sidebar.addWidget(self.animaSettingsPanel)

    # 4. Action buttons
    self.actionsGroup = QGroupBox()
    actions_layout = QVBoxLayout(self.actionsGroup)
    actions_layout.setSpacing(6)

    run_cancel_row = QHBoxLayout()
    self.decensorButton = QPushButton()
    self.decensorButton.setMinimumHeight(38)
    self.decensorButton.setStyleSheet("font-weight: bold;")
    self.decensorButton.clicked.connect(self._on_decensor_clicked)
    run_cancel_row.addWidget(self.decensorButton, 2)

    self.cancelButton = QPushButton()
    self.cancelButton.setMinimumHeight(38)
    self.cancelButton.setEnabled(False)
    self.cancelButton.clicked.connect(self._on_cancel_clicked)
    run_cancel_row.addWidget(self.cancelButton, 1)
    actions_layout.addLayout(run_cancel_row)

    self.progressBar = QProgressBar()
    self.progressBar.setRange(0, 100)
    self.progressBar.setValue(0)
    actions_layout.addWidget(self.progressBar)

    self.statusLabel = QLabel()
    self.statusLabel.setWordWrap(True)
    actions_layout.addWidget(self.statusLabel)

    self.saveResultButton = QPushButton()
    self.saveResultButton.clicked.connect(self.save_result_as)
    actions_layout.addWidget(self.saveResultButton)

    self.saveMaskButton = QPushButton()
    self.saveMaskButton.clicked.connect(self.save_masked_png)
    actions_layout.addWidget(self.saveMaskButton)

    sidebar.addWidget(self.actionsGroup)
    sidebar.addStretch(1)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(sidebar_container)
    scroll.setFixedWidth(320)
    scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    body.addWidget(scroll)

    root.addLayout(body, 1)

  def _update_window_title(self):
    if self.original_image is not None:
      title_name = (
        self.source_path.name if self.source_path is not None else self.output_stem
      )
      self.setWindowTitle(t("mask_editor.window_title_with_file", name=title_name))
    else:
      self.setWindowTitle(t("mask_editor.window_title"))

  def retranslate_ui(self):
    self._update_window_title()

    self.openButton.setText(t("mask_editor.open_image"))
    self.pasteButton.setText(t("mask_editor.paste"))
    self.pasteButton.setToolTip(t("mask_editor.paste_tooltip"))

    tool_specs = {
      TOOL_AUTO_DETECT: (
        "mask_editor.tool_auto_detect",
        "mask_editor.tool_auto_detect_tooltip",
      ),
      TOOL_RECT_FILL: (
        "mask_editor.tool_rect_fill",
        "mask_editor.tool_rect_fill_tooltip",
      ),
      TOOL_BRUSH: ("mask_editor.tool_brush", "mask_editor.tool_brush_tooltip"),
      TOOL_ERASER: ("mask_editor.tool_eraser", "mask_editor.tool_eraser_tooltip"),
    }
    for tool_id, (label_key, tip_key) in tool_specs.items():
      btn = self.toolButtons[tool_id]
      btn.setText(t(label_key))
      btn.setToolTip(t(tip_key))

    self.undoButton.setText(t("mask_editor.undo"))
    self.undoButton.setToolTip(t("mask_editor.undo_tooltip"))
    self.redoButton.setText(t("mask_editor.redo"))
    self.redoButton.setToolTip(t("mask_editor.redo_tooltip"))
    self.clearMaskButton.setText(t("mask_editor.clear_mask"))

    self.zoomOutButton.setToolTip(t("mask_editor.zoom_out_tooltip"))
    self.fitButton.setText(t("mask_editor.fit"))
    self.fitButton.setToolTip(t("mask_editor.fit_tooltip"))
    self.zoomInButton.setToolTip(t("mask_editor.zoom_in_tooltip"))

    self.editViewButton.setText(t("mask_editor.view_edit"))
    self.resultViewButton.setText(t("mask_editor.view_result"))
    self.compareButton.setText(t("mask_editor.hold_to_compare"))
    self.compareButton.setToolTip(t("mask_editor.hold_to_compare_tooltip"))

    if self.result_images:
      current_idx = self.variationCombo.currentIndex()
      self.variationCombo.blockSignals(True)
      for i in range(self.variationCombo.count()):
        self.variationCombo.setItemText(i, t("mask_editor.variation_item", index=i))
      if 0 <= current_idx < self.variationCombo.count():
        self.variationCombo.setCurrentIndex(current_idx)
      self.variationCombo.blockSignals(False)

    self.detectGroup.setTitle(t("mask_editor.detect_group"))
    self.detectBlackRadio.setText(t("mask_editor.detect_black"))
    self.detectWhiteRadio.setText(t("mask_editor.detect_white"))
    self.detectBothRadio.setText(t("mask_editor.detect_both"))

    self.blackThreshLabel.setText(t("mask_editor.black_max"))
    self.whiteThreshLabel.setText(t("mask_editor.white_min"))
    self.maxSatLabel.setText(t("mask_editor.max_chroma"))
    self.minThickLabel.setText(t("mask_editor.line_filter"))
    self.minThickSlider.setToolTip(t("mask_editor.line_filter_tooltip"))
    self.dilationLabel.setText(t("mask_editor.dilation"))
    self.dilationSlider.setToolTip(t("mask_editor.dilation_tooltip"))

    self.applyPendingButton.setText(t("mask_editor.apply_pending"))
    self.cancelPendingButton.setText(t("mask_editor.cancel_pending"))

    self.displayGroup.setTitle(t("mask_editor.brush_display_group"))
    self.brushSizeLabel.setText(t("mask_editor.brush_size"))
    self.opacityLabel.setText(t("mask_editor.mask_alpha"))

    self.settingsGroup.setTitle(t("mask_editor.decensor_settings_group"))
    self.censorTypeLabel.setText(t("mask_editor.type_label"))
    self.barRadio.setText(t("mask_editor.type_bar"))
    self.mosaicRadio.setText(t("mask_editor.type_mosaic"))
    self.modelLabel.setText(t("mask_editor.model_label"))
    self.ttaCheckbox.setText(t("mask_editor.use_tta"))
    self.variationsLabel.setText(t("mask_editor.variations_label"))

    self.animaSettingsPanel.retranslate_ui()
    self._refresh_lama_availability()
    self._refresh_anima_availability()

    self.actionsGroup.setTitle(t("mask_editor.actions_group"))
    self.cancelButton.setToolTip(t("mask_editor.cancel_tooltip"))
    self.saveResultButton.setText(t("mask_editor.save_result_as"))
    self.saveMaskButton.setText(t("mask_editor.save_masked_png"))

    if self._busy_reason_key:
      self._busy_reason = t(self._busy_reason_key)
    if self._status_is_initial:
      self.statusLabel.setText(t("mask_editor.status_initial"))
    elif self._cancelling:
      self.statusLabel.setText(t("mask_editor.status_cancelling"))

    self._update_ui_state()
    self.canvas.update()

  @staticmethod
  def _vsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep

  @staticmethod
  def _add_slider_row(
    grid: QGridLayout,
    row: int,
    label_text: str,
    minimum: int,
    maximum: int,
    default: int,
    slot,
    *,
    suffix: str = "",
  ) -> tuple[QLabel, QSlider, QLabel]:
    label = QLabel(label_text)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(default)
    value_label = QLabel(f"{default}{suffix}")
    value_label.setFixedWidth(36)
    value_label.setAlignment(
      Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    slider.valueChanged.connect(
      lambda v, lbl=value_label, s=suffix: lbl.setText(f"{v}{s}")
    )
    slider.valueChanged.connect(slot)
    grid.addWidget(label, row, 0)
    grid.addWidget(slider, row, 1)
    grid.addWidget(value_label, row, 2)
    return label, slider, value_label

  def _setup_shortcuts(self):
    QShortcut(QKeySequence.StandardKey.Open, self, activated=self.open_image_dialog)
    QShortcut(QKeySequence.StandardKey.Paste, self, activated=self.paste_from_clipboard)
    QShortcut(QKeySequence.StandardKey.Undo, self, activated=self._undo)
    QShortcut(QKeySequence.StandardKey.Redo, self, activated=self._redo)
    QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._redo)
    QShortcut(QKeySequence("Return"), self, activated=self._commit_pending)
    QShortcut(QKeySequence("Enter"), self, activated=self._commit_pending)
    QShortcut(QKeySequence("Escape"), self, activated=self._cancel_pending)
    QShortcut(QKeySequence("0"), self, activated=lambda: self.canvas.fit_to_view())
    QShortcut(QKeySequence("["), self, activated=lambda: self._step_brush_size(-5))
    QShortcut(QKeySequence("]"), self, activated=lambda: self._step_brush_size(5))

  def _step_brush_size(self, delta: int):
    self.brushSizeSlider.setValue(
      max(
        self.brushSizeSlider.minimum(),
        min(self.brushSizeSlider.maximum(), self.brushSizeSlider.value() + delta),
      )
    )

  # --- Settings sync ---

  def _refresh_lama_availability(self, is_mosaic: bool | None = None) -> bool:
    if is_mosaic is None:
      is_mosaic = self.mosaicRadio.isChecked()
    available = lama_weights_available()
    self.lamaRadio.setEnabled(available and not is_mosaic)
    self.lamaRadio.setToolTip(lama_radio_tooltip(available))
    if (is_mosaic or not available) and self.lamaRadio.isChecked():
      self.pepsiRadio.setChecked(True)
    return available

  def _refresh_anima_availability(self, is_mosaic: bool | None = None) -> bool:
    if is_mosaic is None:
      is_mosaic = self.mosaicRadio.isChecked()
    available, reason = anima_availability()
    self.animaRadio.setEnabled(available and not is_mosaic)
    self.animaRadio.setToolTip(anima_radio_tooltip(available, reason))
    if (is_mosaic or not available) and self.animaRadio.isChecked():
      self.pepsiRadio.setChecked(True)
    return available

  def _update_model_specific_controls(self):
    is_pepsi = self.pepsiRadio.isChecked()
    is_anima = self.animaRadio.isChecked() and self.animaRadio.isEnabled()
    self.ttaCheckbox.setEnabled(is_pepsi)
    self.animaSettingsPanel.setVisible(is_anima)
    self.animaSettingsPanel.setEnabled(is_anima)

  def get_decensor_settings(self) -> dict:
    is_mosaic = self.mosaicRadio.isChecked()
    if not is_mosaic and self.animaRadio.isEnabled() and self.animaRadio.isChecked():
      model_name = "anima"
    elif not is_mosaic and self.lamaRadio.isEnabled() and self.lamaRadio.isChecked():
      model_name = "lama"
    else:
      model_name = "pepsi"
    return {
      "is_mosaic": is_mosaic,
      "model_name": model_name,
      "use_tta": self.ttaCheckbox.isChecked(),
      "variations": int(self.variationsCombo.currentText()),
      "anima_settings": self.animaSettingsPanel.get_settings(),
    }

  def apply_synced_settings(self, settings: dict, *, auto_switch_tool: bool = True):
    prev_mosaic = self.mosaicRadio.isChecked()
    self._syncing_settings = True
    try:
      is_mosaic = bool(settings.get("is_mosaic", False))
      model_name = settings.get("model_name", "pepsi")
      use_tta = bool(settings.get("use_tta", True))
      variations = int(settings.get("variations", 1))
      anima_data = settings.get("anima_settings")

      self.mosaicRadio.setChecked(is_mosaic)
      self.barRadio.setChecked(not is_mosaic)
      lama_ok = self._refresh_lama_availability(is_mosaic=is_mosaic)
      anima_ok = self._refresh_anima_availability(is_mosaic=is_mosaic)
      if (is_mosaic or not lama_ok) and model_name == "lama":
        model_name = "pepsi"
      if (is_mosaic or not anima_ok) and model_name == "anima":
        model_name = "pepsi"
      self.pepsiRadio.setChecked(model_name == "pepsi")
      self.lamaRadio.setChecked(model_name == "lama")
      self.animaRadio.setChecked(model_name == "anima")
      self.ttaCheckbox.setChecked(use_tta)
      if anima_data is not None:
        self.animaSettingsPanel.set_settings(anima_data)
      self._update_model_specific_controls()
      idx = self.variationsCombo.findText(str(variations))
      if idx >= 0:
        self.variationsCombo.setCurrentIndex(idx)
    finally:
      self._syncing_settings = False

    if auto_switch_tool and prev_mosaic != bool(settings.get("is_mosaic", False)):
      self._apply_tool_for_mode(bool(settings.get("is_mosaic", False)))
    self._update_ui_state()

  def _apply_tool_for_mode(self, is_mosaic: bool):
    if is_mosaic and self.canvas.tool == TOOL_AUTO_DETECT:
      self._select_tool(TOOL_RECT_FILL)
    elif not is_mosaic and self.canvas.tool == TOOL_RECT_FILL:
      self._select_tool(TOOL_AUTO_DETECT)

  def _on_decensor_settings_ui_changed(self):
    if self._syncing_settings:
      return
    is_mosaic = self.mosaicRadio.isChecked()
    self._syncing_settings = True
    try:
      self._refresh_lama_availability(is_mosaic=is_mosaic)
      self._refresh_anima_availability(is_mosaic=is_mosaic)
      self._update_model_specific_controls()
    finally:
      self._syncing_settings = False

    self._apply_tool_for_mode(is_mosaic)
    self.settings_changed.emit(self.get_decensor_settings())
    self._update_ui_state()

  # --- Image loading ---

  def _set_status_message(self, text: str, *, is_initial: bool = False):
    self._status_is_initial = is_initial
    self.statusLabel.setText(text)

  def open_image_dialog(self):
    start_dir = str(self.source_path.parent) if self.source_path else ""
    file_path, _ = QFileDialog.getOpenFileName(
      self,
      t("mask_editor.dialog_open_title"),
      start_dir,
      t("mask_editor.dialog_open_filter"),
    )
    if file_path:
      self.load_image_from_path(file_path)

  def load_image_from_path(self, file_path: str):
    path = Path(file_path)
    try:
      with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode not in ("RGB", "RGBA"):
          has_alpha = "A" in image.getbands() or "transparency" in image.info
          image = image.convert("RGBA" if has_alpha else "RGB")
        else:
          image = image.copy()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
      QMessageBox.warning(
        self, t("mask_editor.dialog_open_failed_title"), f"{path}\n{error}"
      )
      return

    self._set_loaded_image(image, path.stem, source_path=path)

  def paste_from_clipboard(self):
    clipboard = QApplication.clipboard()
    qimage = clipboard.image()
    if qimage.isNull():
      self._set_status_message(t("mask_editor.status_clipboard_empty"))
      return
    image = qimage_to_pil(qimage)
    stem = f"clipboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    self._set_loaded_image(image, stem, source_path=None)

  def _set_loaded_image(
    self, image: Image.Image, output_stem: str, *, source_path: Path | None
  ):
    self.original_image = image
    self.rgb_array = np.asarray(image.convert("RGB")).copy()
    self.source_path = source_path
    self.output_stem = output_stem
    self.result_images.clear()
    self.saved_paths.clear()
    self.variationCombo.blockSignals(True)
    self.variationCombo.clear()
    self.variationCombo.blockSignals(False)

    self.canvas.load_image(image)
    self.editViewButton.setChecked(True)
    title_name = source_path.name if source_path is not None else output_stem
    self._update_window_title()
    self._set_status_message(
      t(
        "mask_editor.status_loaded",
        name=title_name,
        width=image.width,
        height=image.height,
      )
    )
    self.progressBar.setValue(0)
    self._update_ui_state()

  # --- Tools & Auto-Detection ---

  def _select_tool(self, tool_id: str):
    if self.canvas.has_pending():
      self._commit_pending()
    if self.canvas.view_mode != "edit":
      self._set_view_mode("edit")
    self.canvas.set_tool(tool_id)
    if tool_id in self.toolButtons:
      self.toolButtons[tool_id].setChecked(True)
    self._update_ui_state()

  def get_detection_params(self) -> DetectionParams:
    if self.detectWhiteRadio.isChecked():
      mode = "white"
    elif self.detectBothRadio.isChecked():
      mode = "both"
    else:
      mode = "black"
    return DetectionParams(
      mode=mode,
      black_threshold=self.blackThreshSlider.value(),
      white_threshold=self.whiteThreshSlider.value(),
      max_saturation=self.maxSatSlider.value(),
      min_thickness=self.minThickSlider.value(),
      dilation=self.dilationSlider.value(),
    )

  def _run_auto_detect(self, rect: tuple[int, int, int, int]):
    if self.rgb_array is None:
      return
    params = self.get_detection_params()
    clamped, roi_mask = detect_censor_in_rect(self.rgb_array, rect, params)
    self.canvas.set_pending_detection(clamped, roi_mask)
    detected_px = int(roi_mask.sum())
    self._set_status_message(t("mask_editor.status_detected", count=detected_px))
    self._update_ui_state()

  def _on_detect_params_changed(self):
    self._update_detect_slider_enabled()
    if self.canvas.pending_rect is not None and self.rgb_array is not None:
      self._run_auto_detect(self.canvas.pending_rect)

  def _update_detect_slider_enabled(self):
    use_black = self.detectBlackRadio.isChecked() or self.detectBothRadio.isChecked()
    use_white = self.detectWhiteRadio.isChecked() or self.detectBothRadio.isChecked()
    self.blackThreshSlider.setEnabled(use_black)
    self.whiteThreshSlider.setEnabled(use_white)

  def _commit_pending(self):
    if self.canvas.has_pending():
      self.canvas.commit_pending()
      self._set_status_message(t("mask_editor.status_applied_mask"))
      self._update_ui_state()

  def _cancel_pending(self):
    if self.canvas.has_pending():
      self.canvas.clear_pending()
      self._set_status_message(t("mask_editor.status_canceled_detection"))
      self._update_ui_state()

  def _on_pending_changed(self, has_pending: bool):
    self.applyPendingButton.setEnabled(has_pending)
    self.cancelPendingButton.setEnabled(has_pending)

  def _on_brush_size_changed(self, value: int):
    self.canvas.set_brush_size(value)

  def _on_opacity_changed(self, value: int):
    self.canvas.set_mask_opacity(value / 100.0)

  def _on_zoom_changed(self, zoom: float):
    self.zoomLabel.setText(f"{round(zoom * 100)}%")

  def _undo(self):
    if self.canvas.view_mode != "edit":
      self._set_view_mode("edit")
    self.canvas.undo()

  def _redo(self):
    if self.canvas.view_mode != "edit":
      self._set_view_mode("edit")
    self.canvas.redo()

  def _clear_mask(self):
    if self.canvas.view_mode != "edit":
      self._set_view_mode("edit")
    self.canvas.clear_mask()

  # --- View switching & Results ---

  def _set_view_mode(self, mode: str):
    if mode == "result" and not self.result_images:
      self.editViewButton.setChecked(True)
      return
    self.canvas.set_comparing_before(False)
    self.canvas.set_view_mode(mode)
    self.editViewButton.setChecked(mode == "edit")
    self.resultViewButton.setChecked(mode == "result")
    self._update_ui_state()

  def _select_variation(self, index: int):
    if 0 <= index < len(self.result_images):
      self.canvas.set_result_pixmap(pil_to_qpixmap(self.result_images[index]))

  # --- Decensor execution & Progress ---

  def set_busy(
    self,
    busy: bool,
    reason: str = "",
    *,
    can_cancel: bool = False,
    reason_key: str = "",
  ):
    self._busy = busy
    self._busy_reason_key = reason_key if busy else ""
    self._busy_reason = t(reason_key) if (busy and reason_key) else reason
    self._can_cancel = bool(busy and can_cancel)
    if not busy:
      self._cancelling = False
    if busy and self._busy_reason:
      self._set_status_message(self._busy_reason)
    self._update_ui_state()

  def set_cancelling(self):
    self._cancelling = True
    self._set_status_message(t("mask_editor.status_cancelling"))
    self._update_ui_state()

  def set_status_text(self, text: str):
    if text:
      self._set_status_message(text)

  def set_progress_range(self, minimum: int, maximum: int):
    self.progressBar.setRange(minimum, maximum)

  def set_progress_value(self, value: int):
    self.progressBar.setValue(value)

  def _on_cancel_clicked(self):
    if self._busy and self._can_cancel and not self._cancelling:
      self.set_cancelling()
      self.cancel_requested.emit()

  def _build_colored_image(self) -> Image.Image:
    assert self.original_image is not None
    assert self.canvas.mask is not None
    return apply_mask_to_image(self.original_image, self.canvas.mask, self.mask_color)

  def _on_decensor_clicked(self):
    if self.original_image is None or self.canvas.mask is None or self._busy:
      return
    if self.canvas.has_pending():
      self._commit_pending()
    if not self.canvas.mask.any():
      QMessageBox.information(
        self,
        t("mask_editor.dialog_no_mask_title"),
        t("mask_editor.dialog_no_mask_body"),
      )
      return

    colored_img = self._build_colored_image()
    self.progressBar.setValue(0)
    self.decensor_requested.emit(self.original_image, colored_img, self.output_stem)

  def on_decensor_succeeded(self, results: list, saved_paths: list):
    self.result_images = list(results)
    self.saved_paths = list(saved_paths)
    self.variationCombo.blockSignals(True)
    self.variationCombo.clear()
    for i in range(len(self.result_images)):
      self.variationCombo.addItem(t("mask_editor.variation_item", index=i))
    self.variationCombo.setCurrentIndex(0)
    self.variationCombo.blockSignals(False)

    if self.result_images:
      self.canvas.set_result_pixmap(pil_to_qpixmap(self.result_images[0]))
      self._set_view_mode("result")
      if len(self.saved_paths) == 1:
        self._set_status_message(
          t("mask_editor.status_saved_single", path=self.saved_paths[0])
        )
      else:
        self._set_status_message(
          t(
            "mask_editor.status_saved_multiple",
            count=len(self.saved_paths),
            directory=Path(self.saved_paths[0]).parent,
          )
        )
    self._update_ui_state()

  def on_decensor_failed(self, error_message: str):
    self._set_status_message(
      t("mask_editor.status_decensor_failed", error=error_message)
    )
    QMessageBox.warning(
      self, t("mask_editor.dialog_decensor_failed_title"), error_message
    )
    self._update_ui_state()

  def on_decensor_cancelled(self):
    self._cancelling = False
    self._set_status_message(t("mask_editor.status_decensor_cancelled"))
    self._update_ui_state()

  # --- Manual Save actions ---

  def save_result_as(self):
    if not self.result_images:
      return
    idx = max(0, self.variationCombo.currentIndex())
    result = self.result_images[idx]
    default_name = f"{self.output_stem} {idx}.png"
    default_dir = str(self.source_path.parent) if self.source_path else ""
    initial = str(Path(default_dir) / default_name) if default_dir else default_name
    file_path, _ = QFileDialog.getSaveFileName(
      self,
      t("mask_editor.dialog_save_result_title"),
      initial,
      t("mask_editor.dialog_save_result_filter"),
    )
    if not file_path:
      return
    try:
      out_path = Path(file_path)
      img_to_save = result
      if out_path.suffix.lower() in (".jpg", ".jpeg"):
        img_to_save = result.convert("RGB")
      img_to_save.save(out_path)
      self._set_status_message(t("mask_editor.status_saved_result_to", path=out_path))
    except OSError as error:
      QMessageBox.warning(self, t("mask_editor.dialog_save_failed_title"), str(error))

  def save_masked_png(self):
    if self.original_image is None or self.canvas.mask is None:
      return
    if self.canvas.has_pending():
      self._commit_pending()
    default_name = f"{self.output_stem}_masked.png"
    default_dir = str(self.source_path.parent) if self.source_path else ""
    initial = str(Path(default_dir) / default_name) if default_dir else default_name
    file_path, _ = QFileDialog.getSaveFileName(
      self,
      t("mask_editor.dialog_save_mask_title"),
      initial,
      t("mask_editor.dialog_save_mask_filter"),
    )
    if not file_path:
      return
    try:
      colored = self._build_colored_image()
      colored.save(file_path, format="PNG")
      self._set_status_message(t("mask_editor.status_saved_mask_to", path=file_path))
    except OSError as error:
      QMessageBox.warning(self, t("mask_editor.dialog_save_failed_title"), str(error))

  # --- UI State ---

  def _update_ui_state(self):
    has_img = self.canvas.has_image()
    has_results = bool(self.result_images)
    in_result_view = self.canvas.view_mode == "result" and has_results

    self.undoButton.setEnabled(has_img and self.canvas.can_undo())
    self.redoButton.setEnabled(has_img and self.canvas.can_redo())
    self.clearMaskButton.setEnabled(has_img and self.canvas.has_any_mask())
    self.zoomOutButton.setEnabled(has_img)
    self.fitButton.setEnabled(has_img)
    self.zoomInButton.setEnabled(has_img)

    self.editViewButton.setEnabled(has_img)
    self.resultViewButton.setEnabled(has_results)
    self.compareButton.setEnabled(in_result_view)
    self.variationCombo.setEnabled(has_results and len(self.result_images) > 1)

    self.applyPendingButton.setEnabled(self.canvas.has_pending())
    self.cancelPendingButton.setEnabled(self.canvas.has_pending())
    self._update_detect_slider_enabled()

    self.decensorButton.setEnabled(has_img and not self._busy)
    if self._busy:
      self.decensorButton.setText(
        self._busy_reason or t("mask_editor.status_processing")
      )
    else:
      self.decensorButton.setText(t("mask_editor.decensor_button"))

    self.cancelButton.setEnabled(
      self._busy and self._can_cancel and not self._cancelling
    )
    self.cancelButton.setText(
      t("mask_editor.cancel_button_cancelling")
      if self._cancelling
      else t("mask_editor.cancel_button")
    )

    self.saveResultButton.setEnabled(has_results)
    self.saveMaskButton.setEnabled(has_img)

  # --- Window-level Drag & Drop ---

  def dragEnterEvent(self, event):
    if _extract_dropped_image_path(event.mimeData()) is not None:
      event.acceptProposedAction()
    else:
      event.ignore()

  def dropEvent(self, event):
    path = _extract_dropped_image_path(event.mimeData())
    if path is not None:
      self.load_image_from_path(str(path))
      event.acceptProposedAction()
