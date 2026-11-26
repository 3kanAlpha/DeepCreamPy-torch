"""Draw person bounding boxes and save them in the training CSV."""

import argparse
import csv
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
  QApplication,
  QHBoxLayout,
  QLabel,
  QListWidget,
  QMessageBox,
  QPushButton,
  QVBoxLayout,
  QWidget,
)

from annotations import (
  IMAGE_SUFFIXES,
  read_annotations,
  save_image_boxes,
  validate_boxes,
)


class ImageCanvas(QWidget):
  box_drawn = Signal(tuple)

  def __init__(self):
    super().__init__()
    self.pixmap = QPixmap()
    self.boxes = []
    self.selected = -1
    self.start = self.end = None
    self.setMinimumSize(320, 240)

  def show_image(self, pixmap, boxes):
    self.pixmap = pixmap
    self.boxes = boxes
    self.start = self.end = None
    self.update()

  def image_rect(self):
    width, height = self.pixmap.width(), self.pixmap.height()
    scale = min(self.width() / width, self.height() / height)
    width, height = width * scale, height * scale
    return QRectF(
      (self.width() - width) / 2, (self.height() - height) / 2, width, height
    )

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.fillRect(self.rect(), QColor("#252525"))
    if self.pixmap.isNull():
      return
    area = self.image_rect()
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.drawPixmap(area, self.pixmap, QRectF(self.pixmap.rect()))
    painter.translate(area.topLeft())
    painter.scale(
      area.width() / self.pixmap.width(), area.height() / self.pixmap.height()
    )
    for index, (left, top, right, bottom) in enumerate(self.boxes):
      painter.setPen(self.box_pen("#ffcc40" if index == self.selected else "#52e7ff"))
      painter.drawRect(QRectF(left, top, right - left, bottom - top))
    if self.start is not None:
      painter.setPen(self.box_pen("#ffcc40"))
      painter.drawRect(QRectF(self.start, self.end).normalized())

  @staticmethod
  def box_pen(color):
    pen = QPen(QColor(color), 2)
    pen.setCosmetic(True)
    return pen

  def image_point(self, position):
    area = self.image_rect()
    x = (position.x() - area.left()) * self.pixmap.width() / area.width()
    y = (position.y() - area.top()) * self.pixmap.height() / area.height()
    return QPointF(
      min(max(x, 0), self.pixmap.width()), min(max(y, 0), self.pixmap.height())
    )

  def mousePressEvent(self, event):
    if (
      event.button() == Qt.MouseButton.LeftButton
      and not self.pixmap.isNull()
      and self.image_rect().contains(event.position())
    ):
      self.start = self.end = self.image_point(event.position())

  def mouseMoveEvent(self, event):
    if self.start is not None:
      self.end = self.image_point(event.position())
      self.update()

  def mouseReleaseEvent(self, event):
    if event.button() != Qt.MouseButton.LeftButton or self.start is None:
      return
    box = QRectF(self.start, self.image_point(event.position())).normalized()
    self.start = self.end = None
    left, top, right, bottom = (
      round(coordinate)
      for coordinate in (box.left(), box.top(), box.right(), box.bottom())
    )
    if left < right and top < bottom:
      self.box_drawn.emit((left, top, right, bottom))
    self.update()


class AnnotationWindow(QWidget):
  def __init__(self, root, images, csv_path, annotations):
    super().__init__()
    self.root, self.images, self.csv_path = root, images, csv_path
    self.annotations = annotations
    self.index = next(
      (i for i, image in enumerate(images) if image not in annotations), 0
    )
    self.setWindowTitle("人物矩形のアノテーション")
    self.resize(1000, 720)

    self.caption = QLabel()
    self.caption.setWordWrap(True)
    self.canvas = ImageCanvas()
    self.canvas.box_drawn.connect(self.draw_box)
    self.rectangles = QListWidget()
    self.rectangles.setAccessibleName("この画像の矩形")
    self.rectangles.currentRowChanged.connect(self.select_box)
    self.redraw = QPushButton("選択した矩形を描き直す")
    self.redraw.setCheckable(True)
    self.redraw.toggled.connect(self.show_instruction)
    self.delete = QPushButton("選択した矩形を削除")
    self.delete.clicked.connect(self.delete_box)
    self.instruction = QLabel()
    self.instruction.setWordWrap(True)

    sidebar = QVBoxLayout()
    sidebar.addWidget(self.instruction)
    sidebar.addWidget(self.rectangles)
    sidebar.addWidget(self.redraw)
    sidebar.addWidget(self.delete)
    content = QHBoxLayout()
    content.addWidget(self.canvas, 3)
    content.addLayout(sidebar, 1)

    self.previous = QPushButton("前の画像")
    self.previous.setShortcut("Left")
    self.previous.clicked.connect(lambda: self.open_image(self.index - 1))
    self.next = QPushButton("次の画像")
    self.next.setShortcut("Right")
    self.next.clicked.connect(lambda: self.open_image(self.index + 1))
    navigation = QHBoxLayout()
    navigation.addWidget(self.previous)
    navigation.addWidget(self.next)

    layout = QVBoxLayout(self)
    layout.addWidget(self.caption)
    layout.addLayout(content, 1)
    layout.addLayout(navigation)
    self.open_image(self.index)

  @property
  def image(self):
    return self.images[self.index]

  def open_image(self, index):
    if not 0 <= index < len(self.images):
      return
    self.index = index
    self.caption.setText(f"{index + 1} / {len(self.images)}   {self.image.as_posix()}")
    self.previous.setEnabled(index > 0)
    self.next.setEnabled(index + 1 < len(self.images))
    pixmap = QPixmap()
    try:
      with Image.open(self.root / self.image) as image:
        validate_boxes(self.image, self.annotations.get(self.image, []), image.size)
        pixmap = QPixmap.fromImage(image.convert("RGB").toqimage())
    except (OSError, ValueError, Image.DecompressionBombError) as error:
      QMessageBox.warning(self, "画像を開けません", f"{self.image}\n{error}")
    self.canvas.show_image(pixmap, [])
    self.refresh_boxes()

  def refresh_boxes(self, selected=-1):
    boxes = (
      self.annotations.get(self.image, []) if not self.canvas.pixmap.isNull() else []
    )
    self.canvas.boxes = boxes
    self.rectangles.clear()
    self.rectangles.addItems([", ".join(map(str, box)) for box in boxes])
    self.rectangles.setCurrentRow(selected)
    self.select_box(selected)

  def select_box(self, index):
    self.canvas.selected = index
    self.canvas.update()
    self.redraw.setChecked(False)
    self.redraw.setEnabled(index >= 0)
    self.delete.setEnabled(index >= 0)
    self.show_instruction(False)

  def show_instruction(self, replacing):
    text = (
      "新しい範囲をドラッグすると、選択した矩形を置き換えます。"
      if replacing
      else "人物をドラッグで囲みます。編集内容は CSV に自動保存されます。"
    )
    self.instruction.setText(text)

  def draw_box(self, box):
    boxes = self.canvas.boxes.copy()
    if self.redraw.isChecked():
      index = self.rectangles.currentRow()
      boxes[index] = box
    else:
      index = len(boxes)
      boxes.append(box)
    self.save_boxes(boxes, index)

  def delete_box(self):
    index = self.rectangles.currentRow()
    if index >= 0:
      boxes = self.canvas.boxes.copy()
      del boxes[index]
      self.save_boxes(boxes, min(index, len(boxes) - 1))

  def save_boxes(self, boxes, index):
    try:
      self.annotations = save_image_boxes(
        self.csv_path, self.image, self.canvas.boxes, boxes
      )
    except (OSError, ValueError, csv.Error) as error:
      QMessageBox.warning(self, "CSV を保存できません", str(error))
      return
    self.refresh_boxes(index)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--images", type=Path, required=True, help="directory of clean images"
  )
  parser.add_argument(
    "--annotations", type=Path, required=True, help="CSV to create or edit"
  )
  args = parser.parse_args()
  root = args.images.resolve()
  if not root.is_dir():
    parser.error(f"Image directory not found: {root}")
  if args.annotations.suffix.lower() != ".csv":
    parser.error("--annotations must be a .csv file")
  try:
    images = sorted(
      path.relative_to(root)
      for path in root.rglob("*")
      if path.suffix.lower() in IMAGE_SUFFIXES
      and path.is_file()
      and path.resolve().is_relative_to(root)
    )
    annotations = (
      read_annotations(args.annotations) if args.annotations.exists() else {}
    )
  except (OSError, ValueError, csv.Error) as error:
    parser.error(str(error))
  if not images:
    parser.error(f"No PNG, JPEG, or WebP images found in {root}")
  app = QApplication([])
  window = AnnotationWindow(root, images, args.annotations, annotations)
  window.show()
  return app.exec()


if __name__ == "__main__":
  raise SystemExit(main())
