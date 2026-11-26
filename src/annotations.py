import csv
from pathlib import Path

from libs.outputs import atomic_output, output_lock

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CSV_COLUMNS = ("image", "x_min", "y_min", "x_max", "y_max")


def read_annotations(path):
  annotations = {}
  with path.open(newline="", encoding="utf-8-sig") as stream:
    reader = csv.DictReader(stream, strict=True)
    if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
      raise ValueError(f"{path}: expected CSV header {','.join(CSV_COLUMNS)}")
    for row in reader:
      try:
        if None in row or not all(row.values()):
          raise ValueError("expected five nonempty CSV fields")
        image = Path(row["image"])
        if (
          image.anchor
          or ".." in image.parts
          or image.suffix.lower() not in IMAGE_SUFFIXES
        ):
          raise ValueError(
            "image must be a supported path relative to the image directory"
          )
        left, top, right, bottom = (int(row[key]) for key in CSV_COLUMNS[1:])
        if not (0 <= left < right and 0 <= top < bottom):
          raise ValueError("the rectangle must have positive width and height")
      except ValueError as error:
        raise ValueError(f"{path}, line {reader.line_num}: {error}") from error
      annotations.setdefault(image, []).append((left, top, right, bottom))
  return annotations


def validate_boxes(image, boxes, size):
  width, height = size
  for box in boxes:
    if box[2] > width or box[3] > height:
      raise ValueError(
        f"{image}: rectangle {box} is outside the {width}x{height} image"
      )


def save_image_boxes(path, image, previous, boxes):
  path.parent.mkdir(parents=True, exist_ok=True)
  with output_lock(path.with_name(f".{path.name}.lock")):
    annotations = read_annotations(path) if path.exists() else {}
    if annotations.get(image, []) != previous:
      raise ValueError(
        f"{image}: CSV changed since opening this image. Restart the tool."
      )
    if boxes:
      annotations[image] = boxes
    else:
      annotations.pop(image, None)

    with (
      atomic_output(path) as temporary,
      temporary.open("w", newline="", encoding="utf-8") as stream,
    ):
      writer = csv.writer(stream)
      writer.writerow(CSV_COLUMNS)
      for name, rectangles in sorted(annotations.items()):
        writer.writerows((name.as_posix(), *box) for box in rectangles)
  return annotations
