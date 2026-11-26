import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annotations import read_annotations, save_image_boxes
from train_bar import BarDataset, sample_bar, split_samples


class TrainingTests(unittest.TestCase):
  def test_annotations_keep_other_images_and_detect_conflicting_edits(self):
    with tempfile.TemporaryDirectory() as directory:
      csv_path = Path(directory) / "people.csv"
      first = Path("chapter/first, page.png")
      second = Path("second.jpg")
      boxes = [(3, 4, 30, 40), (10, 20, 15, 35)]
      save_image_boxes(csv_path, first, [], boxes)
      save_image_boxes(csv_path, second, [], boxes[:1])
      with self.assertRaisesRegex(ValueError, "CSV changed"):
        save_image_boxes(csv_path, first, [], [])
      self.assertEqual(read_annotations(csv_path), {first: boxes, second: boxes[:1]})
      save_image_boxes(csv_path, first, boxes, [])
      self.assertEqual(read_annotations(csv_path), {second: boxes[:1]})

  def test_validation_keeps_annotations_for_an_image_together(self):
    boxes = [(3, 4, 30, 40), (10, 20, 15, 35)]
    annotations = {Path(f"{i}.png"): boxes for i in range(10)}
    train, validation = split_samples(annotations, 42)
    self.assertEqual((len(train), len(validation)), (18, 2))
    self.assertTrue(
      {name for name, _ in train}.isdisjoint(name for name, _ in validation)
    )

  def test_bar_and_validation_sample_match_training_input_conventions(self):
    size = (400, 300)
    person = (50, 40, 200, 180)
    bar = sample_bar(size, person, random.Random(42))
    left, top, right, bottom = bar
    self.assertTrue(person[0] <= left < right <= person[2])
    self.assertTrue(person[1] <= top < bottom <= person[3])
    short, long = sorted((right - left, bottom - top))
    self.assertTrue(4 <= short <= 12 and 24 <= long <= 64)

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      name = Path("image.png")
      Image.new("RGB", size, "white").save(root / name)
      dataset = BarDataset(root, [(name, person)], 42)
      image, target, known, scored = dataset[0]
      self.assertEqual(image.shape, (3, 256, 256))
      self.assertTrue(scored.any().item())
      self.assertTrue(torch.equal(image, torch.where(known.bool(), target, -1)))
      self.assertTrue(torch.equal(scored, dataset[0][3]))


if __name__ == "__main__":
  unittest.main()
