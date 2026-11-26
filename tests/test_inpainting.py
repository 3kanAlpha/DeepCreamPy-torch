import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decensor import Decensor
from libs.utils import (
  bridge_thin_gaps,
  connected_regions,
  expand_bounding,
  prepare_crop,
)
from model import InpaintModel


class RecordingModel:
  batch_size = 4

  def __init__(self):
    self.masks = []

  def predict(self, image, original, mask):
    self.masks.extend(mask)
    return np.where(mask, original, 0.0).astype(np.float32)


class InpaintingTests(unittest.TestCase):
  def test_crop_contains_every_mark(self):
    cases = [
      ((512, 512), {(511, 200), (511, 210)}),
      ((512, 512), {(200, 511), (210, 511)}),
      ((1000, 200), {(250, 90), (749, 109)}),
      ((180, 140), {(170, 70), (170, 80)}),
      ((1, 1), {(0, 0)}),
    ]
    for size, region in cases:
      with self.subTest(size=size, region=region):
        left, top, right, bottom = expand_bounding(Image.new("RGB", size), region)
        self.assertTrue(0 <= left < right <= size[0])
        self.assertTrue(0 <= top < bottom <= size[1])
        self.assertTrue(all(left <= x < right and top <= y < bottom for x, y in region))

  def test_thin_mask_survives_downsampling(self):
    image = Image.new("RGB", (1000, 1000), (110, 95, 170))
    marked = np.zeros((1000, 1000), dtype=bool)
    marked[300:701, 500] = True
    (region,) = connected_regions(marked)
    _, missing, unpadded = prepare_crop(image, marked, expand_bounding(image, region))
    self.assertTrue(missing[unpadded].any())

  def test_edge_padding_preserves_image_and_mask_alignment(self):
    marked = np.zeros((160, 512), dtype=bool)
    marked[40:60, -390:] = True
    pixels = np.full((160, 512, 3), (110, 95, 170), dtype=np.uint8)
    pixels[marked] = (0, 255, 0)
    image = Image.fromarray(pixels)
    (region,) = connected_regions(marked)
    rgb, missing, unpadded = prepare_crop(image, marked, expand_bounding(image, region))
    self.assertEqual(rgb.shape, (256, 256, 3))
    self.assertEqual(missing[unpadded].shape[1], 240)
    self.assertTrue(missing[unpadded][:, -1].any())
    green = (rgb == (0, 1, 0)).all(axis=-1)
    self.assertTrue(green.any())
    self.assertTrue(missing[green].all())

  def test_paste_back_changes_only_marked_rgb(self):
    pixels = np.full((300, 300, 4), (70, 70, 70, 0), dtype=np.uint8)
    pixels[..., 3] = np.arange(300, dtype=np.uint16) % 256
    marked = np.zeros((300, 300), dtype=bool)
    marked[110:112, 120:122] = True
    marked[140:142, 160:162] = True
    source = Image.fromarray(pixels)
    marked_pixels = pixels.copy()
    marked_pixels[marked, :3] = (0, 255, 0)
    marked_image = Image.fromarray(marked_pixels)
    with patch.object(sys, "argv", [sys.argv[0]]):
      decensor = Decensor()

    for mosaic in (False, True):
      with self.subTest(mosaic=mosaic):
        decensor.model = RecordingModel()
        decensor.is_mosaic = mosaic
        original = source if mosaic else marked_image
        with contextlib.redirect_stdout(io.StringIO()):
          result = np.asarray(
            decensor.decensor_image_variation(original, marked_image, 3, None)
          )

        self.assertEqual(len(decensor.model.masks), 2)
        np.testing.assert_array_equal(result[~marked], pixels[~marked])
        np.testing.assert_array_equal(result[..., 3], pixels[..., 3])
        np.testing.assert_array_equal(result[marked, :3], np.full((8, 3), 128))

  def test_matching_neighbors_continue_a_line_across_a_thin_gap(self):
    source = np.full((7, 7, 3), 200, dtype=np.uint8)
    source[3, :] = 25
    marked = np.zeros((7, 7), dtype=bool)
    marked[2:5, 3] = True
    prediction = source.copy()
    prediction[marked] = 115
    bridge_thin_gaps(source, marked, prediction)
    np.testing.assert_array_equal(prediction, source)

  def test_uniform_attention_candidates_are_finite(self):
    model = InpaintModel()
    features = torch.zeros(1, 256, 5, 5)
    known = torch.ones(1, 3, 40, 40)
    with torch.inference_mode():
      filled = model.contextual_block(features, known)
    self.assertTrue(torch.isfinite(filled).all().item())


if __name__ == "__main__":
  unittest.main()
