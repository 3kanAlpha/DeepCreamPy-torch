import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config
from convert_lama import convert_checkpoint
from decensor import Decensor
from lama import FFCResNetGenerator, infer_lama_config
from libs.utils import (
  bridge_thin_gaps,
  connected_regions,
  expand_bounding,
  prepare_crop,
  prepare_lama_crop,
)
from model import InpaintModel, LamaInpainter, create_inpainter


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
    features = torch.zeros(1, 256, 5, 5, requires_grad=True)
    known = torch.ones(1, 3, 40, 40)
    filled = model.contextual_block(features, known)
    self.assertTrue(torch.isfinite(filled).all().item())
    filled.sum().backward()
    self.assertTrue(torch.isfinite(features.grad).all().item())

  def test_lama_big_lama_structure_and_gradient(self):
    big_lama = FFCResNetGenerator()
    state = big_lama.state_dict()
    self.assertEqual(len(state), 989)
    inferred = infer_lama_config(state)
    self.assertEqual(
      inferred,
      {
        "input_nc": 4,
        "ngf": 64,
        "n_downsampling": 3,
        "n_blocks": 18,
        "ratio_g": 0.75,
        "enable_lfu": False,
      },
    )

    small_lama = FFCResNetGenerator(ngf=16, n_blocks=2)
    image = torch.rand(1, 3, 32, 40, requires_grad=True)
    mask = torch.zeros(1, 1, 32, 40)
    mask[:, :, 8:20, 10:26] = 1.0
    output = small_lama(image, mask)
    self.assertEqual(output.shape, image.shape)
    self.assertTrue((output >= 0.0).all().item() and (output <= 1.0).all().item())
    self.assertTrue(
      torch.equal(
        output[mask.expand_as(output) == 0], image[mask.expand_as(output) == 0]
      )
    )
    output.sum().backward()
    self.assertTrue(torch.isfinite(image.grad).all().item())

  def test_prepare_lama_crop_padding_and_downscaling(self):
    image = Image.new("RGB", (300, 220), (110, 95, 170))
    marked = np.zeros((220, 300), dtype=bool)
    marked[50:70, 80:110] = True
    rgb, missing, unpadded = prepare_lama_crop(image, marked, (10, 10, 260, 200))
    self.assertEqual(rgb.shape, (192, 256, 3))
    self.assertEqual(missing.shape, (192, 256))
    self.assertEqual(rgb[unpadded].shape, (190, 250, 3))

    large = Image.new("RGB", (1000, 1000), (110, 95, 170))
    large_marked = np.zeros((1000, 1000), dtype=bool)
    large_marked[300:701, 500] = True
    rgb_large, missing_large, unpadded_large = prepare_lama_crop(
      large, large_marked, (0, 0, 1000, 1000), max_size=512
    )
    self.assertEqual(rgb_large.shape, (512, 512, 3))
    self.assertTrue(missing_large[unpadded_large].any())

    tiny = Image.new("RGB", (1, 1), (10, 20, 30))
    tiny_marked = np.ones((1, 1), dtype=bool)
    rgb_tiny, missing_tiny, unpadded_tiny = prepare_lama_crop(
      tiny, tiny_marked, (0, 0, 1, 1)
    )
    self.assertEqual(rgb_tiny.shape, (8, 8, 3))
    self.assertEqual(rgb_tiny[unpadded_tiny].shape, (1, 1, 3))
    self.assertTrue(missing_tiny[unpadded_tiny].all())

  def test_lama_conversion_and_decensor_integration(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_ckpt = root / "source.ckpt"
      converted = root / "inpaint.pt"
      net = FFCResNetGenerator(ngf=16, n_blocks=2)
      torch.save(
        {"state_dict": {f"generator.{k}": v for k, v in net.state_dict().items()}},
        source_ckpt,
      )
      info = convert_checkpoint(source_ckpt, converted)
      self.assertEqual(info["n_blocks"], 2)
      self.assertEqual(info["ngf"], 16)

      # Also verify Lightning-like checkpoints referencing missing external modules.
      fake_cls = type("FakeCallback", (), {"__module__": "nonexistent_lightning"})
      fake_mod = type("M", (), {"FakeCallback": fake_cls})()
      lightning_ckpt = root / "lightning.ckpt"
      with patch.dict(sys.modules, {"nonexistent_lightning": fake_mod}):
        torch.save(
          {
            "callbacks": [fake_cls()],
            "state_dict": {f"generator.{k}": v for k, v in net.state_dict().items()},
          },
          lightning_ckpt,
        )
      converted_lightning = root / "inpaint_lightning.pt"
      info_lightning = convert_checkpoint(lightning_ckpt, converted_lightning)
      self.assertEqual(info_lightning["n_blocks"], 2)

      pixels = np.full((120, 120, 4), (70, 80, 90, 200), dtype=np.uint8)
      marked = np.zeros((120, 120), dtype=bool)
      marked[40:52, 45:65] = True
      marked_pixels = pixels.copy()
      marked_pixels[marked, :3] = (0, 255, 0)
      marked_image = Image.fromarray(marked_pixels)

      with patch.object(sys, "argv", [sys.argv[0], "--model", "lama"]):
        decensor = Decensor(cli=True)
      decensor.model = LamaInpainter(weights_path=converted)
      with contextlib.redirect_stdout(io.StringIO()):
        result = np.asarray(
          decensor.decensor_image_variation(marked_image, marked_image, 0, None)
        )
      np.testing.assert_array_equal(result[~marked], pixels[~marked])
      np.testing.assert_array_equal(result[..., 3], pixels[..., 3])

  def test_lama_rejected_in_mosaic_mode(self):
    with self.assertRaisesRegex(ValueError, "bar mode"):
      create_inpainter("lama", is_mosaic=True)

    with (
      patch.object(
        sys, "argv", [sys.argv[0], "--model", "lama", "--is_mosaic", "true"]
      ),
      contextlib.redirect_stderr(io.StringIO()),
    ):
      with self.assertRaises(SystemExit) as ctx:
        config.get_args(cli=True)
      self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
  unittest.main()
