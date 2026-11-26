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
  mask_pixels,
  prepare_crop,
  prepare_lama_crop,
  seamless_blend_region,
  suppress_monochrome_chroma,
)
from model import InpaintModel, InpaintNN, LamaInpainter, create_inpainter


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

  def test_mask_pixels_absorbs_antialiased_fringe_locally(self):
    pixels = np.full((24, 24, 3), (220, 190, 175), dtype=np.uint8)
    pixels[8:16, 8:16] = (0, 255, 0)
    # Anti-aliased green fringe (1px and 2px out) touching the exact mask.
    pixels[7, 8:16] = (120, 255, 120)
    pixels[6, 10:14] = (150, 230, 150)
    # Mild mint-green below chroma threshold touching the exact mask should remain kept.
    pixels[16, 8:16] = (196, 247, 222)
    # Unconnected green object far from the exact mask should never be absorbed.
    pixels[1:4, 1:4] = (10, 220, 10)
    image = Image.fromarray(pixels)

    exact = mask_pixels(image, [0, 255, 0], absorb_fringe=False)
    absorbed = mask_pixels(image, [0, 255, 0], absorb_fringe=True)

    self.assertEqual(int(exact.sum()), 64)
    self.assertTrue(absorbed[7, 8:16].all())
    self.assertTrue(absorbed[6, 10:14].all())
    self.assertFalse(absorbed[16, 8:16].any())
    self.assertFalse(absorbed[1:4, 1:4].any())

    tiny = Image.new("RGB", (1, 1), (0, 255, 0))
    self.assertTrue(mask_pixels(tiny, [0, 255, 0]).all())

  def test_prepare_crop_mask_aware_resampling_avoids_green_bleed(self):
    # Downscaling case (512 -> 256): anti-aliased averaging without green bleed.
    down = np.full((512, 512, 3), (200, 180, 160), dtype=np.uint8)
    down[:, ::2] = (40, 40, 40)
    marked_down = np.zeros((512, 512), dtype=bool)
    marked_down[220:290, 220:290] = True
    down[marked_down] = (0, 255, 0)
    rgb_down, missing_down, _ = prepare_crop(
      Image.fromarray(down), marked_down, (0, 0, 512, 512)
    )
    valid_down = rgb_down[~missing_down]
    self.assertTrue(
      np.allclose(valid_down, (120 / 255, 110 / 255, 100 / 255), atol=0.02)
    )

    # Upscaling case (128 -> 256): smooth gradient on valid pixels without green bleed.
    grad = np.linspace(40, 200, 128, dtype=np.uint8)
    up = np.broadcast_to(grad[None, :, None], (128, 128, 3)).copy()
    marked_up = np.zeros((128, 128), dtype=bool)
    marked_up[50:78, 50:78] = True
    up[marked_up] = (0, 255, 0)
    rgb_up, missing_up, _ = prepare_crop(
      Image.fromarray(up), marked_up, (0, 0, 128, 128)
    )
    valid_up = rgb_up[~missing_up]
    # Pure green would spike G - R well above 0; grayscale gradient keeps G == R.
    self.assertLess(float(np.max(np.abs(valid_up[:, 1] - valid_up[:, 0]))), 1e-4)

  def test_seamless_blend_region_corrects_offset_and_clamps_spikes(self):
    source = np.full((32, 32, 3), 200.0, dtype=np.float32)
    raw = np.full((32, 32, 3), 185.0, dtype=np.float32)
    region = np.zeros((32, 32), dtype=bool)
    region[10:22, 10:22] = True
    known = ~region

    blended = seamless_blend_region(raw, source, region, known)
    # Boundary pixels of the region should closely match source (200) instead of 185.
    self.assertGreater(float(blended[10, 15, 0]), 198.0)
    self.assertLess(float(blended[10, 15, 0]), 201.0)
    # Non-region pixels remain unchanged.
    np.testing.assert_array_equal(blended[known], raw[known])

    # Extreme single-pixel spike on the ring is bounded by max_shift.
    spiky_source = source.copy()
    spiky_source[9, 15] = 0.0
    blended_spiky = seamless_blend_region(
      raw, spiky_source, region, known, max_shift=20.0
    )
    self.assertGreater(float(blended_spiky[region].min()), 164.0)

  def test_suppress_monochrome_chroma_handles_grayscale_and_color(self):
    region = np.zeros((20, 20), dtype=bool)
    region[6:14, 6:14] = True
    known = ~region

    gray_source = np.full((20, 20, 3), 180.0, dtype=np.float32)
    tinted_pred = np.full((20, 20, 3), (170.0, 200.0, 165.0), dtype=np.float32)
    cleaned = suppress_monochrome_chroma(tinted_pred, gray_source, region, known)
    spread = cleaned[region].max(axis=1) - cleaned[region].min(axis=1)
    self.assertLess(float(spread.max()), 1e-4)

    color_source = np.full((20, 20, 3), (240.0, 200.0, 180.0), dtype=np.float32)
    untouched = suppress_monochrome_chroma(tinted_pred, color_source, region, known)
    np.testing.assert_array_equal(untouched, tinted_pred)

  def test_pepsi_tta_alignment_chunking_and_cli_flags(self):
    # Build a lightweight InpaintNN instance with a deterministic identity raw predictor
    # to verify exact spatial alignment across flip + shift views and batch chunking.
    inpainter = InpaintNN.__new__(InpaintNN)
    inpainter.batch_size = 3
    inpainter.use_tta = True
    call_sizes = []

    def fake_predict_raw(images, masks):
      call_sizes.append(len(images))
      return images.astype(np.float32)

    inpainter.predict_raw = fake_predict_raw

    rng = np.random.default_rng(7)
    images = rng.uniform(-1.0, 1.0, size=(2, 32, 32, 3)).astype(np.float32)
    masks = np.ones((2, 32, 32, 3), dtype=np.float32)

    averaged = inpainter.predict_raw_tta(images, masks)
    np.testing.assert_allclose(averaged, images, atol=1e-6)
    # 2 crops x 4 views = 8 items, chunked by batch_size=3 -> [3, 3, 2].
    self.assertEqual(call_sizes, [3, 3, 2])

    # Asymmetric predictor should become horizontally equivariant after TTA.
    def asymmetric_predict_raw(batch_images, batch_masks):
      width = batch_images.shape[2]
      ramp = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, None, :, None]
      return (batch_images * 0.5 + ramp).astype(np.float32)

    inpainter.predict_raw = asymmetric_predict_raw
    out_orig = inpainter.predict_raw_tta(images[:1], masks[:1])[0]
    out_flipped = inpainter.predict_raw_tta(
      images[:1, :, ::-1, :], masks[:1, :, ::-1, :]
    )[0]
    np.testing.assert_allclose(out_flipped, out_orig[:, ::-1, :], atol=1e-5)

    with patch.object(sys, "argv", [sys.argv[0]]):
      self.assertTrue(config.get_args(cli=True).tta)
    with patch.object(sys, "argv", [sys.argv[0], "--no-tta"]):
      self.assertFalse(config.get_args(cli=True).tta)


if __name__ == "__main__":
  unittest.main()
