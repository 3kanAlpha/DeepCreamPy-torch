import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from decensor import Decensor
from main import MainWindow
from mask_detect import (
  DetectionParams,
  apply_mask_to_image,
  detect_censor_in_rect,
)
from mask_editor import (
  TOOL_AUTO_DETECT,
  TOOL_BRUSH,
  TOOL_ERASER,
  TOOL_RECT_FILL,
  MaskEditorWindow,
  pil_to_qpixmap,
  qimage_to_pil,
)


class StubModel:
  batch_size = 4

  def predict(self, image, original, mask):
    # Return a constant gray prediction in [-1, 1] so restored pixels differ visibly.
    return np.zeros_like(original, dtype=np.float32)


class MaskDetectTests(unittest.TestCase):
  def test_black_bar_detected_and_saturated_dark_rejected(self):
    rgb = np.full((100, 120, 3), (235, 200, 185), dtype=np.uint8)
    # 10px tall black bar
    rgb[40:50, 20:90] = (10, 10, 12)
    # Dark saturated navy region (should be rejected by max_saturation)
    rgb[65:75, 20:90] = (5, 10, 95)

    rect, roi = detect_censor_in_rect(
      rgb,
      (10, 30, 100, 85),
      DetectionParams(
        mode="black",
        black_threshold=50,
        max_saturation=30,
        min_thickness=3,
        dilation=0,
      ),
    )
    self.assertEqual(rect, (10, 30, 100, 85))
    # Inside black bar (relative y=10..20, x=10..80)
    self.assertTrue(roi[10:20, 10:80].all())
    # Saturated dark blue must remain unmarked
    self.assertFalse(roi[35:45, 10:80].any())

  def test_white_bar_detected_and_thin_lineart_filtered(self):
    rgb = np.full((100, 100, 3), (180, 140, 120), dtype=np.uint8)
    # 8px thick white bar
    rgb[20:28, 15:85] = (250, 248, 252)
    # 2px thin black anime lineart
    rgb[50:52, 10:90] = (5, 5, 5)
    # 8px thick black bar
    rgb[70:78, 15:85] = (8, 8, 8)

    # White mode finds only white bar
    _, white_roi = detect_censor_in_rect(
      rgb,
      (0, 0, 100, 100),
      DetectionParams(mode="white", white_threshold=230, min_thickness=3, dilation=0),
    )
    self.assertTrue(white_roi[20:28, 15:85].all())
    self.assertFalse(white_roi[70:78, 15:85].any())

    # Both mode with min_thickness=3 finds thick black & white bars, ignores 2px lineart
    _, both_roi = detect_censor_in_rect(
      rgb,
      (0, 0, 100, 100),
      DetectionParams(
        mode="both",
        black_threshold=45,
        white_threshold=230,
        min_thickness=3,
        dilation=0,
      ),
    )
    self.assertTrue(both_roi[20:28, 15:85].all())
    self.assertTrue(both_roi[70:78, 15:85].all())
    self.assertFalse(both_roi[50:52, 10:90].any())

    # Disabling lineart filter (min_thickness=0) keeps the 2px line
    _, raw_roi = detect_censor_in_rect(
      rgb,
      (0, 0, 100, 100),
      DetectionParams(mode="black", black_threshold=45, min_thickness=0, dilation=0),
    )
    self.assertTrue(raw_roi[50:52, 10:90].all())

  def test_thin_black_line_adjacent_to_white_bar_still_filtered_in_both_mode(self):
    rgb = np.full((60, 60, 3), (170, 130, 120), dtype=np.uint8)
    # Thick white bar
    rgb[20:30, 10:50] = (250, 250, 250)
    # 1px black outline immediately adjacent to white bar
    rgb[19, 10:50] = (0, 0, 0)

    _, roi = detect_censor_in_rect(
      rgb,
      (0, 0, 60, 60),
      DetectionParams(mode="both", min_thickness=3, dilation=0),
    )
    self.assertTrue(roi[20:30, 10:50].all())
    self.assertFalse(roi[19, 10:50].any())

  def test_dilation_and_rect_boundary_preservation(self):
    rgb = np.full((80, 80, 3), (200, 170, 150), dtype=np.uint8)
    # Black bar crossing the selection rectangle's left edge
    rgb[30:40, 0:50] = (0, 0, 0)

    _, roi = detect_censor_in_rect(
      rgb,
      (20, 20, 60, 50),
      DetectionParams(mode="black", min_thickness=4, dilation=2),
    )
    # Left edge of ROI (x=20 in image -> col 0 in ROI) must remain detected
    self.assertTrue(roi[10:20, 0:30].all())
    # Dilation extends 2px beyond bar edge (up to col 31, rows 8..21)
    self.assertTrue(roi[8, 10])
    self.assertTrue(roi[21, 10])
    self.assertTrue(roi[15, 31])

  def test_apply_mask_to_image_preserves_alpha(self):
    rgba = np.full((20, 24, 4), (100, 120, 140, 200), dtype=np.uint8)
    rgba[5:10, 6:12, 3] = 77
    img = Image.fromarray(rgba, mode="RGBA")
    mask = np.zeros((20, 24), dtype=bool)
    mask[5:10, 6:12] = True

    masked = apply_mask_to_image(img, mask, (0, 255, 0))
    self.assertEqual(masked.mode, "RGBA")
    out = np.asarray(masked)
    self.assertTrue((out[mask, :3] == (0, 255, 0)).all())
    self.assertTrue((out[mask, 3] == 77).all())
    self.assertTrue((out[~mask, :3] == (100, 120, 140)).all())


class SingleImageDecensorTests(unittest.TestCase):
  def test_single_image_saves_overwrites_and_keeps_input_dir(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      in_dir = tmp_path / "in"
      out_dir = tmp_path / "out"
      in_dir.mkdir()
      out_dir.mkdir()
      sentinel = in_dir / "keep_me.png"
      Image.new("RGB", (16, 16), (10, 20, 30)).save(sentinel)

      with patch.object(sys, "argv", [sys.argv[0]]):
        decensor = Decensor()
      decensor.decensor_input_path = str(in_dir)
      decensor.decensor_output_path = str(out_dir)
      decensor.clean_up_input_dirs = True
      decensor.model = StubModel()
      decensor.loaded_config = (
        decensor.model_name,
        decensor.is_mosaic,
        decensor.bar_weights,
      )
      decensor.variations = 2

      ori = Image.new("RGB", (64, 64), (210, 180, 160))
      mask = np.zeros((64, 64), dtype=bool)
      mask[20:30, 20:40] = True
      colored = apply_mask_to_image(ori, mask)

      with contextlib.redirect_stdout(io.StringIO()):
        results, paths = decensor.decensor_single_image(ori, colored, "sample")

      self.assertEqual(len(results), 2)
      self.assertEqual(len(paths), 2)
      self.assertTrue((out_dir / "sample 0.png").is_file())
      self.assertTrue((out_dir / "sample 1.png").is_file())
      # Single-image decensor must never delete files in decensor_input
      self.assertTrue(sentinel.is_file())

      # Re-running with same stem overwrites rather than creating duplicate files
      with contextlib.redirect_stdout(io.StringIO()):
        decensor.decensor_single_image(ori, colored, "sample")
      self.assertEqual(
        sorted(p.name for p in out_dir.iterdir()), ["sample 0.png", "sample 1.png"]
      )

  def test_single_image_empty_mask_raises_in_bar_mode(self):
    with tempfile.TemporaryDirectory() as tmp:
      with patch.object(sys, "argv", [sys.argv[0]]):
        decensor = Decensor()
      decensor.decensor_output_path = tmp
      decensor.model = StubModel()
      decensor.loaded_config = (
        decensor.model_name,
        decensor.is_mosaic,
        decensor.bar_weights,
      )
      decensor.is_mosaic = False

      clean = Image.new("RGB", (48, 48), (120, 130, 140))
      with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
        decensor.decensor_single_image(clean, clean, "empty")


class MaskEditorGuiTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.app = QApplication.instance() or QApplication([])

  def test_qimage_pil_roundtrip(self):
    arr = np.zeros((24, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 200
    arr[:, :, 1] = 100
    arr[:, :, 2] = 50
    img = Image.fromarray(arr, mode="RGB")
    qimg = pil_to_qpixmap(img).toImage()
    roundtripped = qimage_to_pil(qimg)
    self.assertEqual(roundtripped.mode, "RGB")
    self.assertTrue(np.array_equal(np.asarray(roundtripped), arr))

  def test_auto_detect_live_tuning_commit_undo_redo(self):
    editor = MaskEditorWindow(
      {"is_mosaic": False, "model_name": "pepsi", "use_tta": True, "variations": 1}
    )
    arr = np.full((80, 80, 3), (220, 190, 175), dtype=np.uint8)
    arr[25:35, 20:60] = (20, 20, 20)
    img = Image.fromarray(arr, mode="RGB")
    editor._set_loaded_image(img, "test_img", source_path=None)

    # Trigger auto-detect in rectangle -> pending preview created
    editor._run_auto_detect((10, 15, 70, 45))
    self.assertTrue(editor.canvas.has_pending())
    self.assertFalse(editor.canvas.mask.any())
    initial_pending_count = int(editor.canvas.pending_mask.sum())
    self.assertGreater(initial_pending_count, 0)

    # Adjust dilation slider -> pending mask updates live
    editor.dilationSlider.setValue(4)
    dilated_count = int(editor.canvas.pending_mask.sum())
    self.assertGreater(dilated_count, initial_pending_count)

    # Commit pending -> committed into mask, undoable
    editor._commit_pending()
    self.assertFalse(editor.canvas.has_pending())
    self.assertEqual(int(editor.canvas.mask.sum()), dilated_count)
    self.assertTrue(editor.canvas.can_undo())

    # Undo restores empty mask; redo brings it back
    editor._undo()
    self.assertFalse(editor.canvas.mask.any())
    self.assertTrue(editor.canvas.can_redo())
    editor._redo()
    self.assertEqual(int(editor.canvas.mask.sum()), dilated_count)

    # Brush & eraser mutate mask cleanly
    editor._select_tool(TOOL_BRUSH)
    editor.canvas.set_brush_size(8)
    editor.canvas._paint_capsule(QPointF(5, 5), QPointF(15, 5), True)
    self.assertTrue(editor.canvas.mask[5, 10])
    editor._select_tool(TOOL_ERASER)
    editor.canvas._paint_capsule(QPointF(5, 5), QPointF(15, 5), False)
    self.assertFalse(editor.canvas.mask[5, 10])

    # Result view switching
    res0 = Image.new("RGB", (80, 80), (100, 100, 100))
    res1 = Image.new("RGB", (80, 80), (150, 150, 150))
    editor.on_decensor_succeeded(
      [res0, res1], ["out/test_img 0.png", "out/test_img 1.png"]
    )
    self.assertEqual(editor.canvas.view_mode, "result")
    self.assertEqual(editor.variationCombo.count(), 2)
    self.assertTrue(editor.compareButton.isEnabled())
    editor.close()

  def test_main_window_two_way_settings_sync_and_mosaic_tool_switch(self):
    with (
      patch.object(MainWindow, "load_model", lambda self: None),
      patch.object(sys, "argv", [sys.argv[0]]),
    ):
      win = MainWindow()
    win.openMaskEditor()
    editor = win.maskEditorWindow
    self.assertIsNotNone(editor)
    self.assertEqual(editor.canvas.tool, TOOL_AUTO_DETECT)

    # Switch to Mosaic in MainWindow -> syncs to editor & switches tool to Rect Fill
    win.mosaicButton.setChecked(True)
    self.assertTrue(editor.mosaicRadio.isChecked())
    self.assertFalse(editor.lamaRadio.isEnabled())
    self.assertEqual(editor.canvas.tool, TOOL_RECT_FILL)

    # Switch back to Bar + LaMa + 4 variations in Editor -> syncs back to MainWindow
    editor.barRadio.setChecked(True)
    self.assertEqual(editor.canvas.tool, TOOL_AUTO_DETECT)
    editor.lamaRadio.setChecked(True)
    editor.variationsCombo.setCurrentText("4")

    self.assertTrue(win.barButton.isChecked())
    self.assertTrue(win.lamaModelButton.isChecked())
    self.assertFalse(win.ttaCheckbox.isEnabled())
    self.assertEqual(win.get_decensor_settings()["variations"], 4)

    win.close()


if __name__ == "__main__":
  unittest.main()
