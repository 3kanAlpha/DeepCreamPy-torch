import argparse
import contextlib
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annotations import read_annotations, save_image_boxes
from lama import FFCResNetGenerator
from model import LamaInpainter
from train_bar import BarDataset, fit, sample_bar, split_samples


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

  def test_lama_dataset_patchgan_training_resume_and_finetune(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      images_dir = root / "images"
      images_dir.mkdir()
      for name, color in (("a.png", (180, 120, 90)), ("b.png", (90, 140, 200))):
        Image.new("RGB", (128, 128), color).save(images_dir / name)

      annotations = {
        Path("a.png"): [(16, 16, 112, 112)],
        Path("b.png"): [(16, 16, 112, 112)],
      }
      dataset = BarDataset(
        images_dir,
        [(Path("a.png"), (16, 16, 112, 112))],
        42,
        model_name="lama",
        crop_size=64,
      )
      image, target, mask, scored = dataset[0]
      self.assertEqual(image.shape, (3, 64, 64))
      self.assertEqual(mask.shape, (1, 64, 64))
      self.assertTrue((target >= 0.0).all().item() and (target <= 1.0).all().item())
      self.assertTrue(torch.equal(image, target * (1.0 - mask)))
      self.assertTrue(scored.any().item())

      # Train 1 epoch from scratch with PatchGAN loss, then resume to epoch 2.
      run_dir = root / "run1"
      run_dir.mkdir()
      args = argparse.Namespace(
        model="lama",
        crop_size=64,
        init_weights=None,
        lama_blocks=1,
        batch_size=1,
        lr=1e-4,
        d_lr=1e-4,
        adv_weight=0.01,
        fm_weight=0.1,
        freeze_bn=None,
        seed=42,
        device="cpu",
        epochs=1,
        resume=False,
      )
      with contextlib.redirect_stdout(io.StringIO()):
        fit(args, images_dir, run_dir, annotations)
      self.assertTrue((run_dir / "inpaint.pt").is_file())
      self.assertTrue((run_dir / "last.pt").is_file())

      args.epochs = 2
      args.resume = True
      with contextlib.redirect_stdout(io.StringIO()):
        fit(args, images_dir, run_dir, annotations)

      # Fine-tune from a saved state_dict with auto BN freeze.
      init_path = root / "init_lama.pt"
      torch.save(FFCResNetGenerator(ngf=16, n_blocks=1).state_dict(), init_path)
      ft_dir = root / "run_ft"
      ft_dir.mkdir()
      ft_args = argparse.Namespace(
        model="lama",
        crop_size=64,
        init_weights=init_path,
        lama_blocks=18,
        batch_size=1,
        lr=1e-4,
        d_lr=1e-4,
        adv_weight=0.0,
        fm_weight=0.1,
        freeze_bn=None,
        seed=42,
        device="cpu",
        epochs=1,
        resume=False,
      )
      with contextlib.redirect_stdout(io.StringIO()):
        fit(ft_args, images_dir, ft_dir, annotations)

      # Exported weights must load directly into LamaInpainter.
      inpainter = LamaInpainter(weights_path=ft_dir / "inpaint.pt")
      self.assertEqual(inpainter.batch_size, 1)


if __name__ == "__main__":
  unittest.main()
