"""Train the DeepCreamPy bar generator on annotated clean illustrations."""

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from annotations import read_annotations, validate_boxes
from libs.outputs import atomic_output, output_lock
from libs.utils import expand_bounding, prepare_crop
from model import MODELS_DIR, WEIGHTS_NAME, InpaintModel

LAST_NAME = "last.pt"


def load_annotated_images(root, csv_path):
  annotations = read_annotations(csv_path)
  if len(annotations) < 2:
    raise ValueError(
      "At least two different annotated images are needed for training and validation"
    )
  for image, boxes in annotations.items():
    path = (root / image).resolve()
    if not path.is_relative_to(root) or not path.is_file():
      raise ValueError(f"Image must exist inside --images: {image}")
    with Image.open(path) as source:
      if min(source.size) < 8:
        raise ValueError(f"{image}: both dimensions must be at least 8 pixels")
      validate_boxes(image, boxes, source.size)
  return annotations


def split_samples(annotations, seed):
  names = sorted(annotations)
  random.Random(seed).shuffle(names)
  n_validation = max(1, round(len(names) * 0.1))
  return tuple(
    [(name, box) for name in split for box in annotations[name]]
    for split in (names[n_validation:], names[:n_validation])
  )


def sample_bar(size, person, rng):
  left, top, right, bottom = person
  scale = min(*size, 256) / 256
  short = max(1, round(rng.randint(4, 12) * scale))
  long = max(1, round(rng.randint(24, 64) * scale))
  width, height = (long, short) if rng.randrange(2) else (short, long)
  width, height = min(width, right - left), min(height, bottom - top)
  x = rng.randint(left, right - width)
  y = rng.randint(top, bottom - height)
  return x, y, x + width, y + height


class BarDataset(Dataset):
  def __init__(self, root, samples, seed, epoch=0):
    self.root, self.samples, self.seed, self.epoch = root, samples, seed, epoch

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    name, person = self.samples[index]
    rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
    with Image.open(self.root / name) as source:
      image = source.convert("RGB")
    left, top, right, bottom = sample_bar(image.size, person, rng)
    marked = np.zeros((image.height, image.width), dtype=bool)
    marked[top:bottom, left:right] = True
    corners = ((left, top), (right - 1, bottom - 1))
    rgb, missing, unpadded = prepare_crop(
      image, marked, expand_bounding(image, corners)
    )
    target = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1) * 2 - 1
    known = torch.from_numpy(~missing).expand_as(target).float()
    image = target.masked_fill(known == 0, -1)
    scored = np.zeros_like(missing)
    scored[unpadded] = missing[unpadded]
    return image, target, known, torch.from_numpy(scored)


def run_epoch(model, batches, device, optimizer=None):
  training = optimizer is not None
  model.train(training)
  total = 0.0
  previews = []
  with torch.set_grad_enabled(training):
    for step, batch in enumerate(batches, 1):
      image, target, known, scored = (tensor.to(device) for tensor in batch)
      if training:
        optimizer.zero_grad(set_to_none=True)
      predicted = model(image, image, known)
      error = (predicted - target).abs().mean(dim=1)
      loss = ((error * scored).sum(dim=(1, 2)) / scored.sum(dim=(1, 2))).mean()
      value = loss.item()
      if not math.isfinite(value):
        raise ValueError(
          f"Non-finite {'training' if training else 'validation'} loss at batch {step}"
        )
      if training:
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
          print(f"  Batch {step}/{len(batches)}", flush=True)
      elif len(previews) < 4:
        count = min(4 - len(previews), len(image))
        panels = torch.cat((image[:count], predicted[:count], target[:count]), dim=3)
        pixels = ((panels + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        previews.extend(pixels.permute(0, 2, 3, 1).cpu().numpy())
      total += value * len(image)
  return total / len(batches.dataset), previews


def save_preview(path, panels):
  banner = 24
  height, width, _ = panels[0].shape
  preview = Image.new("RGB", (width, banner + len(panels) * height), (30, 30, 30))
  draw = ImageDraw.Draw(preview)
  for index, label in enumerate(("Input", "Restored", "Target")):
    draw.text((index * (width // 3) + 8, 5), label, fill="white")
  for row, panel in enumerate(panels):
    preview.paste(Image.fromarray(panel), (0, banner + row * height))
  with atomic_output(path) as temporary:
    preview.save(temporary, format="PNG")


def save_weights(path, state):
  with atomic_output(path) as temporary, temporary.open("wb") as stream:
    torch.save(state, stream)


def fit(args, root, output, annotations):
  device = torch.device(args.device)
  train, validation = split_samples(annotations, args.seed)
  print(
    f"{device}: {len(train)} training and {len(validation)} validation rectangles "
    f"from {len(annotations)} images",
    flush=True,
  )
  manifest = [(name.as_posix(), boxes) for name, boxes in sorted(annotations.items())]
  settings = {
    "annotations_sha256": hashlib.sha256(repr(manifest).encode("utf-8")).hexdigest(),
    "seed": args.seed,
  }
  validation_batches = DataLoader(
    BarDataset(root, validation, args.seed), batch_size=args.batch_size
  )
  torch.manual_seed(args.seed % 2**63)
  model = InpaintModel().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  next_epoch, best_loss, best_model = 1, math.inf, None
  weights_path = output / WEIGHTS_NAME
  last_path = output / LAST_NAME

  if args.resume:
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
    if checkpoint["settings"] != settings:
      raise ValueError("--resume requires the same annotations and --seed")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    next_epoch = checkpoint["epoch"] + 1
    best_loss, best_model = checkpoint["best_loss"], checkpoint["best_model"]
    # Recover the app export if the previous process stopped after saving last.pt.
    save_weights(weights_path, best_model)
    if next_epoch > args.epochs:
      print(f"Already completed {next_epoch - 1} epochs")
      return

  for epoch in range(next_epoch, args.epochs + 1):
    print(f"Epoch {epoch}/{args.epochs}", flush=True)
    generator = torch.Generator().manual_seed((args.seed + epoch) % 2**63)
    train_batches = DataLoader(
      BarDataset(root, train, args.seed, epoch),
      batch_size=args.batch_size,
      shuffle=True,
      generator=generator,
    )
    train_loss, _ = run_epoch(model, train_batches, device, optimizer)
    val_loss, previews = run_epoch(model, validation_batches, device)
    improved = val_loss < best_loss
    if improved:
      best_loss = val_loss
      best_model = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
      }
    save_preview(output / f"preview-{epoch:04d}.png", previews)
    save_weights(
      last_path,
      {
        "epoch": epoch,
        "settings": settings,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
        "best_model": best_model,
      },
    )
    if improved:
      save_weights(weights_path, best_model)
    print(
      f"  Train L1 {train_loss:.5f} | validation L1 {val_loss:.5f} | best {best_loss:.5f}",
      flush=True,
    )

  print(f"App weights: {weights_path}")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--images", type=Path, required=True, help="directory of clean images"
  )
  parser.add_argument(
    "--annotations", type=Path, required=True, help="person rectangle CSV"
  )
  parser.add_argument(
    "--output", type=Path, required=True, help="separate output directory"
  )
  parser.add_argument(
    "--epochs", type=int, required=True, help="total number of epochs"
  )
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
  parser.add_argument(
    "--resume", action="store_true", help="continue from last.pt in --output"
  )
  args = parser.parse_args()
  if args.epochs < 1 or args.batch_size < 1:
    parser.error("--epochs and --batch-size must be positive")
  if args.device == "cuda" and not torch.cuda.is_available():
    parser.error(
      "CUDA is unavailable. Check the PyTorch installation or use --device cpu"
    )
  root = args.images.resolve()
  output = args.output.resolve()
  if output in {(MODELS_DIR / mode).resolve() for mode in ("bar", "mosaic")}:
    parser.error("--output must be separate from the bundled model directories")

  try:
    annotations = load_annotated_images(root, args.annotations)
    if args.resume:
      if not (output / LAST_NAME).is_file():
        raise ValueError(f"No {LAST_NAME} to resume in {output}")
    else:
      if output.exists():
        raise ValueError(
          "--output must be a new directory. Use --resume to continue a run"
        )
      output.mkdir(parents=True)
    with output_lock(output / ".training.lock"):
      fit(args, root, output, annotations)
  except (OSError, ValueError, csv.Error, Image.DecompressionBombError) as error:
    parser.exit(1, f"Training stopped: {error}\n")
  except torch.cuda.OutOfMemoryError:
    parser.exit(
      1,
      "Training stopped: CUDA ran out of memory. Try a smaller batch or --device cpu.\n",
    )


if __name__ == "__main__":
  main()
