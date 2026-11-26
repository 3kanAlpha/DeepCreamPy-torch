"""Train the DeepCreamPy bar generator (PEPSI or LaMa) on annotated clean illustrations."""

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader, Dataset

from annotations import read_annotations, validate_boxes
from lama import FFCResNetGenerator, infer_lama_config
from libs.outputs import atomic_output, output_lock
from libs.utils import expand_bounding, prepare_crop, prepare_lama_training_crop
from model import MODEL_CHOICES, MODELS_DIR, WEIGHTS_NAME, InpaintModel

LAST_NAME = "last.pt"
DEFAULT_CROP_SIZE = {"pepsi": 256, "lama": 512}


class PatchGANDiscriminator(nn.Module):
  """Lightweight spectral-normalized PatchGAN discriminator (BatchNorm-free)."""

  def __init__(self, in_channels=3, ndf=64):
    super().__init__()
    self.blocks = nn.ModuleList(
      (
        nn.Sequential(
          spectral_norm(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
          ),
          nn.LeakyReLU(0.2, inplace=True),
        ),
        nn.Sequential(
          spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
          nn.LeakyReLU(0.2, inplace=True),
        ),
        nn.Sequential(
          spectral_norm(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
          ),
          nn.LeakyReLU(0.2, inplace=True),
        ),
        nn.Sequential(
          spectral_norm(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1)
          ),
          nn.LeakyReLU(0.2, inplace=True),
        ),
      )
    )
    self.head = spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1))

  def forward(self, x):
    features = []
    current = x
    for block in self.blocks:
      current = block(current)
      features.append(current)
    return self.head(current), features


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
  def __init__(
    self, root, samples, seed, epoch=0, *, model_name="pepsi", crop_size=None
  ):
    self.root = root
    self.samples = samples
    self.seed = seed
    self.epoch = epoch
    self.model_name = model_name
    self.crop_size = DEFAULT_CROP_SIZE[model_name] if crop_size is None else crop_size

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
    box = expand_bounding(image, corners, min_size=self.crop_size)

    if self.model_name == "lama":
      rgb, missing, unpadded = prepare_lama_training_crop(
        image, marked, box, size=self.crop_size
      )
      target = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)
      mask = torch.from_numpy(missing.astype(np.float32)).unsqueeze(0)
      masked_image = target * (1.0 - mask)
      scored = np.zeros_like(missing)
      scored[unpadded] = missing[unpadded]
      return masked_image, target, mask, torch.from_numpy(scored)

    rgb, missing, unpadded = prepare_crop(image, marked, box)
    target = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1) * 2 - 1
    known = torch.from_numpy(~missing).expand_as(target).float()
    masked_image = target.masked_fill(known == 0, -1)
    scored = np.zeros_like(missing)
    scored[unpadded] = missing[unpadded]
    return masked_image, target, known, torch.from_numpy(scored)


def set_batchnorm_eval(module):
  for submodule in module.modules():
    if isinstance(submodule, nn.BatchNorm2d):
      submodule.eval()


def _to_discriminator_range(tensor, model_name):
  return tensor * 2.0 - 1.0 if model_name == "lama" else tensor


def _to_preview_uint8(panels, model_name):
  if model_name == "lama":
    return (panels * 255.0).round().clamp(0, 255).to(torch.uint8)
  return ((panels + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


def _predict_generator(model, model_name, image, mask_or_known):
  if model_name == "lama":
    return model(image, mask_or_known)
  return model(image, image, mask_or_known)


def run_epoch(
  model,
  batches,
  device,
  *,
  model_name="pepsi",
  optimizer=None,
  discriminator=None,
  d_optimizer=None,
  adv_weight=0.0,
  fm_weight=0.1,
  freeze_bn=False,
):
  training = optimizer is not None
  use_gan = training and discriminator is not None and adv_weight > 0
  model.train(training)
  if training and freeze_bn:
    set_batchnorm_eval(model)
  if discriminator is not None:
    discriminator.train(training)

  total_l1 = 0.0
  total_adv = 0.0
  total_d = 0.0
  previews = []

  with torch.set_grad_enabled(training):
    for step, batch in enumerate(batches, 1):
      image, target, mask_or_known, scored = (tensor.to(device) for tensor in batch)
      predicted = _predict_generator(model, model_name, image, mask_or_known)

      if use_gan:
        d_optimizer.zero_grad(set_to_none=True)
        real_in = _to_discriminator_range(target, model_name)
        fake_in = _to_discriminator_range(predicted.detach(), model_name)
        real_logits, _ = discriminator(real_in)
        fake_logits, _ = discriminator(fake_in)
        d_loss = 0.5 * (
          F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
        )
        d_value = d_loss.item()
        if not math.isfinite(d_value):
          raise ValueError(f"Non-finite discriminator loss at batch {step}")
        d_loss.backward()
        d_optimizer.step()
        total_d += d_value * len(image)

      if training:
        optimizer.zero_grad(set_to_none=True)
      error = (predicted - target).abs().mean(dim=1)
      l1_loss = ((error * scored).sum(dim=(1, 2)) / scored.sum(dim=(1, 2))).mean()
      l1_value = l1_loss.item()
      if not math.isfinite(l1_value):
        raise ValueError(
          f"Non-finite {'training' if training else 'validation'} loss at batch {step}"
        )

      total_loss = l1_loss
      if use_gan:
        real_in = _to_discriminator_range(target, model_name)
        fake_in = _to_discriminator_range(predicted, model_name)
        with torch.no_grad():
          _, real_features = discriminator(real_in)
        fake_logits, fake_features = discriminator(fake_in)
        adv_loss = -fake_logits.mean()
        fm_loss = sum(
          F.l1_loss(fake_feat, real_feat)
          for fake_feat, real_feat in zip(fake_features, real_features, strict=True)
        ) / len(fake_features)
        total_loss = l1_loss + adv_weight * adv_loss + fm_weight * fm_loss
        adv_value = adv_loss.item()
        if not math.isfinite(total_loss.item()):
          raise ValueError(f"Non-finite adversarial generator loss at batch {step}")
        total_adv += adv_value * len(image)

      if training:
        total_loss.backward()
        optimizer.step()
        if step % 100 == 0:
          print(f"  Batch {step}/{len(batches)}", flush=True)
      elif len(previews) < 4:
        count = min(4 - len(previews), len(image))
        panels = torch.cat((image[:count], predicted[:count], target[:count]), dim=3)
        pixels = _to_preview_uint8(panels, model_name)
        previews.extend(pixels.permute(0, 2, 3, 1).cpu().numpy())

      total_l1 += l1_value * len(image)

  dataset_size = len(batches.dataset)
  metrics = {"l1": total_l1 / dataset_size}
  if use_gan:
    metrics["g_adv"] = total_adv / dataset_size
    metrics["d_loss"] = total_d / dataset_size
  return metrics, previews


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


def build_training_model(model_name, *, init_weights=None, lama_blocks=18):
  init_state = None
  if init_weights is not None:
    init_state = torch.load(init_weights, map_location="cpu", weights_only=True)

  if model_name == "lama":
    if init_state is not None:
      lama_config = infer_lama_config(init_state)
    else:
      lama_config = {
        "input_nc": 4,
        "ngf": 64,
        "n_downsampling": 3,
        "n_blocks": lama_blocks,
        "ratio_g": 0.75,
        "enable_lfu": False,
      }
    model = FFCResNetGenerator(**lama_config)
    if init_state is not None:
      model.load_state_dict(init_state, strict=True)
    return model, lama_config

  model = InpaintModel()
  if init_state is not None:
    model.load_state_dict(init_state, strict=True)
  return model, None


def fit(args, root, output, annotations):
  device = torch.device(args.device)
  model_name = args.model
  crop_size = (
    DEFAULT_CROP_SIZE[model_name] if args.crop_size is None else args.crop_size
  )
  train, validation = split_samples(annotations, args.seed)
  print(
    f"{device} ({model_name}, crop={crop_size}): {len(train)} training and "
    f"{len(validation)} validation rectangles from {len(annotations)} images",
    flush=True,
  )

  weights_path = output / WEIGHTS_NAME
  last_path = output / LAST_NAME
  manifest = [(name.as_posix(), boxes) for name, boxes in sorted(annotations.items())]

  torch.manual_seed(args.seed % 2**63)
  if args.resume:
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
    saved_settings = checkpoint["settings"]
    saved_model = saved_settings.get("model", "pepsi")
    saved_crop_size = saved_settings.get("crop_size", DEFAULT_CROP_SIZE[saved_model])
    saved_lama_config = saved_settings.get("lama_config")
    if saved_model != model_name or saved_crop_size != crop_size:
      raise ValueError(
        f"--resume requires matching --model ({saved_model}) and --crop-size ({saved_crop_size})"
      )
    if model_name == "lama":
      model = FFCResNetGenerator(**saved_lama_config).to(device)
      lama_config = saved_lama_config
    else:
      model = InpaintModel().to(device)
      lama_config = None
  else:
    model, lama_config = build_training_model(
      model_name, init_weights=args.init_weights, lama_blocks=args.lama_blocks
    )
    model = model.to(device)

  settings = {
    "annotations_sha256": hashlib.sha256(repr(manifest).encode("utf-8")).hexdigest(),
    "seed": args.seed,
    "model": model_name,
    "crop_size": crop_size,
    "lama_config": lama_config,
    "adv_weight": args.adv_weight,
    "fm_weight": args.fm_weight if args.adv_weight > 0 else 0.0,
  }

  freeze_bn = args.freeze_bn
  if freeze_bn is None:
    freeze_bn = (
      model_name == "lama" and args.init_weights is not None and args.batch_size == 1
    )
    if freeze_bn:
      print(
        "Freezing BatchNorm2d statistics for batch_size=1 LaMa fine-tuning "
        "(pass --no-freeze-bn to override).",
        flush=True,
      )

  use_gan = args.adv_weight > 0
  discriminator = PatchGANDiscriminator().to(device) if use_gan else None
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  d_optimizer = (
    torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    if use_gan
    else None
  )

  validation_batches = DataLoader(
    BarDataset(
      root,
      validation,
      args.seed,
      model_name=model_name,
      crop_size=crop_size,
    ),
    batch_size=args.batch_size,
  )

  next_epoch, best_loss, best_model = 1, math.inf, None
  if args.resume:
    saved_settings = checkpoint["settings"]
    compatible_legacy = (
      model_name == "pepsi"
      and crop_size == 256
      and not use_gan
      and saved_settings.get("annotations_sha256") == settings["annotations_sha256"]
      and saved_settings.get("seed") == settings["seed"]
      and "model" not in saved_settings
    )
    if saved_settings != settings and not compatible_legacy:
      raise ValueError(
        "--resume requires the same annotations, --seed, --model, --crop-size, and GAN weights"
      )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if use_gan:
      if "discriminator" not in checkpoint or "d_optimizer" not in checkpoint:
        raise ValueError("Checkpoint does not contain discriminator state for --resume")
      discriminator.load_state_dict(checkpoint["discriminator"])
      d_optimizer.load_state_dict(checkpoint["d_optimizer"])
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
      BarDataset(
        root,
        train,
        args.seed,
        epoch,
        model_name=model_name,
        crop_size=crop_size,
      ),
      batch_size=args.batch_size,
      shuffle=True,
      generator=generator,
    )
    train_metrics, _ = run_epoch(
      model,
      train_batches,
      device,
      model_name=model_name,
      optimizer=optimizer,
      discriminator=discriminator,
      d_optimizer=d_optimizer,
      adv_weight=args.adv_weight,
      fm_weight=args.fm_weight,
      freeze_bn=freeze_bn,
    )
    val_metrics, previews = run_epoch(
      model,
      validation_batches,
      device,
      model_name=model_name,
    )
    val_loss = val_metrics["l1"]
    improved = val_loss < best_loss
    if improved:
      best_loss = val_loss
      best_model = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
      }
    save_preview(output / f"preview-{epoch:04d}.png", previews)
    checkpoint_payload = {
      "epoch": epoch,
      "settings": settings,
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "best_loss": best_loss,
      "best_model": best_model,
    }
    if use_gan:
      checkpoint_payload["discriminator"] = discriminator.state_dict()
      checkpoint_payload["d_optimizer"] = d_optimizer.state_dict()
    save_weights(last_path, checkpoint_payload)
    if improved:
      save_weights(weights_path, best_model)

    summary = (
      f"  Train L1 {train_metrics['l1']:.5f} | validation L1 {val_loss:.5f} "
      f"| best {best_loss:.5f}"
    )
    if use_gan:
      summary += (
        f" | G_adv {train_metrics['g_adv']:.5f} | D {train_metrics['d_loss']:.5f}"
      )
    print(summary, flush=True)

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
  parser.add_argument(
    "--model",
    choices=MODEL_CHOICES,
    default="pepsi",
    help="generator architecture to train (pepsi or lama)",
  )
  parser.add_argument(
    "--crop-size",
    type=int,
    default=None,
    help="training crop size (default: 256 for pepsi, 512 for lama; multiple of 8 for lama)",
  )
  parser.add_argument(
    "--init-weights",
    type=Path,
    default=None,
    help="pretrained state_dict .pt file to fine-tune from",
  )
  parser.add_argument(
    "--lama-blocks",
    type=int,
    default=18,
    help="number of FFC residual blocks when training LaMa from scratch (default: 18)",
  )
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--lr", type=float, default=1e-4, help="generator learning rate")
  parser.add_argument(
    "--d-lr", type=float, default=1e-4, help="discriminator learning rate"
  )
  parser.add_argument(
    "--adv-weight",
    type=float,
    default=0.0,
    help="PatchGAN adversarial loss weight (0 disables GAN training; e.g. 0.01 to enable)",
  )
  parser.add_argument(
    "--fm-weight",
    type=float,
    default=0.1,
    help="PatchGAN feature matching loss weight when --adv-weight > 0",
  )
  parser.add_argument(
    "--freeze-bn",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="freeze BatchNorm2d running stats (auto-enabled for batch_size=1 LaMa fine-tuning)",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
  parser.add_argument(
    "--resume", action="store_true", help="continue from last.pt in --output"
  )
  args = parser.parse_args()

  if args.epochs < 1 or args.batch_size < 1:
    parser.error("--epochs and --batch-size must be positive")
  if args.lr <= 0 or args.d_lr <= 0:
    parser.error("--lr and --d-lr must be positive")
  if args.adv_weight < 0 or args.fm_weight < 0:
    parser.error("--adv-weight and --fm-weight must be non-negative")
  if args.lama_blocks < 1:
    parser.error("--lama-blocks must be positive")
  if args.resume and args.init_weights is not None:
    parser.error("--init-weights cannot be combined with --resume")
  if args.init_weights is not None and not args.init_weights.is_file():
    parser.error(f"Initial weights not found: {args.init_weights}")

  crop_size = (
    DEFAULT_CROP_SIZE[args.model] if args.crop_size is None else args.crop_size
  )
  if args.model == "pepsi" and crop_size != 256:
    parser.error("--crop-size must be 256 when --model pepsi")
  if args.model == "lama" and (crop_size < 8 or crop_size % 8 != 0):
    parser.error("--crop-size must be a positive multiple of 8 when --model lama")

  if args.device == "cuda" and not torch.cuda.is_available():
    parser.error(
      "CUDA is unavailable. Check the PyTorch installation or use --device cpu"
    )
  root = args.images.resolve()
  output = args.output.resolve()
  if output in {(MODELS_DIR / mode).resolve() for mode in ("bar", "mosaic", "lama")}:
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
