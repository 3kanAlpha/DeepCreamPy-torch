import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from anima import (
  ANIMA_DEFAULT_CACHE_PROMPT as ANIMA_DEFAULT_CACHE_PROMPT,
  ANIMA_DEFAULT_CFG as ANIMA_DEFAULT_CFG,
  ANIMA_DEFAULT_CFG_SKIP_TAIL as ANIMA_DEFAULT_CFG_SKIP_TAIL,
  ANIMA_DEFAULT_LLLITE_STRENGTH as ANIMA_DEFAULT_LLLITE_STRENGTH,
  ANIMA_DEFAULT_MAX_SIZE as ANIMA_DEFAULT_MAX_SIZE,
  ANIMA_DEFAULT_NEGATIVE_PROMPT as ANIMA_DEFAULT_NEGATIVE_PROMPT,
  ANIMA_DEFAULT_SEED as ANIMA_DEFAULT_SEED,
  ANIMA_DEFAULT_STEPS as ANIMA_DEFAULT_STEPS,
  ANIMA_MAX_CFG_SKIP_TAIL as ANIMA_MAX_CFG_SKIP_TAIL,
  ANIMA_SUBDIRS as ANIMA_SUBDIRS,
  AnimaCancelledError as AnimaCancelledError,
  AnimaInpainter as AnimaInpainter,
  AnimaSettings as AnimaSettings,
  anima_availability as anima_availability,
  anima_cuda_available as anima_cuda_available,
  anima_default_fp8_storage as anima_default_fp8_storage,
  anima_dependencies_available as anima_dependencies_available,
  anima_models_dir as anima_models_dir,
  anima_radio_tooltip as anima_radio_tooltip,
  list_anima_files as list_anima_files,
)
from lama import FFCResNetGenerator, infer_lama_config
from libs.utils import (
  prepare_crop,
  prepare_lama_crop,
  resize_lama_prediction,
  resize_prediction_float,
  seamless_blend_region,
  suppress_monochrome_chroma,
)


def _app_root():
  if getattr(sys, "frozen", False):
    return Path(sys.executable).resolve().parent
  return Path(__file__).resolve().parent.parent


ROOT = Path(__file__).resolve().parent
if not getattr(sys, "frozen", False):
  ROOT = ROOT.parent
MODELS_DIR = ROOT / "models"
WEIGHTS_NAME = "inpaint.pt"
LAMA_WEIGHTS_PATH = _app_root() / "models" / "lama" / WEIGHTS_NAME
MODEL_CHOICES = ("pepsi", "lama", "anima")


def lama_weights_available(path=None):
  target = LAMA_WEIGHTS_PATH if path is None else Path(path)
  return target.is_file()


def lama_radio_tooltip(available=None):
  if available is None:
    available = lama_weights_available()
  if available:
    return "LaMa is available in Bar censor mode only."
  return (
    "LaMa model not found. Place a converted inpaint.pt at "
    f"{LAMA_WEIGHTS_PATH} to enable LaMa."
  )


# (horizontal_flip, dy, dx) in 256x256 input space
PEPSI_TTA_VIEWS = (
  (False, 0, 0),
  (True, 0, 0),
  (False, 2, 2),
  (True, 2, 2),
)


def _shift_reflect(array, dy, dx):
  if dy == 0 and dx == 0:
    return array
  height, width = array.shape[:2]
  pad_y, pad_x = abs(dy), abs(dx)
  padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="reflect")
  y0 = pad_y - dy
  x0 = pad_x - dx
  return padded[y0 : y0 + height, x0 : x0 + width]


def _unshift_with_valid(prediction, dy, dx):
  height, width = prediction.shape[:2]
  if dy == 0 and dx == 0:
    return prediction, np.ones((height, width), dtype=np.float32)
  out = np.zeros_like(prediction)
  valid = np.zeros((height, width), dtype=np.float32)
  src_y0, src_y1 = max(0, dy), min(height, height + dy)
  src_x0, src_x1 = max(0, dx), min(width, width + dx)
  dst_y0 = max(0, -dy)
  dst_x0 = max(0, -dx)
  dst_y1 = dst_y0 + (src_y1 - src_y0)
  dst_x1 = dst_x0 + (src_x1 - src_x0)
  out[dst_y0:dst_y1, dst_x0:dst_x1] = prediction[src_y0:src_y1, src_x0:src_x1]
  valid[dst_y0:dst_y1, dst_x0:dst_x1] = 1.0
  return out, valid


class InpaintModel(nn.Module):
  def __init__(self):
    super().__init__()
    channels = (6, 32, 64, 64, 128, 128, 256, 256, 256, 256, 256)
    strides = (1, 2, 1, 2, 1, 2, 1, 1, 1, 1)
    dilations = (1, 1, 1, 1, 1, 1, 2, 4, 8, 16)
    self.encoder = nn.ModuleList(
      nn.Conv2d(
        channels[i],
        channels[i + 1],
        5 if i == 0 else 3,
        stride=stride,
        dilation=dilation,
      )
      for i, (stride, dilation) in enumerate(zip(strides, dilations, strict=True))
    )
    self.context_merge = nn.Conv2d(512, 256, 1)
    channels = (256, 128, 128, 64, 64, 32, 32, 16, 16, 3)
    self.decoder = nn.ModuleList(
      nn.Conv2d(in_channels, out_channels, 3, padding=1 if out_channels == 3 else 0)
      for in_channels, out_channels in zip(channels, channels[1:])
    )

  def contextual_block(self, features, mask):
    size = features.shape[-2:]
    patch_size = 3
    padding = patch_size // 2
    mask = F.interpolate(mask[:, :1], size=size, mode="nearest")
    background = features * mask
    keys = F.unfold(background, kernel_size=patch_size).transpose(1, 2)
    queries = F.unfold(features, kernel_size=patch_size, padding=padding)

    distances = (
      keys.square().sum(dim=2, keepdim=True)
      + queries.square().sum(dim=1, keepdim=True)
      - 2 * torch.bmm(keys, queries)
    )
    centered = distances - distances.mean(dim=1, keepdim=True)
    variance = centered.square().mean(dim=1, keepdim=True)
    # Bound the variance before sqrt so uniform candidates also have finite gradients.
    deviation = variance.clamp_min(torch.finfo(distances.dtype).eps ** 2).sqrt()
    attention = torch.softmax(-50 * torch.tanh(centered / deviation), dim=1)

    patches = torch.bmm(keys.transpose(1, 2), attention)
    copied = (
      F.fold(patches, size, kernel_size=patch_size, padding=padding) / patch_size**2
    )
    filled = background + copied * (1 - mask)
    return F.elu(self.context_merge(torch.cat((features, filled), dim=1)))

  def forward_raw(self, image, mask):
    features = torch.cat((image, mask), dim=1)
    for convolution in self.encoder:
      border = (convolution.kernel_size[0] // 2) * convolution.dilation[0]
      features = F.elu(convolution(F.pad(features, (border,) * 4, mode="reflect")))

    features = self.contextual_block(features, mask)
    for i, convolution in enumerate(self.decoder[:-1]):
      features = F.elu(convolution(F.pad(features, (1, 1, 1, 1), mode="reflect")))
      if i in (1, 3, 5):
        features = F.interpolate(features, scale_factor=2, mode="nearest")

    return self.decoder[-1](features).clamp(-1, 1)

  def forward(self, image, original, mask):
    filled = self.forward_raw(image, mask)
    return filled * (1 - mask) + original * mask


class InpaintNN:
  def __init__(self, is_mosaic=False, *, weights_path=None, use_tta=True):
    mode = "mosaic" if is_mosaic else "bar"
    path = (
      weights_path if weights_path is not None else MODELS_DIR / mode / WEIGHTS_NAME
    )
    if not path.is_file():
      if weights_path is not None:
        raise FileNotFoundError(f"Missing weights file: {path}")
      raise FileNotFoundError(
        f"Missing {path}. Run the model conversion in docs/INSTALLATION.md."
      )

    state = torch.load(path, map_location="cpu", weights_only=True)
    self.model = InpaintModel().eval()
    self.model.load_state_dict(state)
    self.use_tta = use_tta
    self.device = torch.device("cpu")
    self.fallback_reason = "CUDA is unavailable"

    try:
      if torch.cuda.is_available():
        # Match the float32 arithmetic used by the original model.
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        self.model.to("cuda")
        torch.cuda.synchronize()
        self.device = torch.device("cuda")
        self.fallback_reason = None
    except (RuntimeError, AssertionError) as error:
      self.model = InpaintModel().eval()
      self.model.load_state_dict(state)
      self.fallback_reason = f"CUDA initialization failed: {error}"

    self.batch_size = 4 if self.device.type == "cuda" else 1

  @torch.inference_mode()
  def predict(self, image, original, mask):
    try:
      return self._predict(image, original, mask)
    except torch.cuda.OutOfMemoryError:
      if self.device.type != "cuda" or len(image) <= 1:
        raise
      self.batch_size = 1

    # Retry after leaving the exception handler so the failed batch can be freed.
    return np.concatenate(
      [
        self._predict(image[i : i + 1], original[i : i + 1], mask[i : i + 1])
        for i in range(len(image))
      ]
    )

  def _predict(self, image, original, mask):
    inputs = [
      torch.as_tensor(array, dtype=torch.float32, device=self.device).permute(
        0, 3, 1, 2
      )
      for array in (image, original, mask)
    ]
    return self.model(*inputs).permute(0, 2, 3, 1).cpu().numpy()

  @torch.inference_mode()
  def predict_raw(self, image, mask):
    try:
      return self._predict_raw(image, mask)
    except torch.cuda.OutOfMemoryError:
      if self.device.type != "cuda" or len(image) <= 1:
        raise
      self.batch_size = 1

    # Retry after leaving the exception handler so the failed batch can be freed.
    return np.concatenate(
      [self._predict_raw(image[i : i + 1], mask[i : i + 1]) for i in range(len(image))]
    )

  def _predict_raw(self, image, mask):
    inputs = [
      torch.as_tensor(array, dtype=torch.float32, device=self.device).permute(
        0, 3, 1, 2
      )
      for array in (image, mask)
    ]
    return self.model.forward_raw(*inputs).permute(0, 2, 3, 1).cpu().numpy()

  def predict_raw_tta(self, images, masks, *, views=PEPSI_TTA_VIEWS):
    """Run raw inference across flipped/shifted views and average aligned predictions."""
    count = len(images)
    view_images = []
    view_masks = []
    for flip, dy, dx in views:
      for i in range(count):
        image = images[i]
        mask = masks[i]
        if flip:
          image = image[:, ::-1, :]
          mask = mask[:, ::-1, :]
        view_images.append(_shift_reflect(image, dy, dx))
        view_masks.append(_shift_reflect(mask, dy, dx))

    stacked_images = np.stack(view_images)
    stacked_masks = np.stack(view_masks)
    chunks = []
    start = 0
    while start < len(stacked_images):
      step = max(1, self.batch_size)
      chunks.append(
        self.predict_raw(
          stacked_images[start : start + step],
          stacked_masks[start : start + step],
        )
      )
      start += step
    raw_all = np.concatenate(chunks, axis=0)

    averaged = []
    for i in range(count):
      acc = np.zeros_like(images[i], dtype=np.float32)
      weight = np.zeros(images[i].shape[:2], dtype=np.float32)
      for view_index, (flip, dy, dx) in enumerate(views):
        pred = raw_all[view_index * count + i]
        pred, valid = _unshift_with_valid(pred, dy, dx)
        if flip:
          pred = pred[:, ::-1, :]
          valid = valid[:, ::-1]
        acc += pred * valid[..., None]
        weight += valid
      averaged.append(acc / np.maximum(weight, 1.0)[..., None])
    return np.stack(averaged, axis=0)

  def inpaint_crops(
    self, ori, marked, boxes, *, is_mosaic=False, regions=None, use_tta=None
  ):
    prediction_slices = []
    crops = []
    masks = []
    for box in boxes:
      crop, missing, prediction_slice = prepare_crop(ori, marked, box)
      prediction_slices.append(prediction_slice)
      if not is_mosaic:
        crop[missing] = 0
      crops.append(crop)
      masks.append(np.repeat((~missing)[..., None], 3, axis=-1))

    images = np.stack(crops) * 2.0 - 1
    masks_stack = np.stack(masks)
    apply_tta = self.use_tta if use_tta is None else use_tta
    if apply_tta:
      raw_predictions = self.predict_raw_tta(images, masks_stack)
    else:
      raw_predictions = self.predict_raw(images, masks_stack)
    region_list = regions if regions is not None else [None] * len(boxes)
    results = []
    for box, prediction_slice, raw_prediction, region in zip(
      boxes, prediction_slices, raw_predictions, region_list, strict=True
    ):
      left, top, right, bottom = box
      width, height = right - left, bottom - top
      pred_float = resize_prediction_float(
        raw_prediction[prediction_slice], (width, height)
      )
      source_crop = np.asarray(ori.crop(box), dtype=np.float32)
      crop_marked = marked[top:bottom, left:right]
      known_mask = ~crop_marked
      if region is not None:
        region_mask = np.zeros((height, width), dtype=bool)
        rx, ry = np.asarray(list(region)).T
        region_mask[ry - top, rx - left] = True
      else:
        region_mask = crop_marked

      pred_float = seamless_blend_region(
        pred_float, source_crop, region_mask, known_mask
      )
      pred_float = suppress_monochrome_chroma(
        pred_float, source_crop, region_mask, known_mask
      )
      results.append(np.rint(pred_float.clip(0.0, 255.0)).astype(np.uint8))
    return results


class LamaInpainter:
  def __init__(self, *, weights_path=None, max_size=512):
    path = weights_path if weights_path is not None else LAMA_WEIGHTS_PATH
    if not path.is_file():
      if weights_path is not None:
        raise FileNotFoundError(f"Missing weights file: {path}")
      if getattr(sys, "frozen", False):
        raise FileNotFoundError(
          f"Missing {path}. Place a converted LaMa state_dict (inpaint.pt) "
          f"at {path} to enable LaMa."
        )
      raise FileNotFoundError(
        f"Missing {path}. Convert a LaMa checkpoint with src/convert_lama.py "
        "or pass --bar-weights."
      )

    state = torch.load(path, map_location="cpu", weights_only=True)
    config = infer_lama_config(state)
    self.model = FFCResNetGenerator(**config).eval()
    self.model.load_state_dict(state, strict=True)
    self.max_size = max_size
    self.device = torch.device("cpu")
    self.fallback_reason = "CUDA is unavailable"

    try:
      if torch.cuda.is_available():
        self.model.to("cuda")
        torch.cuda.synchronize()
        self.device = torch.device("cuda")
        self.fallback_reason = None
    except (RuntimeError, AssertionError) as error:
      self.model = FFCResNetGenerator(**config).eval()
      self.model.load_state_dict(state, strict=True)
      self.fallback_reason = f"CUDA initialization failed: {error}"

    self.batch_size = 1

  @torch.inference_mode()
  def predict(self, image, mask):
    image_tensor = torch.as_tensor(
      image, dtype=torch.float32, device=self.device
    ).permute(0, 3, 1, 2)
    mask_tensor = torch.as_tensor(
      mask, dtype=torch.float32, device=self.device
    ).permute(0, 3, 1, 2)
    return self.model(image_tensor, mask_tensor).permute(0, 2, 3, 1).cpu().numpy()

  def inpaint_crops(self, ori, marked, boxes, *, is_mosaic=False, regions=None):
    if is_mosaic:
      raise ValueError("LaMa is only supported in bar mode")
    results = []
    for box in boxes:
      crop, missing, prediction_slice = prepare_lama_crop(
        ori, marked, box, max_size=self.max_size
      )
      mask = missing.astype(np.float32)[..., None]
      prediction = self.predict(crop[None, ...], mask[None, ...])[0]
      results.append(
        resize_lama_prediction(
          prediction[prediction_slice], (box[2] - box[0], box[3] - box[1])
        )
      )
    return results


def create_inpainter(
  model_name="pepsi",
  *,
  is_mosaic=False,
  weights_path=None,
  use_tta=True,
  anima_settings=None,
):
  if model_name == "pepsi":
    return InpaintNN(is_mosaic=is_mosaic, weights_path=weights_path, use_tta=use_tta)
  if model_name == "lama":
    if is_mosaic:
      raise ValueError("LaMa is only supported in bar mode")
    return LamaInpainter(weights_path=weights_path)
  if model_name == "anima":
    if is_mosaic:
      raise ValueError("Anima is only supported in bar mode")
    return AnimaInpainter(settings=anima_settings)
  raise ValueError(f"Unsupported model: {model_name}")
