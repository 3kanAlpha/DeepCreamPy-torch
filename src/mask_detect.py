"""Auto-detection and compositing helpers for black/white censor masks."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectionParams:
  mode: str = "black"  # "black", "white", or "both"
  black_threshold: int = 50
  white_threshold: int = 220
  max_saturation: int = 35
  min_thickness: int = 3
  dilation: int = 2


def _erode_1d(mask: np.ndarray, size: int, axis: int) -> np.ndarray:
  """1D binary erosion with window length ``size`` along ``axis`` using edge padding."""
  if size <= 1 or mask.shape[axis] == 0:
    return mask.copy()
  before = (size - 1) // 2
  after = size - 1 - before
  pad = [(0, 0), (0, 0)]
  pad[axis] = (before, after)
  padded = np.pad(mask.astype(np.int32), pad, mode="edge")
  if axis == 0:
    csum = np.zeros((padded.shape[0] + 1, padded.shape[1]), dtype=np.int32)
    np.cumsum(padded, axis=0, out=csum[1:])
    window_sum = csum[size:] - csum[:-size]
  else:
    csum = np.zeros((padded.shape[0], padded.shape[1] + 1), dtype=np.int32)
    np.cumsum(padded, axis=1, out=csum[:, 1:])
    window_sum = csum[:, size:] - csum[:, :-size]
  return window_sum == size


def _dilate_1d(
  mask: np.ndarray, size: int, axis: int, *, reflected: bool = False
) -> np.ndarray:
  """1D binary dilation with window length ``size`` along ``axis``."""
  if size <= 1 or mask.shape[axis] == 0:
    return mask.copy()
  before = (size - 1) // 2
  after = size - 1 - before
  if reflected:
    before, after = after, before
  pad = [(0, 0), (0, 0)]
  pad[axis] = (before, after)
  padded = np.pad(mask.astype(np.int32), pad, mode="constant", constant_values=0)
  if axis == 0:
    csum = np.zeros((padded.shape[0] + 1, padded.shape[1]), dtype=np.int32)
    np.cumsum(padded, axis=0, out=csum[1:])
    window_sum = csum[size:] - csum[:-size]
  else:
    csum = np.zeros((padded.shape[0], padded.shape[1] + 1), dtype=np.int32)
    np.cumsum(padded, axis=1, out=csum[:, 1:])
    window_sum = csum[:, size:] - csum[:, :-size]
  return window_sum > 0


def morphological_open(mask: np.ndarray, min_thickness: int) -> np.ndarray:
  """Remove strokes thinner than ``min_thickness`` pixels via centered box opening."""
  if min_thickness <= 1 or not mask.any():
    return mask.copy()
  eroded = _erode_1d(_erode_1d(mask, min_thickness, axis=0), min_thickness, axis=1)
  if not eroded.any():
    return eroded
  return _dilate_1d(
    _dilate_1d(eroded, min_thickness, axis=0, reflected=True),
    min_thickness,
    axis=1,
    reflected=True,
  )


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
  """Expand a boolean mask by ``radius`` pixels using an octagonal structuring element."""
  if radius <= 0 or not mask.any():
    return mask.copy()
  result = mask.copy()
  height, width = result.shape
  for step in range(radius):
    grown = result.copy()
    if height > 1:
      grown[:-1, :] |= result[1:, :]
      grown[1:, :] |= result[:-1, :]
    if width > 1:
      grown[:, :-1] |= result[:, 1:]
      grown[:, 1:] |= result[:, :-1]
    # Alternate 8-connected and 4-connected steps for a rounded octagonal margin.
    if step % 2 == 0 and height > 1 and width > 1:
      grown[:-1, :-1] |= result[1:, 1:]
      grown[:-1, 1:] |= result[1:, :-1]
      grown[1:, :-1] |= result[:-1, 1:]
      grown[1:, 1:] |= result[:-1, :-1]
    result = grown
  return result


def clamp_rect(
  rect: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
  """Normalize and clamp ``(left, top, right, bottom)`` to ``[0, width] x [0, height]``."""
  x0, y0, x1, y1 = rect
  left = max(0, min(width, min(x0, x1)))
  right = max(0, min(width, max(x0, x1)))
  top = max(0, min(height, min(y0, y1)))
  bottom = max(0, min(height, max(y0, y1)))
  return left, top, right, bottom


def detect_censor_in_rect(
  rgb_array: np.ndarray,
  rect: tuple[int, int, int, int],
  params: DetectionParams | None = None,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
  """Detect black and/or white censor pixels inside ``rect`` of an RGB uint8 array.

  Returns ``(clamped_rect, roi_mask)`` where ``roi_mask`` has shape
  ``(bottom - top, right - left)`` and dtype ``bool``.
  """
  if params is None:
    params = DetectionParams()
  if params.mode not in ("black", "white", "both"):
    raise ValueError(f"Unsupported detection mode: {params.mode!r}")

  height, width = rgb_array.shape[:2]
  left, top, right, bottom = clamp_rect(rect, width, height)
  if left >= right or top >= bottom:
    return (left, top, right, bottom), np.zeros((0, 0), dtype=bool)

  # Include real image context around the selection so morphological opening
  # does not clip bars touching the rectangle border.
  context = max(0, params.min_thickness) + max(0, params.dilation)
  pad_left = max(0, left - context)
  pad_top = max(0, top - context)
  pad_right = min(width, right + context)
  pad_bottom = min(height, bottom + context)

  patch = rgb_array[pad_top:pad_bottom, pad_left:pad_right]
  r = patch[..., 0].astype(np.int32)
  g = patch[..., 1].astype(np.int32)
  b = patch[..., 2].astype(np.int32)

  cmax = np.maximum(np.maximum(r, g), b)
  cmin = np.minimum(np.minimum(r, g), b)
  spread = cmax - cmin
  # Rec.601 luminance in [0, 255]
  luma = (77 * r + 150 * g + 29 * b) >> 8
  achromatic = spread <= params.max_saturation

  candidates = np.zeros(patch.shape[:2], dtype=bool)
  if params.mode in ("black", "both"):
    black = achromatic & (luma <= params.black_threshold)
    if params.min_thickness > 1:
      black = morphological_open(black, params.min_thickness)
    candidates |= black
  if params.mode in ("white", "both"):
    white = achromatic & (luma >= params.white_threshold)
    if params.min_thickness > 1:
      white = morphological_open(white, params.min_thickness)
    candidates |= white

  if params.dilation > 0:
    candidates = dilate_mask(candidates, params.dilation)

  rel_left = left - pad_left
  rel_top = top - pad_top
  roi_mask = candidates[
    rel_top : rel_top + (bottom - top),
    rel_left : rel_left + (right - left),
  ].copy()
  return (left, top, right, bottom), roi_mask


def apply_mask_to_image(
  image: Image.Image,
  mask: np.ndarray,
  mask_color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
  """Return a copy of ``image`` with ``mask`` pixels painted in ``mask_color``."""
  if mask.shape != (image.height, image.width):
    raise ValueError(
      f"Mask shape {mask.shape} does not match image size {(image.height, image.width)}."
    )
  has_alpha = "A" in image.getbands() or "transparency" in image.info
  rgb = np.asarray(image.convert("RGB")).copy()
  rgb[mask] = np.asarray(mask_color, dtype=np.uint8)
  result = Image.fromarray(rgb, mode="RGB")
  if has_alpha:
    result.putalpha(image.convert("RGBA").getchannel("A"))
  return result
