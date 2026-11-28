import numpy as np
from PIL import Image, ImageDraw


# convert PIL image to numpy array
def image_to_array(image):
  array = np.asarray(image)
  return np.array(array / 255.0)


def absorb_mask_fringe(
  image,
  exact_mask,
  mask_color,
  *,
  max_steps=2,
  color_tolerance=24,
  chroma_tolerance=40,
):
  """Expand an exact mask by up to ``max_steps`` 8-connected rings into anti-aliased fringe pixels."""
  if not exact_mask.any() or max_steps <= 0:
    return exact_mask

  pixels = np.asarray(image, dtype=np.int16)
  target = np.rint(mask_color).astype(np.int16)
  near_exact = np.max(np.abs(pixels - target), axis=-1) <= color_tolerance

  high = target >= 128
  low = ~high
  if high.any() and low.any() and (int(target.max()) - int(target.min()) >= 128):
    high_min = pixels[..., high].min(axis=-1)
    low_max = pixels[..., low].max(axis=-1)
    fringe_candidate = near_exact | ((high_min - low_max) >= chroma_tolerance)
  else:
    fringe_candidate = near_exact

  mask = exact_mask.copy()
  height, width = mask.shape
  for _ in range(max_steps):
    dilated = mask.copy()
    if height > 1:
      dilated[:-1, :] |= mask[1:, :]
      dilated[1:, :] |= mask[:-1, :]
    if width > 1:
      dilated[:, :-1] |= mask[:, 1:]
      dilated[:, 1:] |= mask[:, :-1]
    if height > 1 and width > 1:
      dilated[:-1, :-1] |= mask[1:, 1:]
      dilated[:-1, 1:] |= mask[1:, :-1]
      dilated[1:, :-1] |= mask[:-1, 1:]
      dilated[1:, 1:] |= mask[:-1, :-1]
    grown = mask | (dilated & fringe_candidate)
    if np.array_equal(grown, mask):
      break
    mask = grown
  return mask


def mask_pixels(image, mask_color, *, absorb_fringe=True):
  red, green, blue = np.asarray(image).transpose(2, 0, 1)
  exact = (red == mask_color[0]) & (green == mask_color[1]) & (blue == mask_color[2])
  if absorb_fringe:
    return absorb_mask_fringe(image, exact, mask_color)
  return exact


def find_regions(image, mask_color):
  return connected_regions(mask_pixels(image, mask_color))


def connected_regions(mask):
  # Scan columns first to retain the ordering of regions of the same size.
  columns, rows = np.nonzero(mask.T)
  pixels = list(zip(columns.tolist(), rows.tolist(), strict=True))
  pending = set(pixels)
  regions = []
  for pixel in pixels:
    if pixel not in pending:
      continue
    pending.remove(pixel)
    region = {pixel}
    frontier = [pixel]
    while frontier:
      x, y = frontier.pop()
      for adjacent in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if adjacent in pending:
          pending.remove(adjacent)
          region.add(adjacent)
          frontier.append(adjacent)
    regions.append(region)
  regions.sort(key=len, reverse=True)
  return regions


def expand_bounding(img, region, expand_factor=1.5, min_size=256):
  x, y = zip(*region)
  left, top, right, bottom = min(x), min(y), max(x) + 1, max(y) + 1
  width, height = img.size
  span = max(right - left, bottom - top)
  limit = min(width, height)
  if span > limit:
    limit = max(width, height)
  side = min(max(span, min_size, int(span * expand_factor)), limit)
  crop_width, crop_height = min(side, width), min(side, height)
  x0 = min(max((left + right - crop_width) // 2, 0), width - crop_width)
  y0 = min(max((top + bottom - crop_height) // 2, 0), height - crop_height)
  return x0, y0, x0 + crop_width, y0 + crop_height


def prepare_crop(image, marked, box):
  left, top, right, bottom = box
  width, height = right - left, bottom - top
  crop_mask = marked[top:bottom, left:right]
  touches_image_edge = (
    (left == 0 and crop_mask[:, 0].any())
    or (right == image.width and crop_mask[:, -1].any())
    or (top == 0 and crop_mask[0].any())
    or (bottom == image.height and crop_mask[-1].any())
  )
  # Keep edge marks away from the decoder's zero padding.
  edge_margin = 8 if touches_image_edge else 0
  size = 256
  scale = (size - 2 * edge_margin) / max(width, height)
  resized = max(1, round(width * scale)), max(1, round(height * scale))
  cropped = image.crop(box)

  if resized == (width, height):
    rgb = image_to_array(cropped)
    missing = crop_mask.copy()
  elif scale < 1:
    # Average in float so even a single marked source pixel survives; BOX on RGB
    # shares the same support, so unmasked pixels never bleed mask colors.
    mask = Image.fromarray(crop_mask.astype(np.float32)).resize(
      resized, Image.Resampling.BOX
    )
    missing = np.asarray(mask) > 0
    rgb = image_to_array(cropped.resize(resized, Image.Resampling.BOX))
  else:
    mask = Image.fromarray(crop_mask).resize(resized, Image.Resampling.NEAREST)
    missing = np.asarray(mask)
    rgb = image_to_array(cropped.resize(resized, Image.Resampling.NEAREST))
    valid = (~crop_mask).astype(np.float32)
    if valid.any():
      src = np.asarray(cropped, dtype=np.float32) / 255.0
      weight = np.asarray(
        Image.fromarray(valid).resize(resized, Image.Resampling.BILINEAR)
      )
      channels = [
        np.asarray(
          Image.fromarray(src[..., c] * valid).resize(
            resized, Image.Resampling.BILINEAR
          )
        )
        for c in range(3)
      ]
      upscaled = np.stack(channels, axis=-1)
      ok = (~missing) & (weight > 1e-6)
      rgb[ok] = np.clip(upscaled[ok] / weight[ok, None], 0.0, 1.0)

  pad_x, pad_y = size - resized[0], size - resized[1]
  x0, y0 = pad_x // 2, pad_y // 2
  if pad_x or pad_y:
    # Use the same padding for RGB and mask to keep marks out of known context.
    padding = ((y0, pad_y - y0), (x0, pad_x - x0))
    rgb = np.pad(rgb, (*padding, (0, 0)), mode="reflect")
    missing = np.pad(missing, padding, mode="reflect")
  return rgb, missing, (slice(y0, y0 + resized[1]), slice(x0, x0 + resized[0]))


def resize_prediction_float(prediction, size):
  """Resize a ``[-1, 1]`` PEPSI prediction to ``size`` as ``float32`` in ``[0, 255]``."""
  channels = [
    np.asarray(Image.fromarray(channel).resize(size, Image.Resampling.BICUBIC))
    for channel in prediction.transpose(2, 0, 1)
  ]
  rgb = (np.stack(channels, axis=-1) + 1.0) * 127.5
  return rgb.clip(0.0, 255.0).astype(np.float32)


def resize_prediction(prediction, size):
  rgb = resize_prediction_float(prediction, size)
  return np.rint(rgb).astype(np.uint8)


def suppress_monochrome_chroma(
  prediction,
  source,
  region_mask,
  known_mask,
  *,
  p95_threshold=4.0,
  p99_threshold=8.0,
):
  """Project restored pixels onto the monochrome luminance axis when context is achromatic."""
  if not region_mask.any() or not known_mask.any():
    return prediction

  known_rgb = source[known_mask].astype(np.float32)
  spread = known_rgb.max(axis=1) - known_rgb.min(axis=1)
  if (
    np.percentile(spread, 95) > p95_threshold
    or np.percentile(spread, 99) > p99_threshold
  ):
    return prediction

  luma_weights = np.array((0.299, 0.587, 0.114), dtype=np.float32)
  known_y = known_rgb @ luma_weights
  mean_tint = (known_rgb - known_y[:, None]).mean(axis=0)

  out = prediction.copy()
  pred_y = out[region_mask] @ luma_weights
  out[region_mask] = np.clip(pred_y[:, None] + mean_tint, 0.0, 255.0)
  return out


def seamless_blend_region(
  raw_prediction,
  source,
  region_mask,
  known_mask,
  *,
  screening=0.015,
  max_shift=28.0,
  max_iters=160,
  tol=0.05,
):
  """Reduce mask-boundary seams by solving a screened harmonic residual correction."""
  if not region_mask.any() or not known_mask.any():
    return raw_prediction

  height, width = region_mask.shape
  ys, xs = np.nonzero(region_mask)
  y0 = max(0, int(ys.min()) - 1)
  y1 = min(height, int(ys.max()) + 2)
  x0 = max(0, int(xs.min()) - 1)
  x1 = min(width, int(xs.max()) + 2)

  inside = region_mask[y0:y1, x0:x1]
  known = known_mask[y0:y1, x0:x1]
  sub_h, sub_w = inside.shape

  ring = np.zeros_like(inside)
  if sub_h > 1:
    ring[:-1, :] |= inside[1:, :]
    ring[1:, :] |= inside[:-1, :]
  if sub_w > 1:
    ring[:, :-1] |= inside[:, 1:]
    ring[:, 1:] |= inside[:, :-1]
  ring &= known & ~inside
  if not ring.any():
    return raw_prediction

  sub_raw = raw_prediction[y0:y1, x0:x1].astype(np.float32)
  sub_src = source[y0:y1, x0:x1].astype(np.float32)
  residual = np.zeros_like(sub_raw)
  residual[ring] = np.clip(sub_src[ring] - sub_raw[ring], -max_shift, max_shift)

  active = inside | ring
  valid_nb = np.zeros((sub_h, sub_w), dtype=np.float32)
  if sub_h > 1:
    valid_nb[1:, :] += active[:-1, :]
    valid_nb[:-1, :] += active[1:, :]
  if sub_w > 1:
    valid_nb[:, 1:] += active[:, :-1]
    valid_nb[:, :-1] += active[:, 1:]
  denom = (valid_nb + screening)[..., None]

  delta = np.zeros((sub_h + 2, sub_w + 2, 3), dtype=np.float32)
  cur = delta[1:-1, 1:-1]
  cur[ring] = residual[ring]
  cur[inside] = residual[ring].mean(axis=0)

  yy, xx = np.indices((sub_h, sub_w))
  red = inside & (((yy + xx) & 1) == 0)
  black = inside & (((yy + xx) & 1) == 1)
  omega = 1.65

  for step in range(max_iters):
    max_change = 0.0
    check = step % 5 == 4
    for color_mask in (red, black):
      if not color_mask.any():
        continue
      nb_sum = delta[:-2, 1:-1] + delta[2:, 1:-1] + delta[1:-1, :-2] + delta[1:-1, 2:]
      step_delta = omega * (nb_sum[color_mask] / denom[color_mask] - cur[color_mask])
      cur[color_mask] += step_delta
      if check:
        max_change = max(max_change, float(np.max(np.abs(step_delta))))
    if check and max_change < tol:
      break

  out = raw_prediction.copy()
  out[y0:y1, x0:x1][inside] = np.clip(sub_raw[inside] + cur[inside], 0.0, 255.0)
  return out


def prepare_lama_crop(image, marked, box, max_size=512):
  """Prepare a ``[0, 1]`` RGB crop and boolean missing mask padded to a multiple of 8."""
  left, top, right, bottom = box
  width, height = right - left, bottom - top
  crop_mask = marked[top:bottom, left:right]

  if max(width, height) > max_size:
    scale = max_size / max(width, height)
    resized = max(1, round(width * scale)), max(1, round(height * scale))
    rgb = (
      np.asarray(
        image.crop(box).resize(resized, Image.Resampling.LANCZOS),
        dtype=np.float32,
      )
      / 255.0
    )
    mask = Image.fromarray(crop_mask.astype(np.float32)).resize(
      resized, Image.Resampling.BOX
    )
    missing = np.asarray(mask) > 0
  else:
    resized = (width, height)
    rgb = np.asarray(image.crop(box), dtype=np.float32) / 255.0
    missing = crop_mask.copy()

  pad_x = (8 - resized[0] % 8) % 8
  pad_y = (8 - resized[1] % 8) % 8
  if pad_x or pad_y:
    padding = ((0, pad_y), (0, pad_x))
    mode = "reflect" if min(resized) > 1 else "edge"
    rgb = np.pad(rgb, (*padding, (0, 0)), mode=mode)
    missing = np.pad(missing, padding, mode=mode)

  return rgb, missing, (slice(0, resized[1]), slice(0, resized[0]))


def resize_lama_prediction(prediction, size):
  """Convert a ``[0, 1]`` unpadded LaMa prediction back to ``uint8`` at ``size``."""
  if (prediction.shape[1], prediction.shape[0]) != size:
    channels = [
      np.asarray(Image.fromarray(channel).resize(size, Image.Resampling.BICUBIC))
      for channel in prediction.transpose(2, 0, 1)
    ]
    prediction = np.stack(channels, axis=-1)
  rgb = prediction * 255.0
  return np.rint(rgb.clip(0, 255)).astype(np.uint8)


def group_anima_regions(
  image_size,
  regions,
  *,
  max_size=1024,
  min_crop_size=512,
  expand_factor=2.5,
  merge_margin=64,
  whole_image_coverage=0.55,
):
  """Group nearby censored regions into merged context crops or a whole-image pass.

  Returns a list of ``(box, merged_region_pixels)`` pairs where each ``box`` is
  ``(left, top, right, bottom)`` within ``image_size``.
  """
  if not regions:
    return []

  img_w, img_h = image_size
  all_pixels = set().union(*regions)
  whole_box = (0, 0, img_w, img_h)

  # Small/medium images fit cleanly in a single full-image pass with full style context.
  if (img_w * img_h <= max_size * max_size) and (
    max(img_w, img_h) <= int(max_size * 1.25)
  ):
    return [(whole_box, all_pixels)]

  def _expand_rect(left, top, right, bottom):
    span = max(right - left, bottom - top)
    target = max(span, min_crop_size, int(round(span * expand_factor)))
    crop_w = min(max(right - left, target), img_w)
    crop_h = min(max(bottom - top, target), img_h)
    x0 = min(max((left + right - crop_w) // 2, 0), img_w - crop_w)
    y0 = min(max((top + bottom - crop_h) // 2, 0), img_h - crop_h)
    return (x0, y0, x0 + crop_w, y0 + crop_h)

  clusters = []
  for region in regions:
    xs, ys = zip(*region)
    box = _expand_rect(min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    clusters.append([box, set(region)])

  # Iteratively merge overlapping or nearby context boxes.
  merged = True
  while merged and len(clusters) > 1:
    merged = False
    next_clusters = []
    used = [False] * len(clusters)
    for i in range(len(clusters)):
      if used[i]:
        continue
      used[i] = True
      l1, t1, r1, b1 = clusters[i][0]
      pix = set(clusters[i][1])
      changed = True
      while changed:
        changed = False
        for j in range(i + 1, len(clusters)):
          if used[j]:
            continue
          l2, t2, r2, b2 = clusters[j][0]
          if (
            l1 - merge_margin <= r2
            and l2 - merge_margin <= r1
            and t1 - merge_margin <= b2
            and t2 - merge_margin <= b1
          ):
            used[j] = True
            l1, t1 = min(l1, l2), min(t1, t2)
            r1, b1 = max(r1, r2), max(b1, b2)
            pix |= clusters[j][1]
            changed = True
            merged = True
      next_clusters.append([(l1, t1, r1, b1), pix])
    clusters = next_clusters

  # If the combined area already covers most of the image, run one full-image pass.
  if len(clusters) == 1:
    l0, t0, r0, b0 = clusters[0][0]
    area_ratio = ((r0 - l0) * (b0 - t0)) / max(1, img_w * img_h)
    if area_ratio >= whole_image_coverage:
      return [(whole_box, all_pixels)]
  else:
    ul = min(c[0][0] for c in clusters)
    ut = min(c[0][1] for c in clusters)
    ur = max(c[0][2] for c in clusters)
    ub = max(c[0][3] for c in clusters)
    union_ratio = ((ur - ul) * (ub - ut)) / max(1, img_w * img_h)
    if union_ratio >= whole_image_coverage and max(img_w, img_h) <= int(max_size * 1.5):
      return [(whole_box, all_pixels)]

  return [(box, pix) for box, pix in clusters]


def prepare_anima_crop(image, marked, box, *, min_size=512, max_size=1024, multiple=16):
  """Prepare a ``uint8`` RGB crop and boolean mask aligned to ``multiple`` for Anima.

  Preserves aspect ratio, keeps long side within ``[min_size, max_size]`` (subject to
  ``min_size <= max_size``), avoids mask-color bleed during resampling, and reflection-pads
  both RGB and mask to multiples of ``multiple``.
  """
  left, top, right, bottom = box
  width, height = right - left, bottom - top
  crop_mask = marked[top:bottom, left:right]
  cropped = image.crop(box)

  long_side = max(width, height)
  effective_min = min(min_size, max_size)
  if long_side > max_size:
    scale = max_size / long_side
  elif long_side < effective_min:
    scale = effective_min / long_side
  else:
    scale = 1.0

  resized = (
    max(1, round(width * scale)),
    max(1, round(height * scale)),
  )

  if resized == (width, height):
    rgb = np.asarray(cropped, dtype=np.uint8).copy()
    missing = crop_mask.copy()
  elif scale < 1.0:
    mask_f = Image.fromarray(crop_mask.astype(np.float32)).resize(
      resized, Image.Resampling.BOX
    )
    missing = np.asarray(mask_f) > 0
    rgb = np.asarray(
      cropped.resize(resized, Image.Resampling.BOX), dtype=np.uint8
    ).copy()
  else:
    mask_u = Image.fromarray(crop_mask).resize(resized, Image.Resampling.NEAREST)
    missing = np.asarray(mask_u)
    rgb_f = (
      np.asarray(cropped.resize(resized, Image.Resampling.NEAREST), dtype=np.float32)
      / 255.0
    )
    valid = (~crop_mask).astype(np.float32)
    if valid.any():
      src = np.asarray(cropped, dtype=np.float32) / 255.0
      weight = np.asarray(
        Image.fromarray(valid).resize(resized, Image.Resampling.BILINEAR)
      )
      channels = [
        np.asarray(
          Image.fromarray(src[..., c] * valid).resize(
            resized, Image.Resampling.BILINEAR
          )
        )
        for c in range(3)
      ]
      upscaled = np.stack(channels, axis=-1)
      ok = (~missing) & (weight > 1e-6)
      rgb_f[ok] = np.clip(upscaled[ok] / weight[ok, None], 0.0, 1.0)
    rgb = np.rint(rgb_f.clip(0.0, 1.0) * 255.0).astype(np.uint8)

  pad_x = (multiple - resized[0] % multiple) % multiple
  pad_y = (multiple - resized[1] % multiple) % multiple
  if pad_x or pad_y:
    padding = ((0, pad_y), (0, pad_x))
    mode = "reflect" if min(resized) > 1 else "edge"
    rgb = np.pad(rgb, (*padding, (0, 0)), mode=mode)
    missing = np.pad(missing, padding, mode=mode)

  return rgb, missing, (slice(0, resized[1]), slice(0, resized[0]))


def resize_anima_prediction(prediction_unit, size):
  """Resize a ``[0, 1]`` float Anima prediction back to ``size`` as ``float32`` in ``[0, 255]``."""
  if (prediction_unit.shape[1], prediction_unit.shape[0]) != size:
    channels = [
      np.asarray(Image.fromarray(channel).resize(size, Image.Resampling.BICUBIC))
      for channel in prediction_unit.astype(np.float32).transpose(2, 0, 1)
    ]
    prediction_unit = np.stack(channels, axis=-1)
  return (prediction_unit.astype(np.float32) * 255.0).clip(0.0, 255.0)


def prepare_lama_training_crop(image, marked, box, size=512):
  """Prepare a fixed ``size x size`` ``[0, 1]`` RGB crop and missing mask for LaMa training."""
  left, top, right, bottom = box
  width, height = right - left, bottom - top
  crop_mask = marked[top:bottom, left:right]
  scale = size / max(width, height)
  resized = max(1, round(width * scale)), max(1, round(height * scale))

  if resized == (width, height):
    rgb = np.asarray(image.crop(box), dtype=np.float32) / 255.0
    missing = crop_mask.copy()
  elif scale < 1:
    rgb = (
      np.asarray(
        image.crop(box).resize(resized, Image.Resampling.LANCZOS),
        dtype=np.float32,
      )
      / 255.0
    )
    mask = Image.fromarray(crop_mask.astype(np.float32)).resize(
      resized, Image.Resampling.BOX
    )
    missing = np.asarray(mask) > 0
  else:
    rgb = (
      np.asarray(
        image.crop(box).resize(resized, Image.Resampling.BICUBIC),
        dtype=np.float32,
      )
      / 255.0
    )
    mask = Image.fromarray(crop_mask).resize(resized, Image.Resampling.NEAREST)
    missing = np.asarray(mask)

  pad_x, pad_y = size - resized[0], size - resized[1]
  x0, y0 = pad_x // 2, pad_y // 2
  if pad_x or pad_y:
    padding = ((y0, pad_y - y0), (x0, pad_x - x0))
    mode = "reflect" if min(resized) > 1 else "edge"
    rgb = np.pad(rgb, (*padding, (0, 0)), mode=mode)
    missing = np.pad(missing, padding, mode=mode)

  return rgb, missing, (slice(y0, y0 + resized[1]), slice(x0, x0 + resized[0]))


def bridge_thin_gaps(original, marked, output):
  y, x = np.nonzero(marked)
  height, width = marked.shape
  prediction = output[y, x].astype(np.int16)
  best_error = np.full(x.shape, np.inf)
  for dx, dy in ((1, 0), (0, 1)):
    inside = (dx <= x) & (x < width - dx) & (dy <= y) & (y < height - dy)
    candidates = np.flatnonzero(inside)
    px, py = x[candidates], y[candidates]
    known = ~(marked[py - dy, px - dx] | marked[py + dy, px + dx])
    candidates = candidates[known]
    px, py = x[candidates], y[candidates]

    before = original[py - dy, px - dx].astype(np.int16)
    after = original[py + dy, px + dx].astype(np.int16)
    agree = (np.abs(before - after) <= 4).all(axis=1)
    color = (before + after + 1) // 2
    # Where both directions agree internally, favor the one closer to the model.
    error = np.abs(color - prediction[candidates]).sum(axis=1)
    use = agree & (error < best_error[candidates])
    output[py[use], px[use]] = color[use]
    best_error[candidates[use]] = error[use]


if __name__ == "__main__":
  image = Image.open("")
  no_alpha_image = image.convert("RGB")
  draw = ImageDraw.Draw(no_alpha_image)
  for region in find_regions(no_alpha_image, [0, 255, 0]):
    left, top, right, bottom = expand_bounding(no_alpha_image, region)
    draw.rectangle((left, top, right - 1, bottom - 1), outline=(0, 255, 0))
  no_alpha_image.show()
