import numpy as np
from PIL import Image, ImageDraw


# convert PIL image to numpy array
def image_to_array(image):
  array = np.asarray(image)
  return np.array(array / 255.0)


def mask_pixels(image, mask_color):
  red, green, blue = np.asarray(image).transpose(2, 0, 1)
  return (red == mask_color[0]) & (green == mask_color[1]) & (blue == mask_color[2])


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
  rgb = image_to_array(image.crop(box).resize(resized, Image.Resampling.NEAREST))
  if scale < 1:
    # Average in float so even a single marked source pixel survives.
    mask = Image.fromarray(crop_mask.astype(np.float32)).resize(
      resized, Image.Resampling.BOX
    )
    missing = np.asarray(mask) > 0
  else:
    mask = Image.fromarray(crop_mask).resize(resized, Image.Resampling.NEAREST)
    missing = np.asarray(mask)

  pad_x, pad_y = size - resized[0], size - resized[1]
  x0, y0 = pad_x // 2, pad_y // 2
  if pad_x or pad_y:
    # Use the same padding for RGB and mask to keep marks out of known context.
    padding = ((y0, pad_y - y0), (x0, pad_x - x0))
    rgb = np.pad(rgb, (*padding, (0, 0)), mode="reflect")
    missing = np.pad(missing, padding, mode="reflect")
  return rgb, missing, (slice(y0, y0 + resized[1]), slice(x0, x0 + resized[0]))


def resize_prediction(prediction, size):
  channels = [
    np.asarray(Image.fromarray(channel).resize(size, Image.Resampling.BICUBIC))
    for channel in prediction.transpose(2, 0, 1)
  ]
  rgb = (np.stack(channels, axis=-1) + 1) * 127.5
  return np.rint(rgb.clip(0, 255)).astype(np.uint8)


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
