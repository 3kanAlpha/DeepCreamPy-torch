"""Convert the original TensorFlow weights for both inpainting models."""

import tempfile
from pathlib import Path

import tensorflow as tf
import torch

from model import MODELS_DIR, WEIGHTS_NAME, InpaintModel


def convert(mode):
  directory = MODELS_DIR / mode
  checkpoint = tf.train.latest_checkpoint(str(directory))
  if checkpoint is None:
    raise FileNotFoundError(f"No TensorFlow checkpoint found in {directory}")

  reader = tf.train.load_checkpoint(checkpoint)
  model = InpaintModel()
  layers = {"context_merge": "CB1/ML"}
  for part, scope in (("encoder", "G_en"), ("decoder", "G_de")):
    for i in range(len(getattr(model, part))):
      suffix = f"_{i}" if i else ""
      layers[f"{part}.{i}"] = f"{scope}/Conv{suffix}"

  weights = {}
  for target, source in layers.items():
    kernel = reader.get_tensor(f"{source}/weights")
    weights[f"{target}.weight"] = (
      torch.from_numpy(kernel).permute(3, 2, 0, 1).contiguous()
    )
    weights[f"{target}.bias"] = torch.from_numpy(reader.get_tensor(f"{source}/biases"))
  model.load_state_dict(weights)

  output = directory / WEIGHTS_NAME
  with tempfile.NamedTemporaryFile(dir=directory, delete=False) as file:
    temporary = Path(file.name)
  try:
    torch.save(model.state_dict(), temporary)
    temporary.replace(output)
  finally:
    temporary.unlink(missing_ok=True)
  print(f"Converted {mode}: {output}")


if __name__ == "__main__":
  for model_mode in ("bar", "mosaic"):
    convert(model_mode)
