"""Convert pretrained LaMa checkpoints into Pure PyTorch state_dict weights.

Supports:
  - TorchScript archives (e.g. ``big-lama.pt``, ``anime-manga-big-lama.pt``)
  - PyTorch / Lightning checkpoints (e.g. ``best.ckpt``, ``lama_large_512px.ckpt``)
  - Safetensors files (e.g. ``lama-manga.safetensors``, run with ``--with safetensors``)
"""

import argparse
import pickle
import types
from pathlib import Path

import torch

from lama import FFCResNetGenerator, infer_lama_config
from libs.outputs import atomic_output
from model import LAMA_WEIGHTS_PATH

GENERATOR_PREFIXES = (
  "model.generator.",
  "generator.",
  "module.generator.",
  "netG.",
)


def extract_generator_state(raw_state):
  """Extract and normalize generator keys to ``model.*``."""
  if isinstance(raw_state, dict) and "state_dict" in raw_state:
    raw_state = raw_state["state_dict"]
  if not isinstance(raw_state, dict):
    raise TypeError("Unsupported checkpoint payload: expected a state dict mapping")

  for prefix in GENERATOR_PREFIXES:
    matched = {
      key[len(prefix) :]: value
      for key, value in raw_state.items()
      if key.startswith(prefix)
    }
    if matched:
      return _normalize_model_keys(matched)

  return _normalize_model_keys(raw_state)


def _normalize_model_keys(state):
  if any(key.startswith("model.") for key in state):
    return {
      key: value.detach().cpu()
      for key, value in state.items()
      if key.startswith("model.")
    }
  if any(key.split(".")[0].isdigit() for key in state):
    return {
      f"model.{key}": value.detach().cpu()
      for key, value in state.items()
      if key.split(".")[0].isdigit()
    }
  raise ValueError("Could not find FFCResNetGenerator weights in checkpoint")


class _MissingClassStub:
  """Placeholder for non-tensor metadata classes (e.g. pytorch_lightning) in .ckpt files."""

  def __init__(self, *args, **kwargs):
    pass

  def __setstate__(self, state):
    pass

  def __setitem__(self, key, value):
    pass

  def append(self, value):
    pass

  def extend(self, values):
    pass


class _TolerantUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    try:
      return super().find_class(module, name)
    except (AttributeError, ModuleNotFoundError):
      return _MissingClassStub


def _make_tolerant_pickle_module():
  module = types.ModuleType("_tolerant_pickle")
  module.__dict__.update(pickle.__dict__)
  module.Unpickler = _TolerantUnpickler
  return module


def load_source_state(source):
  suffix = source.suffix.lower()
  if suffix == ".safetensors":
    try:
      from safetensors.torch import load_file
    except ImportError as error:
      raise RuntimeError(
        "Reading .safetensors requires the safetensors package. "
        "Run with: uv run --locked --with safetensors python src/convert_lama.py ..."
      ) from error
    return load_file(str(source), device="cpu")

  # Prefer safe state_dict loading first, then TorchScript, then legacy checkpoints.
  try:
    return torch.load(source, map_location="cpu", weights_only=True)
  except (RuntimeError, ValueError, pickle.UnpicklingError):
    pass

  try:
    scripted = torch.jit.load(str(source), map_location="cpu")
  except (RuntimeError, ValueError):
    scripted = None
  if scripted is not None:
    return scripted.state_dict()

  return torch.load(
    source,
    map_location="cpu",
    pickle_module=_make_tolerant_pickle_module(),
    weights_only=False,
  )


def convert_checkpoint(source, output):
  raw_state = load_source_state(source)
  weights = extract_generator_state(raw_state)
  config = infer_lama_config(weights)
  generator = FFCResNetGenerator(**config)
  generator.load_state_dict(weights, strict=True)

  output.parent.mkdir(parents=True, exist_ok=True)
  with atomic_output(output) as temporary, temporary.open("wb") as stream:
    torch.save(generator.state_dict(), stream)
  return config


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "source",
    type=Path,
    help="path to a pretrained LaMa checkpoint (.pt, .ckpt, or .safetensors)",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=LAMA_WEIGHTS_PATH,
    help=f"destination path for the converted state_dict (default: {LAMA_WEIGHTS_PATH})",
  )
  args = parser.parse_args()
  if not args.source.is_file():
    parser.error(f"Source checkpoint not found: {args.source}")

  try:
    config = convert_checkpoint(args.source, args.output)
  except (OSError, RuntimeError, TypeError, ValueError) as error:
    parser.exit(1, f"Conversion failed: {error}\n")

  print(
    f"Converted LaMa ({config['n_blocks']} blocks, ngf={config['ngf']}) -> {args.output}"
  )


if __name__ == "__main__":
  main()
