"""Anima-Base + 4-channel ControlNet-LLLite GPU inpainting engine.

Loads ComfyUI-style standalone ``.safetensors`` weights from ``models/anima/``
fully offline using ``diffusers`` + ``transformers`` and injects lightweight
ControlNet-LLLite hooks onto ``CosmosTransformer3DModel``.
"""

from __future__ import annotations

import gc
import importlib.util
import random
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from anima_lllite import ControlNetLLLiteDiT
from libs.utils import (
  prepare_anima_crop,
  resize_anima_prediction,
  seamless_blend_region,
  suppress_monochrome_chroma,
)

ANIMA_SUBDIRS = ("diffusion", "text_encoder", "vae", "controlnet")
ANIMA_DEFAULT_NEGATIVE_PROMPT = "censored, bar censor, mosaic censoring"
ANIMA_DEFAULT_STEPS = 20
ANIMA_DEFAULT_CFG = 4.0
ANIMA_DEFAULT_LLLITE_STRENGTH = 1.0
ANIMA_DEFAULT_MAX_SIZE = 1024
ANIMA_DEFAULT_SEED = -1
ANIMA_DEFAULT_CACHE_PROMPT = False
ANIMA_DEFAULT_CFG_SKIP_TAIL = 0.0
ANIMA_MAX_CFG_SKIP_TAIL = 0.3
ANIMA_PROMPT_CACHE_MAX_ENTRIES = 8
# Threshold (GiB) below which block-level CPU offloading is automatically enabled.
ANIMA_BLOCK_OFFLOAD_VRAM_GIB = 10.5
# Threshold (GiB) when FP8 weight storage is enabled (~2.1 GiB DiT fits on 8 GiB GPUs).
ANIMA_FP8_BLOCK_OFFLOAD_VRAM_GIB = 6.5

_REQUIRED_PACKAGES = (
  "diffusers",
  "transformers",
  "safetensors",
  "accelerate",
  "tokenizers",
)


class AnimaCancelledError(RuntimeError):
  """Raised when an Anima inference run is cancelled by the user."""


@dataclass
class AnimaSettings:
  positive_prompt: str = ""
  negative_prompt: str = ANIMA_DEFAULT_NEGATIVE_PROMPT
  steps: int = ANIMA_DEFAULT_STEPS
  cfg_scale: float = ANIMA_DEFAULT_CFG
  lllite_strength: float = ANIMA_DEFAULT_LLLITE_STRENGTH
  seed: int = ANIMA_DEFAULT_SEED
  max_size: int = ANIMA_DEFAULT_MAX_SIZE
  use_fp8_storage: bool | None = None
  cache_prompt_embeds: bool = ANIMA_DEFAULT_CACHE_PROMPT
  cfg_skip_tail: float = ANIMA_DEFAULT_CFG_SKIP_TAIL
  diffusion_file: str = ""
  controlnet_file: str = ""

  def normalized(self) -> AnimaSettings:
    steps = max(1, min(100, int(self.steps)))
    cfg = max(1.0, min(20.0, float(self.cfg_scale)))
    strength = max(0.0, min(2.0, float(self.lllite_strength)))
    seed = int(self.seed)
    max_size = max(512, min(1536, int(self.max_size)))
    max_size = (max_size // 16) * 16
    use_fp8 = (
      anima_default_fp8_storage()
      if self.use_fp8_storage is None
      else bool(self.use_fp8_storage)
    )
    cfg_skip_tail = max(0.0, min(ANIMA_MAX_CFG_SKIP_TAIL, float(self.cfg_skip_tail)))
    return AnimaSettings(
      positive_prompt=str(self.positive_prompt or ""),
      negative_prompt=str(self.negative_prompt or ""),
      steps=steps,
      cfg_scale=cfg,
      lllite_strength=strength,
      seed=seed,
      max_size=max_size,
      use_fp8_storage=use_fp8,
      cache_prompt_embeds=bool(self.cache_prompt_embeds),
      cfg_skip_tail=cfg_skip_tail,
      diffusion_file=str(self.diffusion_file or ""),
      controlnet_file=str(self.controlnet_file or ""),
    )

  def to_dict(self) -> dict:
    s = self.normalized()
    return {
      "positive_prompt": s.positive_prompt,
      "negative_prompt": s.negative_prompt,
      "steps": s.steps,
      "cfg_scale": s.cfg_scale,
      "lllite_strength": s.lllite_strength,
      "seed": s.seed,
      "max_size": s.max_size,
      "use_fp8_storage": bool(s.use_fp8_storage),
      "cache_prompt_embeds": bool(s.cache_prompt_embeds),
      "cfg_skip_tail": float(s.cfg_skip_tail),
      "diffusion_file": s.diffusion_file,
      "controlnet_file": s.controlnet_file,
    }

  @classmethod
  def from_dict(cls, data: dict | None) -> AnimaSettings:
    if not data:
      return cls().normalized()
    return cls(
      positive_prompt=data.get("positive_prompt", ""),
      negative_prompt=data.get("negative_prompt", ANIMA_DEFAULT_NEGATIVE_PROMPT),
      steps=data.get("steps", ANIMA_DEFAULT_STEPS),
      cfg_scale=data.get("cfg_scale", ANIMA_DEFAULT_CFG),
      lllite_strength=data.get("lllite_strength", ANIMA_DEFAULT_LLLITE_STRENGTH),
      seed=data.get("seed", ANIMA_DEFAULT_SEED),
      max_size=data.get("max_size", ANIMA_DEFAULT_MAX_SIZE),
      use_fp8_storage=data.get("use_fp8_storage", None),
      cache_prompt_embeds=data.get("cache_prompt_embeds", ANIMA_DEFAULT_CACHE_PROMPT),
      cfg_skip_tail=data.get("cfg_skip_tail", ANIMA_DEFAULT_CFG_SKIP_TAIL),
      diffusion_file=data.get("diffusion_file", ""),
      controlnet_file=data.get("controlnet_file", ""),
    ).normalized()


def _app_root() -> Path:
  if getattr(sys, "frozen", False):
    return Path(sys.executable).resolve().parent
  return Path(__file__).resolve().parent.parent


def _bundle_root() -> Path:
  if getattr(sys, "frozen", False):
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
      return Path(meipass)
    return Path(sys.executable).resolve().parent
  return Path(__file__).resolve().parent.parent


def anima_models_dir() -> Path:
  return _app_root() / "models" / "anima"


def anima_tokenizers_dir() -> Path:
  # Prefer user-override beside exe/repo, fall back to bundled assets.
  user_dir = anima_models_dir() / "tokenizers"
  if (user_dir / "qwen3" / "tokenizer.json").is_file() and (
    user_dir / "t5" / "tokenizer.json"
  ).is_file():
    return user_dir
  return _bundle_root() / "assets" / "anima_tokenizers"


def anima_dependencies_available() -> bool:
  return all(importlib.util.find_spec(pkg) is not None for pkg in _REQUIRED_PACKAGES)


def anima_cuda_available() -> bool:
  return bool(torch.cuda.is_available())


def anima_gpu_vram_gib() -> float | None:
  if not anima_cuda_available():
    return None
  try:
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)
  except (RuntimeError, AssertionError):
    return None


def anima_default_fp8_storage() -> bool:
  """Enable FP8 weight storage by default on <=10.5 GiB GPUs (e.g. 8GB VRAM)."""
  vram_gib = anima_gpu_vram_gib()
  return vram_gib is not None and vram_gib <= ANIMA_BLOCK_OFFLOAD_VRAM_GIB


def anima_tokenizers_available(tok_dir: Path | None = None) -> bool:
  base = anima_tokenizers_dir() if tok_dir is None else Path(tok_dir)
  return (base / "qwen3" / "tokenizer.json").is_file() and (
    base / "t5" / "tokenizer.json"
  ).is_file()


def list_anima_files(subdir: str, base_dir: Path | None = None) -> list[Path]:
  root = anima_models_dir() if base_dir is None else Path(base_dir)
  folder = root / subdir
  if not folder.is_dir():
    return []
  return sorted(
    (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".safetensors"),
    key=lambda p: p.name.casefold(),
  )


def _select_weight_file(
  subdir: str, preferred: str | Path | None = None, base_dir: Path | None = None
) -> Path | None:
  if preferred:
    candidate = Path(preferred)
    if candidate.is_file():
      return candidate
    root = anima_models_dir() if base_dir is None else Path(base_dir)
    named = root / subdir / str(preferred)
    if named.is_file():
      return named
  files = list_anima_files(subdir, base_dir=base_dir)
  return files[0] if files else None


def resolve_anima_paths(
  *,
  base_dir: Path | None = None,
  diffusion_file: str | Path | None = None,
  text_encoder_file: str | Path | None = None,
  vae_file: str | Path | None = None,
  controlnet_file: str | Path | None = None,
) -> dict[str, Path | None]:
  return {
    "diffusion": _select_weight_file("diffusion", diffusion_file, base_dir=base_dir),
    "text_encoder": _select_weight_file(
      "text_encoder", text_encoder_file, base_dir=base_dir
    ),
    "vae": _select_weight_file("vae", vae_file, base_dir=base_dir),
    "controlnet": _select_weight_file("controlnet", controlnet_file, base_dir=base_dir),
  }


def anima_availability(
  *,
  base_dir: Path | None = None,
  tokenizers_dir: Path | None = None,
  require_cuda: bool = True,
) -> tuple[bool, str]:
  """Return ``(available, reason)`` for enabling Anima in UI/CLI."""
  if not anima_dependencies_available():
    missing = [
      pkg for pkg in _REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None
    ]
    return (
      False,
      (
        f"Anima dependencies not installed ({', '.join(missing)}). "
        "Run `uv sync --locked --group anima` or use an Anima-enabled build."
      ),
    )
  if require_cuda and not anima_cuda_available():
    return (
      False,
      "Anima requires a CUDA-capable NVIDIA GPU (CUDA is unavailable on this system).",
    )
  if not anima_tokenizers_available(tokenizers_dir):
    return (
      False,
      f"Anima offline tokenizers not found under {anima_tokenizers_dir()}.",
    )
  paths = resolve_anima_paths(base_dir=base_dir)
  missing_dirs = [name for name, path in paths.items() if path is None]
  if missing_dirs:
    root = anima_models_dir() if base_dir is None else Path(base_dir)
    folders = ", ".join(f"{root / name}" for name in missing_dirs)
    return (
      False,
      f"Anima .safetensors weights missing in: {folders}.",
    )
  return (
    True,
    "Anima (Base + 4ch Inpaint LLLite) is available in Bar censor mode (GPU only).",
  )


def anima_radio_tooltip(
  available: bool | None = None, reason: str | None = None
) -> str:
  if available is None or reason is None:
    available, reason = anima_availability()
  return reason


def _ensure_cosmos_torchvision_shim() -> None:
  """Provide a lightweight PyTorch resize shim if ``torchvision`` is absent.

  ``CosmosTransformer3DModel.forward`` calls ``transforms.functional.resize`` with
  nearest-neighbor interpolation only to resize the all-zero ``padding_mask``.
  """
  from diffusers.models.transformers import transformer_cosmos

  if hasattr(transformer_cosmos, "transforms"):
    return

  def _resize(
    tensor: torch.Tensor, size: list[int] | tuple[int, int], interpolation=None
  ):
    return F.interpolate(tensor, size=tuple(size), mode="nearest")

  transformer_cosmos.transforms = SimpleNamespace(
    functional=SimpleNamespace(resize=_resize),
    InterpolationMode=SimpleNamespace(NEAREST="nearest"),
  )


def _anima_transformer_config() -> dict:
  return {
    "in_channels": 16,
    "out_channels": 16,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "max_size": (128, 240, 240),
    "patch_size": (1, 2, 2),
    "rope_scale": (1.0, 4.0, 4.0),
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
    "use_crossattn_projection": False,
    "crossattn_proj_in_channels": 1024,
    "encoder_hidden_states_channels": 1024,
    "controlnet_block_every_n": None,
    "img_context_dim_in": None,
    "img_context_num_tokens": 256,
    "img_context_dim_out": 2048,
  }


def _anima_text_conditioner_config() -> dict:
  return {
    "source_dim": 1024,
    "target_dim": 1024,
    "model_dim": 1024,
    "num_layers": 6,
    "num_attention_heads": 16,
    "mlp_ratio": 4.0,
    "target_vocab_size": 32128,
    "use_self_attention": True,
    "use_layer_norm": False,
    "min_sequence_length": 512,
  }


def _anima_qwen3_config():
  from transformers import Qwen3Config

  return Qwen3Config(
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=False,
    tie_word_embeddings=False,
    rope_theta=1000000.0,
    attention_bias=False,
    attention_dropout=0.0,
  )


def _anima_scheduler():
  from diffusers import FlowMatchEulerDiscreteScheduler

  return FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000,
    shift=3.0,
    use_dynamic_shifting=False,
    time_shift_type="exponential",
  )


def _strip_optional_prefix(key: str, prefixes: tuple[str, ...]) -> str:
  for prefix in prefixes:
    if key.startswith(prefix):
      return key[len(prefix) :]
  return key


def _convert_anima_diffusion_state_dict(
  raw_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
  """Split a ComfyUI Anima diffusion safetensors into transformer & text_conditioner dicts."""
  from diffusers.loaders.single_file_utils import (
    convert_cosmos_transformer_checkpoint_to_diffusers,
  )

  transformer_raw: dict[str, torch.Tensor] = {}
  conditioner_sd: dict[str, torch.Tensor] = {}
  for key, value in raw_state.items():
    clean = _strip_optional_prefix(key, ("model.diffusion_model.", "diffusion_model."))
    if clean.startswith("llm_adapter."):
      conditioner_sd[clean[len("llm_adapter.") :]] = value
    elif clean.startswith("net.llm_adapter."):
      conditioner_sd[clean[len("net.llm_adapter.") :]] = value
    else:
      transformer_raw[clean] = value

  already_diffusers = any(k.startswith("transformer_blocks.") for k in transformer_raw)
  if already_diffusers:
    transformer_sd = transformer_raw
  else:
    transformer_sd = convert_cosmos_transformer_checkpoint_to_diffusers(transformer_raw)
  return transformer_sd, conditioner_sd


def _convert_qwen3_state_dict(
  raw_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  cleaned: dict[str, torch.Tensor] = {}
  for key, value in raw_state.items():
    if key.startswith(("lm_head.", "model.lm_head.")):
      continue
    clean = _strip_optional_prefix(key, ("transformer.", "model."))
    cleaned[clean] = value
  return cleaned


def _convert_qwen_vae_state_dict(
  raw_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

  cleaned = {
    _strip_optional_prefix(k, ("first_stage_model.", "vae.")): v
    for k, v in raw_state.items()
  }
  if any(k.startswith("encoder.downsamples.") for k in cleaned):
    return convert_wan_vae_to_diffusers(cleaned)
  return cleaned


def _load_state_dict_with_dtype(
  module: torch.nn.Module,
  state_dict: dict[str, torch.Tensor],
  dtype: torch.dtype,
) -> None:
  """Load ``state_dict`` into ``module`` in ``dtype`` without calling ``ModelMixin.to(dtype=...)``.

  Respects ``_keep_in_fp32_modules`` and uses ``assign=True`` so floating-point weights
  land directly in the target dtype without triggering diffusers' cast warning.
  """
  fp32_patterns = tuple(getattr(module, "_keep_in_fp32_modules", None) or ())
  cast_sd: dict[str, torch.Tensor] = {}
  for key, tensor in state_dict.items():
    if tensor.is_floating_point():
      keep_fp32 = any(pattern in key for pattern in fp32_patterns)
      target = torch.float32 if keep_fp32 else dtype
      cast_sd[key] = tensor.to(dtype=target) if tensor.dtype != target else tensor
    else:
      cast_sd[key] = tensor
  module.load_state_dict(cast_sd, strict=True, assign=True)


def _fill_masked_pixels_for_vae(rgb_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """Replace masked RGB pixels with nearest known border colors before VAE encoding.

  This prevents green mask color (or black bar color) from bleeding across the
  mask boundary through VAE convolutional receptive fields.
  """
  if not mask.any() or mask.all():
    return rgb_uint8.copy()

  filled = rgb_uint8.astype(np.float32)
  known = ~mask
  mean_color = filled[known].mean(axis=0)
  filled[mask] = mean_color

  # Iterative neighbor diffusion from the known boundary inward (fast pure NumPy).
  current_known = known.copy()
  remaining = mask.copy()
  for _ in range(16):
    if not remaining.any():
      break
    padded_img = np.pad(filled, ((1, 1), (1, 1), (0, 0)), mode="edge")
    padded_k = np.pad(
      current_known.astype(np.float32), ((1, 1), (1, 1)), mode="constant"
    )
    weight = (
      padded_k[:-2, 1:-1]
      + padded_k[2:, 1:-1]
      + padded_k[1:-1, :-2]
      + padded_k[1:-1, 2:]
    )
    frontier = remaining & (weight > 0)
    if not frontier.any():
      break
    sum_rgb = (
      padded_img[:-2, 1:-1] * padded_k[:-2, 1:-1, None]
      + padded_img[2:, 1:-1] * padded_k[2:, 1:-1, None]
      + padded_img[1:-1, :-2] * padded_k[1:-1, :-2, None]
      + padded_img[1:-1, 2:] * padded_k[1:-1, 2:, None]
    )
    filled[frontier] = sum_rgb[frontier] / weight[frontier, None]
    current_known[frontier] = True
    remaining[frontier] = False

  return np.rint(filled.clip(0.0, 255.0)).astype(np.uint8)


def _cfg_active_steps(total_steps: int, cfg_skip_tail: float) -> int:
  """Return how many leading denoising steps keep CFG active before switching to CFG=1.0."""
  if total_steps <= 1 or cfg_skip_tail <= 0.0:
    return total_steps
  ratio = max(0.0, min(ANIMA_MAX_CFG_SKIP_TAIL, float(cfg_skip_tail)))
  skip_count = min(total_steps - 1, round(total_steps * ratio))
  return total_steps - skip_count


def _build_soft_latent_mask(
  mask01: torch.Tensor, latent_height: int, latent_width: int
) -> torch.Tensor:
  """Build a feathered ``[1, 1, 1, H_lat, W_lat]`` mask preserving 1.0 on masked cells."""
  # Ensure every latent cell touching a masked pixel stays >= 1.0 so bar pixels never leak.
  hard = F.adaptive_max_pool2d(
    mask01.float(), output_size=(latent_height, latent_width)
  )
  hard = (hard > 0.0).float()
  # Add a gentle 1-cell feathered halo outside the hard mask for smooth latent transition.
  dilated = F.max_pool2d(hard, kernel_size=3, stride=1, padding=1)
  soft = torch.maximum(hard, dilated * 0.5)
  return soft.unsqueeze(2)


class AnimaInpainter:
  """GPU-only Anima-Base + 4-channel ControlNet-LLLite inpainter with 8GB VRAM offloading."""

  def __init__(
    self,
    *,
    settings: AnimaSettings | None = None,
    base_dir: Path | None = None,
    tokenizers_dir: Path | None = None,
    diffusion_file: str | Path | None = None,
    text_encoder_file: str | Path | None = None,
    vae_file: str | Path | None = None,
    controlnet_file: str | Path | None = None,
  ):
    self.settings = (settings or AnimaSettings()).normalized()
    if diffusion_file is not None:
      self.settings.diffusion_file = str(diffusion_file)
    if controlnet_file is not None:
      self.settings.controlnet_file = str(controlnet_file)

    ok, reason = anima_availability(
      base_dir=base_dir, tokenizers_dir=tokenizers_dir, require_cuda=True
    )
    if not ok:
      raise RuntimeError(reason)

    self.paths = resolve_anima_paths(
      base_dir=base_dir,
      diffusion_file=self.settings.diffusion_file or None,
      text_encoder_file=text_encoder_file,
      vae_file=vae_file,
      controlnet_file=self.settings.controlnet_file or None,
    )
    for role, path in self.paths.items():
      if path is None or not path.is_file():
        raise FileNotFoundError(f"Missing Anima {role} .safetensors file.")

    self.tokenizers_dir = (
      anima_tokenizers_dir() if tokenizers_dir is None else Path(tokenizers_dir)
    )
    self.device = torch.device("cuda")
    self.fallback_reason = None
    self.batch_size = 1

    self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    self.use_fp8_storage = bool(self.settings.use_fp8_storage)
    total_vram_gib = anima_gpu_vram_gib() or 0.0
    offload_threshold = (
      ANIMA_FP8_BLOCK_OFFLOAD_VRAM_GIB
      if self.use_fp8_storage
      else ANIMA_BLOCK_OFFLOAD_VRAM_GIB
    )
    self.use_block_offload = total_vram_gib <= offload_threshold
    self._prompt_cache: OrderedDict[
      tuple[str, str, bool], tuple[torch.Tensor, torch.Tensor | None]
    ] = OrderedDict()

    _ensure_cosmos_torchvision_shim()
    self._load_components()

  def _load_components(self) -> None:
    from diffusers.hooks import apply_group_offloading
    from diffusers.models import (
      AnimaTextConditioner,
      AutoencoderKLQwenImage,
      CosmosTransformer3DModel,
    )
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3Model, T5TokenizerFast

    # 1. Tokenizers (100% offline)
    self.tokenizer = AutoTokenizer.from_pretrained(
      str(self.tokenizers_dir / "qwen3"), local_files_only=True
    )
    self.t5_tokenizer = T5TokenizerFast.from_pretrained(
      str(self.tokenizers_dir / "t5"), local_files_only=True
    )

    # 2. Qwen3-0.6B Text Encoder (kept on CPU until encoding)
    qwen_raw = load_file(str(self.paths["text_encoder"]), device="cpu")
    qwen_sd = _convert_qwen3_state_dict(qwen_raw)
    del qwen_raw
    self.text_encoder = Qwen3Model(_anima_qwen3_config()).eval()
    _load_state_dict_with_dtype(self.text_encoder, qwen_sd, self.dtype)
    del qwen_sd

    # 3. Anima Diffusion Transformer + llm_adapter (TextConditioner)
    diff_raw = load_file(str(self.paths["diffusion"]), device="cpu")
    transformer_sd, conditioner_sd = _convert_anima_diffusion_state_dict(diff_raw)
    del diff_raw

    self.text_conditioner = AnimaTextConditioner(
      **_anima_text_conditioner_config()
    ).eval()
    if not conditioner_sd:
      raise ValueError(
        f"Diffusion checkpoint {self.paths['diffusion']} does not contain llm_adapter weights."
      )
    _load_state_dict_with_dtype(self.text_conditioner, conditioner_sd, self.dtype)
    del conditioner_sd

    self.transformer = CosmosTransformer3DModel(**_anima_transformer_config()).eval()
    _load_state_dict_with_dtype(self.transformer, transformer_sd, self.dtype)
    del transformer_sd
    if self.use_fp8_storage:
      self.transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=self.dtype,
      )

    # 4. Qwen-Image VAE (kept on CPU until encode/decode; tiling & slicing enabled)
    vae_raw = load_file(str(self.paths["vae"]), device="cpu")
    vae_sd = _convert_qwen_vae_state_dict(vae_raw)
    del vae_raw
    self.vae = AutoencoderKLQwenImage().eval()
    _load_state_dict_with_dtype(self.vae, vae_sd, self.dtype)
    del vae_sd
    self.vae.enable_tiling()
    self.vae.enable_slicing()

    # 5. ControlNet-LLLite (4-channel Inpaint adapter)
    self.lllite = ControlNetLLLiteDiT.from_safetensors(
      self.paths["controlnet"],
      self.transformer,
      multiplier=self.settings.lllite_strength,
      dtype=self.dtype,
    )

    # 6. Block-level CPU offloading for <= 8GB VRAM GPUs
    if self.use_block_offload:
      apply_group_offloading(
        self.transformer,
        onload_device=self.device,
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=False,
      )

    self.scheduler = _anima_scheduler()
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  def update_settings(self, settings: AnimaSettings) -> None:
    self.settings = settings.normalized()
    if hasattr(self, "lllite") and self.lllite is not None:
      self.lllite.set_multiplier(self.settings.lllite_strength)
    if not self.settings.cache_prompt_embeds and hasattr(self, "_prompt_cache"):
      self._prompt_cache.clear()

  def unload(self) -> None:
    if hasattr(self, "_prompt_cache"):
      self._prompt_cache.clear()
    if hasattr(self, "lllite") and self.lllite is not None:
      self.lllite.restore()
      self.lllite.offload_modules_to_cpu()
    for attr in ("transformer", "text_conditioner", "text_encoder", "vae", "lllite"):
      if hasattr(self, attr):
        setattr(self, attr, None)
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  @staticmethod
  def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
      raise AnimaCancelledError("Cancelled by user.")

  @torch.inference_mode()
  def _encode_prompts(
    self,
    positive_prompt: str,
    negative_prompt: str,
    *,
    do_cfg: bool,
    cancel_check: Callable[[], bool] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    self._check_cancel(cancel_check)
    cache_key = (positive_prompt, negative_prompt if do_cfg else "", bool(do_cfg))
    if self.settings.cache_prompt_embeds and cache_key in self._prompt_cache:
      self._prompt_cache.move_to_end(cache_key)
      cached_pos, cached_neg = self._prompt_cache[cache_key]
      pos_gpu = cached_pos.to(device=self.device, dtype=self.dtype)
      neg_gpu = (
        cached_neg.to(device=self.device, dtype=self.dtype)
        if cached_neg is not None
        else None
      )
      return pos_gpu, neg_gpu

    max_seq = 512

    def _qwen_encode(text: str) -> tuple[torch.Tensor, torch.Tensor]:
      tokens = self.tokenizer(
        [text],
        padding="longest",
        max_length=max_seq,
        truncation=True,
        return_tensors="pt",
      )
      input_ids = tokens.input_ids.to(self.device)
      attn_mask = tokens.attention_mask.to(self.device)
      if input_ids.shape[-1] == 0:
        input_ids = input_ids.new_zeros((1, 1))
        attn_mask = attn_mask.new_zeros((1, 1))
      hidden = self.text_encoder(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=False,
      ).last_hidden_state.to(dtype=self.dtype)
      hidden = hidden * attn_mask.to(hidden).unsqueeze(-1)
      return hidden, attn_mask

    def _t5_ids(text: str) -> tuple[torch.Tensor, torch.Tensor]:
      tokens = self.t5_tokenizer(
        [text],
        padding="longest",
        max_length=max_seq,
        truncation=True,
        return_tensors="pt",
      )
      return tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)

    # Phase 1: Qwen3 text encoder on GPU -> immediately offload back to CPU
    self.text_encoder.to(self.device)
    try:
      pos_qwen, pos_qwen_mask = _qwen_encode(positive_prompt)
      neg_qwen = neg_qwen_mask = None
      if do_cfg:
        self._check_cancel(cancel_check)
        neg_qwen, neg_qwen_mask = _qwen_encode(negative_prompt)
    finally:
      self.text_encoder.to("cpu")
      torch.cuda.empty_cache()

    pos_t5_ids, pos_t5_mask = _t5_ids(positive_prompt)
    neg_t5_ids, neg_t5_mask = _t5_ids(negative_prompt) if do_cfg else (None, None)

    # Phase 2: AnimaTextConditioner (llm_adapter) on GPU -> immediately offload
    self._check_cancel(cancel_check)
    self.text_conditioner.to(self.device)
    try:
      pos_embeds = self.text_conditioner(
        source_hidden_states=pos_qwen,
        target_input_ids=pos_t5_ids,
        target_attention_mask=pos_t5_mask,
        source_attention_mask=pos_qwen_mask,
      ).to(dtype=self.dtype)
      neg_embeds = None
      if do_cfg and neg_qwen is not None:
        neg_embeds = self.text_conditioner(
          source_hidden_states=neg_qwen,
          target_input_ids=neg_t5_ids,
          target_attention_mask=neg_t5_mask,
          source_attention_mask=neg_qwen_mask,
        ).to(dtype=self.dtype)
    finally:
      self.text_conditioner.to("cpu")
      torch.cuda.empty_cache()

    if self.settings.cache_prompt_embeds:
      self._prompt_cache[cache_key] = (
        pos_embeds.detach().cpu(),
        neg_embeds.detach().cpu() if neg_embeds is not None else None,
      )
      self._prompt_cache.move_to_end(cache_key)
      while len(self._prompt_cache) > ANIMA_PROMPT_CACHE_MAX_ENTRIES:
        self._prompt_cache.popitem(last=False)

    return pos_embeds, neg_embeds

  @torch.inference_mode()
  def _encode_image_latents(self, rgb_filled_pm1: torch.Tensor) -> torch.Tensor:
    self.vae.to(self.device)
    try:
      inp = rgb_filled_pm1.unsqueeze(2).to(device=self.device, dtype=self.dtype)
      latents = self.vae.encode(inp).latent_dist.mode()
      latents_mean = torch.tensor(
        self.vae.config.latents_mean, device=self.device, dtype=latents.dtype
      ).view(1, self.vae.config.z_dim, 1, 1, 1)
      latents_std = torch.tensor(
        self.vae.config.latents_std, device=self.device, dtype=latents.dtype
      ).view(1, self.vae.config.z_dim, 1, 1, 1)
      return ((latents - latents_mean) / latents_std).to(dtype=torch.float32)
    finally:
      self.vae.to("cpu")
      torch.cuda.empty_cache()

  @torch.inference_mode()
  def _decode_latents(self, latents: torch.Tensor) -> np.ndarray:
    self.vae.to(self.device)
    try:
      z = latents.to(device=self.device, dtype=self.dtype)
      latents_mean = torch.tensor(
        self.vae.config.latents_mean, device=self.device, dtype=z.dtype
      ).view(1, self.vae.config.z_dim, 1, 1, 1)
      latents_std = torch.tensor(
        self.vae.config.latents_std, device=self.device, dtype=z.dtype
      ).view(1, self.vae.config.z_dim, 1, 1, 1)
      z = z * latents_std + latents_mean
      decoded = self.vae.decode(z, return_dict=False)[0][:, :, 0]
      image = ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 0.5)[0]
      return image.permute(1, 2, 0).cpu().numpy()
    finally:
      self.vae.to("cpu")
      torch.cuda.empty_cache()

  @torch.inference_mode()
  def _denoise_crop(
    self,
    rgb_uint8: np.ndarray,
    missing_mask: np.ndarray,
    *,
    pos_embeds: torch.Tensor,
    neg_embeds: torch.Tensor | None,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
    step_callback: Callable[[int, int, str], None] | None = None,
  ) -> np.ndarray:
    height, width = missing_mask.shape
    if height % 16 != 0 or width % 16 != 0:
      raise ValueError(
        f"Anima crop dimensions must be divisible by 16, got {width}x{height}."
      )

    # 1. Build 4ch LLLite conditioning tensor & VAE-encoded known-region latents
    self._check_cancel(cancel_check)
    if step_callback:
      step_callback(0, self.settings.steps, "Encoding image & conditioning...")

    rgb_pm1 = (
      torch.from_numpy(rgb_uint8.astype(np.float32) / 127.5 - 1.0)
      .permute(2, 0, 1)
      .unsqueeze(0)
    )
    mask01 = torch.from_numpy(missing_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    vae_rgb_uint8 = _fill_masked_pixels_for_vae(rgb_uint8, missing_mask)
    vae_rgb_pm1 = (
      torch.from_numpy(vae_rgb_uint8.astype(np.float32) / 127.5 - 1.0)
      .permute(2, 0, 1)
      .unsqueeze(0)
    )
    image_latents = self._encode_image_latents(vae_rgb_pm1)
    latent_h, latent_w = image_latents.shape[-2:]
    soft_mask = _build_soft_latent_mask(mask01, latent_h, latent_w).to(
      device=self.device, dtype=torch.float32
    )

    self._check_cancel(cancel_check)
    self.lllite.set_multiplier(self.settings.lllite_strength)
    self.lllite.encode_conditioning(
      rgb_pm1,
      mask01,
      device=self.device,
      dtype=self.dtype,
      offload_trunk_to_cpu=True,
    )

    # 2. Pure-noise initialization (Denoise = 1.0) + Rectified Flow schedule
    steps = self.settings.steps
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
    self.scheduler.set_begin_index(0)
    timesteps = self.scheduler.timesteps

    generator = torch.Generator(device=self.device).manual_seed(int(seed))
    noise = torch.randn(
      image_latents.shape,
      generator=generator,
      device=self.device,
      dtype=torch.float32,
    )
    latents = noise.clone()
    padding_mask = torch.zeros(
      (1, 1, height, width), device=self.device, dtype=self.dtype
    )
    do_cfg = neg_embeds is not None and self.settings.cfg_scale > 1.0

    # 3. Denoising loop with sequential CFG & step-wise known-region latent preservation
    self.lllite.apply_to()
    if not self.use_block_offload:
      self.transformer.to(self.device)
      self.lllite.lllite_modules.to(device=self.device, dtype=self.dtype)
    try:
      total = len(timesteps)
      cfg_active_until = (
        _cfg_active_steps(total, self.settings.cfg_skip_tail) if do_cfg else 0
      )
      for idx, t in enumerate(timesteps):
        self._check_cancel(cancel_check)
        use_cfg_step = do_cfg and (idx < cfg_active_until)
        if step_callback:
          suffix = "" if (not do_cfg or use_cfg_step) else " [CFG=1]"
          step_callback(idx, total, f"Denoising step {idx + 1}/{total}{suffix}")

        latent_input = latents.to(dtype=self.dtype)
        t_norm = (t.expand(1) / self.scheduler.config.num_train_timesteps).to(
          dtype=self.dtype
        )

        # Sequential CFG (batch=1 twice) keeps peak activation VRAM low on 8GB GPUs.
        cond_pred = self.transformer(
          hidden_states=latent_input,
          timestep=t_norm,
          encoder_hidden_states=pos_embeds,
          padding_mask=padding_mask,
          return_dict=False,
        )[0].float()

        if use_cfg_step:
          self._check_cancel(cancel_check)
          uncond_pred = self.transformer(
            hidden_states=latent_input,
            timestep=t_norm,
            encoder_hidden_states=neg_embeds,
            padding_mask=padding_mask,
            return_dict=False,
          )[0].float()
          noise_pred = uncond_pred + self.settings.cfg_scale * (cond_pred - uncond_pred)
        else:
          noise_pred = cond_pred

        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Known-region latent preservation (Differential Diffusion / soft composite)
        if idx + 1 < total:
          next_sigma = self.scheduler.sigmas[idx + 1].to(
            device=self.device, dtype=torch.float32
          )
          known_noisy = next_sigma * noise + (1.0 - next_sigma) * image_latents
          latents = latents * soft_mask + known_noisy * (1.0 - soft_mask)

        if step_callback:
          step_callback(idx + 1, total, f"Denoising step {idx + 1}/{total}")
    finally:
      self.lllite.restore()
      self.lllite.offload_modules_to_cpu()
      if not self.use_block_offload:
        self.transformer.to("cpu")
      torch.cuda.empty_cache()

    # 4. VAE decode
    self._check_cancel(cancel_check)
    if step_callback:
      step_callback(steps, steps, "Decoding latents...")
    return self._decode_latents(latents)

  def inpaint_crops(
    self,
    ori: Image.Image,
    marked: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    *,
    is_mosaic: bool = False,
    regions: list | None = None,
    seed: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
  ) -> list[np.ndarray]:
    if is_mosaic:
      raise ValueError("Anima is only supported in bar mode.")
    if not boxes:
      return []

    base_seed = (
      int(seed)
      if seed is not None and seed >= 0
      else (
        self.settings.seed if self.settings.seed >= 0 else random.randint(0, 2**31 - 1)
      )
    )
    do_cfg = self.settings.cfg_scale > 1.0
    total_steps = len(boxes) * self.settings.steps

    if progress_callback:
      progress_callback(0, total_steps, "Encoding prompts...")

    pos_embeds, neg_embeds = self._encode_prompts(
      self.settings.positive_prompt,
      self.settings.negative_prompt,
      do_cfg=do_cfg,
      cancel_check=cancel_check,
    )

    results: list[np.ndarray] = []
    region_list = regions if regions is not None else [None] * len(boxes)
    for crop_idx, (box, region) in enumerate(zip(boxes, region_list, strict=True)):
      self._check_cancel(cancel_check)
      left, top, right, bottom = box
      width, height = right - left, bottom - top
      crop_rgb, crop_missing, unpadded_slice = prepare_anima_crop(
        ori, marked, box, max_size=self.settings.max_size
      )

      def _on_step(
        step_in_crop: int,
        crop_steps: int,
        stage: str,
        *,
        _crop_idx: int = crop_idx,
      ) -> None:
        if progress_callback:
          overall = _crop_idx * self.settings.steps + min(
            step_in_crop, self.settings.steps
          )
          prefix = f"Crop {_crop_idx + 1}/{len(boxes)} - " if len(boxes) > 1 else ""
          progress_callback(overall, total_steps, f"{prefix}{stage}")

      pred_unit = self._denoise_crop(
        crop_rgb,
        crop_missing,
        pos_embeds=pos_embeds,
        neg_embeds=neg_embeds,
        seed=base_seed + crop_idx * 997,
        cancel_check=cancel_check,
        step_callback=_on_step,
      )
      pred_unpadded = pred_unit[unpadded_slice]
      pred_float = resize_anima_prediction(pred_unpadded, (width, height))

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
