"""Pure-PyTorch ControlNet-LLLite (v2/v1) adapter for Anima's CosmosTransformer3DModel.

Mirrors the architecture and 4-channel inpainting conditioning format used by
``kohya-ss/sd-scripts`` and ``kohya-ss/ComfyUI-Anima-LLLite`` while mapping
target linear layers onto diffusers' ``CosmosTransformer3DModel``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ASPP_DEFAULT_DILATIONS = (1, 2, 4, 6)

# Maps ComfyUI / sd-scripts block-local linear names to diffusers CosmosTransformer3DModel names.
_ATOMIC_SUFFIX_TO_DIFFUSERS = {
  "self_attn_q_proj": "attn1.to_q",
  "self_attn.q_proj": "attn1.to_q",
  "attn1_to_q": "attn1.to_q",
  "attn1.to_q": "attn1.to_q",
  "self_attn_k_proj": "attn1.to_k",
  "self_attn.k_proj": "attn1.to_k",
  "attn1_to_k": "attn1.to_k",
  "attn1.to_k": "attn1.to_k",
  "self_attn_v_proj": "attn1.to_v",
  "self_attn.v_proj": "attn1.to_v",
  "attn1_to_v": "attn1.to_v",
  "attn1.to_v": "attn1.to_v",
  "self_attn_output_proj": "attn1.to_out.0",
  "self_attn.output_proj": "attn1.to_out.0",
  "attn1_to_out_0": "attn1.to_out.0",
  "attn1.to_out.0": "attn1.to_out.0",
  "cross_attn_q_proj": "attn2.to_q",
  "cross_attn.q_proj": "attn2.to_q",
  "attn2_to_q": "attn2.to_q",
  "attn2.to_q": "attn2.to_q",
  "cross_attn_k_proj": "attn2.to_k",
  "cross_attn.k_proj": "attn2.to_k",
  "attn2_to_k": "attn2.to_k",
  "attn2.to_k": "attn2.to_k",
  "cross_attn_v_proj": "attn2.to_v",
  "cross_attn.v_proj": "attn2.to_v",
  "attn2_to_v": "attn2.to_v",
  "attn2.to_v": "attn2.to_v",
  "cross_attn_output_proj": "attn2.to_out.0",
  "cross_attn.output_proj": "attn2.to_out.0",
  "attn2_to_out_0": "attn2.to_out.0",
  "attn2.to_out.0": "attn2.to_out.0",
  "mlp_layer1": "ff.net.0.proj",
  "mlp.layer1": "ff.net.0.proj",
  "ff_net_0_proj": "ff.net.0.proj",
  "ff.net.0.proj": "ff.net.0.proj",
  "mlp_layer2": "ff.net.2",
  "mlp.layer2": "ff.net.2",
  "ff_net_2": "ff.net.2",
  "ff.net.2": "ff.net.2",
}

_BLOCK_KEY_RE = re.compile(
  r"^(?:lllite_dit_)?(?:diffusion_model[._])?(?:net[._])?(?:blocks|transformer_blocks)[._](\d+)[._](.+)$"
)


def _gn(channels: int) -> nn.GroupNorm:
  groups = 8
  while groups > 1 and channels % groups != 0:
    groups //= 2
  return nn.GroupNorm(groups, channels)


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
  if isinstance(value, bool):
    return value
  if value is None:
    return default
  text = str(value).strip().lower()
  if text in ("true", "1", "yes", "y"):
    return True
  if text in ("false", "0", "no", "n"):
    return False
  return default


def _parse_int_tuple(
  value: str | tuple[int, ...] | list[int] | None, default: tuple[int, ...]
) -> tuple[int, ...]:
  if value is None:
    return default
  if isinstance(value, (tuple, list)):
    return tuple(int(v) for v in value)
  parts = [p.strip() for p in str(value).strip("()[] ").split(",") if p.strip()]
  if not parts:
    return default
  return tuple(int(p) for p in parts)


def map_lllite_name_to_diffusers(lllite_name: str) -> str | None:
  """Map a ControlNet-LLLite target module name to a ``CosmosTransformer3DModel`` path."""
  match = _BLOCK_KEY_RE.match(lllite_name)
  if not match:
    return None
  block_idx, rest = match.group(1), match.group(2)
  mapped_suffix = _ATOMIC_SUFFIX_TO_DIFFUSERS.get(rest)
  if mapped_suffix is None:
    return None
  return f"transformer_blocks.{block_idx}.{mapped_suffix}"


def build_inpaint_cond_image(
  rgb_pm1: torch.Tensor, mask01: torch.Tensor, masked_input: bool = True
) -> torch.Tensor:
  """Build a 4-channel LLLite inpainting conditioning tensor.

  Args:
    rgb_pm1: ``(B, 3, H, W)`` float tensor in ``[-1, 1]``.
    mask01: ``(B, 1, H, W)`` float tensor in ``[0, 1]`` where ``>= 0.5`` marks pixels to fill.
    masked_input: If True, zero out RGB channels inside the mask (matching ``sd-scripts``).

  Returns:
    ``(B, 4, H, W)`` conditioning tensor with mask mapped to ``[-1, 1]``.
  """
  if masked_input:
    keep = (mask01 < 0.5).to(dtype=rgb_pm1.dtype, device=rgb_pm1.device)
    rgb_pm1 = rgb_pm1 * keep
  mask_pm1 = mask01.to(dtype=rgb_pm1.dtype, device=rgb_pm1.device) * 2.0 - 1.0
  return torch.cat([rgb_pm1, mask_pm1], dim=1)


class _ResBlock(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    self.norm1 = _gn(channels)
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.norm2 = _gn(channels)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.conv1(F.silu(self.norm1(x)))
    h = self.conv2(F.silu(self.norm2(h)))
    return x + h


class _ASPP(nn.Module):
  def __init__(self, channels: int, dilations: tuple[int, ...]):
    super().__init__()
    if not dilations:
      raise ValueError("ASPP requires at least one dilation rate.")
    branches = []
    for dilation in dilations:
      if dilation == 1:
        conv = nn.Conv2d(channels, channels, kernel_size=1)
      else:
        conv = nn.Conv2d(
          channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
      branches.append(nn.Sequential(conv, _gn(channels), nn.SiLU()))
    self.branches = nn.ModuleList(branches)
    self.global_pool = nn.AdaptiveAvgPool2d(1)
    self.global_conv = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=1),
      _gn(channels),
      nn.SiLU(),
    )
    self.proj = nn.Sequential(
      nn.Conv2d(channels * (len(dilations) + 1), channels, kernel_size=1),
      _gn(channels),
      nn.SiLU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    height, width = x.shape[-2:]
    outs = [branch(x) for branch in self.branches]
    pooled = self.global_conv(self.global_pool(x))
    pooled = F.interpolate(
      pooled, size=(height, width), mode="bilinear", align_corners=False
    )
    outs.append(pooled)
    return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
  def __init__(
    self,
    cond_dim: int,
    cond_emb_dim: int,
    n_resblocks: int = 0,
    use_aspp: bool = False,
    aspp_dilations: tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
    cond_in_channels: int = 4,
  ):
    super().__init__()
    if cond_dim % 2 != 0:
      raise ValueError(f"cond_dim must be even, got {cond_dim}")
    ch_half = cond_dim // 2
    self.cond_in_channels = cond_in_channels
    self.conv1 = nn.Conv2d(
      cond_in_channels, ch_half, kernel_size=4, stride=4, padding=0
    )
    self.norm1 = _gn(ch_half)
    self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
    self.norm2 = _gn(ch_half)
    self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
    self.norm3 = _gn(cond_dim)
    self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
    self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None
    self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
    self.out_norm = nn.LayerNorm(cond_emb_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = F.silu(self.norm1(self.conv1(x)))
    h = F.silu(self.norm2(self.conv2(h)))
    h = F.silu(self.norm3(self.conv3(h)))
    for block in self.resblocks:
      h = block(h)
    if self.aspp is not None:
      h = self.aspp(h)
    h = self.proj(h)
    batch, channels, hh, ww = h.shape
    h = h.view(batch, channels, hh * ww).permute(0, 2, 1).contiguous()
    return self.out_norm(h)


class LLLiteModuleDiT(nn.Module):
  """Single ControlNet-LLLite low-rank input-perturbation adapter for a DiT Linear layer."""

  def __init__(
    self,
    name: str,
    org_module: nn.Linear,
    cond_emb_dim: int,
    mlp_dim: int,
    *,
    has_film: bool = True,
    has_depth_embed: bool = True,
    multiplier: float = 1.0,
  ):
    super().__init__()
    self.lllite_name = name
    # Store original Linear in a list so its weights are not registered in this module.
    self.org_module = [org_module]
    self.org_forward = org_module.forward
    self.cond_emb_dim = cond_emb_dim
    self.mlp_dim = mlp_dim
    self.multiplier = float(multiplier)
    self._applied = False

    in_dim = org_module.in_features
    self.depth_embed = (
      nn.Parameter(torch.zeros(cond_emb_dim)) if has_depth_embed else None
    )
    self.down = nn.Linear(in_dim, mlp_dim)
    self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)
    if has_film:
      self.cond_to_film: nn.Linear | None = nn.Linear(cond_emb_dim, 2 * mlp_dim)
      nn.init.zeros_(self.cond_to_film.weight)
      nn.init.zeros_(self.cond_to_film.bias)
    else:
      self.cond_to_film = None
    self.up = nn.Linear(mlp_dim, in_dim)
    nn.init.zeros_(self.up.weight)
    nn.init.zeros_(self.up.bias)

    self.cond_emb: torch.Tensor | None = None

  def apply_to(self) -> None:
    if self._applied:
      return
    self.org_forward = self.org_module[0].forward
    self.org_module[0].forward = self.forward
    self._applied = True

  def restore(self) -> None:
    if not self._applied:
      return
    self.org_module[0].forward = self.org_forward
    self._applied = False

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.cond_emb is None or self.multiplier == 0.0:
      return self.org_forward(x)

    target_device = x.device
    target_dtype = x.dtype
    if (
      self.down.weight.device != target_device or self.down.weight.dtype != target_dtype
    ):
      self.to(device=target_device, dtype=target_dtype)

    cond = self.cond_emb.to(device=target_device, dtype=target_dtype)
    if cond.shape[0] == 1 and x.shape[0] > 1:
      cond = cond.expand(x.shape[0], -1, -1)

    if self.depth_embed is not None:
      cond_local = cond + self.depth_embed
    else:
      cond_local = cond

    h = F.silu(self.down(x))
    m = self.mid(torch.cat([cond_local, h], dim=-1))
    if self.cond_to_film is not None:
      gamma, beta = self.cond_to_film(cond_local).chunk(2, dim=-1)
      m = m * (1.0 + gamma) + beta
    m = F.silu(m)
    delta = self.up(m) * self.multiplier
    return self.org_forward(x + delta)


@dataclass(frozen=True)
class LLLiteConfig:
  cond_in_channels: int
  cond_dim: int
  cond_emb_dim: int
  n_resblocks: int
  use_aspp: bool
  aspp_dilations: tuple[int, ...]
  inpaint_masked_input: bool


def infer_lllite_config(
  state_dict: dict[str, torch.Tensor], metadata: dict[str, str] | None = None
) -> LLLiteConfig:
  """Infer ControlNet-LLLite architectural config from safetensors metadata and shapes."""
  meta = metadata or {}

  conv1_w = state_dict.get("lllite_conditioning1.conv1.weight")
  conv3_w = state_dict.get("lllite_conditioning1.conv3.weight")
  proj_w = state_dict.get("lllite_conditioning1.proj.weight")
  if conv1_w is None or conv3_w is None or proj_w is None:
    raise ValueError(
      "ControlNet-LLLite checkpoint is missing lllite_conditioning1 weights."
    )

  cond_in_channels = int(meta.get("lllite.cond_in_channels", conv1_w.shape[1]))
  cond_dim = int(meta.get("lllite.cond_dim", conv3_w.shape[0]))
  cond_emb_dim = int(meta.get("lllite.cond_emb_dim", proj_w.shape[0]))

  resblock_indices = [
    int(match.group(1))
    for key in state_dict
    if (match := re.match(r"^lllite_conditioning1\.resblocks\.(\d+)\.", key))
  ]
  inferred_resblocks = (max(resblock_indices) + 1) if resblock_indices else 0
  n_resblocks = int(meta.get("lllite.cond_resblocks", inferred_resblocks))

  has_aspp_keys = any(k.startswith("lllite_conditioning1.aspp.") for k in state_dict)
  use_aspp = _parse_bool(meta.get("lllite.use_aspp"), default=has_aspp_keys)

  aspp_branch_indices = [
    int(match.group(1))
    for key in state_dict
    if (match := re.match(r"^lllite_conditioning1\.aspp\.branches\.(\d+)\.", key))
  ]
  if "lllite.aspp_dilations" in meta:
    aspp_dilations = _parse_int_tuple(
      meta.get("lllite.aspp_dilations"), ASPP_DEFAULT_DILATIONS
    )
  elif aspp_branch_indices:
    branch_count = max(aspp_branch_indices) + 1
    aspp_dilations = ASPP_DEFAULT_DILATIONS[:branch_count]
  else:
    aspp_dilations = ASPP_DEFAULT_DILATIONS

  inpaint_masked_input = _parse_bool(
    meta.get("lllite.inpaint_masked_input"), default=(cond_in_channels == 4)
  )

  return LLLiteConfig(
    cond_in_channels=cond_in_channels,
    cond_dim=cond_dim,
    cond_emb_dim=cond_emb_dim,
    n_resblocks=n_resblocks,
    use_aspp=use_aspp,
    aspp_dilations=aspp_dilations,
    inpaint_masked_input=inpaint_masked_input,
  )


class ControlNetLLLiteDiT(nn.Module):
  """Manages the conditioning trunk and per-layer LLLite hooks for a DiT model."""

  def __init__(
    self,
    config: LLLiteConfig,
    conditioning1: _Conditioning1,
    modules: list[LLLiteModuleDiT],
  ):
    super().__init__()
    self.config = config
    self.conditioning1 = conditioning1
    self.lllite_modules = nn.ModuleList(modules)
    self._applied = False

  @classmethod
  def from_state_dict(
    cls,
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
    *,
    metadata: dict[str, str] | None = None,
    multiplier: float = 1.0,
    dtype: torch.dtype | None = None,
  ) -> ControlNetLLLiteDiT:
    config = infer_lllite_config(state_dict, metadata)
    if config.cond_in_channels != 4:
      raise ValueError(
        f"Anima inpainting requires a 4-channel inpaint ControlNet-LLLite checkpoint "
        f"(got cond_in_channels={config.cond_in_channels})."
      )

    conditioning1 = _Conditioning1(
      cond_dim=config.cond_dim,
      cond_emb_dim=config.cond_emb_dim,
      n_resblocks=config.n_resblocks,
      use_aspp=config.use_aspp,
      aspp_dilations=config.aspp_dilations,
      cond_in_channels=config.cond_in_channels,
    )
    cond_prefix = "lllite_conditioning1."
    cond_sd = {
      k[len(cond_prefix) :]: v
      for k, v in state_dict.items()
      if k.startswith(cond_prefix)
    }
    conditioning1.load_state_dict(cond_sd, strict=True)

    # Group per-layer weights by module prefix.
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
      if key.startswith(cond_prefix):
        continue
      parts = key.split(".")
      if len(parts) < 2:
        continue
      mod_name = parts[0]
      sub_key = ".".join(parts[1:])
      grouped.setdefault(mod_name, {})[sub_key] = tensor

    named_linears = {
      name: mod
      for name, mod in transformer.named_modules()
      if isinstance(mod, nn.Linear)
    }

    lllite_modules: list[LLLiteModuleDiT] = []
    unmatched: list[str] = []
    for raw_name, sub_sd in sorted(grouped.items()):
      if "down.weight" not in sub_sd or "up.weight" not in sub_sd:
        continue
      target_path = (
        raw_name
        if raw_name in named_linears
        else map_lllite_name_to_diffusers(raw_name)
      )
      if target_path is None or target_path not in named_linears:
        unmatched.append(raw_name)
        continue
      org_linear = named_linears[target_path]
      mlp_dim = int(sub_sd["down.weight"].shape[0])
      has_film = "cond_to_film.weight" in sub_sd
      has_depth_embed = "depth_embed" in sub_sd
      module = LLLiteModuleDiT(
        name=raw_name,
        org_module=org_linear,
        cond_emb_dim=config.cond_emb_dim,
        mlp_dim=mlp_dim,
        has_film=has_film,
        has_depth_embed=has_depth_embed,
        multiplier=multiplier,
      )
      module.load_state_dict(sub_sd, strict=True)
      lllite_modules.append(module)

    if not lllite_modules:
      sample = ", ".join(unmatched[:5]) if unmatched else "none"
      raise ValueError(
        f"No ControlNet-LLLite modules matched the Anima transformer (unmatched sample: {sample})."
      )

    network = cls(config=config, conditioning1=conditioning1, modules=lllite_modules)
    if dtype is not None:
      network.to(dtype=dtype)
    network.eval()
    return network

  @classmethod
  def from_safetensors(
    cls,
    path: str | Path,
    transformer: nn.Module,
    *,
    multiplier: float = 1.0,
    dtype: torch.dtype | None = None,
  ) -> ControlNetLLLiteDiT:
    from safetensors import safe_open
    from safetensors.torch import load_file

    file_path = Path(path)
    with safe_open(str(file_path), framework="pt", device="cpu") as handle:
      metadata = handle.metadata() or {}
    state_dict = load_file(str(file_path), device="cpu")
    return cls.from_state_dict(
      state_dict,
      transformer,
      metadata=metadata,
      multiplier=multiplier,
      dtype=dtype,
    )

  def set_multiplier(self, multiplier: float) -> None:
    value = float(multiplier)
    for module in self.lllite_modules:
      module.multiplier = value

  def apply_to(self) -> None:
    if self._applied:
      return
    for module in self.lllite_modules:
      module.apply_to()
    self._applied = True

  def restore(self) -> None:
    if not self._applied:
      return
    for module in self.lllite_modules:
      module.restore()
    self._applied = False

  @torch.inference_mode()
  def encode_conditioning(
    self,
    rgb_pm1: torch.Tensor,
    mask01: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    offload_trunk_to_cpu: bool = True,
  ) -> torch.Tensor:
    """Compute shared conditioning embeddings and broadcast them to all LLLite modules."""
    cond_input = build_inpaint_cond_image(
      rgb_pm1, mask01, masked_input=self.config.inpaint_masked_input
    )
    self.conditioning1.to(device=device, dtype=dtype)
    try:
      cond_emb = self.conditioning1(cond_input.to(device=device, dtype=dtype))
    finally:
      if offload_trunk_to_cpu and device.type == "cuda":
        self.conditioning1.to(device="cpu")
        torch.cuda.empty_cache()

    for module in self.lllite_modules:
      module.cond_emb = cond_emb
    return cond_emb

  def clear_conditioning(self) -> None:
    for module in self.lllite_modules:
      module.cond_emb = None

  def offload_modules_to_cpu(self) -> None:
    self.clear_conditioning()
    self.conditioning1.to(device="cpu")
    self.lllite_modules.to(device="cpu")
