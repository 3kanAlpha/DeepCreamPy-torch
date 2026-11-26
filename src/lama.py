"""Pure PyTorch implementation of LaMa's FFCResNetGenerator.

Matches the layer structure and state_dict keys of ``advimman/lama`` so that
converted ``big-lama`` (18 blocks, ~51M params) and standard LaMa (9 blocks,
~27M params) checkpoints load with ``strict=True`` and run on both CPU and CUDA.
"""

import torch
from torch import nn


class FourierUnit(nn.Module):
  def __init__(self, in_channels, out_channels, groups=1, fft_norm="ortho"):
    super().__init__()
    self.groups = groups
    self.fft_norm = fft_norm
    self.conv_layer = nn.Conv2d(
      in_channels=in_channels * 2,
      out_channels=out_channels * 2,
      kernel_size=1,
      stride=1,
      padding=0,
      groups=self.groups,
      bias=False,
    )
    self.bn = nn.BatchNorm2d(out_channels * 2)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    batch = x.shape[0]
    orig_dtype = x.dtype
    x_fp32 = x.float()

    # (B, C, H, W // 2 + 1) complex
    ffted = torch.fft.rfftn(x_fp32, dim=(-2, -1), norm=self.fft_norm)
    ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
    # Interleave real/imag per channel to match official LaMa checkpoints.
    ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
    ffted = ffted.view((batch, -1) + ffted.size()[3:])

    ffted = self.conv_layer(ffted)
    ffted = self.relu(self.bn(ffted))

    ffted = ffted.view((batch, -1, 2) + ffted.size()[2:])
    ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
    ffted = torch.complex(ffted[..., 0].float(), ffted[..., 1].float())
    output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
    return output.to(orig_dtype)


class SpectralTransform(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
    super().__init__()
    self.enable_lfu = enable_lfu
    self.downsample = (
      nn.AvgPool2d(kernel_size=(2, 2), stride=2) if stride == 2 else nn.Identity()
    )
    self.stride = stride
    self.conv1 = nn.Sequential(
      nn.Conv2d(
        in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False
      ),
      nn.BatchNorm2d(out_channels // 2),
      nn.ReLU(inplace=True),
    )
    self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
    if self.enable_lfu:
      self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
    self.conv2 = nn.Conv2d(
      out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False
    )

  def forward(self, x):
    x = self.downsample(x)
    x = self.conv1(x)
    output = self.fu(x)

    if self.enable_lfu:
      _, channels, height, width = x.shape
      split_no = 2
      split_h = height // split_no
      split_w = width // split_no
      xs = torch.cat(torch.split(x[:, : channels // 4], split_h, dim=-2), dim=1)
      xs = torch.cat(torch.split(xs, split_w, dim=-1), dim=1)
      xs = self.lfu(xs)
      xs = xs.repeat(1, 1, split_no, split_no).contiguous()
      output = output + xs

    return self.conv2(x + output)


class FFC(nn.Module):
  def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size,
    ratio_gin,
    ratio_gout,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=False,
    enable_lfu=False,
    padding_type="reflect",
  ):
    super().__init__()
    in_cg = int(in_channels * ratio_gin)
    in_cl = in_channels - in_cg
    out_cg = int(out_channels * ratio_gout)
    out_cl = out_channels - out_cg

    self.ratio_gin = ratio_gin
    self.ratio_gout = ratio_gout

    self.convl2l = (
      nn.Identity()
      if in_cl == 0 or out_cl == 0
      else nn.Conv2d(
        in_cl,
        out_cl,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode=padding_type,
      )
    )
    self.convl2g = (
      nn.Identity()
      if in_cl == 0 or out_cg == 0
      else nn.Conv2d(
        in_cl,
        out_cg,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode=padding_type,
      )
    )
    self.convg2l = (
      nn.Identity()
      if in_cg == 0 or out_cl == 0
      else nn.Conv2d(
        in_cg,
        out_cl,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode=padding_type,
      )
    )
    self.convg2g = (
      nn.Identity()
      if in_cg == 0 or out_cg == 0
      else SpectralTransform(
        in_cg,
        out_cg,
        stride,
        1 if groups == 1 else groups // 2,
        enable_lfu=enable_lfu,
      )
    )

  def forward(self, x):
    x_l, x_g = x if isinstance(x, tuple) else (x, 0)
    out_xl, out_xg = 0, 0

    if self.ratio_gout != 1:
      out_xl = self.convl2l(x_l) + self.convg2l(x_g)
    if self.ratio_gout != 0:
      out_xg = self.convl2g(x_l) + self.convg2g(x_g)

    return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
  def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size,
    ratio_gin,
    ratio_gout,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=False,
    norm_layer=nn.BatchNorm2d,
    activation_layer=nn.Identity,
    padding_type="reflect",
    enable_lfu=False,
  ):
    super().__init__()
    self.ffc = FFC(
      in_channels,
      out_channels,
      kernel_size,
      ratio_gin,
      ratio_gout,
      stride,
      padding,
      dilation,
      groups,
      bias,
      enable_lfu=enable_lfu,
      padding_type=padding_type,
    )
    out_cg = int(out_channels * ratio_gout)
    out_cl = out_channels - out_cg
    self.bn_l = nn.Identity() if ratio_gout == 1 else norm_layer(out_cl)
    self.bn_g = nn.Identity() if ratio_gout == 0 else norm_layer(out_cg)
    self.act_l = nn.Identity() if ratio_gout == 1 else activation_layer(inplace=True)
    self.act_g = nn.Identity() if ratio_gout == 0 else activation_layer(inplace=True)

  def forward(self, x):
    x_l, x_g = self.ffc(x)
    x_l = self.act_l(self.bn_l(x_l))
    x_g = self.act_g(self.bn_g(x_g))
    return x_l, x_g


class FFCResnetBlock(nn.Module):
  def __init__(
    self,
    dim,
    padding_type="reflect",
    norm_layer=nn.BatchNorm2d,
    activation_layer=nn.ReLU,
    dilation=1,
    ratio_gin=0.75,
    ratio_gout=0.75,
    enable_lfu=False,
  ):
    super().__init__()
    self.conv1 = FFC_BN_ACT(
      dim,
      dim,
      kernel_size=3,
      padding=dilation,
      dilation=dilation,
      norm_layer=norm_layer,
      activation_layer=activation_layer,
      padding_type=padding_type,
      ratio_gin=ratio_gin,
      ratio_gout=ratio_gout,
      enable_lfu=enable_lfu,
    )
    self.conv2 = FFC_BN_ACT(
      dim,
      dim,
      kernel_size=3,
      padding=dilation,
      dilation=dilation,
      norm_layer=norm_layer,
      activation_layer=activation_layer,
      padding_type=padding_type,
      ratio_gin=ratio_gin,
      ratio_gout=ratio_gout,
      enable_lfu=enable_lfu,
    )

  def forward(self, x):
    x_l, x_g = x
    id_l, id_g = x_l, x_g
    x_l, x_g = self.conv1((x_l, x_g))
    x_l, x_g = self.conv2((x_l, x_g))
    return x_l + id_l, x_g + id_g


class ConcatTupleLayer(nn.Module):
  def forward(self, x):
    x_l, x_g = x
    if not torch.is_tensor(x_g):
      return x_l
    if not torch.is_tensor(x_l):
      return x_g
    return torch.cat((x_l, x_g), dim=1)


class FFCResNetGenerator(nn.Module):
  def __init__(
    self,
    input_nc=4,
    output_nc=3,
    ngf=64,
    n_downsampling=3,
    n_blocks=18,
    padding_type="reflect",
    max_features=1024,
    ratio_g=0.75,
    enable_lfu=False,
    add_out_act="sigmoid",
  ):
    super().__init__()
    if n_blocks < 0:
      raise ValueError("n_blocks must be non-negative")

    layers = [
      nn.ReflectionPad2d(3),
      FFC_BN_ACT(
        input_nc,
        ngf,
        kernel_size=7,
        padding=0,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        ratio_gin=0,
        ratio_gout=0,
        enable_lfu=enable_lfu,
      ),
    ]

    for i in range(n_downsampling):
      mult = 2**i
      ratio_gout = ratio_g if i == n_downsampling - 1 else 0
      layers.append(
        FFC_BN_ACT(
          min(max_features, ngf * mult),
          min(max_features, ngf * mult * 2),
          kernel_size=3,
          stride=2,
          padding=1,
          norm_layer=nn.BatchNorm2d,
          activation_layer=nn.ReLU,
          ratio_gin=0,
          ratio_gout=ratio_gout,
          enable_lfu=enable_lfu,
        )
      )

    mult = 2**n_downsampling
    bottleneck_channels = min(max_features, ngf * mult)
    for _ in range(n_blocks):
      layers.append(
        FFCResnetBlock(
          bottleneck_channels,
          padding_type=padding_type,
          norm_layer=nn.BatchNorm2d,
          activation_layer=nn.ReLU,
          ratio_gin=ratio_g,
          ratio_gout=ratio_g,
          enable_lfu=enable_lfu,
        )
      )

    layers.append(ConcatTupleLayer())

    for i in range(n_downsampling):
      mult = 2 ** (n_downsampling - i)
      layers.extend(
        (
          nn.ConvTranspose2d(
            min(max_features, ngf * mult),
            min(max_features, ngf * mult // 2),
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
          ),
          nn.BatchNorm2d(min(max_features, ngf * mult // 2)),
          nn.ReLU(inplace=True),
        )
      )

    layers.extend(
      (
        nn.ReflectionPad2d(3),
        nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
      )
    )
    if add_out_act == "sigmoid":
      layers.append(nn.Sigmoid())
    elif add_out_act == "tanh":
      layers.append(nn.Tanh())

    self.model = nn.Sequential(*layers)

  def forward(self, image, mask):
    """Predict an inpainted ``[0, 1]`` RGB image.

    Args:
      image: ``[B, 3, H, W]`` float tensor in ``[0, 1]`` (unmasked pixels intact).
      mask: ``[B, 1, H, W]`` float tensor where ``1`` marks pixels to inpaint.
    """
    masked = image * (1.0 - mask)
    predicted = self.model(torch.cat((masked, mask), dim=1))
    return predicted * mask + image * (1.0 - mask)


def infer_lama_config(state_dict):
  """Infer ``FFCResNetGenerator`` hyperparameters from a LaMa state dict."""
  block_indices = set()
  enable_lfu = False
  for key in state_dict:
    parts = key.split(".")
    if (
      len(parts) >= 3
      and parts[0] == "model"
      and parts[1].isdigit()
      and parts[2] in ("conv1", "conv2")
    ):
      block_indices.add(int(parts[1]))
    if ".lfu." in key:
      enable_lfu = True

  if not block_indices:
    raise ValueError("State dict does not look like an FFCResNetGenerator checkpoint")

  first_block = min(block_indices)
  n_downsampling = first_block - 2
  if n_downsampling < 1:
    raise ValueError(f"Unsupported LaMa downsampling count: {n_downsampling}")
  n_blocks = len(block_indices)

  init_weight = state_dict.get("model.1.ffc.convl2l.weight")
  if init_weight is None:
    raise ValueError("Missing model.1.ffc.convl2l.weight in LaMa state dict")
  ngf = int(init_weight.shape[0])
  input_nc = int(init_weight.shape[1])

  last_down_key = f"model.{first_block - 1}.ffc.convl2g.weight"
  last_down_l_key = f"model.{first_block - 1}.ffc.convl2l.weight"
  if last_down_key in state_dict and last_down_l_key in state_dict:
    global_ch = int(state_dict[last_down_key].shape[0])
    local_ch = int(state_dict[last_down_l_key].shape[0])
    ratio_g = global_ch / (global_ch + local_ch)
  elif last_down_key in state_dict:
    ratio_g = 1.0
  else:
    ratio_g = 0.0

  return {
    "input_nc": input_nc,
    "ngf": ngf,
    "n_downsampling": n_downsampling,
    "n_blocks": n_blocks,
    "ratio_g": ratio_g,
    "enable_lfu": enable_lfu,
  }
