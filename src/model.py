import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
if not getattr(sys, "frozen", False):
    ROOT = ROOT.parent
MODELS_DIR = ROOT / "models"
WEIGHTS_NAME = "inpaint.pt"


class InpaintModel(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (6, 32, 64, 64, 128, 128, 256, 256, 256, 256, 256)
        strides = (1, 2, 1, 2, 1, 2, 1, 1, 1, 1)
        dilations = (1, 1, 1, 1, 1, 1, 2, 4, 8, 16)
        self.encoder = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], 5 if i == 0 else 3,
                      stride=stride, dilation=dilation)
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

        distances = (keys.square().sum(dim=2, keepdim=True)
                     + queries.square().sum(dim=1, keepdim=True)
                     - 2 * torch.bmm(keys, queries))
        centered = distances - distances.mean(dim=1, keepdim=True)
        deviation = centered.square().mean(dim=1, keepdim=True).sqrt()
        attention = torch.softmax(-50 * torch.tanh(centered / deviation), dim=1)

        patches = torch.bmm(keys.transpose(1, 2), attention)
        copied = F.fold(patches, size, kernel_size=patch_size, padding=padding) / patch_size**2
        filled = background + copied * (1 - mask)
        return F.elu(self.context_merge(torch.cat((features, filled), dim=1)))

    def forward(self, image, original, mask):
        features = torch.cat((image, mask), dim=1)
        for convolution in self.encoder:
            border = (convolution.kernel_size[0] // 2) * convolution.dilation[0]
            features = F.elu(convolution(F.pad(features, (border,) * 4, mode="reflect")))

        features = self.contextual_block(features, mask)
        for i, convolution in enumerate(self.decoder[:-1]):
            features = F.elu(convolution(F.pad(features, (1, 1, 1, 1), mode="reflect")))
            if i in (1, 3, 5):
                features = F.interpolate(features, scale_factor=2, mode="nearest")

        filled = self.decoder[-1](features).clamp(-1, 1)
        return filled * (1 - mask) + original * mask


class InpaintNN:
    def __init__(self, is_mosaic=False):
        mode = "mosaic" if is_mosaic else "bar"
        path = MODELS_DIR / mode / WEIGHTS_NAME
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run the model conversion in docs/INSTALLATION.md.")

        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model = InpaintModel().eval()
        self.model.load_state_dict(state)
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
        return np.concatenate([
            self._predict(image[i:i + 1], original[i:i + 1], mask[i:i + 1])
            for i in range(len(image))
        ])

    def _predict(self, image, original, mask):
        inputs = [
            torch.as_tensor(array, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
            for array in (image, original, mask)
        ]
        return self.model(*inputs).permute(0, 2, 3, 1).cpu().numpy()
