"""HistoChrome: virtual multiplex IHC from H&E (HIMICO; github.com/measty/HistoChromeCode, MIT).

Inference-only port of the HistoChromeCode `he2mihc` model. Module and parameter names mirror the original so its
checkpoint loads unchanged:

    StainPredictor
      base_model:   HEStainSeparation   RGB -> optical density -> 2 H&E stain channels (learned stain matrix)
        base_model: GlobalGenerator     pix2pixHD generator, 2 -> 5 mIHC stain concentrations (softplus)
      stain_decoder: Decoder             StainQuant basis (5 x 3): RGB = exp(-sum_k C_k * basis_k)

Tiles are 832 px at 0.5 um/px, RGB scaled to [0, 1]; the output is the reconstructed mIHC RGB image in [0, 1].
Weights are not public yet: they load from a local weights_dir (histochrome/histochrome_he2mihc.safetensors) or from
the private pkeller01/HistoChrome repo for accounts with access. Inference runs locally; no API key is involved.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .. import weights
from ..registry import register_model
from .base import VirtualStainer

WEIGHTS_KEY = "histochrome/histochrome_he2mihc.safetensors"


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, activation: nn.Module):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), activation,
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GlobalGenerator(nn.Module):
    """pix2pixHD global generator: 3 stride-2 downsamplings, 9 ResNet blocks, instance norm, reflect padding."""

    def __init__(self, input_nc: int, output_nc: int, ngf: int = 64, n_down: int = 3, n_blocks: int = 9):
        super().__init__()
        act = nn.ReLU(True)
        layers: list[nn.Module] = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7), nn.InstanceNorm2d(ngf), act]
        for i in range(n_down):
            m = 2 ** i
            layers += [nn.Conv2d(ngf * m, ngf * m * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf * m * 2), act]
        layers += [ResnetBlock(ngf * 2 ** n_down, act) for _ in range(n_blocks)]
        for i in range(n_down):
            m = 2 ** (n_down - i)
            layers += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(ngf * m // 2), act]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Softplus()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class HEStainSeparation(nn.Module):
    """Optical density through the pseudo-inverse of a learned (stains x RGB) matrix."""

    def __init__(self, base_model: nn.Module, n_stains: int = 2):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("stain_matrix_init", torch.zeros(n_stains, 3))
        self.stain_matrix = nn.Parameter(torch.zeros(n_stains, 3))

    def forward(self, x):
        od = -torch.log(torch.clamp(x.float(), min=1e-4))
        # 2 x 3 pseudo-inverse on the CPU (not implemented on every accelerator); same result everywhere
        pinv = torch.linalg.pinv(self.stain_matrix.detach().float().cpu()).to(od.device)
        stains = torch.einsum("bchw,cs->bshw", od, pinv)
        return self.base_model(stains)


class _Basis(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self._basis = nn.Parameter(torch.zeros(k, 3))

    @property
    def basis(self):
        return F.normalize(self._basis, dim=1)


class Decoder(nn.Module):
    """StainQuant decoder: mix stain concentrations with the colour basis (Beer-Lambert) and convert to RGB."""

    def __init__(self, k: int = 5):
        super().__init__()
        self.basis_module = _Basis(k)

    def forward(self, c):
        b = self.basis_module.basis.view(1, -1, 3, 1, 1)
        return torch.exp(-(c.unsqueeze(2) * b).sum(1))


class StainPredictor(nn.Module):
    def __init__(self, n_markers: int = 5):
        super().__init__()
        self.base_model = HEStainSeparation(GlobalGenerator(2, n_markers))
        self.stain_decoder = Decoder(n_markers)

    def forward(self, x):
        """x: [B, 3, H, W] RGB in [0, 1] -> (stain concentrations [B, 5, H, W], mIHC RGB [B, 3, H, W])"""
        stains = self.base_model(x)
        return stains, self.stain_decoder(stains)


@register_model("histochrome")
class HistoChrome(VirtualStainer):
    description = "Virtual multiplex IHC from H&E (HIMICO; 5-marker panel rendered as RGB)"
    paper = "https://github.com/measty/HistoChromeCode"
    weights = (WEIGHTS_KEY,)
    tile = 832
    mpp = 0.5
    overlap = 96
    min_tissue = 0.2

    def __init__(self, device: torch.device, weights_dir: str | None = None):
        super().__init__(device, weights_dir)
        from safetensors.torch import load_file

        net = StainPredictor()
        missing, unexpected = net.load_state_dict(load_file(weights.resolve(WEIGHTS_KEY, weights_dir)), strict=False)
        # init_rgb is a training-only buffer of the StainQuant basis; nothing else may be missing
        if unexpected or missing:
            raise RuntimeError(f"HistoChrome weights do not match the network: missing {missing}, unexpected {unexpected}")
        self.net = net.to(device).eval()

    @torch.no_grad()
    def __call__(self, tiles: torch.Tensor) -> torch.Tensor:
        """tiles: [B, H, W, 3] uint8 -> mIHC RGB [B, 3, H, W] float in [0, 1]"""
        x = tiles.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return self.net(x)[1].clamp(0, 1)
