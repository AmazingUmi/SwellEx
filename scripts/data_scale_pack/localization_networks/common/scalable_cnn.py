"""Complex CNN primitives and the shared depth-scaled model implementation."""

from __future__ import annotations

import torch
from torch import nn

from .scaling import ScaleModelConfig, stage_channels


class ComplexConv2d(nn.Module):
    """Complex convolution using the existing two-real-kernel implementation."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.imag = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.chunk(2, dim=1)
        return torch.cat([self.real(xr) - self.imag(xi), self.real(xi) + self.imag(xr)], dim=1)


class ComplexBatchNorm2d(nn.Module):
    """Separate real/imaginary normalization, not full covariance whitening."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.real = nn.BatchNorm2d(channels)
        self.imag = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.chunk(2, dim=1)
        return torch.cat([self.real(xr), self.imag(xi)], dim=1)


class ComplexModReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.chunk(2, dim=1)
        mag = torch.sqrt(xr.square() + xi.square() + 1.0e-8)
        scale = torch.relu(mag + self.bias) / (mag + 1.0e-8)
        return torch.cat([xr * scale, xi * scale], dim=1)


class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            ComplexBatchNorm2d(out_channels),
            ComplexModReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScalableComplexCNN(nn.Module):
    """Instantiate the saved layout; extra blocks never add downsampling."""

    def __init__(self, config: ScaleModelConfig) -> None:
        super().__init__()
        blocks = []
        in_channels = 1
        for channels, repeats, stride in zip(
            stage_channels(config.base_channels, config.selected_profile),
            config.stage_repeats,
            config.stage_strides,
            strict=True,
        ):
            for index in range(repeats):
                blocks.append(ComplexConvBlock(in_channels, channels, stride if index == 0 else (1, 1)))
                in_channels = channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        zr, zi = z.chunk(2, dim=1)
        mag = torch.sqrt(zr.square() + zi.square() + 1.0e-8)
        return self.regressor(self.pool(mag))
