"""Complex-valued CNN model for flattened SCM pair-vector inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ComplexConv2d(nn.Module):
    """Complex convolution implemented with two real Conv2d layers."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.imag = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.chunk(2, dim=1)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        return torch.cat([yr, yi], dim=1)


class ComplexBatchNorm2d(nn.Module):
    """Apply real BatchNorm independently to real and imaginary channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.real = nn.BatchNorm2d(channels)
        self.imag = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.chunk(2, dim=1)
        return torch.cat([self.real(xr), self.imag(xi)], dim=1)


class ComplexModReLU(nn.Module):
    """Magnitude-gated complex activation preserving phase."""

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
            ComplexConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            ComplexBatchNorm2d(out_channels),
            ComplexModReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class ModelConfig:
    input_pairs: int
    input_freq_bins: int
    base_channels: int = 16
    dropout: float = 0.15


class ScmComplexRangeCNN(nn.Module):
    """Complex CNN regressor for flattened SCM pair-vector features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.features = nn.Sequential(
            ComplexConvBlock(1, c, stride=(1, 2)),
            ComplexConvBlock(c, c * 2, stride=(2, 2)),
            ComplexConvBlock(c * 2, c * 4, stride=(2, 2)),
            ComplexConvBlock(c * 4, c * 4, stride=(1, 2)),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(c * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        zr, zi = z.chunk(2, dim=1)
        mag = torch.sqrt(zr.square() + zi.square() + 1.0e-8)
        return self.regressor(self.pool(mag))
