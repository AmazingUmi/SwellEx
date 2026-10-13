"""Real-valued CNN model for flattened ELM pairwise-ratio inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class ModelConfig:
    input_pairs: int
    input_freq_bins: int
    base_channels: int = 16
    dropout: float = 0.15


class ElmRealRangeCNN(nn.Module):
    """Real CNN regressor fed by normalized ELM complex magnitude."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.features = nn.Sequential(
            ConvBlock(1, c, stride=(1, 2)),
            ConvBlock(c, c * 2, stride=(2, 2)),
            ConvBlock(c * 2, c * 4, stride=(2, 2)),
            ConvBlock(c * 4, c * 4, stride=(1, 2)),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(c * 4, 1),
        )

    @staticmethod
    def magnitude_normalize(x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0:1]
        imag = x[:, 1:2]
        mag = torch.sqrt(real.square() + imag.square() + 1.0e-8)
        mean = mag.mean(dim=(2, 3), keepdim=True)
        std = mag.std(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
        return (mag - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = self.magnitude_normalize(x)
        z = self.features(mag)
        return self.regressor(self.pool(z))
