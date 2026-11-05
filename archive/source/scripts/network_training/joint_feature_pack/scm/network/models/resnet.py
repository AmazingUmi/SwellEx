"""Torchvision ResNet models adapted for flattened SCM pair-vector inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn
from torchvision.models import ResNet, resnet18, resnet50


@dataclass
class ModelConfig:
    input_pairs: int
    input_freq_bins: int
    dropout: float = 0.15


def _adapt_resnet_for_scm(
    model_factory: Callable[..., ResNet], config: ModelConfig
) -> ResNet:
    model = model_factory(weights=None)
    model.conv1 = nn.Conv2d(
        2,
        model.conv1.out_channels,
        kernel_size=3,
        stride=(1, 2),
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(config.dropout),
        nn.Linear(in_features, 1),
    )
    return model


class ScmResNet18Range(nn.Module):
    """ResNet-18 regressor for normalized range targets."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model = _adapt_resnet_for_scm(resnet18, config)

    def forward(self, x):
        return self.model(x)


class ScmResNet50Range(nn.Module):
    """ResNet-50 regressor for normalized range targets."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model = _adapt_resnet_for_scm(resnet50, config)

    def forward(self, x):
        return self.model(x)
