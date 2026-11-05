"""Torchvision ResNet regressors shared by the RBD (element) and SCM (pair) inputs."""

from __future__ import annotations

from typing import Callable

from torch import nn
from torchvision.models import ResNet, resnet18, resnet50


def _adapt_resnet(model_factory: Callable[..., ResNet], config) -> ResNet:
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


class ResNet18Range(nn.Module):
    """ResNet-18 regressor for normalized range targets."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = _adapt_resnet(resnet18, config)

    def forward(self, x):
        return self.model(x)


class ResNet50Range(nn.Module):
    """ResNet-50 regressor for normalized range targets."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = _adapt_resnet(resnet50, config)

    def forward(self, x):
        return self.model(x)
