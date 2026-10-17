"""Sample-routed complex residual experts with no dropped observations."""

from __future__ import annotations

import torch
from torch import nn

from .scaling import MoEModelConfig, normalization_groups


class ComplexConv2d(nn.Module):
    """Two bias-free real kernels implement a complex convolution."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.imag = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.chunk(2, dim=1)
        return torch.cat((self.real(xr) - self.imag(xi), self.real(xi) + self.imag(xr)), dim=1)


class ComplexGroupNorm(nn.Module):
    """Separate real/imaginary GroupNorm; no cross-sample running state."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = normalization_groups(channels)
        self.real = nn.GroupNorm(groups, channels)
        self.imag = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.chunk(2, dim=1)
        return torch.cat((self.real(xr), self.imag(xi)), dim=1)


class ComplexModReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.chunk(2, dim=1)
        magnitude = torch.sqrt(xr.square() + xi.square() + 1.0e-8)
        scale = torch.relu(magnitude + self.bias) / (magnitude + 1.0e-8)
        return torch.cat((xr * scale, xi * scale), dim=1)


class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            ComplexGroupNorm(out_channels), ComplexModReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ComplexResidualBlock(nn.Module):
    """Identity shortcut at constant width/resolution: exactly x + F(x)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            ComplexConv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            ComplexGroupNorm(channels), ComplexModReLU(channels),
            ComplexConv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            ComplexGroupNorm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)


def _magnitude(x: torch.Tensor) -> torch.Tensor:
    xr, xi = x.chunk(2, dim=1)
    return torch.sqrt(xr.square() + xi.square() + 1.0e-8)


class ResidualExpert(nn.Module):
    def __init__(self, config: MoEModelConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.projection = ComplexConvBlock(2 * c, 4 * c, config.expert_projection_stride)
        self.residual = ComplexResidualBlock(4 * c)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(4 * c, 4 * c), nn.ReLU(),
            nn.Dropout(config.dropout), nn.Linear(4 * c, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.residual(self.projection(x))
        return self.regressor(self.pool(_magnitude(z)))


class ComplexResidualMoE(nn.Module):
    """Shared fixed stem; only selected expert tails execute for each sample.

    active_parameter_count records a per-sample footprint; a whole batch can
    activate all experts. All experts are separately initialized modules.
    """

    def __init__(self, config: MoEModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        c = config.base_channels
        self.stem = nn.Sequential(
            ComplexConvBlock(1, c, config.stem_strides[0]),
            ComplexConvBlock(c, 2 * c, config.stem_strides[1]),
        )
        self.experts = nn.ModuleList(ResidualExpert(config) for _ in range(self.num_experts))
        self.router = nn.Linear(2 * c, self.num_experts) if self.num_experts > 1 else None
        self.input_rows = getattr(config, "input_elements", getattr(config, "input_pairs", getattr(config, "input_features", None)))

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("MoE input must be a real tensor [batch,2,feature_rows,frequency_bins]")
        if x.shape[0] <= 0 or x.shape[1:] != (2, self.input_rows, self.config.input_freq_bins):
            raise ValueError("MoE input shape must match its complete saved input contract")
        if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
            raise ValueError("MoE input must contain finite real floating-point values")

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_input(x)
        shared = self.stem(x)
        batch = x.shape[0]
        if self.router is None:
            prediction = self.experts[0](shared)
            probabilities = shared.new_ones((batch, 1))
            indices = torch.zeros((batch, 1), dtype=torch.long, device=x.device)
            weights = probabilities
            load = torch.full((1,), batch, dtype=torch.long, device=x.device)
            importance = weights.sum(dim=0)
            balance = prediction.sum() * 0.0
            entropy = balance
        else:
            router_input = _magnitude(shared).mean(dim=(-2, -1))
            logits = self.router(router_input)
            probabilities = torch.softmax(logits, dim=1)
            # Stable tie handling makes inspection and evaluation reproducible.
            indices = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :self.top_k]
            weights = torch.softmax(logits.gather(1, indices), dim=1)
            prediction = shared.new_zeros((batch, 1))
            for expert_index, expert in enumerate(self.experts):
                positions = (indices == expert_index).nonzero(as_tuple=False)
                if positions.numel() == 0:
                    continue
                rows, slots = positions.unbind(dim=1)
                values = expert(shared.index_select(0, rows)) * weights[rows, slots].unsqueeze(1)
                prediction = prediction.index_add(0, rows, values)
            load = torch.bincount(indices.reshape(-1), minlength=self.num_experts)
            importance = shared.new_zeros(self.num_experts).scatter_add(0, indices.reshape(-1), weights.reshape(-1))
            mean_probability = probabilities.mean(dim=0)
            uniform = 1.0 / self.num_experts
            # Importance term remains trainable for E=2, when dispatch loads are
            # necessarily equal. Detached load correlation adds sparse pressure.
            importance_penalty = self.num_experts * (mean_probability - uniform).square().sum()
            load_fraction = load.to(probabilities.dtype).detach() / (batch * self.top_k)
            load_penalty = self.num_experts * (load_fraction * mean_probability).sum() - 1.0
            balance = importance_penalty + load_penalty
            entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=1).mean()
        return prediction, {
            "balance_loss": balance, "expert_load": load, "expert_importance": importance,
            "router_entropy": entropy, "router_probabilities": probabilities,
            "selected_experts": indices, "routing_weights": weights,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)[0]
