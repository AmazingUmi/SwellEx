"""Complex residual experts with separate training and deployment states."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    input_features: int
    input_freq_bins: int
    base_channels: int = 16
    num_experts: int = 2
    dropout: float = 0.15
    architecture: str = "continual-complex-residual-moe-v1"

    def __post_init__(self):
        for name in ("input_features", "input_freq_bins", "base_channels", "num_experts"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.base_channels < 2 or self.num_experts > 8:
            raise ValueError("base_channels must be >=2 and num_experts <=8")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (float, int)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite in [0,1)")
        if self.architecture != "continual-complex-residual-moe-v1":
            raise ValueError("Unsupported continual MoE architecture")


def _groups(channels):
    return max(g for g in range(1, min(8, channels // 2) + 1) if channels % g == 0)


class ComplexConv(nn.Module):
    def __init__(self, channels_in, channels_out, stride=(1, 1)):
        super().__init__()
        self.real = nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False)
        self.imag = nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        xr, xi = x.chunk(2, dim=1)
        return torch.cat((self.real(xr) - self.imag(xi), self.real(xi) + self.imag(xr)), dim=1)


class ComplexNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.real = nn.GroupNorm(_groups(channels), channels)
        self.imag = nn.GroupNorm(_groups(channels), channels)

    def forward(self, x):
        xr, xi = x.chunk(2, dim=1)
        return torch.cat((self.real(xr), self.imag(xi)), dim=1)


class ModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        xr, xi = x.chunk(2, dim=1)
        amplitude = torch.sqrt(xr.square() + xi.square() + 1e-8)
        scale = torch.relu(amplitude + self.bias) / (amplitude + 1e-8)
        return torch.cat((xr * scale, xi * scale), dim=1)


def conv_block(channels_in, channels_out, stride):
    return nn.Sequential(ComplexConv(channels_in, channels_out, stride), ComplexNorm(channels_out), ModReLU(channels_out))


def magnitude(x):
    xr, xi = x.chunk(2, dim=1)
    return torch.sqrt(xr.square() + xi.square() + 1e-8)


class ResidualExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.base_channels
        self.projection = conv_block(2 * c, 4 * c, (2, 1))
        self.residual = nn.Sequential(ComplexConv(4 * c, 4 * c), ComplexNorm(4 * c), ModReLU(4 * c),
                                      ComplexConv(4 * c, 4 * c), ComplexNorm(4 * c))
        self.head = nn.Sequential(nn.Linear(4 * c, 4 * c), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(4 * c, 1))

    def forward(self, shared):
        z = self.projection(shared)
        z = z + self.residual(z)
        return self.head(magnitude(z).mean(dim=(-2, -1)))


class ContinualMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config.base_channels
        self.stem = nn.Sequential(conv_block(1, c, (1, 2)), conv_block(c, 2 * c, (2, 2)))
        self.experts = nn.ModuleList(ResidualExpert(config) for _ in range(config.num_experts))
        # Always retain router rows, including E1, to preserve stable expert IDs.
        self.router = nn.Linear(2 * c, config.num_experts)
        self.frozen_experts: set[int] = set()
        self.stem_frozen = False

    @property
    def num_experts(self):
        return len(self.experts)

    def _validate_input(self, x):
        expected = (2, self.config.input_features, self.config.input_freq_bins)
        if x.ndim != 4 or x.shape[0] <= 0 or tuple(x.shape[1:]) != expected or not x.is_floating_point():
            raise ValueError(f"Expected finite floating [batch,{expected}] input")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Nonfinite model input")

    def train(self, mode=True):
        super().train(mode)
        if self.stem_frozen:
            self.stem.eval()
        for index in self.frozen_experts:
            self.experts[index].eval()
        return self

    def configure_training(self, trainable_experts, *, train_router=True):
        """Keep the shared representation fixed; explicitly reopen expert gradients."""
        trainable = set(trainable_experts)
        if any(type(i) is not int or not 0 <= i < self.num_experts for i in trainable):
            raise ValueError("Invalid trainable expert IDs")
        self.stem_frozen = True
        self.stem.requires_grad_(False)
        for parameter in self.stem.parameters():
            parameter.grad = None
        self.frozen_experts = set(range(self.num_experts)) - trainable
        for index, expert in enumerate(self.experts):
            expert.requires_grad_(index in trainable)
            if index not in trainable:
                for parameter in expert.parameters():
                    parameter.grad = None
        self.router.requires_grad_(train_router)
        if not train_router:
            for parameter in self.router.parameters():
                parameter.grad = None
        self.train(self.training)

    def freeze_for_deployment(self):
        """Inference state only; later candidate fitting reopens expert gradients."""
        self.configure_training((), train_router=False)
        self.eval()

    def add_expert(self, maximum):
        if type(maximum) is not int or not 1 <= maximum <= 8 or self.num_experts >= maximum:
            raise ValueError("Expert budget exhausted or invalid")
        old_count = self.num_experts
        values = asdict(self.config)
        values["num_experts"] = old_count + 1
        self.config = ModelConfig(**values)
        parameter = next(self.parameters())
        self.experts.append(ResidualExpert(self.config).to(device=parameter.device, dtype=parameter.dtype))
        expanded = nn.Linear(2 * self.config.base_channels, old_count + 1).to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            expanded.weight[:old_count].copy_(self.router.weight)
            expanded.bias[:old_count].copy_(self.router.bias)
            # New row starts near old logit scale; independent warmup precedes routing.
            expanded.bias[old_count].copy_(self.router.bias.mean())
        self.router = expanded
        self.configure_training((old_count,), train_router=False)
        return old_count

    def descriptor(self, x):
        self._validate_input(x)
        amplitude = magnitude(self.stem(x))
        return torch.cat((amplitude.mean(dim=(-2, -1)), amplitude.std(dim=(-2, -1), correction=0)), dim=1)

    def expert_prediction(self, x, expert_id):
        self._validate_input(x)
        return self.experts[expert_id](self.stem(x))

    def forward_with_aux(self, x):
        self._validate_input(x)
        shared = self.stem(x)
        logits = self.router(magnitude(shared).mean(dim=(-2, -1)))
        probabilities = torch.softmax(logits, dim=1)
        k = min(2, self.num_experts)
        indices = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :k]
        weights = torch.softmax(logits.gather(1, indices), dim=1)
        prediction = shared.new_zeros((len(x), 1))
        for index, expert in enumerate(self.experts):
            positions = (indices == index).nonzero(as_tuple=False)
            if not positions.numel():
                continue
            rows, slots = positions.unbind(dim=1)
            values = expert(shared.index_select(0, rows)) * weights[rows, slots, None]
            prediction = prediction.index_add(0, rows, values)
        load = torch.bincount(indices.flatten(), minlength=self.num_experts)
        mean_probability = probabilities.mean(dim=0)
        balance = self.num_experts * (mean_probability - 1 / self.num_experts).square().sum()
        balance = balance + self.num_experts * (load.to(x.dtype).detach() / (len(x) * k) * mean_probability).sum() - 1
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(x.dtype).tiny).log()).sum(dim=1).mean()
        return prediction, {"balance_loss": balance, "expert_load": load,
                            "router_entropy": entropy, "selected_experts": indices,
                            "routing_weights": weights, "router_probabilities": probabilities}

    def forward(self, x):
        return self.forward_with_aux(x)[0]
