"""Model configuration dataclasses; field names are part of the checkpoint contract."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RbdCnnConfig:
    input_elements: int
    input_freq_bins: int
    base_channels: int = 16
    dropout: float = 0.15


@dataclass
class ScmCnnConfig:
    input_pairs: int
    input_freq_bins: int
    base_channels: int = 16
    dropout: float = 0.15


@dataclass
class RbdResNetConfig:
    input_elements: int
    input_freq_bins: int
    dropout: float = 0.15


@dataclass
class ScmResNetConfig:
    input_pairs: int
    input_freq_bins: int
    dropout: float = 0.15
