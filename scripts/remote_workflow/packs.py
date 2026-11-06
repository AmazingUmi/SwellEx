"""Explicit experiment-pack registry; safe to import without numerical packages.

Only ``base_pack`` (RBD/SCM features; complex CNN, ResNet and GRNN networks) is
active. ELM, joint-feature, data-scale and MoE experiments live under
``archive/`` and are not dispatchable.
"""
from __future__ import annotations

from dataclasses import dataclass

from scripts.data_processing.input_contract import METHODS
from scripts.network_training.catalog import ARCHITECTURES


DEFAULT_PACK = "base_pack"
NETWORKS = ARCHITECTURES


@dataclass(frozen=True)
class Pack:
    name: str
    methods: tuple[str, ...]
    local_grnn: bool = False
    networks: tuple[str, ...] = NETWORKS
    tag: str = ""

    @property
    def dispatcher(self) -> str:
        return "scripts.remote_workflow.remote"


PACKS = {
    DEFAULT_PACK: Pack(DEFAULT_PACK, METHODS, local_grnn=True, tag="base"),
}


def get_pack(name: str) -> Pack:
    if name not in PACKS:
        raise ValueError(f"PACK must be one of {', '.join(PACKS)}")
    return PACKS[name]
