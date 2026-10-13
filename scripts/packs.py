"""Explicit experiment-pack registry; safe to import without numerical packages."""
from __future__ import annotations

from dataclasses import dataclass


NETWORKS = ("complex_CNN", "real_CNN", "ResNet18", "ResNet50", "GRNN")


@dataclass(frozen=True)
class Pack:
    name: str
    methods: tuple[str, ...]
    fusion_modes: tuple[str, ...] = ()
    local_grnn: bool = False

    @property
    def dispatcher(self) -> str:
        return f"scripts.{self.name}.localization_networks.workflow.remote"


PACKS = {
    "base_pack": Pack("base_pack", ("RBD", "ELM", "SCM"), local_grnn=True),
    "joint_feature_pack": Pack("joint_feature_pack", (
        "RBD", "ELM", "SCM", "RBD_ELM", "RBD_SCM", "ELM_SCM", "RBD_ELM_SCM",
    ), fusion_modes=("raw_concat", "block_rms")),
}


def get_pack(name: str) -> Pack:
    if name not in PACKS:
        raise ValueError(f"PACK must be one of {', '.join(PACKS)}")
    return PACKS[name]
