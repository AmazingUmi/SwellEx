"""Explicit experiment-pack registry; safe to import without numerical packages."""
from __future__ import annotations

from dataclasses import dataclass


NETWORKS = ("complex_CNN", "real_CNN", "ResNet18", "ResNet50", "GRNN")
SCALE_NETWORKS = ("complex_CNN", "complex_CNN_small", "complex_CNN_medium",
                  "complex_CNN_large", "complex_CNN_xlarge", "complex_CNN_baseline")
MOE_NETWORKS = ("complex_MoE", "complex_MoE_e1", "complex_MoE_e2", "complex_MoE_e4", "complex_MoE_e8")
MOE_LEARN_NETWORKS = ("complex_MoE_learn",)
ALL_NETWORKS = tuple(dict.fromkeys((*NETWORKS, *SCALE_NETWORKS, *MOE_NETWORKS, *MOE_LEARN_NETWORKS)))


@dataclass(frozen=True)
class Pack:
    name: str
    methods: tuple[str, ...]
    fusion_modes: tuple[str, ...] = ()
    local_grnn: bool = False
    networks: tuple[str, ...] = NETWORKS
    tag: str = ""

    @property
    def dispatcher(self) -> str:
        return f"scripts.{self.name}.localization_networks.workflow.remote"


PACKS = {
    "base_pack": Pack("base_pack", ("RBD", "ELM", "SCM"), local_grnn=True, tag="base"),
    "joint_feature_pack": Pack("joint_feature_pack", (
        "RBD", "ELM", "SCM", "RBD_ELM", "RBD_SCM", "ELM_SCM", "RBD_ELM_SCM",
    ), fusion_modes=("raw_concat", "block_rms"), tag="joint"),
    "data_scale_pack": Pack("data_scale_pack", ("RBD", "ELM", "SCM"),
                            networks=SCALE_NETWORKS, tag="scale"),
    "MoE_data_scale": Pack("MoE_data_scale", ("RBD", "ELM", "SCM", "RBD_SCM"),
                           networks=MOE_NETWORKS, tag="moe"),
    "MoE_learn": Pack("MoE_learn", ("RBD_SCM",), networks=MOE_LEARN_NETWORKS,
                      tag="moelearn"),
}


def get_pack(name: str) -> Pack:
    if name not in PACKS:
        raise ValueError(f"PACK must be one of {', '.join(PACKS)}")
    return PACKS[name]
