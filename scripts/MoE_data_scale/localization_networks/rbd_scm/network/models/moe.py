"""Method-specific physical-input contract for residual complex MoE."""

from dataclasses import dataclass

from scripts.MoE_data_scale.localization_networks.common.scaling import (
    MoEModelConfig, validate_model_config,
)
from scripts.MoE_data_scale.localization_networks.common.moe import ComplexResidualMoE


@dataclass(frozen=True)
class ModelConfig(MoEModelConfig):
    input_features: int
    input_freq_bins: int

    def __post_init__(self) -> None:
        validate_model_config(self, "input_features")


class RbdScmComplexRangeMoE(ComplexResidualMoE):
    """Predict normalized range from RBD_SCM using frozen independent experts."""

    def __init__(self, config: ModelConfig) -> None:
        config.__post_init__()
        super().__init__(config)
