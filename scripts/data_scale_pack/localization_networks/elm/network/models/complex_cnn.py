"""Method-specific input contract for the complex CNN capacity family."""

from dataclasses import dataclass

from scripts.data_scale_pack.localization_networks.common.scaling import (
    ScaleModelConfig,
    validate_model_config,
)
from scripts.data_scale_pack.localization_networks.common.scalable_cnn import ScalableComplexCNN


@dataclass(frozen=True)
class ModelConfig(ScaleModelConfig):
    input_pairs: int
    input_freq_bins: int

    def __post_init__(self) -> None:
        validate_model_config(self, "input_pairs")


class ElmComplexRangeCNN(ScalableComplexCNN):
    """Predict normalized range from ELM features using the saved layout."""

    def __init__(self, config: ModelConfig) -> None:
        config.__post_init__()
        super().__init__(config)
