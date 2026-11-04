"""Method-specific input contract for the complex CNN capacity family."""

from dataclasses import dataclass

from scripts.network_training.data_scale_pack.common.scaling import (
    ScaleModelConfig,
    validate_model_config,
)
from scripts.network_training.data_scale_pack.common.scalable_cnn import ScalableComplexCNN


@dataclass(frozen=True)
class ModelConfig(ScaleModelConfig):
    input_elements: int
    input_freq_bins: int

    def __post_init__(self) -> None:
        validate_model_config(self, "input_elements")


class ComplexRangeCNN(ScalableComplexCNN):
    """Predict normalized range from RBD features using the saved layout."""

    def __init__(self, config: ModelConfig) -> None:
        config.__post_init__()
        super().__init__(config)
