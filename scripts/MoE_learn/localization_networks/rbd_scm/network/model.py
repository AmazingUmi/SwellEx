"""Canonical model registry for the staged RBD_SCM demo."""
from .models.moe import ContinualMoE, ModelConfig

MODEL_REGISTRY = {"rbd_scm_complex_moe_learn_range": (ModelConfig, ContinualMoE)}
MODEL_DATASET_INPUT_NORM = {"rbd_scm_complex_moe_learn_range": True}


def build_model_config(model_name: str, input_shape, **kwargs) -> ModelConfig:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown MoE_learn model: {model_name}")
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3 or input_shape[0] != 2:
        raise ValueError("RBD_SCM input must be [2,feature,frequency]")
    return ModelConfig(input_features=input_shape[1], input_freq_bins=input_shape[2], **kwargs)


def build_model(model_name: str, config: ModelConfig) -> ContinualMoE:
    if model_name not in MODEL_REGISTRY or not isinstance(config, ModelConfig):
        raise ValueError("MoE_learn requires its canonical model and configuration")
    config.__post_init__()
    return MODEL_REGISTRY[model_name][1](config)
