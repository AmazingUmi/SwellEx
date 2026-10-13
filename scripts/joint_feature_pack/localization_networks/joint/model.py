"""Existing neural architectures with a truthful feature-count configuration."""
from dataclasses import asdict, dataclass

from scripts.joint_feature_pack.localization_networks.rbd.network.models.complex_cnn import ComplexRangeCNN
from scripts.joint_feature_pack.localization_networks.rbd.network.models.real_cnn import RealRangeCNN
from scripts.joint_feature_pack.localization_networks.rbd.network.models.resnet import ResNet18Range, ResNet50Range


@dataclass
class ModelConfig:
    input_features: int
    input_freq_bins: int
    base_channels: int = 16
    dropout: float = 0.15


MODEL_REGISTRY = {"joint_complex_cnn_range": (ModelConfig, ComplexRangeCNN),
                  "joint_real_cnn_range": (ModelConfig, RealRangeCNN),
                  "joint_resnet18_range": (ModelConfig, ResNet18Range),
                  "joint_resnet50_range": (ModelConfig, ResNet50Range)}
MODEL_DATASET_INPUT_NORM = {name: name != "joint_real_cnn_range" for name in MODEL_REGISTRY}


def build_model_config(model_name, input_shape, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown joint model: {model_name}")
    return ModelConfig(input_features=input_shape[1], input_freq_bins=input_shape[2],
                       **{key: value for key, value in kwargs.items()
                          if key in {"base_channels", "dropout"} and value is not None})


def build_model(model_name, config):
    return MODEL_REGISTRY[model_name][1](config)


def model_uses_dataset_input_norm(model_name):
    return MODEL_DATASET_INPUT_NORM[model_name]


def model_config_from_checkpoint(checkpoint):
    name = checkpoint["model_name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown joint checkpoint model: {name}")
    return name, ModelConfig(**checkpoint["model_config"])


def serialize_model_config(config):
    return asdict(config)
