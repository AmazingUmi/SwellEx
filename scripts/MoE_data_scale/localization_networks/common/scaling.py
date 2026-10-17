"""Strict, tensor-free resolution of an uncalibrated expert-count policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import json
from typing import Any


POLICY_VERSION = "train-count-moe-v1"
POLICY_STATUS = "uncalibrated"
ARCHITECTURE_VERSION = "complex-residual-moe-v1"
ALLOWED_EXPERTS = (1, 2, 4, 8)
DEFAULT_SCALING_POLICY = {
    "reference_samples": 600, "reference_experts": 2, "exponent": 0.5, "max_experts": 8,
}
STEM_STRIDES = ((1, 2), (2, 2))
EXPERT_PROJECTION_STRIDE = (2, 1)
NORMALIZATION = "separate-real-imag-groupnorm"
ROUTING_POLICY = "sample-top2-selected-softmax-no-drop"
BALANCE_POLICY = "full-softmax-importance-plus-detached-load-v1"


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _base_channels(value: Any) -> int:
    value = positive_int(value, "base_channels")
    if value < 2:
        raise ValueError("base_channels must be at least 2 for singleton-safe GroupNorm")
    return value


def _experts(value: Any, name: str = "num_experts") -> int:
    if type(value) is not int or value not in ALLOWED_EXPERTS:
        raise ValueError(f"{name} must be one of {ALLOWED_EXPERTS}")
    return value


def validate_scaling_policy(policy: Any) -> dict[str, int | float]:
    if not isinstance(policy, dict) or set(policy) != set(DEFAULT_SCALING_POLICY):
        raise ValueError("scaling_policy requires exactly reference_samples, reference_experts, exponent, max_experts")
    reference = positive_int(policy["reference_samples"], "reference_samples")
    reference_experts = _experts(policy["reference_experts"], "reference_experts")
    maximum = _experts(policy["max_experts"], "max_experts")
    if reference_experts > maximum:
        raise ValueError("reference_experts cannot exceed max_experts")
    exponent = policy["exponent"]
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        raise ValueError("exponent must be a finite positive number")
    try:
        exponent = float(exponent)
    except (OverflowError, ValueError) as exc:
        raise ValueError("exponent must be a finite positive number") from exc
    if not math.isfinite(exponent) or exponent <= 0:
        raise ValueError("exponent must be a finite positive number")
    return {"reference_samples": reference, "reference_experts": reference_experts,
            "exponent": exponent, "max_experts": maximum}


def parameter_count(num_experts: int, base_channels: int) -> int:
    """All kernels, GroupNorm affine terms, modReLU biases, heads and router."""
    e, c = _experts(num_experts), _base_channels(base_channels)
    shared = 36 * c * c + 33 * c
    expert = 736 * c * c + 64 * c + 1
    router = e * (2 * c + 1) if e > 1 else 0
    return shared + e * expert + router


def active_parameter_count(num_experts: int, base_channels: int) -> int:
    """Per-sample parameter footprint, including shared stem and full router."""
    e, c = _experts(num_experts), _base_channels(base_channels)
    return 36 * c * c + 33 * c + min(2, e) * (736 * c * c + 64 * c + 1) + (e * (2 * c + 1) if e > 1 else 0)


def normalization_groups(channels: int) -> int:
    """At least two channels per group, even for one routed 1x1 sample."""
    positive_int(channels, "channels")
    return max(group for group in range(1, min(8, channels // 2) + 1) if channels % group == 0)


def expert_budget(train_count: int, policy: dict[str, Any]) -> float:
    n = positive_int(train_count, "train_count")
    policy = validate_scaling_policy(policy)
    log_budget = math.log(policy["reference_experts"]) + policy["exponent"] * (
        math.log(n) - math.log(policy["reference_samples"]))
    if log_budget <= 0:
        return 1.0
    if log_budget >= math.log(policy["max_experts"]):
        return float(policy["max_experts"])
    return min(float(policy["max_experts"]), max(1.0, math.exp(log_budget)))


def resolve_capacity(
    selection: str | int, *, train_count: int, base_channels: int,
    scaling_policy: dict[str, Any],
) -> dict[str, Any]:
    """Freeze expert count once; fixed controls may exceed the auto policy cap."""
    policy = validate_scaling_policy(scaling_policy)
    _base_channels(base_channels)
    budget = expert_budget(train_count, policy)
    if selection == "auto":
        mode = "auto"
        # A small relative tolerance prevents downward jumps at exact log thresholds.
        e = max(value for value in ALLOWED_EXPERTS if value <= budget * (1.0 + 1e-12))
    else:
        mode, e = "fixed", _experts(selection)
    return {
        "train_count": train_count, "scaling_policy_version": POLICY_VERSION,
        "scaling_policy": policy, "selection_mode": mode, "num_experts": e,
        "top_k": min(2, e), "parameter_count": parameter_count(e, base_channels),
        "active_parameter_count": active_parameter_count(e, base_channels),
        "expert_budget": budget, "architecture_version": ARCHITECTURE_VERSION,
        "stem_strides": STEM_STRIDES, "expert_projection_stride": EXPERT_PROJECTION_STRIDE,
        "residual_blocks": 1, "normalization": NORMALIZATION,
        "normalization_max_groups": 8, "routing_policy": ROUTING_POLICY,
        "balance_policy": BALANCE_POLICY,
    }


@dataclass(frozen=True)
class MoEModelConfig:
    """Every field is mandatory; legacy or partial layouts cannot be restored."""

    base_channels: int
    dropout: float
    train_count: int
    scaling_policy_version: str
    scaling_policy: dict[str, Any]
    selection_mode: str
    num_experts: int
    top_k: int
    parameter_count: int
    active_parameter_count: int
    expert_budget: float
    architecture_version: str
    stem_strides: tuple[tuple[int, int], ...]
    expert_projection_stride: tuple[int, int]
    residual_blocks: int
    normalization: str
    normalization_max_groups: int
    routing_policy: str
    balance_policy: str


def validate_serialized_config(model_config: Any, input_dimension: str) -> dict[str, Any]:
    """Validate and canonicalize JSON config without importing torch."""
    if input_dimension not in ("input_elements", "input_pairs", "input_features"):
        raise ValueError("Unknown method input dimension")
    expected = {field.name for field in fields(MoEModelConfig)} | {input_dimension, "input_freq_bins"}
    if not isinstance(model_config, dict) or set(model_config) != expected:
        raise ValueError("Model configuration requires exactly the complete residual MoE schema")
    values = dict(model_config)
    positive_int(values[input_dimension], input_dimension)
    positive_int(values["input_freq_bins"], "input_freq_bins")
    c = _base_channels(values["base_channels"])
    positive_int(values["train_count"], "train_count")
    dropout = values["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (float, int)):
        raise ValueError("dropout must be finite and satisfy 0 <= dropout < 1")
    try:
        dropout = float(dropout)
    except (ValueError, OverflowError) as exc:
        raise ValueError("dropout must be finite and satisfy 0 <= dropout < 1") from exc
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and satisfy 0 <= dropout < 1")
    values["dropout"] = dropout
    if values["selection_mode"] not in ("auto", "fixed"):
        raise ValueError("selection_mode must be auto or fixed")
    e = _experts(values["num_experts"])
    resolved = resolve_capacity("auto" if values["selection_mode"] == "auto" else e,
                                train_count=values["train_count"], base_channels=c,
                                scaling_policy=values["scaling_policy"])
    for key, expected_value in resolved.items():
        actual = values[key]
        if key in ("stem_strides", "expert_projection_stride"):
            try:
                actual = tuple(tuple(stride) for stride in actual) if key == "stem_strides" else tuple(actual)
            except TypeError as exc:
                raise ValueError(f"Invalid saved {key}") from exc
            flattened = [item for stride in actual for item in stride] if key == "stem_strides" else actual
            if any(type(item) is not int for item in flattened):
                raise ValueError(f"Invalid saved {key}")
        elif key == "expert_budget":
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                raise ValueError("Invalid saved expert_budget")
            try:
                if not math.isfinite(actual) or not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Saved expert_budget disagrees with training count and policy")
            except OverflowError as exc:
                raise ValueError("Invalid saved expert_budget") from exc
            actual = expected_value
        elif type(expected_value) is int and type(actual) is not int:
            raise ValueError(f"Invalid saved {key}")
        if actual != expected_value:
            raise ValueError(f"Saved {key} disagrees with the frozen architecture or selection policy")
        values[key] = expected_value
    return values


def validate_model_config(config: MoEModelConfig, input_dimension: str) -> None:
    values = validate_serialized_config(asdict(config), input_dimension)
    for key, value in values.items():
        object.__setattr__(config, key, value)


def _resolution_metadata(values: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: value for key, value in values.items() if key not in (
        "input_elements", "input_pairs", "input_features", "input_freq_bins", "dropout")} | {
        "calibration_status": POLICY_STATUS,
        "active_parameter_count_scope": "per_sample_including_shared_stem_and_router",
    }
    return json.loads(json.dumps(metadata, allow_nan=False))


def resolution_metadata(config: MoEModelConfig) -> dict[str, Any]:
    dimension = next(name for name in ("input_elements", "input_pairs", "input_features") if hasattr(config, name))
    validate_model_config(config, dimension)
    return _resolution_metadata(asdict(config))


def serialized_resolution_metadata(model_config: Any, input_dimension: str) -> dict[str, Any]:
    return _resolution_metadata(validate_serialized_config(model_config, input_dimension))
