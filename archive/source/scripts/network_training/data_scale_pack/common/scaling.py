"""Frozen training-count capacity policy; no tensor or numerical dependencies.

This is an uncalibrated experimental rule, not a claim of optimal capacity or
independent sample count. Only unique rows used for gradient updates count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


POLICY_VERSION = "train-count-v1"
POLICY_STATUS = "uncalibrated"
DEFAULT_SCALING_POLICY = {"reference_samples": 600, "exponent": 0.5}
PROFILE_REPEATS = {
    "small": (1, 1, 1),
    "medium": (1, 2, 1),
    "large": (2, 2, 2),
    "xlarge": (2, 3, 3),
}
STAGE_STRIDES = ((1, 2), (2, 2), (2, 1))
BASELINE_REPEATS = (1, 1, 1, 1)
BASELINE_STRIDES = ((1, 2), (2, 2), (2, 2), (1, 2))


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_scaling_policy(policy: Any) -> dict[str, int | float]:
    if not isinstance(policy, dict) or set(policy) != set(DEFAULT_SCALING_POLICY):
        raise ValueError("scaling_policy requires exactly reference_samples and exponent")
    reference = positive_int(policy["reference_samples"], "reference_samples")
    exponent = policy["exponent"]
    if (
        isinstance(exponent, bool)
        or not isinstance(exponent, (int, float))
        or not math.isfinite(exponent)
        or exponent <= 0
    ):
        raise ValueError("exponent must be a finite positive number")
    return {"reference_samples": reference, "exponent": float(exponent)}


def profile_layout(profile: str) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    if profile == "baseline":
        return BASELINE_REPEATS, BASELINE_STRIDES
    try:
        return PROFILE_REPEATS[profile], STAGE_STRIDES
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown capacity profile: {profile}") from exc


def stage_channels(base_channels: int, profile: str) -> tuple[int, ...]:
    c = positive_int(base_channels, "base_channels")
    profile_layout(profile)
    return (c, 2 * c, 4 * c, 4 * c) if profile == "baseline" else (c, 2 * c, 4 * c)


def parameter_count(profile: str, base_channels: int) -> int:
    """Exact trainable count: two bias-free kernels, two BNs, one modReLU."""
    repeats, _ = profile_layout(profile)
    channels = stage_channels(base_channels, profile)
    count, in_channels = 0, 1
    for out_channels, repeat in zip(channels, repeats, strict=True):
        for _ in range(repeat):
            count += 18 * in_channels * out_channels + 5 * out_channels
            in_channels = out_channels
    # Linear(4c,4c) + Linear(4c,1), including both biases.
    count += in_channels * in_channels + 2 * in_channels + 1
    return count


def parameter_budget(train_count: int, base_channels: int, policy: dict[str, Any]) -> float:
    n = positive_int(train_count, "train_count")
    policy = validate_scaling_policy(policy)
    low = parameter_count("small", base_channels)
    reference = parameter_count("medium", base_channels)
    high = parameter_count("xlarge", base_channels)
    log_factor = policy["exponent"] * (math.log(n) - math.log(policy["reference_samples"]))
    if log_factor <= math.log(low / reference):
        return float(low)
    if log_factor >= math.log(high / reference):
        return float(high)
    return min(float(high), max(float(low), reference * math.exp(log_factor)))


def resolve_capacity(
    profile: str,
    *,
    train_count: int,
    base_channels: int,
    scaling_policy: dict[str, Any],
) -> dict[str, Any]:
    """Resolve exactly once before initialization; saved layouts drive restoration."""
    policy = validate_scaling_policy(scaling_policy)
    budget = parameter_budget(train_count, base_channels, policy)
    mode = "auto" if profile == "auto" else "fixed"
    if mode == "auto":
        profile = max(
            (name for name in PROFILE_REPEATS if parameter_count(name, base_channels) <= budget),
            key=lambda name: parameter_count(name, base_channels),
        )
    repeats, strides = profile_layout(profile)
    return {
        "train_count": train_count,
        "scaling_policy_version": POLICY_VERSION,
        "scaling_policy": policy,
        "selection_mode": mode,
        "selected_profile": profile,
        "stage_repeats": repeats,
        "stage_strides": strides,
        "parameter_count": parameter_count(profile, base_channels),
        "parameter_budget": budget,
    }


@dataclass(frozen=True)
class ScaleModelConfig:
    """All fields required so incomplete or legacy checkpoints fail explicitly."""

    base_channels: int
    dropout: float
    train_count: int
    scaling_policy_version: str
    scaling_policy: dict[str, Any]
    selection_mode: str
    selected_profile: str
    stage_repeats: tuple[int, ...]
    stage_strides: tuple[tuple[int, int], ...]
    parameter_count: int
    parameter_budget: float


def validate_model_config(config: ScaleModelConfig, input_dimension: str) -> None:
    positive_int(getattr(config, input_dimension), input_dimension)
    positive_int(config.input_freq_bins, "input_freq_bins")
    positive_int(config.base_channels, "base_channels")
    positive_int(config.train_count, "train_count")
    if (
        isinstance(config.dropout, bool)
        or not isinstance(config.dropout, (float, int))
        or not math.isfinite(config.dropout)
        or not 0 <= config.dropout < 1
    ):
        raise ValueError("dropout must be finite and satisfy 0 <= dropout < 1")
    if config.scaling_policy_version != POLICY_VERSION:
        raise ValueError(f"scaling_policy_version must be {POLICY_VERSION}")
    policy = validate_scaling_policy(config.scaling_policy)
    if config.selection_mode not in ("auto", "fixed"):
        raise ValueError("selection_mode must be auto or fixed")
    repeats, strides = profile_layout(config.selected_profile)
    try:
        stored_repeats = tuple(config.stage_repeats)
        stored_strides = tuple(tuple(pair) for pair in config.stage_strides)
    except TypeError as exc:
        raise ValueError("Invalid saved stage layout") from exc
    if (
        any(type(value) is not int for value in stored_repeats)
        or any(type(value) is not int for pair in stored_strides for value in pair)
        or stored_repeats != repeats
        or stored_strides != strides
    ):
        raise ValueError("Saved stage layout disagrees with selected_profile")
    count = parameter_count(config.selected_profile, config.base_channels)
    if type(config.parameter_count) is not int or config.parameter_count != count:
        raise ValueError("Saved parameter_count disagrees with architecture")
    budget = parameter_budget(config.train_count, config.base_channels, policy)
    if (
        isinstance(config.parameter_budget, bool)
        or not isinstance(config.parameter_budget, (float, int))
        or not math.isfinite(config.parameter_budget)
        or not math.isclose(config.parameter_budget, budget, rel_tol=1e-12, abs_tol=1e-9)
    ):
        raise ValueError("Saved parameter_budget disagrees with training count and policy")
    # Verify saved choice without resolving a new configuration or using new data.
    if config.selection_mode == "auto":
        if config.selected_profile not in PROFILE_REPEATS or count > budget:
            raise ValueError("Saved automatic profile exceeds its capacity budget")
        if any(count < parameter_count(name, config.base_channels) <= budget for name in PROFILE_REPEATS):
            raise ValueError("Saved automatic profile is not the largest within budget")
    object.__setattr__(config, "stage_repeats", stored_repeats)
    object.__setattr__(config, "stage_strides", stored_strides)
    object.__setattr__(config, "scaling_policy", policy)


def resolution_metadata(config: ScaleModelConfig) -> dict[str, Any]:
    values = asdict(config)
    keys = (
        "train_count", "scaling_policy_version", "scaling_policy", "selection_mode",
        "selected_profile", "stage_repeats", "stage_strides", "parameter_count", "parameter_budget",
    )
    metadata = {**{key: values[key] for key in keys}, "calibration_status": POLICY_STATUS}
    metadata["stage_repeats"] = list(config.stage_repeats)
    metadata["stage_strides"] = [list(stride) for stride in config.stage_strides]
    return metadata
