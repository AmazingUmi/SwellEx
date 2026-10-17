"""Semantic identities of feature axes, shared by neural networks and GRNN."""
from __future__ import annotations

from typing import Any
import math


def validate_saved_input_signature(signature: Any, input_shape: tuple[int, int, int], method: str) -> None:
    """Reject incomplete physical axes before a saved model can be restored."""
    if method == "RBD_SCM":
        from scripts.MoE_learn.localization_networks.rbd_scm.contract import validate_signature
        validate_signature(signature, input_shape, method)
        return
    if method not in {"RBD", "ELM", "SCM"}:
        raise ValueError("Unsupported feature method")
    keys = {"method", "input_shape", "frequency_hz", "element_idx", "depth_m", "pair_indices"}
    if not isinstance(signature, dict) or set(signature) != keys or signature["method"] != method:
        raise ValueError("Saved input signature requires the complete method and physical-axis schema")
    if signature["input_shape"] != list(input_shape):
        raise ValueError("Saved input signature shape differs from model configuration")
    for name, count in (("frequency_hz", input_shape[2]), ("depth_m", None)):
        values = signature[name]
        if (not isinstance(values, list) or not values or (count is not None and len(values) != count)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values)
                or any(left >= right for left, right in zip(values, values[1:]))):
            raise ValueError(f"Saved input signature {name} must be finite, nonnegative and strictly increasing")
    elements = signature["element_idx"]
    if (not isinstance(elements, list) or any(type(value) is not int for value in elements)
            or elements != list(range(1, len(signature["depth_m"]) + 1))):
        raise ValueError("Saved input signature requires ordered 1-based element positions")
    expected_pairs = [] if method == "RBD" else [
        [i, j] for j in range(1, len(elements) + 1) for i in range(1, j + (method == "SCM"))]
    pairs = signature["pair_indices"]
    if (not isinstance(pairs, list) or any(not isinstance(pair, list) or len(pair) != 2
                                         or any(type(value) is not int for value in pair) for pair in pairs)
            or pairs != expected_pairs):
        raise ValueError("Saved input signature has invalid column-major pair axes")
    if input_shape[1] != (len(elements) if method == "RBD" else len(pairs)):
        raise ValueError("Saved feature row count differs from the physical array axes")


def _vector(handle, key: str):
    import numpy as np
    if key not in handle:
        raise ValueError(f"{handle.filename} missing required input metadata /{key}.")
    values = np.asarray(handle[key])
    if values.ndim > 2 or (values.ndim == 2 and 1 not in values.shape):
        raise ValueError(f"{handle.filename} /{key} must be a vector.")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{handle.filename} /{key} must contain finite numeric values.")
    return values.reshape(-1)


def read_input_signature(handle, input_shape: tuple[int, int, int], method: str) -> dict[str, Any]:
    """Read a JSON-compatible signature for PyTorch shape ``(2, E|P, F)``.

    Pair IDs refer to 1-based array positions, traversed column first. Axis
    identity is exact: silently interpolating/reordering frequencies is unsafe.
    """
    import numpy as np
    if method not in {"RBD", "ELM", "SCM"}:
        raise ValueError(f"Unsupported feature method: {method}")
    if len(input_shape) != 3 or input_shape[0] != 2 or any(v <= 0 for v in input_shape):
        raise ValueError(f"Invalid feature input shape: {input_shape}")
    frequencies = _vector(handle, "frequency/freq_hz").astype(np.float64)
    elements = _vector(handle, "array/element_idx")
    depths = _vector(handle, "array/depth_m").astype(np.float64)
    if frequencies.size != input_shape[2] or (frequencies < 0).any() or not (np.diff(frequencies) > 0).all():
        raise ValueError(f"{handle.filename} frequency axis must match input bins and increase strictly.")
    if not elements.size or elements.size != depths.size:
        raise ValueError(f"{handle.filename} element IDs and depths must have equal nonzero lengths.")
    if not np.array_equal(elements, np.arange(1, elements.size + 1)):
        raise ValueError(f"{handle.filename} element IDs must be ordered 1-based array positions.")
    if (depths < 0).any() or not (np.diff(depths) > 0).all():
        raise ValueError(f"{handle.filename} array depths must be nonnegative and increase strictly.")
    pairs: list[list[int]] = []
    if method == "RBD":
        if input_shape[1] != elements.size:
            raise ValueError(f"{handle.filename} RBD feature height differs from array element count.")
    else:
        numerator = _vector(handle, "pair/numerator_element_idx")
        denominator = _vector(handle, "pair/denominator_element_idx")
        expected = [(i, j) for j in range(1, elements.size + 1)
                    for i in range(1, j + (method == "SCM"))]
        if input_shape[1] != len(expected) or numerator.size != len(expected) or denominator.size != len(expected):
            raise ValueError(f"{handle.filename} {method} pair count does not match array geometry.")
        if not np.array_equal(np.column_stack((numerator, denominator)), np.asarray(expected)):
            relation = "i<j" if method == "ELM" else "i<=j"
            raise ValueError(f"{handle.filename} {method} pairs must be complete 1-based column-major {relation}.")
        pairs = [[i, j] for i, j in expected]
    return {"method": method, "input_shape": list(input_shape),
            "frequency_hz": frequencies.astype(float).tolist(),
            "element_idx": elements.astype(int).tolist(),
            "depth_m": depths.astype(float).tolist(), "pair_indices": pairs}


def require_matching_input_signature(expected: dict[str, Any], actual: dict[str, Any], context: str) -> None:
    if expected != actual:
        changed = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
        raise ValueError(f"{context}: input signature mismatch ({', '.join(changed)}).")
