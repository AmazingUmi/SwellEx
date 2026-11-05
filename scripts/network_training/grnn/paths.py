"""Method-specific GRNN artifacts; shared path operations live in common."""

from pathlib import Path

from scripts.network_training.common.paths import default_output_dir as method_output_dir, model_output_dir
from scripts.network_training.grnn.bindings import model_name, validate_method


def default_output_dir(method: str) -> Path:
    return method_output_dir(validate_method(method))


def reference_output_dir(output_dir: Path, *, method: str) -> Path:
    return model_output_dir(output_dir, model_name(method)) / "reference_outputs"


def test_output_dir(output_dir: Path, *, method: str) -> Path:
    return model_output_dir(output_dir, model_name(method)) / "test_outputs"
