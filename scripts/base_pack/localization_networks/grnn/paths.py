"""Method-specific GRNN artifacts; shared path operations live in common."""

from pathlib import Path

from scripts.base_pack.localization_networks.common.paths import PROJECT_ROOT, model_output_dir
from scripts.base_pack.localization_networks.grnn.bindings import model_name, validate_method


def default_output_dir(method: str) -> Path:
    return PROJECT_ROOT / "outputs" / "networks_results" / validate_method(method)


def reference_output_dir(output_dir: Path, *, method: str) -> Path:
    return model_output_dir(output_dir, model_name(method)) / "reference_outputs"


def test_output_dir(output_dir: Path, *, method: str) -> Path:
    return model_output_dir(output_dir, model_name(method)) / "test_outputs"
