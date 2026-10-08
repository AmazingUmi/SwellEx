"""SCM-GRNN artifact directories; shared path operations live in common."""

from pathlib import Path

from scripts.localization_networks.common.paths import PROJECT_ROOT, model_output_dir

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "networks_results" / "SCM"
MODEL_DIRNAME = "scm_grnn_range"


def reference_output_dir(output_dir: Path) -> Path:
    return model_output_dir(output_dir, MODEL_DIRNAME) / "reference_outputs"


def test_output_dir(output_dir: Path) -> Path:
    return model_output_dir(output_dir, MODEL_DIRNAME) / "test_outputs"
