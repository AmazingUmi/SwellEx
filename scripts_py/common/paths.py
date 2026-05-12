"""Shared path helpers for Python network scripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "outputs" / "Datasets"
TRAIN_OUTPUTS_DIRNAME = "train_outputs"
TEST_OUTPUTS_DIRNAME = "test_outputs"


def time_suffix() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


def with_time_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def safe_name(value: str) -> str:
    name = Path(value).name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def dataset_dir(dataset: str) -> Path:
    path = Path(dataset)
    if path.exists():
        return path
    return DATASETS_DIR / dataset


def dataset_train_glob(dataset: str) -> str:
    return str(dataset_dir(dataset) / "*_train.h5")


def dataset_val_glob(dataset: str) -> str:
    return str(dataset_dir(dataset) / "*_val.h5")


def dataset_test_glob(dataset: str) -> str:
    return str(dataset_dir(dataset) / "*_test.h5")


def model_output_dir(output_root: Path, model_name: str) -> Path:
    if output_root.name == model_name:
        return output_root
    return output_root / model_name


def train_output_dir(output_root: Path, model_name: str) -> Path:
    return model_output_dir(output_root, model_name) / TRAIN_OUTPUTS_DIRNAME


def test_output_dir(output_root: Path, model_name: str) -> Path:
    return model_output_dir(output_root, model_name) / TEST_OUTPUTS_DIRNAME


def latest_time_suffixed_path(path: Path) -> Path | None:
    matches = sorted(path.parent.glob(f"{path.stem}_*{path.suffix}"))
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime)
