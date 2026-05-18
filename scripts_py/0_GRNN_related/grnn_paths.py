"""Path helpers for standalone SCM-GRNN workflows."""

from __future__ import annotations

import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "outputs" / "Datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "networks_results" / "0_GRNN_related"
MODEL_DIRNAME = "scm_grnn_range"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def time_suffix() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


def with_time_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def dataset_train_glob(dataset_name: str) -> str:
    return str(DATASETS_DIR / dataset_name / "*_train.h5")


def dataset_val_glob(dataset_name: str) -> str:
    return str(DATASETS_DIR / dataset_name / "*_val.h5")


def dataset_test_glob(dataset_name: str) -> str:
    return str(DATASETS_DIR / dataset_name / "*_test.h5")


def resolve_h5_paths(patterns: Iterable[str], required: bool = True) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            path = Path(pattern)
            if path.is_file():
                paths.append(path)
    paths = sorted({path.resolve() for path in paths})
    if required and not paths:
        raise FileNotFoundError("No HDF5 files matched the provided path or glob.")
    return paths


def model_output_dir(output_dir: Path) -> Path:
    if output_dir.name == MODEL_DIRNAME:
        return output_dir
    return output_dir / MODEL_DIRNAME


def reference_output_dir(output_dir: Path) -> Path:
    return model_output_dir(output_dir) / "reference_outputs"


def test_output_dir(output_dir: Path) -> Path:
    return model_output_dir(output_dir) / "test_outputs"


def latest_time_suffixed_path(base_path: Path) -> Path | None:
    pattern = base_path.with_name(f"{base_path.stem}_*{base_path.suffix}")
    matches = sorted(base_path.parent.glob(pattern.name), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None
