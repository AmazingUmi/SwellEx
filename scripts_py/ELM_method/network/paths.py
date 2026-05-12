"""ELM network path defaults."""

from __future__ import annotations

from common.paths import (
    DATASETS_DIR,
    TEST_OUTPUTS_DIRNAME,
    TRAIN_OUTPUTS_DIRNAME,
    dataset_dir,
    dataset_test_glob,
    dataset_train_glob,
    dataset_val_glob,
    latest_time_suffixed_path,
    model_output_dir,
    safe_name,
    test_output_dir,
    time_suffix,
    train_output_dir,
    with_time_suffix,
)
from common.paths import PROJECT_ROOT


DEFAULT_DATASET_STRATEGY = "periodic_4_1_elm_pairwise_ratio_upper_mel64"
DEFAULT_DATASET_DIR = DATASETS_DIR / DEFAULT_DATASET_STRATEGY
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "networks_results" / "ELM_method"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT
