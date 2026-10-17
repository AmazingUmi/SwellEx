"""Small continual-learning plans; no data-size or expert-count matrix."""
from __future__ import annotations

import copy

from scripts.packs import get_pack
from scripts.run_identity import validate_filters
from scripts.MoE_learn.localization_networks.workflow.config import validate_config

PACK = "MoE_learn"
SUITES = ("quick", "smoke", "experiment", "selected")
METHODS = get_pack(PACK).methods
ARCHITECTURES = get_pack(PACK).networks


def model_plan(suite: str, selection: dict | None = None) -> list[dict]:
    if suite not in SUITES:
        raise ValueError(f"Unknown execution suite: {suite}")
    if suite == "selected":
        validate_filters(PACK, selection)
    elif selection is not None:
        raise ValueError("Selection is allowed only for the selected suite")
    if suite == "quick":
        return []
    return [{"method": METHODS[0], "architecture": ARCHITECTURES[0],
             "model": "rbd_scm_complex_moe_learn_range"}]


def resolved_config(config: dict, suite: str) -> dict:
    """Freeze validation reductions separately from the submitted configuration."""
    model_plan(suite, {"method": METHODS[0], "network": None} if suite == "selected" else None)
    result = copy.deepcopy(validate_config(config))
    if suite == "quick":
        result["training"].update(initial_epochs=1, adapt_epochs=1, warmup_epochs=1,
                                  batch_size=4, base_channels=2)
        result["continual"].update(stages=2, initial_experts=2, max_experts=4)
        if result["dataset"]["stage_mode"] == "distance_holdout":
            # The bounded quick demo is synthetic; real distance exclusion is checked by its fixtures.
            result["dataset"]["stage_mode"] = "interleaved"
            del result["dataset"]["distance_edges_km"]
            del result["dataset"]["heldout_group"]
        if result["dataset"].get("after_grouping") == "range_bins":
            # Quick has one synthetic increment; retain only the outer declared bounds.
            edges = result["dataset"]["after_range_edges_km"]
            result["dataset"]["after_range_edges_km"] = [edges[0], edges[-1]]
        if result["continual"]["replay_mode"] == "buffer":
            result["continual"]["replay_size"] = min(16, result["continual"]["replay_size"])
    if suite == "smoke":
        for key in ("initial_epochs", "adapt_epochs", "warmup_epochs"):
            result["training"][key] = min(1, result["training"][key])
        dataset = result["dataset"]
        if dataset["split_strategy"] == "full_interleaved":
            dataset["segment_end_s"] = min(dataset["segment_end_s"], dataset["segment_start_s"] + 39)
    return validate_config(result)


def expected_phases(suite: str, selection: dict | None = None) -> set[str]:
    model_plan(suite, selection)
    phases = {"snapshot_integrity", "environment", "demo", "source_unchanged_after_validation"}
    if suite != "quick":
        phases.add("original_data_identity")
    return phases
