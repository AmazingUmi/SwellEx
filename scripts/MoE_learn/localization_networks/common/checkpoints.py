"""Self-contained deployment artifacts; stage continuation is a separate contract."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch

from .model import ContinualMoE, ModelConfig
from .input_contract import require_matching_input_signature, validate_saved_input_signature
from scripts.MoE_learn.localization_networks.rbd_scm.contract import validate_preprocessing
from scripts.MoE_learn.localization_networks.workflow.config import validate_config


def validate_metadata(metadata, config):
    required = {"pack", "method", "fusion", "input_signature", "input_shape", "source_paths",
                "preprocessing", "preprocessing_fit_stage", "input_normalized_per_sample", "stages", "test_source_ids"}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("Incomplete continual input metadata")
    if metadata["pack"] != "MoE_learn" or metadata["method"] != "RBD_SCM" or metadata["fusion"] != "block_rms":
        raise ValueError("Checkpoint input metadata belongs to another pack/method/fusion")
    shape = (2, config.input_features, config.input_freq_bins)
    if metadata["input_shape"] != list(shape):
        raise ValueError("Metadata input shape differs from architecture")
    validate_saved_input_signature(metadata["input_signature"], shape, "RBD_SCM")
    validate_preprocessing(metadata["preprocessing"], metadata["input_signature"])
    if metadata["preprocessing_fit_stage"] != 1 or metadata["input_normalized_per_sample"] is not True:
        raise ValueError("Continual demo requires immutable stage-one preprocessing and sample normalization")
    if not isinstance(metadata["source_paths"], list) or not metadata["source_paths"] or any(not isinstance(x, str) or not x for x in metadata["source_paths"]):
        raise ValueError("Input source paths are missing")
    stages = metadata["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("Stage memberships are required")
    all_ids = []
    for index, stage in enumerate(stages, 1):
        if not isinstance(stage, dict) or stage.get("stage") != index:
            raise ValueError("Stage identity must be consecutive and one based")
        for field in ("train_source_ids", "val_source_ids"):
            ids = stage.get(field)
            if not isinstance(ids, list) or not ids or any(type(x) is not int or x <= 0 for x in ids) or len(set(ids)) != len(ids):
                raise ValueError("Stage source identities are incomplete")
            all_ids.extend(ids)
    test_ids = metadata["test_source_ids"]
    if not isinstance(test_ids, list) or not test_ids or any(type(x) is not int or x <= 0 for x in test_ids):
        raise ValueError("Test membership is missing")
    if len(set(all_ids + test_ids)) != len(all_ids + test_ids):
        raise ValueError("Train/validation/test or stage identities overlap")
    first_ids = stages[0]["train_source_ids"]
    fitted = metadata["preprocessing"]
    digest = hashlib.sha256(",".join(str(i) for i in first_ids).encode()).hexdigest()
    if fitted["fit_sample_count"] != len(first_ids) or fitted["fit_source_identity_sha256"] != digest:
        raise ValueError("Preprocessing was not fitted exclusively on stage-one optimizer rows")


def validate_runtime_metadata(runtime, metadata):
    """Bind the saved experiment protocol to its actual data allocation."""
    dataset = runtime["dataset"]
    mode, recorded_mode = dataset["stage_mode"], metadata.get("stage_mode")
    if recorded_mode is None:
        generator = metadata["input_signature"]["source_contract"]["generator_version"]
        if mode == "distance_holdout" or generator != "moe-learn-synthetic-v1":
            raise ValueError("Real checkpoint metadata must declare its stage_mode")
    elif mode != recorded_mode:
        raise ValueError("Runtime stage_mode differs from checkpoint metadata")
    if mode == "distance_holdout":
        allocation = metadata.get("allocation")
        if (not isinstance(allocation, dict)
                or type(allocation.get("heldout_group")) is not int
                or allocation["heldout_group"] != dataset["heldout_group"]):
            raise ValueError("Runtime heldout_group differs from the recorded distance allocation")
        edges = allocation.get("distance_edges_km")
        if (not isinstance(edges, list)
                or any(type(edge) not in (int, float) or not math.isfinite(edge) for edge in edges)
                or edges != dataset["distance_edges_km"]):
            raise ValueError("Runtime distance_edges_km differ from the recorded distance allocation")


def state_digest(model, prefixes=None):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if prefixes is not None and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, *, target, runtime_config, metadata, descriptor_reference, controller_records, stage):
    validate_config(runtime_config)
    validate_metadata(metadata, model.config)
    validate_runtime_metadata(runtime_config, metadata)
    payload = {
        "checkpoint_version": 1, "pack": "MoE_learn", "method": "RBD_SCM",
        "artifact_role": "deployment", "stage": stage,
        "model_config": asdict(model.config), "model_state": model.state_dict(),
        "model_state_sha256": state_digest(model),
        "frozen_experts": sorted(model.frozen_experts), "stem_frozen": model.stem_frozen,
        "expert_ids": list(range(model.num_experts)),
        "target": target, "runtime_config": runtime_config, "metadata": metadata,
        "descriptor_reference": descriptor_reference, "controller_records": controller_records,
        "resume_training_supported": False,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def load_checkpoint(path, device, *, expected_metadata=None):
    payload = torch.load(path, map_location=device, weights_only=True)
    required = {"checkpoint_version", "pack", "method", "artifact_role", "stage", "model_config", "model_state",
                "model_state_sha256", "frozen_experts", "stem_frozen", "expert_ids", "target", "runtime_config",
                "metadata", "descriptor_reference", "controller_records", "resume_training_supported"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Incomplete or obsolete continual checkpoint schema")
    if (payload["checkpoint_version"] != 1 or payload["pack"] != "MoE_learn" or payload["method"] != "RBD_SCM"
            or payload["artifact_role"] != "deployment" or payload["resume_training_supported"] is not False):
        raise ValueError("Unsupported or foreign checkpoint identity")
    config = ModelConfig(**payload["model_config"])
    validate_metadata(payload["metadata"], config)
    validate_config(payload["runtime_config"])
    runtime = payload["runtime_config"]
    validate_runtime_metadata(runtime, payload["metadata"])
    if expected_metadata is not None:
        validate_metadata(expected_metadata, config)
        validate_runtime_metadata(runtime, expected_metadata)
        require_matching_input_signature(expected_metadata["input_signature"], payload["metadata"]["input_signature"], "Restored checkpoint")
        if expected_metadata["preprocessing"] != payload["metadata"]["preprocessing"]:
            raise ValueError("Restored preprocessing differs from expected frozen scales")
        for field in ("stages", "test_source_ids", "allocation"):
            if expected_metadata.get(field) != payload["metadata"].get(field):
                raise ValueError(f"Restored {field} differs from expected data membership/allocation")
    target = payload["target"]
    if not isinstance(target, dict) or set(target) != {"mean_km", "std_km", "fit_stage", "fit_sample_count", "fit_source_identity_sha256"}:
        raise ValueError("Incomplete target normalization contract")
    for key in ("mean_km", "std_km"):
        if isinstance(target[key], bool) or not isinstance(target[key], (int, float)) or not math.isfinite(target[key]):
            raise ValueError("Nonfinite target normalization")
    fitted = payload["metadata"]["preprocessing"]
    if (target["std_km"] <= 0 or target["fit_stage"] != 1 or target["fit_sample_count"] != fitted["fit_sample_count"]
            or target["fit_source_identity_sha256"] != fitted["fit_source_identity_sha256"]):
        raise ValueError("Target normalization must retain stage-one membership")
    if payload["expert_ids"] != list(range(config.num_experts)):
        raise ValueError("Stable expert IDs do not match model architecture")
    if payload["stem_frozen"] is not True or payload["frozen_experts"] != payload["expert_ids"]:
        raise ValueError("Deployment checkpoint must freeze its stem and all expert branches")
    if type(payload["stage"]) is not int or not 1 <= payload["stage"] <= len(payload["metadata"]["stages"]):
        raise ValueError("Invalid deployment stage")
    if (runtime["training"]["base_channels"] != config.base_channels
            or not runtime["continual"]["initial_experts"] <= config.num_experts <= runtime["continual"]["max_experts"]
            or runtime["continual"]["stages"] != len(payload["metadata"]["stages"])):
        raise ValueError("Saved runtime configuration differs from resolved model/stage identities")
    if not isinstance(payload["controller_records"], list) or len(payload["controller_records"]) != payload["stage"]:
        raise ValueError("Missing configuration or controller history")
    reference = payload["descriptor_reference"]
    if not isinstance(reference, dict) or set(reference) != {"mean", "variance", "count", "fit_stage", "version"}:
        raise ValueError("Incomplete descriptor reference")
    if reference["fit_stage"] != 1 or reference["version"] != "frozen-stem-magnitude-mean-std-v1" or reference["count"] != fitted["fit_sample_count"]:
        raise ValueError("Descriptor reference provenance differs")
    for key in ("mean", "variance"):
        values = reference[key]
        if not isinstance(values, list) or len(values) != 4 * config.base_channels or any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("Invalid descriptor reference dimensions/values")
    if any(v < 0 for v in reference["variance"]):
        raise ValueError("Negative descriptor variance")
    model = ContinualMoE(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.freeze_for_deployment()
    model.eval()
    if state_digest(model) != payload["model_state_sha256"]:
        raise ValueError("Checkpoint model-state digest mismatch")
    # All non-tensor contracts must remain plain JSON-compatible evidence.
    json.dumps({k: v for k, v in payload.items() if k != "model_state"}, allow_nan=False)
    return model, payload
