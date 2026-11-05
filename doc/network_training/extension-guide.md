# Python Extension Guide

Read the [documentation index](../README.md#experiment-pack) and [HDF5 contract](../data_processing/hdf5-datasets-and-splits.md) before modifying loaders or models.

## Directory Map and Layer Isolation

Code is partitioned across three top-level packages under `scripts/` (enforced by [`test_layer_isolation.py`](../../scripts/remote_workflow/tests/test_layer_isolation.py)):

| Path | Responsibility & isolation boundary |
|---|---|
| `scripts/run.sh`, `scripts/remote_workflow/{run.py,pack_cli.py,packs.py,run_identity.py}` | Standard-library-only launcher, shell argument contract (`--preset`/`--method`/`--network`), pack registry, and run IDs |
| `scripts/data_processing/` | Pure CPU raw readers, feature math, splits, HDF5 writer, and validation (`generate.py`, `validate.py`, `input_contract.py` for `METHODS`/axis identities); never imports PyTorch, matplotlib, `network_training`, or `remote_workflow` |
| `scripts/network_training/{common,neural,grnn}/`, `catalog.py` | Pure HDF5 loaders, the single neural runtime (`neural/`: CLI, data, model registry, training, prediction), the standalone GRNN, shared paths/argument types (`common/`), and the stdlib-only model catalog; never imports `scipy`, `Origindata`, `generation_core`, or `remote_workflow` |
| `scripts/remote_workflow/` | SSH snapshot/dispatch (`remote.py`), worker (`worker.py`), execution plan (`model_plan.py`), TOML `presets/`, runtime checks (`validation/`), and report plots (`visualization/`) |

Only `base_pack` is active. Archived packs (ELM, joint-feature, data-scale, MoE) are kept as reference snapshots under [`archive/`](../../archive/README.md) and are not importable from `scripts/`.

## Naming and Shared Logic

Use `snake_case` modules and canonical `scripts.data_processing.*`, `scripts.network_training.*`, and `scripts.remote_workflow.*` imports without compatibility aliases. Reuse `network_training/common/{h5_utils.py,paths.py,cli_args.py}`, `data_processing/input_contract.py`, and `neural/bindings.py:MethodBindings` for shared neural train/predict logic; RBD and SCM differ only by the data class (`neural/data.py`) and config dataclass (`neural/models/configs.py`). `scripts/remote_workflow/run_identity.py` owns descriptive run IDs; `scripts/remote_workflow/run_monitor.py` follows remote logs with transport retry. Keep `scripts/data_processing/matlab_reference/` static.

## Add a Model

1. Implement the `nn.Module` in `scripts/network_training/neural/models/` (shared by RBD and SCM) returning normalized `[batch, 1]`, and add one config dataclass per method in `neural/models/configs.py` (`RBD` uses `input_elements`, `SCM` uses `input_pairs`, plus `input_freq_bins`; field names are checkpoint contract).
2. Register the family in `neural/model.py` (`_MODEL_CLASSES`, `_CONFIG_CLASSES`) and in `catalog.py` (`NEURAL_MODELS`; keep GRNN outside trainable registries). Registry names follow `catalog.neural_model_name` (RBD unprefixed, SCM `scm_`).
3. Check `remote_workflow/validation/quick.py` registry assertions, add the architecture to expectations in `remote_workflow/tests`, and verify checkpoint round-trip plus a remote CUDA forward/backward check.

## Add or Change a Loader

Extend `neural/data.py` using `network_training/common/h5_utils.py` and `data_processing/input_contract.py`, preserving the `DatasetBundle` fields:

```text
train_dataset, val_dataset, train_idx, val_idx, train_labels,
input_shape, split_mode, source_paths, input_signature
```

Compare full physical frequency, element/depth, and pair signatures rather than tensor shapes alone, and add malformed-input tests in `neural/tests/`. Validate changes on `haiqin` via `./scripts/run.sh base_pack validate --suite quick` (and `smoke` when changing generation, checkpoints, or exports).
