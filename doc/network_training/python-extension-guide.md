# Python Development Guide

Read the [documentation index](../README.md#experiment-packs) and [HDF5 contract](../data_processing/hdf5-datasets-and-splits.md) before modifying loaders or models.

## Directory Map and Layer Isolation

Code is partitioned across three top-level packages under `scripts/` (enforced by [`test_layer_isolation.py`](../../scripts/remote_workflow/tests/test_layer_isolation.py)):

| Path | Responsibility & isolation boundary |
|---|---|
| `scripts/run.sh`, `scripts/remote_workflow/{run.py,pack_cli.py,packs.py}` | Standard-library-only launcher, shell argument contract, and pack registry |
| `scripts/data_processing/{common,<pack>}/` | Pure CPU raw readers, feature math, joint HDF5 builders, splits, and validation (`generate.py`, `validate.py`); never imports PyTorch, matplotlib, `network_training`, or `remote_workflow` |
| `scripts/network_training/<pack>/{common,rbd,elm,scm,grnn,...}/` | Pure HDF5 loaders, `MODEL_REGISTRY`, neural/GRNN CLI, checkpoints, and `<pack>/catalog.py`; never imports `scipy`, `Origindata`, `generation_core`, or `remote_workflow` |
| `scripts/remote_workflow/<pack>/` | SSH snapshot/dispatch (`remote.py`), workers, TOML presets, `study`/`budget-study` coordinators, runtime checks (`validation/`), and report plots (`visualization/`) |

Pack-specific `network_training` layouts:
- `joint_feature_pack`: neural entry is `scripts/network_training/joint_feature_pack/joint/main.py` (`input_features` height) alongside `grnn/main.py`.
- `data_scale_pack`: scalable CNN profiles and `scaling.py` in `scripts/network_training/data_scale_pack/common/`; no GRNN.
- `MoE_data_scale`: shared `common/moe.py` and `rbd_scm/` combined route; no GRNN.
- `MoE_learn`: pure-HDF5 staged `scripts/network_training/MoE_learn/demo.py` (`--train-h5` / `--test-h5`), `common/controller.py`, and `visualization/plot_stages.py`; single `rbd_scm` route.

## Naming and Shared Logic

Use `snake_case` modules and canonical `scripts.data_processing.*`, `scripts.network_training.*`, and `scripts.remote_workflow.*` imports without compatibility aliases. Reuse `<pack>/common/{h5_utils.py,input_contract.py,paths.py}` and `MethodBindings` for shared neural CLI/train/predict logic. `scripts/remote_workflow/run_identity.py` owns descriptive run IDs; `scripts/remote_workflow/run_monitor.py` follows remote logs with transport retry. Keep `scripts/data_processing/matlab_reference/` static.

## Add a Model

1. Implement config dataclass + `nn.Module` under `scripts/network_training/<pack>/<method>/network/models/` returning normalized `[batch, 1]` (`RBD` config uses `input_elements`, `ELM`/`SCM` use `input_pairs`, joint/`RBD_SCM` use `input_features`, plus `input_freq_bins`).
2. Register in `network/model.py:MODEL_REGISTRY` and `MODEL_DATASET_INPUT_NORM` (keep GRNN outside trainable registries).
3. Update `scripts/remote_workflow/packs.py`, `scripts/network_training/<pack>/catalog.py`, and `scripts/remote_workflow/<pack>/validation/quick.py` registry assertions, and verify checkpoint round-trip plus a remote CUDA forward/backward check.

## Add or Change a Loader

Extend `network/data.py` using `common/h5_utils.py` and `common/input_contract.py`, preserving the `DatasetBundle` fields:

```text
train_dataset, val_dataset, train_idx, val_idx, train_labels,
input_shape, split_mode, source_paths, input_signature
```

Compare full physical frequency, element/depth, and pair signatures rather than tensor shapes alone, and add malformed-input tests in `tests/`. Validate changes on `haiqin` via `./scripts/run.sh PACK validate --suite quick` (and `smoke` when changing generation, checkpoints, or exports).
