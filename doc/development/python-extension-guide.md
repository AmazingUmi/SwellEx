# Python Development Guide

Use this guide to locate modules and extend models or loaders. Read the
[pack contracts](../packs/README.md) for ownership and the
[HDF5 contract](../datasets/hdf5-datasets-and-splits.md) before loader changes.
ELM and SCM have the same tensor rank but different pair semantics.

## Directory Map

Paths below are relative to `scripts/base_pack/`, except the shared entry row.
Joint experiments own corresponding modules under `scripts/joint_feature_pack/`;
their neural entry is `localization_networks/joint/main.py`. Read the
[joint feature contract](../packs/joint-feature-pack.md) before changing fusion
loaders or fitted preprocessing. Keep joint-only changes out of the base pack.
Training-count capacity changes belong in `scripts/data_scale_pack/`; see its
[resolved architecture contract](../packs/data-scale-pack.md#artifacts-and-calibration).
The shared pack registry declares each pack's accepted network families.
Residual expert-count models belong to `scripts/MoE_data_scale/`, with the
user-selected canonical pack spelling. Its [MoE contract](../packs/moe-data-scale.md)
defines the identity shortcuts, routing diagnostics and architecture-only checks.
Its `rbd_scm/` method uses `input_features` and aligned RBD+SCM blocks with fixed
`block_rms`; fit preprocessing after training subsampling. Its separate `study`
coordinator freezes data-size/expert-count/seed cells and resumes without redispatch.

| Directory | Responsibility | Entry point |
|---|---|---|
| Repository `scripts/` | Pack selection and shared shell contract | `run.sh` → `run.py` |
| `data_processing/` | Raw inputs, feature math, windows/splits, and HDF5 generation | `generate.py`, `validate.py` |
| `localization_networks/rbd/`, `elm/`, `scm/` | Method loaders, registries, and neural models | Each package's `main.py train/predict` |
| `localization_networks/grnn/` | Reference samples, sigma selection, and kernel prediction | `main.py build/predict --method METHOD` |
| `localization_networks/common/` | Shared neural CLI, training/prediction, checkpoints, input contracts, and paths | Imported by method packages |
| `localization_networks/workflow/` | Dispatch, remote execution/cache/retrieval, local CPU GRNN, and saved-result comparison | `remote.py`, `local_grnn.py`, `compare_results.py`; `worker.py` is internal |
| `localization_networks/validation/` | Runtime/model, dataset, fixture, and report checks | `validate.py quick/datasets` |
| `visualization/` | Result discovery, consistency checks, figures, and provenance | `plot_results.py` |

## Naming and Shared Logic

Use lowercase snake_case for modules and canonical `scripts.<pack>.*` imports.
Entry scripts add the repository root once; internal modules use package imports.
Do not add aliases for removed code roots. Model modules are `complex_cnn.py`,
`real_cnn.py`, and `resnet.py`; registry names remain method-specific.

`scripts/pack_cli.py` owns argument parsing; `scripts/packs.py` owns the
pack/method/network registry. `scripts/run_identity.py` owns task IDs and filter
identity; `scripts/run_monitor.py` follows logs and retries transport failures
without resubmission. `run.sh` locates the root and launches the standard-library
dispatcher through uv.

Within each pack, `validation/quick.py` checks runtime/models, `datasets.py`
checks generated data, and `reporting.py` writes reports. In base,
`workflow/local_grnn.py` owns local generation/execution/delivery;
`workflow/compare_results.py` reads saved evidence using only the standard
library and never launches a model.

## Add a Model

1. Add a module and its dataclass config to the method's `network/models/`.
2. Register the config/module pair in `network/model.py:MODEL_REGISTRY` and
   define its loader-normalization policy in `MODEL_DATASET_INPUT_NORM`.
3. RBD configs accept `input_elements` plus `input_freq_bins`; ELM/SCM accept
   `input_pairs` plus `input_freq_bins`. Return normalized predictions shaped
   `[batch,1]`. Shared training handles target statistics and km metrics.
4. Check checkpoint config serialization/restoration. Update the four-model
   registry-count expectation in `validation/quick.py` and the public
   architecture mapping/plans in `workflow/tasks.py` when adding a family.
   Update `scripts/packs.py` when adding a public method or network family.
   The quick suite checks those mappings against the actual registries;
   cover the new model with a focused remote forward/backward check.

Existing families are described in [model architecture](../methods/trainable-model-architecture.md).
GRNN stores method-specific reference samples and is not a trainable
`nn.Module`; keep it outside these registries.

GRNN feature selection uses explicit `--method RBD|ELM|SCM`; `grnn/bindings.py`
owns canonical method/model/artifact identities. The common GRNN loader reuses
each method's HDF5 dataset with neural normalization disabled, then applies
the requested GRNN preprocessing and preserves CHW flatten order.

## Add or Change a Loader

Extend the method's `network/data.py`; reuse `common/h5_utils.py` for HDF5
path/glob resolution, axis handling, split indices, and label subsets.
Keep the bundle interface stable:

```text
train_dataset, val_dataset
train_idx, val_idx
train_labels
input_shape
split_mode
source_paths
input_signature
```

Preserve real/imag channel order, valid-label filtering, pair order, and sample
identity/time alignment. Reuse `common/input_contract.py` to compare physical
frequency, element/depth, and pair axes across files and against artifacts.
Matching tensor dimensions alone are insufficient. Add a focused malformed-input
fixture when changing this contract.

Check training, prediction exports, and continuation against the loader.
Continuation rebuilds the split and requires its indices, mode, and seed to
match the checkpoint. It starts a new cosine learning-rate phase; see the
[resume rules](../workflows/training-and-prediction.md#resume-and-checkpoint-lookup).

## Configuration and Validation

Use `common/cli.py`, `training.py`, and `prediction.py` for neural command
behavior. A method's thin wrappers supply `MethodBindings` for its loader,
model registry, paths, and shape field. Keep method-specific layouts out of
the shared loop.

Generation schema changes belong in
`scripts/base_pack/data_processing/generation_config.py`; reject unknown/invalid fields
and keep cache fingerprints/metadata in step with the new behavior.

Run numerical checks and model validation on haiqin through the
[remote workflow](../workflows/remote-validation.md), using the frozen uv
environment. A quick suite checks contracts; use a smoke run when generation,
loading, checkpointing, or result exports change. Update the relevant topic
document and index instead of duplicating commands or environment tables.
Focused tests live beside the code in `tests/`. Keep
`base_pack/data_processing/matlab_reference/` static; do not execute MATLAB or
commit datasets, model artifacts, generated figures, or local environments.
