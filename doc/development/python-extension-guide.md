# Python Extension Guide

Neural packages are `scripts/localization_networks/rbd/`, `elm/`, and `scm/`;
standalone RBD/ELM/SCM GRNN lives in `grnn/`. Each has one `main.py` entry point.
Keep shared path, split, training, and prediction logic in `common/`, and
method-specific tensor layouts in each package. Read the
[HDF5 contract](../datasets/hdf5-datasets-and-splits.md) before loader changes;
ELM and SCM have the same tensor rank but different pair semantics.

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
the shared loop. The [scripts map](scripts-organization.md) lists module ownership.

Generation schema changes belong in
`scripts/data_processing/generation_config.py`; reject unknown/invalid fields
and keep cache fingerprints/metadata in step with the new behavior.

Run numerical checks and model validation on haiqin through the
[remote workflow](../workflows/remote-validation.md), using the frozen uv
environment. A quick suite checks contracts; use a smoke run when generation,
loading, checkpointing, or result exports change. Update the relevant topic
document and index instead of duplicating commands or environment tables.
