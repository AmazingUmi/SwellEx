# Python Extension Guide

Network packages live under `scripts/localization_networks/<METHOD>_method/`.
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
   registry-count expectation in `validation/validate.py` and the hardcoded
   experiment model/expected-phase lists in `workflow/worker.py` when adding
   a family; then cover the new model with a focused remote forward/backward check.

Existing families are described in [model architecture](../methods/trainable-model-architecture.md).
The current SCM-GRNN stores reference samples and is not a trainable
`nn.Module`; keep it outside these registries.

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
```

Preserve real/imag channel order, valid-label filtering, pair order, and sample
identity/time alignment. A tensor shape alone does not prove compatible
frequency or pair semantics. Add a small fixture that distinguishes the
changed behavior, including rejection of malformed inputs when relevant.
Check training, prediction exports, and continuation against the new loader;
continuation rebuilds the split from current inputs/options.

## Configuration and Validation

Add common CLI behavior in shared helpers where possible; keep method parsers
consistent. Generation schema changes belong in
`scripts/data_processing/generation_config.py`; reject unknown/invalid fields
and keep cache fingerprints/metadata in step with the new behavior.

Run numerical checks and model validation on haiqin through the
[remote workflow](../workflows/remote-validation.md), using the frozen uv
environment. A quick suite checks contracts; use a smoke run when generation,
loading, checkpointing, or result exports change. Update the relevant topic
document and index instead of duplicating commands or environment tables.
