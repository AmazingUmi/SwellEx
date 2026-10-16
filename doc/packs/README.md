# Experiment Packs

A pack owns generation, input representation, models, workflow, and plots under
`scripts/<pack>/`. Select it explicitly with `./scripts/run.sh PACK ...`.

| Pack | Feature selections | Smoke / full model count | Fusion |
|---|---|---|---|
| [base_pack](#base-pack) | `RBD`, `ELM`, `SCM` | 6 / 15 | None |
| [joint_feature_pack](joint-feature-pack.md) | Three singles, three pairs, and the three-way combination | 14 / 35 | `block_rms` (default) or `raw_concat` |
| [data_scale_pack](data-scale-pack.md) | `RBD`, `ELM`, `SCM` | 6 / 18 | None; training-count-selected complex CNN and fixed controls |

Base and joint support `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50`, and standalone
`GRNN`. A full joint plan evaluates one fusion mode. Pack and model names are
case-sensitive. Commands, option tables, and examples are maintained in the
[command guide](../workflows/remote-validation.md). Data scale supports six
complex CNN choices described in its [pack contract](data-scale-pack.md).

## Base Pack

`scripts/base_pack/` implements independent RBD, ELM, and SCM regressors.
RBD uses element features; ELM uses strict-upper pairs; SCM includes diagonal
pairs. Each route owns its neural models and GRNN reference/sigma search.
Base rejects both `--fusion` and `[fusion]`; joint requires a fusion configuration.

Pack migration changed source paths, not feature definitions or split rules.
Artifact reuse still requires the current checkpoint/reference contract.
See the [HDF5 contract](../datasets/hdf5-datasets-and-splits.md) for inputs,
[model architecture](../methods/trainable-model-architecture.md) for neural
layers, and [GRNN guide](../workflows/standalone-grnn.md) for build/predict.

## Configuration and Results

Use complete presets from the selected pack's `localization_networks/workflow/`.
The [command guide](../workflows/remote-validation.md#advanced-configuration)
lists presets and overrides; its [results section](../workflows/remote-validation.md#results-and-failures)
defines delivery paths. Requests, reports, and manifests retain pack identity;
status/fetch must use the launching pack.

Compare runs with explicit pack, split, feature combination, fusion, and seed.
Historical base results are not matched joint controls. Recorded validation
and remaining work are in [project progress](../overview/project-progress.md).
