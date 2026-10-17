# Documentation Index

## Common Commands

Run from the **local project root**. Choose one launch command for the desired pack.

| Task | Command |
|---|---|
| Start all joint models with train-before / test-after split | `./scripts/run.sh joint_feature_pack --preset train_before` |
| Start all base models with the same split | `./scripts/run.sh base_pack --preset train_before` |
| Check residual MoE structure using synthetic CUDA inputs | `./scripts/run.sh MoE_data_scale validate --suite quick` |
| Inspect the RBD+SCM MoE matrix without running it | `./scripts/run.sh MoE_data_scale study --plan scripts/MoE_data_scale/localization_networks/workflow/rbd_scm_expert_study_v1.json --dry-run` |
| Check joint task status | `./scripts/run.sh joint_feature_pack status --run 'RUN_ID'` |
| Fetch joint results | `./scripts/run.sh joint_feature_pack fetch --run 'RUN_ID'` |

Replace `RUN_ID` with the full ID printed after `Run:`. For base tasks, replace
`joint_feature_pack` with `base_pack` in status/fetch; no preset is needed.
Results: `output/data/<run_id>/` and `output/figure/<run_id>/`.
See the [command guide](workflows/remote-validation.md#common-commands) for
selection/defaults and [status details](workflows/remote-validation.md#task-id-status-and-fetch)
for recovery. Use [project progress](overview/project-progress.md) for verified
capabilities and remaining scientific work.

## Experiment Packs

| Task | Guide |
|---|---|
| Choose a pack and understand ownership | [Pack index](packs/README.md) |
| Run the preserved independent-feature scheme | [Base pack](packs/README.md#base-pack) |
| Compare joint inputs with existing architectures | [Joint feature pack](packs/joint-feature-pack.md) |
| Choose complex CNN capacity from training sample count | [Data scale pack](packs/data-scale-pack.md) |
| Configure RBD+SCM data size × residual MoE expert count | [MoE data scale pack](packs/moe-data-scale.md) |
| Calibrate capacity thresholds and compare equal optimizer updates | [SCM controlled-budget study](packs/data-scale-budget-study.md) |

The workflow and method references below describe `base_pack` unless they
explicitly identify another pack. Pack-specific differences belong in `packs/`.

## Workflows

| Task | Guide |
|---|---|
| Run all train-before models or select methods/networks | [Shell commands and remote workflow](workflows/remote-validation.md) |
| Check task status or fetch results after disconnection | [Status and fetch commands](workflows/remote-validation.md#task-id-status-and-fetch) |
| Check TOML parameters, defaults, and constraints | [Configuration defaults](workflows/command-defaults.md) |
| Enter a frozen remote run or inspect runtime files | [Frozen environment and run files](workflows/remote-environment.md) |
| Train, resume, or predict with a neural model manually | [Training and prediction](workflows/training-and-prediction.md) |
| Build a GRNN reference or choose sigma manually | [Standalone RBD/ELM/SCM GRNN](workflows/standalone-grnn.md) |
| Plan local CPU GRNN or merge existing results | [Local workflow and comparison](workflows/remote-validation.md#maintenance-commands) |
| Find figures, understand them, or render a model selection | [Result visualization](workflows/result-visualization.md) |

## Datasets and Development

| Task | Guide |
|---|---|
| Understand generation, HDF5 fields, windows, frequencies, or splits | [HDF5 datasets and split strategies](datasets/hdf5-datasets-and-splits.md) |
| Add a model or change a loader | [Python development guide](development/python-extension-guide.md) |
| Locate a script, shared module, or method entry point | [Directory map](development/python-extension-guide.md#directory-map) |

## Methods

| Topic | Reference |
|---|---|
| CNN/ResNet structure and regression output | [Trainable model architecture](methods/trainable-model-architecture.md) |
| RBD/ELM/SCM features, GRNN kernels, and sigma selection | [Feature representations and GRNN](methods/grnn.md) |
| RBD peak selection, aggregation, and diagnostics | [RBD multipath peak detection](methods/rbd-multipath-detection.md) |
| Papers and reference tracking | [papers.xlsx](references/papers.xlsx) |

## Documentation Rules

- Keep one authority per topic; link shared details instead of duplicating them.
- Use parameter/default/meaning tables and short examples; distinguish shell
  settings from direct Python defaults.
- Preserve each document's language and update links when moving content.
- Use English lowercase directories and hyphen-separated filenames without
  numeric ordering prefixes. Keep `README.md` for indexes.
- Keep run logs in their artifacts; progress pages summarize key evidence and gaps.
