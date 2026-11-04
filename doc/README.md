# Documentation Index

Documentation mirrors the three functional layers under `scripts/` (`data_processing/`, `network_training/`, `remote_workflow/`). Quick-start commands live in the root [README](../README.md) and [remote validation](remote_workflow/remote-validation.md); decoupled Colab/local notebooks live in [01_data_processing_cpu.ipynb](../notebooks/01_data_processing_cpu.ipynb) and [02_network_training_gpu.ipynb](../notebooks/02_network_training_gpu.ipynb).

- **Verified results & scientific boundaries**: [project-progress.md](project-progress.md)
- **Reference literature**: [papers.xlsx](papers.xlsx)

## Experiment Packs

Each pack spans isolated generation (`scripts/data_processing/<pack>/`), loaders and models (`scripts/network_training/<pack>/`), and SSH workflow dispatch/plotting (`scripts/remote_workflow/<pack>/`). Select explicitly via `./scripts/run.sh PACK ...` (case-sensitive). Every artifact records pack identity; `status`, `fetch`, and checkpoint/reference loading reject cross-pack reuse.

| Pack | Feature selections | Smoke / full | Data processing | Network training | Remote workflow |
|---|---|---|---|---|---|
| [base_pack](#base-pack) | `RBD`, `ELM`, `SCM` | 6 / 15 | [HDF5 & splits](data_processing/hdf5-datasets-and-splits.md) | [Neural](network_training/trainable-model-architecture.md), [GRNN](network_training/grnn.md) | [Commands](remote_workflow/remote-validation.md), [defaults](remote_workflow/command-defaults.md) |
| `joint_feature_pack` | 3 singles, 3 pairs, `RBD_ELM_SCM` | 14 / 35 | [Joint concatenation](data_processing/joint-features.md) | [Joint scaling & loaders](network_training/joint-feature-pack.md) | [Commands](remote_workflow/remote-validation.md), [figures](remote_workflow/result-visualization.md) |
| `data_scale_pack` | `RBD`, `ELM`, `SCM` | 6 / 18 | [HDF5 & splits](data_processing/hdf5-datasets-and-splits.md) | [Scalable CNN & count rule](network_training/data-scale-pack.md) | [SCM calibration & budget study](remote_workflow/matrix-studies.md#scm-capacity-calibration-and-budget-study) |
| `MoE_data_scale` | `RBD`, `ELM`, `SCM`, `RBD_SCM` | 8 / 20 | [Joint concatenation](data_processing/joint-features.md) | [Residual MoE & routing](network_training/moe-data-scale.md) | [RBD+SCM MoE matrix](remote_workflow/matrix-studies.md#rbd-scm-moe-expert-study) |
| `MoE_learn` | `RBD_SCM` | 1 staged run | [Holdout & staged splits](data_processing/hdf5-datasets-and-splits.md#continual-stage-and-distance-holdout-splits) | [Continual adaptation & growth](network_training/moe-learn.md) | [Commands](remote_workflow/remote-validation.md), [staged TOMLs](remote_workflow/command-defaults.md#continual-moe-staged-defaults) |

### Base Pack

`base_pack` trains independent `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50`, and standalone `GRNN` regressors per feature route (`RBD` element vectors, `ELM` `i<j` pairs, `SCM` `i<=j` pairs) and rejects `--fusion` / `[fusion]`. Compare runs only under matched splits, feature selections, fusion modes, and seeds.

## Data Processing

| Topic | Guide |
|---|---|
| Raw inputs, HDF5 schema, feature math, windows, frequencies, and splits | [HDF5 datasets and split strategies](data_processing/hdf5-datasets-and-splits.md) |
| Bartlett peak selection, sidelobe rejection, and `/rbd` diagnostics | [RBD multipath peak detection](data_processing/rbd-multipath-detection.md) |
| Canonical `RBD → ELM → SCM` row concatenation and `Ns=1` alignment | [Joint feature concatenation](data_processing/joint-features.md) |

## Network Training

| Topic | Guide |
|---|---|
| Three-layer code boundaries, `DatasetBundle` contract, and adding models/loaders | [Python extension guide](network_training/python-extension-guide.md) |
| Complex CNN, magnitude CNN, ResNet regressors, and target normalization | [Trainable model architecture](network_training/trainable-model-architecture.md) |
| Direct neural `train` / `predict` CLI, checkpoints, and resume | [Neural training and prediction](network_training/training-and-prediction.md) |
| Three-route flattened features, Gaussian kernel math, and sigma CV | [Feature representations and GRNN](network_training/grnn.md) |
| Direct GRNN `build` / `predict` CLI and reference artifacts | [Standalone RBD/ELM/SCM GRNN](network_training/standalone-grnn.md) |
| `raw_concat` vs training-fitted `block_rms` scaling on concatenated features | [Joint feature pack](network_training/joint-feature-pack.md) |
| Training-count-selected complex CNN depth and frozen architecture metadata | [Data scale pack](network_training/data-scale-pack.md) |
| Shared complex frontend, GroupNorm residual experts, and Top-2 routing | [MoE data scale pack](network_training/moe-data-scale.md) |
| Stage-1 frozen frontend, grouped LRs, candidate growth, and replay/history guards | [Continual MoE pack](network_training/moe-learn.md) |

## Remote Workflow

| Topic | Guide |
|---|---|
| `run.sh` commands, filters, run IDs, `status`/`fetch`, and `local-grnn` | [Run models and validation on haiqin](remote_workflow/remote-validation.md) |
| Standard workflow TOML parameters and `MoE_learn` staged TOML defaults | [Workflow configuration defaults](remote_workflow/command-defaults.md) |
| Source snapshot, `haiqin` paths, frozen `uv`/CUDA runtime, and dataset cache | [Frozen environment and remote run files](remote_workflow/remote-environment.md) |
| Detached `study` / `budget-study` coordinators, SCM calibration, and MoE matrix | [Controlled matrix studies](remote_workflow/matrix-studies.md) |
| Delivered `output/{data,figure}/` layout, PDF plots, and manual re-rendering | [Result figures and manual plotting](remote_workflow/result-visualization.md) |

## Documentation Rules

- Keep one authority per topic inside its matching layer (`data_processing`, `network_training`, `remote_workflow`); link shared contracts instead of repeating tables.
- Distinguish `run.sh` workflow defaults from direct Python CLI defaults.
- Preserve each document's language; keep run logs in `output/data/` rather than design guides.
