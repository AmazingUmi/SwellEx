# Documentation Index

Documentation mirrors the three functional layers under `scripts/` (`data_processing/`, `network_training/`, `remote_workflow/`). Quick-start commands live in the root [README](../README.md) and [remote validation](remote_workflow/run-and-validate.md); decoupled Colab/local notebooks live in [01_data_processing_cpu.ipynb](../notebooks/01_data_processing_cpu.ipynb) and [02_network_training_gpu.ipynb](../notebooks/02_network_training_gpu.ipynb).

- **Verified results & scientific boundaries**: [project-progress.md](project-progress.md)
- **Reference literature**: [papers.xlsx](papers.xlsx)

## Experiment Pack

Only `base_pack` is active: generation (`scripts/data_processing/`), loaders and models (`scripts/network_training/`), and SSH workflow dispatch/plotting (`scripts/remote_workflow/`). Select it via `./scripts/run.sh base_pack ...`. Every artifact records pack identity; `status`, `fetch`, and checkpoint/reference loading reject cross-pack reuse.

| Pack | Feature selections | Networks | Smoke / full | Data processing | Network training | Remote workflow |
|---|---|---|---|---|---|---|
| `base_pack` | `RBD`, `SCM` | `complex_CNN`, `ResNet18`, `ResNet50`, `GRNN` | 4 / 8 | [HDF5 & splits](data_processing/hdf5-datasets-and-splits.md) | [Neural](network_training/neural-models.md), [GRNN](network_training/grnn-method.md) | [Commands](remote_workflow/run-and-validate.md), [defaults](remote_workflow/workflow-config.md) |

`base_pack` trains independent neural regressors per feature route (`RBD` element vectors, `SCM` `i<=j` pairs) plus a standalone `GRNN`. Compare runs only under matched splits, feature selections, and seeds. The ELM feature, `real_CNN`, joint-feature, data-scale and MoE experiments (code, documents and results) are archived: see the [archive index](../archive/README.md).

## Data Processing

| Topic | Guide |
|---|---|
| Raw inputs, HDF5 schema, feature math, windows, frequencies, and splits | [HDF5 datasets and split strategies](data_processing/hdf5-datasets-and-splits.md) |
| Bartlett peak selection, sidelobe rejection, and `/rbd` diagnostics | [RBD multipath peak detection](data_processing/rbd-multipath-detection.md) |

## Network Training

| Topic | Guide |
|---|---|
| Three-layer code boundaries, `DatasetBundle` contract, and adding models/loaders | [Python extension guide](network_training/extension-guide.md) |
| Complex CNN, ResNet regressors, and target normalization | [Trainable model architecture](network_training/neural-models.md) |
| Direct neural `train` / `predict` CLI, checkpoints, and resume | [Neural training and prediction](network_training/neural-cli.md) |
| Two-route flattened features, Gaussian kernel math, and sigma CV | [Feature representations and GRNN](network_training/grnn-method.md) |
| Direct GRNN `build` / `predict` CLI and reference artifacts | [GRNN CLI](network_training/grnn-cli.md) |

## Remote Workflow

| Topic | Guide |
|---|---|
| `run.sh` commands, filters, run IDs, `status`/`fetch`, and `local-grnn` | [Run models and validation on haiqin](remote_workflow/run-and-validate.md) |
| Workflow TOML parameters | [Workflow configuration defaults](remote_workflow/workflow-config.md) |
| Source snapshot, `haiqin` paths, frozen `uv`/CUDA runtime, and dataset cache | [Frozen environment and remote run files](remote_workflow/remote-environment.md) |
| Delivered `output/{data,figure}/` layout (retired results under `archive/output/`), PDF plots, and manual re-rendering | [Result figures and manual plotting](remote_workflow/result-visualization.md) |

## Documentation Rules

- Keep one authority per topic inside its matching layer (`data_processing`, `network_training`, `remote_workflow`); link shared contracts instead of repeating tables.
- Distinguish `run.sh` workflow defaults from direct Python CLI defaults.
- Preserve each document's language; keep run logs in `output/data/` rather than design guides.
