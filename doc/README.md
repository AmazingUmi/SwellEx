# Documentation Index

Choose a guide by the task you want to perform. Use
[project progress](overview/project-progress.md) for verified capabilities
and remaining scientific work.

## Workflows

| Task | Guide |
|---|---|
| Run a model, validate, check status, or fetch results | [Shell commands and remote workflow](workflows/remote-validation.md) |
| Check or change the default settings | [Command defaults](workflows/command-defaults.md) |
| Enter a frozen remote run or inspect runtime files | [Frozen environment and run files](workflows/remote-environment.md) |
| Train, resume, or predict with a neural model manually | [Training and prediction](workflows/training-and-prediction.md) |
| Build a GRNN reference or choose sigma manually | [Standalone SCM-GRNN](workflows/standalone-scm-grnn.md) |
| Find figures, understand them, or render a model selection | [Result visualization](workflows/result-visualization.md) |

## Datasets and Development

| Task | Guide |
|---|---|
| Understand generation, HDF5 fields, windows, frequencies, or splits | [HDF5 datasets and split strategies](datasets/hdf5-datasets-and-splits.md) |
| Add a model or change a loader | [Python extension guide](development/python-extension-guide.md) |
| Locate a script, shared module, or method entry point | [Scripts organization](development/scripts-organization.md) |

## Methods

| Topic | Reference |
|---|---|
| CNN/ResNet structure and regression output | [Trainable model architecture](methods/trainable-model-architecture.md) |
| SCM construction, GRNN kernels, and sigma selection | [SCM features and GRNN](methods/scm-grnn.md) |
| RBD peak selection, aggregation, and diagnostics | [RBD multipath peak detection](methods/rbd-multipath-detection.md) |
| Papers and reference tracking | [papers.xlsx](references/papers.xlsx) |

## Documentation Rules

- Keep one authority per topic; link shared details instead of duplicating them.
- Preserve each document's language and update links when moving content.
- Use English lowercase directories and hyphen-separated filenames without
  numeric ordering prefixes. Keep `README.md` for indexes.
- Name files for their subject. Meaningful numbers, such as years, may remain.
- These rules apply to `doc/`; code and generated artifacts have their own contracts.
