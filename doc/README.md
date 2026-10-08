# Documentation Index

Start with project progress for verified scope, then select the guide for your
task. Commands and environment setup belong to workflows; algorithms and
stored data contracts have separate references.

## Naming Rules

- Use English lowercase topic directories and hyphen-separated filenames
  (`kebab-case`), with no numeric ordering prefixes. Keep `README.md` for indexes.
- Name each file for its subject; meaningful numbers such as publication years
  may remain. The index provides reading order.
- Keep one authority per topic, link shared details, and update links after
  merges or moves. Preserve each document's language.
- These rules apply to `doc/`; code and generated artifacts follow their own
  naming contracts.

## Overview

- [Project progress](overview/project-progress.md): completed capabilities,
  latest validation evidence, and remaining scientific work.

## Datasets

- [HDF5 datasets and split strategies](datasets/hdf5-datasets-and-splits.md):
  generation inputs, stored axes/fields, windows, frequencies, and splits.

## Workflows

- [Local development and remote validation](workflows/remote-validation.md):
  haiqin environment, source snapshots, suites, detached runs, and retrieval.
- [Training and prediction](workflows/training-and-prediction.md):
  RBD/ELM/SCM commands, loss units, continuation, checkpoints, and exports.
- [Standalone SCM-GRNN](workflows/standalone-scm-grnn.md):
  reference building, sigma selection, prediction, and artifacts.
- [Result visualization](workflows/result-visualization.md):
  figure definitions, comparison checks, and delivery provenance.

## Methods

- [Trainable model architecture](methods/trainable-model-architecture.md):
  CNN/ResNet families, input processing, and regression output.
- [SCM features and GRNN](methods/scm-grnn.md): covariance construction,
  vectorization, kernel regression, numerical behavior, and evaluation limits.
- [RBD multipath peak detection](methods/rbd-multipath-detection.md):
  peak selection, sidelobe rejection, Green aggregation, and diagnostics.

## Development

- [Python extension guide](development/python-extension-guide.md):
  adding models or loaders without changing shared contracts.

## References

- [papers.xlsx](references/papers.xlsx): paper and reference tracking.
