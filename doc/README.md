# Documentation Index

The documentation is organized by numbered subdirectories so the order is
stable in file browsers and each area has an obvious home.

## 00 Overview

- [00 Project progress](00_overview/00_project_progress.md): current repository
  status, completed work, and next steps.
- [10 Environment](00_overview/10_environment.md): Python, MATLAB, and local
  smoke-test commands.

## 10 Datasets

- [00 HDF5 datasets and split strategies](10_datasets/00_hdf5_datasets_and_splits.md):
  MATLAB dataset generation routes, split strategies, frequency selection, and
  HDF5 fields.
- [10 SCM feature note](10_datasets/10_scm_feature_note.txt): concise SCM
  feature-construction sketch.

## 20 Python Workflows

- [00 Trainable training](20_python_workflows/00_trainable_training.md):
  CNN/ResNet training for RBD, ELM, and SCM.
- [10 Trainable prediction](20_python_workflows/10_trainable_prediction.md):
  checkpoint-based prediction for trained RBD, ELM, and SCM networks.
- [20 Standalone SCM-GRNN](20_python_workflows/20_standalone_scm_grnn.md):
  non-iterative GRNN reference building and prediction.

## 30 Methods

- [00 Trainable model architecture](30_methods/00_trainable_model_architecture.md):
  tensor layouts and trainable model families.
- [10 GRNN Wang and Peng 2018 architecture](30_methods/10_grnn_wang_peng_2018_architecture.md):
  paper-derived GRNN structure and normalized SCM input design.
- [20 RBD multipath peak detection](30_methods/20_rbd_multipath_detection.md):
  multipath Bartlett-peak selection options and behavior.

## 40 Extension

- [00 Python extension guide](40_extension/00_python_extension_guide.md): how to
  add trainable models or dataset loaders to the RBD/ELM/SCM network packages.

## 90 References

- [papers.xlsx](90_references/papers.xlsx): paper and reference tracking
  spreadsheet.
