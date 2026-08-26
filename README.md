# SwellEx Range Regression

This repository builds range-regression datasets and PyTorch models for the
SwellEx VLA data. The code currently supports three parallel trainable feature
routes plus a standalone non-trainable regressor:

- RBD: beamformer/RBD-derived Green-function features.
- ELM: direct element pairwise frequency-ratio features without RBD.
- SCM: spatial covariance features (upper-triangle entries including the
  diagonal).
- GRNN: standalone Gaussian-kernel regression on SCM features (reference
  build + kernel prediction, no training).

## Quick Start

Run commands from the project root.

RBD training and prediction:

```bash
python3 scripts_py/RBD_method/Network_main.py train \
  --model complex_cnn_range \
  --data <rbd_dataset_name>

python3 scripts_py/RBD_method/Network_main.py predict \
  --model complex_cnn_range \
  --data <rbd_dataset_name>
```

ELM training and prediction:

```bash
python3 scripts_py/ELM_method/Network_main.py train \
  --model elm_complex_cnn_range \
  --data <elm_dataset_name>

python3 scripts_py/ELM_method/Network_main.py predict \
  --model elm_complex_cnn_range \
  --data <elm_dataset_name>
```

SCM training and prediction:

```bash
python3 scripts_py/SCM_method/Network_main.py train \
  --model scm_complex_cnn_range \
  --data <scm_dataset_name>

python3 scripts_py/SCM_method/Network_main.py predict \
  --model scm_complex_cnn_range \
  --data <scm_dataset_name>
```

Standalone GRNN reference build and prediction (SCM datasets):

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py build \
  --data <scm_dataset_name>

python3 scripts_py/0_GRNN_related/GRNN_main.py predict \
  --data <scm_dataset_name>
```

RBD, ELM, and SCM training all support target-normalized loss and
physical-error loss:

```bash
--loss-space normalized
--loss-space km --huber-beta 0.5
```

Default outputs are written under:

```text
outputs/networks_results/RBD_method/
outputs/networks_results/ELM_method/
outputs/networks_results/SCM_method/
outputs/networks_results/0_GRNN_related/
```

## Code Layout

MATLAB dataset generation:

```text
scripts_matlab/RBD_method/
scripts_matlab/ELM_method/
scripts_matlab/SCM_method/
scripts_matlab/function/
```

Python training and prediction:

```text
scripts_py/common/
scripts_py/RBD_method/
scripts_py/ELM_method/
scripts_py/SCM_method/
scripts_py/0_GRNN_related/
```

`scripts_py/common/` contains shared path, HDF5 split, training, and prediction
helpers. RBD, ELM, and SCM keep separate dataset loaders and model registries
because their HDF5 layouts or pair semantics differ. The standalone GRNN lives
outside the trainable-network packages.

Built-in RBD models:

- `complex_cnn_range`
- `real_cnn_range`
- `resnet18_range`
- `resnet50_range`

Built-in ELM models:

- `elm_complex_cnn_range`
- `elm_real_cnn_range`
- `elm_resnet18_range`
- `elm_resnet50_range`

Built-in SCM models:

- `scm_complex_cnn_range`
- `scm_real_cnn_range`
- `scm_resnet18_range`
- `scm_resnet50_range`

GRNN is not a registered trainable model; use
`scripts_py/0_GRNN_related/GRNN_main.py` with the `build` and `predict`
subcommands.

## Documentation Index

- [Documentation index by category](doc/README.md)
- [Project progress](doc/00_overview/00_project_progress.md)
- [HDF5 datasets and MATLAB split strategies](doc/10_datasets/00_hdf5_datasets_and_splits.md)
- [Training workflow](doc/20_python_workflows/00_trainable_training.md)
- [Prediction workflow](doc/20_python_workflows/10_trainable_prediction.md)
- [Standalone SCM-GRNN workflow](doc/20_python_workflows/20_standalone_scm_grnn.md)
- [Local toolchain and environment](doc/00_overview/10_environment.md)
