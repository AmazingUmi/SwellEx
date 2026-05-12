# SwellEx Range Regression

This repository builds range-regression datasets and PyTorch models for the
SwellEx VLA data. The code currently supports two parallel feature routes:

- RBD: beamformer/RBD-derived Green-function features.
- ELM: direct element pairwise frequency-ratio features without RBD.

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

Both methods support target-normalized loss and physical-error loss:

```bash
--loss-space normalized
--loss-space km --huber-beta 0.5
```

Default outputs are written under:

```text
outputs/networks_results/RBD_method/
outputs/networks_results/ELM_method/
```

## Code Layout

MATLAB dataset generation:

```text
scripts_matlab/RBD_method/
scripts_matlab/ELM_method/
scripts_matlab/function/
```

Python training and prediction:

```text
scripts_py/common/
scripts_py/RBD_method/
scripts_py/ELM_method/
```

`scripts_py/common/` contains shared path, HDF5 split, training, and prediction
helpers. RBD and ELM keep separate dataset loaders and model registries because
their HDF5 layouts differ.

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

## Documentation Index

- [Local Toolchain and Environment](doc/scripts_environment.md)
- [HDF5 datasets and MATLAB split strategies](doc/scripts_py_datasets.md)
- [RBD multipath peak detection](doc/rbd_multipath_detection.md)
- [Training workflow](doc/scripts_py_training.md)
- [Prediction workflow](doc/scripts_py_prediction.md)
- [Shared interfaces and extension guide](doc/scripts_py_extension.md)
- [Model notes and network architecture](doc/scripts_py_model_architecture.md)
- [Project progress](doc/project_progress.md)
