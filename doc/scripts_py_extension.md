# Python Extension Guide

The Python code is intentionally split by feature family:

```text
scripts_py/RBD_method/network/
scripts_py/ELM_method/network/
```

Use the RBD method when the HDF5 input is:

```text
[sample, element, frequency, real_imag]
```

Use the ELM method when the HDF5 input is:

```text
[sample, numerator_element, denominator_element, frequency, real_imag]
```

## Add A Model

Each method has its own registry:

```text
scripts_py/RBD_method/network/model.py
scripts_py/ELM_method/network/model.py
```

Add the implementation under the matching `models/` directory, then register
the config class and module class in `MODEL_REGISTRY`.

RBD configs use:

```python
input_elements: int
input_freq_bins: int
```

ELM flat-pair configs use:

```python
input_pairs: int
input_freq_bins: int
```

All models should return normalized range predictions with shape:

```text
[batch, 1]
```

The shared training loop handles target normalization and converts metrics back
to km.

## Add A Dataset Format

For RBD, extend:

```text
scripts_py/RBD_method/network/data.py
```

For ELM, extend:

```text
scripts_py/ELM_method/network/data.py
```

Keep the `DatasetBundle` fields stable:

```text
train_dataset
val_dataset
train_idx
val_idx
train_labels
input_shape
split_mode
source_paths
```

ELM additionally tracks:

```text
pair_grid_shape
```

This lets training, checkpoint resume, prediction CSV export, and plotting stay
unchanged.
