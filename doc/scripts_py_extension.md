# Python Extension Guide

The Python code is intentionally split by feature family:

```text
scripts_py/common/
scripts_py/RBD_method/network/
scripts_py/ELM_method/network/
```

`scripts_py/common/` contains shared path helpers, HDF5 path resolution and
split helpers, range-regression training utilities, and prediction CSV/plot
helpers. Keep method-specific tensor layout logic in the RBD or ELM method
package.

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
to km. It can compute SmoothL1 loss either in normalized target space or
directly in physical kilometers via `--loss-space`.

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

Shared HDF5 helper functions live in:

```text
scripts_py/common/h5_utils.py
```

Use them for path resolution, random train/validation split indices, and label
subsets instead of duplicating that logic inside method-specific loaders.
