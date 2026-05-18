# Standalone SCM-GRNN

This folder keeps the GRNN workflow separate from the trainable SCM CNN/ResNet
network package. GRNN does not run epochs or backpropagation; it builds a
reference artifact by memorizing SCM features and source ranges, then predicts
with Gaussian-kernel weighted averaging.

Build a reference artifact:

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py build \
  --data <scm_dataset_name> \
  --spread 0.01
```

Build with k-fold sigma selection:

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py build \
  --data <scm_dataset_name> \
  --cv-folds 5
```

Predict with the latest reference artifact:

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py predict \
  --data <scm_dataset_name>
```

The HDF5 input remains compatible with existing SCM datasets:

```text
/X          4-D SCM upper-triangle pair-vector with real/imag axis
/y_range_km source range labels in km
```

The loader infers sample, pair, frequency, and real/imag axes from the dataset
shape and metadata, matching the previous SCM HDF5 layout.
