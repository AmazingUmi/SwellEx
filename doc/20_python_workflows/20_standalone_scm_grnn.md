# Standalone SCM-GRNN Workflow

GRNN is kept outside the trainable RBD/ELM/SCM network packages because it does
not use epochs, backpropagation, optimizers, or learned weights. It builds a
reference artifact by memorizing SCM features and range labels, then predicts
with Gaussian-kernel weighted averaging.

Implementation:

```text
scripts_py/0_GRNN_related/
  GRNN_main.py
  grnn_cli.py
  grnn_data.py
  grnn_model.py
  grnn_paths.py
```

## Compatible Input

The standalone loader is compatible with existing MATLAB-exported SCM HDF5
datasets:

```text
/X: 4-D SCM upper-triangle pair vector with a real/imag axis
/y_range_km: source range labels in km
```

The loader infers sample, pair, frequency, and real/imag axes from shape and
metadata, matching the SCM network loader. The GRNN feature vector is:

```text
flatten([real, imag] x pair x frequency)
feature_dim = 2 * pair_count * frequency_count
```

For a 21-element SCM upper triangle with diagonal:

```text
pair_count = 21 * 22 / 2 = 231
feature_dim = 462 * Q
```

## Build A Reference Artifact

Use `build` instead of `train`:

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py build \
  --data <scm_dataset_name> \
  --spread 0.01
```

This writes:

```text
outputs/networks_results/0_GRNN_related/scm_grnn_range/reference_outputs/
  <dataset>_reference_MMDD_HHMMSS.pt
```

The artifact stores:

```text
X_reference
y_reference_km
input_shape
feature_dim
spread
optional feature mean/std
source HDF5 paths
```

## Select Sigma

Use k-fold cross-validation on the reference set:

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py build \
  --data <scm_dataset_name> \
  --cv-folds 5
```

Or use an explicit validation HDF5 file:

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py build \
  --data <scm_dataset_name> \
  --val-data "outputs/Datasets/<dataset>/*_val.h5"
```

The default sigma candidates follow the Wang and Peng GRNN note:

```text
0.01, 0.02, ..., 0.10, 0.20, 0.30, ..., 2.00
```

Override them with:

```bash
--sigma-candidates 0.005 0.01 0.02 0.05 0.1
```

## Predict

Use the newest reference artifact for the dataset:

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py predict \
  --data <scm_dataset_name>
```

Or pass a specific artifact:

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py predict \
  --data <scm_dataset_name> \
  --reference outputs/networks_results/0_GRNN_related/scm_grnn_range/reference_outputs/<file>.pt
```

Prediction outputs:

```text
outputs/networks_results/0_GRNN_related/scm_grnn_range/test_outputs/
  <dataset>_predictions_MMDD_HHMMSS.csv
  <dataset>_range_prediction_MMDD_HHMMSS.png
```

CSV fields:

```text
sample
source_segment_idx
window_center_s
pred_range_km
true_range_km
abs_error_km
```

Use `--no-plot` to skip plot generation.

## Useful Options

```bash
--standardize-input          # z-score each feature coordinate using reference stats
--per-sample-channel-norm    # normalize real/imag channels per sample before flattening
--batch-size 128             # prediction distance batch size
--device cuda                # or cpu
--max-reference-samples N    # smoke tests and debugging
--max-predict-samples N      # smoke tests and debugging
```

## Conceptual Difference From CNN/ResNet

Trainable networks write checkpoints with learned weights, optimizer state,
schedulers, and target-normalization metadata. GRNN writes a reference artifact
containing the reference feature matrix and labels. Prediction complexity scales
with:

```text
O(N_reference * feature_dim)
```
