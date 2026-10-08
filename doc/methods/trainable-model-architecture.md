# Trainable Model Architecture

RBD, ELM, and SCM have independent loaders and `network/model.py` registries
under `scripts/localization_networks/<method>/`. Their model families
share the structure below. Input is `[batch,2,element|pair,frequency]`, with
real then imaginary channels; [HDF5 contracts](../datasets/hdf5-datasets-and-splits.md)
define the feature and pair semantics.

| Family | RBD model name | ELM / SCM model name |
|---|---|---|
| Complex CNN | `complex_cnn_range` | Prefix `elm_` / `scm_` |
| Magnitude CNN | `real_cnn_range` | Prefix `elm_` / `scm_` |
| ResNet-18 | `resnet18_range` | Prefix `elm_` / `scm_` |
| ResNet-50 | `resnet50_range` | Prefix `elm_` / `scm_` |

RBD configs use `input_elements` and `input_freq_bins`; ELM/SCM use
`input_pairs` and `input_freq_bins`. Implementations live in each method's
`network/models/` directory. GRNN is a separate
[kernel regressor](scm-grnn.md), outside these registries.

## Complex CNN

Each complex convolution applies two real kernels:

```text
(Wr + jWi) * (xr + jxi)
  = (Wr*xr - Wi*xi) + j(Wr*xi + Wi*xr)
```

Real and imaginary maps have independent batch normalization. ComplexModReLU
gates magnitude while retaining phase. Four blocks use channel widths
`c,2c,4c,4c` and strides `(1,2),(2,2),(2,2),(1,2)`. Defaults are `c=16` and
dropout `0.15`. The head takes final complex magnitudes and adaptive average
pooling, then applies a hidden linear layer, ReLU, dropout, and scalar linear
output.

## Magnitude CNN and ResNet

The real CNN first computes `sqrt(real²+imag²+1e-8)` and normalizes the
magnitude per sample across element/pair and frequency axes, with standard
deviation floored at `1e-6`. Four real convolution blocks feed a pooled MLP
head with the same hidden linear/ReLU/dropout structure. It defaults to bypassing
the loader's real/imag channel normalization; complex CNN and ResNet default to
using that normalization.

ResNet variants use torchvision ResNet-18/50 with `weights=None`. Their first
layer is `Conv2d(2,64,kernel_size=3,stride=(1,2),padding=1)`, max pooling is
replaced by `Identity`, and the final classifier becomes dropout plus a
single-output linear layer. They do not load pretrained image weights.

## Regression Contract

Every model returns normalized range `[batch,1]`:

```text
y_norm = (y_range_km - train_mean_km) / train_std_km
pred_range_km = pred_norm * train_std_km + train_mean_km
```

Fresh training fits statistics on training labels only, using population
standard deviation; non-finite or below-`1e-6` standard deviation is replaced
by 1. Checkpoints preserve these statistics for prediction and continuation.
MAE/RMSE are reported in km. Loss units, CLI options, and continuation behavior
are defined by [training and prediction](../workflows/training-and-prediction.md).
