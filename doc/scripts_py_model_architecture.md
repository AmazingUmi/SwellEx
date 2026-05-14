# Model Notes and Network Architecture

The repository currently keeps RBD, ELM, and SCM networks separate:

```text
scripts_py/common/
scripts_py/RBD_method/network/
scripts_py/ELM_method/network/
scripts_py/SCM_method/network/
```

The common package contains shared training and prediction infrastructure. The
method packages keep their own dataset loaders and model registries.

Both methods predict normalized range with shape:

```text
[batch, 1]
```

The training loop converts normalized predictions back to km for metrics. The
training loss can be computed in normalized target space or directly in
kilometers via `--loss-space`.

## RBD Models

RBD input:

```text
x: [batch, 2, element, frequency]
   channel 0 = real(green_freq)
   channel 1 = imag(green_freq)
```

Built-in RBD model names:

- `complex_cnn_range`
- `real_cnn_range`
- `resnet18_range`
- `resnet50_range`

Implementations live in:

```text
scripts_py/RBD_method/network/models/
```

`complex_cnn_range` uses explicit complex convolutions:

```text
(Wr + jWi) * (xr + jxi)
  = (Wr*xr - Wi*xi) + j(Wr*xi + Wi*xr)
```

Complex feature maps use independent real/imaginary batch normalization and a
magnitude-gated activation. Before the regression head, the final complex maps
are converted to magnitude.

`real_cnn_range` computes the complex magnitude first:

```text
mag = sqrt(real^2 + imag^2 + 1e-8)
mag_norm = (mag - sample_mean) / sample_std
```

The ResNet models use torchvision ResNet-18 or ResNet-50 with:

```text
Conv2d(2, 64, kernel_size=3, stride=(1, 2), padding=1)
maxpool = Identity()
fc = Dropout + Linear(..., 1)
```

## ELM Models

ELM HDF5 input:

```text
/X: [sample, pair, frequency, real_imag]
```

The current Python loader reads the strict upper-triangle pair vector:

```text
x: [batch, 2, pair, frequency]
pair = strict upper-triangle element ratio with i < j
```

The current ELM MATLAB dataset computes the pair vector with a least-squares
ratio estimator across snapshots, but the Python tensor layout is unchanged:

```text
ratio(i,j,f)
  = sum_s FFT_i,s(f) * conj(FFT_j,s(f))
    / (sum_s abs(FFT_j,s(f))^2 + floor)
```

Built-in ELM model names:

- `elm_complex_cnn_range`
- `elm_real_cnn_range`
- `elm_resnet18_range`
- `elm_resnet50_range`

Implementations live in:

```text
scripts_py/ELM_method/network/models/
```

The ELM complex, real-magnitude, and ResNet models mirror the RBD model family,
but their configs use:

```text
input_pairs
input_freq_bins
```

instead of:

```text
input_elements
input_freq_bins
```

## SCM Models

SCM HDF5 input:

```text
/X: [sample, pair, frequency, real_imag]
```

The SCM loader reads an upper-triangle pair vector with diagonal:

```text
x: [batch, 2, pair, frequency]
pair = SCM entries with i <= j
```

Built-in SCM model names:

- `scm_complex_cnn_range`
- `scm_real_cnn_range`
- `scm_resnet18_range`
- `scm_resnet50_range`

Implementations live in:

```text
scripts_py/SCM_method/network/models/
```

The SCM model configs also use:

```text
input_pairs
input_freq_bins
```

## Target Normalization

For all methods:

```text
y_norm = (y_range_km - train_mean_km) / train_std_km
pred_range_km = pred_norm * train_std_km + train_mean_km
```

Checkpoints store the target mean/std so prediction can restore the original
range units.

## Loss Space

RBD, ELM, and SCM training commands support:

```text
--loss-space normalized
--loss-space km
```

`normalized` is the original behavior and computes SmoothL1 loss on `pred_norm`
and `y_norm`. `km` computes SmoothL1 on `pred_range_km` and `y_range_km`; in
that mode `--huber-beta` is also measured in kilometers.
