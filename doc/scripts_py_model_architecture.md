# Model Notes and Network Architecture

## Complex Model Notes

The network uses explicit complex convolutions:

```text
(Wr + jWi) * (xr + jxi)
  = (Wr*xr - Wi*xi) + j(Wr*xi + Wi*xr)
```

Complex feature maps pass through batch normalization and a magnitude-gated
activation. Before the final MLP head, real and imaginary maps are converted to
magnitude features for scalar range regression. Targets are normalized with the
training-set mean and standard deviation, while reported metrics are in km.

## Complex Network Architecture

The model implementation is in `network/models/model_complex_cnn_range.py`:

- `ComplexRangeCNN`: full range-regression model
- `ComplexConvBlock`: complex convolution + complex batch normalization + modReLU
- `ComplexConv2d`: complex convolution represented by two real `Conv2d` layers
- `ComplexBatchNorm2d`: independent batch normalization for real and imaginary maps
- `ComplexModReLU`: magnitude-gated activation that preserves complex phase

Input samples are loaded as real and imaginary channels:

```text
x: [batch, 2, element, frequency]
   channel 0 = real
   channel 1 = imaginary
```

The complex layers treat this as one complex input channel. Each complex feature
channel is stored as two real tensor channels, so an internal complex width of
`C` appears in PyTorch tensors as `2*C` channels.

With the default `--base-channels 16`, the feature extractor is:

```text
Input: [B, 2, E, F]

Block 1: ComplexConv2d  1 -> 16, kernel 3x3, stride (1, 2), padding 1
         ComplexBatchNorm2d(16)
         ComplexModReLU(16)
         Output complex shape: [B, 16, E, ceil(F/2)]

Block 2: ComplexConv2d 16 -> 32, kernel 3x3, stride (2, 2), padding 1
         ComplexBatchNorm2d(32)
         ComplexModReLU(32)
         Output complex shape: [B, 32, ceil(E/2), ceil(F/4)]

Block 3: ComplexConv2d 32 -> 64, kernel 3x3, stride (2, 2), padding 1
         ComplexBatchNorm2d(64)
         ComplexModReLU(64)
         Output complex shape: [B, 64, ceil(E/4), ceil(F/8)]

Block 4: ComplexConv2d 64 -> 64, kernel 3x3, stride (1, 2), padding 1
         ComplexBatchNorm2d(64)
         ComplexModReLU(64)
         Output complex shape: [B, 64, ceil(E/4), ceil(F/16)]
```

After the complex feature extractor:

```text
1. Split output into real and imaginary maps.
2. Convert to magnitude: sqrt(real^2 + imag^2 + 1e-8).
3. AdaptiveAvgPool2d((1, 1)).
4. Flatten.
5. Linear(64, 64).
6. ReLU.
7. Dropout(p=0.15 by default).
8. Linear(64, 1).
```

The final scalar is the normalized range prediction. During training, labels are
normalized as:

```text
y_norm = (y_range_km - train_mean_km) / train_std_km
```

During reporting and prediction, the model output is converted back to km:

```text
pred_range_km = pred_norm * train_std_km + train_mean_km
```

## Built-in Real Magnitude Model

`real_cnn_range` receives the same dataset tensor shape as the complex model:

```text
x: [batch, 2, element, frequency]
```

Inside `model_real_cnn_range.py`, it computes:

```text
mag = sqrt(real^2 + imag^2 + 1e-8)
mag_norm = (mag - sample_mean) / sample_std
```

Then `mag_norm` is passed to a real-valued CNN. Because this model performs
magnitude normalization internally, the training pipeline automatically disables
the dataset-level real/imaginary channel normalization for `real_cnn_range`.
