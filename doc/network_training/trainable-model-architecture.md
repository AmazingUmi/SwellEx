# Trainable Model Architecture

`base_pack` (and `joint_feature_pack` on concatenated rows) registers four trainable neural families per route (`RBD` unprefixed; `ELM` / `SCM` prefixed with `elm_` / `scm_`; joint prefixed with `joint_`) taking `[batch, 2, rows, frequency]` `(real, imag)` tensors (`GRNN` stays outside trainable registries):

| Shell `--network` | Base registry name | Input normalization & structure |
|---|---|---|
| `complex_CNN` | `complex_cnn_range` | Loader real/imag norm on; 4 two-kernel complex conv blocks (`c,2c,4c,4c`, strides `(1,2),(2,2),(2,2),(1,2)`, `c=16`, separate real/imag BatchNorm + `ComplexModReLU`), magnitude adaptive avg pool, `Linear -> ReLU -> Dropout(0.15) -> Linear` |
| `real_CNN` | `real_cnn_range` | Loader norm off; computes `sqrt(real^2 + imag^2 + 1e-8)` with per-sample z-score (SD floor `1e-6`), 4 real conv blocks + pooled MLP head |
| `ResNet18` / `ResNet50` | `resnet18_range` / `resnet50_range` | Loader norm on; `torchvision` ResNet (`weights=None`) with `Conv2d(2,64,3,stride=(1,2),padding=1)`, `maxpool=Identity`, and dropout + scalar linear head |

Complex convolution applies `(Wr + jWi)*(xr + jxi) = (Wr*xr - Wi*xi) + j(Wr*xi + Wi*xr)`. Pack-specific architectural variants live in [data scale pack](data-scale-pack.md), [MoE data scale](moe-data-scale.md), and [continual MoE](moe-learn.md).

## Regression Output Contract

All trainable models output normalized `[batch, 1]` predictions:

```text
y_norm = (y_range_km - train_mean_km) / train_std_km
pred_range_km = pred_norm * train_std_km + train_mean_km
```

Population mean and SD (`train_std_km` floored to `1.0` if `< 1e-6` or nonfinite) are fitted strictly on optimizer-training labels and persisted in checkpoints. Evaluation metrics (MAE, RMSE) are always reported in km.
