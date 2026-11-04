# Data Scale Pack Architecture and Capacity Rule

`scripts/network_training/data_scale_pack/` selects complex CNN depth from valid unique optimizer-training sample count before weight initialization on independent `RBD`, `ELM`, or `SCM` inputs. It has no GRNN or feature fusion, and its default count rule is **uncalibrated**. Launch commands live in [remote validation](../remote_workflow/remote-validation.md); multi-seed calibration and equal-update experiments live in [controlled matrix studies](../remote_workflow/matrix-studies.md#scm-capacity-calibration-and-budget-study).

## Scalable Complex CNN Profiles

| `--network` | Stage repetitions | Role |
|---|---|---|
| `complex_CNN` | Resolved before initialization | Automatic training-count rule (`train-count-v1`) |
| `complex_CNN_small` / `medium` / `large` / `xlarge` | `[1,1,1]` / `[1,2,1]` / `[2,2,2]` / `[2,3,3]` | Fixed 3 / 4 / 6 / 8 blocks |
| `complex_CNN_baseline` | Original 4-block layout | Architecture control (`c,2c,4c,4c`, strides `(1,2),(2,2),(2,2),(1,2)`) |

Scalable stages use channels `c, 2c, 4c` (`c=16`), entry strides `(1,2), (2,2), (2,1)`, and constant stride `(1,1)` on repeated blocks without downsampling or residual shortcuts. Each block applies two-kernel complex convolution, separate real/imag BatchNorm, and `ComplexModReLU`, followed by magnitude pooling and a two-layer real head returning normalized `[batch, 1]`.

## Training Count and Subsampling Contract

`N` counts unique valid optimizer-training windows after validation splitting and optional `train_fraction` subsampling (epochs, batch repeats, and validation/test rows never increase `N`):

```text
budget = clip(P_medium * (N / reference_samples) ** exponent, P_small, P_xlarge)
profile = largest scalable profile with trainable_params <= budget
```

```toml
# Custom TOML overrides (--config PATH):
# Inside [training]:
train_fraction = 0.5          # (0, 1], retaining >= 2 rows

[scaling]
reference_samples = 600       # positive integer (uncalibrated default)
exponent = 0.5                # finite positive float
```

Validation splitting happens first and stays fixed across `train_fraction` values for a given seed; remaining training candidates are time-sorted into 32-row blocks and prefix-sampled after a seeded block shuffle. Adaptive frequency selection (`adapt`) is rejected to prevent pre-subsample leakage; inherited whole-recording mean centering remains a documented boundary.

## Frozen Architecture Metadata

Before step 1, `*_architecture_MMDD_HHMMSS.json` freezes the resolved profile, stage block counts, parameter budget, and subset SHA256 identities. Resume and prediction restore the persisted profile unchanged rather than recomputing capacity from test size.
