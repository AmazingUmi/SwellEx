# MoE Data Scale Architecture and Routing

`scripts/network_training/MoE_data_scale/` varies the count of independent residual experts (`E1/E2/E4/E8` or uncalibrated auto `complex_MoE`) behind a fixed complex frontend on `RBD`, `ELM`, `SCM`, or concatenated `RBD_SCM` inputs (`21` RBD + `231` SCM rows; see [joint concatenation](../data_processing/joint-features.md)). GRNN and `--fusion` overrides are rejected. Launch and 48-cell matrix study commands live in [controlled matrix studies](../remote_workflow/matrix-studies.md#rbd-scm-moe-expert-study).

## Shared Frontend, Residual Experts, and Top-2 Dispatch

The shared complex frontend uses channels `c, 2c` (`c=16`) with strides `(1,2), (2,2)`. Each expert owns an independent `2c -> 4c` stride-`(2,1)` projection, a constant-width identity residual block (`x + F(x)` with stride-1 complex convolutions, separate real/imag GroupNorm, and `ComplexModReLU`), magnitude pooling, and a `4c -> 4c -> 1` real head:

| Experts (`--network`) | Total parameters (`c=16`) | Per-sample active parameters | Routing behavior |
|---|---:|---:|---|
| 1 (`complex_MoE_e1`) | 199,185 | 199,185 | Bypasses router |
| 2 (`complex_MoE_e2`) | 388,692 | 388,692 | Top-2 evaluates both experts |
| 4 (`complex_MoE_e4`) | 767,640 | 388,758 | Sparse Top-2 dispatch, normalized weights |
| 8 (`complex_MoE_e8`) | 1,525,536 | 388,890 | Sparse Top-2 dispatch, normalized weights |

The router pools shared-feature magnitudes into `Linear(2c, E)`; no samples are dropped. Full pack runs evaluate all 5 networks × 4 methods (20 models); smoke runs `E1` and `E4` (8 models); `validate --suite quick` checks CUDA structure on synthetic inputs without launching a matrix.

## Subset-Fitted Scales, Count Rule, and Diagnostics

For `RBD_SCM`, `block_rms` scales are fitted **only after retaining the optimizer-training subset** and persisted for validation, prediction, and resume.

```toml
[scaling]
reference_samples = 600
reference_experts = 2
exponent = 0.5
max_experts = 8

[moe]
balance_loss_weight = 0.01
```

Auto `complex_MoE` resolves `reference_experts * (N / reference_samples) ** exponent` down to `{1, 2, 4, 8}` before optimization using valid unique training rows `N` (`adapt` frequency mode is rejected) and freezes expert count in checkpoints. Training minimizes SmoothL1 regression plus `balance_loss_weight` load-balance loss while selecting best checkpoints strictly by validation RMSE; histories record regression loss, balance loss, router entropy, and expert loads separately. Completed matrix results are summarized in [project progress](../project-progress.md).
