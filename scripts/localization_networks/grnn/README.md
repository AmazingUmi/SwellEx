# Standalone SCM-GRNN

Use `scripts/localization_networks/grnn/main.py build` to save reference samples
and `predict` to apply Gaussian-kernel regression. This package stays outside
the trainable SCM CNN/ResNet registry and has no epochs or optimizer.

`--data` accepts a dataset name under `outputs/Datasets/` or an existing
directory path. References and predictions default to
`outputs/networks_results/SCM/scm_grnn_range/`. Input axes must match the saved
reference signature; references missing the current contract fields must be rebuilt.

See the [SCM-GRNN workflow](../../../doc/workflows/standalone-scm-grnn.md)
for commands and artifact fields, and the
[method note](../../../doc/methods/scm-grnn.md) for formulas and evaluation limits.
