# Standalone RBD/ELM/SCM GRNN

Use `scripts/localization_networks/grnn/main.py build --method RBD|ELM|SCM` to save
reference samples and `predict --method RBD|ELM|SCM` to apply Gaussian-kernel regression. This package stays outside
all trainable CNN/ResNet registries and has no epochs or optimizer.

`--data` accepts a dataset name under `outputs/Datasets/` or an existing
directory path. References and predictions default to
`outputs/networks_results/<METHOD>/<method>_grnn_range/`. Input method and axes must match the saved
reference signature; references missing the current contract fields must be rebuilt.

See the [GRNN workflow](../../../doc/workflows/standalone-grnn.md)
for commands and artifact fields, and the
[method note](../../../doc/methods/grnn.md) for formulas and evaluation limits.
