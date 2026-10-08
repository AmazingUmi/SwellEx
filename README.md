# SwellEx Range Regression

Range regression on SwellEx-96 vertical line array data. Python generates
HDF5 features; PyTorch CNN/ResNet models predict source range in km. Three
feature routes have separate loaders and model registries:

- **RBD**: beamformer-derived equivalent Green features.
- **ELM**: element-pair least-squares frequency ratios, excluding the diagonal.
- **SCM**: normalized spatial covariance entries, including the diagonal.

Standalone **SCM-GRNN** stores reference samples and predicts by Gaussian
kernel regression, without epochs or an optimizer.

## Local Development, Remote Execution

Edit source and configuration locally. Dataset generation, numerical checks,
training, prediction, and plotting run on SSH host `haiqin`. Dependencies are
managed by `pyproject.toml` and `uv.lock`; MATLAB source is retained as a static
migration reference.

From the local project root:

```bash
./scripts/run.sh RBD complex_CNN
./scripts/run.sh SCM GRNN
./scripts/run.sh validate --suite smoke
```

The selected-model command validates or reuses the aligned RBD/ELM/SCM datasets,
then trains (or builds GRNN), predicts, plots, and retrieves only the requested
model. `METHOD` is `RBD`, `ELM`, or `SCM`; `ARCH` is exactly `complex_CNN`,
`real_CNN`, `ResNet18`, `ResNet50`, or `GRNN` (SCM only). Optional
`--config PATH`, `--epochs N`, and `--seed N` override the frozen run settings;
GRNN rejects `--epochs`. Defaults retain S5 window starts 0–59 s and two neural
epochs, which establish pipeline execution without an accuracy acceptance
threshold.

The [remote workflow](doc/workflows/remote-validation.md) explains all command
options, prerequisites, frozen environments, detached runs, and retrieval. Local logs,
reports, history, and prediction CSVs go to `output/data/<run_id>/`; figures go
to `output/figure/<run_id>/`. Remote datasets and model artifacts stay on haiqin
under the run's `outputs/` directory.

## Layout and Documentation

| Path | Purpose |
|---|---|
| `scripts/data_processing/` | Generation, feature math, configuration, numerical oracles |
| `scripts/localization_networks/` | Shared helpers, RBD/ELM/SCM models, GRNN, remote workflow |
| `scripts/visualization/` | Run-based plots and result integrity checks |
| `Origindata/` | Original experiment inputs; excluded from Git |
| `simulation/` | Propagation model inputs |
| `doc/` | Topic-based guides and references |

Start with the [documentation index](doc/README.md) or
[project progress](doc/overview/project-progress.md). Operation guides cover
[training and prediction](doc/workflows/training-and-prediction.md),
[SCM-GRNN](doc/workflows/standalone-scm-grnn.md), and
[result visualization](doc/workflows/result-visualization.md).
