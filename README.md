# SwellEx Range Regression

Predict source range in km from SwellEx-96 vertical line array data. Python
creates HDF5 features; PyTorch CNN/ResNet models or standalone SCM-GRNN perform
regression.

| Feature route | Input features |
|---|---|
| RBD | Beamformer-derived equivalent Green estimates |
| ELM | Element-pair least-squares frequency ratios, excluding the diagonal |
| SCM | Normalized spatial covariance entries, including the diagonal |

## Run a Model

Edit code and configuration locally. Generation, numerical tests, training,
prediction, and plotting run on SSH host `haiqin`. MATLAB source is retained
only as a static reference.

From the local project root, with uv, Python 3.11+, Git, SSH, and rsync available:

```bash
# Run one neural model through generation, training, prediction, and plotting
./scripts/run.sh ELM complex_CNN

# Run SCM-GRNN through reference building, prediction, and plotting
./scripts/run.sh SCM GRNN

# Check the full pipeline with three complex CNNs and SCM-GRNN
./scripts/run.sh validate --suite smoke
```

Choose `RBD`, `ELM`, or `SCM` with `complex_CNN`, `real_CNN`, `ResNet18`, or
`ResNet50`. `GRNN` is available only for SCM and has no epochs or optimizer.
A model command prepares all three feature routes and evaluates the chosen model.

Bundled defaults: the complete 4500 s S5 VLA recording (window starts 0–4499 s),
100 neural epochs, A5000-tuned batch size 64, seed 2026, and CUDA. SCM-GRNN uses
five-fold sigma search. The periodic 4:1 split is a within-record baseline; independent
time/track generalization and accuracy acceptance require separate experiments.
See the [command guide](doc/workflows/remote-validation.md) for options,
validation suites, background runs, and failure recovery; see
[command defaults](doc/workflows/command-defaults.md) for all settings.

## Find Results

Foreground runs retrieve results automatically when they finish:

| Location | Contents |
|---|---|
| `output/data/<run_id>/` | Logs, reports, prediction CSVs, history, and provenance |
| `output/figure/<run_id>/` | PDF figures |
| Remote run's `project/outputs/` | Datasets, model checkpoints, and GRNN references |

For figure meanings and manual plotting, see
[result visualization](doc/workflows/result-visualization.md).

## Project Layout

| Path | Purpose |
|---|---|
| `scripts/data_processing/` | Generation, feature math, and numerical checks |
| `scripts/localization_networks/` | RBD/ELM/SCM models, GRNN, shared helpers, and remote workflow |
| `scripts/visualization/` | Plots and result integrity checks |
| `Origindata/` | Original inputs; excluded from Git |
| `simulation/` | Propagation model inputs |
| `doc/` | Workflow guides, data contracts, and method references |
| `pyproject.toml`, `uv.lock` | Dependency specifications and frozen versions |

Use the [documentation index](doc/README.md) to find a guide by task.
[Project progress](doc/overview/project-progress.md) records verified scope
and remaining scientific work.
