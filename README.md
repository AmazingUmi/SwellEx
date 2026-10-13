# SwellEx Range Regression

Predict source range in km from SwellEx-96 vertical line array data. Python
creates HDF5 features; PyTorch CNN/ResNet models or standalone GRNN perform
regression. Experiments are organized as explicit packs: `base_pack` preserves
the original independent feature routes; `joint_feature_pack` evaluates
combinations of those features with the existing network architectures.

| Feature route | Input features |
|---|---|
| RBD | Beamformer-derived equivalent Green estimates |
| ELM | Element-pair least-squares frequency ratios, excluding the diagonal |
| SCM | Normalized spatial covariance entries, including the diagonal |

## Run an Experiment or Model

Edit code and configuration locally. Generation, numerical tests, training,
prediction, and plotting run on SSH host `haiqin`. MATLAB source is retained
only as a static reference.

From the local project root, with uv, Python 3.11+, Git, SSH, and rsync available:

```bash
# Run the selected pack's complete experiment
./scripts/run.sh base_pack
./scripts/run.sh joint_feature_pack

# Inspect the full joint plan without SSH or computation
./scripts/run.sh joint_feature_pack --dry-run

# Run one neural model through generation, training, prediction, and plotting
./scripts/run.sh base_pack ELM complex_CNN

# Run GRNN for any feature route through reference building, prediction, and plotting
./scripts/run.sh base_pack RBD GRNN
./scripts/run.sh base_pack ELM GRNN
./scripts/run.sh base_pack SCM GRNN

# Check the full pipeline with three complex CNNs and three GRNNs
./scripts/run.sh base_pack validate --suite smoke

# Joint input with a training-fitted scale for each feature block
./scripts/run.sh joint_feature_pack RBD_ELM_SCM complex_CNN --fusion block_rms
```

For `base_pack`, choose `RBD`, `ELM`, or `SCM` with `complex_CNN`, `real_CNN`, `ResNet18`, or
`ResNet50`, or `GRNN`. GRNN has no epochs or optimizer. The complete experiment
contains 12 neural models and three method-specific GRNNs.
A base-pack model command prepares all three feature routes and evaluates the
chosen model. The pack argument is required; commands without it are rejected.
The joint pack supports seven feature combinations and the same five model
families. See the [pack guide](doc/packs/README.md) for commands, fusion rules,
and comparison boundaries.

Specifying only the pack selects `validate --suite experiment`: 15 models for
`base_pack`, or 35 for `joint_feature_pack` with `block_rms` by default. Pack
options such as `--config`, `--detach`, and joint `--fusion` apply to that full
plan. Explicit `validate` still defaults to `smoke`; add `METHOD ARCH` to run
one model. `--help` lists commands without launching work.

Bundled defaults: the complete 4500 s S5 VLA recording (window starts 0–4499 s),
100 neural epochs, A5000-tuned batch size 64, seed 2026, and CUDA. Each GRNN uses
five-fold sigma search. The full_interleaved 4:1 split is a within-record baseline; independent
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
| `scripts/run.sh`, `run.py`, `pack_cli.py`, `packs.py` | Thin shell launcher, pack routing, shared argument contract, and registry |
| `scripts/base_pack/` | Original generation, independent RBD/ELM/SCM models, GRNN, workflow, and plots |
| `scripts/joint_feature_pack/` | Joint feature inputs, existing model families, and independent experiment workflow |
| `Origindata/` | Original inputs; excluded from Git |
| `simulation/` | Propagation model inputs |
| `doc/` | Workflow guides, data contracts, method references, and `packs/` experiment descriptions |
| `pyproject.toml`, `uv.lock` | Dependency specifications and frozen versions |

Use the [documentation index](doc/README.md) to find a guide by task.
[Project progress](doc/overview/project-progress.md) records verified scope
and remaining scientific work.
