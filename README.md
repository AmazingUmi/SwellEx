# SwellEx Range Regression

Predict source range (km) from SwellEx-96 VLA recordings using Python-generated
HDF5 features and CNN/ResNet or GRNN regressors.

| Feature | Representation |
|---|---|
| RBD | Beamformer-derived equivalent Green estimates |
| ELM | Element-pair least-squares frequency ratios, excluding the diagonal |
| SCM | Normalized spatial covariance entries, including the diagonal |

## Quick Start

Edit locally; generation, numerical tests, training, prediction, and plotting
run on SSH host `haiqin`. Local commands need uv, Python 3.11+, Git, SSH, and
rsync. MATLAB source is a static reference.

```bash
# All 15 base models: range_overlap, train before / test after closest approach
./scripts/run.sh base_pack --preset train_before

# One model, including generation, prediction, figures, and retrieval
./scripts/run.sh base_pack --preset train_before --method SCM --network ResNet18

# All 35 joint models with the same split
./scripts/run.sh joint_feature_pack --preset train_before
```

| Parameter | Default | Purpose |
|---|---|---|
| `PACK` | Required | `base_pack` or `joint_feature_pack` |
| `--preset` | `full` | `full`, `train_before`, or `train_after` split |
| `--method` | All in the pack | Feature method/combination; e.g. `SCM`, joint `RBD_ELM_SCM` |
| `--network` | All five families | `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50`, or `GRNN` |
| `--fusion` | Joint preset: `block_rms` | Joint only; `block_rms` or `raw_concat` |

**Pack alone starts the full experiment.** Omitted filters select all methods
or networks. Bundled presets use 100 neural epochs and five-fold GRNN sigma
search; GRNN has no epochs or optimizer. Epochs/seed and other advanced settings
belong in a complete TOML passed with `--config`, instead of `--preset`.

Remote runs start immediately, print a descriptive task ID, follow status/logs,
and fetch results on completion. Once launched, they survive disconnection;
Ctrl-C stops local monitoring. Use `status --run ID` and `fetch --run ID` to
recover results. Task IDs include pack, split, method, network, joint fusion,
and a unique suffix. `--help` shows syntax without launching work.

See [commands and options](doc/workflows/remote-validation.md),
[configuration defaults](doc/workflows/command-defaults.md), and
[pack contracts](doc/packs/README.md) before changing an experiment.

## Results and Source

| Path | Contents |
|---|---|
| `output/data/<run_id>/` | Retrieved logs, reports, prediction CSVs, and history |
| `output/figure/<run_id>/` | Retrieved PDF figures |
| Remote run's `project/outputs/` | Datasets, checkpoints, and GRNN references |
| `scripts/<pack>/` | Pack-specific generation, networks, workflow, and visualization |
| `scripts/pack_cli.py`, `scripts/packs.py` | Shared public command contract and pack registry |
| `Origindata/` | Original inputs; excluded from Git |
| `simulation/` | Propagation model inputs |
| `pyproject.toml`, `uv.lock` | Frozen numerical dependencies |

Use the [documentation index](doc/README.md) for detailed guides.
[Project progress](doc/overview/project-progress.md) separates recorded pipeline
checks from scientific validation; a PASS run alone does not establish
accuracy or cross-track generalization.
