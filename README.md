# SwellEx Range Regression

Predict source range (km) from SwellEx-96 VLA recordings using RBD and SCM complex features with `complex_CNN`, `ResNet18`, `ResNet50`, and standalone `GRNN` regressors.

## Quick Start

Run from the **local project root**; generation, training, prediction, and plotting execute on `haiqin`.

| Task | Command |
|---|---|
| Run all 8 base models (`train_before`) | `./scripts/run.sh base_pack --preset train_before` |
| Run selected methods / networks | `./scripts/run.sh base_pack --preset train_before --method SCM --network complex_CNN ResNet18` |
| Query status / fetch results of a run | `./scripts/run.sh base_pack status --run 'RUN_ID'` / `./scripts/run.sh base_pack fetch --run 'RUN_ID'` |
| Inspect options without launching work | `./scripts/run.sh base_pack --help` |

Replace `RUN_ID` with the complete ID printed after `Run:`. Do not pass `--preset`, `--method`, or `--network` to `status`/`fetch`. Disconnection stops local monitoring only; remote workers continue on `haiqin`.

## Experiment Selection

| Option | Default | Scope |
|---|---|---|
| `PACK` | Required | `base_pack` (the only active pack) |
| `--preset` | `full` | `full`, `train_before`, `train_after` |
| `--method` | From `[experiment]` | `1..N` unique values from `RBD`, `SCM` |
| `--network` | From `[experiment]` | `1..N` unique values from `complex_CNN`, `ResNet18`, `ResNet50`, `GRNN` |

`PACK` alone launches its default preset (`[experiment]` lists all methods and networks, 100 neural epochs, and 5-fold GRNN sigma CV). Pass `--config PATH` instead of `--preset` to customize `[experiment]`, multi-snapshot `[scm]`, windowing, splits, or training hyperparameters; every configuration generates isolated dataset fingerprints and run outputs. Local dispatch requires `uv`, Python 3.11+, Git, SSH, and `rsync`; remote execution uses the locked Python 3.14 + CUDA environment.

## Repository Layout

| Path | Contents |
|---|---|
| `scripts/data_processing/` | Pure CPU feature generation (`Origindata/` -> HDF5), dataset validation, and static `matlab_reference/` |
| `scripts/network_training/` | Pure HDF5 -> PyTorch/GRNN training and prediction: `neural/` (shared RBD/SCM runtime), `grnn/`, `common/`, and the model `catalog.py` |
| `scripts/remote_workflow/` | Shared CLI/pack dispatcher (`run.py`, `packs.py`, `pack_cli.py`), `haiqin` SSH workers (`remote.py`, `worker.py`), `presets/*.toml`, validation, and run report renderers |
| `doc/{data_processing,network_training,remote_workflow}/` | [Documentation index](doc/README.md), [HDF5 contracts](doc/data_processing/hdf5-datasets-and-splits.md), [models](doc/network_training/neural-models.md), and [remote workflow](doc/remote_workflow/run-and-validate.md) |
| `notebooks/` | Split Colab/local notebooks: [CPU data processing](notebooks/01_data_processing_cpu.ipynb) and [GPU training & analysis](notebooks/02_network_training_gpu.ipynb) |
| `output/data/<run_id>/`, `output/figure/<run_id>/` | Retrieved local reports/CSVs/logs and PDF figures (excluded from Git); see [`output/data/README.md`](output/data/README.md) |
| `archive/` | Retired code, documents, notebooks, and results (ELM, `real_CNN`, joint-feature, data-scale, MoE packs); see [archive index](archive/README.md) |
| `Origindata/`, `simulation/` | Raw SwellEx-96 inputs (untracked) and acoustic propagation model inputs |

See [project progress](doc/project-progress.md) for retained experimental results and scientific interpretation boundaries.
