# SwellEx Range Regression

Predict source range (km) from SwellEx-96 VLA recordings using RBD, ELM, and SCM complex features.

## Quick Start

Run from the **local project root**; generation, training, prediction, and plotting execute on `haiqin`.

| Task | Command |
|---|---|
| Run all 15 base models (`train_before`) | `./scripts/run.sh base_pack --preset train_before` |
| Run all 35 joint models (`train_before`, `block_rms`) | `./scripts/run.sh joint_feature_pack --preset train_before` |
| Run all 18 scalable CNN models (`train_before`) | `./scripts/run.sh data_scale_pack --preset train_before` |
| Check residual MoE contracts on synthetic CUDA | `./scripts/run.sh MoE_data_scale validate --suite quick` |
| Preview the RBD+SCM MoE data-size × expert matrix | `./scripts/run.sh MoE_data_scale study --plan scripts/remote_workflow/MoE_data_scale/rbd_scm_expert_study_v1.json --dry-run` |
| Run or dry-run the continual MoE demo | `./scripts/run.sh MoE_learn` / `./scripts/run.sh MoE_learn --dry-run` |
| Run one selected model | `./scripts/run.sh base_pack --preset train_before --method SCM --network ResNet18` |
| Query status / fetch results of a run or study | `./scripts/run.sh PACK status --run 'RUN_ID'` / `./scripts/run.sh PACK fetch --run 'RUN_ID'` |
| Inspect pack options without launching work | `./scripts/run.sh PACK --help` |

Replace `PACK` and `RUN_ID` with the launching pack and the complete ID printed after `Run:`. Do not pass `--preset`, `--method`, `--network`, or `--fusion` to `status`/`fetch`. Disconnection stops local monitoring only; remote workers continue on `haiqin`.

## Experiment Selection

| Option | Default | Scope |
|---|---|---|
| `PACK` | Required | `base_pack`, `joint_feature_pack`, `data_scale_pack`, `MoE_data_scale`, `MoE_learn` |
| `--preset` | `full` (`demo` for `MoE_learn`) | `full`, `train_before`, `train_after`; `MoE_learn`: `demo` |
| `--method` | All in pack | Single or joint feature route (e.g. `SCM`, `RBD_SCM`, `RBD_ELM_SCM`) |
| `--network` | All in pack | Base/joint: 4 neural + `GRNN`; [data scale](doc/network_training/data-scale-pack.md): 6 CNN choices; [MoE](doc/network_training/moe-data-scale.md): auto + E1/E2/E4/E8; [MoE_learn](doc/network_training/moe-learn.md): `complex_MoE_learn` |
| `--fusion` | `block_rms` | `joint_feature_pack` only: `block_rms` or `raw_concat` |

`PACK` alone launches its default preset (100 neural epochs and 5-fold GRNN sigma CV for standard packs; a short 2-stage check for `MoE_learn`). Pass `--config PATH` instead of `--preset` to customize epochs, seed, or stage rules. Local dispatch requires `uv`, Python 3.11+, Git, SSH, and `rsync`; remote execution uses the locked Python 3.14 + CUDA environment.

## Repository Layout

| Path | Contents |
|---|---|
| `scripts/data_processing/` | Pure CPU feature generation (`Origindata/` -> HDF5), joint feature concatenation, dataset validation, and static `matlab_reference/` |
| `scripts/network_training/` | Pure HDF5 -> PyTorch/GRNN training, prediction, model selection catalogs, and `MoE_learn` continual stage runner/plots |
| `scripts/remote_workflow/` | Shared CLI/pack dispatcher (`run.py`, `packs.py`, `pack_cli.py`), `haiqin` SSH workers, TOML presets, studies, and run/study report renderers |
| `doc/{data_processing,network_training,remote_workflow}/` | [Documentation index & pack matrix](doc/README.md), [HDF5 contracts](doc/data_processing/hdf5-datasets-and-splits.md), [models](doc/network_training/trainable-model-architecture.md), and [remote workflow](doc/remote_workflow/remote-validation.md) |
| `notebooks/` | Split Colab/local notebooks: [CPU data processing](notebooks/01_data_processing_cpu.ipynb) and [GPU training & analysis](notebooks/02_network_training_gpu.ipynb) |
| `output/data/<run_id>/`, `output/figure/<run_id>/` | Retrieved local reports/CSVs/logs and PDF figures (excluded from Git) |
| `Origindata/`, `simulation/` | Raw SwellEx-96 inputs (untracked) and acoustic propagation model inputs |

See [project progress](doc/project-progress.md) for retained experimental results and scientific interpretation boundaries.
