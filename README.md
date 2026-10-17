# SwellEx Range Regression

Predict source range (km) from SwellEx-96 VLA recordings.

## Common Commands

Run from the **local project root**. Experiments execute on `haiqin`.
Choose the command for your task; the launch commands below are alternatives.

| Task | Command |
|---|---|
| Start all 35 joint models, train before / test after | `./scripts/run.sh joint_feature_pack --preset train_before` |
| Start all 15 base models, train before / test after | `./scripts/run.sh base_pack --preset train_before` |
| Compare automatic CNN capacity with fixed sizes (18 models) | `./scripts/run.sh data_scale_pack --preset train_before` |
| Check residual MoE architecture on synthetic CUDA inputs | `./scripts/run.sh MoE_data_scale validate --suite quick` |
| Inspect the RBD+SCM data-size × expert-count matrix | `./scripts/run.sh MoE_data_scale study --plan scripts/MoE_data_scale/localization_networks/workflow/rbd_scm_expert_study_v1.json --dry-run` |
| Check / fetch a MoE run | `./scripts/run.sh MoE_data_scale status --run 'RUN_ID'` / `./scripts/run.sh MoE_data_scale fetch --run 'RUN_ID'` |
| Check a joint run's status | `./scripts/run.sh joint_feature_pack status --run 'RUN_ID'` |
| Fetch a joint run's results | `./scripts/run.sh joint_feature_pack fetch --run 'RUN_ID'` |
| Show help without starting a run | `./scripts/run.sh joint_feature_pack --help` |

Replace `RUN_ID` with the **complete ID printed after `Run:` at launch**.
For a base run, use `base_pack` in status/fetch. These commands identify an
existing run; do not add `--preset`, `--method`, or `--network`.

Launch follows progress and fetches results on completion. Once the remote
worker has started, disconnection or Ctrl-C stops only local monitoring.
Manual fetch retrieves what is available now; if the run is still active,
fetch again after completion. Reports/CSVs/logs go to `output/data/<run_id>/`;
figures go to `output/figure/<run_id>/`. HDF5/checkpoints remain on haiqin.

## Choose an Experiment

| Parameter | Default | Purpose |
|---|---|---|
| `PACK` | Required | `base_pack`, `joint_feature_pack`, `data_scale_pack`, or `MoE_data_scale` |
| `--preset` | `full` | `full`, `train_before`, or `train_after` split |
| `--method` | All in the pack | Feature method/combination; e.g. `SCM`, joint `RBD_ELM_SCM` |
| `--network` | All in the pack | Base/joint: five families. Data scale: auto, four fixed sizes and original complex CNN baseline |
| `--fusion` | Joint preset: `block_rms` | Joint only; `block_rms` or `raw_concat` |

**Pack alone starts the full experiment.** Omitted filters select all methods
or networks. Bundled presets use 100 neural epochs and five-fold GRNN sigma
search; GRNN has no epochs or optimizer. Epochs/seed and other advanced settings
belong in a complete TOML passed with `--config`, instead of `--preset`.
The [data scale pack](doc/packs/data-scale-pack.md) selects capacity before
training using an uncalibrated count rule; it has no GRNN or fusion.
The [MoE pack](doc/packs/moe-data-scale.md) instead varies residual expert count.
Its RBD+SCM matrix contains four training fractions × four expert counts × three
seeds; the pack guide provides launch/recovery commands. `validate --suite quick`
checks structure without a data-size run.

See [commands and options](doc/workflows/remote-validation.md),
[configuration defaults](doc/workflows/command-defaults.md), and
[pack contracts](doc/packs/README.md) before changing an experiment.

To run one model instead of the complete pack:

```bash
./scripts/run.sh base_pack --preset train_before --method SCM --network ResNet18
```

Local prerequisites: uv, Python 3.11+, Git, SSH, and rsync. Edit locally;
generation, numerical tests, training, prediction, and plotting run on haiqin.
MATLAB source is a static reference.

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
