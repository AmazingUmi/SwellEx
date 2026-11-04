# Run Models and Validation on haiqin

## Common Commands

Run from the **local project root**. `scripts/run.sh` invokes standard-library `scripts/remote_workflow/run.py` (`pack_cli.py`, `packs.py`), snapshots local source, executes inside the [frozen CUDA environment](remote-environment.md) on `haiqin`, survives terminal disconnection, and delivers verified outputs on completion.

| Task | Command |
|---|---|
| Run all models in a standard pack (`train_before`) | `./scripts/run.sh PACK --preset train_before` (`base_pack`: 15, `joint_feature_pack`: 35, `data_scale_pack`: 18, `MoE_data_scale`: 20) |
| Run one selected method / network | `./scripts/run.sh base_pack --preset train_before --method SCM --network ResNet18` |
| Run joint pack with raw concatenation | `./scripts/run.sh joint_feature_pack --preset train_before --fusion raw_concat` |
| Synthetic CUDA architecture check | `./scripts/run.sh MoE_data_scale validate --suite quick` |
| Run or dry-run the continual MoE demo | `./scripts/run.sh MoE_learn` / `./scripts/run.sh MoE_learn --dry-run` |
| Run `MoE_learn` distance-holdout or before-then-after | `./scripts/run.sh MoE_learn --config scripts/remote_workflow/MoE_learn/haiqin_holdout_middle.toml` (`near`, `middle`, `far`, `haiqin_before_then_after.toml`, `..._range.toml`) |
| Launch/inspect a detached matrix coordinator | `./scripts/run.sh MoE_data_scale study --plan scripts/remote_workflow/MoE_data_scale/rbd_scm_expert_study_v1.json --dry-run` |
| Check status / fetch results of a run or study | `./scripts/run.sh PACK status --run 'RUN_ID'` / `./scripts/run.sh PACK fetch --run 'RUN_ID'` |
| Inspect syntax without launching work | `./scripts/run.sh PACK --help` |

Pass only the launching `PACK` and `RUN_ID` to `status`/`fetch` (plus `--config PATH` only if `remote.host`/`remote.workspace` was overridden). Manual `fetch` retrieves currently available exports once without blocking.

## Experiment Parameters

```text
./scripts/run.sh PACK [--preset PRESET | --config PATH] [--method METHOD] [--network NETWORK] [--fusion MODE] [--no-cache]
```

| Parameter | Default | Values and purpose |
|---|---|---|
| `PACK` | Required | `base_pack`, `joint_feature_pack`, `data_scale_pack`, `MoE_data_scale`, `MoE_learn` |
| `--preset` | `full` (`demo` for `MoE_learn`) | `full` (`haiqin_full_interleaved.toml`), `train_before` (`haiqin_range_overlap_train_before.toml`), `train_after` (`..._train_after.toml`), or `demo` (`haiqin_demo.toml`) |
| `--method` | All in pack | `RBD`, `ELM`, `SCM`; joint adds `RBD_ELM`, `RBD_SCM`, `ELM_SCM`, `RBD_ELM_SCM`; `MoE_data_scale` adds `RBD_SCM`; `MoE_learn` uses `RBD_SCM` |
| `--network` | All in pack | Base/joint: `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50`, `GRNN`; pack variants in [documentation index](../README.md#experiment-packs) |
| `--fusion` | `block_rms` | `joint_feature_pack` only: `block_rms` or `raw_concat` |
| `--config PATH` | Preset TOML | Complete custom TOML (mutually exclusive with `--preset`; unknown/missing fields fail); see [configuration defaults](command-defaults.md) |
| `--no-cache` | Off | Force fresh HDF5 generation instead of reusing verified cached datasets |

For a fast pipeline smoke test, copy `haiqin_full_interleaved.toml`, set `dataset.segment_end_s = 59` and `training.epochs = 1`, and pass `--config` to `validate` (`range_overlap` requires spanning closest approach, so `0–59 s` is invalid there).

## Task ID, Status, and Fetch

Experiment and validation launches print `Run:` with a descriptive ID (`base`, `joint`, `scale`, `moe`, `moelearn`):

```text
UTC_timestamp-pack-split-method-network[-fusion]-8hex
20261013T120000Z-base-before-SCM-ResNet18-a1b2c3d4
20261013T120000Z-joint-after-all-GRNN-block_rms-a1b2c3d4
```

Matrix studies register an explicit study ID (e.g. `scm-capacity-budget-study-v2`, `rbd-scm-moe-expert-study-v1`) accepted by the same `status`/`fetch` commands.

| Status | Exit code | Meaning |
|---|---:|---|
| `STARTING` / `WAITING_FOR_RESOURCE` / `RUNNING` / `EXPORTING` | `2` | Active or queued behind the remote GPU lock |
| `PASS` | `0` | Pipeline completed |
| `FAIL` | `1` | Failed; run `fetch` to inspect `report.json` and `logs/` |

## Maintenance Commands

| Command | Scope and options |
|---|---|
| `check` | Preflight SSH, remote paths, `Origindata`, and `uv` (`--config` optional; skips CUDA execution) |
| `validate` | `--suite smoke` (default), `quick`, or `experiment`; accepts `--preset`/`--config`, `--no-cache`, and joint `--fusion` (`smoke` keeps configured intervals/epochs except `MoE_learn`, which caps to 1 epoch) |
| `status` / `fetch` | Requires `--run RUN_ID`; verifies retrieved files via `rsync` checksums and manifest SHA256 |
| `local-grnn` | `base_pack` only (`scripts/remote_workflow/base_pack/local_grnn.py`): foreground local CPU `range_overlap` GRNN (`--method RBD\|ELM\|SCM`, `--python PATH` for a local Python 3.14 env matching `pyproject.toml`, `--preset train_before\|train_after` or `--config PATH`, optional `--data PATH`); see [standalone GRNN](../network_training/standalone-grnn.md) |
| `study` / `budget-study` | Detached matrix coordinators on `haiqin` (`--plan PATH [--study-dir PATH] [--dry-run]`); see [controlled matrix studies](matrix-studies.md) |

## Results and Offline Comparison

- `output/data/<run_id>/`: `report.json`, `request.json`, `manifest.json`, `logs/`, CSVs/histories, and `delivery-verification.json` (study children nest under `output/{data,figure}/<study_id>/children/<run_id>/`).
- `output/figure/<run_id>/`: Rendered PDFs (see [result visualization](result-visualization.md)). Raw HDF5, neural checkpoints, and GRNN references stay on `haiqin` (except the small `MoE_learn` deployment checkpoints).
- Merge saved local `base_pack` run summaries offline without SSH via `python3 -B -m scripts.remote_workflow.base_pack.compare_results [--input-dir output/data] [--output-dir PATH]`.
