# Run Models and Validation on haiqin

## Common Commands

Run from the **local project root**. `scripts/run.sh` invokes standard-library `scripts/remote_workflow/run.py` (`pack_cli.py`, `packs.py`), snapshots local source, executes inside the [frozen CUDA environment](remote-environment.md) on `haiqin`, survives terminal disconnection, and delivers verified outputs on completion.

| Task | Command |
|---|---|
| Run all 8 `base_pack` models (`train_before`) | `./scripts/run.sh base_pack --preset train_before` |
| Run one selected method / network | `./scripts/run.sh base_pack --preset train_before --method SCM --network ResNet18` |
| Check status / fetch results of a run | `./scripts/run.sh base_pack status --run 'RUN_ID'` / `./scripts/run.sh base_pack fetch --run 'RUN_ID'` |
| Inspect syntax without launching work | `./scripts/run.sh base_pack --help` |

Pass only `base_pack` and `RUN_ID` to `status`/`fetch` (plus `--config PATH` only if `remote.host`/`remote.workspace` was overridden). Manual `fetch` retrieves currently available exports once without blocking.

## Experiment Parameters

```text
./scripts/run.sh base_pack [--preset PRESET | --config PATH] [--method METHOD] [--network NETWORK] [--no-cache]
```

| Parameter | Default | Values and purpose |
|---|---|---|
| `PACK` | Required | `base_pack` (the only active pack; ELM, joint-feature, data-scale and MoE packs are archived, see [archive](../../archive/README.md)) |
| `--preset` | `full` | `full`, `train_before`, or `train_after` (`scripts/remote_workflow/presets/<preset>.toml`) |
| `--method` | All in pack | `RBD` or `SCM` |
| `--network` | All in pack | `complex_CNN`, `ResNet18`, `ResNet50`, `GRNN` |
| `--config PATH` | Preset TOML | Complete custom TOML (mutually exclusive with `--preset`; unknown/missing fields fail); see [configuration defaults](workflow-config.md) |
| `--no-cache` | Off | Force fresh HDF5 generation instead of reusing verified cached datasets |

For a fast pipeline smoke test, copy `presets/full.toml`, set `dataset.segment_end_s = 59` and `training.epochs = 1`, and pass `--config` to `validate` (`range_overlap` requires spanning closest approach, so `0–59 s` is invalid there).

## Task ID, Status, and Fetch

Experiment and validation launches print `Run:` with a descriptive ID (pack tag `base`):

```text
UTC_timestamp-pack-split-method-network-8hex
20261013T120000Z-base-before-SCM-ResNet18-a1b2c3d4
20261013T120000Z-base-after-all-GRNN-a1b2c3d4
```

Runs recorded before the cleanup (15-model plan with ELM and `real_CNN`, or other pack tags) are historical evidence: active `status`/`fetch`/visualization validate against the current 8-model plan and reject them.

| Status | Exit code | Meaning |
|---|---:|---|
| `STARTING` / `WAITING_FOR_RESOURCE` / `RUNNING` / `EXPORTING` | `2` | Active or queued behind the remote GPU lock |
| `PASS` | `0` | Pipeline completed |
| `FAIL` | `1` | Failed; run `fetch` to inspect `report.json` and `logs/` |

## Maintenance Commands

| Command | Scope and options |
|---|---|
| `check` | Preflight SSH, remote paths, `Origindata`, and `uv` (`--config` optional; skips CUDA execution) |
| `validate` | `--suite smoke` (default), `quick`, or `experiment`; accepts `--preset`/`--config` and `--no-cache` (`smoke` keeps configured intervals/epochs) |
| `status` / `fetch` | Requires `--run RUN_ID`; verifies retrieved files via `rsync` checksums and manifest SHA256 |
| `local-grnn` | `base_pack` only (`scripts/remote_workflow/local_grnn.py`): foreground local CPU `range_overlap` GRNN (`--method RBD\|SCM`, `--python PATH` for a local Python 3.14 env matching `pyproject.toml`, `--preset train_before\|train_after` or `--config PATH`, optional `--data PATH`); see [standalone GRNN](../network_training/grnn-cli.md) |

## Results and Offline Comparison

- `output/data/<run_id>/`: `report.json`, `request.json`, `manifest.json`, `logs/`, CSVs/histories, and `delivery-verification.json` (archived studies live under `archive/output/`).
- `output/figure/<run_id>/`: Rendered PDFs (see [result visualization](result-visualization.md)). Raw HDF5, neural checkpoints, and GRNN references stay on `haiqin`.
- Merge saved local `base_pack` run summaries offline without SSH via `python3 -B -m scripts.remote_workflow.compare_results [--input-dir output/data] [--output-dir PATH]`.
