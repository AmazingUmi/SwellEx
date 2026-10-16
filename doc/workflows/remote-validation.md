# Run Models and Validation on haiqin

## Common Commands

Run from the **local project root**. Launch commands are alternatives; choose the
pack and split you want. Replace `RUN_ID` with the complete ID printed after
`Run:` at launch.

| Task | Command |
|---|---|
| All 35 joint models, train before / test after | `./scripts/run.sh joint_feature_pack --preset train_before` |
| All 15 base models, train before / test after | `./scripts/run.sh base_pack --preset train_before` |
| All 18 data-scale models, train before / test after | `./scripts/run.sh data_scale_pack --preset train_before` |
| All 35 joint models, train after / test before | `./scripts/run.sh joint_feature_pack --preset train_after` |
| Check joint task status | `./scripts/run.sh joint_feature_pack status --run 'RUN_ID'` |
| Fetch joint results | `./scripts/run.sh joint_feature_pack fetch --run 'RUN_ID'` |
| Check base task status | `./scripts/run.sh base_pack status --run 'RUN_ID'` |
| Fetch base results | `./scripts/run.sh base_pack fetch --run 'RUN_ID'` |
| Check a data-scale child or whole study | `./scripts/run.sh data_scale_pack status --run 'RUN_ID'` |
| Fetch data-scale child or study results | `./scripts/run.sh data_scale_pack fetch --run 'RUN_ID'` |
| Show help without starting a run | `./scripts/run.sh joint_feature_pack --help` |

**Status/fetch need only the launching pack and run ID.** Do not add `--preset`,
`--method`, `--network`, or `--fusion`. If a custom configuration changes the
remote host/workspace, also supply the same `--config PATH`.

Launching starts remote work, follows status/logs, and fetches results when done.
Once the worker has started, disconnection or Ctrl-C stops only local monitoring;
a temporary monitoring connection failure retries automatically. Interruption
before launch completes can prevent startup; use the printed ID to check.
Manual fetch retrieves available files once and does not wait for completion.
If status is still running, fetch again after the task finishes.

Data-scale `study` and `budget-study` launches return after starting a remote
coordinator. The whole matrix continues on haiqin without the local computer.
Use its printed study ID with the same `status`/`fetch` commands; repeat the
original matrix launch to recover it without rerunning completed children.
See the [budget study commands](../packs/data-scale-budget-study.md#run-and-recover).

| Retrieved files | Local directory |
|---|---|
| Reports, prediction CSVs, training history, and logs | `output/data/<run_id>/` |
| PDF figures | `output/figure/<run_id>/` |

For a registered data-scale study, child deliveries are grouped under
`output/{data,figure}/<study_id>/children/<run_id>/`. Both whole-study and direct
child fetch use this location once the study ledger has been retrieved locally.

HDF5/checkpoints/GRNN references stay on haiqin. Detailed status meanings and ID
format are in [Task ID, Status, and Fetch](#task-id-status-and-fetch).

Edit source/configuration locally. Generation, numerical tests, training,
prediction, and plotting run on haiqin. Local prerequisites: uv, Python 3.11+,
Git, SSH, and rsync. Remote computation uses the [frozen environment](remote-environment.md).

## Experiment Parameters

```text
./scripts/run.sh PACK [--preset PRESET] [--method METHOD] [--network NETWORK] [--fusion MODE]
```

| Parameter | Default | Values and purpose |
|---|---|---|
| `PACK` | Required | `base_pack`: 3 methods; `joint_feature_pack`: 7 combinations; `data_scale_pack`: 3 methods with scalable complex CNNs |
| `--preset` | `full` | `full`: four train windows then one test; `train_before`: range-overlap before→after; `train_after`: after→before |
| `--method` | All methods in the pack | Base: `RBD`, `ELM`, `SCM`. Joint also supports `RBD_ELM`, `RBD_SCM`, `ELM_SCM`, `RBD_ELM_SCM` |
| `--network` | All in the pack | Base/joint: `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50`, `GRNN`. Data scale: see its [network choices](../packs/data-scale-pack.md#network-family) |
| `--fusion` | Joint TOML's value; bundled `block_rms` | Joint only: `block_rms` or `raw_concat` |

Omitting either filter selects all values of that dimension. Pack alone runs its
full experiment with the `full` preset: 15 base, 35 joint or 18 data-scale models. Bundled
presets use full S5 candidate coverage, 100 neural epochs, and five-fold GRNN
sigma search. GRNN builds references without epochs or an optimizer.
Data scale has no GRNN; its [configuration and capacity rule](../packs/data-scale-pack.md#training-count-and-capacity-rule)
control training fractions and pre-training architecture selection.

`train_before` and `train_after` use `range_overlap`: up to 800 s per candidate
side, a 15 s central gap, and a shared distance interval selected using both
sides' true ranges. Side counts may differ. “All models” does not mean all
recorded windows are retained. See the [split contract](../datasets/hdf5-datasets-and-splits.md#train-and-test-splits).

Names are case-sensitive; unknown, repeated, incomplete, or misplaced arguments
fail. Old positional `METHOD ARCH`, `--epochs`, `--seed`, and `--detach` are
removed from the shell interface. `--dry-run` is accepted only by data-scale
`budget-study`. Use `--help` to inspect syntax without launching a run.

### Select Methods, Networks, or Fusion

Each row starts a separate experiment. Omitted method/network filters select all.

| Task | Command |
|---|---|
| SCM with all five networks | `./scripts/run.sh base_pack --preset train_before --method SCM` |
| GRNN with all base methods | `./scripts/run.sh base_pack --preset train_before --network GRNN` |
| One SCM ResNet18 model | `./scripts/run.sh base_pack --preset train_before --method SCM --network ResNet18` |
| Joint full plan with raw concatenation | `./scripts/run.sh joint_feature_pack --preset train_before --fusion raw_concat` |

### Advanced Configuration

| Option | Default | Purpose |
|---|---|---|
| `--config PATH` | Pack's bundled preset | Complete custom TOML; mutually exclusive with `--preset` |
| `--no-cache` | Off | Regenerate feature datasets for an experiment or validation run |

Copy a pack's TOML to change epochs, seed, interval, or other numerical settings;
unknown/missing fields fail. `--fusion` may override the joint configuration.
The effective configuration and filters are frozen in `request.json`.
See [configuration defaults](command-defaults.md).

Presets live in `scripts/<pack>/localization_networks/workflow/`:

| Preset | TOML file |
|---|---|
| `full` | `haiqin_full_interleaved.toml` |
| `train_before` | `haiqin_range_overlap_train_before.toml` |
| `train_after` | `haiqin_range_overlap_train_after.toml` |

For a short pipeline check, copy `haiqin_full_interleaved.toml`, set
`dataset.segment_end_s = 59` and `training.epochs = 1`, then run:

```bash
./scripts/run.sh base_pack validate --config /tmp/swellex-smoke.toml
```

A short range-overlap configuration must span both sides of closest approach
with overlapping distances; 0–59 s is unsuitable.

## Task ID, Status, and Fetch

Use the complete [status/fetch commands at the top of this page](#common-commands).
The pack must match the original task. `PASS` means complete; `FAIL` means the
run failed and its logs should be fetched for diagnosis.

The launcher prints `Run:` before remote preparation and saves the request locally.
New IDs describe the effective split and selection, including custom TOML settings:

```text
UTC timestamp-pack-split-method-network[-fusion]-random suffix
20261013T120000Z-base-before-all-all-a1b2c3d4
20261013T120000Z-base-before-SCM-ResNet18-a1b2c3d4
20261013T120000Z-joint-after-all-GRNN-block_rms-a1b2c3d4
```

`base`/`joint`/`scale` identify the pack; `full`/`before`/`after` identify the split;
`all` denotes an omitted filter. The eight hexadecimal random characters avoid
collisions between otherwise identical submissions. Epochs, seed, suite, full
configuration, and source identity remain in the request/manifest. Workers reject
IDs inconsistent with the frozen parameters. Status/fetch reject old-format and
cross-pack IDs; existing historical results remain on disk.

Data-scale matrices additionally register a study ID such as
`scm-capacity-budget-study-v2`. Its remote registry verifies pack, frozen plan,
source and child identities before status, fetch or recovery. Study IDs are not
filesystem paths. Malformed or cross-pack child IDs are rejected rather than
treated as study names. Study status checks coordinator liveness and remote child
reports instead of trusting a local ledger's last `RUNNING` entry.

| Status | Meaning | Status command exit code |
|---|---|---|
| `STARTING` | Preparing snapshot/environment | `2` |
| `WAITING_FOR_RESOURCE` | Waiting for the numerical-run lock | `2` |
| `RUNNING` | Computing or checking | `2` |
| `EXPORTING` | Collecting deliverables | `2` |
| `PASS` | Configured pipeline completed | `0` |
| `FAIL` | Failed; inspect report/logs | `1` |

Automatic following retries transport failures without resubmitting the task.
Configuration, remote application, and delivery-integrity errors fail instead of
retrying indefinitely. Fetch verifies transfers using rsync checksums and refreshes
terminal exports if completion occurs during retrieval.

## Maintenance Commands

| Command after `PACK` | Purpose / options |
|---|---|
| `check` | Check SSH, repository, raw inputs, and uv; does not test CUDA. Optional `--config` |
| `validate` | Default `--suite smoke`; also accepts `quick` or `experiment`, `--preset` or `--config`, `--no-cache`, and joint `--fusion` |
| `status --run RUN_ID` | Query a run, or a registered data-scale study; optional `--config` |
| `fetch --run RUN_ID` | Retrieve and verify available run/study results; optional `--config` |
| `local-grnn` | Explicit local CPU exception; base only. See [local GRNN](standalone-grnn.md#local-cpu-range-overlap-computation) |
| `study --plan PATH [--study-dir PATH]` | Data scale only: start/recover a detached haiqin data-size/seed coordinator; see [matrix studies](../packs/data-scale-pack.md#data-size-matrix) |
| `budget-study --plan PATH [--study-dir PATH] [--dry-run]` | Data scale only: start/recover a detached haiqin controlled-budget coordinator, or inspect its plan without launching; see [budget protocol](../packs/data-scale-budget-study.md) |

`quick` checks runtime, CLI/model/loader contracts, numerical oracles, and
workflow/plot tests. `smoke` adds generation, complex CNN + GRNN per method,
prediction, a resume check, and figures (6 base / 14 joint models).
For base/joint, `experiment` runs all five families. **Smoke does not shorten configured intervals
or epochs.** Experiments with a method/network filter omit global quick/numerical
and resume checks, while still checking data, predictions, and delivery.
For data scale, smoke uses auto + original baseline for each method (6 models),
and experiment uses all six choices (18). Quick covers all registered sizes.

## Results and Failures

| Local path | Contents |
|---|---|
| `output/data/<run_id>/report.json` | Status, checks, metrics, and failures |
| `output/data/<run_id>/request.json` | Effective configuration and filters |
| `output/data/<run_id>/logs/` | Commands and logs |
| `output/data/<run_id>/delivery-verification.json` | Visualization file size/SHA256 verification, when present |
| `output/figure/<run_id>/` | PDF figures |

HDF5/checkpoints/references stay in the remote run's `project/outputs/`.
PASS model runs require a visualization manifest; quick runs may lack one.
See [figure layout and verification scope](result-visualization.md).

| Failure | Action |
|---|---|
| Invalid argument/configuration | Correct the named field; consult `--help` or the tables above |
| SSH/path/uv preflight | Run `PACK check`; inspect configured paths |
| CUDA/dependency | Inspect logs and `environment.json`; see the [environment guide](remote-environment.md) |
| GPU memory | Check competing workloads; reduce `training.batch_size` if needed |
| Dataset cache | Inspect validation failure; use `--no-cache` to regenerate |
| Interrupted run | Query status, then fetch logs using the printed ID |

Existing local base results can be merged without computation or SSH:

```bash
python3 -B -m scripts.base_pack.localization_networks.workflow.compare_results
```

| Option | Default | Meaning |
|---|---|---|
| `--input-dir PATH` | Project `output/data/` | Direct child run directories to inspect |
| `--output-dir PATH` | New timestamped comparison directory | New output directory; must not exist |

The reader verifies CSV metrics, test identities, histories, and sigma selection,
then writes `comparison.json`, `metrics.csv`, and `provenance.json`. Duplicate
predictions retain their sources; records without local CSVs remain `summary_only`.
It does not run models or render figures.
