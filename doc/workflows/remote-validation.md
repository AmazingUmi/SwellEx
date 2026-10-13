# Run Models and Validation on haiqin

Run commands from the local project root. Edit source/configuration locally;
generation, numerical tests, training, prediction, and plotting run on `haiqin`.
Local prerequisites: uv, Python 3.11+, Git, SSH, and rsync. The dispatcher uses
the standard library; remote computation uses the [frozen environment](remote-environment.md).

## Common Experiments

```bash
# All 15 base models, range_overlap: train before / test after closest approach
./scripts/run.sh base_pack --preset train_before

# All 35 joint models for the same split
./scripts/run.sh joint_feature_pack --preset train_before

# One method with all five networks; or one network with all methods
./scripts/run.sh base_pack --preset train_before --method SCM
./scripts/run.sh base_pack --preset train_before --network GRNN

# One model
./scripts/run.sh base_pack --preset train_before --method SCM --network ResNet18

# Joint comparison with a different fusion condition
./scripts/run.sh joint_feature_pack --preset train_before --fusion raw_concat
```

Every remote experiment starts immediately, follows status and new logs, and
fetches results on completion. A temporary monitoring connection failure retries
automatically. Once launched, the remote worker survives client disconnection;
Ctrl-C stops only local monitoring (exit 130). Before launch completes, interruption
can prevent the worker from starting. The printed ID lets you check what happened.

## Experiment Parameters

```text
./scripts/run.sh PACK [--preset PRESET] [--method METHOD] [--network NETWORK] [--fusion MODE]
```

| Parameter | Default | Values and purpose |
|---|---|---|
| `PACK` | Required | `base_pack`: 3 feature methods; `joint_feature_pack`: 7 feature combinations |
| `--preset` | `full` | `full`: four train windows then one test; `train_before`: range-overlap before→after; `train_after`: after→before |
| `--method` | All methods in the pack | Base: `RBD`, `ELM`, `SCM`. Joint also supports `RBD_ELM`, `RBD_SCM`, `ELM_SCM`, `RBD_ELM_SCM` |
| `--network` | All five families | `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50`, `GRNN` |
| `--fusion` | Joint TOML's value; bundled `block_rms` | Joint only: `block_rms` or `raw_concat` |

Omitting either filter selects all values of that dimension. Pack alone runs its
full experiment with the `full` preset: 15 base or 35 joint models. Bundled
presets use full S5 candidate coverage, 100 neural epochs, and five-fold GRNN
sigma search. GRNN builds references without epochs or an optimizer.

`train_before` and `train_after` use `range_overlap`: up to 800 s per candidate
side, a 15 s central gap, and a shared distance interval selected using both
sides' true ranges. Side counts may differ. “All models” does not mean all
recorded windows are retained. See the [split contract](../datasets/hdf5-datasets-and-splits.md#train-and-test-splits).

Names are case-sensitive; unknown, repeated, incomplete, or misplaced arguments
fail. Old positional `METHOD ARCH`, `--epochs`, `--seed`, `--detach`, and
`--dry-run` are removed from the shell interface. Use `--help` to inspect syntax
without launching a run.

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

The launcher prints `Run:` before remote preparation and saves the request locally.
New IDs describe the effective split and selection, including custom TOML settings:

```text
UTC timestamp-pack-split-method-network[-fusion]-random suffix
20261013T120000Z-base-before-all-all-a1b2c3d4
20261013T120000Z-base-before-SCM-ResNet18-a1b2c3d4
20261013T120000Z-joint-after-all-GRNN-block_rms-a1b2c3d4
```

`base`/`joint` identify the pack; `full`/`before`/`after` identify the split;
`all` denotes an omitted filter. The eight hexadecimal random characters avoid
collisions between otherwise identical submissions. Epochs, seed, suite, full
configuration, and source identity remain in the request/manifest. Workers reject
IDs inconsistent with the frozen parameters. Status/fetch reject old-format and
cross-pack IDs; existing historical results remain on disk.

```bash
RUN_ID='<exact ID printed at launch>'
./scripts/run.sh base_pack status --run "$RUN_ID"
./scripts/run.sh base_pack fetch --run "$RUN_ID"
```

Use the launching pack and the same `--config PATH` if it changes the remote
host/workspace. Fetch retrieves available results once; it does not wait for a
running task. For an interrupted monitor, query status and fetch again after completion.

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
| `status --run RUN_ID` | Query a run; optional `--config` |
| `fetch --run RUN_ID` | Retrieve and verify available results; optional `--config` |
| `local-grnn` | Explicit local CPU exception; base only. See [local GRNN](standalone-grnn.md#local-cpu-range-overlap-computation) |

`quick` checks runtime, CLI/model/loader contracts, numerical oracles, and
workflow/plot tests. `smoke` adds generation, complex CNN + GRNN per method,
prediction, a resume check, and figures (6 base / 14 joint models).
`experiment` runs all five families. **Smoke does not shorten configured intervals
or epochs.** Experiments with a method/network filter omit global quick/numerical
and resume checks, while still checking data, predictions, and delivery.

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
