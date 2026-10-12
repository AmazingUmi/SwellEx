# Run Models and Validation on haiqin

Run the commands below from the local project root. Source and configuration
are edited locally; generation, tests, training, prediction, and plotting run
on SSH host `haiqin`.

Local prerequisites: uv, Python 3.11+, Git, SSH, and rsync. The remote worker
uses the frozen dependencies in `pyproject.toml` and `uv.lock`.
Explicit local CPU GRNN execution uses the separate entry described below.

## Select One Model

```bash
# ELM complex CNN, using the bundled configuration
./scripts/run.sh ELM complex_CNN

# Override the neural epoch count and seed
./scripts/run.sh SCM ResNet18 --epochs 200 --seed 2027

# GRNN reference building and prediction for any feature route
./scripts/run.sh RBD GRNN
./scripts/run.sh ELM GRNN
./scripts/run.sh SCM GRNN
```

Both positional arguments are required. Names are case-sensitive:

| Argument | Accepted values |
|---|---|
| `METHOD` | `RBD`, `ELM`, `SCM` |
| `ARCH` for any method | `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50`, `GRNN` |

Each command performs these steps:

1. Freeze the current source and merged configuration in a new remote run.
2. Generate or reuse all three aligned RBD/ELM/SCM feature datasets, then validate them.
3. Train only the selected neural model, or build the selected method's GRNN reference.
4. Predict test ranges, check the prediction CSV, and render figures.
5. Retrieve logs, tables, reports, and figures to the local project.

Neural runs start fresh and predict with that run's best checkpoint. GRNN has
no epochs or optimizer. A selected-run PASS means the pipeline completed;
it does not apply an accuracy acceptance threshold.

## Options and Defaults

The bundled configuration uses the complete 4500 s S5 VLA recording: window
starts from 0 through 4499 s, 100 neural epochs, batch size 64, seed 2026, and
CUDA. Each GRNN independently uses five-fold CV over 29 sigma candidates. The full_interleaved 4:1
split provides a within-record baseline, with no accuracy acceptance threshold.
See [all command defaults](command-defaults.md) for the complete settings.

| Option | Default | Use |
|---|---|---|
| `--config PATH` | Bundled `haiqin_full_interleaved.toml` | Load a complete TOML configuration |
| `--epochs N` | `100` | Set a positive neural epoch count; rejected for GRNN |
| `--seed N` | `2026` | Set the run seed, from 0 to 4294967295 |
| `--detach` | Off | Launch and return the run ID immediately |
| `--dry-run` | Off | Print the merged request without contacting haiqin |
| `--no-cache` | Off | Regenerate all three feature datasets |

`--epochs` and `--seed` override the loaded TOML. Unknown names, options,
fields, and invalid values are rejected before launch.

### Split Configurations

Three complete, independent configurations share the same remote, feature,
training, and GRNN settings. Choose one with `--config PATH`:

| Configuration in `scripts/localization_networks/workflow/` | Training and test selection |
|---|---|
| [haiqin_full_interleaved.toml](../../scripts/localization_networks/workflow/haiqin_full_interleaved.toml) (default) | Full S5 recording; four train windows, then one test window |
| [haiqin_range_overlap_train_before.toml](../../scripts/localization_networks/workflow/haiqin_range_overlap_train_before.toml) | Shared range interval; before closest approach for training, after it for testing |
| [haiqin_range_overlap_train_after.toml](../../scripts/localization_networks/workflow/haiqin_range_overlap_train_after.toml) | Shared range interval; after closest approach for training, before it for testing |

Both `range_overlap` configurations take up to 800 s of candidate windows per
side, skip 15 s nearest closest approach on each side, and retain only labels
inside the sides' shared range interval. The test step is 1 s. Selected side
counts may differ; samples are not paired one-to-one by distance.
`train_test_ratio` is ignored. Neural training still reserves 25% of the selected
training-side rows for validation when no validation files exist. See the
[split contract](../datasets/hdf5-datasets-and-splits.md#train-and-test-splits).

```bash
./scripts/run.sh ELM complex_CNN --config scripts/localization_networks/workflow/haiqin_range_overlap_train_before.toml
./scripts/run.sh ELM complex_CNN --config scripts/localization_networks/workflow/haiqin_range_overlap_train_after.toml
```

For smoke validation, keep a candidate interval containing both sides of
closest approach with overlapping range coverage; the 0–59 s short interval
below is for `full_interleaved`.

To change other settings, copy and edit the complete configuration, then
inspect it before launching:

```bash
cp scripts/localization_networks/workflow/haiqin_full_interleaved.toml /tmp/swellex-experiment.toml
# Edit /tmp/swellex-experiment.toml, then inspect the request:
./scripts/run.sh ELM complex_CNN --config /tmp/swellex-experiment.toml --dry-run
# Launch with the same configuration:
./scripts/run.sh ELM complex_CNN --config /tmp/swellex-experiment.toml
```

Learning rate and dropout are direct Python CLI options; `run.sh` does not
accept them. The workflow TOML also has no fields for them.

## Local GRNN and Existing-Result Comparison

The local GRNN worker belongs to `workflow/local_grnn.py`. Inspect a
`range_overlap` direction through the same shell dispatcher:

```bash
./scripts/run.sh local-grnn --method SCM \
  --config scripts/localization_networks/workflow/haiqin_range_overlap_train_before.toml \
  --python /private/tmp/swellex-grnn-local-env/bin/python --dry-run
```

The dry run performs no computation or SSH. Actual execution requires omitting
`--dry-run` and an explicitly selected CPU environment. It generates only the selected method,
or accepts a validated matching directory with `--data PATH`; see
[local GRNN setup and delivery](standalone-grnn.md#local-cpu-range-overlap-computation).

To merge results already present locally, use the standard-library reader:

```bash
python3 -B -m scripts.localization_networks.workflow.compare_results
```

It reads direct run directories under `output/data/`, verifies CSV metrics,
test identities, epoch histories and GRNN sigma selection, and writes a fresh
comparison directory containing `comparison.json`, `metrics.csv`, and
`provenance.json`. Use `--input-dir PATH` or `--output-dir NEW_PATH` to choose
the locations. Duplicate GRNN predictions retain their source runs; summaries
without locally available CSV files remain `summary_only`. This command does
not generate data, build references, predict, render figures, or contact haiqin.

## Run Validation Suites

```bash
./scripts/run.sh check
./scripts/run.sh validate --suite quick
./scripts/run.sh validate --suite smoke
./scripts/run.sh validate --suite experiment
```

| Command | What it checks or runs |
|---|---|
| `check` | SSH access, remote repository, original-input directory, and uv; it does not test CUDA |
| `validate --suite quick` | Frozen environment, Python/CLI/model and loader contracts, GRNN invariants, numerical oracles, and visualization tests |
| `validate --suite smoke` | Quick checks, three-route generation/validation, one complex CNN and one GRNN per method (six models), prediction, checkpoint resume, and figures |
| `validate --suite experiment` | The same workflow with all four neural architectures and one GRNN per method (fifteen models) |

`validate` defaults to `smoke`. These suites accept `--config`, `--detach`,
`--dry-run`, and `--no-cache`; set epochs and seed in their TOML configuration.
Selected-model commands omit the global quick/numerical/resume checks.

Smoke and experiment use the configured interval and epochs. Their errors
describe those runs; scientific accuracy or model superiority needs separate
acceptance criteria and repeated seeds.

With the bundled configuration, smoke trains three neural models for 100 epochs
each plus three GRNN references; experiment trains all twelve plus three GRNNs.
For a short pipeline check, copy the TOML
as shown above and set `dataset.segment_end_s = 59` and `training.epochs = 2`,
then pass that file with `--config`. The bundled configuration remains the
normal experiment baseline.

## Detach, Check Status, and Fetch

```bash
./scripts/run.sh RBD ResNet50 --epochs 100 --detach

# Replace <run_id> with the ID printed by the launch command.
RUN_ID='<run_id>'
./scripts/run.sh status --run "$RUN_ID"
./scripts/run.sh fetch --run "$RUN_ID"
```

Use the same `--config PATH` for status/fetch if the launch configuration
changes the remote host or workspace. Foreground runs fetch automatically
when they finish; detached runs need `fetch`.

| Status | Meaning |
|---|---|
| `STARTING` | Preparing the run and frozen environment |
| `WAITING_FOR_RESOURCE` | Waiting for the server's numerical-run lock |
| `RUNNING` | Executing generation, checks, or model commands |
| `EXPORTING` | Collecting deliverable files |
| `PASS` | Completed successfully |
| `FAIL` | Failed; inspect the report and retained logs |

Status exit codes are `0` for PASS, `1` for FAIL, and `2` while pending.
Fetch uses the final queried status and returns `1` on a transfer or delivery
verification error.
A successful detached launch returns `0` before the work completes. Ctrl-C
stops foreground waiting with exit code `130`; the remote worker continues.
Workers also continue after an SSH disconnect.

Fetch uses rsync checksums, then queries status again. If a pending run became
terminal during retrieval, it retrieves the final data/figure export again
before saving the final `report.json`. Terminal retrieval verifies any
visualization manifest's listed files and records the result in
`delivery-verification.json`; see [verification scope](result-visualization.md#input-checks-and-provenance).
A PASS selected/smoke/experiment run requires that manifest; missing delivery
evidence makes fetch fail. Quick runs may have no rendering manifest.

## Find Results and Diagnose Failures

| Location | Contents |
|---|---|
| `output/data/<run_id>/report.json` | Final status, checks, metrics, and failure details |
| `output/data/<run_id>/request.json` | Frozen configuration and selected model |
| `output/data/<run_id>/delivery-verification.json` | Verification of visualization manifest files, when present after terminal retrieval |
| `output/data/<run_id>/logs/` | Executed commands and their logs |
| `output/data/<run_id>/` | Reports, provenance, prediction CSVs, and history JSONs |
| `output/figure/<run_id>/` | Aggregate and per-model PDF figures |
| Remote run's `project/outputs/` | HDF5 datasets, checkpoints, and GRNN references |

HDF5 and `.pt` files remain on haiqin. Do not commit `output/` or `outputs/`.
See [result visualization](result-visualization.md) for figure meanings.

| Problem | Next action |
|---|---|
| Invalid command or configuration | Correct the named argument or TOML field; use `--dry-run` to inspect the request |
| SSH, raw-data, or uv preflight failure | Run `./scripts/run.sh check`; inspect the configured remote paths |
| CUDA or dependency failure | Read the logs and `environment.json` if present; follow the [frozen environment guide](remote-environment.md) |
| CUDA out of memory | Check other GPU workloads; run when enough memory is free, or lower `training.batch_size` in a complete TOML copy passed with `--config` |
| Dataset cache validation failure | Inspect the failure; use `--no-cache` for a fresh generation |
| Run failed after launch | Use `status` and `fetch` with its run ID to retrieve the report and available logs |
| Foreground waiting was interrupted | Use `status` with the printed run ID; fetch after completion |

For manual train/predict/plot commands, first [enter a completed frozen run](remote-environment.md#manual-commands-in-a-frozen-run).
For data fields and validation coverage, see the
[dataset contract](../datasets/hdf5-datasets-and-splits.md) and
[validation details](remote-environment.md#validation-coverage).
