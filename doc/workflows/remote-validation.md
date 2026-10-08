# Run Models and Validation on haiqin

Run the commands below from the local project root. Source and configuration
are edited locally; generation, tests, training, prediction, and plotting run
on SSH host `haiqin`.

Local prerequisites: uv, Python 3.11+, Git, SSH, and rsync. The remote worker
uses the frozen dependencies in `pyproject.toml` and `uv.lock`.

## Select One Model

```bash
# ELM complex CNN, using the bundled configuration
./scripts/run.sh ELM complex_CNN

# Override the neural epoch count and seed
./scripts/run.sh SCM ResNet18 --epochs 100 --seed 2026

# SCM-GRNN reference building and prediction
./scripts/run.sh SCM GRNN
```

Both positional arguments are required. Names are case-sensitive:

| Argument | Accepted values |
|---|---|
| `METHOD` | `RBD`, `ELM`, `SCM` |
| `ARCH` for any method | `complex_CNN`, `real_CNN`, `ResNet18`, `ResNet50` |
| Additional SCM architecture | `GRNN` |

Each command performs these steps:

1. Freeze the current source and merged configuration in a new remote run.
2. Generate or reuse all three aligned RBD/ELM/SCM feature datasets, then validate them.
3. Train only the selected neural model, or build the SCM-GRNN reference.
4. Predict test ranges, check the prediction CSV, and render figures.
5. Retrieve logs, tables, reports, and figures to the local project.

Neural runs start fresh and predict with that run's best checkpoint. GRNN has
no epochs or optimizer. A selected-run PASS means the pipeline completed;
it does not apply an accuracy acceptance threshold.

## Options and Defaults

The bundled configuration uses S5 window starts from 0 through 59 s, two
neural epochs, batch size 8, seed 2026, and CUDA. These small defaults check
the pipeline. See [all command defaults](command-defaults.md) before choosing
a longer experiment.

| Option | Default | Use |
|---|---|---|
| `--config PATH` | Bundled `haiqin.toml` | Load a complete TOML configuration |
| `--epochs N` | `2` | Set a positive neural epoch count; rejected for GRNN |
| `--seed N` | `2026` | Set the run seed, from 0 to 4294967295 |
| `--detach` | Off | Launch and return the run ID immediately |
| `--dry-run` | Off | Print the merged request without contacting haiqin |
| `--no-cache` | Off | Regenerate all three feature datasets |

`--epochs` and `--seed` override the loaded TOML. Unknown names, options,
fields, and invalid values are rejected before launch.

To change other settings, copy and edit the complete configuration, then
inspect it before launching:

```bash
cp scripts/localization_networks/workflow/haiqin.toml /tmp/swellex-experiment.toml
# Edit /tmp/swellex-experiment.toml, then inspect the request:
./scripts/run.sh ELM complex_CNN --config /tmp/swellex-experiment.toml --dry-run
# Launch with the same configuration:
./scripts/run.sh ELM complex_CNN --config /tmp/swellex-experiment.toml
```

Learning rate and dropout are direct Python CLI options; `run.sh` does not
accept them. The workflow TOML also has no fields for them.

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
| `validate --suite smoke` | Quick checks, three-route generation/validation, one complex CNN per method, SCM-GRNN, prediction, checkpoint resume, and figures |
| `validate --suite experiment` | The same workflow with all four neural architectures per method, plus SCM-GRNN |

`validate` defaults to `smoke`. These suites accept `--config`, `--detach`,
`--dry-run`, and `--no-cache`; set epochs and seed in their TOML configuration.
Selected-model commands omit the global quick/numerical/resume checks.

Smoke and experiment use the configured interval and epochs. Their errors
describe those runs; scientific accuracy or model superiority needs separate
acceptance criteria and repeated seeds.

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

Status/fetch exit codes are `0` for PASS, `1` for FAIL, and `2` while pending.
A successful detached launch returns `0` before the work completes. Ctrl-C
stops foreground waiting with exit code `130`; the remote worker continues.
Workers also continue after an SSH disconnect.

## Find Results and Diagnose Failures

| Location | Contents |
|---|---|
| `output/data/<run_id>/report.json` | Final status, checks, metrics, and failure details |
| `output/data/<run_id>/request.json` | Frozen configuration and selected model |
| `output/data/<run_id>/logs/` | Executed commands and their logs |
| `output/data/<run_id>/` | Reports, provenance, prediction CSVs, and history JSONs |
| `output/figure/<run_id>/` | Aggregate and per-model PNG/PDF figures |
| Remote run's `project/outputs/` | HDF5 datasets, checkpoints, and GRNN references |

HDF5 and `.pt` files remain on haiqin. Do not commit `output/` or `outputs/`.
See [result visualization](result-visualization.md) for figure meanings.

| Problem | Next action |
|---|---|
| Invalid command or configuration | Correct the named argument or TOML field; use `--dry-run` to inspect the request |
| SSH, raw-data, or uv preflight failure | Run `./scripts/run.sh check`; inspect the configured remote paths |
| CUDA or dependency failure | Read the logs and `environment.json` if present; follow the [frozen environment guide](remote-environment.md) |
| Dataset cache validation failure | Inspect the failure; use `--no-cache` for a fresh generation |
| Run failed after launch | Use `status` and `fetch` with its run ID to retrieve the report and available logs |
| Foreground waiting was interrupted | Use `status` with the printed run ID; fetch after completion |

For manual train/predict/plot commands, first [enter a completed frozen run](remote-environment.md#manual-commands-in-a-frozen-run).
For data fields and validation coverage, see the
[dataset contract](../datasets/hdf5-datasets-and-splits.md) and
[validation details](remote-environment.md#validation-coverage).
