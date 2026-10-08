# Local Development and haiqin Validation

Edit source/configuration locally; run dataset generation, numerical tests,
GPU training, GRNN, and plotting on `haiqin` (`HQserver`). Dependencies have
one authority: `pyproject.toml` and `uv.lock`. The dispatcher imports only the
standard library; the worker uses the frozen Linux/CUDA environment. No MATLAB
runtime is needed.

## Select One Model

Run from the local project root with local uv and Python 3.11+, Git, SSH, and
rsync available:

```bash
./scripts/run.sh RBD complex_CNN
./scripts/run.sh ELM ResNet18 --epochs 100 --seed 2026
./scripts/run.sh SCM GRNN --config /absolute/path/experiment.toml
```

The shell entry delegates to the standard-library dispatcher. `METHOD` must be
`RBD`, `ELM`, or `SCM`; `ARCH` must be exactly `complex_CNN`, `real_CNN`,
`ResNet18`, `ResNet50`, or `GRNN`. GRNN is supported only for SCM and rejects
`--epochs`; it builds reference samples and selects sigma instead of training
with an optimizer. METHOD and ARCH are required and have no defaults.
Unknown names, options, or invalid configuration fail before launch.

| Selected-run option | Default | Behavior when supplied |
|---|---|---|
| `--config PATH` | Bundled `scripts/localization_networks/workflow/haiqin.toml` | Load a strict TOML configuration |
| `--epochs N` | Configuration's `training.epochs` | Override positive neural epoch count |
| `--seed N` | Configuration's `validation.seed` | Override the run seed, from 0 to 4294967295 |
| `--detach` | Off: wait in the foreground and retrieve results automatically | Launch remotely and return the run ID without waiting |
| `--dry-run` | Off: execute the request | Inspect the merged request without contacting haiqin |
| `--no-cache` | Off: allow hash-verified dataset reuse | Force regeneration of all three feature routes |

CLI overrides take precedence over TOML. `request.json` freezes the merged
configuration and canonical method/architecture/model selection; the report
records the selection and configuration hash. The selected workflow validates
or reuses aligned RBD/ELM/SCM datasets,
executes only the chosen model, checks its prediction CSV, renders its figures,
and retrieves its results. Dataset generation still produces all three routes;
single-route generation, automatic resume, and training-only runs are outside
this command's scope. Global quick checks, numerical oracles, and checkpoint
resume checks belong to the validation suites below.

A PASS selected report means the configured pipeline completed; no accuracy
acceptance threshold is applied. The effective defaults are listed below.

## Command Defaults

These values apply to `./scripts/run.sh METHOD ARCH` with the bundled
[haiqin.toml](../../scripts/localization_networks/workflow/haiqin.toml).
CLI `--epochs` and `--seed` override the selected TOML values; change the other
TOML settings with `--config PATH`. The merged configuration is frozen in
`request.json`. The bundled data interval and epoch count are for a small
pipeline check.

| Execution setting | Default |
|---|---|
| SSH host / device | `haiqin` / `cuda`; an unavailable requested GPU fails |
| Seed (`validation.seed`) | `2026` |
| CPU threads (`validation.threads`) | `4` |
| Quick-check timeout (`quick_timeout_s`) | `600` s |
| Generation timeout (`generation_timeout_s`) | `900` s |
| Each train/build/predict command timeout (`training_timeout_s`) | `900` s |

Remote paths and the frozen interpreter are defined in
[Configuration and Frozen Environment](#configuration-and-frozen-environment).

| Dataset setting | Default |
|---|---|
| Event (`event_name`) | `S5` |
| Sampling rate (`fs`) | `1500` Hz |
| Window start range (`segment_start_s`, `segment_end_s`) | `0`–`59` s, inclusive window starts |
| Window duration / step (`segment_duration_s`, `segment_step_s`) | `1.0` s / `1.0` s |
| Snapshots per segment / overlap (`num_snapshots_per_segment`, `snapshot_overlap_count`) | `1` / `0` |
| Frequency modes (`frequency_selection_modes`) | `["deep", "shallow"]` |
| Split (`split_strategy`, `train_test_ratio`) | `periodic`, train:test = `4:1` |
| Range-nearby settings (`train_side`, `half_duration_s`, `gap_s`, `test_step_s`) | `before`, `800` s, `15` s, `1` s; inactive for `periodic` |
| Variant suffix (`manual_dataset_variant_tag`) | Empty string |

RBD defaults are `beam_selection="best"`, `frequency_estimation="selected"`,
`use_plane_wave=false`, and `normalize_spectrum=false`. All three feature
routes are generated or validated; only the requested model is evaluated.

| Neural setting | Effective default |
|---|---|
| Epochs (`training.epochs`) | `2` |
| Train / predict batch size (`training.batch_size`) | `8` / `8` |
| CNN base channels (`training.base_channels`) | `8`; ResNet uses its architecture's channel layout |
| Loss space / SmoothL1 beta (`training.loss_space`, `training.huber_beta`) | `km` / `0.5` km |
| Validation fraction (`training.val_fraction`) | `0.25`, split from training samples when no validation files exist |
| Learning rate / weight decay | `3e-4` / `1e-4`, inherited from the network CLI |
| Dropout | `0.15`, inherited from the network CLI |
| DataLoader workers | `0` |

Learning rate and dropout settings above are network CLI defaults, not fields
in the workflow TOML or options accepted by `run.sh`. Each selected neural run
starts fresh and predicts with that run's best checkpoint. Input channel
normalization is enabled for complex CNN/ResNet and disabled for real CNN.

| SCM-GRNN setting | Default |
|---|---|
| Cross-validation folds (`grnn.cv_folds`) | `3` |
| Sigma candidates (`grnn.sigma_candidates`) | `[0.01, 0.1, 1.0]` |
| Build / predict batch size (`grnn.batch_size`) | `64` / `64` |
| Feature-coordinate standardization / per-sample channel normalization | Off / off |
| Epochs / optimizer | None; use build/predict |

GRNN uses the shared seed above, builds a fresh reference, and predicts with
that explicit reference. Automatic figures are PNG at 300 dpi and PDF.
Logs/tables/reports go to local `output/data/<run_id>/`; figures go to
`output/figure/<run_id>/`. HDF5 and model/reference artifacts remain on haiqin.

## Validate and Retrieve

```bash
./scripts/run.sh check
./scripts/run.sh validate --suite quick
./scripts/run.sh validate --suite smoke
./scripts/run.sh validate --suite experiment
```

| Suite | Scope |
|---|---|
| `quick` | Snapshot/environment integrity, Python/CLI/model contracts, synthetic loaders, GRNN invariants, data-processing numerical oracles, visualization integrity/export tests |
| `smoke` | Quick checks plus aligned RBD/ELM/SCM generation, HDF5 validation, one complex CNN per method, prediction, GRNN build/predict, checkpoint resume, and result figures |
| `experiment` | The same workflow with all four registered architectures per method, using configured interval and epochs |

`./scripts/run.sh validate` defaults to `--suite smoke`. Validation suites use
the same [configured defaults](#command-defaults). Smoke MAE/RMSE demonstrate
pipeline functionality; scientific accuracy or model superiority requires
independent acceptance criteria and repeated seeds.

Use `validate --suite experiment` for the configured experiment. Add
`--dry-run` to inspect a request without contacting haiqin, or `--no-cache` to
force generation. Foreground completion retrieves results automatically.
Long runs can detach:

```bash
./scripts/run.sh RBD ResNet50 --epochs 100 --detach
./scripts/run.sh status --run <run_id>
./scripts/run.sh fetch --run <run_id>
```

Workers persist after SSH disconnects; numerical runs serialize on a server
lock. Status/fetch return 0 for PASS, 1 for FAIL, and 2 for pending work.
Ctrl-C stops foreground waiting and leaves the worker running. Failed launch,
dependency sync, or a dead worker produces FAIL with retained logs.
Status exposes launch, resource-waiting, execution, and export states until a
final PASS/FAIL is available. Fetch a detached run after it completes to populate
the local delivery roots.

## Configuration and Frozen Environment

`scripts/localization_networks/workflow/haiqin.toml` defines the complete
configuration; unknown fields and invalid values are rejected. Pass
`--config /absolute/path/experiment.toml` to the public shell command. The
underlying dispatcher remains available as
`uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py`;
its global `--config` option goes before the subcommand.

| Setting | Current configured location |
|---|---|
| Remote repository/raw data | `/mnt/SDD/project/SwellEx` |
| Validation workspace | `/mnt/SDD/project/SwellEx_validation` |
| Base Python | `/home/yiyang-lu/miniforge3/envs/pytorch/bin/python` |
| Remote uv | `/mnt/SDD/project/SwellEx_validation/runtime_profile/tools/uv-0.12.19/uv-x86_64-unknown-linux-gnu/uv` |
| Runtime profile | Linux x86-64, Python 3.14, official PyTorch/torchvision CUDA 12.8 wheels |
| Configured hardware | NVIDIA RTX A5000 on haiqin |

The local development machine is macOS. Its numerical environment is not used
for this workflow. Versions in `pyproject.toml`/`uv.lock` define dependencies;
the run's `environment.json` is authoritative for the actual installed runtime
and operating system.

Remote uv is not on the default SSH PATH. The dispatcher invokes that absolute
binary with `uv run --frozen`, the configured base Python, and the snapshot
project. `UV_PROJECT_ENVIRONMENT` selects
`environments/<uv_lock_sha256>/.venv`; downloads share `uv_cache/`, and Python
downloads are disabled. Do not activate the base Conda environment or create
an unrelated `.venv` for manual commands.

`environment.json` records the actual interpreter/packages, CUDA device, uv
version, `uv_lock_sha256`, and `uv_environment`. The configured device is
required; the validation workflow rejects silent CPU fallback. If CUDA fails
only inside a sandbox, recheck outside it before changing drivers/packages.
Update dependency specifications and the lockfile together, then validate the
frozen environment on haiqin.

## Manual Commands in a Frozen Run

Use a completed run and read its `uv_environment` from retrieved
`output/data/<run_id>/environment.json`. On haiqin, replace the placeholders
below with that run ID and exact recorded environment path:

```bash
ssh haiqin
export SWELLEX_RUN="/mnt/SDD/project/SwellEx_validation/runs/<run_id>"
export SWELLEX_PYTHON="<uv_environment>/bin/python"
cd "$SWELLEX_RUN/project"
```

`SWELLEX_PYTHON` is the existing locked environment's interpreter, not the
base Python. Follow [training/prediction](training-and-prediction.md),
[GRNN](standalone-scm-grnn.md), or [visualization](result-visualization.md) for
operation-specific commands. Additional manual results do not become part of
an already completed run's automatic report/export.

## Snapshot, Data, and Delivery

Each selected-run or validate invocation snapshots eligible tracked and new
source files,
including uncommitted changes and uv files. Hashes and Git state are stored
in `manifest.json`; source symlinks are rejected. The remote repository supplies
raw `Origindata` through a link, and is never overwritten.

| Remote run path | Contents |
|---|---|
| `runs/<run_id>/project/` | Frozen source, raw-data link, remote `outputs/Datasets/` and `outputs/networks_results/` |
| `runs/<run_id>/` | Request, generation config, source/data identities, environment and final report |
| `runs/<run_id>/logs/`, `reports/` | Exact commands, durations, checks, metrics, and failures |
| `runs/<run_id>/export/data/` | Small JSON/CSV/log artifacts |
| `runs/<run_id>/export/figure/` | Model and aggregate PNG/PDF figures |
| `datasets/<fingerprint>/` | Cached HDF5 and provenance |

Dataset reuse requires matching generator/validation code, configuration, uv
lock, numerical package versions, and raw-input hashes. Content is hash-checked
and validated again; reports retain the producing run ID. Cross-method
acceptance requires aligned S5/Ns=1, non-overlapping windows, splits, labels,
and frequencies. Adaptive frequencies fit training windows only.

Retrieval preserves relative artifact paths: logs, reports, history, CSVs,
configuration and provenance go to `output/data/<run_id>/`; all figures go to
`output/figure/<run_id>/`. HDF5 and `.pt` artifacts remain on haiqin. Neither
`output/` nor `outputs/` is committed. A failed phase stops dependent work;
retained reports distinguish PASS, FAIL, and intentionally skipped generation.
The [visualization guide](result-visualization.md) defines figure semantics
and delivery manifests.

## Validation Reports and Boundaries

Python quick checks cover all 12 real model classes at their default scale,
forward/backward contracts, loader storage-axis handling, shared paths/splits,
workflow unit tests, and GRNN invariants. CUDA requests fail if unavailable.
CPU threads use `OMP_NUM_THREADS`, capped at four.

HDF5 checks validate complete finite features/labels, raw-track label
interpolation, frequencies, source-window times/indices, column-major ELM/SCM
pair ordering, and non-overlapping train/test sampled intervals. All three
methods must agree on labels, masks, times, frequencies, and element depths;
RBD single-window versus different ELM/SCM snapshot geometry fails this strict
cross-method check. See the [dataset contract](../datasets/hdf5-datasets-and-splits.md)
for fields and RBD feature-normalization tolerances.

`validation/validate.py datasets --expected-config` accepts a canonical
Python generation JSON or flat `num_elements`/`freq_hz`/`min_samples` constraints;
unknown fields fail. Full configuration checks source membership, window/sample
endpoints, FFT grid, split membership, and generation metadata. Adaptive mode
checks the frozen frequency union/count/range, without recomputing the raw
signal's spectral ranking.

Reports contain `status: PASS|FAIL` and individual checks with timing/details or
error stacks; failure returns nonzero and preserves earlier evidence.
Dataset reports include directory lists by method, plus a singular mapping
when exactly one dataset directory exists per method. Numerical oracles and
smoke acceptance validate implementation contracts; they do not certify
scientific accuracy or the propagation model.
