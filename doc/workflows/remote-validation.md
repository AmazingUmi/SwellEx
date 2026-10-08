# Local Development and haiqin Validation

Edit source/configuration locally; run dataset generation, numerical tests,
GPU training, GRNN, and plotting on `haiqin` (`HQserver`). Dependencies have
one authority: `pyproject.toml` and `uv.lock`. The dispatcher imports only the
standard library; the worker uses the frozen Linux/CUDA environment. No MATLAB
runtime is needed.

## Dispatch and Retrieve

Run from the local project root with local uv and Python 3.11+, Git, SSH, and
rsync available:

```bash
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py check
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py validate --suite quick
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py validate --suite smoke
```

| Suite | Scope |
|---|---|
| `quick` | Snapshot/environment integrity, Python/CLI/model contracts, synthetic loaders, GRNN invariants, data-processing numerical oracles, visualization integrity/export tests |
| `smoke` | Quick checks plus aligned RBD/ELM/SCM generation, HDF5 validation, one complex CNN per method, prediction, GRNN build/predict, checkpoint resume, and result figures |
| `experiment` | The same workflow with all four registered architectures per method, using configured interval and epochs |

Default smoke uses S5 window starts 0–59 s, 1 s non-overlapping windows,
periodic 4:1 train/test split, two epochs, and GRNN 3-fold sigma selection.
Its MAE/RMSE demonstrate pipeline functionality; scientific accuracy or model
superiority requires independent acceptance criteria and repeated seeds.

Use `validate --suite experiment` for the configured experiment. Add
`--dry-run` to inspect a request without contacting haiqin, or `--no-cache` to
force generation. Foreground completion retrieves results automatically.
Long runs can detach:

```bash
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py validate --suite smoke --detach
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py status --run <run_id>
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py fetch --run <run_id>
```

Workers persist after SSH disconnects; numerical runs serialize on a server
lock. Status/fetch return 0 for PASS, 1 for FAIL, and 2 for pending work.
Ctrl-C stops foreground waiting and leaves the worker running. Failed launch,
dependency sync, or a dead worker produces FAIL with retained logs.

## Configuration and Frozen Environment

`scripts/localization_networks/workflow/haiqin.toml` defines the complete
configuration; unknown fields and invalid values are rejected. A global
`--config /absolute/path/experiment.toml` goes before the subcommand.

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

Each validate invocation snapshots eligible tracked and new source files,
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
