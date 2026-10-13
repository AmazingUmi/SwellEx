# Frozen Environment and Remote Run Files

The dispatcher freezes local source and runs it in a Linux/CUDA environment
on `haiqin`. Everyday commands are in the [shell guide](remote-validation.md).

## Configured Paths and Runtime

| TOML parameter | Bundled value | Role |
|---|---|---|
| `remote.repository` | `/mnt/SDD/project/SwellEx` | Existing checkout supplying `Origindata` |
| `remote.workspace` | `/mnt/SDD/project/SwellEx_validation` | Isolated runs, environments, and cache |
| `remote.base_python` | `/home/yiyang-lu/miniforge3/envs/pytorch/bin/python` | Base interpreter for uv |
| `remote.uv` | `/mnt/SDD/project/SwellEx_validation/runtime_profile/tools/uv-0.12.19/uv-x86_64-unknown-linux-gnu/uv` | Absolute uv executable |

These are configured paths, not a live server-health check; use `PACK check`.
The locked profile is Linux x86-64, Python 3.14, and PyTorch/torchvision CUDA
12.8 wheels. Each run's `environment.json` records the actual interpreter,
packages, OS, GPU, uv version, lock hash, and environment path.

The worker uses `uv run --frozen`, with `UV_PROJECT_ENVIRONMENT` selecting
`environments/<uv_lock_sha256>/.venv`, shared `uv_cache/`, and Python downloads
disabled. Update `pyproject.toml` and `uv.lock` together. MATLAB is not executed.
Unavailable requested CUDA fails; if it fails only inside a sandbox, recheck
outside it before changing drivers/packages.

Commands use separate process groups. Timeouts terminate the group and kill
remaining processes after five seconds; nested test runners propagate termination.
Timeout settings are in [configuration defaults](command-defaults.md#execution).

## Manual Commands in a Frozen Run

Choose a PASS selected/smoke/experiment run with the desired pack and datasets.
Read its local `output/data/<run_id>/request.json` for
`config.remote.workspace`; the remote run is `<workspace>/runs/<run_id>`.
Read `environment.json` for the exact `uv_environment`.

```bash
ssh haiqin
export SWELLEX_RUN='<remote_run_dir>'
export SWELLEX_PYTHON='<uv_environment>/bin/python'
cd "$SWELLEX_RUN/project"
```

Use this interpreter for [neural commands](training-and-prediction.md#manual-train-and-predict),
[GRNN commands](standalone-grnn.md#manual-build-and-predict), or
[plotting](result-visualization.md#render-manually-on-haiqin).
Use the entry paths present in that snapshot; pre-pack runs have historical paths.
Manual results require separate retrieval and do not update the completed
run's report or recorded export.

## Source Snapshot and Remote Files

Snapshots include eligible tracked/new source and uncommitted changes.
`manifest.json` records hashes and Git state; source symlinks are rejected.
The snapshot links to the existing raw `Origindata`; neither the remote
checkout nor original inputs are overwritten.

| Path relative to `remote.workspace` | Contents |
|---|---|
| `runs/<run_id>/project/` | Frozen source, raw-data link, and intermediate `outputs/` |
| `runs/<run_id>/` | Request, environment, identities, and report |
| `runs/<run_id>/logs/` | Commands and logs |
| `runs/<run_id>/reports/` | Checks, metrics, and failures |
| `runs/<run_id>/export/data/` | JSON/CSV/log files for retrieval |
| `runs/<run_id>/export/figure/` | PDF figures for retrieval |
| `datasets/<fingerprint>/` | Cached HDF5 and provenance |

## Dataset Cache and Alignment

Reuse requires matching generation/validation code, configuration, uv lock,
numerical package versions, and raw-input hashes. Cached content is hash-checked
and validated again; the report retains the producing run ID.
Cross-method comparisons require S5/Ns=1, disjoint windows, equal split/sample
identities, labels, times, frequencies, and array depths. See the
[dataset contract](../datasets/hdf5-datasets-and-splits.md).

## Validation Coverage

| Group | Coverage |
|---|---|
| Quick | Runtime, registered model forward/backward contracts, loaders, GRNN, workflow, and plots; joint adds fusion/artifact checks |
| Numerical | Feature calculations against independent numerical expectations |
| HDF5 | Finite features/labels, raw-track label interpolation, physical axes, times/indices, pair order, and sampled split separation |
| Cross-method | Matching labels, masks, times, frequencies, and array depths |

The dataset validator's `--expected-config` accepts canonical generation JSON
or flat `num_elements`/`freq_hz`/`min_samples` constraints; unknown fields fail.
Full-config checks include source membership, window endpoints, FFT grid,
splits, and metadata. Adaptive mode checks the frozen axis/count/range without
recomputing raw-signal spectral rankings.

A failed phase stops dependent work; verified generation reuse is marked SKIP.
Reports retain timings, errors, and earlier evidence. These checks establish
implementation contracts; [scientific validation](../overview/project-progress.md)
requires separate evidence.
