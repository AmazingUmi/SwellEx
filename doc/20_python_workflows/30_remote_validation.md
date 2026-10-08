# Local Development and haiqin Validation

Edit source and configuration locally. Python dataset generation, numerical
checks, GPU training, GRNN building, and prediction run on SSH host `haiqin`.
Dependencies are defined in `pyproject.toml` and pinned in `uv.lock`. Numerical
execution uses the Linux x86-64 CUDA 12.8 profile; the local dispatcher imports
only the Python standard library. Run commands from the project root.

## Commands

The local dispatcher uses uv without syncing the remote numerical dependencies:

```bash
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py check
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py validate --suite quick
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py validate --suite smoke
```

Append `--dry-run` to inspect a frozen request without contacting haiqin.
`quick` checks snapshot integrity, the locked environment, workflow unit tests,
imports/CLI, all 12 models' forward/backward contracts, synthetic HDF5 loaders,
GRNN invariants, and data-processing numerical oracles.

`smoke` adds RBD/ELM/SCM generation, complete HDF5 validation, one complex CNN
per route, prediction on the same test windows, SCM-GRNN, and checkpoint resume.
The default covers window starts 0–59 s and two training epochs. MAE/RMSE are
functional evidence; scientific accuracy acceptance requires a separate design.

Long runs can detach:

```bash
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py validate --suite smoke --detach
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py status --run <run_id>
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py fetch --run <run_id>
```

The remote worker persists after SSH disconnects. Numerical runs serialize on a
server lock. Status/fetch return 0 for PASS, 1 for FAIL, and 2 for pending work.
Ctrl-C stops foreground waiting and leaves the remote task running. A dead
worker or failed dependency sync produces FAIL, with retained logs.

## Configuration and uv

`scripts/localization_networks/workflow/haiqin.toml` is the complete canonical
configuration. Unknown fields and invalid values are rejected. Override it
with the global `--config /absolute/path/experiment.toml` option.

`experiment` uses the configured interval and epochs for all four registered
architectures per route plus GRNN. Repeated seeds and scientific acceptance
criteria belong in the experiment design. Cross-route comparison currently
requires S5, Ns=1, no snapshot overlap, and non-overlapping base windows.
Adaptive frequency selection fits valid training windows only, then freezes
that axis for testing.

The dispatcher starts `uv run --frozen` on each remote source snapshot, using
the existing Python 3.14.4 interpreter as the base. Each lockfile SHA-256 has
its own virtual environment under `environments/<uv_lock_sha256>/.venv`.
Shared downloads live in `uv_cache/`. Python downloads are disabled. Runtime
reports record the actual interpreter, package versions, CUDA device, uv
version, and lock hash. Update dependencies by editing `pyproject.toml` and
running `uv lock`, then validate the resulting lock on haiqin.

## Source, Data, and Results

```text
SSH host:       haiqin (hostname HQserver)
Repository:     /mnt/SDD/project/SwellEx
Validation:     /mnt/SDD/project/SwellEx_validation
Base Python:    /home/yiyang-lu/miniforge3/envs/pytorch/bin/python
```

Every invocation freezes eligible tracked and new local source files,
including uncommitted changes, plus the uv project and lock files. SHA-256
checks establish the exact source used by the worker:

```text
SwellEx_validation/
  runs/<run_id>/
    project/           fixed source; Origindata links to remote raw data
    manifest.json      file hashes, Git HEAD/status, configuration identity
    request.json
    generation_config.json
    logs/
    reports/
    report.json
    export/            small artifacts for retrieval
  environments/<uv_lock_sha256>/.venv/
  uv_cache/
  datasets/<fingerprint>/Datasets/
  datasets/<fingerprint>/provenance.json
```

The existing remote checkout supplies raw data. Source symlinks are rejected.
Each run has independent network outputs; baseline training uses `--no-resume`,
and resume is verified in a separate output directory.

Dataset reuse requires identical generation configuration, Python generator
and validation code, uv lock, numerical package versions, and raw-input hashes.
Cache content is hash-checked and validated again before use. `--no-cache`
forces fresh generation. Reports retain the original producing run ID.

Reports, logs, history, prediction CSVs, and plots are retrieved into local
`outputs/remote_validation/<run_id>/`. HDF5 and `.pt` models stay on haiqin.
All outputs remain ignored by Git. Exact commands, durations, failure details,
and test-window metrics are retained; failures stop dependent phases.

Data processing runs in `scripts/data_processing/`, localization networks in
`scripts/localization_networks/`. The former MATLAB implementation is retained
under `data_processing/matlab_reference/` for source comparison. Python reads
original numeric MAT files with SciPy/h5py and reads CTD values from the raw
`Origindata/environments/ctds/i9605.prn`; execution requires Python only.
