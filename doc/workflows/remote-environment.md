# Frozen Environment and Remote Run Files

The dispatcher snapshots local source and runs it in a locked Linux/CUDA
environment on `haiqin`. Use the [shell command guide](remote-validation.md)
for everyday operations; this reference covers runtime setup, manual commands,
and run integrity.

## Configured Paths and Runtime

The bundled [haiqin_full_interleaved.toml](../../scripts/base_pack/localization_networks/workflow/haiqin_full_interleaved.toml)
defines these locations:

| Setting | Configured value |
|---|---|
| `remote.repository` | `/mnt/SDD/project/SwellEx` |
| `remote.workspace` | `/mnt/SDD/project/SwellEx_validation` |
| `remote.base_python` | `/home/yiyang-lu/miniforge3/envs/pytorch/bin/python` |
| `remote.uv` | `/mnt/SDD/project/SwellEx_validation/runtime_profile/tools/uv-0.12.19/uv-x86_64-unknown-linux-gnu/uv` |

The frozen runtime profile is Linux x86-64, Python 3.14, and official
PyTorch/torchvision CUDA 12.8 wheels. Validation has run on the NVIDIA RTX A5000
on haiqin. A run's `environment.json` records its actual interpreter, packages,
operating system, CUDA device, uv version, `uv_lock_sha256`, and `uv_environment`.

The dispatcher uses only the standard library locally. On haiqin it invokes
the absolute uv binary with `uv run --frozen` against the source snapshot:

- `UV_PROJECT_ENVIRONMENT` selects `environments/<uv_lock_sha256>/.venv`.
- Downloads share `uv_cache/`; Python downloads are disabled.
- `pyproject.toml` and `uv.lock` are the dependency authority. Update them together.
- MATLAB files are static references; the workflow does not execute MATLAB.

The requested CUDA device must be available; the workflow rejects silent CPU
fallback. If CUDA fails only inside a sandbox, recheck outside it before
changing drivers or packages.

Worker commands, including aggregate rendering, run in separate process groups.
Timeouts and interruptions stop the whole group; remaining processes are
force-killed after five seconds, even if the group leader has exited. Rendering
uses `validation.quick_timeout_s`; train/build/predict commands use
`validation.training_timeout_s`.
Nested validation/test runners propagate termination into their child groups
and use shorter cleanup grace periods so children stop before the parent.

## Manual Commands in a Frozen Run

1. Choose a PASS selected/smoke/experiment run; these runs have datasets for manual commands.
2. Open local `output/data/<run_id>/request.json`; combine `config.remote.workspace`
   and `run_id` as `<workspace>/runs/<run_id>` to get the remote run directory.
3. Open `environment.json` in the same directory and copy `uv_environment`.
4. On haiqin, replace both placeholders below with those exact paths.

```bash
ssh haiqin
export SWELLEX_RUN='<remote_run_dir>'
export SWELLEX_PYTHON='<uv_environment>/bin/python'
cd "$SWELLEX_RUN/project"
```

Use this existing locked interpreter for
[neural train/predict](training-and-prediction.md#manual-train-and-predict),
[GRNN build/predict](standalone-grnn.md#manual-build-and-predict), or
[manual plotting](result-visualization.md#render-manually-on-haiqin).
Do not activate the base Conda environment or create another `.venv`.

Additional manual results do not update a completed run's report or automatic
export. Retrieve those files separately; `fetch` retrieves the recorded export.

## Source Snapshot and Remote Files

Every model or validation run snapshots eligible tracked and new source,
including uncommitted changes and uv files. `manifest.json` records file hashes
and Git state. Source symlinks are rejected.

The snapshot links to the remote repository's raw `Origindata`. The existing
checkout and original inputs are never overwritten.

Paths below are relative to `remote.workspace`:

| Path | Contents |
|---|---|
| `runs/<run_id>/project/` | Frozen source, raw-data link, and intermediate `outputs/` |
| `runs/<run_id>/` | Request, generation config, source/data identities, environment, and final report |
| `runs/<run_id>/logs/` | Executed commands and logs |
| `runs/<run_id>/reports/` | Checks, metrics, and failures |
| `runs/<run_id>/export/data/` | Small JSON/CSV/log files for retrieval |
| `runs/<run_id>/export/figure/` | PDF figures for retrieval |
| `datasets/<fingerprint>/` | Cached HDF5 and provenance |

The underlying dispatcher can also be invoked directly:

```bash
uv run --no-project --python python3 scripts/base_pack/localization_networks/workflow/remote.py \
  --config /tmp/swellex-experiment.toml validate --suite smoke
```

Both the dispatcher and shell entry accept `--config` before or after the
subcommand. The public shell requires the pack first; its options can follow
that pack or `METHOD ARCH`. Direct Python entry points select a pack through
their source path. Use the selected pack's configuration and entry point.

## Dataset Cache and Alignment

Reuse requires matching generation/validation code, configuration, uv lock,
numerical package versions, and raw-input hashes. Cached content is hash-checked
and validated again; the report retains the producing run ID.
The validator fingerprint includes all implementation Python files under
`validation/`, plus `common/h5_utils.py` and `common/input_contract.py`.
Validation test files are excluded from that fingerprint.

Cross-method validation requires aligned S5/Ns=1 datasets with non-overlapping
windows and matching splits, labels, frequencies, times, and element depths.
Adaptive frequencies are fitted using training windows only. See the
[dataset contract](../datasets/hdf5-datasets-and-splits.md).

## Validation Coverage

| Check group | Coverage |
|---|---|
| Python quick checks | All 12 neural classes at default scale, forward/backward contracts, loaders, shared paths/splits, workflow tests, GRNN invariants, and visualization tests |
| Numerical oracles | Data-processing calculations checked against independent numerical expectations |
| HDF5 checks | Finite features/labels, raw-track label interpolation, frequencies, window times/indices, ELM/SCM pair order, and non-overlapping sampled train/test intervals |
| Cross-method checks | Equal labels, masks, times, frequencies, and element depths across RBD/ELM/SCM |

RBD single-window geometry paired with different ELM/SCM snapshot geometry
fails the cross-method check. Quick checks cap PyTorch threads at four; other
worker commands use the configured `validation.threads` value.

`validation/validate.py datasets --expected-config` accepts either a canonical
generation JSON or flat `num_elements`/`freq_hz`/`min_samples` constraints.
Unknown fields fail. Full configuration validation checks source membership,
window/sample endpoints, FFT grid, splits, and generation metadata. Adaptive
mode checks the frozen frequency union, count, and range; it does not recompute
the raw signal's spectral ranking.

Reports preserve timings, details, error stacks, and evidence from earlier
phases. A failed phase stops dependent work; reused generation is marked
skipped. Dataset reports list directories by method and add a singular mapping
when each method has exactly one directory.

These checks establish implementation contracts. They do not certify
scientific range accuracy or the propagation model. See
[project progress](../overview/project-progress.md) for recorded evidence and
remaining scientific work.
