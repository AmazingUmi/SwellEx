# Frozen Environment and Remote Run Files

The dispatcher snapshots local source (`scripts/`, `simulation/`, `doc/`, and root project files) and executes on `haiqin` without modifying the remote checkout or `Origindata/`.

## Configured Paths and Runtime

| TOML parameter | Bundled value | Role |
|---|---|---|
| `remote.repository` | `/mnt/SDD/project/SwellEx` | Existing remote checkout providing `Origindata` |
| `remote.workspace` | `/mnt/SDD/project/SwellEx_validation` | Isolated runs, uv environments, and dataset cache |
| `remote.base_python` | `/home/yiyang-lu/miniforge3/envs/pytorch/bin/python` | Bootstrap interpreter for `uv` |
| `remote.uv` | `/mnt/SDD/project/SwellEx_validation/runtime_profile/tools/uv-0.12.19/uv-x86_64-unknown-linux-gnu/uv` | Locked `uv` binary |

Remote execution uses Linux x86-64, Python 3.14, and PyTorch CUDA 12.8 wheels via `uv run --frozen` with `UV_PROJECT_ENVIRONMENT` under `environments/<uv_lock_sha256>/.venv`. Update `pyproject.toml` and `uv.lock` together. Each run writes `environment.json` with exact OS, GPU, interpreter, and package hashes.

## Manual Commands in a Frozen Run

Inspect `output/data/<run_id>/request.json` (`config.remote.workspace`) and `environment.json` (`uv_environment`), then enter the snapshot on `haiqin`:

```bash
ssh haiqin
export SWELLEX_RUN='<workspace>/runs/<run_id>'
export SWELLEX_PYTHON='<uv_environment>/bin/python'
cd "$SWELLEX_RUN/project"
```

Use `"$SWELLEX_PYTHON"` for manual [neural](../network_training/neural-cli.md#manual-train-and-predict), [GRNN](../network_training/grnn-cli.md#manual-build-and-predict), or [plotting](result-visualization.md#render-manually-on-haiqin) commands; retrieve manual outputs separately.

## Source Snapshot, Cache, and Validation

| Path under `remote.workspace` | Contents |
|---|---|
| `runs/<run_id>/project/` | Frozen source, symlink to `Origindata`, and intermediate `outputs/` |
| `runs/<run_id>/{logs,reports,export}/` | Command logs, phase reports, and staged `export/{data,figure}/` |
| `datasets/<METHOD>/<fingerprint>/` | Per-method hash-verified HDF5 cache and provenance |

Per-method dataset cache reuse requires matching generation/validation code, method-scoped config (`[rbd]` and `ctd_file` do not pollute `SCM`, and `[scm]` does not pollute `RBD`), `uv.lock`, package versions, and raw-file SHA256 hashes (CTD files are required and hashed only when `RBD` is active). Validation phases cover `Quick` (runtime, model forward/backward, loaders, GRNN, workflow/plot tests), `Numerical` (feature math oracles), `HDF5` (finite values, physical axes, pair order, split separation), and route/cross-method checks.
