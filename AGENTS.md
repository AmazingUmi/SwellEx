# AGENTS.md

Workspace instructions for agents working in SwellEx.

## Purpose and Layout

Range regression on SwellEx-96 VLA data. Python builds HDF5 features from the
original experiment files; PyTorch CNN/ResNet models regress range in km.

```text
Origindata/                          original inputs; never committed
scripts/data_processing/            Python generation, numerical oracles, configuration
scripts/data_processing/matlab_reference/  static implementation reference
scripts/localization_networks/common/      shared paths, HDF5, train/predict helpers
scripts/localization_networks/{RBD,ELM,SCM}_method/  independent network packages
scripts/localization_networks/0_GRNN_related/       standalone SCM-GRNN
scripts/localization_networks/workflow/            local dispatch/remote worker
scripts/localization_networks/validation/          contract validation
simulation/                         propagation model inputs
outputs/                            generated data/results; never committed
doc/                                categorized documentation
pyproject.toml / uv.lock             canonical dependencies and frozen versions
```

RBD `X` is `[sample, element, frequency, real_imag]`; ELM/SCM use
`[sample, pair, frequency, real_imag]`. ELM pairs are strict upper triangle
`i<j`; SCM includes diagonal `i<=j`. Pair ordering is column-major and stored
1-based. GRNN is standalone `build`/`predict`, with no epochs or optimizer.

## Development and Execution

Edit code/config/docs locally. Run data generation, numerical tests, GPU
training and prediction on SSH host `haiqin` (`HQserver`). Use uv; no MATLAB
runtime is required. From the local project root:

```bash
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py validate --suite quick
uv run --no-project --python python3 scripts/localization_networks/workflow/remote.py validate --suite smoke
```

The local dispatcher imports only the standard library. Configuration is
`scripts/localization_networks/workflow/haiqin.toml`. Source snapshots include
uncommitted/new files; never overwrite the remote checkout or original inputs.
Remote `uv run --frozen` creates a separate environment per lockfile hash.

Remote repository: `/mnt/SDD/project/SwellEx`; validation workspace:
`/mnt/SDD/project/SwellEx_validation`. Base Python:
`/home/yiyang-lu/miniforge3/envs/pytorch/bin/python` (3.14.4). GPU: RTX A5000.
Runtime package/CUDA versions are verified in each run's environment report.
If CUDA appears unavailable only in a sandbox, recheck outside that sandbox
before changing drivers/packages. Reject silent CPU fallback.

Standalone commands inside a configured remote uv project:

```bash
uv run --frozen python scripts/data_processing/generate.py --config generation_config.json
uv run --frozen python scripts/localization_networks/RBD_method/Network_main.py train --model complex_cnn_range --data <dataset>
uv run --frozen python scripts/localization_networks/RBD_method/Network_main.py predict --model complex_cnn_range --data <dataset>
uv run --frozen python scripts/localization_networks/0_GRNN_related/GRNN_main.py build --data <scm_dataset>
```

## Architecture

- Use canonical imports `scripts.localization_networks.*`; shared network
  logic belongs in `common/`, method-specific tensor layouts in each package.
  Reuse `common/h5_utils.py` for paths/splits.
- Generation configuration is defined in `data_processing/generation_config.py`.
  Unknown/invalid fields are rejected. Adaptive frequencies fit training windows.
- Each method owns `network/model.py:MODEL_REGISTRY`. New models belong in
  `network/models/`. RBD config uses `input_elements` + `input_freq_bins`;
  ELM/SCM use `input_pairs` + `input_freq_bins`. Return normalized predictions
  with shape `[batch,1]`.
- Keep DatasetBundle fields stable: datasets, indices, labels, input_shape,
  split_mode, source_paths. Keep GRNN outside trainable model registries.
- Time-suffixed outputs use `MMDD_HHMMSS`; use
  `common/paths.py:latest_time_suffixed_path` for latest resolution.
- Dependencies have one authority: `pyproject.toml`/`uv.lock`.

## Git and Reading

Whitelist tracked roots: README/AGENTS, doc, scripts, simulation, uv project
files. Never commit Origindata, outputs, local environments, OS junk or stray
files. Commit only on user request; messages are detailed Chinese summaries.

Before sensitive changes, read `doc/40_extension/00_python_extension_guide.md`
(loaders/models), `doc/10_datasets/00_hdf5_datasets_and_splits.md` (generation,
splits/HDF5), and `doc/20_python_workflows/` (execution). Current progress:
`doc/00_overview/00_project_progress.md`. Keep the edited file's language.
