# AGENTS.md

Workspace instructions for ZCode agents working in this repository.

## Project Purpose

Range-regression research on SwellEx-96 vertical line array (VLA) data. MATLAB
scripts build HDF5 feature datasets from the original experiment data; Python/
PyTorch trains CNN/ResNet models to regress source range in km. Four feature
routes:

- **RBD** — beamformer/RBD-derived Green-function features. HDF5 `X` layout:
  `[sample, element, frequency, real_imag]`.
- **ELM** — element pairwise least-squares frequency-ratio features. `X`:
  `[sample, pair, frequency, real_imag]`, strict upper-triangle pairs `i < j`.
- **SCM** — spatial covariance features. Same rank as ELM, but upper-triangle
  pairs include the diagonal (`i <= j`). ELM and SCM loaders are separate.
- **GRNN** (`scripts_py/0_GRNN_related/`) — standalone non-trainable
  Gaussian-kernel regression (`build` / `predict`, no epochs/optimizer).

## Layout

```text
Origindata/            original SwellEx data (events, positions, environments, source) — NOT committed
scripts_matlab/        dataset generation: RBD_method/, ELM_method/, SCM_method/ entry scripts
                       function/ helpers grouped by prefix: DS_*, SPL_*, RBD_*, ELM_*, SCM_*
scripts_py/common/     shared helpers: paths.py, h5_utils.py, training_utils.py, prediction_utils.py
scripts_py/*_method/   per-method network packages (cli, data, model registry, models/, training, prediction)
scripts_py/0_GRNN_related/  standalone SCM-GRNN (outside the trainable-network structure)
simulation/            bellhop / kraken propagation models
doc/                   numbered documentation; see doc/README.md index
outputs/               generated datasets and results — NOT committed
```

Data flow: `Origindata/` → MATLAB `Signals_Segmentation.m` →
`outputs/Datasets/<name>/{*_train,*_val,*_test}.h5` → Python training →
`outputs/networks_results/<method>/<model>/{train_outputs,test_outputs}/`.

## Commands

Run from the project root. Datasets resolve from `outputs/Datasets/<name>/`
unless a direct path is given.

```bash
# Train / predict (per method; RBD shown; same shape for ELM_method, SCM_method)
python3 scripts_py/RBD_method/Network_main.py train   --model complex_cnn_range --data <dataset>
python3 scripts_py/RBD_method/Network_main.py predict --model complex_cnn_range --data <dataset>

# Loss options: --loss-space normalized | km  (--huber-beta is in km when loss-space=km)

# Standalone GRNN
python3 scripts_py/0_GRNN_related/GRNN_main.py build   --data <scm_dataset> [--cv-folds 5]
python3 scripts_py/0_GRNN_related/GRNN_main.py predict --data <scm_dataset>

# Python syntax smoke check (no test suite exists; use this after edits)
conda activate pytorch
python -m py_compile scripts_py/common/*.py \
  scripts_py/RBD_method/Network_main.py \
  scripts_py/ELM_method/Network_main.py \
  scripts_py/SCM_method/Network_main.py

# MATLAB lint
/usr/local/bin/matlab -batch "checkcode('scripts_matlab/RBD_method/Signals_Segmentation.m')"
```

## Environment (this host, `HQserver`)

- Repo is mounted at `/mnt/SDD/project/SwellEx` (doc/00_overview/10_environment.md
  lists other machines' paths, e.g. HaiQin1's old `/home/yiyang-lu/project/SwellEx`).
- Python: conda env `pytorch` at `/home/yiyang-lu/miniforge3/envs/pytorch`
  (Python 3.14.4, PyTorch 2.11.0+cu128, CUDA available). Use this interpreter,
  not bare `python`.
- MATLAB: `/usr/local/bin/matlab` (R2025b).
- GPU: NVIDIA RTX A5000. If `torch.cuda.is_available()` is `False` only inside a
  sandboxed command, re-check in an interactive shell before touching CUDA or
  PyTorch settings — device access can be hidden by the sandbox.

## Architecture Rules

- Shared logic goes in `scripts_py/common/`; method-specific tensor layouts stay
  in the RBD/ELM/SCM method packages. Do not duplicate h5 path/split logic in
  method loaders — use `common/h5_utils.py`.
- Each method has its own `MODEL_REGISTRY` in `<method>/network/model.py`.
  New models go in `<method>/network/models/` and must be registered there.
  Configs: RBD uses `input_elements` + `input_freq_bins`; ELM/SCM use
  `input_pairs` + `input_freq_bins`. Models must return normalized range
  predictions of shape `[batch, 1]`.
- Keep `DatasetBundle` fields stable (train/val datasets, indices, labels,
  input_shape, split_mode, source_paths) so training, checkpointing, and
  prediction stay unchanged.
- Do not add GRNN to the RBD/ELM/SCM model registries unless intentionally
  converting it to a trainable `nn.Module`.
- Output files carry `MMDD_HHMMSS` time suffixes; "latest" resolution uses
  `common/paths.py:latest_time_suffixed_path`.

## Git Conventions

- `.gitignore` is whitelist-style: only `README.md`, `doc/`, `scripts_py/`,
  `scripts_matlab/`, `simulation/` are tracked. Never commit `outputs/`,
  `Origindata/`, or other generated data.
- Commit messages are written in Chinese, summarizing the change in detail.
- macOS junk files (`._*`, `.DS_Store`) and stray files like `_proxy -u all_proxy \`
  (accidentally saved `less` help output) are junk — do not commit or edit them.

## Read Before Changing Sensitive Areas

- Adding models or dataset loaders: `doc/40_extension/00_python_extension_guide.md`
- Dataset generation, split strategies, frequency modes, HDF5 fields:
  `doc/10_datasets/00_hdf5_datasets_and_splits.md`
- Workflows: `doc/20_python_workflows/` (training, prediction, standalone GRNN)
- Method internals: `doc/30_methods/` (model architectures, GRNN, RBD multipath)
- Current status and history: `doc/00_overview/00_project_progress.md` (in Chinese)

Docs are mixed Chinese/English; keep the language of the file you are editing.
